"""MediaPipe hand-tracking gRPC publisher.

Run this on the operator's machine (any OS with a webcam). It captures hand
landmarks via MediaPipe and streams them to the robot-side ``IngressServer``
over gRPC.

Usage::

    # Stream to robot on the same machine
    python -m orca_teleop.ingress.mediapipe.publisher

    # Stream to a remote robot
    python -m orca_teleop.ingress.mediapipe.publisher --server 192.168.1.42:50051

    # Left hand, high confidence
    python -m orca_teleop.ingress.mediapipe.publisher --hand left --confidence 0.9

    # Image-cue baseline: same Orbbec color stream, but no metric palm point
    python -m orca_teleop.ingress.mediapipe.publisher --depth off

    # Plain OpenCV webcam even when an Orbbec RGB-D camera is attached
    python -m orca_teleop.ingress.mediapipe.publisher --depth webcam

With an Orbbec camera and ``pyorbbecsdk2`` installed (``--depth auto``, the
default), the wrist hint gains a metric palm point: ``x, y, palm_width, X, Y, Z``
with X/Y/Z in meters in the color camera frame (NaN when depth is missing).
``--depth off`` keeps the Orbbec capture path (same frames and rate) and sends
only ``x, y, palm_width``, so the two arm modes can be compared like for like.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import grpc
import mediapipe as mp
import numpy as np

from orca_teleop.ingress import hand_stream_pb2, hand_stream_pb2_grpc
from orca_teleop.ingress.mediapipe.orbbec import OrbbecRGBDCamera, open_orbbec

# auto: Orbbec depth if present; orbbec: require it; off: Orbbec color only (3-value
# hint); webcam: OpenCV webcam, never the Orbbec SDK.
DEPTH_MODES = ("auto", "orbbec", "off", "webcam")
_DEPTH_CACHE = 8  # depth maps kept for matching async MediaPipe results by timestamp

_HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


def _draw_hand_landmarks(frame: np.ndarray, landmarks, color: tuple = (0, 255, 0)) -> None:
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in _HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, color, -1)


logger = logging.getLogger(__name__)


WINDOW_NAME = "MediaPipe Publisher"


def _prepare_opencv_gui() -> bool:
    """Create the preview window. Return False if this OpenCV build has no GUI."""
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    qt_fonts = Path(cv2.__file__).resolve().parent / "qt" / "fonts"
    if not qt_fonts.is_dir():
        for candidate in (
            Path("/usr/share/fonts/truetype/dejavu"),
            Path("/usr/share/fonts/truetype"),
            Path("/usr/share/fonts"),
        ):
            if candidate.is_dir():
                os.environ.setdefault("QT_QPA_FONTDIR", str(candidate))
                break
    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 640, 480)
        cv2.moveWindow(WINDOW_NAME, 80, 80)
        cv2.waitKey(1)
    except cv2.error as err:
        logger.warning(
            "OpenCV GUI is unavailable (%s). Hand tracking continues without a preview. "
            "This usually means opencv-python-headless replaced opencv-python.",
            err,
        )
        return False
    logger.info("Opened webcam preview window %r (DISPLAY=%s)", WINDOW_NAME, os.environ.get("DISPLAY"))
    return True


_COLOR_FORMATS = {"MJPG", "JPEG", "YUYV", "YUY2", "RGB3", "BGR3"}
_NON_COLOR_FORMATS = {
    "Z16",
    "GREY",
    "Y8",
    "Y16",
    "BA81",
    "GBRG",
    "RGGB",
    "GRBG",
    "NV12",
    "YV12",
    "Y10",
    "Y12",
}


def _normalize_fourcc(label: str) -> str:
    return label.replace(" ", "").strip().upper()


def _v4l2_formats(index: int) -> list[str]:
    """Pixel formats advertised by ``/dev/video{index}``."""
    try:
        out = subprocess.check_output(
            ["v4l2-ctl", "-d", f"/dev/video{index}", "--list-formats"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return []
    formats: list[str] = []
    for line in out.splitlines():
        if "'" not in line:
            continue
        formats.append(_normalize_fourcc(line.split("'")[1]))
    return formats


def _is_color_v4l_device(index: int) -> bool:
    formats = _v4l2_formats(index)
    if not formats:
        return True
    return any(fmt in _COLOR_FORMATS for fmt in formats)


def _webcam_indices_to_try(preferred: int | None = None) -> list[int]:
    if sys.platform.startswith("linux"):
        scored: list[tuple[int, int]] = []
        for idx in range(8):
            formats = _v4l2_formats(idx)
            if formats and not any(fmt in _COLOR_FORMATS for fmt in formats):
                logger.info(
                    "Skipping /dev/video%s (not RGB: %s)",
                    idx,
                    ",".join(formats),
                )
                continue
            score = 0
            if "MJPG" in formats or "JPEG" in formats:
                score += 100
            if "YUYV" in formats or "YUY2" in formats:
                score += 50
            scored.append((score, idx))
        scored.sort(key=lambda item: (-item[0], item[1]))
        indices = [idx for _, idx in scored]
        if not indices:
            indices = list(range(8))
    else:
        indices = list(range(8))
    if preferred is not None and _is_color_v4l_device(preferred):
        return [preferred] + [idx for idx in indices if idx != preferred]
    if preferred is not None:
        logger.info(
            "Ignoring --camera %s because it is not an RGB/MJPEG node; using color devices %s",
            preferred,
            indices,
        )
    return indices


def _in_video_group() -> bool:
    if not sys.platform.startswith("linux"):
        return True
    try:
        import grp

        return grp.getgrnam("video").gr_gid in os.getgroups()
    except KeyError:
        return True


def _fourcc_str(cap: cv2.VideoCapture) -> str:
    value = int(cap.get(cv2.CAP_PROP_FOURCC))
    chars = "".join(chr((value >> (8 * i)) & 0xFF) for i in range(4))
    return _normalize_fourcc("".join(ch if ch.isprintable() else " " for ch in chars)) or "?"


def configure_webcam(cap: cv2.VideoCapture) -> str:
    """Request MJPEG 640x480 before the first real frame."""
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
    for _ in range(3):
        cap.read()
    fourcc = _fourcc_str(cap)
    logger.info(
        "Webcam fourcc=%s size=%sx%s",
        fourcc,
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    return fourcc


def frame_to_bgr_rgb(frame: np.ndarray, fourcc: str = "") -> tuple[np.ndarray, np.ndarray]:
    """Return ``(bgr_for_imshow, rgb_for_mediapipe)`` from an OpenCV capture frame."""
    fourcc = _normalize_fourcc(fourcc)
    if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
        bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    channels = frame.shape[2] if frame.ndim == 3 else 0
    if channels == 2 or (fourcc in {"YUYV", "YUY2"} and channels != 3):
        bgr = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)
        return bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if channels == 4:
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # MJPEG / OpenCV V4L2 with CONVERT_RGB delivers BGR.
    bgr = np.ascontiguousarray(frame)
    return bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _try_open_index(idx: int) -> cv2.VideoCapture | None:
    from orca_teleop.cameras import _suppress_native_stderr

    if sys.platform.startswith("linux") and not _is_color_v4l_device(idx):
        return None

    backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
    with _suppress_native_stderr():
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            cap.release()
            return None
        configure_webcam(cap)
        fourcc = _fourcc_str(cap)
        known_color = _is_color_v4l_device(idx)
        if fourcc in _NON_COLOR_FORMATS and not known_color:
            logger.info("Skipping /dev/video%s after open (fourcc=%s)", idx, fourcc)
            cap.release()
            return None
        ok, frame = cap.read()
        if not ok or frame is None or getattr(frame, "size", 0) == 0:
            cap.release()
            return None
        height, width = frame.shape[:2]
        channels = frame.shape[2] if frame.ndim == 3 else 1
        if width < 160 or height < 120:
            cap.release()
            return None
        if fourcc in _NON_COLOR_FORMATS or (channels != 3 and fourcc not in _COLOR_FORMATS):
            logger.info("Skipping /dev/video%s after frame (fourcc=%s shape=%s)", idx, fourcc, frame.shape)
            cap.release()
            return None
        logger.info("Using RGB camera /dev/video%s (%sx%s, %s)", idx, width, height, fourcc)
        return cap


def find_webcam_index(camera_index: int | None = None) -> int:
    """Probe and return a working camera index without keeping the device open."""
    cap = open_webcam(camera_index, retry=False)
    # Best-effort: we already logged the source; caller only needs any working index.
    cap.release()
    time.sleep(0.2)
    return camera_index if camera_index is not None else 0


def open_webcam(
    camera_index: int | None = None,
    *,
    retry: bool = True,
    interval: float = 1.0,
) -> cv2.VideoCapture:
    """Open the RGB/MJPEG capture node, skipping depth/IR/metadata nodes.

    Orbbec Gemini cameras expose Z16 depth, GREY IR, and Bayer nodes first.
    Color lives on the node that advertises MJPEG/YUYV (commonly /dev/video6).
    When ``retry`` is true, keep scanning until that device yields a frame.
    """
    attempt = 0
    while True:
        attempt += 1
        for idx in _webcam_indices_to_try(camera_index):
            cap = _try_open_index(idx)
            if cap is not None:
                return cap
        if not retry:
            if camera_index is not None:
                raise RuntimeError(f"Failed to open webcam at index {camera_index}")
            raise RuntimeError(
                "Failed to open webcam. Join the `video` group (`newgrp video`) "
                "and retry, or pass an explicit --camera index (on WSL try --camera 2)."
            )
        if attempt == 1 or attempt % 3 == 0:
            video_hint = (
                ""
                if _in_video_group()
                else " Current user is not in the `video` group — run `newgrp video` in this terminal."
            )
            logger.warning(
                "Webcam not found (scan %s); retrying every %.1fs.%s",
                attempt,
                interval,
                video_hint,
            )
        time.sleep(interval)


def _raise_keyboard_interrupt(_signum, _frame) -> None:
    raise KeyboardInterrupt


class MediaPipePublisher:
    """Captures hand landmarks from a webcam and streams them over gRPC."""

    def __init__(
        self,
        server_address: str = "localhost:50051",
        handedness: str = "right",
        confidence: float = 0.7,
        show_video: bool = False,
        camera_index: int | None = None,
        depth: str = "auto",
    ) -> None:
        if depth not in DEPTH_MODES:
            raise ValueError(f"depth must be one of {DEPTH_MODES} (got {depth!r})")
        self._server_address = server_address
        self._handedness = handedness.lower()
        self._confidence = confidence
        self._show_video = show_video
        self._camera_index = camera_index
        self._landmarker = None
        self._capture_fourcc = ""
        self._depth_mode = depth
        self._send_depth = depth in {"auto", "orbbec"}
        self._rgbd: OrbbecRGBDCamera | None = None
        self._depth_by_ts: OrderedDict[int, np.ndarray] = OrderedDict()

        # Latest frame data (written by callback, read by stream generator)
        self._lock = threading.Lock()
        self._latest_keypoints: np.ndarray | None = None
        self._latest_wrist_image: np.ndarray | None = None
        self._fresh = False

        # Visualization state
        self._latest_frame: np.ndarray | None = None
        self._latest_image_landmarks = None
        self._latest_palm_xyz: np.ndarray | None = None
        self._mp_timestamp_ms = 0

    def _on_result(self, result, _output_image, timestamp_ms: int) -> None:
        """MediaPipe async callback — fires on each detection."""
        if not result.hand_landmarks:
            return

        # Only accept the hand we care about
        detected_hand = result.handedness[0][0].category_name.lower()
        if detected_hand != self._handedness:
            return

        world_landmarks = result.hand_world_landmarks[0]
        keypoints = np.array([[lm.x, lm.y, lm.z] for lm in world_landmarks], dtype=np.float32)
        image_landmarks = result.hand_landmarks[0]
        wrist = image_landmarks[0]
        index_mcp = image_landmarks[5]
        pinky_mcp = image_landmarks[17]
        palm_w = float(np.hypot(index_mcp.x - pinky_mcp.x, index_mcp.y - pinky_mcp.y))
        wrist_image = np.array([wrist.x, wrist.y, palm_w], dtype=np.float32)

        palm_xyz = None
        if self._rgbd is not None and self._send_depth:
            with self._lock:
                depth_m = self._depth_by_ts.get(timestamp_ms)
            if depth_m is not None:
                palm_xyz = self._rgbd.palm_point(depth_m, image_landmarks)
            nan3 = np.full(3, np.nan, dtype=np.float32)
            wrist_image = np.concatenate([wrist_image, nan3 if palm_xyz is None else palm_xyz])

        with self._lock:
            self._latest_keypoints = keypoints
            self._latest_wrist_image = wrist_image
            self._fresh = True
            if self._show_video:
                self._latest_image_landmarks = image_landmarks
                self._latest_palm_xyz = palm_xyz

    def _frame_generator(self):
        """Yield HandFrame protos as fast as new data arrives."""
        while True:
            with self._lock:
                if not self._fresh:
                    kp = None
                    wrist = None
                else:
                    kp = self._latest_keypoints.copy()
                    wrist = (
                        None
                        if self._latest_wrist_image is None
                        else self._latest_wrist_image.copy()
                    )
                    self._fresh = False

            if kp is None:
                time.sleep(0.001)
                continue

            packed = kp.ravel().tolist()
            if wrist is not None:
                packed.extend(wrist.tolist())
            yield hand_stream_pb2.HandFrame(
                keypoints=packed,
                handedness=self._handedness,
                timestamp_ns=time.time_ns(),
            )

    def _create_landmarker(self) -> None:
        mediapipe_task_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task"
        )
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(mediapipe_task_path),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=self._confidence,
            min_hand_presence_confidence=self._confidence,
            min_tracking_confidence=self._confidence,
            result_callback=self._on_result,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def run(self) -> None:
        """Open the webcam, connect to the server, and stream until interrupted."""
        # Grab the camera before MediaPipe creates an EGL context; otherwise WSL
        # camera probing often fails to see /dev/video2.
        cap = None
        if threading.current_thread() is threading.main_thread():
            # record_dataset terminates the publisher process; unwind so the camera is
            # stopped cleanly (an Orbbec pipeline left running stalls the next start).
            signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
        if self._depth_mode != "webcam":
            self._rgbd = open_orbbec()
            if self._rgbd is None and self._depth_mode == "orbbec":
                raise RuntimeError("--depth orbbec requested but no Orbbec RGB-D camera opened")
        if self._rgbd is None:
            logger.info("Searching for RGB/MJPEG camera (will keep retrying until one opens)")
            cap = open_webcam(self._camera_index, retry=True)
            self._capture_fourcc = _fourcc_str(cap)
        self._create_landmarker()
        if self._show_video:
            self._show_video = _prepare_opencv_gui()

        logger.info(
            "Connecting to %s (hand=%s, confidence=%.2f)",
            self._server_address,
            self._handedness,
            self._confidence,
        )
        channel = grpc.insecure_channel(self._server_address)
        stub = hand_stream_pb2_grpc.HandStreamStub(channel)

        # Start the gRPC stream in a background thread
        stream_future = stub.StreamHandFrames.future(self._frame_generator())

        try:
            missed = 0
            while True:
                depth_m = None
                if self._rgbd is not None:
                    ret, frame_rgb, depth_m = self._rgbd.read()
                    if not ret:
                        missed += 1
                        if missed == 15:
                            logger.warning("Orbbec camera stopped delivering color+depth frames")
                        continue
                    missed = 0
                    frame_bgr = (
                        cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR) if self._show_video else None
                    )
                else:
                    ret, frame = cap.read()
                    if not ret:
                        missed += 1
                        if missed >= 15:
                            logger.warning("Webcam dropped frames; searching again")
                            cap.release()
                            cap = open_webcam(self._camera_index, retry=True)
                            self._capture_fourcc = _fourcc_str(cap)
                            missed = 0
                        time.sleep(0.05)
                        continue
                    missed = 0
                    frame_bgr, frame_rgb = frame_to_bgr_rgb(frame, self._capture_fourcc)

                if self._show_video:
                    with self._lock:
                        self._latest_frame = frame_bgr

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                timestamp_ms = max(self._mp_timestamp_ms + 1, int(time.time() * 1000))
                self._mp_timestamp_ms = timestamp_ms
                if depth_m is not None and self._send_depth:
                    with self._lock:
                        self._depth_by_ts[timestamp_ms] = depth_m
                        while len(self._depth_by_ts) > _DEPTH_CACHE:
                            self._depth_by_ts.popitem(last=False)
                self._landmarker.detect_async(mp_image, timestamp_ms)

                if self._show_video:
                    self._display_frame()

                key = cv2.waitKey(1) & 0xFF
                if key in {ord("q"), 27}:  # q or Esc
                    break

                if self._rgbd is None:
                    time.sleep(1.0 / 30.0)  # the Orbbec read already blocks on the next frame

        except KeyboardInterrupt:
            pass
        finally:
            stream_future.cancel()
            channel.close()
            if cap is not None:
                cap.release()
            if self._rgbd is not None:
                self._rgbd.release()
            if self._landmarker is not None:
                self._landmarker.close()
            if self._show_video:
                cv2.destroyAllWindows()
            logger.info("Publisher shut down.")

    def _display_frame(self) -> None:
        """Show the webcam feed with landmarks overlaid."""
        with self._lock:
            if self._latest_frame is None:
                return
            frame = self._latest_frame.copy()
            image_landmarks = self._latest_image_landmarks
            palm_xyz = self._latest_palm_xyz

        if image_landmarks:
            _draw_hand_landmarks(frame, image_landmarks)
        if self._rgbd is not None:
            label = (
                "depth: off (image cue)"
                if not self._send_depth
                else "depth: --"
                if palm_xyz is None
                else f"palm X={palm_xyz[0]:+.2f} Y={palm_xyz[1]:+.2f} Z={palm_xyz[2]:.2f} m"
            )
            cv2.putText(frame, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow(WINDOW_NAME, frame)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream hand landmarks from a webcam to the orca_teleop server via gRPC.",
    )
    parser.add_argument(
        "--server",
        default="localhost:50051",
        help="gRPC server address (default: localhost:50051)",
    )
    parser.add_argument(
        "--hand",
        default="right",
        choices=["left", "right"],
        help="Which hand to track (default: right)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.7,
        help="MediaPipe detection confidence (default: 0.7)",
    )
    parser.add_argument(
        "--show-video",
        action="store_true",
        help="Show webcam feed with landmarks overlay",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="OpenCV camera index. Default: first device that yields a frame.",
    )
    parser.add_argument(
        "--depth",
        default="auto",
        choices=DEPTH_MODES,
        help="Metric palm position from an Orbbec RGB-D camera. 'auto' (default) uses it "
        "when pyorbbecsdk2 finds one and otherwise falls back to the OpenCV webcam; "
        "'orbbec' requires it; 'off' keeps the Orbbec color stream but sends only the "
        "image cue (baseline); 'webcam' forces the OpenCV webcam.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    publisher = MediaPipePublisher(
        server_address=args.server,
        handedness=args.hand,
        confidence=args.confidence,
        show_video=args.show_video,
        camera_index=args.camera,
        depth=args.depth,
    )
    publisher.run()


if __name__ == "__main__":
    main()
