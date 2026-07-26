"""OpenCV camera capture for dataset recording.

Wraps ``cv2.VideoCapture`` behind a small abstraction tuned for the
kind of USB / built-in / phone cameras one points at the workspace while
recording teleop demonstrations.

Frames are returned as contiguous ``uint8`` RGB arrays shaped ``(H, W, 3)``.
"""

from __future__ import annotations

import contextlib
import logging
import os
import platform
import threading
import time
from dataclasses import dataclass

import numpy as np

from orca_teleop.constants import (
    DEFAULT_CAMERA_FPS,
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_WIDTH,
    DEFAULT_MAX_PROBE_INDEX,
    MACOS_PLATFORM,
    STALE_FRAME_TIMEOUT_S,
)

_IS_MACOS = platform.system() == MACOS_PLATFORM

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpenCVCameraConfig:
    """Configuration for a single OpenCV-backed camera."""

    name: str
    index: int = 0
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    warmup_frames: int = 5
    rotate: int | None = None


def parse_camera_spec(spec: str) -> OpenCVCameraConfig:
    """Parse a ``--camera`` CLI string into an :class:`OpenCVCameraConfig`.

    Accepted forms (everything after ``NAME`` is optional)::

        NAME
        NAME:INDEX
        NAME:INDEX:WIDTHxHEIGHT
        NAME:INDEX:WIDTHxHEIGHT@FPS
        NAME:INDEX@FPS

    Examples::

        front                     -> index 0, native resolution
        iphone:1                  -> index 1, native resolution
        iphone:1:1280x720         -> index 1, forced 1280x720
        iphone:1:1280x720@30      -> index 1, forced 1280x720, request 30 fps
    """
    body, _, fps_str = spec.partition("@")
    fps = float(fps_str) if fps_str else None

    parts = body.split(":")
    name = parts[0]
    if not name:
        raise ValueError(f"Camera spec {spec!r} is missing a name (use NAME[:INDEX][:WxH][@FPS]).")

    index = int(parts[1]) if len(parts) > 1 and parts[1] else 0

    width: int | None = None
    height: int | None = None
    if len(parts) > 2 and parts[2]:
        res = parts[2].lower()
        w_str, sep, h_str = res.partition("x")
        if not sep or not w_str or not h_str:
            raise ValueError(
                f"Camera spec {spec!r} has an invalid resolution {parts[2]!r}; use WIDTHxHEIGHT."
            )
        width, height = int(w_str), int(h_str)

    return OpenCVCameraConfig(name=name, index=index, width=width, height=height, fps=fps)


def with_recording_defaults(config: OpenCVCameraConfig) -> OpenCVCameraConfig:
    """Fill omitted resolution/fps with the standard recording defaults."""
    return OpenCVCameraConfig(
        name=config.name,
        index=config.index,
        width=config.width if config.width is not None else DEFAULT_CAMERA_WIDTH,
        height=config.height if config.height is not None else DEFAULT_CAMERA_HEIGHT,
        fps=config.fps if config.fps is not None else DEFAULT_CAMERA_FPS,
        warmup_frames=config.warmup_frames,
        rotate=config.rotate,
    )


def _rotate_code(rotate: int | None):
    if not rotate:
        return None

    import cv2

    codes = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    if rotate not in codes:
        raise ValueError(f"Unsupported rotate={rotate!r}; use one of 90, 180, 270.")
    return codes[rotate]


def _open_backend():
    """Return the OpenCV backend flag best suited to the current OS."""
    import cv2

    return cv2.CAP_AVFOUNDATION if _IS_MACOS else cv2.CAP_ANY


class OpenCVCamera:
    """A single OpenCV camera with a background reader thread.

    Call :meth:`open` once, :meth:`read` per frame, and :meth:`close` when done.
    ``read`` returns the most recent frame captured by the background thread as a
    contiguous ``uint8`` RGB array of shape :attr:`shape`.
    """

    def __init__(self, config: OpenCVCameraConfig) -> None:
        self.config = config
        self._cap = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._shape: tuple[int, int, int] | None = None
        self._last_ok_ts: float = 0.0

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def shape(self) -> tuple[int, int, int]:
        if self._shape is None:
            raise RuntimeError(f"Camera {self.name!r} is not open; call open() first.")
        return self._shape

    def open(self) -> tuple[int, int, int]:
        """Open the device, prime it, and start the background reader.

        Returns the ``(height, width, channels)`` frame shape.
        """
        import cv2

        cap = cv2.VideoCapture(self.config.index, _open_backend())
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                f"Failed to open camera {self.name!r} (index {self.config.index}). "
                "Run with --list-cameras to see available indices."
            )

        if self.config.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.config.width))
        if self.config.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.config.height))
        if self.config.fps:
            cap.set(cv2.CAP_PROP_FPS, float(self.config.fps))

        last_good = None
        for _ in range(max(1, self.config.warmup_frames)):
            ok, frame = cap.read()
            if ok and frame is not None:
                last_good = frame
            time.sleep(0.01)

        if last_good is None:
            cap.release()
            raise RuntimeError(
                f"Camera {self.name!r} (index {self.config.index}) opened but returned no "
                "frames. On macOS make sure the terminal has Camera permission (System "
                "Settings > Privacy & Security > Camera) and the iPhone is unlocked/nearby."
            )

        self._cap = cap
        processed = self._process(last_good)
        self._shape = processed.shape  # type: ignore[assignment]
        self._latest = processed
        self._last_ok_ts = time.monotonic()

        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name=f"camera-{self.name}", daemon=True)
        self._thread.start()

        logger.info(
            "Camera %r opened (index=%d, shape=%s%s)",
            self.name,
            self.config.index,
            self._shape,
            f", requested {self.config.width}x{self.config.height}"
            if self.config.width and self.config.height
            else "",
        )
        return self._shape

    def _process(self, frame: np.ndarray) -> np.ndarray:
        import cv2

        if self.config.width and self.config.height:
            if frame.shape[1] != self.config.width or frame.shape[0] != self.config.height:
                frame = cv2.resize(frame, (self.config.width, self.config.height))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        code = _rotate_code(self.config.rotate)
        if code is not None:
            rgb = cv2.rotate(rgb, code)
        return np.ascontiguousarray(rgb)

    def _loop(self) -> None:
        assert self._cap is not None
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            processed = self._process(frame)
            with self._lock:
                self._latest = processed
                self._last_ok_ts = time.monotonic()

    def read(self) -> np.ndarray:
        """Return the most recently captured RGB frame.

        Raises ``RuntimeError`` if the reader thread has stopped or hasn't
        produced a fresh frame within :data:`STALE_FRAME_TIMEOUT_S`.
        """
        if self._thread is None or not self._thread.is_alive():
            raise RuntimeError(f"Camera {self.name!r} reader thread is not running.")
        with self._lock:
            frame = self._latest
            age = time.monotonic() - self._last_ok_ts
        if frame is None:
            raise RuntimeError(f"Camera {self.name!r} has no frame available.")
        if age > STALE_FRAME_TIMEOUT_S:
            raise RuntimeError(
                f"Camera {self.name!r} produced no fresh frame for {age:.1f}s; assuming it died."
            )
        return frame

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        with self._lock:
            self._latest = None
        self._shape = None


class CameraManager:
    """Owns a set of :class:`OpenCVCamera` built from configs.

    This deliberately lives *outside* the robot sink: camera choice is a
    recording concern, independent of which sink (hardware / sim) drives the
    robot. The recorder owns the cameras; the sink only drives commands.
    """

    def __init__(self, configs: list[OpenCVCameraConfig]) -> None:
        names = [c.name for c in configs]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"Duplicate camera name(s) in configs: {duplicates}")
        self._configs = list(configs)
        self._cameras: list[OpenCVCamera] = []
        self._shapes: dict[str, tuple[int, int, int]] = {}

    def __len__(self) -> int:
        return len(self._cameras)

    @property
    def names(self) -> list[str]:
        return [config.name for config in self._configs]

    @property
    def shapes(self) -> dict[str, tuple[int, int, int]]:
        return dict(self._shapes)

    def open(self) -> dict[str, tuple[int, int, int]]:
        """Open every configured camera. Returns ``{name: (H, W, C)}``.

        On any failure, already-opened cameras are released before re-raising.
        """
        try:
            for config in self._configs:
                camera = OpenCVCamera(config)
                shape = camera.open()
                self._cameras.append(camera)
                self._shapes[config.name] = tuple(int(x) for x in shape)
        except Exception:
            self.close()
            raise
        if self._cameras:
            logger.info("CameraManager opened %d camera(s): %s", len(self._cameras), self._shapes)
        return dict(self._shapes)

    def capture(self) -> dict[str, np.ndarray]:
        """Return the latest frame from each camera as ``{name: RGB array}``."""
        return {camera.name: camera.read() for camera in self._cameras}

    def ensure_live(self, samples: int = 3, interval_s: float = 0.05) -> None:
        """Verify every open camera keeps producing fresh frames.

        Raises ``RuntimeError`` on the first stale/dead camera. A single successful
        ``open()`` is not enough — Continuity Camera in particular can open, then
        stop delivering frames — so recording callers should probe repeatedly.
        """
        if samples < 1:
            raise ValueError(f"samples must be >= 1, got {samples}")
        if not self._cameras:
            return
        for i in range(samples):
            self.capture()
            if i + 1 < samples:
                time.sleep(interval_s)

    def close(self) -> None:
        for camera in self._cameras:
            try:
                camera.close()
            except Exception:
                pass
        self._cameras.clear()
        self._shapes.clear()


@contextlib.contextmanager
def _suppress_native_stderr():
    """Silence C-level stderr (the noisy OpenCV/AVFoundation probe warnings).

    OpenCV prints "out device of bound" / "camera failed to properly initialize"
    straight from C++ to fd 2, so Python-level logging config can't touch it.
    We redirect the underlying fd for the duration of the probe only.
    """
    try:
        saved_fd = os.dup(2)
    except OSError:
        yield
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(devnull)
        os.close(saved_fd)


def _probe_cameras(max_index: int) -> list[tuple[int, bool, int, int]]:
    """Probe indices ``0..max_index-1``, stopping once past the valid range.

    Returns ``(index, frame_ok, width, height)`` for every index that *opens*
    (whether or not it yielded a frame). ``frame_ok`` is ``False`` for devices
    that open but return nothing (e.g. an iPhone that's asleep/not yet active).
    """
    import cv2

    backend = _open_backend()
    probed: list[tuple[int, bool, int, int]] = []
    consecutive_unopened = 0
    with _suppress_native_stderr():
        for idx in range(max_index):
            cap = cv2.VideoCapture(idx, backend)
            try:
                if not cap.isOpened():
                    consecutive_unopened += 1
                    # AVFoundation indexes devices contiguously from 0, so two
                    # misses in a row means we've walked off the end.
                    if consecutive_unopened >= 2:
                        break
                    continue
                consecutive_unopened = 0
                ok, frame = cap.read()
                if ok and frame is not None:
                    h, w = frame.shape[:2]
                    probed.append((idx, True, int(w), int(h)))
                else:
                    probed.append((idx, False, 0, 0))
            finally:
                cap.release()
    return probed


def list_available_cameras(max_index: int = DEFAULT_MAX_PROBE_INDEX) -> list[tuple[int, int, int]]:
    """Probe camera indices and return the ones that open *and* yield a frame.

    Returns a list of ``(index, width, height)``.
    """
    return [(idx, w, h) for idx, frame_ok, w, h in _probe_cameras(max_index) if frame_ok]


def print_available_cameras(max_index: int = DEFAULT_MAX_PROBE_INDEX) -> None:
    """Print discovered cameras in a form ready to paste into ``--camera``."""
    probed = _probe_cameras(max_index)
    working = [(idx, w, h) for idx, frame_ok, w, h in probed if frame_ok]
    asleep = [idx for idx, frame_ok, _w, _h in probed if not frame_ok]

    if not working and not asleep:
        print(
            "No cameras found. On macOS grant Camera permission to your terminal "
            "(System Settings > Privacy & Security > Camera) and, for an iPhone, make "
            "sure it is unlocked and near the Mac (Continuity Camera)."
        )
        return

    if working:
        print(f"Found {len(working)} usable camera(s):")
        for idx, w, h in working:
            print(f"  index {idx}: {w}x{h}   ->  --camera cam{idx}:{idx}:{w}x{h}")
    if asleep:
        print(
            "\nOpened but returned no frame (device may be asleep/off — for an iPhone, "
            f"unlock it and keep it near the Mac, then re-run): indices {asleep}"
        )
    print(
        "\nTip: the iPhone (Continuity Camera) usually shows up as a high-resolution "
        "device (e.g. 1920x1080). Pick its index for --camera."
    )
