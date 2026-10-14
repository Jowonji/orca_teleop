"""Orbbec RGB-D capture for the MediaPipe publisher.

Streams color + depth through pyorbbecsdk (``pip install pyorbbecsdk2``) and
aligns depth to the color image, so a MediaPipe landmark pixel indexes the
depth map directly. ``palm_point`` turns palm landmarks into a metric 3D point
in the color camera frame (x right, y down, z forward).
"""

from __future__ import annotations

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)

WIDTH = 640
HEIGHT = 480
FPS = 30
FRAME_TIMEOUT_MS = 200
# A process killed without pipeline.stop() leaves the next start() silent; a restart fixes it.
STALL_RESTART_S = 1.5

# Palm landmarks: wrist + index/middle/ring/pinky MCP. Fingertips miss depth too often.
PALM_LANDMARKS = (0, 5, 9, 13, 17)
PATCH_RADIUS = 3  # px, (2r+1)^2 window per landmark
MIN_DEPTH_M = 0.15
MAX_DEPTH_M = 1.5
OUTLIER_M = 0.06  # palm landmarks farther than this from the palm median are dropped


class OrbbecRGBDCamera:
    """Color + depth-aligned-to-color from an Orbbec camera (e.g. Gemini 335)."""

    def __init__(
        self,
        width: int = WIDTH,
        height: int = HEIGHT,
        fps: int = FPS,
        serial: str | None = None,
        laser_level: int | None = None,
    ) -> None:
        """``serial`` picks one Orbbec when several are attached (default: the first found).

        With a serial the SDK uses the kernel V4L2 backend: its default libuvc backend
        detaches uvcvideo from the cameras, which takes the other Orbbec's color node away
        from the OpenCV recorder. ``laser_level`` (0-6, Gemini 335) lowers the IR projector:
        on a bus-powered hub, color + depth with the laser at 5-6 reset the camera, 1-4 ran.
        """
        from pyorbbecsdk import (
            AlignFilter,
            Config,
            Context,
            OBFormat,
            OBFrameAggregateOutputMode,
            OBPropertyID,
            OBSensorType,
            OBStreamType,
            OBUvcBackendType,
            Pipeline,
        )

        if serial:
            # The device list is only valid while its Context lives.
            self._context = Context()
            self._context.set_uvc_backend_type(OBUvcBackendType.V4L2)
            device = self._context.query_devices().get_device_by_serial_number(serial)
            self._pipeline = Pipeline(device)
        else:
            self._pipeline = Pipeline()
        if laser_level is not None:
            device = self._pipeline.get_device()
            device.set_int_property(OBPropertyID.OB_PROP_LASER_CONTROL_INT, 1)
            device.set_int_property(OBPropertyID.OB_PROP_LASER_POWER_LEVEL_CONTROL_INT, laser_level)
        color_profile = self._pipeline.get_stream_profile_list(
            OBSensorType.COLOR_SENSOR
        ).get_video_stream_profile(width, height, OBFormat.RGB, fps)
        depth_profile = self._pipeline.get_stream_profile_list(
            OBSensorType.DEPTH_SENSOR
        ).get_video_stream_profile(width, height, OBFormat.Y16, fps)
        config = Config()
        config.enable_stream(color_profile)
        config.enable_stream(depth_profile)
        config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        self._config = config
        self._align = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

        intr = color_profile.get_intrinsic()
        self.fx, self.fy, self.cx, self.cy = intr.fx, intr.fy, intr.cx, intr.cy
        self.width, self.height = width, height

        self._pipeline.start(config)
        self._last_frame = time.monotonic()
        info = self._pipeline.get_device().get_device_info()
        logger.info(
            "Orbbec %s (fw %s, %s): color+depth %dx%d@%d, fx=%.1f fy=%.1f",
            info.get_name(),
            info.get_firmware_version(),
            info.get_connection_type(),
            width,
            height,
            fps,
            self.fx,
            self.fy,
        )

    def read(self) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
        """Return (ok, rgb uint8 HxWx3, depth meters float32 HxW; 0 = invalid)."""
        frames = self._pipeline.wait_for_frames(FRAME_TIMEOUT_MS)
        if frames is None:
            if time.monotonic() - self._last_frame > STALL_RESTART_S:
                self._restart()
            return False, None, None
        aligned = self._align.process(frames)
        if aligned is None:
            return False, None, None
        aligned = aligned.as_frame_set()
        color = aligned.get_color_frame()
        depth = aligned.get_depth_frame()
        if color is None or depth is None:
            return False, None, None
        rgb = np.frombuffer(color.get_data(), dtype=np.uint8).reshape(
            color.get_height(), color.get_width(), 3
        )
        raw = np.frombuffer(depth.get_data(), dtype=np.uint16).reshape(
            depth.get_height(), depth.get_width()
        )
        self._last_frame = time.monotonic()
        depth_m = raw.astype(np.float32) * (float(depth.get_depth_scale()) * 1e-3)
        return True, rgb.copy(), depth_m

    def palm_point(self, depth_m: np.ndarray, image_landmarks) -> np.ndarray | None:
        """Wrist pixel deprojected at the robust palm depth, in meters (camera frame)."""
        h, w = depth_m.shape
        samples = []
        for i in PALM_LANDMARKS:
            u = int(round(image_landmarks[i].x * w))
            v = int(round(image_landmarks[i].y * h))
            if not (0 <= u < w and 0 <= v < h):
                continue
            patch = depth_m[
                max(v - PATCH_RADIUS, 0) : v + PATCH_RADIUS + 1,
                max(u - PATCH_RADIUS, 0) : u + PATCH_RADIUS + 1,
            ]
            valid = patch[(patch > MIN_DEPTH_M) & (patch < MAX_DEPTH_M)]
            if valid.size:
                samples.append(float(np.median(valid)))
        if len(samples) < 2:
            return None
        samples = np.asarray(samples)
        center = float(np.median(samples))
        z = float(np.mean(samples[np.abs(samples - center) < OUTLIER_M]))

        wrist = image_landmarks[0]
        u = wrist.x * self.width
        v = wrist.y * self.height
        return np.array(
            [(u - self.cx) * z / self.fx, (v - self.cy) * z / self.fy, z],
            dtype=np.float32,
        )

    def _restart(self) -> None:
        logger.warning("Orbbec stream stalled for %.1fs; restarting the pipeline", STALL_RESTART_S)
        try:
            self._pipeline.stop()
        except Exception:  # noqa: BLE001 - stop on a stalled pipeline may fail
            pass
        time.sleep(0.5)
        self._pipeline.start(self._config)
        self._last_frame = time.monotonic()

    def release(self) -> None:
        try:
            self._pipeline.stop()
        except Exception:  # noqa: BLE001 - best-effort shutdown
            pass


def open_orbbec(
    serial: str | None = None, laser_level: int | None = None
) -> OrbbecRGBDCamera | None:
    """Open the Orbbec RGB-D camera (``serial``, else the first), or None when unavailable."""
    try:
        return OrbbecRGBDCamera(serial=serial, laser_level=laser_level)
    except ImportError:
        logger.info("pyorbbecsdk not installed; using the OpenCV webcam without depth")
    except Exception as err:  # noqa: BLE001 - SDK raises plain RuntimeError/OBError
        logger.warning("Orbbec RGB-D unavailable (%s); using the OpenCV webcam without depth", err)
    return None
