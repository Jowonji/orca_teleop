"""Tests for orca_teleop.cameras.

Spec parsing runs anywhere; the OpenCVCamera tests fake ``cv2.VideoCapture`` so
they exercise the threading/frame-processing plumbing without real cameras.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from orca_teleop.cameras import (
    CameraManager,
    OpenCVCamera,
    OpenCVCameraConfig,
    list_available_cameras,
    parse_camera_spec,
    with_recording_defaults,
)

cv2 = pytest.importorskip("cv2")


def test_parse_spec_name_only():
    cfg = parse_camera_spec("front")
    assert cfg == OpenCVCameraConfig(name="front", index=0)


def test_parse_spec_name_index():
    cfg = parse_camera_spec("iphone:1")
    assert cfg.name == "iphone"
    assert cfg.index == 1
    assert cfg.width is None and cfg.height is None and cfg.fps is None


def test_parse_spec_with_resolution():
    cfg = parse_camera_spec("iphone:1:1280x720")
    assert (cfg.index, cfg.width, cfg.height) == (1, 1280, 720)
    assert cfg.fps is None


def test_parse_spec_with_resolution_and_fps():
    cfg = parse_camera_spec("iphone:2:1920x1080@30")
    assert (cfg.index, cfg.width, cfg.height, cfg.fps) == (2, 1920, 1080, 30.0)


def test_parse_spec_index_and_fps_no_resolution():
    cfg = parse_camera_spec("front:0@60")
    assert (cfg.index, cfg.fps) == (0, 60.0)
    assert cfg.width is None and cfg.height is None


def test_parse_spec_missing_name_raises():
    with pytest.raises(ValueError, match="missing a name"):
        parse_camera_spec(":0")


def test_parse_spec_bad_resolution_raises():
    with pytest.raises(ValueError, match="invalid resolution"):
        parse_camera_spec("cam:0:1280")


def test_with_recording_defaults_fills_missing_resolution_and_fps():
    cfg = with_recording_defaults(parse_camera_spec("iphone:1"))
    assert (cfg.width, cfg.height, cfg.fps) == (640, 480, 30.0)


def test_with_recording_defaults_preserves_explicit_values():
    cfg = with_recording_defaults(parse_camera_spec("iphone:1:1280x720@60"))
    assert (cfg.width, cfg.height, cfg.fps) == (1280, 720, 60.0)


class _FakeCapture:
    """Minimal stand-in for cv2.VideoCapture yielding solid-color BGR frames."""

    def __init__(self, index, *_args, width=640, height=480, opened=True, frames_ok=True):
        self.index = index
        self._width = width
        self._height = height
        self._opened = opened
        self._frames_ok = frames_ok
        self.released = False

    def isOpened(self):
        return self._opened

    def set(self, _prop, _value):
        return True

    def read(self):
        if not self._frames_ok:
            return False, None
        # Distinct per-channel values so BGR->RGB conversion is observable.
        frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        frame[..., 0] = 10  # B
        frame[..., 1] = 20  # G
        frame[..., 2] = 30  # R
        return True, frame

    def release(self):
        self.released = True


def _patch_capture(monkeypatch, **kwargs):
    created: list[_FakeCapture] = []

    def factory(index, *args, **_kw):
        cap = _FakeCapture(index, *args, **kwargs)
        created.append(cap)
        return cap

    monkeypatch.setattr(cv2, "VideoCapture", factory)
    return created


def test_camera_open_read_close_converts_to_rgb(monkeypatch):
    _patch_capture(monkeypatch, width=640, height=480)
    cam = OpenCVCamera(OpenCVCameraConfig(name="cam", index=0))
    try:
        shape = cam.open()
        assert shape == (480, 640, 3)
        frame = cam.read()
        assert frame.shape == (480, 640, 3)
        assert frame.dtype == np.uint8
        # BGR (10,20,30) -> RGB (30,20,10)
        assert tuple(int(x) for x in frame[0, 0]) == (30, 20, 10)
    finally:
        cam.close()
    assert cam._thread is None


def test_camera_resizes_to_configured_resolution(monkeypatch):
    _patch_capture(monkeypatch, width=1920, height=1080)
    cam = OpenCVCamera(OpenCVCameraConfig(name="iphone", index=1, width=1280, height=720))
    try:
        shape = cam.open()
        assert shape == (720, 1280, 3)
        assert cam.read().shape == (720, 1280, 3)
    finally:
        cam.close()


def test_camera_rotate_swaps_dims(monkeypatch):
    _patch_capture(monkeypatch, width=640, height=480)
    cam = OpenCVCamera(OpenCVCameraConfig(name="cam", index=0, rotate=90))
    try:
        shape = cam.open()
        assert shape == (640, 480, 3)
    finally:
        cam.close()


def test_camera_open_failure_when_not_opened(monkeypatch):
    _patch_capture(monkeypatch, opened=False)
    cam = OpenCVCamera(OpenCVCameraConfig(name="cam", index=9))
    with pytest.raises(RuntimeError, match="Failed to open camera"):
        cam.open()


def test_camera_open_failure_when_no_frames(monkeypatch):
    _patch_capture(monkeypatch, frames_ok=False)
    cam = OpenCVCamera(OpenCVCameraConfig(name="cam", index=0, warmup_frames=2))
    with pytest.raises(RuntimeError, match="returned no"):
        cam.open()


def test_camera_close_is_idempotent(monkeypatch):
    _patch_capture(monkeypatch)
    cam = OpenCVCamera(OpenCVCameraConfig(name="cam", index=0))
    cam.open()
    cam.close()
    cam.close()  # must not raise


def test_camera_does_not_leak_reader_thread(monkeypatch):
    before = {t.ident for t in threading.enumerate()}
    _patch_capture(monkeypatch)
    cam = OpenCVCamera(OpenCVCameraConfig(name="cam", index=0))
    cam.open()
    cam.close()
    leaked = [t for t in threading.enumerate() if t.ident not in before and t.is_alive()]
    for t in leaked:
        t.join(timeout=1.0)
    assert not [t for t in leaked if t.is_alive()]


def test_list_available_cameras(monkeypatch):
    def factory(index, *_args, **_kw):
        # Only indices 0 and 2 "exist".
        return _FakeCapture(index, opened=index in (0, 2))

    monkeypatch.setattr(cv2, "VideoCapture", factory)
    found = list_available_cameras(max_index=4)
    assert [idx for idx, _w, _h in found] == [0, 2]
    for _idx, w, h in found:
        assert (w, h) == (640, 480)


def test_camera_manager_open_capture_close(monkeypatch):
    _patch_capture(monkeypatch, width=640, height=480)
    manager = CameraManager(
        [OpenCVCameraConfig(name="front", index=0), OpenCVCameraConfig(name="side", index=1)]
    )
    try:
        shapes = manager.open()
        assert shapes == {"front": (480, 640, 3), "side": (480, 640, 3)}
        assert manager.shapes == shapes
        assert len(manager) == 2
        assert manager.names == ["front", "side"]
        frames = manager.capture()
        assert set(frames) == {"front", "side"}
        assert frames["front"].shape == (480, 640, 3)
    finally:
        manager.close()
    assert len(manager) == 0
    assert manager.shapes == {}


def test_camera_manager_empty():
    manager = CameraManager([])
    assert manager.open() == {}
    assert manager.capture() == {}
    manager.ensure_live()
    manager.close()


def test_camera_manager_ensure_live_requires_fresh_frames(monkeypatch):
    _patch_capture(monkeypatch, width=640, height=480)
    manager = CameraManager([OpenCVCameraConfig(name="front", index=0)])
    try:
        manager.open()
        manager.ensure_live(samples=3, interval_s=0.0)
    finally:
        manager.close()


def test_camera_manager_ensure_live_raises_when_stale(monkeypatch):
    _patch_capture(monkeypatch, width=640, height=480)
    manager = CameraManager([OpenCVCameraConfig(name="front", index=0)])
    try:
        manager.open()
        # Freeze the freshness clock so the next read looks stale.
        cam = manager._cameras[0]
        cam._last_ok_ts = 0.0
        with pytest.raises(RuntimeError, match="no fresh frame"):
            manager.ensure_live(samples=1)
    finally:
        manager.close()


def test_camera_manager_rejects_duplicate_names():
    with pytest.raises(ValueError, match="Duplicate camera name"):
        CameraManager(
            [OpenCVCameraConfig(name="cam", index=0), OpenCVCameraConfig(name="cam", index=1)]
        )


def test_camera_manager_open_failure_releases_opened(monkeypatch):
    # Index 0 opens fine; index 1 fails to open -> manager must release index 0.
    created: list[_FakeCapture] = []

    def factory(index, *_args, **_kw):
        cap = _FakeCapture(index, opened=(index == 0))
        created.append(cap)
        return cap

    monkeypatch.setattr(cv2, "VideoCapture", factory)
    manager = CameraManager(
        [OpenCVCameraConfig(name="ok", index=0), OpenCVCameraConfig(name="bad", index=1)]
    )
    with pytest.raises(RuntimeError, match="Failed to open camera"):
        manager.open()
    assert len(manager) == 0
    # The successfully-opened capture must have been released during rollback.
    assert created[0].released is True
