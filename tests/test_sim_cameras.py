"""Tests for how the sim sink composes its observation.

The sim sink owns its observation: the intrinsic MuJoCo render plus proprioception,
and optionally real (config-driven) cameras. These avoid constructing a real
MuJoCo env (which needs a GL context) by injecting a fake renderer / camera
manager, so only the observation-composition logic is exercised.
"""

from __future__ import annotations

import inspect
import types

import numpy as np
import pytest

pytest.importorskip("orca_sim")

from orca_teleop.pipeline import SinkObservation  # noqa: E402
from orca_teleop.sim import OrcaHandSimSink, SimCameraConfig  # noqa: E402


class _FakeRenderer:
    def __init__(self, height: int, width: int) -> None:
        self._height = height
        self._width = width
        self.closed = False

    def update_scene(self, _data, camera=None):
        return None

    def render(self):
        return np.full((self._height, self._width, 3), 7, dtype=np.uint8)

    def close(self):
        self.closed = True


class _FakeCameras:
    """Stand-in for a ``CameraManager`` with pre-opened cameras."""

    def __init__(self, frames: dict[str, np.ndarray]) -> None:
        self._frames = frames
        self.closed = False

    @property
    def names(self) -> list[str]:
        return list(self._frames)

    @property
    def shapes(self) -> dict[str, tuple[int, int, int]]:
        return {name: tuple(img.shape) for name, img in self._frames.items()}

    def capture(self) -> dict[str, np.ndarray]:
        return {name: img.copy() for name, img in self._frames.items()}

    def close(self) -> None:
        self.closed = True


def _sink_with_fakes(real_frames: dict[str, np.ndarray] | None = None):
    sink = OrcaHandSimSink(camera_config=SimCameraConfig(name="frontal", width=320, height=240))
    # get_observation() touches env.data.qpos, the qpos addresses, a renderer and
    # the record camera.
    sink._env = types.SimpleNamespace(data=types.SimpleNamespace(qpos=np.deg2rad([10.0, 20.0])))
    sink._joint_qpos_adr = [0, 1]
    sink._renderer = _FakeRenderer(height=240, width=320)
    sink._record_camera = object()
    if real_frames is not None:
        sink._cameras = _FakeCameras(real_frames)
    return sink


def test_camera_shapes_reports_render():
    sink = OrcaHandSimSink(camera_config=SimCameraConfig(name="frontal", width=320, height=240))
    assert sink.camera_shapes == {"frontal": (240, 320, 3)}


def test_camera_shapes_includes_real_cameras():
    sink = _sink_with_fakes({"wrist": np.zeros((120, 160, 3), dtype=np.uint8)})
    assert sink.camera_shapes == {"frontal": (240, 320, 3), "wrist": (120, 160, 3)}


def test_get_observation_composes_render_and_state():
    sink = _sink_with_fakes()
    obs = sink.get_observation()
    assert isinstance(obs, SinkObservation)
    np.testing.assert_allclose(obs.joint_state, [10.0, 20.0], atol=1e-4)
    assert set(obs.images) == {"frontal"}
    assert obs.images["frontal"].shape == (240, 320, 3)
    assert int(obs.images["frontal"][0, 0, 0]) == 7


def test_get_observation_includes_real_cameras():
    sink = _sink_with_fakes({"wrist": np.full((120, 160, 3), 3, dtype=np.uint8)})
    obs = sink.get_observation()
    assert set(obs.images) == {"frontal", "wrist"}
    assert obs.images["wrist"].shape == (120, 160, 3)
    assert int(obs.images["wrist"][0, 0, 0]) == 3


def test_sim_sink_accepts_camera_configs():
    # Real cameras live inside the sink again.
    params = inspect.signature(OrcaHandSimSink.__init__).parameters
    assert "camera_configs" in params
