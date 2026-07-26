"""URDF-backed kinematic teleop sink.

This is intentionally lighter than ``orca_teleop.sim``: it does not simulate
contacts, actuator dynamics, or MuJoCo physics. It only displays the latest
retargeted joint command on the ORCA URDF.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
from pathlib import Path
from typing import Any

import numpy as np
import orca_core
from orca_core import OrcaHand, OrcaJointPositions

from orca_teleop.pipeline import _SHUTDOWN, RobotSink
from orca_teleop.retargeting.urdf_offsets import load_ref_offsets
from orca_teleop.utils import RateTicker

logger = logging.getLogger(__name__)

_ORCAHAND_DESCRIPTION_DIR_ENV = "ORCAHAND_DESCRIPTION_DIR"
_ORCAHAND_DESCRIPTION_DIR_DEFAULT = os.path.join(
    os.path.expanduser("~"), "Documents", "orcahand_description"
)

_JOINT_ALIASES: dict[str, dict[str, str]] = {
    "right": {"thumb_pip": "thumb_cmc"},
    "left": {"thumb_pip": "thumb_cmc"},
}


def _default_model_path(hand_type: str, version: str) -> str:
    path = Path(orca_core.__file__).resolve().parent / "models" / version / f"orcahand_{hand_type}"
    config_path = path / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find OrcaCore model config: {config_path}")
    return str(config_path)


def _default_urdf_path(hand_type: str, version: str) -> str:
    base = os.environ.get(_ORCAHAND_DESCRIPTION_DIR_ENV, _ORCAHAND_DESCRIPTION_DIR_DEFAULT)
    path = Path(base) / version / "models" / "urdf" / f"orcahand_{hand_type}.urdf"
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find ORCA URDF at {path}. Set {_ORCAHAND_DESCRIPTION_DIR_ENV} "
            "or pass --urdf_path."
        )
    return str(path)


def _package_dirs_for_urdf(urdf_path: str) -> list[str]:
    path = Path(urdf_path).resolve()
    # orcahand_description layout:
    #   ~/Documents/orcahand_description/v2/models/urdf/orcahand_right.urdf
    # package://orcahand_description/... resolves from ~/Documents.
    for parent in path.parents:
        if parent.name == "orcahand_description":
            return [str(parent.parent)]
    return [str(path.parent)]


class OrcaHandUrdfVizSink(RobotSink):
    """Kinematic Meshcat viewer sink for ``OrcaJointPositions`` commands."""

    def __init__(
        self,
        hand_type: str = "right",
        version: str = "v2",
        model_path: str | None = None,
        urdf_path: str | None = None,
        rate_hz: float = 90.0,
        open_browser: bool = True,
    ) -> None:
        self._hand_type = hand_type
        self._version = version
        self._model_path = model_path
        self._urdf_path = urdf_path
        self._rate_hz = float(rate_hz)
        self._open_browser = open_browser

        self._hand: OrcaHand | None = None
        self._pin: Any | None = None
        self._viz: Any | None = None
        self._model: Any | None = None
        self._qpos: np.ndarray | None = None
        self._last_action: OrcaJointPositions | None = None
        self._joint_q_indices: dict[str, int] = {}
        self._ref_offsets: dict[str, float] = {}

    @property
    def retarget_model_path(self) -> str:
        if self._model_path is None:
            return _default_model_path(self._hand_type, self._version)
        return self._model_path

    def connect(self) -> None:
        try:
            import pinocchio as pin
            from pinocchio.visualize import MeshcatVisualizer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "URDF visualization requires optional dependencies. Run with "
                "`uv run --extra adaptive --extra urdf-viz python scripts/teleop_urdf.py ...`."
            ) from exc

        self._pin = pin
        model_path = self.retarget_model_path
        urdf_path = self._urdf_path or _default_urdf_path(self._hand_type, self._version)

        hand = OrcaHand(model_path)
        if hand.config.type != self._hand_type:
            raise ValueError(
                f"Model hand type {hand.config.type!r} does not match sink hand "
                f"{self._hand_type!r}"
            )
        self._hand = hand
        self._urdf_path = urdf_path

        model = pin.buildModelFromUrdf(urdf_path)
        visual_model = pin.buildGeomFromUrdf(
            model,
            urdf_path,
            pin.GeometryType.VISUAL,
            package_dirs=_package_dirs_for_urdf(urdf_path),
        )
        self._model = model
        self._qpos = np.zeros(model.nq, dtype=np.float64)
        self._ref_offsets = load_ref_offsets(urdf_path, self._hand_type) or {}
        self._build_joint_mapping()

        self._last_action = OrcaJointPositions(hand.config.neutral_position)
        self._qpos[:] = self._to_qpos(self._last_action)

        self._viz = MeshcatVisualizer(model, pin.GeometryModel(), visual_model)
        self._viz.initViewer(open=self._open_browser)
        self._viz.loadViewerModel(rootNodeName=f"orca_{self._hand_type}_urdf")
        self._viz.display(self._qpos)

        logger.info(
            "URDF Viz connected: hand=%s version=%s joints=%d rate=%.1fHz url=%s",
            self._hand_type,
            self._version,
            len(self._joint_q_indices),
            self._rate_hz,
            self._viewer_url(),
        )

    def run_loop(
        self,
        actions_q: queue.Queue[OrcaJointPositions | object],
        stop_event: threading.Event,
    ) -> None:
        assert self._last_action is not None, "connect() must be called before run_loop()"
        ticker = RateTicker(dt=1.0 / self._rate_hz)

        while not stop_event.is_set():
            shutdown_received = False
            latest = self._last_action

            while True:
                try:
                    item = actions_q.get_nowait()
                except queue.Empty:
                    break

                if item is _SHUTDOWN:
                    shutdown_received = True
                    break
                if isinstance(item, OrcaJointPositions):
                    latest = item

            if shutdown_received:
                break

            if latest is not self._last_action:
                self._last_action = latest
                self._display(latest)

            ticker.tick()

    def close(self) -> None:
        # Meshcat has no hard process to close here; leaving the browser tab open
        # is useful after Ctrl-C for visual inspection.
        self._viz = None

    def _build_joint_mapping(self) -> None:
        assert self._hand is not None
        assert self._model is not None

        aliases = _JOINT_ALIASES.get(self._hand_type, {})
        missing = []
        for joint_id in self._hand.config.joint_ids:
            urdf_suffix = aliases.get(joint_id, joint_id)
            urdf_name = f"{self._hand_type}_{urdf_suffix}"
            pin_joint_id = self._model.getJointId(urdf_name)
            if pin_joint_id >= self._model.njoints:
                missing.append(urdf_name)
                continue
            self._joint_q_indices[joint_id] = int(self._model.idx_qs[pin_joint_id])

        if missing:
            raise ValueError(f"URDF is missing joints required by Orca config: {missing}")

    def _to_qpos(self, positions: OrcaJointPositions) -> np.ndarray:
        assert self._hand is not None
        assert self._model is not None

        qpos = np.zeros(self._model.nq, dtype=np.float64)
        aliases = _JOINT_ALIASES.get(self._hand_type, {})
        position_data = positions.as_dict()
        for joint_id in self._hand.config.joint_ids:
            q_idx = self._joint_q_indices[joint_id]
            urdf_suffix = aliases.get(joint_id, joint_id)
            ref_offset = self._ref_offsets.get(
                joint_id,
                self._ref_offsets.get(urdf_suffix, 0.0),
            )
            value_deg = float(
                position_data.get(joint_id, self._hand.config.neutral_position[joint_id])
            )
            qpos[q_idx] = np.deg2rad(value_deg) - ref_offset
        return qpos

    def _display(self, action: OrcaJointPositions) -> None:
        assert self._viz is not None
        self._qpos = self._to_qpos(action)
        self._viz.display(self._qpos)

    def _viewer_url(self) -> str:
        if self._viz is None:
            return "?"
        viewer = getattr(self._viz, "viewer", None)
        if viewer is None or not hasattr(viewer, "url"):
            return "?"
        try:
            return str(viewer.url())
        except Exception:
            return "?"
