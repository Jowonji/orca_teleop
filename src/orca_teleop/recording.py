"""Recording pipeline readiness helpers.

Recording must not begin until every sensor the episode depends on is live:
teleop actions flowing, and ``get_observation()`` succeeding repeatedly. Mid-
episode sensor failures should discard the in-progress episode rather than
persist broken data.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

from orca_core import OrcaJointPositions

from orca_teleop.constants import HEARTBEAT_INTERVAL

logger = logging.getLogger(__name__)

READY_ACTION_STREAK = 5
READY_OBS_STREAK = 5
READY_STATUS_INTERVAL_S = 5.0

T = TypeVar("T")


@dataclass
class TeleopActionMirror:
    """Thread-safe view of the most recent teleop command.

    A dedicated consumer thread owns ``actions_q`` and updates this mirror;
    the fixed-rate recorder reads snapshots without dequeuing commands.
    """

    _lock: threading.Lock = field(default_factory=threading.Lock)
    _latest: OrcaJointPositions | None = None
    _update_count: int = 0

    def update(self, action: OrcaJointPositions) -> None:
        with self._lock:
            self._latest = action
            self._update_count += 1

    def snapshot(self) -> OrcaJointPositions | None:
        with self._lock:
            return self._latest

    @property
    def update_count(self) -> int:
        with self._lock:
            return self._update_count

    def reset(self) -> None:
        with self._lock:
            self._latest = None
            self._update_count = 0


def teleop_consumer_loop(
    actions_q: queue.Queue[OrcaJointPositions | object],
    *,
    mirror: TeleopActionMirror,
    stop_event: threading.Event,
    shutdown_sentinel: object,
    dispatch_action: Callable[[OrcaJointPositions], None] | None = None,
    dispatch_enabled: threading.Event | None = None,
    heartbeat_interval: float = HEARTBEAT_INTERVAL,
) -> None:
    """Drain ``actions_q``, mirror each command, and optionally dispatch it."""
    while not stop_event.is_set():
        try:
            item = actions_q.get(timeout=heartbeat_interval)
        except queue.Empty:
            continue
        if item is shutdown_sentinel:
            stop_event.set()
            break
        if not isinstance(item, OrcaJointPositions):
            continue
        mirror.update(item)
        if dispatch_action is None:
            continue
        if dispatch_enabled is not None and not dispatch_enabled.is_set():
            continue
        try:
            dispatch_action(item)
        except Exception:
            logger.exception("Teleop dispatch failed")


def drain_actions_queue(
    actions_q: queue.Queue[OrcaJointPositions | object],
    *,
    stop_event: threading.Event,
    shutdown_sentinel: object,
) -> int:
    """Drop stale teleop commands so a new episode starts from a clean queue."""
    drained = 0
    while True:
        try:
            item = actions_q.get_nowait()
        except queue.Empty:
            break
        if item is shutdown_sentinel:
            stop_event.set()
            break
        drained += 1
    return drained


def poll_latest_action(
    actions_q: queue.Queue[OrcaJointPositions | object],
    *,
    last_action: OrcaJointPositions | None,
    stop_event: threading.Event,
    shutdown_sentinel: object,
) -> OrcaJointPositions | None:
    """Return the newest queued teleop command without blocking."""
    latest = last_action
    while True:
        try:
            item = actions_q.get_nowait()
        except queue.Empty:
            break
        if item is shutdown_sentinel:
            stop_event.set()
            return None
        if isinstance(item, OrcaJointPositions):
            latest = item
    return latest


def wait_for_teleop_mirror_ready(
    *,
    get_observation: Callable[[], T],
    mirror: TeleopActionMirror,
    stop_event: threading.Event,
    heartbeat_interval: float = HEARTBEAT_INTERVAL,
    min_action_streak: int = READY_ACTION_STREAK,
    min_obs_streak: int = READY_OBS_STREAK,
    status_interval_s: float = READY_STATUS_INTERVAL_S,
) -> bool:
    """Block until the teleop mirror and observations are both healthy.

    Observes the mirror updated by :func:`teleop_consumer_loop` without
    dequeuing from ``actions_q`` itself.
    """
    if min_action_streak < 1 or min_obs_streak < 1:
        raise ValueError("readiness streaks must be >= 1")

    logger.info(
        "Waiting for recording readiness: need %d consecutive teleop updates and "
        "%d consecutive healthy observations.",
        min_action_streak,
        min_obs_streak,
    )
    seen_updates = mirror.update_count
    action_streak = 0
    obs_streak = 0
    last_status = time.monotonic()
    last_obs_error: str | None = None

    while not stop_event.is_set():
        time.sleep(heartbeat_interval)
        update_count = mirror.update_count
        if update_count > seen_updates:
            action_streak += 1
            seen_updates = update_count
        else:
            action_streak = 0

        try:
            get_observation()
        except Exception as exc:
            action_streak = 0
            obs_streak = 0
            last_obs_error = f"{type(exc).__name__}: {exc}"
            now = time.monotonic()
            if now - last_status >= status_interval_s:
                logger.warning(
                    "Observation pipeline not ready (%s). Not recording until sensors recover.",
                    last_obs_error,
                )
                last_status = now
            continue

        obs_streak += 1
        last_obs_error = None

        if action_streak >= min_action_streak and obs_streak >= min_obs_streak:
            logger.info("Recording pipeline ready (teleop stream + observations healthy).")
            return True

        now = time.monotonic()
        if now - last_status >= status_interval_s:
            logger.info(
                "Warming up sensors (teleop=%d/%d, observations=%d/%d)",
                action_streak,
                min_action_streak,
                obs_streak,
                min_obs_streak,
            )
            last_status = now

    return False


def wait_for_recording_ready(
    *,
    get_observation: Callable[[], T],
    actions_q: queue.Queue[OrcaJointPositions | object],
    stop_event: threading.Event,
    dispatch_action: Callable[[OrcaJointPositions], None] | None,
    shutdown_sentinel: object,
    heartbeat_interval: float = HEARTBEAT_INTERVAL,
    min_action_streak: int = READY_ACTION_STREAK,
    min_obs_streak: int = READY_OBS_STREAK,
    status_interval_s: float = READY_STATUS_INTERVAL_S,
) -> bool:
    """Block until teleop actions and observations are both healthy.

    Consumes actions from ``actions_q`` (optionally dispatching them so the
    operator can verify teleop) and repeatedly probes ``get_observation``.
    Returns ``True`` once both streaks are satisfied, or ``False`` if
    ``stop_event`` is set / a shutdown sentinel arrives first.

    Nothing should be written to the dataset while this function is running.
    """
    if min_action_streak < 1 or min_obs_streak < 1:
        raise ValueError("readiness streaks must be >= 1")

    logger.info(
        "Waiting for recording readiness: need %d consecutive actions and "
        "%d consecutive healthy observations.",
        min_action_streak,
        min_obs_streak,
    )
    action_streak = 0
    obs_streak = 0
    last_status = time.monotonic()
    last_obs_error: str | None = None

    while not stop_event.is_set():
        try:
            action = actions_q.get(timeout=heartbeat_interval)
        except queue.Empty:
            action_streak = 0
            now = time.monotonic()
            if now - last_status >= status_interval_s:
                logger.info(
                    "Still waiting for teleop stream (actions=%d/%d, observations=%d/%d)%s",
                    action_streak,
                    min_action_streak,
                    obs_streak,
                    min_obs_streak,
                    f"; last observation error: {last_obs_error}" if last_obs_error else "",
                )
                last_status = now
            continue

        if action is shutdown_sentinel:
            stop_event.set()
            return False
        if not isinstance(action, OrcaJointPositions):
            action_streak = 0
            continue

        try:
            get_observation()
        except Exception as exc:
            action_streak = 0
            obs_streak = 0
            last_obs_error = f"{type(exc).__name__}: {exc}"
            now = time.monotonic()
            if now - last_status >= status_interval_s:
                logger.warning(
                    "Observation pipeline not ready (%s). Not recording until sensors recover.",
                    last_obs_error,
                )
                last_status = now
            continue

        obs_streak += 1
        action_streak += 1
        last_obs_error = None

        if dispatch_action is not None:
            try:
                dispatch_action(action)
            except Exception:
                logger.exception(
                    "Dispatch failed during readiness probe; treating pipeline as not ready."
                )
                action_streak = 0
                obs_streak = 0
                continue

        if action_streak >= min_action_streak and obs_streak >= min_obs_streak:
            logger.info("Recording pipeline ready (teleop stream + observations healthy).")
            return True

        now = time.monotonic()
        if now - last_status >= status_interval_s:
            logger.info(
                "Warming up sensors (actions=%d/%d, observations=%d/%d)",
                action_streak,
                min_action_streak,
                obs_streak,
                min_obs_streak,
            )
            last_status = now

    return False
