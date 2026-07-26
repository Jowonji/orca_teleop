"""Tests for the repository-owned Quest Browser stream."""

import asyncio
import queue
import socket
import threading

import numpy as np
import pytest

from orca_teleop.ingress.metaquest.bridge import (
    QuestTelemetryState,
    WebXRHandSample,
)
from orca_teleop.ingress.metaquest.landmarks import (
    WEBXR_TO_RETARGETER_LANDMARK_INDICES,
)
from orca_teleop.ingress.metaquest.publisher import MetaQuestPublisher
from orca_teleop.ingress.server import HandLandmarks, IngressServer


def _webxr_payload(side: str = "right") -> dict:
    return {
        "type": "telemetry",
        "client_wall_ms": 1234.5,
        "hands": {
            side: {
                "wrist": np.eye(4).T.ravel().tolist(),
                "landmarks": np.arange(75, dtype=float).reshape(25, 3).tolist(),
            }
        },
    }


def test_telemetry_state_keeps_latest_valid_hand_sample():
    state = QuestTelemetryState()

    state.update(_webxr_payload())
    first = state.get_hand_sample("right")
    state.update(_webxr_payload())
    second = state.get_hand_sample("right")

    assert first is not None
    assert second is not None
    assert first.sequence_id == 1
    assert second.sequence_id == 2
    assert first.timestamp_ns == 1_234_500_000
    assert first.landmarks.shape == (25, 3)
    np.testing.assert_allclose(first.wrist_matrix, np.eye(4))


def test_telemetry_state_drops_incomplete_webxr_hand():
    state = QuestTelemetryState()
    state.update(_webxr_payload())
    state.update(
        {
            "type": "telemetry",
            "hands": {"right": {"landmarks": np.zeros((24, 3)).tolist()}},
        }
    )

    assert state.get_hand_sample("right") is None


def test_webxr_publisher_emits_reduced_points_and_wrist_angle():
    publisher = MetaQuestPublisher(wrist_enabled=True)
    landmarks = np.arange(75, dtype=float).reshape(25, 3)
    sample = WebXRHandSample(
        sequence_id=1,
        timestamp_ns=42,
        landmarks=landmarks,
        wrist_matrix=np.eye(4),
    )

    first = publisher._sample_to_proto(sample)
    pitched_wrist = np.eye(4)
    pitch = np.radians(-10.0)
    pitched_wrist[:3, 0] = [np.cos(pitch), 0.0, np.sin(pitch)]
    second = publisher._sample_to_proto(
        WebXRHandSample(
            sequence_id=2,
            timestamp_ns=43,
            landmarks=landmarks,
            wrist_matrix=pitched_wrist,
        )
    )

    expected = landmarks[list(WEBXR_TO_RETARGETER_LANDMARK_INDICES)].astype(np.float32)
    np.testing.assert_allclose(np.asarray(first.keypoints).reshape(21, 3), expected)
    assert first.timestamp_ns == 42
    assert first.handedness == "right"
    assert first.wrist_angle_degrees == pytest.approx(0.0)
    assert second.wrist_angle_degrees == pytest.approx(10.0)


def _unused_local_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_websocket_frame_reaches_grpc_ingress():
    aiohttp = pytest.importorskip("aiohttp")
    landmarks_q: queue.Queue = queue.Queue(maxsize=4)
    stop_event = threading.Event()
    ingress = IngressServer(landmarks_q, stop_event, port=0)
    grpc_port = ingress.start()
    web_port = _unused_local_port()
    publisher = MetaQuestPublisher(
        server_address=f"localhost:{grpc_port}",
        quest_host="127.0.0.1",
        quest_port=web_port,
        fps=60,
        wrist_enabled=False,
    )
    publisher_thread = threading.Thread(target=publisher.run, daemon=True)
    publisher_thread.start()

    async def send_when_ready() -> None:
        async with aiohttp.ClientSession() as session:
            for _ in range(100):
                try:
                    async with session.ws_connect(f"http://127.0.0.1:{web_port}/ws") as ws:
                        await ws.send_json(_webxr_payload())
                        await asyncio.sleep(0.1)
                        return
                except aiohttp.ClientError:
                    await asyncio.sleep(0.02)
        raise AssertionError("Quest WebXR bridge did not become ready")

    try:
        asyncio.run(send_when_ready())
        item = landmarks_q.get(timeout=3.0)

        assert isinstance(item, HandLandmarks)
        assert item.keypoints.shape == (21, 3)
        assert item.handedness == "right"
    finally:
        publisher.stop()
        stop_event.set()
        publisher_thread.join(timeout=3.0)
        ingress.stop()

    assert not publisher_thread.is_alive()
