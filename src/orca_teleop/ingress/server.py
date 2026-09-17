"""gRPC ingress server.

The robot side runs this server. Any publisher connects as a gRPC client and
streams ``HandFrame``messages. The server drops each frame onto a queue
consumed by the retargeter.

Typical usage inside the pipeline::

```python
    from orca_teleop.ingress import IngressServer

    server = IngressServer(landmarks_q, stop_event)
    server.start(port=50051) # publishes HandLandmarks to be consumed, e.g. by the retargeter
    server.stop()
```
"""

import logging
import queue
import threading
from concurrent import futures
from dataclasses import dataclass
from typing import Literal

import grpc
import numpy as np

from orca_teleop.constants import _COORDS_PER_POINT, _EXPECTED_LEN, _NUM_KEYPOINTS, DEFAULT_PORT

# MediaPipe wrist hint: image x, y, palm_width [+ palm X, Y, Z meters from RGB-D].
_ARM_HINT_LENS = (3, 6)
from orca_teleop.ingress import hand_stream_pb2, hand_stream_pb2_grpc

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HandLandmarks:
    """Canonical frame that travels on the landmarks queue."""

    keypoints: np.ndarray  # (21, 3) float32, wrist-relative, meters
    handedness: Literal["left", "right"]
    timestamp_ns: int
    wrist_angle_degrees: float = 0.0
    # (3,) image x, y, palm_width, or (6,) with camera-frame palm X, Y, Z (m, NaN if no depth).
    wrist_image: np.ndarray | None = None  # MediaPipe only


class _HandStreamServicer(hand_stream_pb2_grpc.HandStreamServicer):
    """Receives HandFrame streams from publishers and enqueues them."""

    def __init__(
        self,
        landmarks_q: queue.Queue,
        stop_event: threading.Event,
    ) -> None:
        self._q = landmarks_q
        self._stop = stop_event

    def StreamHandFrames(self, request_iterator, context):
        frames_received = 0
        peer = context.peer() or "unknown"
        logger.info("Publisher connected: %s", peer)

        try:
            for frame in request_iterator:
                if self._stop.is_set():
                    break

                n = len(frame.keypoints)
                wrist_image = None
                if n - _EXPECTED_LEN in _ARM_HINT_LENS:
                    raw = np.array(frame.keypoints, dtype=np.float32)
                    keypoints = raw[:_EXPECTED_LEN].reshape(_NUM_KEYPOINTS, _COORDS_PER_POINT)
                    wrist_image = raw[_EXPECTED_LEN:]
                elif n == _EXPECTED_LEN:
                    keypoints = np.array(frame.keypoints, dtype=np.float32).reshape(
                        _NUM_KEYPOINTS, _COORDS_PER_POINT
                    )
                else:
                    logger.warning(
                        "Dropping frame: expected %d (+%s hint) floats, got %d",
                        _EXPECTED_LEN,
                        "/".join(str(k) for k in _ARM_HINT_LENS),
                        n,
                    )
                    continue

                handedness = frame.handedness.lower()
                if handedness not in ("left", "right"):
                    logger.warning("Dropping frame: invalid handedness %r", frame.handedness)
                    continue

                landmark = HandLandmarks(
                    keypoints=keypoints,
                    handedness=handedness,
                    timestamp_ns=frame.timestamp_ns,
                    wrist_angle_degrees=frame.wrist_angle_degrees,
                    wrist_image=wrist_image,
                )

                # Always keep the latest frame; drop stale ones.
                try:
                    self._q.put_nowait(landmark)
                except queue.Full:
                    try:
                        self._q.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self._q.put_nowait(landmark)
                    except queue.Full:
                        pass

                frames_received += 1
        except Exception:
            logger.exception("Error in stream from %s", peer)
        finally:
            logger.info("Publisher disconnected: %s (received %d frames)", peer, frames_received)

        return hand_stream_pb2.StreamSummary(frames_received=frames_received)


class IngressServer:
    """gRPC server that accepts hand-frame streams from publishers.

    Args:
        landmarks_q: Queue onto which ``HandLandmarks`` are placed.
        stop_event: When set, the server drains and shuts down.
        port: TCP port to listen on.
        max_workers: gRPC thread-pool size.
    """

    def __init__(
        self,
        landmarks_q: queue.Queue,
        stop_event: threading.Event,
        port: int = DEFAULT_PORT,
        max_workers: int = 4,
    ) -> None:
        self._port = port
        self._executor = futures.ThreadPoolExecutor(max_workers=max_workers)
        self._server = grpc.server(self._executor)
        self._stopped = False
        servicer = _HandStreamServicer(landmarks_q, stop_event)
        hand_stream_pb2_grpc.add_HandStreamServicer_to_server(servicer, self._server)

    def start(self) -> int:
        """Start listening. Returns the actual port (useful when port=0)."""
        bound_port = self._server.add_insecure_port(f"[::]:{self._port}")
        self._server.start()
        logger.info("Ingress server listening on port %d", bound_port)
        return bound_port

    def stop(self, grace: float = 2.0) -> None:
        """Gracefully shut down the server."""
        if self._stopped:
            return
        self._stopped = True
        termination = self._server.stop(grace=grace)
        termination.wait(timeout=grace + 1.0)
        self._executor.shutdown(wait=True)
        logger.info("Ingress server stopped")

    def wait_for_termination(self, timeout: float | None = None) -> None:
        """Block until the server shuts down."""
        self._server.wait_for_termination(timeout=timeout)
