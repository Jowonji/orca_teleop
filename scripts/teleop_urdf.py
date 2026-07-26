"""Teleoperate a kinematic ORCA URDF viewer.

This runs the same ingress + retargeter stack as ``teleop_sim.py``, but the
sink is a Meshcat/Pinocchio URDF viewer instead of a MuJoCo environment. It is
useful for separating retargeter responsiveness from MuJoCo physics/rendering.

Examples:
    # One-command local MediaPipe teleop against the right-hand v2 URDF.
    python scripts/teleop_urdf.py --hand right --local --show-video

    # Wait for an external publisher.
    python scripts/teleop_urdf.py --hand right --port 50052
"""

from __future__ import annotations

import argparse
import logging

from orca_teleop.ingress.server import DEFAULT_PORT
from orca_teleop.pipeline import run, run_local
from orca_teleop.urdf_viz import OrcaHandUrdfVizSink


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orca Hand teleoperation pipeline against a kinematic URDF viewer."
    )
    parser.add_argument("--model_path", default=None, help="OrcaHand model config")
    parser.add_argument("--urdf_path", default=None, help="Hand URDF file")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"gRPC port (default: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--hand",
        default="right",
        choices=["left", "right"],
        help="Hand side for both the URDF and the local publisher (default: right)",
    )
    parser.add_argument(
        "--version",
        default="v2",
        help="ORCA embodiment version for default model/URDF resolution (default: v2)",
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=90.0,
        help="Meshcat display update rate (default: 90 Hz)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Start Meshcat without opening a browser tab automatically",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Also launch a local MediaPipe publisher for one-command teleop",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.7, help="MediaPipe confidence (default: 0.7)"
    )
    parser.add_argument("--show-video", action="store_true", help="Show webcam feed with landmarks")
    parser.add_argument(
        "--retargeter",
        default="adaptive_analytical",
        choices=["rmsprop", "adaptive_analytical"],
        help="Retargeter backend (default: adaptive_analytical)",
    )
    parser.add_argument(
        "--retarget-config",
        default=None,
        help="YAML config for --retargeter adaptive_analytical",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    sink = OrcaHandUrdfVizSink(
        hand_type=args.hand,
        version=args.version,
        model_path=args.model_path,
        urdf_path=args.urdf_path,
        rate_hz=args.rate_hz,
        open_browser=not args.no_browser,
    )

    if args.local:
        run_local(
            model_path=args.model_path,
            urdf_path=args.urdf_path,
            port=args.port,
            handedness=args.hand,
            confidence=args.confidence,
            show_video=args.show_video,
            sink=sink,
            retargeter_backend=args.retargeter,
            retargeter_config_path=args.retarget_config,
        )
    else:
        run(
            model_path=args.model_path,
            urdf_path=args.urdf_path,
            port=args.port,
            sink=sink,
            retargeter_backend=args.retargeter,
            retargeter_config_path=args.retarget_config,
        )


if __name__ == "__main__":
    main()
