"""Live preview of OpenCV cameras before recording a dataset.

Use the same ``--camera NAME:INDEX[:WIDTHxHEIGHT][@FPS]`` specs as
``record_dataset.py``. Open a window per camera, physically aim them, then
press ``q`` / Esc to quit.

Examples::

    # Discover indices
    uv run python scripts/preview_cameras.py --list-cameras

    # Preview the iPhone Continuity Camera at index 1
    uv run python scripts/preview_cameras.py --camera iphone:1

    # Multiple cameras
    uv run python scripts/preview_cameras.py --camera front:0 --camera wrist:1
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from orca_teleop.cameras import (
    CameraManager,
    parse_camera_spec,
    print_available_cameras,
)

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if "--list-cameras" in argv:
        print_available_cameras()
        return 0

    parser = argparse.ArgumentParser(
        description="Live-preview OpenCV cameras so you can aim them before recording."
    )
    parser.add_argument(
        "--camera",
        action="append",
        default=[],
        metavar="SPEC",
        help=(
            "Camera spec NAME[:INDEX][:WIDTHxHEIGHT][@FPS], repeatable. "
            "Same format as record_dataset.py."
        ),
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Probe available camera indices and exit.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Preview refresh rate (default: 30).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.camera:
        parser.error("pass at least one --camera SPEC, or use --list-cameras")

    configs = [parse_camera_spec(spec) for spec in args.camera]
    manager = CameraManager(configs)

    import cv2

    period = 1.0 / max(args.fps, 1e-3)
    try:
        shapes = manager.open()
        logger.info("Previewing %d camera(s): %s", len(shapes), shapes)
        print("Aim the cameras as needed. Press 'q' or Esc to quit.")

        while True:
            t0 = time.monotonic()
            frames = manager.capture()
            for name, rgb in frames.items():
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imshow(f"preview: {name}", bgr)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

            elapsed = time.monotonic() - t0
            if elapsed < period:
                time.sleep(period - elapsed)
    except KeyboardInterrupt:
        print()
    finally:
        manager.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
