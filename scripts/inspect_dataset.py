"""Summarise a recorded LeRobot dataset: motion, joint usage, and tracking quality.

Answers the questions you actually care about after a recording session:
is there real motion in each episode, which joints were exercised, and does the
sim hand follow the retargeted command?

    python scripts/inspect_dataset.py --root datasets/orca-sim-mediapipe
    python scripts/inspect_dataset.py --root datasets/orca-sim-mediapipe --plot out.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# A frame whose commanded pose barely changed from the previous one. Teleop
# jitter alone keeps an actively moved hand well above this.
IDLE_DEG_PER_FRAME = 0.05


def load(root: Path):
    info = json.loads((root / "meta/info.json").read_text())
    df = pd.read_parquet(next((root / "data").rglob("*.parquet")))
    return (
        info["features"]["action"]["names"],
        info["fps"],
        np.stack(df["action"].to_numpy()),
        np.stack(df["observation.state"].to_numpy()),
        df["episode_index"].to_numpy(),
    )


def report(joints, fps, action, state, ep) -> None:
    print(f"{len(action)} frames | {ep.max() + 1} episodes | {fps} fps | {len(joints)} joints\n")

    print("--- per episode " + "-" * 55)
    print(f"{'ep':>3}{'frames':>8}{'sec':>7}{'motion':>12}{'idle':>9}  verdict")
    for e in range(ep.max() + 1):
        a = action[ep == e]
        d = np.abs(np.diff(a, axis=0)).mean(axis=1)
        idle = (d < IDLE_DEG_PER_FRAME).mean() * 100
        verdict = "ok" if idle < 20 and d.mean() > 0.3 else "SUSPECT"
        print(f"{e:>3}{len(a):>8}{len(a) / fps:>7.1f}{d.mean():>9.2f} deg/f{idle:>8.1f}%  {verdict}")

    print("\n--- joint usage (commanded range, deg) " + "-" * 32)
    rng = action.max(axis=0) - action.min(axis=0)
    for i in np.argsort(-rng):
        bar = "#" * int(rng[i] / max(rng.max(), 1e-9) * 30)
        note = "  <- never moved" if rng[i] < 1.0 else ""
        print(
            f"{joints[i]:<13}{action[:, i].min():>8.1f}{action[:, i].max():>8.1f}"
            f"{rng[i]:>8.1f}  {bar}{note}"
        )

    print("\n--- tracking (does state follow action?) " + "-" * 30)
    err = np.abs(state - action)
    print(f"mean abs error {err.mean():.2f} deg | median {np.median(err):.2f} deg")
    j = int(np.argmax(rng))
    lags = [(np.corrcoef(action[: len(action) - k, j], state[k:, j])[0, 1], k) for k in range(16)]
    corr, lag = max(lags)
    print(f"most active joint '{joints[j]}': best lag {lag} frames "
          f"({lag / fps * 1000:.0f} ms), correlation {corr:.3f}")


def plot(joints, fps, action, state, ep, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 7))

    speed = np.r_[0, np.abs(np.diff(action, axis=0)).mean(axis=1)]
    t = np.arange(len(speed)) / fps
    ax0.plot(t, speed, lw=0.7, color="#4C78A8")
    ax0.axhline(IDLE_DEG_PER_FRAME, color="#E45756", ls="--", lw=1,
                label=f"idle threshold ({IDLE_DEG_PER_FRAME} deg/frame)")
    ax0.fill_between(t, 0, speed.max(), where=speed < IDLE_DEG_PER_FRAME,
                     color="#E45756", alpha=0.18, step="mid")
    for e in range(ep.max() + 1):
        start = np.argmax(ep == e) / fps
        ax0.axvline(start, color="grey", lw=1)
        ax0.text(start + 0.4, speed.max() * 0.92, f"ep{e}", fontsize=9, color="grey")
    ax0.set(xlabel="session time (s)", ylabel="mean |d action| (deg/frame)", xlim=(0, t[-1]))
    ax0.set_title("Motion timeline - red shading = hand not moving", fontsize=11)
    ax0.legend(fontsize=8, loc="upper right")

    j = int(np.argmax(action.max(axis=0) - action.min(axis=0)))
    m = ep == min(1, ep.max())
    tt = np.arange(m.sum()) / fps
    ax1.plot(tt, action[m, j], lw=1.4, color="#E45756", label="action  (commanded)")
    ax1.plot(tt, state[m, j], lw=1.4, color="#4C78A8", label="observation.state  (measured)")
    ax1.set(xlabel=f"episode {min(1, ep.max())} time (s)", ylabel="angle (deg)", xlim=(0, tt[-1]))
    ax1.set_title(f"'{joints[j]}' - sim hand tracking the command", fontsize=11)
    ax1.legend(fontsize=9)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nwrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Dataset root directory.")
    parser.add_argument("--plot", type=Path, default=None, help="Also write a PNG summary here.")
    args = parser.parse_args()

    joints, fps, action, state, ep = load(args.root)
    report(joints, fps, action, state, ep)
    if args.plot is not None:
        plot(joints, fps, action, state, ep, args.plot)


if __name__ == "__main__":
    main()
