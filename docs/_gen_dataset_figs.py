"""Sample frames out of the recorded video, one strip per episode.

Regenerates docs/assets/dataset-sample-frames.png for the data-collection report.
OpenCV cannot decode the AV1 stream LeRobot writes, so frames go through ffmpeg.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path("datasets/orca-sim-mediapipe")
OUT = Path("docs/assets")
OUT.mkdir(parents=True, exist_ok=True)

info = json.loads((ROOT / "meta/info.json").read_text())
fps = info["fps"]
ep = pd.read_parquet(next((ROOT / "data").rglob("*.parquet")))["episode_index"].to_numpy()
video = next((ROOT / "videos").rglob("*.mp4"))


def grab(global_frame: int, tmp: Path) -> np.ndarray:
    out = tmp / f"{global_frame}.png"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(global_frame / fps),
         "-i", str(video), "-frames:v", "1", "-y", str(out)],
        check=True,
    )
    return np.array(Image.open(out))


n_ep, n_col = int(ep.max()) + 1, 4
fig, axes = plt.subplots(n_ep, n_col, figsize=(n_col * 2.3, n_ep * 1.85))

with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    for e in range(n_ep):
        idx = np.where(ep == e)[0]
        for c, g in enumerate(np.linspace(idx[0], idx[-1] - 1, n_col).astype(int)):
            ax = axes[e, c]
            ax.imshow(grab(int(g), tmp))
            ax.axis("off")
            ax.set_title(f"t={(g - idx[0]) / fps:.1f}s", fontsize=7)
        axes[e, 0].text(
            -0.10, 0.5, f"ep{e}\n{len(idx)}f", transform=axes[e, 0].transAxes,
            fontsize=9, va="center", ha="right",
        )

fig.suptitle(
    f"observation.images.frontal — {len(ep)} frames, {fps} fps, 320x240", fontsize=11, y=1.0
)
fig.tight_layout()
fig.savefig(OUT / "dataset-sample-frames.png", dpi=130, bbox_inches="tight")
print(f"wrote {OUT}/dataset-sample-frames.png")
