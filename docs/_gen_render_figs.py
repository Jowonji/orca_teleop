"""Regenerate the figures embedded in docs/sim-render-bottleneck-report.md."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from orca_teleop.sim import OrcaHandSimSink, SimCameraConfig

OUT = Path(__file__).resolve().parent / "assets"
OUT.mkdir(exist_ok=True)

# --- Figure 1: rendered observation, shadows on vs off ---------------------
frames = {}
for shadows in (True, False):
    sink = OrcaHandSimSink(
        env_name="right",
        version="v2",
        render_mode=None,
        camera_config=SimCameraConfig(name="frontal", width=320, height=240, shadows=shadows),
    )
    sink.connect()
    frames[shadows] = sink.get_observation().images["frontal"]
    sink.close()

fig, axes = plt.subplots(1, 2, figsize=(9, 4.0))
for ax, shadows, title in (
    (axes[0], True, "BEFORE  shadows ON\n288.4 ms/frame  (max 3.5 fps)"),
    (axes[1], False, "AFTER  shadows OFF\n52.7 ms/frame  (max 19.0 fps)"),
):
    ax.imshow(frames[shadows])
    ax.set_title(title, fontsize=10)
    ax.axis("off")
fig.suptitle("observation.images.frontal (320x240)", fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "render-quality-compare.png", dpi=130, bbox_inches="tight")
plt.close(fig)

diff = np.abs(frames[True].astype(int) - frames[False].astype(int))
print(f"mean abs pixel diff: {diff.mean():.1f}/255")

# --- Figure 2: where the frame time goes ----------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.4))

labels = ["BEFORE\nshadows ON", "AFTER\nshadows OFF"]
base = np.array([50.0, 50.0])  # geometry + rasterisation
shadow = np.array([238.4, 0.0])  # shadow map pass
ax1.barh(labels, base, color="#4C78A8", label="base render (geometry, raster)")
ax1.barh(labels, shadow, left=base, color="#E45756", label="8192px shadow map pass")
for y, total in enumerate([288.4, 52.7]):
    ax1.text(total + 6, y, f"{total:.1f} ms", va="center", fontsize=10, fontweight="bold")
ax1.set_xlabel("get_observation() latency per frame (ms)")
ax1.set_xlim(0, 350)
ax1.legend(fontsize=8, loc="lower right")
ax1.set_title("Where the frame time goes", fontsize=11)
ax1.invert_yaxis()

stages = ["mj_step\n(physics)", "update_scene", "render\n(before)", "render\n(after)"]
vals = [0.2, 0.03, 290.8, 52.7]
colors = ["#54A24B", "#54A24B", "#E45756", "#4C78A8"]
ax2.bar(stages, vals, color=colors)
ax2.set_yscale("log")
ax2.set_ylabel("ms per call (log scale)")
ax2.set_title("Per-stage cost: rendering dominates, physics is free", fontsize=11)
for i, v in enumerate(vals):
    ax2.text(i, v * 1.25, f"{v:g} ms", ha="center", fontsize=9)
ax2.set_ylim(0.01, 2000)

fig.tight_layout()
fig.savefig(OUT / "render-timing-breakdown.png", dpi=130, bbox_inches="tight")
plt.close(fig)

print(f"wrote {OUT}/render-quality-compare.png")
print(f"wrote {OUT}/render-timing-breakdown.png")
