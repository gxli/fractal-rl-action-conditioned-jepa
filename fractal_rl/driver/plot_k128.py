import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUNS = Path("dumps")
OUT = RUNS / "ar4k128_analysis"
OUT.mkdir(exist_ok=True)
ARMS = [
    ("sac", "SAC (separate encoders)", "#111827"),
    ("sac_shared", "SAC (shared critic encoder)", "#16a34a"),
    ("jepa_shared", "SAC+JEPA K=128 (shared critic encoder)", "#dc2626"),
    ("jepa_sigreg_shared", "SAC+JEPA K=128 + SIGReg (shared critic-trained encoder)", "#7c3aed"),
]
PANELS = [
    ("source_a3", "source alpha=3"),
    ("frozen_a2", "frozen alpha=2"),
    ("frozen_a3.5", "frozen alpha=3.5"),
]


def load(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    return (
        [float(row["steps"]) * 4 / 1000 for row in rows],
        [float(row["evaluation_mean_speed"]) for row in rows],
        [float(row["evaluation_std_speed"]) for row in rows],
    )


fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.5), constrained_layout=True)
fig.patch.set_facecolor("white")
for ax, (stage, title) in zip(axes, PANELS):
    for arm, label, color in ARMS:
        path = RUNS / f"ar4k128_{arm}_{stage}" / "metrics" / "training.csv"
        if not path.exists():
            continue
        x, y, std = load(path)
        ax.plot(x, y, "-o", ms=3, label=label, color=color)
        ax.fill_between(x, [a - b for a, b in zip(y, std)], [a + b for a, b in zip(y, std)], color=color, alpha=.12)
    ax.set(title=title, xlabel="physics steps (thousands)", ylabel="held-out speed", xlim=(0, 80))
    ax.grid(alpha=.2)
axes[0].legend(frameon=False, fontsize=6.3, loc="lower right")
fig.savefig(OUT / "k128_curves_white.png", dpi=220, facecolor="white", transparent=False)
