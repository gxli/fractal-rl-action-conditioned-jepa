"""Plots for the repeat-4, long-effective-horizon SAC/JEPA sweep."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("dumps")
OUT = ROOT / "ar4_analysis"
ARMS = [("ar4_sac", "SAC, separate", "#334155"), ("ar4_sac_sharedcritic", "SAC, critic-trained shared", "#64748b"), ("ar4_jepa_k32", "JEPA K=32, critic-trained", "#2563eb"), ("ar4_jepa_k64", "JEPA K=64, critic-trained", "#dc2626"), ("ar4_jepa_actorstop_k32", "JEPA K=32, actor-owned", "#9333ea"), ("ar4_jepa_separate_k32", "JEPA K=32, separate (source only)", "#f59e0b"), ("ar4_jepa_sigreg_k32", "JEPA + SIGReg K=32 (source only)", "#0f766e"), ("ar4_jepa_spherical_k32", "JEPA + spherical MMD K=32", "#059669")]

def read(run):
    with open(ROOT / run / "metrics" / "training.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {key: np.asarray([float(row[key]) for row in rows]) for key in rows[0]}

def run_name(prefix, split):
    return f"{prefix}_source_a3" if split == "source" else f"{prefix}_frozen_{split}"

def speed(ax, split, title):
    for prefix, label, color in ARMS:
        if not (ROOT / run_name(prefix, split) / "metrics" / "training.csv").exists():
            continue
        data = read(run_name(prefix, split)); x = data["steps"] * 4 / 1000
        y, s = data["evaluation_mean_speed"], data["evaluation_std_speed"]
        ax.plot(x, y, marker="o", ms=3.5, lw=1.8, label=label, color=color)
        ax.fill_between(x, y - s, y + s, color=color, alpha=.12)
    ax.set(title=title, xlabel="physics steps (thousands)", ylabel="held-out mean speed", xlim=(0, 80)); ax.grid(alpha=.2)

def main():
    OUT.mkdir(exist_ok=True); plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.6), constrained_layout=True)
    speed(axes[0], "source", "Source: alpha = 3"); speed(axes[1], "a2", "Frozen encoder transfer: alpha = 2"); speed(axes[2], "a3.5", "Frozen encoder transfer: alpha = 3.5")
    axes[0].legend(frameon=False, fontsize=6.4, loc="lower right"); fig.savefig(OUT / "actionrepeat4_speed_curves.png", dpi=220); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.5), constrained_layout=True)
    for prefix, label, color in ARMS[1:]:
        data = read(run_name(prefix, "source")); x = data["steps"] * 4 / 1000
        axes[0].plot(x, data["jepa_loss"], marker="o", ms=3, label=label, color=color)
        axes[1].plot(x, data["jepa_real_to_shuffled_ratio"], marker="o", ms=3, label=label, color=color)
        axes[2].plot(x, data["jepa_embedding_std"], marker="o", ms=3, label=label, color=color)
    for ax, title, ylabel in zip(axes, ["JEPA prediction loss", "Action-shuffle loss ratio", "Online embedding standard deviation"], ["cosine distance", "real / shuffled", "standard deviation"]):
        ax.set(title=title, xlabel="physics steps (thousands)", ylabel=ylabel, xlim=(0, 80)); ax.grid(alpha=.2)
    axes[0].legend(frameon=False, fontsize=7.2); fig.savefig(OUT / "actionrepeat4_jepa_diagnostics.png", dpi=220); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.5), constrained_layout=True)
    for prefix, label, color in ARMS:
        data = read(run_name(prefix, "source")); x = data["steps"] * 4 / 1000
        axes[0].plot(x, data["actor_loss"], marker="o", ms=3, label=label, color=color)
        axes[1].plot(x, data["critic_loss"], marker="o", ms=3, label=label, color=color)
        axes[2].plot(x, data["actor_total_loss"], marker="o", ms=3, label=label, color=color)
    for ax, title, ylabel in zip(axes, ["SAC actor objective", "TD critic objective", "Total actor objective"], ["actor loss", "critic loss", "actor loss"]):
        ax.set(title=title, xlabel="physics steps (thousands)", ylabel=ylabel, xlim=(0, 80)); ax.grid(alpha=.2)
    axes[0].legend(frameon=False, fontsize=7.2); fig.savefig(OUT / "actionrepeat4_losses.png", dpi=220); plt.close(fig)
    splits = [("source", "alpha = 3"), ("a2", "frozen alpha = 2"), ("a3.5", "frozen alpha = 3.5")]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.4), constrained_layout=True)
    for ax, (split, title) in zip(axes, splits):
        values = [(read(run_name(prefix, split))["evaluation_mean_speed"][-1], read(run_name(prefix, split))["evaluation_std_speed"][-1], label.replace(", ", "\n"), color) for prefix, label, color in ARMS if (ROOT / run_name(prefix, split) / "metrics" / "training.csv").exists()]
        x = np.arange(len(values)); ax.bar(x, [v[0] for v in values], yerr=[v[1] for v in values], capsize=3, color=[v[3] for v in values], alpha=.9)
        ax.set(title=title, ylabel="final held-out speed", ylim=(2.8, 3.2)); ax.set_xticks(x, [v[2] for v in values], fontsize=7); ax.grid(axis="y", alpha=.2)
    fig.savefig(OUT / "actionrepeat4_final_speeds.png", dpi=220)

if __name__ == "__main__":
    main()
