"""Plot the published paired measurements; requires the plots extra."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLORS = ("#8297ad", "#218b83")


def draw_bars(axis, labels, before, after, title, unit):
    positions = np.arange(len(labels))
    maximum = max(*before, *after)
    for offset, values, color in ((-0.19, before, COLORS[0]), (0.19, after, COLORS[1])):
        bars = axis.bar(positions + offset, values, width=0.34, color=color)
        axis.bar_label(
            bars,
            labels=[f"{value:.1f}" if unit == "MB" else f"{value:.3f}" for value in values],
            padding=4,
            fontsize=10,
        )
    axis.set_xticks(positions, labels)
    axis.set_title(title, fontsize=12, pad=13)
    axis.set_ylabel("Traced allocations (MB)" if unit == "MB" else "Full fit (s)")
    axis.set_ylim(0, maximum * 1.22)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#dce3e9", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/fitting-performance.png")
    args = parser.parse_args()
    data = json.loads((ROOT / "docs/fitting-performance.json").read_text())
    plt.rcParams["font.family"] = "DejaVu Sans"
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    grid = data["grid_preparation_and_fits"]["cases"]
    paths = ("subdivide/cpu", "open3d/cpu", "warp/cuda")
    before, after = [], []
    for backend in paths:
        row = grid[f"{backend}/cube112/refined-cover"]
        before.append(row["reference"]["traced_allocation_peak_bytes"] / 1e6)
        after.append(row["optimized"]["traced_allocation_peak_bytes"] / 1e6)
    draw_bars(
        axes[0],
        ["subdivide", "Open3D", "Warp CUDA"],
        before,
        after,
        "refined-cover: cube112, budget 32",
        "MB",
    )
    before, after = [], []
    for kind in ("gaussian", "shell"):
        row = data["enclosing_clouds"]["cases"][f"{kind}/C/1000000"]
        before.append(row["reference"]["traced_allocation_peak_bytes"] / 1e6)
        after.append(row["optimized"]["traced_allocation_peak_bytes"] / 1e6)
    draw_bars(
        axes[1], ["Gaussian", "Shell"], before, after, "tighten_ball: 1M points, C layout", "MB"
    )
    before, after = [], []
    for robot in ("ur5e", "panda"):
        rows = data["complete_cover_fits"][robot]["cases"]
        selected = [row for name, row in rows.items() if name.endswith("/cover/N24")]
        before.append(sum(row["reference"]["median_s"] for row in selected))
        after.append(sum(row["optimized"]["median_s"] for row in selected))
    draw_bars(axes[2], ["UR5e", "Panda arm"], before, after, "cover: 24 balls per link", "s")
    figure.suptitle("AssetSpheres: measured memory and fitting time", fontsize=16, y=1.01)
    figure.legend(
        [plt.Rectangle((0, 0), 1, 1, color=color) for color in COLORS],
        ["Before (e4dab66)", "After"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=2,
        frameon=False,
    )
    figure.text(
        0.5,
        0.01,
        "Paired warm runs; allocation peaks are not RSS or device memory. "
        "Robot times sum per-link medians; nonexclusive host.",
        ha="center",
        fontsize=10,
        color="#526477",
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.88), w_pad=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
