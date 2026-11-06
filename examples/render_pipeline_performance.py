"""Plot published fitting-pipeline measurements; requires the plots extra."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLORS = ("#8297ad", "#218b83")


def draw_panel(axis, labels, before, after, title, unit):
    positions = np.arange(len(labels))
    for offset, values, color in ((-0.19, before, COLORS[0]), (0.19, after, COLORS[1])):
        bars = axis.bar(positions + offset, values, width=0.34, color=color)
        axis.bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=4, fontsize=10)
    axis.set_xticks(positions, labels)
    axis.set_title(title, fontsize=12, pad=15)
    axis.set_ylabel(unit)
    axis.set_ylim(0, max(*before, *after) * 1.22)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#dce3e9", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/fitting-pipeline.png")
    args = parser.parse_args()
    data = json.loads((ROOT / "docs/fitting-pipeline.json").read_text())
    figure, panels = plt.subplots(2, 2, figsize=(12, 8))
    axes = panels.ravel()
    rows = [
        data["preparation_cover"]["cases"]["ico7/cold/fit"],
        data["preparation_surface"]["cases"]["ico7/cold/fit"],
    ]
    draw_panel(
        axes[0],
        ["cover", "surface"],
        [row["reference"]["traced_allocation_peak_bytes"] / 1e6 for row in rows],
        [row["optimized"]["traced_allocation_peak_bytes"] / 1e6 for row in rows],
        "Complete fit: 327,680 triangles",
        "Traced allocation peak (MB)",
    )
    rows = [data["preparation_surface"]["cases"][f"ico{level}/cold/fit"] for level in (6, 7)]
    draw_panel(
        axes[1],
        ["81,920 triangles", "327,680 triangles"],
        [row["reference"]["median_s"] * 1000 for row in rows],
        [row["optimized"]["median_s"] * 1000 for row in rows],
        "Complete surface fit, budget 24",
        "Elapsed time (ms)",
    )
    rows = [data["preparation_slab"]["cases"][f"ico{level}/cold/fit"] for level in (6, 7)]
    draw_panel(
        axes[2],
        ["81,920 triangles", "327,680 triangles"],
        [row["reference"]["median_s"] * 1000 for row in rows],
        [row["optimized"]["median_s"] * 1000 for row in rows],
        "Complete slab fit, budget 24",
        "Elapsed time (ms)",
    )
    cases = data["process_fits"]["synthetic"]["cases"]
    rows = [cases[label] for label in ("bare", "texture512", "texture4096")]
    draw_panel(
        axes[3],
        ["No texture", "512px PBR", "4096px PBR"],
        [row["reference"]["median_s"] * 1000 for row in rows],
        [row["optimized"]["median_s"] * 1000 for row in rows],
        "Four cover jobs, two spawn workers",
        "Elapsed time (ms)",
    )
    figure.suptitle("AssetSpheres: measured fitting-pipeline costs", fontsize=16)
    figure.legend(
        [plt.Rectangle((0, 0), 1, 1, color=color) for color in COLORS],
        ["Before (83c4df6)", "After"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=2,
        frameon=False,
    )
    figure.text(
        0.5,
        0.02,
        "11 alternating pairs; budget 24, 6,000 samples; nonexclusive host.\n"
        "Mesh inputs have no precomputed area cache. Allocation peaks are not RSS. "
        "Loading/export excluded.",
        ha="center",
        fontsize=9,
        color="#526477",
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.9), w_pad=2, h_pad=3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
