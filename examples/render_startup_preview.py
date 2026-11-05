"""Plot published startup and preview timings; requires the plots extra."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLORS = ("#8297ad", "#218b83")


def draw_panel(axis, cases, keys, labels, scope, title):
    positions = np.arange(len(keys))
    maximum = 0
    for offset, implementation, color in zip((-0.19, 0.19), ("reference", "optimized"), COLORS):
        values = [
            (cases[key][scope] if scope else cases[key])[implementation]["median_s"] * 1000
            for key in keys
        ]
        maximum = max(maximum, *values)
        bars = axis.bar(positions + offset, values, width=0.34, color=color)
        axis.bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=4, fontsize=10)
    axis.set_xticks(positions, labels)
    axis.set_title(title, fontsize=12, pad=15)
    axis.set_ylabel("Median elapsed time (ms)")
    axis.set_ylim(0, maximum * 1.22)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#dce3e9", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/startup-and-preview.png")
    args = parser.parse_args()
    data = json.loads((ROOT / "docs/startup-and-preview.json").read_text())
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    draw_panel(
        axes[0],
        data["startup"]["cases"],
        ("help", "methods-json", "report-json", "import-fit-api"),
        ("CLI help", "Methods JSON", "Saved report", "Import fit API"),
        None,
        "Fresh interpreter; warm filesystem caches",
    )
    draw_panel(
        axes[1],
        data["preview"]["cases"],
        ("none/image0", "simple/image4096", "pbr/image4096"),
        ("No texture", "4096px Simple", "4096px PBR"),
        "construction_and_glb",
        "Preloaded preview construction + GLB serialization",
    )
    figure.suptitle("AssetSpheres: measured startup and preview time", fontsize=16)
    figure.legend(
        [plt.Rectangle((0, 0), 1, 1, color=color) for color in COLORS],
        ["Before (a8ebfc6)", "After"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=2,
        frameon=False,
    )
    figure.text(
        0.5,
        0.02,
        "21 alternating paired runs; nonexclusive host. Preview input: 642 vertices, 24 spheres. "
        "No fitting or file writes timed.",
        ha="center",
        fontsize=9,
        color="#526477",
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.86), w_pad=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
