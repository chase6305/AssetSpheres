"""Persist a portable measurement snapshot and plot its quality trade-offs."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from assetspheres import __version__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/benchmark-voxelizers.json"))
    parser.add_argument("--output", type=Path, default=Path("docs"))
    parser.add_argument("--date", default=date.today().isoformat())
    arguments = parser.parse_args()
    results = json.loads(arguments.input.read_text())
    results["source_version"] = __version__
    results["date"] = arguments.date
    results["interpretation"] = (
        "Legacy shell filling and strict-center SDF sampling differ; speed and quality "
        "must be read together. Non-exclusive workstation, cached Warp binaries."
    )
    arguments.output.mkdir(parents=True, exist_ok=True)
    snapshot_path = arguments.output / f"voxelizers-{arguments.date}.json"
    snapshot_path.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    plotted_methods = {
        "voxel/subdivide-cpu": ("Voxel: legacy CPU", "#377eb8", "o"),
        "voxel/warp-cuda": ("Voxel: Warp CUDA", "#a65628", "D"),
        "refined/subdivide-cuda": ("Refined: legacy grid + CUDA", "#ff7f00", "P"),
        "refined/warp-cuda": ("Refined: Warp CUDA", "#e41a1c", "X"),
    }
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, (shape_name, measurements) in zip(axes.flat, results["cases"].items()):
        for method, (label, color, marker) in plotted_methods.items():
            quality = measurements["quality"].get(method)
            if quality is None:
                continue
            axis.scatter(
                quality["outside_volume_over_mesh"] * 100,
                quality["surface_coverage"] * 100,
                label=label,
                c=color,
                marker=marker,
                s=75,
                edgecolors="white",
                linewidths=0.6,
            )
        axis.set(
            xlim=(0, 100),
            ylim=(0, 103),
            title=shape_name.capitalize(),
            xlabel="Excess sphere-union volume / mesh volume (%)",
            ylabel="Surface coverage (%)",
        )
        axis.grid(True, alpha=0.2)
        axis.set_axisbelow(True)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.suptitle("Grid backends change speed and geometric fit", fontsize=17, y=0.98)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    figure.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.08, hspace=0.42, wspace=0.3)
    plot_path = arguments.output / "voxelizer-quality.png"
    figure.savefig(plot_path, dpi=160)
    print(snapshot_path)
    print(plot_path)


if __name__ == "__main__":
    main()
