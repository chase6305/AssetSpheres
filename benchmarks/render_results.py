"""Save a portable benchmark snapshot and plot the measured quality trade-offs."""

from __future__ import annotations

import argparse
import json
import math
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from assetspheres import __version__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, default=Path("docs"))
    parser.add_argument("--date", default=date.today().isoformat())
    args = parser.parse_args()
    algorithms = json.loads((args.input / "benchmark-algorithms.json").read_text())
    tensor = json.loads((args.input / "benchmark-tensor.json").read_text())
    parallel = json.loads((args.input / "benchmark-parallel.json").read_text())
    environment = parallel["environment"]
    snapshot = {
        "date": args.date,
        "source_version": __version__,
        "environment": {
            key: environment[key]
            for key in (
                "cpu_model",
                "logical_cpu_count",
                "physical_core_count_from_proc",
                "thread_environment",
                "architecture",
            )
        },
        "parallel": {
            "repeats": parallel["repeats"],
            "timing_scope": parallel["timing_scope"],
            "cases": [
                {
                    key: case[key]
                    for key in (
                        "name",
                        "link_count",
                        "configs",
                        "mesh_pickle_bytes_sum",
                        "measurements",
                        "sphere_output_sha256",
                    )
                }
                for case in parallel["cases"]
            ],
        },
        "tensor": tensor,
        "algorithms": algorithms,
        "limitations": [
            "Non-exclusive workstation with desktop and another GPU process",
            "Monte Carlo estimates, not exact volume integration",
            "No native Lula or downstream planner integration benchmark",
            "Algorithm timings include full fit_mesh but exclude mesh loading/export",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"benchmarks-{args.date}.json").write_text(
        json.dumps(snapshot, indent=2, allow_nan=False), encoding="utf-8"
    )
    methods = {
        "voxel/cpu": ("Voxel", "#377eb8", "o"),
        "cover/cpu": ("Cover", "#888888", "s"),
        "cover-tight/cpu": ("Cover tight", "#1b9e77", "D"),
        "slab/cpu": ("Slab", "#984ea3", "^"),
        "refined/cpu": ("Refined: voxel labels", "#ff7f00", "P"),
        "refined-mesh/cpu": ("Refined: mesh labels", "#e41a1c", "X"),
    }
    cases = list(algorithms["cases"].items())
    rows = math.ceil(len(cases) / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(11, rows * 3.4 + 1.2), squeeze=False)
    for axis, (name, case) in zip(axes.flat, cases):
        for method, (label, color, marker) in methods.items():
            metrics = case["quality"].get(method)
            if metrics is None:
                continue
            axis.scatter(
                metrics["outside_volume_over_mesh"] * 100,
                metrics["surface_coverage"] * 100,
                label=label,
                c=color,
                marker=marker,
                s=70,
                alpha=0.9,
                edgecolors="white",
                linewidths=0.6,
            )
        axis.set_xscale("log")
        axis.set(
            xlim=(3, 1200),
            ylim=(40, 103),
            title=name.capitalize(),
            xlabel="Excess sphere-union volume / mesh volume (%)",
            ylabel="Surface coverage (%)",
        )
        axis.grid(True, alpha=0.2)
        axis.set_axisbelow(True)
    for axis in list(axes.flat)[len(cases) :]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle("Surface coverage and excess volume", fontsize=17, y=0.98)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.08, hspace=0.42, wspace=0.3)
    fig.savefig(args.output / "benchmark-pareto.png", dpi=160)
    print(args.output / "benchmark-pareto.png")


if __name__ == "__main__":
    main()
