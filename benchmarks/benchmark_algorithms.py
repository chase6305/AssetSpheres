"""Compare quality on shared held-out samples, plus complete fit latency.

Run: python benchmarks/benchmark_algorithms.py --cuda --repeats 3
Quality samples are NEVER fed back into either fitting or hyperparameter choice.
The common evaluation domain encloses all compared spheres, including protrusions.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from dataclasses import asdict, replace
from pathlib import Path

import trimesh

from assetspheres import FitConfig, fit_mesh
from assetspheres.quality import compare_sphere_models


def shapes():
    return {
        "box": trimesh.creation.box(extents=(0.3, 0.16, 0.1)),
        "cylinder": trimesh.creation.cylinder(radius=0.05, height=0.3, sections=48),
        "sphere": trimesh.creation.icosphere(radius=0.1, subdivisions=3),
        "torus": trimesh.creation.torus(major_radius=0.09, minor_radius=0.025),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark-algorithms.json"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--refine-samples", type=int, default=1024)
    parser.add_argument("--volume-samples", type=int, default=20000)
    parser.add_argument("--evaluation-seed", type=int, default=88003)
    parser.add_argument("--padding", type=float, default=0.003)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument(
        "--mesh-reference",
        action="store_true",
        help="Also benchmark original-mesh classification training",
    )
    parser.add_argument("--shape", action="append", choices=list(shapes()))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    config = FitConfig(
        count=args.count,
        samples=3000,
        iterations=args.iterations,
        refine_samples=args.refine_samples,
        padding=args.padding,
    )
    methods = [(method, "cpu") for method in ("voxel", "cover", "cover-tight", "slab", "refined")]
    if args.cuda:
        # Intentionally do not import torch until the first measured CUDA fit.
        methods.extend([("voxel", "cuda"), ("refined", "cuda")])
    results = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config": asdict(config),
        "repeats": args.repeats,
        "timing_scope": "full fit_mesh, preloaded mesh; no load/export/evaluation",
        "quality": f"common original-mesh Monte Carlo; held-out seed {args.evaluation_seed}",
        "cases": {},
    }
    for name, mesh in shapes().items():
        if args.shape and name not in args.shape:
            continue
        fitted, timings, reports = {}, {}, {}
        for method, device in methods:
            label = f"{method}/{device}"
            local = replace(config, method=method, device=device)
            durations = []
            for repeat in range(args.repeats + 1):
                if device == "cuda" and repeat > 0:
                    import torch

                    torch.cuda.synchronize()
                started = time.perf_counter()
                spheres, report = fit_mesh(mesh, local)
                if device == "cuda":
                    import torch

                    torch.cuda.synchronize()
                durations.append(time.perf_counter() - started)
            fitted[label] = spheres
            reports[label] = report
            timings[label] = {
                "first_call_s": durations[0],
                "warm_times_s": durations[1:],
                "warm_median_s": statistics.median(durations[1:]),
            }
            print(
                f"{name:10} {label:20} first={durations[0]:.4f}s "
                f"warm={timings[label]['warm_median_s']:.4f}s",
                flush=True,
            )
        if args.mesh_reference:
            local = replace(config, method="refined", refine_reference="mesh")
            duration = []
            for _ in range(args.repeats + 1):
                started = time.perf_counter()
                spheres, report = fit_mesh(mesh, local)
                duration.append(time.perf_counter() - started)
            label = "refined-mesh/cpu"
            fitted[label], reports[label] = spheres, report
            timings[label] = {
                "first_call_s": duration[0],
                "warm_times_s": duration[1:],
                "warm_median_s": statistics.median(duration[1:]),
            }
            print(f"{name:10} {label:20} warm={timings[label]['warm_median_s']:.4f}s", flush=True)
        quality = compare_sphere_models(
            mesh, fitted, volume_samples=args.volume_samples, seed=args.evaluation_seed
        )
        results["cases"][name] = {
            "triangles": len(mesh.faces),
            "timing": timings,
            "quality": quality,
            "fit_reports": reports,
        }
        # Save after each shape so completed evidence survives an interrupted run.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
