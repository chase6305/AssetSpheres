"""Measure complete fits and held-out quality for legacy and mesh-distance grids.

No candidate backend is substituted automatically. The legacy surface-shell fill
and strict-center SDF grids have DIFFERENT sampling/radius semantics, so the
quality table matters as much as the speed table. GPU timings include transfers
and synchronization. The first call includes library/runtime preparation.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import trimesh

from assetspheres import FitConfig, fit_mesh
from assetspheres.quality import compare_sphere_models

BACKENDS = {
    "subdivide-cpu": ("subdivide", "cpu"),
    "subdivide-cuda": ("subdivide", "cuda"),
    "open3d-cpu": ("open3d", "cpu"),
    "warp-cpu": ("warp", "cpu"),
    "warp-cuda": ("warp", "cuda"),
}


def benchmark_shapes():
    return {
        "box": trimesh.creation.box(extents=(0.3, 0.16, 0.1)),
        "cylinder": trimesh.creation.cylinder(radius=0.05, height=0.3, sections=48),
        "sphere": trimesh.creation.icosphere(radius=0.1, subdivisions=3),
        "torus": trimesh.creation.torus(major_radius=0.09, minor_radius=0.025),
    }


def synchronize_gpu(device):
    if device == "cuda":
        import torch

        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark-voxelizers.json"))
    parser.add_argument("--backend", action="append", choices=list(BACKENDS))
    parser.add_argument("--shape", action="append", choices=list(benchmark_shapes()))
    parser.add_argument("--method", action="append", choices=["voxel", "refined"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--evaluation-seed", type=int, default=99101)
    parser.add_argument(
        "--cold", action="store_true", help="Also time one fresh cylinder process per backend"
    )
    parser.add_argument("--cold-probe", choices=list(BACKENDS), help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.repeats < 1:
        parser.error("repeats must be positive")
    selected_backends = arguments.backend or list(BACKENDS)
    selected_methods = arguments.method or ["voxel"]
    if arguments.cold_probe:
        voxelizer, device = BACKENDS[arguments.cold_probe]
        fit_mesh(
            benchmark_shapes()["cylinder"],
            FitConfig(
                method="voxel",
                voxelizer=voxelizer,
                device=device,
                count=24,
                resolution=48,
                padding=0.003,
                samples=3000,
            ),
        )
        return
    results = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "timing_scope": "full fit_mesh on preloaded mesh; includes grid/BVH and reports, no export",
        "cold_note": "First call includes runtime preparation; fresh-process timing is separate",
        "evaluation_seed": arguments.evaluation_seed,
        "shared_surface_samples": 6000,
        "shared_volume_samples": 20000,
        "configurations": {},
        "cases": {},
    }
    if arguments.cold:
        results["cold_cylinder_process_s"] = {}
        for backend_label in selected_backends:
            started = time.perf_counter()
            subprocess.run(
                [sys.executable, __file__, "--cold-probe", backend_label],
                check=True,
                capture_output=True,
                text=True,
            )
            results["cold_cylinder_process_s"][backend_label] = time.perf_counter() - started
    for shape_name, triangle_mesh in benchmark_shapes().items():
        if arguments.shape and shape_name not in arguments.shape:
            continue
        models = {}
        fit_reports = {}
        timings = {}
        for method in selected_methods:
            for backend_label in selected_backends:
                voxelizer, device = BACKENDS[backend_label]
                configuration = FitConfig(
                    method=method,
                    voxelizer=voxelizer,
                    device=device,
                    count=24,
                    resolution=48,
                    padding=0.003,
                    samples=3000,
                    refine_samples=1024,
                    iterations=120,
                )
                label = f"{method}/{backend_label}"
                results["configurations"][label] = asdict(configuration)
                durations = []
                for repetition in range(arguments.repeats + 1):
                    if repetition:
                        synchronize_gpu(device)
                    started = time.perf_counter()
                    spheres, report = fit_mesh(triangle_mesh, configuration)
                    synchronize_gpu(device)
                    durations.append(time.perf_counter() - started)
                models[label] = spheres
                fit_reports[label] = report
                timings[label] = {
                    "first_call_s": durations[0],
                    "warm_times_s": durations[1:],
                    "warm_median_s": statistics.median(durations[1:]),
                }
                print(
                    f"{shape_name:9} {label:26} first={durations[0]:.4f}s "
                    f"warm={timings[label]['warm_median_s']:.4f}s",
                    flush=True,
                )
        quality = compare_sphere_models(
            triangle_mesh,
            models,
            seed=arguments.evaluation_seed,
            surface_samples=6000,
            volume_samples=20000,
        )
        results["cases"][shape_name] = {
            "timings": timings,
            "quality": quality,
            "fit_reports": fit_reports,
            "triangle_count": len(triangle_mesh.faces),
        }
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(results, indent=2, allow_nan=False), encoding="utf-8"
        )
    print(arguments.output)


if __name__ == "__main__":
    main()
