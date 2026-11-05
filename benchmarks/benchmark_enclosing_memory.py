"""Compare complete enclosing-ball calls or mesh fits against a prior checkout.

Inputs are prepared before measurement. Calls alternate order after warm-up;
tracemalloc measures allocations separately, not RSS or native runtime memory.
Centers, radii and diagnostic reports must match the reference exactly.
The reference supplies enclosing.py, fit.py and surface_fit.py; other shared
helpers come from the current checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import statistics
import sys
import time
import tracemalloc
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import scipy
import trimesh

import assetspheres.enclosing as optimized
import assetspheres.fit as fitting
import assetspheres.slab as slab
import assetspheres.surface_fit as surface_fitting
from assetspheres import FitConfig, load_asset


def compare(reference, current, equal, repeats):
    assert equal(reference(), current()), "Reference and current outputs differ"
    calls = {"reference": reference, "optimized": current}
    times = {name: [] for name in calls}
    for repeat in range(repeats):
        for name in list(calls)[:: 1 if repeat % 2 == 0 else -1]:
            started = time.perf_counter()
            calls[name]()
            times[name].append(time.perf_counter() - started)
    records = {"exact_output_match": True}
    for name, call in calls.items():
        tracemalloc.start()
        try:
            call()
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        records[name] = {
            "times_s": times[name],
            "median_s": statistics.median(times[name]),
            "traced_allocation_peak_bytes": peak,
        }
    return records


def load_reference(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def dense_meshes():
    box = trimesh.creation.box(extents=(0.3, 0.16, 0.1))
    for _ in range(6):
        box = box.subdivide()
    return {
        "icosphere6": trimesh.creation.icosphere(subdivisions=6, radius=0.1),
        "box6": box,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", type=Path, required=True, help="Previous src/assetspheres directory"
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark-enclosing.json"))
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--points", type=int, action="append")
    parser.add_argument("--seed", type=int, default=591041)
    meshes = parser.add_mutually_exclusive_group()
    meshes.add_argument("--input", type=Path, help="Compare complete fits on a URDF/mesh")
    meshes.add_argument("--dense", action="store_true", help="Compare fits on dense test meshes")
    parser.add_argument("--count", type=int, action="append", help="Per-link full-fit ball budget")
    parser.add_argument(
        "--method", choices=("cover", "cover-tight", "cover-opt", "slab"), action="append"
    )
    args = parser.parse_args()
    if args.repeats < 1 or any(count < 1 for count in args.points or []):
        parser.error("repeats and point counts must be positive")
    for name in ("enclosing.py", "fit.py", "surface_fit.py"):
        if not (args.reference / name).is_file():
            parser.error(f"--reference must contain the previous {name}")
    reference = load_reference(args.reference / "enclosing.py", "assetspheres._enclosing_reference")
    reference_fit = load_reference(
        args.reference / "fit.py", "assetspheres._enclosing_fit_reference"
    )
    reference_surface = load_reference(
        args.reference / "surface_fit.py", "assetspheres._surface_fit_reference"
    )
    records = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "trimesh": trimesh.__version__,
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "threads": {
            name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
        },
        "reference_sha256": hashlib.sha256(
            (args.reference / "enclosing.py").read_bytes()
        ).hexdigest(),
        "optimized_sha256": hashlib.sha256(Path(optimized.__file__).read_bytes()).hexdigest(),
        "reference_fit_sha256": hashlib.sha256(
            Path(reference_fit.__file__).read_bytes()
        ).hexdigest(),
        "optimized_fit_sha256": hashlib.sha256(Path(fitting.__file__).read_bytes()).hexdigest(),
        "reference_surface_sha256": hashlib.sha256(
            (args.reference / "surface_fit.py").read_bytes()
        ).hexdigest(),
        "optimized_surface_sha256": hashlib.sha256(
            Path(surface_fitting.__file__).read_bytes()
        ).hexdigest(),
        "seed": args.seed,
        "repeats": args.repeats,
        "timing_scope": (
            "complete warm fit_mesh, preloaded input; no load/export"
            if args.input or args.dense
            else "complete warm tighten_ball call, input preparation excluded"
        ),
        "memory_scope": "tracemalloc allocation peak, not RSS or native runtimes",
        "cases": {},
    }
    generator = np.random.default_rng(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save(label, row):
        records["cases"][label] = row
        args.output.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
        print(label, "exact match", flush=True)

    if args.input or args.dense:
        if args.input:
            asset = load_asset(args.input, geometry="collision")
            selected_meshes = {name: link.mesh for name, link in asset.links.items()}
            records["input"] = {
                "path": f"{asset.path.parent.name}/{asset.path.name}",
                "sha256": hashlib.sha256(asset.path.read_bytes()).hexdigest(),
            }
        else:
            selected_meshes = dense_meshes()
        for name, mesh in selected_meshes.items():
            for count in args.count or (1, 24, 64):
                for method in dict.fromkeys(
                    args.method or ("cover", "cover-tight", "cover-opt", "slab")
                ):
                    configuration = FitConfig(method=method, count=count, padding=0.003)

                    def old_fit():
                        with (
                            patch.object(optimized, "tighten_ball", reference.tighten_ball),
                            patch.object(slab, "tighten_ball", reference.tighten_ball),
                            patch.object(
                                reference_fit,
                                "fit_triangle_cover",
                                reference_surface.fit_triangle_cover,
                            ),
                        ):
                            return reference_fit.fit_mesh(mesh, configuration)

                    row = compare(
                        old_fit,
                        lambda: fitting.fit_mesh(mesh, configuration),
                        lambda first, second: first == second,
                        args.repeats,
                    )
                    row.update(
                        configuration=asdict(configuration),
                        triangles=len(mesh.faces),
                        vertices=len(mesh.vertices),
                        mesh_sha256=hashlib.sha256(
                            np.asarray(mesh.vertices).tobytes() + np.asarray(mesh.faces).tobytes()
                        ).hexdigest(),
                    )
                    save(f"{name}/{method}/N{count}", row)
        print(args.output)
        return
    for count in args.points or (1000, 100000, 1000000):
        gaussian = generator.normal(size=(count, 3)) * [1, 0.4, 0.2]
        shell = generator.normal(size=(count, 3))
        shell /= np.linalg.norm(shell, axis=1)[:, None]
        shell *= generator.uniform(0.7, 1, size=(count, 1))
        for kind, cloud in (("gaussian", gaussian), ("shell", shell)):
            for order in ("C", "F"):
                points = np.array(cloud, order=order)
                center = points.mean(axis=0) + [2, 1, -1]
                label = f"{kind}/{order}/{count}"
                row = compare(
                    lambda: reference.tighten_ball(points, center),
                    lambda: optimized.tighten_ball(points, center),
                    lambda first, second: (
                        np.array_equal(first[0], second[0]) and first[1:] == second[1:]
                    ),
                    args.repeats,
                )
                save(label, row)
    print(args.output)


if __name__ == "__main__":
    main()
