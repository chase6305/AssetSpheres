"""Compare voxel preparation against a prior voxel_fit.py checkout.

Example:
    python benchmarks/benchmark_voxel_memory.py \
        --reference /tmp/baseline/src/assetspheres/voxel_fit.py --full-fit

The reference imports current shared helpers; only voxel_fit.py differs. Checks
require identical coordinates, candidates, sphere parameters and report values.
Allocation peaks use tracemalloc, not RSS, native runtimes or device memory.
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
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import scipy
import trimesh

import assetspheres.fit as fitting
import assetspheres.refine as refinement
import assetspheres.voxel_fit as optimized
from assetspheres import FitConfig


def load_reference(path, name="assetspheres._voxel_memory_reference"):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def same_candidates(first, second):
    fields = ("candidate_centers", "candidate_radii", "coverage_samples", "occupied_points")
    return (
        all(getattr(first, name).tobytes() == getattr(second, name).tobytes() for name in fields)
        and first.voxel_size_m == second.voxel_size_m
        and first.backend_details == second.backend_details
    )


def paired(reference, current, equal, repeats):
    before, after = reference(), current()
    assert equal(before, after), "Reference and current output differ"
    del before, after
    calls = {"reference": reference, "optimized": current}
    durations = {name: [] for name in calls}
    for repeat in range(repeats):
        for name in list(calls)[:: 1 if repeat % 2 == 0 else -1]:
            started = time.perf_counter()
            calls[name]()
            durations[name].append(time.perf_counter() - started)
    records = {}
    for name, call in calls.items():
        tracemalloc.start()
        try:
            call()
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        records[name] = {
            "times_s": durations[name],
            "median_s": statistics.median(durations[name]),
            "traced_allocation_peak_bytes": peak,
        }
    return {"exact_output_match": True, **records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--reference-refine", type=Path, help="Also compare the prior refinement sampler"
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark-voxel-memory.json"))
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--resolution", type=int, action="append")
    parser.add_argument("--backend", choices=("subdivide", "open3d", "warp"), action="append")
    parser.add_argument("--warp-cuda", action="store_true")
    parser.add_argument("--full-fit", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if not args.reference.is_file():
        parser.error("--reference must point to a previous voxel_fit.py")
    reference = load_reference(args.reference)
    reference_refine = (
        load_reference(args.reference_refine, "assetspheres._refine_memory_reference")
        if args.reference_refine
        else refinement
    )
    records = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "trimesh": trimesh.__version__,
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "optimized_sha256": hashlib.sha256(Path(optimized.__file__).read_bytes()).hexdigest(),
        "reference_refine_sha256": hashlib.sha256(
            Path(reference_refine.__file__).read_bytes()
        ).hexdigest(),
        "optimized_refine_sha256": hashlib.sha256(
            Path(refinement.__file__).read_bytes()
        ).hexdigest(),
        "threads": {
            name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
        },
        "memory_scope": "tracemalloc allocation peak; not RSS, native runtimes or device memory",
        "timing_scope": "warm paired calls, preloaded mesh, alternating call order; no export",
        "cases": {},
    }
    backends = [(name, "cpu") for name in dict.fromkeys(args.backend or ["subdivide"])]
    if args.warp_cuda:
        backends.append(("warp", "cuda"))
    mesh = trimesh.creation.box(extents=(0.3, 0.3, 0.3))

    def save(name, result):
        records["cases"][name] = result
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
        print(name, "exact match", flush=True)

    for backend, device in backends:
        for resolution in args.resolution or (24, 112):
            config = FitConfig(
                voxelizer=backend,
                device=device,
                resolution=resolution,
                count=32,
                samples=6000,
            )
            label = f"{backend}/{device}/cube{resolution}"
            save(
                label + "/preparation",
                paired(
                    lambda: reference._prepare_voxel_candidates(mesh, config),
                    lambda: optimized._prepare_voxel_candidates(mesh, config),
                    same_candidates,
                    args.repeats,
                ),
            )
            if not args.full_fit:
                continue
            for method in ("voxel", "refined", "refined-cover"):
                local = replace(config, method=method)

                def old_fit():
                    with (
                        patch.object(fitting, "fit_voxel_spheres", reference.fit_voxel_spheres),
                        patch.object(refinement, "refine_spheres", reference_refine.refine_spheres),
                    ):
                        return fitting.fit_mesh(mesh, local)

                save(
                    label + "/" + method,
                    paired(
                        old_fit,
                        lambda: fitting.fit_mesh(mesh, local),
                        lambda a, b: a[0] == b[0] and a[1] == b[1],
                        args.repeats,
                    ),
                )
    print(args.output)


if __name__ == "__main__":
    main()
