"""Pair complete surface repairs against an exported surface_cover.py baseline.

Example:
    git show 15ecb68:src/assetspheres/surface_cover.py > /tmp/repair-before.py
    python benchmarks/benchmark_repair_workspace.py --baseline /tmp/repair-before.py \
        --output outputs/repair-workspace.json

Inputs are preloaded; timing includes normalization, triangle repair and reports.
Allocation peaks use tracemalloc, not process RSS or third-party native memory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import statistics
import time
import tracemalloc
from pathlib import Path

import numpy as np
import scipy
import trimesh

import assetspheres.surface_cover as current
from assetspheres import Sphere


def fingerprint(result):
    spheres, report = result
    payload = {
        "spheres": [[list(sphere.center), sphere.radius] for sphere in spheres],
        "report": report,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--subdivisions", type=int, action="append")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    spec = importlib.util.spec_from_file_location("assetspheres._repair_before", args.baseline)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    records = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "trimesh": trimesh.__version__,
        "platform": platform.platform(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "threads": {
            key: os.environ.get(key)
            for key in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "current_sha256": hashlib.sha256(Path(current.__file__).read_bytes()).hexdigest(),
        "timing_scope": "complete warm repair calls, preloaded meshes, alternating call order",
        "memory_scope": "tracemalloc allocation peak; not RSS or native allocations",
        "cases": {},
    }
    for subdivisions in args.subdivisions or [2, 5, 6, 7]:
        mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=0.1)
        for name, radius in [("enclosed", 0.2), ("growth", 0.03)]:
            spheres = [Sphere((0, 0, 0), radius)]
            calls = {
                "before": lambda: baseline.repair_surface_cover(mesh, spheres),
                "after": lambda: current.repair_surface_cover(mesh, spheres),
            }
            expected = fingerprint(calls["before"]())
            assert fingerprint(calls["after"]()) == expected
            times = {key: [] for key in calls}
            for repeat in range(args.repeats):
                for key in list(calls)[:: 1 if repeat % 2 == 0 else -1]:
                    started = time.perf_counter()
                    result = calls[key]()
                    times[key].append(time.perf_counter() - started)
                    assert fingerprint(result) == expected
            peaks = {}
            for key, call in calls.items():
                tracemalloc.start()
                try:
                    call()
                    peaks[key] = tracemalloc.get_traced_memory()[1]
                finally:
                    tracemalloc.stop()
            case = {
                "vertex_count": len(mesh.vertices),
                "triangle_count": len(mesh.faces),
                "sphere_count": len(spheres),
                "sphere_radius_m": radius,
                "output_sha256": expected,
                "exact_output": True,
                "times_s": times,
                "median_s": {key: statistics.median(values) for key, values in times.items()},
                "traced_allocation_peak_bytes": peaks,
            }
            label = f"icosphere{subdivisions}/{name}"
            records["cases"][label] = case
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
            print(label, case["median_s"], peaks, flush=True)


if __name__ == "__main__":
    main()
