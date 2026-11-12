"""Compare ball-union distance workspaces against a previous geometry.py.

Paired calls alternate order after warm-up, and every output must match exactly.
Allocation measurements use tracemalloc separately from timing, not RSS or native
runtime memory. Optional complete fits isolate only the sphere_gaps change.
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

import assetspheres.fit as fitting
import assetspheres.geometry as geometry
import assetspheres.voxel_fit as voxel_fitting
from assetspheres import FitConfig, Sphere


def load_reference(path):
    spec = importlib.util.spec_from_file_location("assetspheres._gap_workspace_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fingerprint(result):
    if isinstance(result, np.ndarray):
        return hashlib.sha256(result.tobytes()).hexdigest()
    spheres, report = result
    payload = {"spheres": [sphere.to_dict() for sphere in spheres], "report": report}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def paired(calls, repeats):
    expected = fingerprint(calls["reference"]())
    assert fingerprint(calls["optimized"]()) == expected
    times = {name: [] for name in calls}
    for repeat in range(repeats):
        for name in list(calls)[:: 1 if repeat % 2 == 0 else -1]:
            started = time.perf_counter()
            result = calls[name]()
            times[name].append(time.perf_counter() - started)
            assert fingerprint(result) == expected, (name, repeat)
            del result
    records = {"exact_output_match": True, "output_sha256": expected}
    for name, call in calls.items():
        tracemalloc.start()
        try:
            result = call()
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert fingerprint(result) == expected, name
        records[name] = {
            "times_s": times[name],
            "median_s": statistics.median(times[name]),
            "traced_allocation_peak_bytes": peak,
        }
        del result
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path, help="Previous geometry.py")
    parser.add_argument("--repeats", default=11, type=int)
    parser.add_argument("--seed", default=20261112, type=int)
    parser.add_argument("--full-fit", action="store_true", help="Also compare complete mesh fits")
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark-gap-workspace.json"))
    args = parser.parse_args()
    if args.repeats < 1 or args.seed < 0:
        parser.error("repeats must be positive and seed must be nonnegative")
    if not args.reference.is_file():
        parser.error("--reference must point to the previous geometry.py")
    reference = load_reference(args.reference)
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
            name: os.environ.get(name)
            for name in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "optimized_sha256": hashlib.sha256(Path(geometry.__file__).read_bytes()).hexdigest(),
        "seed": args.seed,
        "repeats": args.repeats,
        "timing_scope": "warm paired complete calls, alternating order; imports/inputs excluded",
        "memory_scope": "tracemalloc allocation peaks, not RSS or native runtime memory",
        "cases": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save(label, row):
        records["cases"][label] = row
        args.output.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
        print(label, "exact match", flush=True)

    generator = np.random.default_rng(args.seed)
    for point_count, sphere_count in (
        (128, 16),
        (6000, 24),
        (100000, 24),
        (6000, 128),
        (100000, 128),
        (20000, 256),
        (100003, 257),
    ):
        points = generator.normal(size=(point_count * 2, 3))[::2]
        spheres = [
            Sphere(tuple(center), float(radius))
            for center, radius in zip(
                generator.normal(size=(sphere_count, 3)),
                generator.uniform(0.01, 1, sphere_count),
            )
        ]
        inputs_before = hashlib.sha256(points.tobytes()).hexdigest()
        row = paired(
            {
                "reference": lambda: reference.sphere_gaps(points, spheres),
                "optimized": lambda: geometry.sphere_gaps(points, spheres),
            },
            args.repeats,
        )
        assert hashlib.sha256(points.tobytes()).hexdigest() == inputs_before
        row.update(points=point_count, spheres=sphere_count, point_layout="strided")
        save(f"gaps/{point_count}/{sphere_count}", row)

    if args.full_fit:
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.1)
        for method in ("surface", "cover", "voxel"):
            for sample_count in (6000, 100000):
                configuration = FitConfig(
                    method=method, count=128, samples=sample_count, padding=0.003
                )

                def reference_fit():
                    with (
                        patch.object(fitting, "sphere_gaps", reference.sphere_gaps),
                        patch.object(voxel_fitting, "sphere_gaps", reference.sphere_gaps),
                    ):
                        return fitting.fit_mesh(mesh, configuration)

                row = paired(
                    {
                        "reference": reference_fit,
                        "optimized": lambda: fitting.fit_mesh(mesh, configuration),
                    },
                    args.repeats,
                )
                row.update(
                    configuration=asdict(configuration),
                    triangles=len(mesh.faces),
                    vertices=len(mesh.vertices),
                    mesh_sha256=hashlib.sha256(
                        np.asarray(mesh.vertices).tobytes() + np.asarray(mesh.faces).tobytes()
                    ).hexdigest(),
                    reference_scope="only sphere_gaps replaced; other helpers use current code",
                )
                save(f"fit/icosphere3/{method}/S{sample_count}", row)


if __name__ == "__main__":
    main()
