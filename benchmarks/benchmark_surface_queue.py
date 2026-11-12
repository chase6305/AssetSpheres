"""Compare surface rasterization with a standalone baseline surface_grid.py.

Pass the old module with --baseline and a JSON path with --output. Run serially
under the workspace's shared resource gate/lock, with numerical libraries limited
to one thread. Inputs and caches are prepared before timing. Grid filling and
hash checks run outside timing and allocation tracking. Peaks use tracemalloc,
not RSS or native runtime totals. Only surface rasterization is measured;
use benchmark_robot_workspaces.py for complete real-link fits.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import statistics
import tracemalloc
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
import trimesh

import assetspheres.surface_grid as current


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_hash(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def mesh_hash(mesh):
    return array_hash(mesh.vertices, mesh.faces, mesh.triangles)


def grid_hash(grid):
    return array_hash(grid.matrix, grid.transform, grid.copy().fill().matrix)


def measure(mesh, pitch, calls, repeats):
    original_mesh = mesh_hash(mesh)
    expected = None

    def check(result):
        nonlocal expected
        actual = grid_hash(result)
        if expected is None:
            expected = actual
        assert actual == expected, "Surface, transform or filled grid differs"
        assert mesh_hash(mesh) == original_mesh, "Input mesh changed"
        return actual

    warm_hashes, pairs, output_pairs = {}, [], []
    for name, call in calls.items():
        result = call(mesh, pitch, 2_000_000)
        warm_hashes[name] = check(result)
        del result
    for repeat in range(repeats):
        seconds, hashes = {}, {}
        for name in list(calls)[:: 1 if repeat % 2 == 0 else -1]:
            started = perf_counter()
            result = calls[name](mesh, pitch, 2_000_000)
            seconds[name] = perf_counter() - started
            hashes[name] = check(result)
            del result
        pairs.append(seconds)
        output_pairs.append(hashes)
    peaks, peak_hashes = {}, {}
    for name, call in calls.items():
        tracemalloc.start()
        try:
            result = call(mesh, pitch, 2_000_000)
            peaks[name] = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        peak_hashes[name] = check(result)
        del result
    return {
        "exact_surface_transform_fill": True,
        "input_unchanged": True,
        "input_sha256": original_mesh,
        "warm_output_sha256": warm_hashes,
        "seconds_pairs": pairs,
        "output_sha256_pairs": output_pairs,
        "seconds_median": {name: statistics.median(row[name] for row in pairs) for name in calls},
        "peak_allocated_bytes": peaks,
        "peak_output_sha256": peak_hashes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Previous surface_grid.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=11)
    args = parser.parse_args()
    baseline_path = args.baseline.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    paths = {
        "baseline": baseline_path,
        "current": Path(current.__file__).resolve(),
        "script": Path(__file__).resolve(),
    }
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if not baseline_path.is_file() or baseline_path.suffix != ".py":
        parser.error("--baseline must be an existing Python source file")
    if baseline_path.samefile(paths["current"]):
        parser.error("--baseline must differ from the current surface_grid.py")
    if output_path in paths.values():
        parser.error("--output must not overwrite the baseline, implementation or benchmark")
    source_before = {name: file_hash(path) for name, path in paths.items()}
    spec = importlib.util.spec_from_file_location("_surface_queue_baseline", baseline_path)
    if spec is None or spec.loader is None:
        parser.error("Cannot load --baseline as a Python module")
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    if not callable(getattr(baseline, "subdivide_surface_grid", None)):
        parser.error("--baseline must define subdivide_surface_grid")
    calls = {"baseline": baseline.subdivide_surface_grid, "current": current.subdivide_surface_grid}
    thread_variables = "OPENBLAS_NUM_THREADS OMP_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS"
    record = {
        "scope": "isolated surface rasterization; no complete-fit measurement",
        "timing_scope": "warm alternating calls; input/cache preparation and verification excluded",
        "memory_scope": "tracemalloc call peak; excludes verification; not RSS/native totals",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "trimesh": trimesh.__version__,
        "platform": platform.platform(),
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "threads": {name: os.environ.get(name) for name in thread_variables.split()},
        "source_paths": {name: str(path) for name, path in paths.items()},
        "source_sha256_before": source_before,
        "repeats": args.repeats,
        "max_grid_cells": 2_000_000,
        "triangle_batch_sizes": {
            "baseline": baseline._TRIANGLE_BATCH_SIZE,
            "current": current._TRIANGLE_BATCH_SIZE,
        },
        "cases": {},
        "completed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    cases = {"box": (48, 112), "cylinder": (48, 96), "icosphere3": (48,)}
    for name, resolutions in cases.items():
        if name == "box":
            mesh = trimesh.creation.box(extents=(0.3, 0.3, 0.3))
        elif name == "cylinder":
            mesh = trimesh.creation.cylinder(radius=0.04, height=0.3, sections=48)
        else:
            mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.1)
        for attribute in ("bounds", "extents", "triangles"):
            getattr(mesh, attribute)
        for resolution in resolutions:
            pitch = float(mesh.extents.max()) / resolution
            row = measure(mesh, pitch, calls, args.repeats)
            row["resolution"], row["pitch_m"] = resolution, pitch
            row["vertices"], row["triangles"] = len(mesh.vertices), len(mesh.faces)
            record["cases"][f"{name}/res{resolution}"] = row
            output_path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
            print(name, resolution, "exact surface/transform/fill; input unchanged", flush=True)
    record["source_sha256_after"] = {name: file_hash(path) for name, path in paths.items()}
    assert record["source_sha256_after"] == source_before, "Audited source changed"
    record["completed"] = True
    output_path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
