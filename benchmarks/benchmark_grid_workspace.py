"""Compare complete EDT and voxel fitting with a previous grid_distance.py.

Run with single-threaded numerical libraries and a preloaded mesh. Timings
alternate implementations. Peaks track Python/NumPy allocations, not RSS.
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
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import scipy
import trimesh

from assetspheres import FitConfig, fit_mesh, grid_distance, refine, voxel_fit


def fingerprint(result):
    if isinstance(result, np.ndarray):
        return hashlib.sha256(result.tobytes()).hexdigest()
    spheres, report = result
    payload = {"spheres": [sphere.to_dict() for sphere in spheres], "report": report}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def paired(calls, repeats):
    expected = fingerprint(calls["before"]())
    assert fingerprint(calls["after"]()) == expected
    pairs = []
    for index in range(repeats):
        pair = {}
        for key in list(calls)[:: 1 if index % 2 == 0 else -1]:
            started = time.perf_counter()
            result = calls[key]()
            pair[key] = time.perf_counter() - started
            assert fingerprint(result) == expected, (key, index)
            del result
        pairs.append(pair)
    peaks = {}
    for key, call in calls.items():
        tracemalloc.start()
        try:
            result = call()
            peaks[key] = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert fingerprint(result) == expected, key
        del result
    return {
        "exact_output_match": True,
        "output_sha256": expected,
        "seconds_pairs": pairs,
        "seconds_median": {key: statistics.median(p[key] for p in pairs) for key in calls},
        "peak_allocated_bytes": peaks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--full-fit", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    spec = importlib.util.spec_from_file_location("grid_before", args.baseline)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    cases = {}
    for shape in [(24, 24, 24), (48, 48, 48), (114, 114, 114), (16, 256, 256), (1, 256, 256)]:
        grid = np.random.default_rng(91).random(shape) > 0.3
        cases[f"grid/{shape}"] = paired(
            {
                "before": lambda: baseline.unit_grid_distances(grid),
                "after": lambda: grid_distance.unit_grid_distances(grid),
            },
            args.repeats,
        )
    if args.full_fit:
        mesh = trimesh.creation.box(extents=(0.3, 0.3, 0.3))
        for resolution in (24, 112):
            for method in ("voxel", "refined", "refined-cover"):
                config = FitConfig(method=method, resolution=resolution, count=24)

                def old_fit():
                    with (
                        patch.object(
                            voxel_fit, "unit_grid_distances", baseline.unit_grid_distances
                        ),
                        patch.object(refine, "unit_grid_distances", baseline.unit_grid_distances),
                    ):
                        return fit_mesh(mesh, config)

                cases[f"cube/{resolution}/{method}"] = paired(
                    {"before": old_fit, "after": lambda: fit_mesh(mesh, config)}, args.repeats
                )
                cases[f"cube/{resolution}/{method}"]["configuration"] = asdict(config)
    record = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "trimesh": trimesh.__version__,
        "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(Path(grid_distance.__file__).read_bytes()).hexdigest(),
        "threads": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "memory_scope": "tracemalloc allocation peak, not RSS, native runtime or device memory",
        "timing_scope": "alternating warm calls, mesh creation and imports excluded",
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
