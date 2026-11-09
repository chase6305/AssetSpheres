"""Compare seeded surface sampling with an exported geometry.py baseline.

Export: git show db6efc1:src/assetspheres/geometry.py > /tmp/sampling-before.py
Run: PYTHONPATH=src python benchmarks/benchmark_surface_sampling.py \
    --baseline /tmp/sampling-before.py --output /tmp/sampling.json
Timing excludes imports, mesh creation and cache warm-up. Peaks use tracemalloc.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import statistics
import time
import tracemalloc
from pathlib import Path

import numpy as np
import trimesh

from assetspheres.geometry import sample_surface


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    spec = importlib.util.spec_from_file_location("assetspheres._sampling_before", args.baseline)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    functions = {"before": baseline.sample_surface, "after": sample_surface}
    cases = {}
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.1)
    for count in [10000, 100000, 1000000]:
        expected = baseline.sample_surface(mesh, count, 831)
        np.testing.assert_array_equal(sample_surface(mesh, count, 831), expected)
        del expected
        pairs = []
        for index in range(args.repeats):
            pair = {}
            for key in list(functions) if index % 2 == 0 else list(reversed(functions)):
                started = time.perf_counter()
                functions[key](mesh, count, 831)
                pair[key] = time.perf_counter() - started
            pairs.append(pair)
        peaks = {}
        for key, function in functions.items():
            tracemalloc.start()
            function(mesh, count, 831)
            peaks[key] = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
        cases[str(count)] = {
            "seconds_pairs": pairs,
            "seconds_median": {key: statistics.median(p[key] for p in pairs) for key in functions},
            "peak_allocated_bytes": peaks,
            "exact_samples": True,
        }
    record = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "trimesh": trimesh.__version__,
        "baseline": str(args.baseline),
        "cases": cases,
    }
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: {k: v for k, v in case.items() if k != "seconds_pairs"}
                for key, case in cases.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
