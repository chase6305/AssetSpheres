"""Compare surface rasterization against an exported baseline module.

Export: git show 9dab365:src/assetspheres/surface_grid.py > /tmp/surface-before.py
Run: PYTHONPATH=src python benchmarks/benchmark_surface_workspace.py \
    --baseline /tmp/surface-before.py --output /tmp/surface-workspace.json
Inputs and imports are outside timing. Peaks are tracemalloc allocations, not RSS.
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

from assetspheres.surface_grid import subdivide_surface_grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    spec = importlib.util.spec_from_file_location("surface_before", args.baseline)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    functions = {"before": baseline.subdivide_surface_grid, "after": subdivide_surface_grid}
    cases = {}
    for name, mesh, resolution in [
        ("box48", trimesh.creation.box(extents=(0.3, 0.1, 0.02)), 48),
        ("box96", trimesh.creation.box(extents=(0.3, 0.1, 0.02)), 96),
        ("cylinder48", trimesh.creation.cylinder(radius=0.04, height=0.3), 48),
        ("sphere48", trimesh.creation.icosphere(subdivisions=3, radius=0.1), 48),
    ]:
        pitch = float(mesh.extents.max()) / resolution
        results = {key: function(mesh, pitch, 2_000_000) for key, function in functions.items()}
        np.testing.assert_array_equal(results["before"].matrix, results["after"].matrix)
        np.testing.assert_array_equal(results["before"].transform, results["after"].transform)
        pairs = []
        for index in range(args.repeats):
            pair = {}
            for key in list(functions) if index % 2 == 0 else list(reversed(functions)):
                started = time.perf_counter()
                functions[key](mesh, pitch, 2_000_000)
                pair[key] = time.perf_counter() - started
            pairs.append(pair)
        peaks = {}
        for key, function in functions.items():
            tracemalloc.start()
            function(mesh, pitch, 2_000_000)
            peaks[key] = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
        cases[name] = {
            "seconds_pairs": pairs,
            "seconds_median": {key: statistics.median(p[key] for p in pairs) for key in functions},
            "peak_allocated_bytes": peaks,
            "exact_surface_and_transform": True,
        }
    record = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "trimesh": trimesh.__version__,
        "baseline": str(args.baseline),
        "cases": cases,
    }
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
