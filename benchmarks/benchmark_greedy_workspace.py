"""Compare isolated CPU greedy cover calls against a previous voxel_fit.py.

Run serially under the workspace resource gate/lock with numerical threads set
to one. Inputs and warm-up are outside timing. Each measured call includes its
KD tree, neighborhoods and selection. Verification is outside timing and memory
tracking; peaks use tracemalloc, not RSS or native runtime totals. This does not
measure complete fits. The old self-contained _greedy_cover_cpu is called alone;
its module's relative imports resolve to current helpers that are not invoked.
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
import tracemalloc
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
import trimesh

import assetspheres.voxel_fit as current


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(arrays, calls, repeats):
    original = tuple(array.tobytes() for array in arrays)
    digest = hashlib.sha256()
    for array, data in zip(arrays, original):
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(data)
    expected = None

    def check(result):
        nonlocal expected
        assert type(result) is list, "Greedy cover must return a selected-index list"
        if expected is None:
            expected = result.copy()
        assert result == expected, "Selected indices or their order differ"
        assert tuple(array.tobytes() for array in arrays) == original, "Input arrays changed"
        return hashlib.sha256(json.dumps(result, separators=(",", ":")).encode()).hexdigest()

    warm_hashes, pairs, output_pairs = {}, [], []
    for name, call in calls.items():
        result = call(*arrays, 24)
        warm_hashes[name] = check(result)
        del result
    for repeat in range(repeats):
        seconds, hashes = {}, {}
        for name in list(calls)[:: 1 if repeat % 2 == 0 else -1]:
            started = perf_counter()
            result = calls[name](*arrays, 24)
            seconds[name] = perf_counter() - started
            hashes[name] = check(result)
            del result
        pairs.append(seconds)
        output_pairs.append(hashes)
    peaks, peak_hashes = {}, {}
    for name, call in calls.items():
        tracemalloc.start()
        try:
            result = call(*arrays, 24)
            peaks[name] = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        peak_hashes[name] = check(result)
        del result
    return {
        "exact_selected_indices": True,
        "selected_indices": expected,
        "input_unchanged": True,
        "input_sha256": digest.hexdigest(),
        "warm_output_sha256": warm_hashes,
        "seconds_pairs": pairs,
        "output_sha256_pairs": output_pairs,
        "seconds_median": {name: statistics.median(row[name] for row in pairs) for name in calls},
        "peak_allocated_bytes": peaks,
        "peak_output_sha256": peak_hashes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Previous voxel_fit.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=11)
    args = parser.parse_args()
    paths = {
        "baseline": args.baseline.expanduser().resolve(),
        "current": Path(current.__file__).resolve(),
        "script": Path(__file__).resolve(),
    }
    output = args.output.expanduser().resolve()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if not paths["baseline"].is_file() or paths["baseline"].suffix != ".py":
        parser.error("--baseline must be an existing Python source file")
    if paths["baseline"].samefile(paths["current"]):
        parser.error("--baseline must differ from the current voxel_fit.py")
    if output in paths.values() or (
        output.exists() and any(output.samefile(path) for path in paths.values())
    ):
        parser.error("--output must not overwrite the baseline, implementation or benchmark")
    source_before = {name: file_hash(path) for name, path in paths.items()}
    spec = importlib.util.spec_from_file_location(
        "assetspheres._greedy_workspace_before", paths["baseline"]
    )
    if spec is None or spec.loader is None:
        parser.error("Cannot load --baseline as a Python module")
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    if not callable(getattr(baseline, "_greedy_cover_cpu", None)):
        parser.error("--baseline must define _greedy_cover_cpu")
    calls = {"baseline": baseline._greedy_cover_cpu, "current": current._greedy_cover_cpu}
    thread_names = "OPENBLAS_NUM_THREADS OMP_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS"
    record = {
        "scope": "isolated CPU greedy cover, including KD tree/neighborhoods; no complete fits",
        "baseline_scope": "old self-contained function; imported current relative helpers unused",
        "timing_scope": "warm alternating calls; input creation and verification excluded",
        "memory_scope": "tracemalloc call peak; verification excluded; not RSS/native totals",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "trimesh": trimesh.__version__,
        "platform": platform.platform(),
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "threads": {name: os.environ.get(name) for name in thread_names.split()},
        "source_paths": {name: str(path) for name, path in paths.items()},
        "source_sha256_before": source_before,
        "seed": 20261113,
        "repeats": args.repeats,
        "sphere_count": 24,
        "cases": {},
        "completed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")

    save()
    generator = np.random.default_rng(record["seed"])
    for name, count in (
        ("default_dense", 6000),
        ("large_dense", 100000),
        ("large_sparse_one", 100000),
    ):
        if name == "large_sparse_one":
            samples = np.zeros((count, 3))
            samples[:, 0] = np.arange(count)
            centers = samples[np.linspace(0, count - 1, 768, dtype=int)].copy()
            radii = np.zeros(768)
            distribution = "unique integer x coordinates; each radius-zero ball hits one sample"
        else:
            samples, centers = generator.random((count, 3)), generator.random((768, 3))
            radii = generator.uniform(0.1, 0.6, size=768)
            distribution = "samples/centers uniform [0,1)^3; radii uniform [0.1,0.6)"
        row = measure((samples, centers, radii), calls, args.repeats)
        row.update(sample_count=count, candidate_count=768, distribution=distribution)
        record["cases"][name] = row
        save()
        print(name, "exact selected indices; input unchanged", flush=True)
    record["source_sha256_after"] = {name: file_hash(path) for name, path in paths.items()}
    assert record["source_sha256_after"] == source_before, "Audited source changed"
    record["completed"] = True
    save()


if __name__ == "__main__":
    main()
