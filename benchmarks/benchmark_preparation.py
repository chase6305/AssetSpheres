"""Compare mesh preparation and optional complete fits with a previous fit.py.

Each call receives a restored input outside the timer. Cold cases start without
area caches; warm cases preserve a precomputed area cache. Reference and current
calls alternate, and their complete outputs must match. Allocation peaks are
measured separately with tracemalloc, not RSS or native runtime memory.
The reference package is loaded under a separate name, including its methods.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.util
import json
import os
import pickle
import platform
import statistics
import sys
import time
import tracemalloc
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy
import trimesh

import assetspheres.fit as fitting
from assetspheres import FitConfig


def source_cache_bytes(mesh):
    """Diagnostic sum of cached array sizes, not a process memory measurement."""
    return sum(
        value.nbytes for value in mesh._cache.cache.values() if isinstance(value, np.ndarray)
    )


def fingerprint(result, complete_fit):
    if complete_fit:
        spheres, report = result
        payload = {"spheres": [sphere.to_dict() for sphere in spheres], "report": report}
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
    digest = hashlib.sha256()
    for array in (result.vertices, result.faces):
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    digest.update(json.dumps(result.metadata, sort_keys=True, allow_nan=False).encode())
    return digest.hexdigest()


def paired(calls, snapshot, repeats, complete_fit):
    expected = None
    source_fingerprint = fingerprint(pickle.loads(snapshot), False)
    times = {name: [] for name in calls}
    for repeat in range(repeats):
        for name in list(calls)[:: 1 if repeat % 2 == 0 else -1]:
            source = pickle.loads(snapshot)
            gc.collect()
            started = time.perf_counter()
            result = calls[name](source)
            times[name].append(time.perf_counter() - started)
            actual = fingerprint(result, complete_fit)
            if expected is None:
                expected = actual
            assert actual == expected, (name, repeat)
            assert fingerprint(source, False) == source_fingerprint, name
            del result, source
    records = {"exact_output_match": True, "output_sha256": expected}
    for name, call in calls.items():
        source = pickle.loads(snapshot)
        gc.collect()
        before_cache = source_cache_bytes(source)
        tracemalloc.start()
        try:
            result = call(source)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert fingerprint(result, complete_fit) == expected, name
        assert fingerprint(source, False) == source_fingerprint, name
        records[name] = {
            "times_s": times[name],
            "median_s": statistics.median(times[name]),
            "traced_allocation_peak_bytes": peak,
            "source_cache_array_bytes_before": before_cache,
            "source_cache_array_bytes_after": source_cache_bytes(source),
        }
        del result, source
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="previous fit.py")
    parser.add_argument(
        "--subdivisions", type=int, action="append", help="icosphere subdivision level"
    )
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--full-fit", action="store_true", help="also compare complete fits")
    parser.add_argument("--method", choices=("cover", "surface", "slab"), default="cover")
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark-preparation.json"))
    args = parser.parse_args()
    if args.repeats < 1 or any(level < 0 or level > 7 for level in args.subdivisions or []):
        parser.error("repeats must be positive; subdivisions must be between 0 and 7")
    if not args.reference.is_file() or not (args.reference.parent / "__init__.py").is_file():
        parser.error("--reference must point to fit.py inside a previous assetspheres package")
    configuration = FitConfig(
        method=args.method, count=args.count, samples=args.samples, padding=0.003
    )
    spec = importlib.util.spec_from_file_location(
        "_assetspheres_preparation_reference",
        args.reference.parent / "__init__.py",
        submodule_search_locations=[str(args.reference.parent)],
    )
    reference_package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference_package
    spec.loader.exec_module(reference_package)
    reference = importlib.import_module(f"{spec.name}.fit")
    records = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "trimesh": trimesh.__version__,
        },
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "threads": {
            name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
        },
        "source_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in (("reference", args.reference), ("optimized", Path(fitting.__file__)))
        },
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_package_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(args.reference.parent.glob("*.py"))
        },
        "shared_sources_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(fitting.__file__).parent.glob("*.py"))
            if path.name != "fit.py"
        },
        "config": asdict(configuration),
        "repeats": args.repeats,
        "timing_scope": "paired calls; input creation/restoration, GC and output checks excluded",
        "memory_scope": "tracemalloc allocation peak; not RSS, native runtime or device memory",
        "cases": {},
    }
    for level in args.subdivisions or (2, 4, 6, 7):
        for warm in (False, True):
            mesh = trimesh.creation.icosphere(subdivisions=level, radius=0.1)
            if warm:
                _ = mesh.area
            snapshot = pickle.dumps(mesh, protocol=pickle.HIGHEST_PROTOCOL)
            for complete_fit in (False, True) if args.full_fit else (False,):
                calls = {
                    name: (lambda source, fit=module.fit_mesh: fit(source, configuration))
                    if complete_fit
                    else module._prepare_mesh
                    for name, module in (("reference", reference), ("optimized", fitting))
                }
                cache_state = "warm" if warm else "cold"
                phase = "fit" if complete_fit else "prepare"
                label = f"ico{level}/{cache_state}/{phase}"
                records["cases"][label] = {
                    "vertices": len(mesh.vertices),
                    "triangles": len(mesh.faces),
                    "area_cache_precomputed": warm,
                    **paired(calls, snapshot, args.repeats, complete_fit),
                }
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
                print(label, "exact match", flush=True)
    print(args.output)


if __name__ == "__main__":
    main()
