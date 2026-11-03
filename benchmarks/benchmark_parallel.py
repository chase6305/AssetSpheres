"""Reproduce CPU link-parallelism measurements, including pool lifecycle cost.

Defaults generate four tiny surface jobs and sixteen voxel jobs locally. Add
``--urdf`` to include a caller-supplied robot. No external assets are required.

Every measured call includes fit_links, pool startup, spawned-worker imports,
process task serialization, fitting, report generation, and pool shutdown. Mesh
creation/loading, restoring identical inputs for each repeat, parent imports,
output equality checks, and JSON export are outside the timer. Pools are never
reused between calls. Thread environment variables are recorded, not changed.

Run from the repository root after installing the project:
    .venv/bin/python benchmarks/benchmark_parallel.py --repeats 3 \
        --output outputs/benchmark-parallel.json

Quick smoke check:
    .venv/bin/python benchmarks/benchmark_parallel.py --case small --repeats 1
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

import trimesh

from assetspheres import FitConfig, load_asset
from assetspheres.parallel import fit_links

MODES = [("serial", 1), ("thread", 2), ("thread", 4), ("process", 2), ("process", 4)]
THREAD_VARIABLES = [
    "OMP_NUM_THREADS",
    "OMP_DYNAMIC",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
]


def positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def cpu_information() -> dict:
    model = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    physical_cores = set()
    if cpuinfo.is_file():
        for block in cpuinfo.read_text().split("\n\n"):
            values = dict(line.split(":", 1) for line in block.splitlines() if ":" in line)
            values = {key.strip(): value.strip() for key, value in values.items()}
            model = values.get("model name", values.get("Hardware", model))
            if "physical id" in values and "core id" in values:
                physical_cores.add((values["physical id"], values["core id"]))
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    try:
        from threadpoolctl import threadpool_info

        native_pools = threadpool_info()
    except ImportError:
        native_pools = None
    packages = {}
    for package in ("numpy", "scipy", "trimesh", "assetspheres"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "cpu_model": model,
        "logical_cpu_count": os.cpu_count(),
        "physical_core_count_from_proc": len(physical_cores) or None,
        "cpu_affinity": affinity,
        "available_logical_cpu_count": len(affinity) if affinity is not None else os.cpu_count(),
        "architecture": platform.machine(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "packages": packages,
        "thread_environment": {name: os.environ.get(name) for name in THREAD_VARIABLES},
        "native_threadpools": native_pools,
    }


def output_fingerprint(results) -> str:
    # Verify exact link order and every sphere parameter. Runtime measurements in
    # future quality reports must not make identical sphere outputs compare unequal.
    payload = [(name, [sphere.to_dict() for sphere in spheres]) for name, spheres, _ in results]
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def benchmark_case(name: str, tasks: list, repeats: int, source: str | None = None) -> dict:
    # Restore this same snapshot before each run, outside the timed region, so
    # trimesh cache warming in one mode cannot change another mode's inputs.
    payload = pickle.dumps(tasks, protocol=pickle.DEFAULT_PROTOCOL)
    mesh_bytes = sum(
        len(pickle.dumps(mesh, protocol=pickle.DEFAULT_PROTOCOL)) for _, mesh, _ in tasks
    )
    task_bytes = sum(len(pickle.dumps(task, protocol=pickle.DEFAULT_PROTOCOL)) for task in tasks)
    reference = None
    measurements = []
    for executor, workers in MODES:
        durations = []
        for repeat in range(repeats):
            fresh_tasks = pickle.loads(payload)
            started = time.perf_counter()
            fitted = fit_links(
                fresh_tasks,
                workers=workers,
                executor="thread" if executor == "serial" else executor,
            )
            elapsed = time.perf_counter() - started
            fingerprint = output_fingerprint(fitted)
            if reference is None:
                reference = fingerprint
            if fingerprint != reference:
                raise RuntimeError(
                    f"{name}: {executor}/{workers}, repeat {repeat + 1} changed sphere output"
                )
            durations.append(elapsed)
        row = {
            "executor": executor,
            "workers_requested": workers,
            "workers_effective": min(workers, len(tasks)),
            "run_times_s": durations,
            "median_s": statistics.median(durations),
            "min_s": min(durations),
            "max_s": max(durations),
            "sphere_output_matches_serial": True,
        }
        measurements.append(row)
        print(json.dumps({"case": name, **row}), flush=True)
    baseline = measurements[0]["median_s"]
    for row in measurements:
        row["speedup_over_serial_median"] = baseline / row["median_s"]
    return {
        "name": name,
        "source": source or "generated",
        "link_count": len(tasks),
        "configs": {link_name: asdict(config) for link_name, _, config in tasks},
        "triangles_per_link": {link_name: len(mesh.faces) for link_name, mesh, _ in tasks},
        "pickle_protocol": pickle.DEFAULT_PROTOCOL,
        "mesh_pickle_bytes_sum": mesh_bytes,
        "task_pickle_bytes_sum": task_bytes,
        "task_list_pickle_bytes": len(payload),
        "serialization_size_note": "pickle estimates, not measured process transport bytes",
        "sphere_output_sha256": reference,
        "measurements": measurements,
    }


def package_roots(values: list[str]) -> dict[str, str]:
    mapping = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise ValueError(f"Expected NAME=PATH for --package-root, got {value!r}")
        if name in mapping:
            raise ValueError(f"Duplicate --package-root for {name!r}")
        mapping[name] = path
    return mapping


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark-parallel.json"))
    parser.add_argument("--repeats", type=positive_integer, default=3)
    parser.add_argument("--urdf", type=Path, help="Optional local URDF to benchmark using cover32")
    parser.add_argument("--package-root", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument(
        "--case",
        action="append",
        choices=["small", "voxel", "urdf"],
        help="Select cases; defaults to small, voxel, and urdf if supplied",
    )
    args = parser.parse_args(argv)
    requested = args.case or ["small", "voxel", *(["urdf"] if args.urdf else [])]
    if "urdf" in requested and args.urdf is None:
        parser.error("--case urdf requires --urdf")
    cases = []
    if "small" in requested:
        config = FitConfig(method="surface", count=8, samples=128, device="cpu")
        tasks = [(f"link_{i}", trimesh.creation.box(), config) for i in range(4)]
        cases.append(("tiny_surface_4", tasks, None))
    if "voxel" in requested:
        config = FitConfig(method="voxel", count=32, samples=6000, resolution=80, device="cpu")
        tasks = [
            (f"link_{i}", trimesh.creation.box(extents=(0.2 + i * 0.002, 0.15, 0.3)), config)
            for i in range(16)
        ]
        cases.append(("voxel_res80_16", tasks, None))
    if "urdf" in requested:
        asset = load_asset(args.urdf, package_roots=package_roots(args.package_root))
        config = FitConfig(method="cover", count=32, samples=2000, device="cpu")
        tasks = [(name, link.mesh, config) for name, link in asset.links.items()]
        cases.append((f"urdf_cover_{len(tasks)}", tasks, str(asset.path)))
    report = {
        "schema_version": 1,
        "benchmark": "cpu_link_parallelism",
        "environment": cpu_information(),
        "repeats": args.repeats,
        "timer": "time.perf_counter",
        "timing_scope": {
            "includes": [
                "fit_links",
                "pool startup",
                "spawned-worker imports",
                "process task serialization",
                "fitting",
                "quality reports",
                "pool shutdown",
            ],
            "excludes": [
                "parent imports",
                "mesh creation/loading",
                "repeat input restoration",
                "equality checks",
                "JSON export",
            ],
            "pool_reused_between_calls": False,
        },
        "output_equality": "exact link order, sphere centers, and radii; excludes quality reports",
        "cases": [
            benchmark_case(name, tasks, args.repeats, source) for name, tasks, source in cases
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Wrote {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
