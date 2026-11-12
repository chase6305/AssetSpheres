"""Compare complete fits on selected robot links against an independent package.

Load URDF geometry before timing; retain the same immutable input for both
packages. Every warm-up, timed and allocation-measured output must have identical
sphere parameters and reports. Allocation peaks use tracemalloc, not process RSS
or native runtime memory. Each link/resolution also receives one untimed current
grid preparation for actual lattice dimensions and occupied-cell counts.

Use --repair-only to freeze one current refined initialization per link and
resolution, then compare complete old/new repair calls on those same spheres.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
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

import numpy as np
import scipy
import trimesh

import assetspheres
from assetspheres import FitConfig, fit_mesh, load_asset


def load_reference(directory):
    name = "_assetspheres_robot_workspace_reference"
    spec = importlib.util.spec_from_file_location(
        name,
        directory / "__init__.py",
        submodule_search_locations=[str(directory)],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    return package


def source_hashes(directory):
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*.py"))
    }


def mesh_fingerprint(mesh):
    digest = hashlib.sha256()
    for array in (np.asarray(mesh.vertices), np.asarray(mesh.faces)):
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def result_payload(result):
    spheres, report = result
    return {"spheres": [sphere.to_dict() for sphere in spheres], "report": report}


def fingerprint(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def paired(calls, repeats, mesh):
    mesh_before = mesh_fingerprint(mesh)
    expected = None
    expected_payload = None

    def check(result):
        nonlocal expected, expected_payload
        payload = result_payload(result)
        actual = fingerprint(payload)
        if expected is None:
            expected, expected_payload = actual, payload
        assert actual == expected, "Complete sphere parameters or report differ"
        assert mesh_fingerprint(mesh) == mesh_before, "Input mesh geometry changed"

    for call in calls.values():
        result = call()
        check(result)
        del result
    pairs = []
    for repeat in range(repeats):
        pair = {}
        for name in list(calls)[:: 1 if repeat % 2 == 0 else -1]:
            started = time.perf_counter()
            result = calls[name]()
            pair[name] = time.perf_counter() - started
            check(result)
            del result
        pairs.append(pair)
    peaks = {}
    for name, call in calls.items():
        tracemalloc.start()
        try:
            result = call()
            peaks[name] = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        check(result)
        del result
    return {
        "exact_output_match": True,
        "output_sha256": expected,
        "result": expected_payload,
        "input_geometry_unchanged": True,
        "seconds_pairs": pairs,
        "seconds_median": {name: statistics.median(pair[name] for pair in pairs) for name in calls},
        "paired_change_median": statistics.median(
            pair["after"] / pair["before"] - 1 for pair in pairs
        ),
        "peak_allocated_bytes": peaks,
    }


def grid_statistics(mesh, configuration):
    fitting = importlib.import_module("assetspheres.fit")
    voxel_fitting = importlib.import_module("assetspheres.voxel_fit")
    prepared_mesh = fitting._prepare_mesh(mesh)
    prepared_grid = voxel_fitting._prepare_voxel_grid(prepared_mesh, configuration)
    matrix = prepared_grid.voxel_grid.matrix
    return {
        "scope": "one extra untimed current grid preparation on independently cleaned geometry",
        "shape": list(matrix.shape),
        "cell_count": int(matrix.size),
        "occupied_count": int(np.count_nonzero(matrix)),
        "voxel_size_m": prepared_grid.voxel_size_m,
        "prepared_vertex_count": len(prepared_mesh.vertices),
        "prepared_triangle_count": len(prepared_mesh.faces),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path, help="Previous package directory")
    parser.add_argument("--input", required=True, type=Path, help="URDF with its mesh references")
    parser.add_argument("--geometry", choices=("collision", "visual"), default="collision")
    parser.add_argument("--link", required=True, action="append", help="Selected link; repeatable")
    parser.add_argument("--resolution", type=int, action="append", help="Default: 48; repeatable")
    parser.add_argument("--method", choices=("voxel", "refined-cover"), action="append")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--repair-only", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or any(value < 4 for value in args.resolution or []):
        parser.error("repeats must be positive; resolutions must be at least four")
    if args.repair_only and args.method:
        parser.error("--method selects complete fits; omit it with --repair-only")
    reference_root = args.reference.expanduser().resolve()
    current_root = Path(assetspheres.__file__).resolve().parent
    if not (reference_root / "__init__.py").is_file():
        parser.error("--reference must contain the previous package's __init__.py")
    if reference_root == current_root:
        parser.error("--reference must be an independent previous package directory")
    reference = load_reference(reference_root)
    source_before = source_hashes(reference_root)
    source_after = source_hashes(current_root)
    input_path = args.input.expanduser().resolve()
    input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    asset = load_asset(input_path, geometry=args.geometry, link_names=args.link)
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
        "input": {"path": str(input_path), "sha256": input_hash, "geometry": args.geometry},
        "source_before_sha256": source_before,
        "source_after_sha256": source_after,
        "reference_scope": "independent previous package, including every fitting helper",
        "timing_scope": (
            "complete warm repair calls after one current refined initialization"
            if args.repair_only
            else "complete warm fit_mesh calls on preloaded link geometry"
        ),
        "excluded_scope": (
            "imports, asset loading, input preparation, grid diagnostics and output fingerprinting"
        ),
        "memory_scope": "tracemalloc allocation peaks; not RSS or third-party native memory",
        "repeats": args.repeats,
        "completed": False,
        "cases": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")

    for link_name, link in asset.links.items():
        mesh = link.mesh
        input_mesh_hash = mesh_fingerprint(mesh)
        mesh_record = {
            "link": link_name,
            "geometry_source": link.geometry_source,
            "geometry_types": list(link.geometry_types),
            "vertex_count": len(mesh.vertices),
            "triangle_count": len(mesh.faces),
            "bounds_m": np.asarray(mesh.bounds).tolist(),
            "sha256": input_mesh_hash,
        }
        for resolution in dict.fromkeys(args.resolution or [48]):
            configuration = FitConfig(count=24, resolution=resolution, padding=0.003)
            grid = grid_statistics(mesh, configuration)
            if args.repair_only:
                initialization_config = FitConfig(
                    method="refined", count=24, resolution=resolution, padding=0.003
                )
                started = time.perf_counter()
                initial_spheres, initial_report = fit_mesh(mesh, initialization_config)
                initialization_seconds = time.perf_counter() - started
                input_spheres = [sphere.to_dict() for sphere in initial_spheres]
                previous_spheres = [
                    reference.Sphere(sphere.center, sphere.radius) for sphere in initial_spheres
                ]
                current_repair = importlib.import_module("assetspheres.surface_cover")
                previous_repair = importlib.import_module(f"{reference.__name__}.surface_cover")
                row = paired(
                    {
                        "before": lambda: previous_repair.repair_surface_cover(
                            mesh, previous_spheres
                        ),
                        "after": lambda: current_repair.repair_surface_cover(mesh, initial_spheres),
                    },
                    args.repeats,
                    mesh,
                )
                assert [sphere.to_dict() for sphere in initial_spheres] == input_spheres
                assert [sphere.to_dict() for sphere in previous_spheres] == input_spheres
                assert initial_report["voxel_count"] == grid["occupied_count"]
                row.update(
                    mesh=mesh_record,
                    grid=grid,
                    initialization={
                        "configuration": asdict(initialization_config),
                        "seconds_excluded_from_measurements": initialization_seconds,
                        "spheres": input_spheres,
                        "report": initial_report,
                        "sphere_budget": initialization_config.count,
                        "actual_sphere_count": len(initial_spheres),
                    },
                    repair_budgets={
                        "maximum_subdivision_depth": current_repair.MAXIMUM_SUBDIVISION_DEPTH,
                        "maximum_triangle_splits": current_repair.MAXIMUM_TRIANGLE_SPLITS,
                        "triangles_per_block": current_repair.TRIANGLES_PER_BLOCK,
                        "centers_per_block": current_repair.CENTERS_PER_BLOCK,
                        "target_edge_divisor": current_repair.TARGET_EDGE_DIVISOR,
                    },
                )
                label = f"{link_name}/repair/res{resolution}"
                records["cases"][label] = row
                save()
                print(label, row["seconds_median"], row["peak_allocated_bytes"], flush=True)
            else:
                for method in dict.fromkeys(args.method or ["voxel", "refined-cover"]):
                    configuration = FitConfig(
                        method=method, count=24, resolution=resolution, padding=0.003
                    )
                    previous_configuration = reference.FitConfig(**asdict(configuration))
                    row = paired(
                        {
                            "before": lambda: reference.fit_mesh(mesh, previous_configuration),
                            "after": lambda: fit_mesh(mesh, configuration),
                        },
                        args.repeats,
                        mesh,
                    )
                    assert row["result"]["report"]["voxel_count"] == grid["occupied_count"]
                    row.update(mesh=mesh_record, grid=grid, configuration=asdict(configuration))
                    label = f"{link_name}/{method}/res{resolution}"
                    records["cases"][label] = row
                    save()
                    print(label, row["seconds_median"], row["peak_allocated_bytes"], flush=True)
            assert mesh_fingerprint(mesh) == input_mesh_hash
    assert source_hashes(reference_root) == source_before, "Reference sources changed during run"
    assert source_hashes(current_root) == source_after, "Current sources changed during run"
    assert hashlib.sha256(input_path.read_bytes()).hexdigest() == input_hash, "Input URDF changed"
    records["completed"] = True
    save()


if __name__ == "__main__":
    main()
