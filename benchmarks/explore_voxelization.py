"""Compare voxel shell filling and point occupancy on one shared lattice.

This is a research script, not a replacement backend. Open3D occupancy is a
point predicate, while trimesh subdivide voxelization rasterizes the surface.
The script records their differences rather than treating them as equivalent.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh


@dataclass
class GeometryCase:
    name: str
    mesh: trimesh.Trimesh
    components: list[trimesh.Trimesh]
    solid_policy: str = "parity"
    note: str = "One closed, non-self-intersecting surface."


def make_cases() -> list[GeometryCase]:
    box = trimesh.creation.box(extents=(0.3, 0.16, 0.1))
    cylinder = trimesh.creation.cylinder(radius=0.05, height=0.3, sections=48)
    torus = trimesh.creation.torus(major_radius=0.09, minor_radius=0.025)
    thin_box = trimesh.creation.box(extents=(0.3, 0.16, 0.002))
    # At resolution 48 its thickness lies strictly between z=0 and z=pitch.
    thin_box.apply_translation((0.0, 0.0, 0.003125))
    large_body = trimesh.creation.box(extents=(0.2, 0.12, 0.1))
    small_body = trimesh.creation.box(extents=(0.003, 0.003, 0.003))
    small_body.apply_translation((0.163, 0.003125, 0.003125))
    outer = trimesh.creation.box(extents=(0.3, 0.24, 0.2))
    cavity = trimesh.creation.box(extents=(0.15, 0.12, 0.1))
    cavity.invert()
    left = trimesh.creation.box(extents=(0.2, 0.16, 0.1))
    right = left.copy()
    left.apply_translation((-0.04, 0.0, 0.0))
    right.apply_translation((0.04, 0.0, 0.0))
    return [
        GeometryCase("box", box, [box]),
        GeometryCase("cylinder", cylinder, [cylinder]),
        GeometryCase("torus", torus, [torus]),
        GeometryCase(
            "thin_box_between_planes",
            thin_box,
            [thin_box],
            note="Closed 2 mm sheet between point-lattice layers; not an open mesh.",
        ),
        GeometryCase(
            "disconnected_small_part",
            trimesh.util.concatenate((large_body, small_body)),
            [large_body, small_body],
            "union",
            "Two disjoint closed solid components.",
        ),
        GeometryCase(
            "nested_cavity",
            trimesh.util.concatenate((outer, cavity)),
            [outer, cavity],
            "parity",
            "Explicit odd-even shell semantics preserve the nested cavity.",
        ),
        GeometryCase(
            "overlapping_parts",
            trimesh.util.concatenate((left, right)),
            [left, right],
            "union",
            "Overlapping closed parts violate Open3D occupancy's premise; "
            "the intended reference is their union, not global parity.",
        ),
    ]


def installed_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def make_shared_lattice(mesh: trimesh.Trimesh, pitch: float):
    first_index = np.floor(mesh.bounds[0] / pitch).astype(np.int64) - 1
    last_index = np.ceil(mesh.bounds[1] / pitch).astype(np.int64) + 1
    axes = [np.arange(first, last + 1) for first, last in zip(first_index, last_index)]
    lattice_indices = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    points = lattice_indices.reshape(-1, 3) * pitch
    return points, lattice_indices.shape[:3], first_index * pitch


def reference_contains(case: GeometryCase, points: np.ndarray) -> np.ndarray:
    masks = [component.contains(points) for component in case.components]
    if case.solid_policy == "union":
        return np.logical_or.reduce(masks)
    return np.logical_xor.reduce(masks)


def occupancy_statistics(mask, reference, nonboundary):
    occupied = np.asarray(mask, dtype=bool)
    confident_reference = reference & nonboundary
    false_negative = int(np.count_nonzero(confident_reference & ~occupied))
    false_positive = int(np.count_nonzero(~reference & nonboundary & occupied))
    return {
        "occupied_centers": int(np.count_nonzero(occupied)),
        "reference_interior_nonboundary_centers": int(np.count_nonzero(confident_reference)),
        "false_negative_nonboundary_centers": false_negative,
        "false_positive_nonboundary_centers": false_positive,
        "center_coverage_nonboundary": (
            float(np.mean(occupied[confident_reference])) if confident_reference.any() else None
        ),
        "boundary_centers_excluded": int(np.count_nonzero(~nonboundary)),
    }


def sample_lattice_mask(mask, positions, shape, origin, pitch):
    """Probe the voxelized field, not the continuous native geometry query."""
    integer_indices = np.rint((positions - origin) / pitch).astype(np.int64)
    in_bounds = np.all((integer_indices >= 0) & (integer_indices < shape), axis=1)
    result = np.zeros(len(positions), dtype=bool)
    flattened_indices = np.ravel_multi_index(integer_indices[in_bounds].T, shape)
    result[in_bounds] = np.asarray(mask, dtype=bool)[flattened_indices]
    return result


def build_open3d_scene(mesh, number_of_threads):
    import open3d as open3d_module

    origin = mesh.bounds[0] + mesh.extents * 0.5
    scale = float(mesh.extents.max())
    normalized_vertices = np.asarray((mesh.vertices - origin) / scale, dtype=np.float32)
    scene = open3d_module.t.geometry.RaycastingScene(nthreads=number_of_threads)
    scene.add_triangles(
        open3d_module.core.Tensor(normalized_vertices),
        open3d_module.core.Tensor(np.asarray(mesh.faces, dtype=np.uint32)),
    )
    # Open3D/Embree may defer BVH commitment until the first query. Include a
    # one-point distance query in setup rather than hiding that cost in warmup.
    scene.compute_distance(
        open3d_module.core.Tensor(normalized_vertices[:1]), nthreads=number_of_threads
    ).numpy()
    return scene, origin, scale


def measure_open3d(mesh, points, repetitions, number_of_threads):
    import open3d as open3d_module

    setups, occupancy_times, signed_distance_times = [], [], []
    for _ in range(repetitions):
        start = time.perf_counter()
        scene, origin, scale = build_open3d_scene(mesh, number_of_threads)
        query = open3d_module.core.Tensor(np.asarray((points - origin) / scale, dtype=np.float32))
        setups.append(time.perf_counter() - start)
        start = time.perf_counter()
        occupancy = scene.compute_occupancy(query, nthreads=number_of_threads, nsamples=3).numpy()
        occupancy_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        signed_distance = (
            scene.compute_signed_distance(query, nthreads=number_of_threads, nsamples=3).numpy()
            * scale
        )
        signed_distance_times.append(time.perf_counter() - start)
    result = {
        "scene_setup_times_s": setups,
        "occupancy_query_times_s": occupancy_times,
        "signed_distance_query_times_s": signed_distance_times,
        "scene_setup_median_s": statistics.median(setups),
        "occupancy_query_median_s": statistics.median(occupancy_times),
        "signed_distance_query_median_s": statistics.median(signed_distance_times),
        "occupancy_full_median_s": statistics.median(np.array(setups) + occupancy_times),
        "signed_distance_full_median_s": statistics.median(
            np.array(setups) + signed_distance_times
        ),
        "dtype": "Float32 after per-mesh translation/scale normalization",
        "nsamples": 3,
    }
    return occupancy > 0.5, signed_distance, result, (scene, origin, scale)


def measure_rasterization(mesh, points, pitch, method, repetitions):
    durations = []
    for _ in range(repetitions):
        # Copies keep reference.contains's parent cache from warming one method.
        candidate_mesh = mesh.copy()
        start = time.perf_counter()
        grid = candidate_mesh.voxelized(pitch, method=method).fill()
        durations.append(time.perf_counter() - start)
    return grid.is_filled(points), {
        "full_times_s": durations,
        "full_median_s": statistics.median(durations),
        "raw_occupied_voxel_count": int(grid.filled_count),
        "grid_transform": grid.transform.tolist(),
    }


def benchmark_case(case, args):
    pitch = float(case.mesh.extents.max()) / args.resolution
    started = time.perf_counter()
    points, shape, origin = make_shared_lattice(case.mesh, pitch)
    grid_preparation = time.perf_counter() - started
    reference = reference_contains(case, points)
    occupied, signed_distance, open3d_result, scene_info = measure_open3d(
        case.mesh, points, args.repeats, args.threads
    )
    # The metric excludes exact/near boundary centers. Float32 ray/SDF values
    # can have ambiguous sign there; report the exclusion count explicitly.
    boundary_tolerance = max(pitch * 1e-4, float(case.mesh.extents.max()) * 2e-6)
    nonboundary = np.abs(signed_distance) > boundary_tolerance
    methods = {
        "open3d_occupancy": {
            **open3d_result,
            **occupancy_statistics(occupied, reference, nonboundary),
        },
        "open3d_signed_interior": occupancy_statistics(signed_distance < 0, reference, nonboundary),
        "open3d_sdf_surface_band": {
            **occupancy_statistics(
                signed_distance <= np.sqrt(3) * pitch / 2, reference, nonboundary
            ),
            "offset_m": float(np.sqrt(3) * pitch / 2),
            "meaning": "Include inside centers plus a half-cell-diagonal exterior band; "
            "different sampling semantics from strict point occupancy.",
        },
    }
    method_masks = {
        "open3d_occupancy": occupied,
        "open3d_signed_interior": signed_distance < 0,
        "open3d_sdf_surface_band": signed_distance <= np.sqrt(3) * pitch / 2,
    }
    for method in ("subdivide", "ray"):
        try:
            mask, timing = measure_rasterization(case.mesh, points, pitch, method, args.repeats)
            methods[f"trimesh_{method}_fill"] = {
                **timing,
                **occupancy_statistics(mask, reference, nonboundary),
            }
            method_masks[f"trimesh_{method}_fill"] = mask
        except (ValueError, RuntimeError, ImportError, IndexError) as error:
            methods[f"trimesh_{method}_fill"] = {"error": str(error)}
    rng = np.random.default_rng(99173)
    heldout_points = rng.uniform(
        case.mesh.bounds[0] - pitch, case.mesh.bounds[1] + pitch, size=(10000, 3)
    )
    heldout_reference = reference_contains(case, heldout_points)
    scene, scene_origin, scene_scale = scene_info
    import open3d as open3d_module

    heldout_query = open3d_module.core.Tensor(
        np.asarray((heldout_points - scene_origin) / scene_scale, dtype=np.float32)
    )
    heldout_distance = (
        scene.compute_signed_distance(heldout_query, nthreads=args.threads, nsamples=3).numpy()
        * scene_scale
    )
    heldout_confident = np.abs(heldout_distance) > boundary_tolerance
    open3d_result["heldout_disagreement_nonboundary"] = int(
        np.count_nonzero(((heldout_distance < 0) != heldout_reference) & heldout_confident)
    )
    open3d_result["heldout_points"] = len(heldout_points)
    methods["open3d_occupancy"].update(
        {key: open3d_result[key] for key in ("heldout_disagreement_nonboundary", "heldout_points")}
    )
    feature_probes = np.array([component.bounds.mean(axis=0) for component in case.components])
    for method, mask in method_masks.items():
        methods[method]["component_center_voxel_probes"] = sample_lattice_mask(
            mask, feature_probes, shape, origin, pitch
        ).tolist()
    return {
        "name": case.name,
        "triangles": len(case.mesh.faces),
        "watertight": bool(case.mesh.is_watertight),
        "surface_bodies": int(case.mesh.body_count),
        "reference_solid_policy": case.solid_policy,
        "note": case.note,
        "pitch_m": pitch,
        "shared_grid_shape": list(shape),
        "shared_grid_points": len(points),
        "shared_grid_origin_m": origin.tolist(),
        "grid_preparation_s": grid_preparation,
        "boundary_tolerance_m": boundary_tolerance,
        "component_center_probe_positions_m": feature_probes.tolist(),
        "component_center_reference_inside": reference_contains(case, feature_probes).tolist(),
        "methods": methods,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=48)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--case", action="append", choices=[case.name for case in make_cases()])
    parser.add_argument("--output", type=Path, default=Path("outputs/explore-voxelization.json"))
    args = parser.parse_args()
    if args.resolution < 4 or args.repeats < 1 or args.threads < 1:
        parser.error("resolution must be >=4; repeats and threads must be positive")
    import_started = time.perf_counter()
    import open3d as open3d_module

    import_seconds = time.perf_counter() - import_started
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            package: installed_version(package)
            for package in ("open3d", "trimesh", "numpy", "scipy", "embreex")
        },
        "trimesh_ray_engine": type(trimesh.creation.box().ray).__module__,
        "open3d_import_s": import_seconds,
        "open3d_core_cuda_available": open3d_module.core.cuda.is_available(),
        "open3d_sycl_available": open3d_module.core.sycl.is_available(),
        "raycasting_device": "CPU:0; this API has no CUDA device implementation",
        "query_threads": args.threads,
        "repeats": args.repeats,
        "lattice_anchor": "integer multiples of pitch in mesh coordinates",
        "limitations": [
            "The workstation is not isolated from other CPU/GPU work",
            "Surface rasterization and point occupancy have different semantics",
            "Overlapping parts intentionally violate Open3D sign-query assumptions",
            "No sphere-fitting quality or downstream collision benchmark here",
        ],
        "cases": [],
    }
    for case in make_cases():
        if args.case and case.name not in args.case:
            continue
        result["cases"].append(benchmark_case(case, args))
        completed = result["cases"][-1]
        metrics = completed["methods"]
        occupancy_milliseconds = metrics["open3d_occupancy"]["occupancy_full_median_s"] * 1000
        subdivide_milliseconds = metrics["trimesh_subdivide_fill"].get("full_median_s", 0) * 1000
        print(
            f"{case.name}: points={completed['shared_grid_points']:,}; "
            f"Open3D occupancy={occupancy_milliseconds:.2f} ms; "
            f"subdivide={subdivide_milliseconds:.2f} ms",
            flush=True,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
