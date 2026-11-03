from __future__ import annotations

import heapq
import importlib
import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree

from .asset import Asset
from .model import FitConfig, FitResult, Sphere


@dataclass
class _PreparedVoxelGrid:
    voxel_grid: trimesh.voxel.VoxelGrid
    voxel_size_m: float
    # Optional attachment point for backends that query the original mesh.
    clearance_grid_m: np.ndarray | None = None
    backend_details: dict[str, Any] = field(default_factory=dict)


@dataclass
class _VoxelCandidates:
    candidate_centers: np.ndarray
    candidate_radii: np.ndarray
    coverage_samples: np.ndarray
    occupied_points: np.ndarray
    voxel_size_m: float
    backend_details: dict[str, Any] = field(default_factory=dict)


def sample_surface(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    """Area-weighted triangle sampling without touching NumPy's global RNG."""
    random_generator = np.random.default_rng(seed)
    triangle_indices = random_generator.choice(
        len(mesh.faces), count, p=mesh.area_faces / mesh.area
    )
    triangles = mesh.triangles[triangle_indices]
    barycentric_offsets = random_generator.random((count, 2))
    outside_triangle = barycentric_offsets.sum(axis=1) > 1
    barycentric_offsets[outside_triangle] = 1 - barycentric_offsets[outside_triangle]
    return (
        triangles[:, 0]
        + barycentric_offsets[:, :1] * (triangles[:, 1] - triangles[:, 0])
        + barycentric_offsets[:, 1:] * (triangles[:, 2] - triangles[:, 0])
    )


def sphere_gaps(points: np.ndarray, spheres: list[Sphere]) -> np.ndarray:
    """Ball-union gap: exact exterior distance, negative inside; bounded memory."""
    if not spheres:
        return np.full(len(points), np.inf)
    sphere_centers = np.asarray([sphere.center for sphere in spheres])
    sphere_radii = np.asarray([sphere.radius for sphere in spheres])
    minimum_gaps = np.full(len(points), np.inf)
    for first_point in range(0, len(points), 2048):
        last_point = first_point + 2048
        for first_sphere in range(0, len(spheres), 128):
            block_gaps = (
                np.linalg.norm(
                    points[first_point:last_point, None, :]
                    - sphere_centers[None, first_sphere : first_sphere + 128, :],
                    axis=2,
                )
                - sphere_radii[None, first_sphere : first_sphere + 128]
            )
            minimum_gaps[first_point:last_point] = np.minimum(
                minimum_gaps[first_point:last_point], block_gaps.min(axis=1)
            )
    return minimum_gaps


def _make_spheres(centers: np.ndarray, radii: np.ndarray) -> list[Sphere]:
    return [
        Sphere(tuple(float(coordinate) for coordinate in center), float(radius))
        for center, radius in zip(centers, radii)
    ]


def _farthest_point_indices(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    sample_count = min(count, len(points))
    selected_indices = np.zeros(sample_count, dtype=int)
    nearest_squared_distance = np.full(len(points), np.inf)
    current_index = int(np.random.default_rng(seed).integers(len(points)))
    for selection_index in range(sample_count):
        selected_indices[selection_index] = current_index
        nearest_squared_distance = np.minimum(
            nearest_squared_distance, np.sum((points - points[current_index]) ** 2, axis=1)
        )
        nearest_squared_distance[selected_indices[: selection_index + 1]] = -1
        current_index = int(np.argmax(nearest_squared_distance))
    return selected_indices


def _fit_surface_spheres(mesh: trimesh.Trimesh, configuration: FitConfig) -> list[Sphere]:
    candidate_points = sample_surface(
        mesh, max(configuration.samples, configuration.count * 4), configuration.seed
    )
    selected_points = candidate_points[
        _farthest_point_indices(candidate_points, configuration.count, configuration.seed)
    ]
    radius = configuration.surface_radius + configuration.padding
    return _make_spheres(selected_points, np.full(len(selected_points), radius))


def _partition_cover_triangles(
    mesh: trimesh.Trimesh, configuration: FitConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Smaller triangles improve coarse caps without changing the mesh boundary.
    target_edge_length = float(mesh.extents.max()) / 12
    if len(mesh.faces) < 15_000:
        vertices, faces = trimesh.remesh.subdivide_to_size(
            mesh.vertices, mesh.faces, max_edge=target_edge_length, max_iter=8
        )
        triangles = vertices[faces]
    else:
        triangles = mesh.triangles
    triangle_centers = triangles.mean(axis=1)
    triangle_area_weights = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    cluster_centers = triangle_centers[
        _farthest_point_indices(triangle_centers, configuration.count, configuration.seed)
    ].copy()
    for _ in range(12):
        cluster_labels = cKDTree(cluster_centers).query(triangle_centers)[1]
        updated_centers = cluster_centers.copy()
        for cluster_index in range(len(cluster_centers)):
            assigned_mask = cluster_labels == cluster_index
            if assigned_mask.any():
                updated_centers[cluster_index] = np.average(
                    triangle_centers[assigned_mask],
                    axis=0,
                    weights=triangle_area_weights[assigned_mask],
                )
        if np.allclose(updated_centers, cluster_centers, rtol=0, atol=target_edge_length * 1e-6):
            break
        cluster_centers = updated_centers
    cluster_labels = cKDTree(cluster_centers).query(triangle_centers)[1]
    return triangles, cluster_centers, cluster_labels


def _fit_triangle_cover(mesh: trimesh.Trimesh, configuration: FitConfig) -> list[Sphere]:
    """Enclose whole triangle partitions, preserving their boundary coverage.

    A ball is convex: containing each triangle's vertices contains its interior.
    This argument concerns the boundary, not the solid enclosed by the mesh.
    """
    triangles, cluster_centers, cluster_labels = _partition_cover_triangles(mesh, configuration)
    spheres = []
    for cluster_index, center in enumerate(cluster_centers):
        assigned_triangles = triangles[cluster_labels == cluster_index]
        if not len(assigned_triangles):
            continue
        radius = np.linalg.norm(assigned_triangles - center, axis=2).max()
        if configuration.method == "cover-tight":
            from .enclosing import tighten_ball

            center, radius, _ = tighten_ball(assigned_triangles.reshape(-1, 3), center)
        radius = np.nextafter(radius, np.inf) + configuration.padding
        spheres.extend(_make_spheres(center[None], np.array([radius])))
    return spheres


def _prepare_voxel_grid(mesh: trimesh.Trimesh, configuration: FitConfig) -> _PreparedVoxelGrid:
    voxel_size_m = configuration.voxel_size or float(mesh.extents.max()) / configuration.resolution
    if configuration.voxelizer != "subdivide":
        from .voxelization import prepare_distance_grid

        distance_grid = prepare_distance_grid(mesh, configuration)
        return _PreparedVoxelGrid(
            distance_grid.voxel_grid,
            voxel_size_m,
            distance_grid.clearance_grid_m,
            distance_grid.backend_details,
        )
    with np.errstate(over="ignore", invalid="ignore"):
        extent_ratios = mesh.extents / voxel_size_m
    if not np.all(np.isfinite(extent_ratios)):
        raise ValueError("Voxel size is too small to represent the grid; increase --voxel-size")
    shape_bound = [math.ceil(float(ratio)) + 7 for ratio in extent_ratios]
    cell_count_bound = math.prod(shape_bound)
    if cell_count_bound > configuration.max_grid_cells:
        raise ValueError(
            f"Voxel grid needs up to {cell_count_bound:,} cells "
            f"(limit {configuration.max_grid_cells:,}); "
            "increase --voxel-size or reduce --resolution"
        )
    voxel_grid = mesh.voxelized(voxel_size_m)
    if math.prod(voxel_grid.shape) > configuration.max_grid_cells:
        raise ValueError("Voxel grid exceeds max_grid_cells; increase voxel_size")
    return _PreparedVoxelGrid(voxel_grid.fill(), voxel_size_m)


def _prepare_voxel_candidates(mesh: trimesh.Trimesh, configuration: FitConfig) -> _VoxelCandidates:
    prepared_grid = _prepare_voxel_grid(mesh, configuration)
    voxel_grid = prepared_grid.voxel_grid
    voxel_size_m = prepared_grid.voxel_size_m
    occupied_cells = voxel_grid.matrix
    occupied_indices = np.argwhere(occupied_cells)
    if not len(occupied_indices):
        raise ValueError("Voxelization produced no occupied cells; increase resolution")
    occupied_points = trimesh.transform_points(occupied_indices, voxel_grid.transform)
    if prepared_grid.clearance_grid_m is None:
        # EDT needs exterior cells even when the mesh fills its entire voxel bounds.
        distance_grid = ndimage.distance_transform_edt(np.pad(occupied_cells, 1))[1:-1, 1:-1, 1:-1]
        clearance_radii = (
            np.maximum(distance_grid[tuple(occupied_indices.T)] - 0.5, 0.5) * voxel_size_m
        )
    else:
        # SDF backends already supply metric radii from the ORIGINAL mesh.
        distance_grid = prepared_grid.clearance_grid_m
        clearance_radii = distance_grid[tuple(occupied_indices.T)]
    local_maxima = distance_grid >= ndimage.maximum_filter(distance_grid, size=3)
    peak_indices = np.flatnonzero(local_maxima[tuple(occupied_indices.T)])
    random_generator = np.random.default_rng(configuration.seed)
    candidate_budget = min(
        max(configuration.max_candidates, configuration.count), len(occupied_points)
    )
    if len(peak_indices) > candidate_budget // 2:
        peak_indices = random_generator.choice(
            peak_indices, max(1, candidate_budget // 2), replace=False
        )
    remaining_indices = np.setdiff1d(np.arange(len(occupied_points)), peak_indices)
    random_indices = random_generator.choice(
        remaining_indices,
        min(candidate_budget - len(peak_indices), len(remaining_indices)),
        replace=False,
    )
    candidate_indices = np.unique(
        np.r_[peak_indices, random_indices, int(np.argmax(clearance_radii))]
    )
    component_grid_indices = prepared_grid.backend_details.get(
        "component_representative_grid_indices"
    )
    if component_grid_indices and len(component_grid_indices) > 1:
        # Small disconnected solids can be missed by area/volume-proportional
        # random sampling. Retain at least their deepest interior candidate.
        occupied_flat_indices = np.ravel_multi_index(occupied_indices.T, occupied_cells.shape)
        component_flat_indices = np.ravel_multi_index(
            np.asarray(component_grid_indices).T, occupied_cells.shape
        )
        representative_indices = np.searchsorted(occupied_flat_indices, component_flat_indices)
        candidate_indices = np.unique(np.r_[candidate_indices, representative_indices])
    coverage_samples = occupied_points
    if len(coverage_samples) > configuration.samples:
        coverage_samples = coverage_samples[
            random_generator.choice(len(coverage_samples), configuration.samples, replace=False)
        ]
    return _VoxelCandidates(
        candidate_centers=occupied_points[candidate_indices],
        candidate_radii=clearance_radii[candidate_indices] + configuration.padding,
        coverage_samples=coverage_samples,
        occupied_points=occupied_points,
        voxel_size_m=voxel_size_m,
        backend_details=prepared_grid.backend_details,
    )


def _greedy_cover_cpu(
    point_samples: np.ndarray,
    candidate_centers: np.ndarray,
    candidate_radii: np.ndarray,
    sphere_count: int,
) -> list[int]:
    sample_tree = cKDTree(point_samples)
    covered_sample_indices = [
        np.asarray(sample_tree.query_ball_point(center, radius), dtype=int)
        for center, radius in zip(candidate_centers, candidate_radii)
    ]
    covered_mask = np.zeros(len(point_samples), dtype=bool)
    priorities = [
        (-len(sample_indices), candidate_index)
        for candidate_index, sample_indices in enumerate(covered_sample_indices)
    ]
    heapq.heapify(priorities)
    selected_indices = []
    while priorities and len(selected_indices) < sphere_count:
        _, candidate_index = heapq.heappop(priorities)
        candidate_samples = covered_sample_indices[candidate_index]
        uncovered_count = int(np.count_nonzero(~covered_mask[candidate_samples]))
        if uncovered_count == 0:
            continue
        # The tuple comparison preserves the shared CPU/CUDA tie rule.
        if priorities and (-uncovered_count, candidate_index) > priorities[0]:
            heapq.heappush(priorities, (-uncovered_count, candidate_index))
            continue
        selected_indices.append(candidate_index)
        covered_mask[candidate_samples] = True
    return selected_indices


def _select_covering_candidates(
    candidates: _VoxelCandidates, configuration: FitConfig
) -> list[int]:
    representatives = candidates.backend_details.get("component_representative_centers_m", [])
    if len(representatives) > 1:
        return _select_component_aware_candidates(
            candidates, configuration, np.asarray(representatives)
        )
    if configuration.device == "cuda":
        from .tensor_backend import greedy_cover

        return greedy_cover(
            candidates.coverage_samples,
            candidates.candidate_centers,
            candidates.candidate_radii,
            count=configuration.count,
            device="cuda",
        ).tolist()
    return _greedy_cover_cpu(
        candidates.coverage_samples,
        candidates.candidate_centers,
        candidates.candidate_radii,
        configuration.count,
    )


def _select_component_aware_candidates(
    candidates: _VoxelCandidates,
    configuration: FitConfig,
    component_representatives: np.ndarray,
) -> list[int]:
    """Cover each sampled component representative before spending spare budget.

    This prevents a tiny sampled component from disappearing behind a large
    volume target. It does not certify coverage of each entire component.
    """
    priority_order = np.argsort(-candidates.candidate_radii, kind="stable")
    priority_selected = _greedy_cover_cpu(
        component_representatives,
        candidates.candidate_centers[priority_order],
        candidates.candidate_radii[priority_order],
        configuration.count,
    )
    selected = priority_order[priority_selected].tolist()
    candidates.backend_details["component_preselection_device"] = "cpu"
    candidates.backend_details["component_preselection_sphere_count"] = len(selected)
    mandatory_spheres = _make_spheres(
        candidates.candidate_centers[selected], candidates.candidate_radii[selected]
    )
    if np.any(sphere_gaps(component_representatives, mandatory_spheres) > 0):
        raise ValueError(
            "Sphere budget cannot retain every component representative; increase --count "
            "or fit the components separately"
        )
    uncovered_samples = candidates.coverage_samples[
        sphere_gaps(candidates.coverage_samples, mandatory_spheres) > 0
    ]
    remaining_budget = configuration.count - len(selected)
    if remaining_budget <= 0 or not len(uncovered_samples):
        return selected
    remaining_indices = np.setdiff1d(np.arange(len(candidates.candidate_centers)), selected)
    remaining_centers = candidates.candidate_centers[remaining_indices]
    remaining_radii = candidates.candidate_radii[remaining_indices]
    if configuration.device == "cuda":
        from .tensor_backend import greedy_cover

        added = greedy_cover(
            uncovered_samples,
            remaining_centers,
            remaining_radii,
            count=remaining_budget,
            device="cuda",
        ).tolist()
    else:
        added = _greedy_cover_cpu(
            uncovered_samples, remaining_centers, remaining_radii, remaining_budget
        )
    return selected + remaining_indices[added].tolist()


def _fit_voxel_spheres(
    mesh: trimesh.Trimesh, configuration: FitConfig
) -> tuple[list[Sphere], dict[str, Any], np.ndarray]:
    candidates = _prepare_voxel_candidates(mesh, configuration)
    selected_indices = _select_covering_candidates(candidates, configuration)
    if not selected_indices:
        raise ValueError("No voxel sphere candidates cover the sampled volume")
    spheres = _make_spheres(
        candidates.candidate_centers[selected_indices], candidates.candidate_radii[selected_indices]
    )
    details = {
        "voxel_size_m": candidates.voxel_size_m,
        "voxel_count": len(candidates.occupied_points),
        "candidate_count": len(candidates.candidate_centers),
        "greedy_device": configuration.device,
        "radius_model": "voxel EDT minus half a voxel; approximate, not exact mesh SDF",
        **candidates.backend_details,
    }
    return spheres, details, candidates.occupied_points


def _fit_lula_spheres(mesh: trimesh.Trimesh, configuration: FitConfig) -> list[Sphere]:
    try:
        lula = importlib.import_module("lula")
    except ImportError as exc:
        raise ValueError(
            "Lula backend requires Isaac Sim's lula bindings in the current interpreter. "
            "Enable its Lula extension before importing AssetSpheres, or use --method voxel."
        ) from exc
    if not hasattr(lula, "create_collision_sphere_generator"):
        raise ValueError("Installed lula module does not expose create_collision_sphere_generator")
    generator = lula.create_collision_sphere_generator(
        [np.asarray(vertex, dtype=np.float64).reshape(3, 1) for vertex in mesh.vertices],
        [np.asarray(face, dtype=np.int32).reshape(3, 1) for face in mesh.faces],
    )
    generated_spheres = generator.generate_spheres(configuration.count, configuration.padding)
    if not generated_spheres:
        raise ValueError("Lula returned no spheres; check mesh topology or use --method cover")
    return [
        Sphere(tuple(np.asarray(sphere.center).reshape(3).tolist()), float(sphere.radius))
        for sphere in generated_spheres
    ]


def _prepare_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError("Expected a nonempty triangle mesh")
    if not np.all(np.isfinite(mesh.vertices)) or not np.isfinite(mesh.area) or mesh.area <= 0:
        raise ValueError("Mesh must have finite vertices and positive surface area")
    prepared_mesh = mesh.copy()
    prepared_mesh.update_faces(prepared_mesh.nondegenerate_faces())
    prepared_mesh.remove_unreferenced_vertices()
    if not len(prepared_mesh.faces) or prepared_mesh.area <= 0:
        raise ValueError("Mesh has no nondegenerate triangles")
    return prepared_mesh


def _resolve_configuration(
    mesh: trimesh.Trimesh, requested: FitConfig
) -> tuple[FitConfig, list[str]]:
    needs_volume = requested.method in {"voxel", "refined", "lula"}
    if not needs_volume or mesh.is_watertight:
        return requested, []
    if requested.non_watertight == "error":
        raise ValueError(
            "Volume fitting requires a watertight mesh. Repair it, use --method surface/cover, "
            "or explicitly pass --non-watertight surface/cover."
        )
    effective = replace(requested, method=requested.non_watertight, device="cpu")
    return effective, [f"Non-watertight mesh: explicitly selected fallback {effective.method}"]


def _fit_with_method(
    mesh: trimesh.Trimesh, configuration: FitConfig
) -> tuple[list[Sphere], dict[str, Any], np.ndarray | None, list[str]]:
    details: dict[str, Any] = {}
    occupied_points = None
    warnings = []
    if configuration.method in {"voxel", "refined"}:
        # Refinement adds user padding only after optimization, exactly once.
        initial_configuration = (
            replace(configuration, padding=0)
            if configuration.method == "refined"
            else configuration
        )
        spheres, details, occupied_points = _fit_voxel_spheres(mesh, initial_configuration)
        if configuration.method == "refined":
            from .refine import refine_spheres

            spheres, refinement_details = refine_spheres(
                mesh, spheres, occupied_points, details["voxel_size_m"], configuration
            )
            details.update(refinement_details)
            warnings.append(
                "Sampled refinement trades coverage against exterior volume; no certificate"
            )
        warnings.append("Voxel spheres approximate volume and can leave gaps; inspect coverage")
        warnings.extend(details.get("voxelizer_warnings", []))
    elif configuration.method == "surface":
        spheres = _fit_surface_spheres(mesh, configuration)
        warnings.append("Fixed-radius surface spheres do not guarantee solid-volume coverage")
    elif configuration.method in {"cover", "cover-tight"}:
        spheres = _fit_triangle_cover(mesh, configuration)
        details["triangle_surface_coverage_certified"] = True
        warnings.append("Triangle boundary is enclosed; solid interior coverage is not certified")
    elif configuration.method == "slab":
        from .slab import slab_spheres

        spheres, details = slab_spheres(mesh, configuration)
        details["triangle_surface_coverage_certified"] = True
        details["bounded_solid_coverage_conditional"] = bool(mesh.is_watertight)
        details["solid_coverage_assumption"] = (
            "Input triangle surface is the complete boundary of a bounded solid; "
            "floating-point clipping and outward radius margin, not interval arithmetic."
        )
        details["self_intersection_checked"] = False
        warnings.append(
            "Parallel-slab balls enclose the surface and, assuming a complete bounded-solid "
            "boundary, its interior; transverse branches/holes may be over-enclosed"
        )
    else:
        spheres = _fit_lula_spheres(mesh, configuration)
        warnings.append("Lula backend quality depends on its installed version and mesh topology")
    return spheres, details, occupied_points, warnings


def _evaluate_sphere_gaps(
    point_samples: np.ndarray, spheres: list[Sphere], device: str
) -> np.ndarray:
    if device == "cuda":
        from .tensor_backend import point_sphere_gaps

        return point_sphere_gaps(
            point_samples,
            np.array([sphere.center for sphere in spheres]),
            np.array([sphere.radius for sphere in spheres]),
            device="cuda",
        )
    return sphere_gaps(point_samples, spheres)


def _quality_report(
    mesh: trimesh.Trimesh,
    spheres: list[Sphere],
    requested: FitConfig,
    effective: FitConfig,
    details: dict[str, Any],
    warnings: list[str],
    occupied_points: np.ndarray | None,
) -> dict[str, Any]:
    surface_samples = sample_surface(mesh, requested.samples, requested.seed + 1)
    surface_gaps = _evaluate_sphere_gaps(surface_samples, spheres, effective.device)
    positive_gaps = np.maximum(surface_gaps, 0)
    report = {
        "requested_method": requested.method,
        "effective_method": effective.method,
        "requested_count": requested.count,
        "sphere_count": len(spheres),
        "watertight": bool(mesh.is_watertight),
        "triangle_count": len(mesh.faces),
        "surface_sample_count": len(surface_samples),
        "surface_coverage": float(np.mean(surface_gaps <= mesh.extents.max() * 1e-9)),
        "surface_gap_mean_m": float(positive_gaps.mean()),
        "surface_gap_p95_m": float(np.quantile(positive_gaps, 0.95)),
        "surface_gap_max_m": float(positive_gaps.max()),
        "sphere_volume_sum_m3": float(sum(4 * np.pi * sphere.radius**3 / 3 for sphere in spheres)),
        "warnings": warnings,
        **details,
    }
    if occupied_points is not None:
        report["voxel_center_coverage"] = float(
            np.mean(_evaluate_sphere_gaps(occupied_points, spheres, effective.device) <= 0)
        )
    representatives = details.get("component_representative_centers_m", [])
    if len(representatives) > 1:
        component_gaps = sphere_gaps(np.asarray(representatives), spheres)
        if np.any(component_gaps > 0):
            raise ValueError(
                "Optimization lost a component representative; use --method voxel, "
                "increase --count, or fit the components separately"
            )
        report["component_representative_coverage"] = 1.0
        report["component_coverage_scope"] = (
            "Deep interior representatives, not full component coverage"
        )
    return report


def fit_mesh(
    mesh: trimesh.Trimesh, config: FitConfig | None = None
) -> tuple[list[Sphere], dict[str, Any]]:
    """Fit a triangle mesh in its existing frame; return spheres and quality report."""
    requested_configuration = config or FitConfig()
    prepared_mesh = _prepare_mesh(mesh)
    effective_configuration, warnings = _resolve_configuration(
        prepared_mesh, requested_configuration
    )
    spheres, details, occupied_points, method_warnings = _fit_with_method(
        prepared_mesh, effective_configuration
    )
    report = _quality_report(
        prepared_mesh,
        spheres,
        requested_configuration,
        effective_configuration,
        details,
        warnings + method_warnings,
        occupied_points,
    )
    return spheres, report


def fit_asset(
    asset: Asset,
    config: FitConfig | None = None,
    *,
    link_counts: dict[str, int] | None = None,
    workers: int = 1,
    executor: str = "thread",
) -> FitResult:
    configuration = config or FitConfig()
    sphere_count_overrides = link_counts or {}
    unknown_links = set(sphere_count_overrides) - asset.links.keys()
    if unknown_links:
        raise ValueError(f"Sphere count overrides refer to unloaded links: {sorted(unknown_links)}")
    if (
        configuration.device == "cuda"
        and isinstance(workers, int)
        and workers > 1
        and len(asset.links) > 1
    ):
        raise ValueError("CUDA fitting uses one GPU owner; use workers=1")
    from .parallel import fit_links

    link_tasks = []
    for link_name, link in asset.links.items():
        link_configuration = replace(
            configuration, count=sphere_count_overrides.get(link_name, configuration.count)
        )
        link_tasks.append((link_name, link.mesh, link_configuration))
    fitted_links, link_reports = {}, {}
    for link_name, spheres, report in fit_links(link_tasks, workers=workers, executor=executor):
        fitted_links[link_name], link_reports[link_name] = spheres, report
        report["geometry_source"] = asset.links[link_name].geometry_source
        geometry_types = asset.links[link_name].geometry_types
        if geometry_types:
            report["input_geometry_types"] = list(geometry_types)
        if {"sphere", "cylinder"} & set(geometry_types):
            report["warnings"].append(
                "URDF analytic shapes are tessellated; guarantees concern the triangle mesh"
            )
    return FitResult(
        str(asset.path),
        configuration,
        fitted_links,
        link_reports,
        {name: transform.tolist() for name, transform in asset.zero_pose_transforms.items()},
        {
            "geometry": asset.geometry,
            "scale": asset.mesh_scale,
            "package_roots": {name: str(path) for name, path in asset.package_roots.items()},
        },
    )
