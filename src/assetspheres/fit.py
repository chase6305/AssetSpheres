"""Coordinate mesh preparation, fitting methods, and quality reporting.

Method implementations live in surface_fit, voxel_fit, slab, and refine.
Shared geometry helpers are re-exported here for existing Python callers.
"""

from __future__ import annotations

import importlib
from dataclasses import replace
from typing import Any

import numpy as np
import trimesh

from .asset import Asset
from .geometry import sample_surface, sphere_gaps
from .model import FitConfig, FitResult, Sphere
from .surface_fit import fit_surface_spheres, fit_triangle_cover
from .voxel_fit import fit_voxel_spheres


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
    needs_volume = requested.method in {"voxel", "refined", "refined-cover", "lula"}
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
    if configuration.method in {"voxel", "refined", "refined-cover"}:
        # Refinement adds user padding only after optimization, exactly once.
        initial_configuration = (
            replace(configuration, padding=0)
            if configuration.method in {"refined", "refined-cover"}
            else configuration
        )
        spheres, details, occupied_points = fit_voxel_spheres(mesh, initial_configuration)
        if configuration.method in {"refined", "refined-cover"}:
            from .refine import refine_spheres

            spheres, refinement_details = refine_spheres(
                mesh, spheres, occupied_points, details["voxel_size_m"], configuration
            )
            details.update(refinement_details)
            if configuration.method == "refined-cover":
                from .surface_cover import repair_surface_cover

                spheres, repair_details = repair_surface_cover(mesh, spheres)
                details.update(repair_details)
                details["triangle_surface_coverage_certified"] = True
                details["surface_repair_device"] = "cpu"
                details["optimization_scope"] = "before user padding and surface repair"
                warnings.append(
                    "Triangle leaves are enclosed after repair; solid interiors are not certified"
                )
                if repair_details["surface_repair_coarse_leaves"]:
                    warnings.append(
                        "Surface repair retained coarse leaves at its subdivision limit; "
                        "inspect exterior volume"
                    )
            else:
                warnings.append(
                    "Sampled refinement trades coverage against exterior volume; no certificate"
                )
        warnings.append("Voxel spheres approximate volume and can leave gaps; inspect coverage")
        warnings.extend(details.get("voxelizer_warnings", []))
    elif configuration.method == "surface":
        spheres = fit_surface_spheres(mesh, configuration)
        warnings.append("Fixed-radius surface spheres do not guarantee solid-volume coverage")
    elif configuration.method in {"cover", "cover-tight", "cover-opt"}:
        spheres = fit_triangle_cover(mesh, configuration)
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
