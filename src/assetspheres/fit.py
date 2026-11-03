from __future__ import annotations

import heapq
import importlib
import math
from dataclasses import replace
from typing import Any

import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree

from .asset import Asset
from .model import FitConfig, FitResult, Sphere


def sample_surface(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    """Area-weighted triangle sampling without touching NumPy's global RNG."""
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(mesh.faces), count, p=mesh.area_faces / mesh.area)
    triangles = mesh.triangles[indices]
    uv = rng.random((count, 2))
    uv[uv.sum(axis=1) > 1] = 1 - uv[uv.sum(axis=1) > 1]
    return (
        triangles[:, 0]
        + uv[:, :1] * (triangles[:, 1] - triangles[:, 0])
        + uv[:, 1:] * (triangles[:, 2] - triangles[:, 0])
    )


def sphere_gaps(points: np.ndarray, spheres: list[Sphere]) -> np.ndarray:
    """Ball-union gap: exact exterior distance, negative inside; bounded memory."""
    if not spheres:
        return np.full(len(points), np.inf)
    centers = np.asarray([s.center for s in spheres])
    radii = np.asarray([s.radius for s in spheres])
    result = np.full(len(points), np.inf)
    for start in range(0, len(points), 2048):
        stop = start + 2048
        for first in range(0, len(spheres), 128):
            distance = (
                np.linalg.norm(
                    points[start:stop, None, :] - centers[None, first : first + 128, :], axis=2
                )
                - radii[None, first : first + 128]
            )
            result[start:stop] = np.minimum(result[start:stop], distance.min(axis=1))
    return result


def _spheres(centers: np.ndarray, radii: np.ndarray) -> list[Sphere]:
    return [Sphere(tuple(float(x) for x in c), float(r)) for c, r in zip(centers, radii)]


def _farthest_indices(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    count = min(count, len(points))
    selected = np.zeros(count, dtype=int)
    distance = np.full(len(points), np.inf)
    current = int(np.random.default_rng(seed).integers(len(points)))
    for i in range(count):
        selected[i] = current
        distance = np.minimum(distance, np.sum((points - points[current]) ** 2, axis=1))
        distance[selected[: i + 1]] = -1
        current = int(np.argmax(distance))
    return selected


def _surface(mesh: trimesh.Trimesh, cfg: FitConfig) -> list[Sphere]:
    points = sample_surface(mesh, max(cfg.samples, cfg.count * 4), cfg.seed)
    points = points[_farthest_indices(points, cfg.count, cfg.seed)]
    return _spheres(points, np.full(len(points), cfg.surface_radius + cfg.padding))


def _cover(mesh: trimesh.Trimesh, cfg: FitConfig) -> list[Sphere]:
    """Partition entire triangles, then enclose each partition in a ball.

    A ball is convex, so containing a triangle's three vertices contains the
    entire triangle. This certifies boundary coverage, not solid-volume coverage.
    """
    # Modest refinement improves coarse boxes/caps without changing their surface.
    max_edge = float(mesh.extents.max()) / 12
    if len(mesh.faces) < 15_000:
        vertices, faces = trimesh.remesh.subdivide_to_size(
            mesh.vertices, mesh.faces, max_edge=max_edge, max_iter=8
        )
        triangles = vertices[faces]
    else:
        triangles = mesh.triangles
    points = triangles.mean(axis=1)
    weights = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    centers = points[_farthest_indices(points, cfg.count, cfg.seed)].copy()
    for _ in range(12):
        labels = cKDTree(centers).query(points)[1]
        updated = centers.copy()
        for i in range(len(centers)):
            mask = labels == i
            if mask.any():
                updated[i] = np.average(points[mask], axis=0, weights=weights[mask])
        if np.allclose(updated, centers, rtol=0, atol=max_edge * 1e-6):
            break
        centers = updated
    labels = cKDTree(centers).query(points)[1]
    output = []
    for i, center in enumerate(centers):
        assigned = triangles[labels == i]
        if not len(assigned):
            continue
        radius = np.linalg.norm(assigned - center, axis=2).max()
        if cfg.method == "cover-tight":
            from .enclosing import tighten_ball

            center, radius, _ = tighten_ball(assigned.reshape(-1, 3), center)
        # Round outward so a zero-padding export still encloses boundary vertices.
        radius = np.nextafter(radius, np.inf) + cfg.padding
        output.extend(_spheres(center[None], np.array([radius])))
    return output


def _voxel(mesh: trimesh.Trimesh, cfg: FitConfig) -> tuple[list[Sphere], dict, np.ndarray]:
    pitch = cfg.voxel_size or float(mesh.extents.max()) / cfg.resolution
    shape_bound = [math.ceil(float(e) / pitch) + 7 for e in mesh.extents]
    cells = math.prod(shape_bound)
    if cells > cfg.max_grid_cells:
        raise ValueError(
            f"Voxel grid needs up to {cells:,} cells (limit {cfg.max_grid_cells:,}); "
            "increase --voxel-size or reduce --resolution"
        )
    grid = mesh.voxelized(pitch)
    if math.prod(grid.shape) > cfg.max_grid_cells:
        raise ValueError("Voxel grid exceeds max_grid_cells; increase voxel_size")
    grid = grid.fill()
    solid = grid.matrix
    indices = np.argwhere(solid)
    if not len(indices):
        raise ValueError("Voxelization produced no occupied cells; increase resolution")
    points = trimesh.transform_points(indices, grid.transform)
    # Padding is essential: EDT must see exterior cells even for a solid box.
    distance = ndimage.distance_transform_edt(np.pad(solid, 1))[1:-1, 1:-1, 1:-1]
    clearance = np.maximum(distance[tuple(indices.T)] - 0.5, 0.5) * pitch
    maxima = distance >= ndimage.maximum_filter(distance, size=3)
    peak_ids = np.flatnonzero(maxima[tuple(indices.T)])
    rng = np.random.default_rng(cfg.seed)
    budget = min(max(cfg.max_candidates, cfg.count), len(points))
    if len(peak_ids) > budget // 2:
        peak_ids = rng.choice(peak_ids, max(1, budget // 2), replace=False)
    remaining = np.setdiff1d(np.arange(len(points)), peak_ids)
    random_ids = rng.choice(remaining, min(budget - len(peak_ids), len(remaining)), replace=False)
    candidate_ids = np.unique(np.r_[peak_ids, random_ids, int(np.argmax(clearance))])
    centers = points[candidate_ids]
    radii = clearance[candidate_ids] + cfg.padding
    # Optimize sampled volume coverage. Final report evaluates all occupied cells.
    target = points
    if len(target) > cfg.samples:
        target = target[rng.choice(len(target), cfg.samples, replace=False)]
    if cfg.device == "cuda":
        from .tensor_backend import greedy_cover

        chosen = greedy_cover(target, centers, radii, count=cfg.count, device="cuda").tolist()
    else:
        tree = cKDTree(target)
        neighbors = [
            np.asarray(tree.query_ball_point(c, r), dtype=int) for c, r in zip(centers, radii)
        ]
        covered = np.zeros(len(target), dtype=bool)
        heap = [(-len(ids), i) for i, ids in enumerate(neighbors)]
        heapq.heapify(heap)
        chosen = []
        while heap and len(chosen) < cfg.count:
            previous, i = heapq.heappop(heap)
            score = int(np.count_nonzero(~covered[neighbors[i]]))
            if score == 0:
                continue
            if heap and (-score, i) > heap[0]:
                heapq.heappush(heap, (-score, i))
                continue
            chosen.append(i)
            covered[neighbors[i]] = True
    if not chosen:
        raise ValueError("No voxel sphere candidates cover the sampled volume")
    spheres = _spheres(centers[chosen], radii[chosen])
    details = {
        "voxel_size_m": pitch,
        "voxel_count": len(points),
        "candidate_count": len(candidate_ids),
        "greedy_device": cfg.device,
        "radius_model": "voxel EDT minus half a voxel; approximate, not exact mesh SDF",
    }
    return spheres, details, points


def _lula(mesh: trimesh.Trimesh, cfg: FitConfig) -> list[Sphere]:
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
        [np.asarray(v, dtype=np.float64).reshape(3, 1) for v in mesh.vertices],
        [np.asarray(f, dtype=np.int32).reshape(3, 1) for f in mesh.faces],
    )
    generated = generator.generate_spheres(cfg.count, cfg.padding)
    if not generated:
        raise ValueError("Lula returned no spheres; check mesh topology or use --method cover")
    return [
        Sphere(tuple(np.asarray(s.center).reshape(3).tolist()), float(s.radius)) for s in generated
    ]


def fit_mesh(
    mesh: trimesh.Trimesh, config: FitConfig | None = None
) -> tuple[list[Sphere], dict[str, Any]]:
    """Fit a triangle mesh in its existing frame; return spheres and quality report."""
    cfg = config or FitConfig()
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError("Expected a nonempty triangle mesh")
    if not np.all(np.isfinite(mesh.vertices)) or not np.isfinite(mesh.area) or mesh.area <= 0:
        raise ValueError("Mesh must have finite vertices and positive surface area")
    mesh = mesh.copy()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if not len(mesh.faces) or mesh.area <= 0:
        raise ValueError("Mesh has no nondegenerate triangles")
    warnings = []
    effective = cfg
    if cfg.method in {"voxel", "refined", "lula"} and not mesh.is_watertight:
        if cfg.non_watertight == "error":
            raise ValueError(
                "Volume fitting requires a watertight mesh. Repair it, use --method surface/cover, "
                "or explicitly pass --non-watertight surface/cover."
            )
        effective = replace(cfg, method=cfg.non_watertight, device="cpu")
        warnings.append(f"Non-watertight mesh: explicitly selected fallback {effective.method}")
    details: dict[str, Any] = {}
    voxels = None
    if effective.method in {"voxel", "refined"}:
        # Refinement trains on unpadded balls; its padding is added exactly once
        # after optimization. Keep the original greedy baseline for voxel fits.
        initial_cfg = replace(effective, padding=0) if effective.method == "refined" else effective
        spheres, details, voxels = _voxel(mesh, initial_cfg)
        if effective.method == "refined":
            from .refine import refine_spheres

            spheres, refinement = refine_spheres(
                mesh, spheres, voxels, details["voxel_size_m"], effective
            )
            details.update(refinement)
            warnings.append(
                "Sampled refinement trades coverage against exterior volume; no certificate"
            )
        warnings.append("Voxel spheres approximate volume and can leave gaps; inspect coverage")
    elif effective.method == "surface":
        spheres = _surface(mesh, effective)
        warnings.append("Fixed-radius surface spheres do not guarantee solid-volume coverage")
    elif effective.method in {"cover", "cover-tight"}:
        spheres = _cover(mesh, effective)
        details["triangle_surface_coverage_certified"] = True
        warnings.append("Triangle boundary is enclosed; solid interior coverage is not certified")
    elif effective.method == "slab":
        from .slab import slab_spheres

        spheres, details = slab_spheres(mesh, effective)
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
        spheres = _lula(mesh, effective)
        warnings.append("Lula backend quality depends on its installed version and mesh topology")
    samples = sample_surface(mesh, cfg.samples, cfg.seed + 1)

    def evaluate_gaps(points):
        if effective.device == "cuda":
            from .tensor_backend import point_sphere_gaps

            return point_sphere_gaps(
                points,
                np.array([s.center for s in spheres]),
                np.array([s.radius for s in spheres]),
                device="cuda",
            )
        return sphere_gaps(points, spheres)

    gaps = evaluate_gaps(samples)
    positive_gaps = np.maximum(gaps, 0)
    report = {
        "requested_method": cfg.method,
        "effective_method": effective.method,
        "requested_count": cfg.count,
        "sphere_count": len(spheres),
        "watertight": bool(mesh.is_watertight),
        "triangle_count": len(mesh.faces),
        "surface_sample_count": len(samples),
        "surface_coverage": float(np.mean(gaps <= mesh.extents.max() * 1e-9)),
        "surface_gap_mean_m": float(positive_gaps.mean()),
        "surface_gap_p95_m": float(np.quantile(positive_gaps, 0.95)),
        "surface_gap_max_m": float(positive_gaps.max()),
        "sphere_volume_sum_m3": float(sum(4 * np.pi * s.radius**3 / 3 for s in spheres)),
        "warnings": warnings,
        **details,
    }
    if voxels is not None:
        report["voxel_center_coverage"] = float(np.mean(evaluate_gaps(voxels) <= 0))
    return spheres, report


def fit_asset(
    asset: Asset,
    config: FitConfig | None = None,
    *,
    link_counts: dict[str, int] | None = None,
    workers: int = 1,
    executor: str = "thread",
) -> FitResult:
    cfg = config or FitConfig()
    overrides = link_counts or {}
    unknown = set(overrides) - asset.links.keys()
    if unknown:
        raise ValueError(f"Sphere count overrides refer to unloaded links: {sorted(unknown)}")
    if cfg.device == "cuda" and isinstance(workers, int) and workers > 1 and len(asset.links) > 1:
        raise ValueError("CUDA fitting uses one GPU owner; use workers=1")
    from .parallel import fit_links

    tasks = []
    for name, link in asset.links.items():
        local_config = replace(cfg, count=overrides.get(name, cfg.count))
        tasks.append((name, link.mesh, local_config))
    links, reports = {}, {}
    for name, spheres, report in fit_links(tasks, workers=workers, executor=executor):
        links[name], reports[name] = spheres, report
        report["geometry_source"] = asset.links[name].geometry_source
        types = asset.links[name].geometry_types
        if types:
            report["input_geometry_types"] = list(types)
        if {"sphere", "cylinder"} & set(types):
            report["warnings"].append(
                "URDF analytic shapes are tessellated; guarantees concern the triangle mesh"
            )
    return FitResult(
        str(asset.path),
        cfg,
        links,
        reports,
        {name: transform.tolist() for name, transform in asset.zero_pose_transforms.items()},
        {
            "geometry": asset.geometry,
            "scale": asset.mesh_scale,
            "package_roots": {name: str(path) for name, path in asset.package_roots.items()},
        },
    )
