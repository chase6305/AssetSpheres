"""PCA-axis slab clipping followed by enclosing balls for triangle surfaces."""

from __future__ import annotations

from typing import Any

import numpy as np
import trimesh

from .enclosing import tighten_ball
from .model import FitConfig, Sphere


def slab_spheres(mesh: trimesh.Trimesh, config: FitConfig) -> tuple[list[Sphere], dict[str, Any]]:
    """Enclose each nonempty clipped surface slab in one tightened ball.

    A triangle clipped between two parallel planes has vertices consisting of
    retained original vertices and edge-plane intersections. All of these points
    are passed to a full-point-set enclosing-ball check. Adjacent slabs share the
    same clipping planes. A scale/coordinate-ULP margin protects the numerical
    reconstruction of intersection coordinates; this is not an exact-arithmetic
    certificate or a claim about arbitrary solid-volume representations.

    Disconnected branches within a slice share one ball and may bridge empty
    space. One global PCA axis is most appropriate for elongated link geometry.
    Input vertices, faces, and metadata are left unchanged.
    """
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError("Expected a nonempty triangle mesh")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    if not len(vertices) or not np.all(np.isfinite(vertices)):
        raise ValueError("Mesh vertices must be finite and nonempty")

    pivot = vertices[0].astype(np.longdouble)
    relative = vertices.astype(np.longdouble) - pivot
    scale = np.max(np.abs(relative))
    if scale == 0 or scale > np.finfo(float).max:
        raise ValueError("Mesh must have a positive finite representable extent")
    normalized = np.asarray(relative / scale, dtype=np.float64)
    triangles = normalized[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    if not np.any(np.any(cross != 0, axis=1)):
        raise ValueError("Mesh must contain a nondegenerate triangle")
    # Release preprocessing buffers before PCA and clipping allocations.
    del relative, triangles, cross

    centered = normalized - normalized.mean(axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    projected_axes = normalized @ axes.T
    ranges = np.ptp(projected_axes, axis=0)
    axis = axes[int(np.argmax(ranges))].copy()
    # Remove the arbitrary SVD sign so equivalent runs label their slabs alike.
    if axis[int(np.argmax(np.abs(axis)))] < 0:
        axis *= -1
    projection = normalized @ axis
    triangle_projection = projection[faces]
    low, high = float(projection.min()), float(projection.max())
    if not high > low:
        raise ValueError("Mesh has no positive span along the PCA axes")
    boundaries = np.linspace(low, high, config.count + 1)
    epsilon = np.finfo(float).eps
    tolerance = 64 * epsilon * max(1.0, abs(low), abs(high))
    max_coordinate = float(np.max(np.abs(vertices)))
    coordinate_ulp = abs(np.spacing(max_coordinate))
    margin = float(256 * epsilon * scale + 4 * np.sqrt(3.0) * coordinate_ulp)
    if not np.isfinite(margin):
        raise ValueError("Mesh coordinates do not permit a finite roundoff margin")

    minimum = triangle_projection.min(axis=1)
    maximum = triangle_projection.max(axis=1)
    physical_triangles = vertices[faces]
    edges = np.array([[0, 1], [1, 2], [2, 0]])
    spheres = []
    point_counts = []
    occupied_intervals = []
    total_iterations = 0
    for slab_low, slab_high in zip(boundaries[:-1], boundaries[1:]):
        selected = (maximum >= slab_low - tolerance) & (minimum <= slab_high + tolerance)
        if not np.any(selected):
            continue
        local_projection = triangle_projection[selected]
        local_triangles = physical_triangles[selected]
        inside = (local_projection >= slab_low - tolerance) & (
            local_projection <= slab_high + tolerance
        )
        pieces = [local_triangles[inside]]
        edge_projection = local_projection[:, edges]
        edge_points = local_triangles[:, edges]
        first_projection = edge_projection[:, :, 0]
        last_projection = edge_projection[:, :, 1]
        denominator = last_projection - first_projection
        nonparallel = denominator != 0
        for boundary in (slab_low, slab_high):
            crosses = nonparallel & (
                (
                    (first_projection <= boundary + tolerance)
                    & (last_projection >= boundary - tolerance)
                )
                | (
                    (last_projection <= boundary + tolerance)
                    & (first_projection >= boundary - tolerance)
                )
            )
            if not np.any(crosses):
                continue
            fraction = np.clip(
                (boundary - first_projection[crosses]) / denominator[crosses], 0.0, 1.0
            ).astype(np.longdouble)
            first = edge_points[:, :, 0, :][crosses].astype(np.longdouble)
            last = edge_points[:, :, 1, :][crosses].astype(np.longdouble)
            pieces.append(np.asarray(first + fraction[:, None] * (last - first), dtype=np.float64))
        points = np.unique(np.concatenate(pieces, axis=0), axis=0)
        if not len(points):
            # With positive-width slabs a real intersection has a retained vertex
            # or an edge crossing. Treat a numerical violation as an error, not
            # permission to omit an intersecting piece of the surface.
            raise ValueError("A selected slab produced no clipped polygon vertices")
        initial = np.asarray(points.astype(np.longdouble).mean(axis=0), dtype=np.float64)
        center, radius, tightening = tighten_ball(points, initial)
        radius = float(np.nextafter(radius + config.padding + margin, np.inf))
        spheres.append(Sphere(tuple(float(x) for x in center), radius))
        point_counts.append(len(points))
        occupied_intervals.append([float(slab_low * scale), float(slab_high * scale)])
        total_iterations += tightening["optimizer_iterations"]
    if not spheres:
        raise ValueError("Slab clipping produced no enclosing spheres")
    details = {
        "slab_axis": axis.tolist(),
        "axis_selection": "PCA axis with largest projected vertex span",
        "projection_origin_m": vertices[0].tolist(),
        "slab_count_requested": config.count,
        "slab_count_nonempty": len(spheres),
        "slab_point_counts": point_counts,
        "slab_intervals_relative_m": occupied_intervals,
        "radius_roundoff_margin_m": margin,
        "tightening_iterations_total": total_iterations,
        "surface_certification": (
            "Each clipped triangle polygon is enclosed by a ball checked against all its "
            "collected vertices, with a floating-point roundoff margin."
        ),
        "coverage_scope": "clipped triangle surfaces with floating-point clipping tolerance",
        "geometric_assumptions": (
            "One PCA axis for the whole mesh; all branches of a slice share a ball, "
            "which may bridge disconnected geometry or a hole. No watertightness is required."
        ),
    }
    return spheres, details
