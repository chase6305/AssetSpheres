"""Construct a triangle-surface enclosure by enlarging an existing ball set."""

from __future__ import annotations

from typing import Any

import numpy as np
import trimesh
from scipy.spatial.distance import cdist

from .model import Sphere

TRIANGLES_PER_BLOCK = 2048
CENTERS_PER_BLOCK = 128
MAXIMUM_SUBDIVISION_DEPTH = 8
MAXIMUM_TRIANGLE_SPLITS = 100_000
TARGET_EDGE_DIVISOR = 96
VERTICES_PER_BLOCK = 32768


def _normalize_repair_geometry(
    vertices: np.ndarray, centers: np.ndarray, radii: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.longdouble]:
    """Translate before scaling, bounding the extended-precision vertex workspace."""
    origin = vertices[0].astype(np.longdouble)
    if len(vertices) <= VERTICES_PER_BLOCK:
        relative_vertices = vertices.astype(np.longdouble) - origin
    else:
        # Subtraction is monotone: coordinate bounds contain the same largest
        # absolute translated value as the full vertex array, including rounding.
        relative_vertices = (
            np.asarray([vertices.min(axis=0), vertices.max(axis=0)], dtype=np.longdouble) - origin
        )
    relative_centers = centers.astype(np.longdouble) - origin
    scale = max(
        np.max(np.abs(relative_vertices)),
        np.max(np.abs(relative_centers)),
        np.max(radii.astype(np.longdouble)),
    )
    if len(vertices) <= VERTICES_PER_BLOCK:
        normalized_vertices = np.asarray(relative_vertices / scale, dtype=np.float64)
    else:
        normalized_vertices = np.empty_like(vertices)
        relative = np.empty((VERTICES_PER_BLOCK, 3), dtype=np.longdouble)
        for first_vertex in range(0, len(vertices), VERTICES_PER_BLOCK):
            block = slice(first_vertex, first_vertex + VERTICES_PER_BLOCK)
            block_relative = relative[: len(vertices[block])]
            np.subtract(vertices[block], origin, out=block_relative, dtype=np.longdouble)
            block_relative /= scale
            normalized_vertices[block] = block_relative
    return (
        normalized_vertices,
        np.asarray(relative_centers / scale, dtype=np.float64),
        np.asarray(radii.astype(np.longdouble) / scale, dtype=np.float64),
        scale,
    )


def _triangle_requirements(
    triangles: np.ndarray, centers: np.ndarray, radii: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find each triangle's least required radius increase, keeping the first tie."""
    best_gaps = np.full(len(triangles), np.inf)
    best_center_indices = np.zeros(len(triangles), dtype=np.intp)
    required_radii = np.zeros(len(triangles))
    for first_center in range(0, len(centers), CENTERS_PER_BLOCK):
        center_block = centers[first_center : first_center + CENTERS_PER_BLOCK]
        # Reduce squared corner distances first: one square root per triangle/ball.
        maximum_distances = cdist(triangles[:, 0], center_block, metric="sqeuclidean")
        temporary = np.empty_like(maximum_distances)
        for corner in (1, 2):
            cdist(triangles[:, corner], center_block, metric="sqeuclidean", out=temporary)
            np.maximum(maximum_distances, temporary, out=maximum_distances)
        del temporary
        np.sqrt(maximum_distances, out=maximum_distances)
        gaps = maximum_distances - radii[first_center : first_center + len(center_block)]
        nearest = gaps.argmin(axis=1)
        candidate_gaps = gaps[np.arange(len(triangles)), nearest]
        improved = candidate_gaps < best_gaps
        best_gaps[improved] = candidate_gaps[improved]
        best_center_indices[improved] = first_center + nearest[improved]
        required_radii[improved] = maximum_distances[np.flatnonzero(improved), nearest[improved]]
        del maximum_distances, gaps
    return best_gaps, best_center_indices, required_radii


def repair_surface_cover(
    mesh: trimesh.Trimesh, spheres: list[Sphere]
) -> tuple[list[Sphere], dict[str, Any]]:
    """Enclose complete triangle pieces without moving centers or shrinking balls.

    A triangle is covered when one ball contains all three vertices. Ambiguous
    triangles are subdivided, and only unresolved leaves enlarge a nearby ball.
    Depth and split budgets bound the extra work: remaining coarse triangles are
    enclosed as whole pieces when either budget is exhausted, never omitted.

    Midpoint subdivision and radius checks use normalized float64 coordinates
    with a scale/coordinate-ULP margin; this is not interval arithmetic. The
    boundary enclosure does not certify solid interiors, but enlarging balls
    preserves every point already covered by the input set. Padding is already
    part of the input radii and is not added again. Inputs are left unchanged.
    """
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError("Surface repair requires a nonempty triangle mesh")
    if not spheres:
        raise ValueError("Surface repair requires at least one sphere")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if not np.all(np.isfinite(vertices)):
        raise ValueError("Surface repair requires finite mesh vertices")
    centers = np.array([sphere.center for sphere in spheres])
    original_radii = np.array([sphere.radius for sphere in spheres])

    # Translating in extended precision preserves small geometry at large offsets.
    normalized_vertices, normalized_centers, normalized_radii, scale = _normalize_repair_geometry(
        vertices, centers, original_radii
    )
    initial_normalized_radii = normalized_radii.copy()
    target_edge = float(np.ptp(normalized_vertices, axis=0).max()) / TARGET_EDGE_DIVISOR
    coordinate_ulp = abs(np.spacing(float(np.abs(vertices).max())))
    margin = float(256 * np.finfo(float).eps + 8 * np.sqrt(3.0) * coordinate_ulp / scale)
    if not np.isfinite(margin):
        raise ValueError("Mesh coordinates do not permit a finite surface-repair margin")

    split_count = 0
    visited_count = 0
    leaf_count = 0
    coarse_leaf_count = 0
    maximum_depth = 0
    for first_face in range(0, len(mesh.faces), TRIANGLES_PER_BLOCK):
        triangles = normalized_vertices[mesh.faces[first_face : first_face + TRIANGLES_PER_BLOCK]]
        pending = [(triangles, 0)]
        while pending:
            triangles, depth = pending.pop()
            visited_count += len(triangles)
            maximum_depth = max(maximum_depth, depth)
            gaps, nearest_center_indices, required_radii = _triangle_requirements(
                triangles, normalized_centers, normalized_radii
            )
            unresolved = gaps > -margin
            longest_edges = np.linalg.norm(triangles - np.roll(triangles, 1, axis=1), axis=2).max(
                axis=1
            )
            split = unresolved & (longest_edges > target_edge)
            if depth >= MAXIMUM_SUBDIVISION_DEPTH:
                split[:] = False
            available_splits = MAXIMUM_TRIANGLE_SPLITS - split_count
            split_indices = np.flatnonzero(split)
            split[split_indices[available_splits:]] = False
            split_count += int(np.count_nonzero(split))

            leaves = ~split
            repair = leaves & unresolved
            leaf_count += int(np.count_nonzero(leaves))
            coarse_leaf_count += int(np.count_nonzero(repair & (longest_edges > target_edge)))
            # One selected ball encloses each entire leaf, including its interior.
            np.maximum.at(
                normalized_radii,
                nearest_center_indices[repair],
                np.nextafter(required_radii[repair] + margin, np.inf),
            )
            if not np.any(split):
                continue
            first, second, third = triangles[split].transpose(1, 0, 2)
            first_second = (first + second) * 0.5
            second_third = (second + third) * 0.5
            first_third = (first + third) * 0.5
            children = np.concatenate(
                (
                    np.stack((first, first_second, first_third), axis=1),
                    np.stack((first_second, second, second_third), axis=1),
                    np.stack((first_third, second_third, third), axis=1),
                    np.stack((first_second, second_third, first_third), axis=1),
                )
            )
            # Reverse the push order so traversal and cross-block ties stay stable.
            for first_child in reversed(range(0, len(children), TRIANGLES_PER_BLOCK)):
                pending.append(
                    (children[first_child : first_child + TRIANGLES_PER_BLOCK], depth + 1)
                )

    repaired_radii = original_radii.copy()
    changed = normalized_radii > initial_normalized_radii
    physical_radii = normalized_radii[changed].astype(np.longdouble) * scale
    maximum_float = np.finfo(float).max
    if np.any(physical_radii > maximum_float):
        raise ValueError("Repaired enclosing radii cannot be represented as finite float64")
    rounded_radii = np.asarray(physical_radii, dtype=np.float64)
    np.nextafter(rounded_radii, np.inf, out=rounded_radii, where=rounded_radii < maximum_float)
    repaired_radii[changed] = np.maximum(
        original_radii[changed],
        rounded_radii,
    )
    if not np.all(np.isfinite(repaired_radii)):
        raise ValueError("Repaired enclosing radii cannot be represented as finite float64")
    repaired = [
        Sphere(sphere.center, float(radius)) for sphere, radius in zip(spheres, repaired_radii)
    ]
    return repaired, {
        "surface_repair_input_triangles": len(mesh.faces),
        "surface_repair_visited_triangles": visited_count,
        "surface_repair_triangle_splits": split_count,
        "surface_repair_leaf_triangles": leaf_count,
        "surface_repair_maximum_depth": maximum_depth,
        "surface_repair_coarse_leaves": coarse_leaf_count,
        "surface_repair_grown_spheres": int(np.count_nonzero(repaired_radii > original_radii)),
        "surface_repair_maximum_radius_increase_m": float(np.max(repaired_radii - original_radii)),
        "surface_repair_roundoff_margin_m": float(margin * scale),
        "surface_repair_certification": (
            "Every triangle leaf is enclosed by one ball, with a floating-point "
            "subdivision/distance margin; not interval arithmetic or solid-volume certification."
        ),
    }
