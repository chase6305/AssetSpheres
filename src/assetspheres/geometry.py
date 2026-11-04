"""Shared sampling and ball-union distances in the input coordinate frame.

These helpers do not select a fitting method or import optional runtimes.
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial.distance import cdist

from .model import Sphere


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
            # cdist computes distances directly in C without an (N, M, 3)
            # difference array or its equally large squared temporary.
            block_gaps = cdist(
                points[first_point:last_point], sphere_centers[first_sphere : first_sphere + 128]
            )
            block_gaps -= sphere_radii[None, first_sphere : first_sphere + 128]
            minimum_gaps[first_point:last_point] = np.minimum(
                minimum_gaps[first_point:last_point], block_gaps.min(axis=1)
            )
    return minimum_gaps


def make_spheres(centers: np.ndarray, radii: np.ndarray) -> list[Sphere]:
    return [
        Sphere(tuple(float(coordinate) for coordinate in center), float(radius))
        for center, radius in zip(centers, radii)
    ]


def farthest_point_indices(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    """Select distinct indices, breaking distance ties by input order."""
    sample_count = min(count, len(points))
    selected_indices = np.empty(sample_count, dtype=int)
    nearest_squared_distance = np.full(len(points), np.inf)
    squared_distances = np.empty((len(points), 1))
    current_index = int(np.random.default_rng(seed).integers(len(points)))
    for selection_index in range(sample_count):
        selected_indices[selection_index] = current_index
        # Compute one distance column directly, avoiding two (N, 3) temporaries.
        cdist(
            points,
            points[current_index : current_index + 1],
            metric="sqeuclidean",
            out=squared_distances,
        )
        np.minimum(nearest_squared_distance, squared_distances[:, 0], out=nearest_squared_distance)
        nearest_squared_distance[selected_indices[: selection_index + 1]] = -1
        current_index = int(np.argmax(nearest_squared_distance))
    return selected_indices
