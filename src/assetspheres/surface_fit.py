"""Sampled fixed-radius spheres and complete triangle-surface enclosures."""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .geometry import farthest_point_indices, make_spheres, sample_surface
from .model import FitConfig, Sphere


def _cluster_indices(labels: np.ndarray, cluster_count: int) -> list[np.ndarray]:
    # Stable grouping preserves each cluster's original summation order.
    ordered_indices = np.argsort(labels, kind="stable")
    boundaries = np.cumsum(np.bincount(labels, minlength=cluster_count))[:-1]
    return np.split(ordered_indices, boundaries)


def fit_surface_spheres(mesh: trimesh.Trimesh, configuration: FitConfig) -> list[Sphere]:
    candidate_points = sample_surface(
        mesh, max(configuration.samples, configuration.count * 4), configuration.seed
    )
    selected_points = candidate_points[
        farthest_point_indices(candidate_points, configuration.count, configuration.seed)
    ]
    radius = configuration.surface_radius + configuration.padding
    return make_spheres(selected_points, np.full(len(selected_points), radius))


def _partition_cover_triangles(
    mesh: trimesh.Trimesh, configuration: FitConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Smaller triangles improve coarse caps without changing the mesh boundary.
    target_edge_length = float(mesh.extents.max()) / 12
    if len(mesh.faces) < 15_000:
        coordinate_ulp = abs(np.spacing(float(np.max(np.abs(mesh.vertices)))))
        if target_edge_length <= 4 * coordinate_ulp:
            raise ValueError(
                "Triangle subdivision is below reliable world-coordinate precision; "
                "recenter the mesh near the origin or use --method slab"
            )
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
        farthest_point_indices(triangle_centers, configuration.count, configuration.seed)
    ].copy()
    for _ in range(12):
        cluster_labels = cKDTree(cluster_centers).query(triangle_centers)[1]
        updated_centers = cluster_centers.copy()
        for cluster_index, indices in enumerate(
            _cluster_indices(cluster_labels, len(cluster_centers))
        ):
            if len(indices):
                updated_centers[cluster_index] = np.average(
                    triangle_centers[indices],
                    axis=0,
                    weights=triangle_area_weights[indices],
                )
        if np.allclose(updated_centers, cluster_centers, rtol=0, atol=target_edge_length * 1e-6):
            break
        cluster_centers = updated_centers
    cluster_labels = cKDTree(cluster_centers).query(triangle_centers)[1]
    return triangles, cluster_centers, cluster_labels


def fit_triangle_cover(mesh: trimesh.Trimesh, configuration: FitConfig) -> list[Sphere]:
    """Enclose whole triangle partitions, preserving their boundary coverage.

    A ball is convex: containing each triangle's vertices contains its interior.
    This argument concerns the boundary, not the solid enclosed by the mesh.
    """
    triangles, cluster_centers, cluster_labels = _partition_cover_triangles(mesh, configuration)
    spheres = []
    for center, indices in zip(
        cluster_centers, _cluster_indices(cluster_labels, len(cluster_centers))
    ):
        if not len(indices):
            continue
        assigned_triangles = triangles[indices]
        if configuration.method == "cover-tight":
            from .enclosing import tighten_ball

            center, radius, _ = tighten_ball(assigned_triangles.reshape(-1, 3), center)
        else:
            radius = np.linalg.norm(assigned_triangles - center, axis=2).max()
        radius = np.nextafter(radius, np.inf) + configuration.padding
        spheres.append(Sphere(tuple(center), float(radius)))
    return spheres
