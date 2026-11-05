"""Sampled fixed-radius spheres and complete triangle-surface enclosures."""

from __future__ import annotations

import math

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

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


def _enclose_triangle_groups(
    triangles: np.ndarray,
    centers: np.ndarray,
    labels: np.ndarray,
    *,
    tighten: bool,
    padding: float,
) -> list[Sphere]:
    spheres = []
    for center, indices in zip(centers, _cluster_indices(labels, len(centers))):
        if not len(indices):
            continue
        assigned_triangles = triangles[indices]
        if tighten:
            from .enclosing import tighten_ball

            center, radius, _ = tighten_ball(assigned_triangles.reshape(-1, 3), center)
        else:
            radius = np.linalg.norm(assigned_triangles - center, axis=2).max()
        radius = np.nextafter(radius, np.inf) + padding
        spheres.append(Sphere(tuple(center), float(radius)))
    return spheres


def _nearest_triangle_centers(triangles: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Assign each whole triangle by its farthest vertex, with bounded workspace."""
    labels = np.zeros(len(triangles), dtype=np.intp)
    for first_triangle in range(0, len(triangles), 2048):
        triangle_block = triangles[first_triangle : first_triangle + 2048]
        best_distances = np.full(len(triangle_block), np.inf)
        block_labels = labels[first_triangle : first_triangle + len(triangle_block)]
        for first_center in range(0, len(centers), 128):
            distances = cdist(
                triangle_block.reshape(-1, 3), centers[first_center : first_center + 128]
            ).reshape(len(triangle_block), 3, -1)
            triangle_distances = distances.max(axis=1)
            nearest = triangle_distances.argmin(axis=1)
            candidate_distances = triangle_distances[np.arange(len(triangle_block)), nearest]
            # Strict improvement keeps the first center when costs tie across blocks.
            improved = candidate_distances < best_distances
            best_distances[improved] = candidate_distances[improved]
            block_labels[improved] = first_center + nearest[improved]
    return labels


def fit_triangle_cover(mesh: trimesh.Trimesh, configuration: FitConfig) -> list[Sphere]:
    """Enclose whole triangles; optionally improve their grouping twice.

    A ball is convex: containing each triangle's vertices contains its interior.
    This argument concerns the boundary, not the solid enclosed by the mesh.
    cover-opt accepts regrouping only when the padded sphere-volume sum falls;
    this proxy counts overlaps and does not measure the union's exterior volume.
    """
    triangles, centers, labels = _partition_cover_triangles(mesh, configuration)
    spheres = _enclose_triangle_groups(
        triangles,
        centers,
        labels,
        tighten=configuration.method in {"cover-tight", "cover-opt"},
        padding=configuration.padding,
    )
    if configuration.method != "cover-opt":
        return spheres
    for _ in range(2):
        centers = np.array([sphere.center for sphere in spheres])
        labels = _nearest_triangle_centers(triangles, centers)
        candidate = _enclose_triangle_groups(
            triangles, centers, labels, tighten=True, padding=configuration.padding
        )
        # A shared scale keeps the cubic comparison finite for very large/small meshes.
        scale = max(sphere.radius for sphere in spheres + candidate)
        current_cost = math.fsum((sphere.radius / scale) ** 3 for sphere in spheres)
        candidate_cost = math.fsum((sphere.radius / scale) ** 3 for sphere in candidate)
        if candidate_cost >= current_cost:
            break
        spheres = candidate
    return spheres
