"""Experimental fixed-budget ball optimization on CPU or CUDA.

This is an independently implemented sampled classification objective, inspired by
gradient-based sphere fitting. It is not the MorphIt reference implementation and
does not provide a continuous-volume or triangle-boundary coverage certificate.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import trimesh
from scipy.spatial.distance import cdist

from .geometry import sample_surface
from .grid_distance import unit_grid_distances
from .model import FitConfig, Sphere

SURFACE_COVERAGE_WEIGHT = 8.0
INTERIOR_COVERAGE_WEIGHT = 1.0
RADIUS_REGULARIZATION_WEIGHT = 0.002
MINIMUM_CENTER_DISTANCE = 1e-12
NORMALIZED_LEARNING_RATE = 0.006
MAXIMUM_TRAINING_PAIRS = 4_000_000
MAXIMUM_GRADIENT_BLOCK_PAIRS = 1_000_000
VOXEL_POINT_BATCH_SIZE = 32768


def _sample_mesh_reference_regions(
    mesh: trimesh.Trimesh,
    interior_candidates: np.ndarray,
    surface_samples: np.ndarray,
    voxel_size_m: float,
    sample_count: int,
    random_generator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if mesh.body_count != 1:
        raise ValueError(
            "Mesh-reference refinement requires one closed surface body; "
            "multiple shells need an explicit solid interpretation. Use voxel reference."
        )
    near_surface_proposals = [
        surface_samples
        + random_generator.normal(size=surface_samples.shape) * voxel_size_m * distance_factor
        for distance_factor in (0.15, 0.6, 1.8)
    ]
    bounding_box_proposals = random_generator.uniform(
        mesh.bounds[0] - voxel_size_m * 2,
        mesh.bounds[1] + voxel_size_m * 2,
        size=(sample_count, 3),
    )
    region_proposals = np.concatenate(near_surface_proposals + [bounding_box_proposals])
    query_points = np.concatenate((interior_candidates, region_proposals))
    try:
        inside_mask = np.concatenate(
            [
                mesh.contains(query_points[first_point : first_point + 2048])
                for first_point in range(0, len(query_points), 2048)
            ]
        )
    except ImportError as exc:
        raise ValueError("Mesh-reference refinement requires the optional rtree package") from exc
    interior_points, exterior_points = query_points[inside_mask], query_points[~inside_mask]
    if not len(interior_points) or not len(exterior_points):
        raise ValueError("Insufficient mesh-reference samples; increase resolution/samples")
    selected_interior = interior_points[
        random_generator.choice(
            len(interior_points), min(sample_count, len(interior_points)), replace=False
        )
    ]
    selected_exterior = exterior_points[
        random_generator.choice(
            len(exterior_points), min(sample_count, len(exterior_points)), replace=False
        )
    ]
    return selected_interior, selected_exterior


def _sample_voxel_exterior(
    occupied_points: np.ndarray,
    voxel_size_m: float,
    sample_count: int,
    random_generator: np.random.Generator,
) -> np.ndarray:
    # Keep negative labels away from the ambiguous rasterized boundary.
    lattice_origin = occupied_points.min(axis=0)
    maximum_coordinates = occupied_points.max(axis=0)
    maximum_indices = np.rint((maximum_coordinates - lattice_origin) / voxel_size_m).astype(int)
    border_cells = 4
    grid_shape = tuple(maximum_indices + 1 + 2 * border_cells)
    occupied_cells = np.zeros(grid_shape, dtype=bool)
    # Avoid retaining a full point-index array during the distance transform.
    for first_point in range(0, len(occupied_points), VOXEL_POINT_BATCH_SIZE):
        point_block = occupied_points[first_point : first_point + VOXEL_POINT_BATCH_SIZE]
        block_indices = np.rint((point_block - lattice_origin) / voxel_size_m).astype(int)
        occupied_cells[tuple((block_indices + border_cells).T)] = True
    del block_indices
    exterior_distance = unit_grid_distances(~occupied_cells)
    exterior_indices = np.argwhere((exterior_distance >= 1.5) & (exterior_distance <= border_cells))
    if not len(exterior_indices):
        raise ValueError("No exterior training samples; increase voxel resolution")
    selected_indices = random_generator.choice(
        len(exterior_indices), min(sample_count, len(exterior_indices)), replace=False
    )
    return lattice_origin + (exterior_indices[selected_indices] - border_cells) * voxel_size_m


def _pack_training_samples(
    surface_samples: np.ndarray,
    interior_samples: np.ndarray,
    exterior_samples: np.ndarray,
    configuration: FitConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    training_points = np.concatenate((surface_samples, interior_samples, exterior_samples))
    positive_mask = np.arange(len(training_points)) < len(surface_samples) + len(interior_samples)
    sample_weights = np.r_[
        np.full(len(surface_samples), SURFACE_COVERAGE_WEIGHT / len(surface_samples)),
        np.full(len(interior_samples), INTERIOR_COVERAGE_WEIGHT / len(interior_samples)),
        np.full(len(exterior_samples), configuration.outside_weight / len(exterior_samples)),
    ]
    sample_counts = {
        "surface": len(surface_samples),
        "interior": len(interior_samples),
        "exterior": len(exterior_samples),
    }
    return training_points, positive_mask, sample_weights, sample_counts


def _prepare_training_samples(
    mesh: trimesh.Trimesh,
    occupied_points: np.ndarray,
    voxel_size_m: float,
    configuration: FitConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    random_generator = np.random.default_rng(configuration.seed + 1009)
    sample_count = configuration.refine_samples
    interior_samples = occupied_points[
        random_generator.choice(
            len(occupied_points), min(sample_count, len(occupied_points)), replace=False
        )
    ]
    surface_samples = sample_surface(mesh, sample_count, configuration.seed + 2017)
    if configuration.refine_reference == "mesh":
        interior_samples, exterior_samples = _sample_mesh_reference_regions(
            mesh, interior_samples, surface_samples, voxel_size_m, sample_count, random_generator
        )
    else:
        exterior_samples = _sample_voxel_exterior(
            occupied_points, voxel_size_m, sample_count, random_generator
        )
    return _pack_training_samples(
        surface_samples, interior_samples, exterior_samples, configuration
    )


def _loss_gradient(
    sphere_parameters: np.ndarray,
    training_points: np.ndarray,
    positive_mask: np.ndarray,
    sample_weights: np.ndarray,
    coverage_margin: float,
) -> tuple[float, np.ndarray]:
    """Float64 training gradient; ties select the first ball deterministically."""
    sphere_centers, sphere_radii = sphere_parameters[:, :3], sphere_parameters[:, 3]
    if len(training_points) * len(sphere_parameters) <= MAXIMUM_GRADIENT_BLOCK_PAIRS:
        all_gaps = cdist(training_points, sphere_centers)
        np.maximum(all_gaps, MINIMUM_CENTER_DISTANCE, out=all_gaps)
        all_gaps -= sphere_radii[None]
        nearest_sphere_indices = np.argmin(all_gaps, axis=1)
        point_indices = np.arange(len(training_points))
        nearest_gaps = all_gaps[point_indices, nearest_sphere_indices]
        # Release the dense workspace before constructing gradient directions.
        del all_gaps
    else:
        # Point rows are independent; retain the global loss/gradient accumulation below.
        nearest_sphere_indices = np.empty(len(training_points), dtype=np.intp)
        nearest_gaps = np.empty(len(training_points), dtype=np.float64)
        points_per_block = max(1, MAXIMUM_GRADIENT_BLOCK_PAIRS // len(sphere_parameters))
        for first_point in range(0, len(training_points), points_per_block):
            last_point = first_point + points_per_block
            all_gaps = cdist(training_points[first_point:last_point], sphere_centers)
            np.maximum(all_gaps, MINIMUM_CENTER_DISTANCE, out=all_gaps)
            all_gaps -= sphere_radii[None]
            block_nearest_indices = np.argmin(all_gaps, axis=1)
            nearest_sphere_indices[first_point:last_point] = block_nearest_indices
            nearest_gaps[first_point:last_point] = all_gaps[
                np.arange(len(block_nearest_indices)), block_nearest_indices
            ]
            del all_gaps
    coverage_errors = np.where(
        positive_mask,
        np.maximum(nearest_gaps + coverage_margin, 0),
        np.minimum(nearest_gaps, 0),
    )
    gap_derivatives = 2 * sample_weights * coverage_errors
    parameter_gradient = np.zeros_like(sphere_parameters)
    # Satisfied samples have zero derivative; only contributing points need directions.
    contributing_samples = gap_derivatives != 0
    nearest_sphere_indices = nearest_sphere_indices[contributing_samples]
    gap_derivatives = gap_derivatives[contributing_samples]
    center_offsets = sphere_centers[nearest_sphere_indices] - training_points[contributing_samples]
    center_distances = np.linalg.norm(center_offsets, axis=1)
    center_directions = (
        center_offsets / np.maximum(center_distances, MINIMUM_CENTER_DISTANCE)[:, None]
    )
    # Match the derivative of torch.clamp_min, including its exact boundary.
    center_directions[center_distances < MINIMUM_CENTER_DISTANCE] = 0
    # Group sample contributions by their active sphere.
    for axis in range(3):
        parameter_gradient[:, axis] = np.bincount(
            nearest_sphere_indices,
            weights=gap_derivatives * center_directions[:, axis],
            minlength=len(sphere_parameters),
        )
    parameter_gradient[:, 3] = np.bincount(
        nearest_sphere_indices, weights=-gap_derivatives, minlength=len(sphere_parameters)
    )
    loss = float(
        np.sum(sample_weights * coverage_errors**2)
        + RADIUS_REGULARIZATION_WEIGHT * np.mean(sphere_radii**3)
    )
    parameter_gradient[:, 3] += (
        3 * RADIUS_REGULARIZATION_WEIGHT * sphere_radii**2 / len(sphere_radii)
    )
    return loss, parameter_gradient


def _numpy_adam(
    sphere_parameters: np.ndarray,
    training_points: np.ndarray,
    positive_mask: np.ndarray,
    sample_weights: np.ndarray,
    coverage_margin: float,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    iterations: int,
) -> tuple[np.ndarray, float, float]:
    """Optimize the shared objective, retaining the lowest-loss training iterate."""
    first_moment = np.zeros_like(sphere_parameters)
    second_moment = np.zeros_like(sphere_parameters)
    best_parameters = sphere_parameters.copy()
    initial_loss, gradient = _loss_gradient(
        sphere_parameters, training_points, positive_mask, sample_weights, coverage_margin
    )
    best_loss = initial_loss
    for step in range(1, iterations + 1):
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * gradient**2
        parameter_update = (
            NORMALIZED_LEARNING_RATE
            * (first_moment / (1 - 0.9**step))
            / (np.sqrt(second_moment / (1 - 0.999**step)) + 1e-8)
        )
        sphere_parameters = np.clip(
            sphere_parameters - parameter_update, lower_bounds, upper_bounds
        )
        current_loss, gradient = _loss_gradient(
            sphere_parameters, training_points, positive_mask, sample_weights, coverage_margin
        )
        if current_loss < best_loss:
            best_loss, best_parameters = current_loss, sphere_parameters.copy()
    return best_parameters, initial_loss, best_loss


def _torch_loss(
    torch, sphere_parameters, training_points, positive_mask, sample_weights, coverage_margin
):
    center_offsets = sphere_parameters[None, :, :3] - training_points[:, None, :]
    center_distances = torch.linalg.vector_norm(center_offsets, dim=2).clamp_min(
        MINIMUM_CENTER_DISTANCE
    )
    nearest_gaps = (center_distances - sphere_parameters[None, :, 3]).min(dim=1).values
    coverage_errors = torch.where(
        positive_mask, (nearest_gaps + coverage_margin).clamp_min(0), nearest_gaps.clamp_max(0)
    )
    return (
        sample_weights * coverage_errors.square()
    ).sum() + RADIUS_REGULARIZATION_WEIGHT * sphere_parameters[:, 3].pow(3).mean()


def _cuda_adam(
    initial_parameters: np.ndarray,
    training_points: np.ndarray,
    positive_mask: np.ndarray,
    sample_weights: np.ndarray,
    coverage_margin: float,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    iterations: int,
) -> tuple[np.ndarray, float, float]:
    try:
        import torch
    except ImportError as exc:
        raise ValueError("CUDA refinement requires the optional torch dependency") from exc
    if not torch.cuda.is_available():
        raise ValueError("CUDA refinement requested, but torch.cuda.is_available() is false")

    def as_cuda_tensor(values):
        return torch.as_tensor(values, device="cuda", dtype=torch.float64)

    sphere_parameters = as_cuda_tensor(initial_parameters).requires_grad_(True)
    training_points, sample_weights = (
        as_cuda_tensor(training_points),
        as_cuda_tensor(sample_weights),
    )
    positive_mask = torch.as_tensor(positive_mask, device="cuda", dtype=torch.bool)
    lower_bounds, upper_bounds = as_cuda_tensor(lower_bounds), as_cuda_tensor(upper_bounds)
    optimizer = torch.optim.Adam([sphere_parameters], lr=NORMALIZED_LEARNING_RATE)

    def objective():
        return _torch_loss(
            torch,
            sphere_parameters,
            training_points,
            positive_mask,
            sample_weights,
            coverage_margin,
        )

    initial_loss = objective().detach()
    best_loss = initial_loss.clone()
    best_parameters = sphere_parameters.detach().clone()
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        current_loss = objective()
        # Device-side selection avoids a CPU synchronization at every iteration.
        improved = current_loss.detach() < best_loss
        best_parameters = torch.where(improved, sphere_parameters.detach(), best_parameters)
        best_loss = torch.minimum(best_loss, current_loss.detach())
        current_loss.backward()
        optimizer.step()
        with torch.no_grad():
            sphere_parameters.clamp_(min=lower_bounds, max=upper_bounds)
    final_loss = objective().detach()
    best_parameters = torch.where(
        final_loss < best_loss, sphere_parameters.detach(), best_parameters
    )
    best_loss = torch.minimum(best_loss, final_loss)
    torch.cuda.synchronize()
    return best_parameters.cpu().numpy(), float(initial_loss.cpu()), float(best_loss.cpu())


def refine_spheres(
    mesh: trimesh.Trimesh,
    initial: list[Sphere],
    occupied: np.ndarray,
    pitch: float,
    cfg: FitConfig,
) -> tuple[list[Sphere], dict[str, Any]]:
    """Refine existing link-local spheres; add user padding only after optimization."""
    # Preserve the existing keyword interface at this public module boundary.
    configuration, voxel_size_m = cfg, pitch
    if not initial:
        raise ValueError("Refinement requires at least one initial sphere")
    # Reject clearly impossible sample requests before allocating point arrays
    # or preparing labels. The final check uses the actual packed sample count.
    minimum_training_count = configuration.refine_samples + min(
        configuration.refine_samples, len(occupied)
    )
    if minimum_training_count * len(initial) > MAXIMUM_TRAINING_PAIRS:
        raise ValueError(
            "Refinement exceeds the 4M point-ball pair budget; reduce refine_samples/count"
        )
    training_points, positive_mask, sample_weights, training_counts = _prepare_training_samples(
        mesh, occupied, voxel_size_m, configuration
    )
    if len(training_points) * len(initial) > MAXIMUM_TRAINING_PAIRS:
        raise ValueError(
            "Refinement exceeds the 4M point-ball pair budget; reduce refine_samples/count"
        )
    normalization_origin = mesh.bounds.mean(axis=0)
    normalization_scale = float(mesh.extents.max())
    training_points = (training_points - normalization_origin) / normalization_scale
    sphere_parameters = np.column_stack(
        (
            (np.array([sphere.center for sphere in initial]) - normalization_origin)
            / normalization_scale,
            np.array([sphere.radius for sphere in initial]) / normalization_scale,
        )
    )
    lower_center_bounds, upper_center_bounds = (
        mesh.bounds - normalization_origin
    ) / normalization_scale
    lower_bounds = np.r_[lower_center_bounds, voxel_size_m * 0.25 / normalization_scale]
    upper_bounds = np.r_[upper_center_bounds, np.linalg.norm(mesh.extents) / normalization_scale]
    coverage_margin = voxel_size_m * 0.15 / normalization_scale
    optimizer = _numpy_adam if configuration.device == "cpu" else _cuda_adam
    optimized_parameters, initial_loss, final_loss = optimizer(
        sphere_parameters,
        training_points,
        positive_mask,
        sample_weights,
        coverage_margin,
        lower_bounds,
        upper_bounds,
        configuration.iterations,
    )
    refined_spheres = [
        Sphere(
            tuple(center * normalization_scale + normalization_origin),
            float(radius * normalization_scale + configuration.padding),
        )
        for center, radius in zip(optimized_parameters[:, :3], optimized_parameters[:, 3])
    ]
    return refined_spheres, {
        "optimizer": "Adam, sampled coverage/exterior classification; independent implementation",
        "optimization_device": configuration.device,
        "optimization_iterations": configuration.iterations,
        "optimization_training_counts": training_counts,
        "optimization_initial_loss": initial_loss,
        "optimization_final_loss": final_loss,
        "optimization_outside_weight": configuration.outside_weight,
        "optimization_reference": (
            "original mesh.contains labels; one closed surface, assuming no self-intersection"
            if configuration.refine_reference == "mesh"
            else "filled voxel grid, not exact mesh SDF"
        ),
        "optimization_positive_margin_m": voxel_size_m * 0.15,
    }
