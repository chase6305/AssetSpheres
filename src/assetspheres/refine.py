"""Experimental fixed-budget ball optimization on CPU or CUDA.

This is an independently implemented sampled classification objective, inspired by
gradient-based sphere fitting. It is not the MorphIt reference implementation and
does not provide a continuous-volume or triangle-boundary coverage certificate.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import trimesh
from scipy import ndimage

from .model import FitConfig, Sphere


def _training_data(mesh, occupied, pitch, cfg):
    from .fit import sample_surface

    rng = np.random.default_rng(cfg.seed + 1009)
    count = cfg.refine_samples
    inside = occupied[rng.choice(len(occupied), min(count, len(occupied)), replace=False)]
    surface = sample_surface(mesh, count, cfg.seed + 2017)
    if cfg.refine_reference == "mesh":
        if mesh.body_count != 1:
            raise ValueError(
                "Mesh-reference refinement requires one closed surface body; "
                "multiple shells need an explicit solid interpretation. Use voxel reference."
            )
        proposals = np.concatenate(
            [
                surface + rng.normal(size=surface.shape) * pitch * factor
                for factor in (0.15, 0.6, 1.8)
            ]
            + [rng.uniform(mesh.bounds[0] - pitch * 2, mesh.bounds[1] + pitch * 2, size=(count, 3))]
        )
        probes = np.concatenate((inside, proposals))
        try:
            contains = np.concatenate(
                [
                    mesh.contains(probes[start : start + 2048])
                    for start in range(0, len(probes), 2048)
                ]
            )
        except ImportError as exc:
            raise ValueError(
                "Mesh-reference refinement requires the optional rtree package"
            ) from exc
        positives, negatives = probes[contains], probes[~contains]
        if not len(positives) or not len(negatives):
            raise ValueError("Insufficient mesh-reference samples; increase resolution/samples")
        inside = positives[rng.choice(len(positives), min(count, len(positives)), replace=False)]
        outside = negatives[rng.choice(len(negatives), min(count, len(negatives)), replace=False)]
        return _pack_training(surface, inside, outside, cfg)
    # Reconstruct a padded lattice from occupied centers. Keep negative samples
    # away from the ambiguous rasterized boundary by at least 1.5 voxel pitches.
    low = occupied.min(axis=0)
    indices = np.rint((occupied - low) / pitch).astype(int)
    border = 4
    shape = tuple(indices.max(axis=0) + 1 + 2 * border)
    solid = np.zeros(shape, dtype=bool)
    solid[tuple((indices + border).T)] = True
    distance = ndimage.distance_transform_edt(~solid)
    exterior_indices = np.argwhere((distance >= 1.5) & (distance <= border))
    if not len(exterior_indices):
        raise ValueError("No exterior training samples; increase voxel resolution")
    selected = rng.choice(len(exterior_indices), min(count, len(exterior_indices)), replace=False)
    outside = low + (exterior_indices[selected] - border) * pitch
    return _pack_training(surface, inside, outside, cfg)


def _pack_training(surface, inside, outside, cfg):
    points = np.concatenate((surface, inside, outside))
    positive = np.arange(len(points)) < len(surface) + len(inside)
    weights = np.r_[
        np.full(len(surface), 8.0 / len(surface)),
        np.full(len(inside), 1.0 / len(inside)),
        np.full(len(outside), cfg.outside_weight / len(outside)),
    ]
    return (
        points,
        positive,
        weights,
        {
            "surface": len(surface),
            "interior": len(inside),
            "exterior": len(outside),
        },
    )


def _loss_gradient(parameters, points, positive, weights, margin):
    """Piecewise analytic gradient; ties select the first ball deterministically."""
    centers, radii = parameters[:, :3], parameters[:, 3]
    delta = centers[None, :, :] - points[:, None, :]
    raw_distances = np.linalg.norm(delta, axis=2)
    distances = np.maximum(raw_distances, 1e-12)
    gaps = distances - radii[None]
    labels = np.argmin(gaps, axis=1)
    row = np.arange(len(points))
    gap = gaps[row, labels]
    error = np.where(positive, np.maximum(gap + margin, 0), np.minimum(gap, 0))
    coefficient = 2 * weights * error
    gradient = np.zeros_like(parameters)
    direction = delta[row, labels] / distances[row, labels, None]
    direction[raw_distances[row, labels] < 1e-12] = 0
    np.add.at(
        gradient[:, :3],
        labels,
        coefficient[:, None] * direction,
    )
    np.add.at(gradient[:, 3], labels, -coefficient)
    regularizer = 0.002
    loss = float(np.sum(weights * error**2) + regularizer * np.mean(radii**3))
    gradient[:, 3] += 3 * regularizer * radii**2 / len(radii)
    return loss, gradient


def _numpy_adam(parameters, points, positive, weights, margin, lower, upper, iterations):
    first = np.zeros_like(parameters)
    second = np.zeros_like(parameters)
    best = parameters.copy()
    initial_loss, _ = _loss_gradient(parameters, points, positive, weights, margin)
    best_loss = initial_loss
    for step in range(1, iterations + 1):
        loss, gradient = _loss_gradient(parameters, points, positive, weights, margin)
        if loss < best_loss:
            best_loss, best = loss, parameters.copy()
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient**2
        update = 0.006 * (first / (1 - 0.9**step)) / (np.sqrt(second / (1 - 0.999**step)) + 1e-8)
        parameters = np.clip(parameters - update, lower, upper)
    loss, _ = _loss_gradient(parameters, points, positive, weights, margin)
    if loss < best_loss:
        best_loss, best = loss, parameters.copy()
    return best, initial_loss, best_loss


def _cuda_adam(parameters, points, positive, weights, margin, lower, upper, iterations):
    try:
        import torch
    except ImportError as exc:
        raise ValueError("CUDA refinement requires the optional torch dependency") from exc
    if not torch.cuda.is_available():
        raise ValueError("CUDA refinement requested, but torch.cuda.is_available() is false")

    def tensor(value):
        return torch.as_tensor(value, device="cuda", dtype=torch.float64)

    parameter = tensor(parameters).requires_grad_(True)
    target, weight = tensor(points), tensor(weights)
    mask = torch.as_tensor(positive, device="cuda", dtype=torch.bool)
    lo, hi = tensor(lower), tensor(upper)
    optimizer = torch.optim.Adam([parameter], lr=0.006)

    def objective():
        delta = parameter[None, :, :3] - target[:, None, :]
        distances = torch.linalg.vector_norm(delta, dim=2).clamp_min(1e-12)
        gap = (distances - parameter[None, :, 3]).min(dim=1).values
        error = torch.where(mask, (gap + margin).clamp_min(0), gap.clamp_max(0))
        return (weight * error.square()).sum() + 0.002 * parameter[:, 3].pow(3).mean()

    initial = objective().detach()
    best_loss = initial.clone()
    best = parameter.detach().clone()
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        # Device-side selection avoids a CPU synchronization at every iteration.
        improved = loss.detach() < best_loss
        best = torch.where(improved, parameter.detach(), best)
        best_loss = torch.minimum(best_loss, loss.detach())
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            parameter.clamp_(min=lo, max=hi)
    final = objective().detach()
    best = torch.where(final < best_loss, parameter.detach(), best)
    best_loss = torch.minimum(best_loss, final)
    torch.cuda.synchronize()
    return best.cpu().numpy(), float(initial.cpu()), float(best_loss.cpu())


def refine_spheres(
    mesh: trimesh.Trimesh,
    initial: list[Sphere],
    occupied: np.ndarray,
    pitch: float,
    cfg: FitConfig,
) -> tuple[list[Sphere], dict[str, Any]]:
    points, positive, weights, counts = _training_data(mesh, occupied, pitch, cfg)
    if len(points) * len(initial) > 4_000_000:
        raise ValueError(
            "Refinement exceeds the 4M point-ball pair budget; reduce refine_samples/count"
        )
    origin = mesh.bounds.mean(axis=0)
    scale = float(mesh.extents.max())
    points = (points - origin) / scale
    parameters = np.column_stack(
        (
            (np.array([s.center for s in initial]) - origin) / scale,
            np.array([s.radius for s in initial]) / scale,
        )
    )
    low, high = (mesh.bounds - origin) / scale
    lower = np.r_[low, pitch * 0.25 / scale]
    upper = np.r_[high, np.linalg.norm(mesh.extents) / scale]
    margin = pitch * 0.15 / scale
    optimize = _numpy_adam if cfg.device == "cpu" else _cuda_adam
    optimized, initial_loss, final_loss = optimize(
        parameters, points, positive, weights, margin, lower, upper, cfg.iterations
    )
    spheres = [
        Sphere(tuple(center * scale + origin), float(radius * scale + cfg.padding))
        for center, radius in zip(optimized[:, :3], optimized[:, 3])
    ]
    return spheres, {
        "optimizer": "Adam, sampled coverage/exterior classification; independent implementation",
        "optimization_device": cfg.device,
        "optimization_iterations": cfg.iterations,
        "optimization_training_counts": counts,
        "optimization_initial_loss": initial_loss,
        "optimization_final_loss": final_loss,
        "optimization_outside_weight": cfg.outside_weight,
        "optimization_reference": (
            "original mesh.contains labels; one closed surface, assuming no self-intersection"
            if cfg.refine_reference == "mesh"
            else "filled voxel grid, not exact mesh SDF"
        ),
        "optimization_positive_margin_m": pitch * 0.15,
    }
