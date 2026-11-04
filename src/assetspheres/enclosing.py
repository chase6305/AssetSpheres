"""Tighten an enclosing sphere without weakening its full-point-set enclosure."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize


def _certified_radius(points: np.ndarray, center: np.ndarray) -> float:
    """Recheck every original point after center reconstruction and round outward."""
    delta = points.astype(np.longdouble) - center.astype(np.longdouble)
    maximum = np.max(np.hypot(np.hypot(delta[:, 0], delta[:, 1]), delta[:, 2]))
    if maximum == 0:
        return 0.0
    if maximum >= np.finfo(float).max:
        raise ValueError("Enclosing radius cannot be represented as a finite float64")
    return float(np.nextafter(float(maximum), np.inf))


def tighten_ball(
    points: np.ndarray,
    center: np.ndarray,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Approximate a smaller enclosing ball, and certify it against all input points.

    Optimizes a bounded support set using SLSQP after translating and scaling the
    point cloud. Violating points are added over at most six rounds. Each round
    is limited to 64 optimizer iterations and the support set to 64 points. Final
    radii always come from *all* original points, not just optimizer constraints.

    The returned radius never exceeds the outward-rounded full-set radius at the
    supplied center. A failed optimizer result is never accepted; if no successful
    improvement exists, the original center is returned with a certified radius.
    A singleton/fully coincident cloud has an exact zero-radius solution.

    This is a bounded numerical approximation, not an exact minimum-ball solver.
    The diagnostic guarantee concerns these points; callers must establish that
    enclosing the points also encloses their intended triangles or other shapes.
    """
    points = np.asarray(points, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("points must be a nonempty (N, 3) array")
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("center must contain three finite values")
    if not np.all(np.isfinite(points)):
        raise ValueError("points must contain only finite values")

    original_radius = _certified_radius(points, center)
    best_center, best_radius = center.copy(), original_radius
    # Extended precision protects translation subtraction at very large offsets
    # and squared lengths at very small scales. Optimization itself stays float64.
    pivot = points[0].astype(np.longdouble)
    relative = points.astype(np.longdouble) - pivot
    scale = np.max(np.abs(relative))
    diagnostics: dict[str, Any] = {
        "point_count": len(points),
        "initial_radius_m": original_radius,
        "optimizer_success": False,
        "optimizer_iterations": 0,
        "constraint_rounds": 0,
        "support_points": 0,
        "all_points_enclosed": True,
        "radius_nonincreasing": True,
    }
    if scale == 0:
        best_center, best_radius = points[0].copy(), 0.0
        diagnostics["optimizer_status"] = "coincident_points_exact"
    else:
        normalized = np.asarray(relative / scale, dtype=np.float64)
        lower_center_bounds, upper_center_bounds = normalized.min(axis=0), normalized.max(axis=0)
        initial_center = normalized.mean(axis=0)
        initial_squared_distances = np.sum((normalized - initial_center) ** 2, axis=1)
        # Extrema in axis and box-diagonal directions provide a useful small
        # support set, including both endpoints of collinear point clouds.
        directions = np.vstack(
            [
                np.eye(3),
                -np.eye(3),
                np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]),
            ]
        )
        if len(normalized) <= 64:
            support_indices = np.arange(len(normalized))
        else:
            support_indices = np.unique(
                np.r_[
                    np.argmax(normalized @ directions.T, axis=0),
                    np.argmax(initial_squared_distances),
                ]
            )
        # Parameters pack the normalized center and squared radius.
        current_parameters = np.r_[initial_center, initial_squared_distances.max()]
        status = "fallback_original"
        for round_index in range(6):
            support = np.unique(normalized[support_indices], axis=0)

            def constraints(parameters):
                delta = support - parameters[:3]
                return parameters[3] - np.sum(delta * delta, axis=1)

            def constraint_jacobian(parameters):
                return np.column_stack((2 * (support - parameters[:3]), np.ones(len(support))))

            try:
                optimized = minimize(
                    lambda parameters: parameters[3],
                    current_parameters,
                    method="SLSQP",
                    jac=lambda parameters: np.array([0.0, 0.0, 0.0, 1.0]),
                    bounds=[*zip(lower_center_bounds, upper_center_bounds), (0.0, None)],
                    constraints={"type": "ineq", "fun": constraints, "jac": constraint_jacobian},
                    options={"maxiter": 64, "ftol": 1e-12},
                )
            except (ValueError, FloatingPointError, RuntimeError) as exc:
                status = f"optimizer_error: {type(exc).__name__}"
                break
            diagnostics["constraint_rounds"] = round_index + 1
            diagnostics["optimizer_iterations"] += int(optimized.nit)
            diagnostics["support_points"] = len(support)
            if not optimized.success or not np.all(np.isfinite(optimized.x)):
                status = "optimizer_failed_previous_certified_ball_retained"
                break
            diagnostics["optimizer_success"] = True
            candidate_center = np.asarray(
                pivot + optimized.x[:3].astype(np.longdouble) * scale, dtype=np.float64
            )
            candidate_radius = _certified_radius(points, candidate_center)
            if candidate_radius < best_radius:
                best_center, best_radius = candidate_center, candidate_radius
                status = "optimized"
            squared_distances = np.sum((normalized - optimized.x[:3]) ** 2, axis=1)
            violations = squared_distances - optimized.x[3]
            if violations.max() <= 1e-10 or len(support_indices) >= 64:
                break
            remaining_mask = violations > 1e-10
            remaining_mask[support_indices] = False
            remaining_indices = np.flatnonzero(remaining_mask)
            if not len(remaining_indices):
                break
            additional_indices = remaining_indices[
                np.argsort(-violations[remaining_indices], kind="stable")[:8]
            ]
            support_indices = np.r_[support_indices, additional_indices][:64]
            current_parameters = np.r_[optimized.x[:3], squared_distances.max()]
        diagnostics["optimizer_status"] = status

    diagnostics["final_radius_m"] = best_radius
    diagnostics["radius_ratio"] = best_radius / original_radius if original_radius else 1.0
    diagnostics["improved"] = best_radius < original_radius
    return best_center, best_radius, diagnostics
