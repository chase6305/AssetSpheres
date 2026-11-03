"""Independent, common-domain Monte Carlo comparison against the original mesh."""

from __future__ import annotations

from typing import Any

import numpy as np
import trimesh

from .fit import sample_surface, sphere_gaps
from .model import Sphere


def compare_sphere_models(
    mesh: trimesh.Trimesh,
    models: dict[str, list[Sphere]],
    *,
    surface_samples: int = 6000,
    volume_samples: int = 20_000,
    seed: int = 73001,
    solid_policy: str = "single",
) -> dict[str, dict[str, Any]]:
    """Use the SAME held-out surface points, volume points and domain for every model.

    The sampling domain encloses the original mesh and every compared ball. This
    prevents large protrusions outside the mesh AABB from escaping the metric.
    Solid classification requires a watertight, non-self-intersecting mesh and
    trimesh's optional rtree dependency. Open meshes get surface metrics only.
    Multiple surface bodies are ambiguous by default: explicit parity preserves
    nested cavities, while union-components treats each closed body as a solid.
    The latter fills nested cavities: original URDF part grouping is unavailable.
    Report interior sample counts: rare/small solids yield noisy volume ratios.
    """
    for value in (surface_samples, volume_samples):
        if not isinstance(value, int) or isinstance(value, bool) or value < 32:
            raise ValueError("Evaluation sample counts must be integers >= 32")
    if not models or any(not spheres for spheres in models.values()):
        raise ValueError("Provide at least one nonempty sphere model")
    if solid_policy not in {"single", "parity", "union-components"}:
        raise ValueError("solid_policy must be single, parity, or union-components")
    rng = np.random.default_rng(seed)
    surface = sample_surface(mesh, surface_samples, seed + 1)
    bounds = [mesh.bounds]
    for spheres in models.values():
        centers = np.array([s.center for s in spheres])
        radii = np.array([s.radius for s in spheres])[:, None]
        bounds.append(np.array([(centers - radii).min(axis=0), (centers + radii).max(axis=0)]))
    low = np.min(np.array(bounds)[:, 0], axis=0)
    high = np.max(np.array(bounds)[:, 1], axis=0)
    margin = max(float(np.max(high - low)) * 0.02, 1e-12)
    low, high = low - margin, high + margin
    points = rng.uniform(low, high, size=(volume_samples, 3))
    inside = None
    skip_reason = "Open mesh: no unambiguous solid reference"
    if mesh.is_watertight and solid_policy == "single" and mesh.body_count > 1:
        skip_reason = "Multiple surface bodies: choose explicit parity or union-components policy"
    elif mesh.is_watertight:
        components = (
            list(mesh.split(only_watertight=False))
            if solid_policy == "union-components"
            else [mesh]
        )
        try:
            inside = np.zeros(len(points), dtype=bool)
            for component in components:
                inside |= np.concatenate(
                    [
                        component.contains(points[start : start + 2048])
                        for start in range(0, len(points), 2048)
                    ]
                )
        except ImportError as exc:
            raise ImportError("Solid quality evaluation needs the optional rtree package") from exc
        if not inside.any():
            raise ValueError("No interior validation points; increase volume_samples")
    reports = {}
    for name, spheres in models.items():
        gap = sphere_gaps(surface, spheres)
        positive_gap = np.maximum(gap, 0)
        report = {
            "surface_coverage": float(np.mean(gap <= mesh.extents.max() * 1e-9)),
            "surface_gap_p95_m": float(np.quantile(positive_gap, 0.95)),
            "sphere_volume_sum_m3": float(sum(4 * np.pi * s.radius**3 / 3 for s in spheres)),
            "sphere_count": len(spheres),
            "evaluation_seed": seed,
            "evaluation_domain_m": [low.tolist(), high.tolist()],
            "evaluation_surface_samples": surface_samples,
            "self_intersection_checked": False,
        }
        if inside is not None:
            covered = sphere_gaps(points, spheres) <= 0
            tp = int(np.count_nonzero(inside & covered))
            fn = int(np.count_nonzero(inside & ~covered))
            fp = int(np.count_nonzero(~inside & covered))
            report.update(
                {
                    "volume_coverage": tp / (tp + fn),
                    "outside_volume_over_mesh": fp / (tp + fn),
                    "sphere_precision": tp / (tp + fp) if tp + fp else 0.0,
                    "volume_iou": tp / (tp + fn + fp),
                    "validation_interior_points": tp + fn,
                    "validation_true_positive": tp,
                    "validation_false_negative": fn,
                    "validation_false_positive": fp,
                    "evaluation_volume_samples": volume_samples,
                    "reference": "original mesh.contains, common domain; Monte Carlo estimate",
                    "solid_policy": solid_policy,
                }
            )
        else:
            report["volume_evaluation_skipped"] = skip_reason
        reports[name] = report
    return reports
