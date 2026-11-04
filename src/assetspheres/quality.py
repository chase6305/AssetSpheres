"""Independent, common-domain Monte Carlo comparison against the original mesh."""

from __future__ import annotations

from typing import Any

import numpy as np
import trimesh

from .geometry import sample_surface, sphere_gaps
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
    """Use the same held-out surface points, volume points and domain for every model.

    The sampling domain encloses the original mesh and every compared ball. This
    prevents large protrusions outside the mesh AABB from escaping the metric.
    Solid classification requires watertight, non-self-intersecting surfaces and
    trimesh's optional rtree dependency. For union-components, these conditions
    apply separately to each component. Open meshes get surface metrics only.
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
    random_generator = np.random.default_rng(seed)
    surface_points = sample_surface(mesh, surface_samples, seed + 1)
    bounds = [mesh.bounds]
    for spheres in models.values():
        centers = np.array([sphere.center for sphere in spheres])
        radii = np.array([sphere.radius for sphere in spheres])[:, None]
        bounds.append(np.array([(centers - radii).min(axis=0), (centers + radii).max(axis=0)]))
    domain_min = np.min(np.array(bounds)[:, 0], axis=0)
    domain_max = np.max(np.array(bounds)[:, 1], axis=0)
    margin = max(float(np.max(domain_max - domain_min)) * 0.02, 1e-12)
    domain_min, domain_max = domain_min - margin, domain_max + margin
    inside_mesh = None
    skip_reason = "Open mesh: no unambiguous solid reference"
    if mesh.is_watertight and solid_policy == "single" and mesh.body_count > 1:
        skip_reason = "Multiple surface bodies: choose explicit parity or union-components policy"
    elif mesh.is_watertight:
        volume_points = random_generator.uniform(domain_min, domain_max, size=(volume_samples, 3))
        components = (
            list(mesh.split(only_watertight=False))
            if solid_policy == "union-components"
            else [mesh]
        )
        try:
            inside_mesh = np.zeros(len(volume_points), dtype=bool)
            for component in components:
                inside_mesh |= np.concatenate(
                    [
                        component.contains(volume_points[start : start + 2048])
                        for start in range(0, len(volume_points), 2048)
                    ]
                )
        except ImportError as exc:
            raise ImportError("Solid quality evaluation needs the optional rtree package") from exc
        if not inside_mesh.any():
            raise ValueError("No interior validation points; increase volume_samples")
    reference_description = (
        "component-wise mesh.contains union"
        if solid_policy == "union-components"
        else "original mesh.contains"
    )
    reports = {}
    for model_name, spheres in models.items():
        surface_gaps = sphere_gaps(surface_points, spheres)
        positive_gaps = np.maximum(surface_gaps, 0)
        report = {
            "surface_coverage": float(np.mean(surface_gaps <= mesh.extents.max() * 1e-9)),
            "surface_gap_p95_m": float(np.quantile(positive_gaps, 0.95)),
            "sphere_volume_sum_m3": float(
                sum(4 * np.pi * sphere.radius**3 / 3 for sphere in spheres)
            ),
            "sphere_count": len(spheres),
            "evaluation_seed": seed,
            "evaluation_domain_m": [domain_min.tolist(), domain_max.tolist()],
            "evaluation_surface_samples": surface_samples,
            "self_intersection_checked": False,
        }
        if inside_mesh is not None:
            inside_spheres = sphere_gaps(volume_points, spheres) <= 0
            true_positive = int(np.count_nonzero(inside_mesh & inside_spheres))
            false_negative = int(np.count_nonzero(inside_mesh & ~inside_spheres))
            false_positive = int(np.count_nonzero(~inside_mesh & inside_spheres))
            interior_count = true_positive + false_negative
            covered_count = true_positive + false_positive
            report.update(
                {
                    "volume_coverage": true_positive / interior_count,
                    "outside_volume_over_mesh": false_positive / interior_count,
                    "sphere_precision": true_positive / covered_count if covered_count else 0.0,
                    "volume_iou": true_positive / (interior_count + false_positive),
                    "validation_interior_points": interior_count,
                    "validation_true_positive": true_positive,
                    "validation_false_negative": false_negative,
                    "validation_false_positive": false_positive,
                    "evaluation_volume_samples": volume_samples,
                    "reference": f"{reference_description}, common domain; Monte Carlo estimate",
                    "solid_policy": solid_policy,
                }
            )
        else:
            report["volume_evaluation_skipped"] = skip_reason
        reports[model_name] = report
    return reports
