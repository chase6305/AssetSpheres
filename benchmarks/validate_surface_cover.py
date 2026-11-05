"""Check repaired surfaces using independent physical-coordinate barycentric points.

Random cases include open triangle soups, convex hulls, anisotropic geometry,
very small/large scales and large translations. These are numerical regression
checks, not a substitute for the triangle-enclosure construction or a formal proof.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import scipy
import trimesh

from assetspheres import Sphere
from assetspheres.surface_cover import repair_surface_cover


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=500)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=481003)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/surface-cover-validation.json")
    )
    args = parser.parse_args()
    if args.cases < 1 or args.samples < 1:
        parser.error("--cases and --samples must be positive")
    random_generator = np.random.default_rng(args.seed)
    records = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "trimesh": trimesh.__version__,
        "longdouble_mantissa_bits": np.finfo(np.longdouble).nmant,
        "seed": args.seed,
        "samples_per_case": args.samples,
        "scale_exponent_range": [-180, 140],
        "maximum_translation_scale_ratio": 1e12,
        "scope": "independent original-triangle barycentric points, extended-precision distances",
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case_index in range(args.cases):
        vertices = random_generator.uniform(-1, 1, size=(48, 3))
        if case_index % 2:
            hull = trimesh.convex.convex_hull(vertices)
            vertices, faces = np.asarray(hull.vertices), np.asarray(hull.faces)
        else:
            faces = random_generator.integers(0, len(vertices), size=(32, 3))
        exponent = float(random_generator.uniform(-180, 140))
        scale = 10**exponent
        anisotropy = 10 ** random_generator.uniform(-8, 0, size=3)
        if case_index % 4 == 0:
            anisotropy[:] = 1
        translation = random_generator.uniform(-1, 1, size=3) * scale
        translation *= 10 ** int(random_generator.integers(0, 13))
        vertices = vertices * anisotropy * scale + translation
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        sphere_count = (1, 3, 8, 24, 129)[case_index % 5]
        centers = random_generator.uniform(-1, 1, size=(sphere_count, 3)) * scale + translation
        radii = random_generator.uniform(0.001, 0.2, size=sphere_count) * scale
        initial = [Sphere(tuple(center), float(radius)) for center, radius in zip(centers, radii)]
        repaired, details = repair_surface_cover(mesh, initial)
        assert len(initial) == len(repaired)
        assert all(a.center == b.center and a.radius <= b.radius for a, b in zip(initial, repaired))
        assert (
            details["surface_repair_visited_triangles"]
            == len(faces) + 4 * details["surface_repair_triangle_splits"]
        )
        assert (
            details["surface_repair_leaf_triangles"]
            == len(faces) + 3 * details["surface_repair_triangle_splits"]
        )

        triangle_indices = random_generator.integers(0, len(faces), size=args.samples)
        triangles = vertices[faces[triangle_indices]].astype(np.longdouble)
        barycentric = random_generator.dirichlet(np.ones(3), size=args.samples).astype(
            np.longdouble
        )
        barycentric /= barycentric.sum(axis=1, keepdims=True)
        points = (
            triangles[:, 0]
            + barycentric[:, 1, None] * (triangles[:, 1] - triangles[:, 0])
            + barycentric[:, 2, None] * (triangles[:, 2] - triangles[:, 0])
        )
        minimum_gaps = np.full(len(points), np.inf, dtype=np.longdouble)
        for sphere in repaired:
            offsets = points - np.array(sphere.center, dtype=np.longdouble)
            distances = np.hypot(np.hypot(offsets[:, 0], offsets[:, 1]), offsets[:, 2])
            np.minimum(minimum_gaps, distances - np.longdouble(sphere.radius), out=minimum_gaps)
        assert minimum_gaps.max() <= 0, (case_index, exponent, str(minimum_gaps.max()))
        records["cases"].append(
            {
                "index": case_index,
                "scale_exponent": exponent,
                "sphere_count": sphere_count,
                "maximum_gap_normalized": float(minimum_gaps.max() / np.longdouble(scale)),
                "repair": details,
            }
        )
        if (case_index + 1) % 20 == 0 or case_index + 1 == args.cases:
            args.output.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
            print(
                f"{case_index + 1}/{args.cases}: covered; centers unchanged, radii only grow",
                flush=True,
            )
    print(args.output)


if __name__ == "__main__":
    main()
