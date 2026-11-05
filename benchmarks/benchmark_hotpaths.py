"""Paired CPU benchmarks with explicit references and exact output checks.

Run: python benchmarks/benchmark_hotpaths.py --repeats 7 --geometry-backend open3d
Timing excludes imports and warm-up. Allocation peaks use tracemalloc, NOT RSS
or native runtime memory. Optional distance-grid runtimes are measured separately.
The full-mesh uniform subdivision reference preserves the legacy 1-to-4 rule;
Trimesh 5's default remesher follows different sampling rules.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import tempfile
import time
import tracemalloc
from pathlib import Path
from unittest.mock import patch

import numpy as np
import scipy
import trimesh
from scipy import ndimage
from scipy.spatial.distance import cdist

from assetspheres import FitConfig, Sphere, asset, voxelization
from assetspheres.fit import sphere_gaps
from assetspheres.geometry import farthest_point_indices
from assetspheres.refine import _loss_gradient, _sample_voxel_exterior
from assetspheres.surface_cover import _triangle_requirements
from assetspheres.surface_grid import subdivide_surface_grid


def reference_surface_grid(mesh, pitch):
    """Whole-mesh reference: split all three edges if any exceeds half a voxel."""
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    surface_samples = []
    for round_index in range(11):
        triangles = vertices[faces]
        edge_offsets = triangles[:, [1, 2, 0]] - triangles
        split = (np.sqrt(np.sum(edge_offsets**2, axis=2)) > pitch / 2).any(axis=1)
        surface_samples.append(vertices[np.unique(faces[~split])])
        if not split.any():
            break
        if round_index == 10:
            raise ValueError("Reference exceeded ten subdivision rounds")
        vertices, faces = trimesh.remesh.subdivide(vertices, faces[split])
    hits = np.unique(np.round(np.vstack(surface_samples) / pitch).astype(np.int64), axis=0)
    origin = hits.min(axis=0)
    return trimesh.voxel.VoxelGrid(
        trimesh.voxel.encoding.SparseBinaryEncoding(hits - origin),
        transform=trimesh.transformations.scale_and_translate(
            scale=pitch, translate=origin * pitch
        ),
    )


def reference_farthest_points(points, count, seed):
    selected = np.empty(min(count, len(points)), dtype=int)
    nearest_squared = np.full(len(points), np.inf)
    current = int(np.random.default_rng(seed).integers(len(points)))
    for step in range(len(selected)):
        selected[step] = current
        nearest_squared = np.minimum(
            nearest_squared, np.sum((points - points[current]) ** 2, axis=1)
        )
        nearest_squared[selected[: step + 1]] = -1
        current = int(np.argmax(nearest_squared))
    return selected


def reference_load_asset(path):
    geometry_loader = asset._geometry

    def uncached_geometry(element, base, packages, shared_meshes):
        return geometry_loader(element, base, packages, {})

    with patch.object(asset, "_geometry", uncached_geometry):
        return asset.load_asset(path)


def same_asset(first, second):
    return first.links.keys() == second.links.keys() and all(
        np.array_equal(first.links[name].mesh.vertices, second.links[name].mesh.vertices)
        and np.array_equal(first.links[name].mesh.faces, second.links[name].mesh.faces)
        and np.array_equal(first.zero_pose_transforms[name], second.zero_pose_transforms[name])
        for name in first.links
    )


def reference_gaps(points, spheres):
    centers = np.asarray([s.center for s in spheres])
    radii = np.asarray([s.radius for s in spheres])
    result = np.full(len(points), np.inf)
    for first in range(0, len(points), 2048):
        for sphere in range(0, len(spheres), 128):
            block = (
                np.linalg.norm(
                    points[first : first + 2048, None] - centers[None, sphere : sphere + 128],
                    axis=2,
                )
                - radii[None, sphere : sphere + 128]
            )
            result[first : first + 2048] = np.minimum(
                result[first : first + 2048], block.min(axis=1)
            )
    return result


def reference_gradient(parameters, points, positive, weights, margin):
    centers, radii = parameters[:, :3], parameters[:, 3]
    offsets = centers[None] - points[:, None]
    distances = np.linalg.norm(offsets, axis=2)
    safe = np.maximum(distances, 1e-12)
    gaps = safe - radii[None]
    nearest = gaps.argmin(axis=1)
    indices = np.arange(len(points))
    active = gaps[indices, nearest]
    errors = np.where(positive, np.maximum(active + margin, 0), np.minimum(active, 0))
    derivatives = 2 * weights * errors
    directions = offsets[indices, nearest] / safe[indices, nearest, None]
    directions[distances[indices, nearest] < 1e-12] = 0
    gradient = np.zeros_like(parameters)
    np.add.at(gradient[:, :3], nearest, derivatives[:, None] * directions)
    np.add.at(gradient[:, 3], nearest, -derivatives)
    gradient[:, 3] += 3 * 0.002 * radii**2 / len(radii)
    loss = float(np.sum(weights * errors**2) + 0.002 * np.mean(radii**3))
    return loss, gradient


def reference_triangle_requirements(triangles, centers, radii):
    """Initial repair kernel: retain distances for every corner in a center block."""
    best_gaps = np.full(len(triangles), np.inf)
    best_centers = np.zeros(len(triangles), dtype=np.intp)
    required_radii = np.zeros(len(triangles))
    for first_center in range(0, len(centers), 128):
        center_block = centers[first_center : first_center + 128]
        distances = cdist(triangles.reshape(-1, 3), center_block)
        maximum_distances = distances.reshape(len(triangles), 3, -1).max(axis=1)
        gaps = maximum_distances - radii[first_center : first_center + len(center_block)]
        nearest = gaps.argmin(axis=1)
        candidate_gaps = gaps[np.arange(len(triangles)), nearest]
        improved = candidate_gaps < best_gaps
        best_gaps[improved] = candidate_gaps[improved]
        best_centers[improved] = first_center + nearest[improved]
        required_radii[improved] = maximum_distances[np.flatnonzero(improved), nearest[improved]]
    return best_gaps, best_centers, required_radii


def reference_voxel_exterior(occupied_points, voxel_size_m, sample_count, random_generator):
    lattice_origin = occupied_points.min(axis=0)
    occupied_indices = np.rint((occupied_points - lattice_origin) / voxel_size_m).astype(int)
    border_cells = 4
    grid_shape = tuple(occupied_indices.max(axis=0) + 1 + 2 * border_cells)
    occupied_cells = np.zeros(grid_shape, dtype=bool)
    occupied_cells[tuple((occupied_indices + border_cells).T)] = True
    exterior_distance = ndimage.distance_transform_edt(~occupied_cells)
    exterior_indices = np.argwhere((exterior_distance >= 1.5) & (exterior_distance <= border_cells))
    if not len(exterior_indices):
        raise ValueError("No exterior training samples; increase voxel resolution")
    selected_indices = random_generator.choice(
        len(exterior_indices), min(sample_count, len(exterior_indices)), replace=False
    )
    return lattice_origin + (exterior_indices[selected_indices] - border_cells) * voxel_size_m


def compare(reference, optimized, equal, repeats, *, allocations=True):
    before, after = reference(), optimized()
    assert equal(before, after), "Optimized output differs from the reference"
    calls = {"reference": reference, "optimized": optimized}
    times = {name: [] for name in calls}
    for repeat in range(repeats):
        # Alternate order to reduce a persistent first/second-call bias.
        for name in list(calls)[:: 1 if repeat % 2 == 0 else -1]:
            started = time.perf_counter()
            calls[name]()
            times[name].append(time.perf_counter() - started)
    result = {
        name: {"times_s": values, "median_s": statistics.median(values)}
        for name, values in times.items()
    }
    result["speedup"] = result["reference"]["median_s"] / result["optimized"]["median_s"]
    result["exact_output_match"] = True
    if allocations:
        for name, call in calls.items():
            tracemalloc.start()
            try:
                call()
                result[name]["traced_allocation_peak_bytes"] = tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()
    return result


def full_grid_batches(layout, bounds):
    cell_count = math.prod(layout.shape)
    for first in range(0, cell_count, voxelization._QUERY_BATCH_SIZE):
        yield np.arange(first, min(first + voxelization._QUERY_BATCH_SIZE, cell_count))


def same_grids(first, second):
    return (
        np.array_equal(first.voxel_grid.transform, second.voxel_grid.transform)
        and np.array_equal(first.clearance_grid_m, second.clearance_grid_m)
        and first.backend_details == second.backend_details
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--geometry-backend", action="append", choices=["open3d", "warp"])
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark-hotpaths.json"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    records = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "trimesh": trimesh.__version__,
        "repeats": args.repeats,
        "timing_scope": "paired warm calls, alternating order; imports/warm-up excluded",
        "memory_scope": "tracemalloc peak allocations, not RSS or native/GPU runtime memory",
        "subdivision_reference": (
            "Whole-mesh legacy 1-to-4 reference using public remesh.subdivide; "
            "not the Trimesh 5 default remesher"
        ),
        "cases": {},
    }

    def save(name, result):
        records["cases"][name] = result
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, indent=2, allow_nan=False), encoding="utf-8")
        print(f"{name:35} speedup={result['speedup']:.2f}x, exact output match", flush=True)

    with tempfile.TemporaryDirectory(prefix="assetspheres-benchmark-") as temporary:
        directory = Path(temporary)
        trimesh.creation.icosphere(subdivisions=5, radius=0.1).export(directory / "part.stl")
        links = ['<link name="root"/>']
        for index in range(16):
            links.append(
                f'<link name="part_{index}"><collision><origin xyz="{index * 0.1} 0 0"/>'
                f'<geometry><mesh filename="part.stl" scale="{1 + index * 0.01} 1 1"/>'
                f'</geometry></collision></link><joint name="joint_{index}" type="fixed">'
                f'<parent link="root"/><child link="part_{index}"/></joint>'
            )
        robot_path = directory / "robot.urdf"
        robot_path.write_text('<robot name="shared">' + "".join(links) + "</robot>")
        save(
            "load-asset/16-shared-stl-instances",
            compare(
                lambda: reference_load_asset(robot_path),
                lambda: asset.load_asset(robot_path),
                same_asset,
                args.repeats,
            ),
        )
    random = np.random.default_rng(20261103)
    for point_count, sphere_count in [(6000, 32), (50000, 128)]:
        points = np.random.default_rng(20261103).normal(size=(point_count, 3))
        save(
            f"farthest/{point_count}/{sphere_count}",
            compare(
                lambda: reference_farthest_points(points, sphere_count, 91),
                lambda: farthest_point_indices(points, sphere_count, 91),
                np.array_equal,
                args.repeats,
            ),
        )
    for point_count, sphere_count in [(6000, 32), (100000, 32), (20000, 256)]:
        points = random.normal(size=(point_count, 3))
        spheres = [
            Sphere(tuple(c), float(r))
            for c, r in zip(
                random.normal(size=(sphere_count, 3)), random.uniform(0.01, 1, sphere_count)
            )
        ]
        save(
            f"gaps/{point_count}/{sphere_count}",
            compare(
                lambda: reference_gaps(points, spheres),
                lambda: sphere_gaps(points, spheres),
                np.array_equal,
                args.repeats,
            ),
        )
    points = random.normal(size=(12000, 3))
    parameters = random.random((128, 4))
    positive, weights = np.arange(len(points)) % 2 == 0, np.full(len(points), 1 / len(points))
    inputs = (parameters, points, positive, weights, 0.001)
    save(
        "gradient/12000/128",
        compare(
            lambda: reference_gradient(*inputs),
            lambda: _loss_gradient(*inputs),
            lambda a, b: a[0] == b[0] and np.array_equal(a[1], b[1]),
            args.repeats,
        ),
    )
    for triangle_count, sphere_count in [(32, 24), (2048, 24), (2048, 128), (2048, 257)]:
        triangles = random.normal(size=(triangle_count, 3, 3))
        centers = random.normal(size=(sphere_count, 3))
        radii = random.uniform(0.01, 1, sphere_count)
        inputs = (triangles, centers, radii)
        save(
            f"surface-repair/{triangle_count}/{sphere_count}",
            compare(
                lambda: reference_triangle_requirements(*inputs),
                lambda: _triangle_requirements(*inputs),
                lambda a, b: all(np.array_equal(x, y) for x, y in zip(a, b)),
                args.repeats,
            ),
        )
    for side in (16, 48, 112):
        indices = np.array(np.meshgrid(*[np.arange(side)] * 3, indexing="ij")).reshape(3, -1).T
        points = np.array(indices * 0.003, order="F")
        save(
            f"voxel-exterior/cube{side}",
            compare(
                lambda: reference_voxel_exterior(points, 0.003, 2048, np.random.default_rng(91)),
                lambda: _sample_voxel_exterior(points, 0.003, 2048, np.random.default_rng(91)),
                np.array_equal,
                args.repeats,
            ),
        )
    for name, mesh, resolution in [
        ("cylinder", trimesh.creation.cylinder(radius=0.05, height=0.3, sections=48), 48),
        ("sphere", trimesh.creation.icosphere(radius=0.1, subdivisions=3), 80),
    ]:
        pitch = float(mesh.extents.max()) / resolution
        save(
            f"subdivide/{name}/res{resolution}",
            compare(
                lambda: reference_surface_grid(mesh, pitch),
                lambda: subdivide_surface_grid(mesh, pitch, 2_000_000),
                lambda a, b: (
                    np.array_equal(a.transform, b.transform) and np.array_equal(a.matrix, b.matrix)
                ),
                args.repeats,
            ),
        )
    components = []
    for corner in np.ndindex((3, 3, 3)):
        mesh = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
        mesh.apply_translation(np.asarray(corner) * 0.2)
        components.append(mesh)
    combined = trimesh.util.concatenate(components)
    for backend in args.geometry_backend or []:
        config = FitConfig(voxelizer=backend, voxel_solid_policy="union-components", resolution=72)

        def full_reference():
            with patch.object(voxelization, "_component_grid_batches", full_grid_batches):
                return voxelization.prepare_distance_grid(combined, config)

        save(
            f"distance-grid/{backend}/27boxes",
            compare(
                full_reference,
                lambda: voxelization.prepare_distance_grid(combined, config),
                same_grids,
                args.repeats,
                allocations=False,
            ),
        )


if __name__ == "__main__":
    main()
