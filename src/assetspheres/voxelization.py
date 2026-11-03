"""Optional signed-distance grid preparation without triangle subdivision.

These backends sample ORIGINAL mesh interiors at cell centers; the legacy
subdivide-and-fill backend instead rasterizes a surface shell and fills it.
They deliberately remain opt-in because their finite-grid semantics differ.
"""

from __future__ import annotations

import importlib
import math
import threading
from contextlib import closing, nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import trimesh

from .model import FitConfig

# Warp allocation, BVH construction, launches and teardown share runtime state.
# This protects AssetSpheres' own preparations, not unrelated application code.
_WARP_PREPARATION_LOCK = threading.RLock()
_QUERY_BATCH_SIZE = 65536


@dataclass(frozen=True)
class PreparedDistanceGrid:
    voxel_grid: trimesh.voxel.VoxelGrid
    clearance_grid_m: np.ndarray
    backend_details: dict[str, Any]


@dataclass(frozen=True)
class GridLayout:
    """A bounded, cell-centered lattice and its dimensionless query frame."""

    shape: tuple[int, int, int]
    voxel_size_m: float
    origin_m: np.ndarray
    normalization_origin_m: np.ndarray
    normalization_scale_m: float

    def transform(self) -> np.ndarray:
        return trimesh.transformations.scale_and_translate(
            scale=self.voxel_size_m,
            translate=self.origin_m,
        )

    def normalized_points(self, first: int, stop: int) -> np.ndarray:
        flat_indices = np.arange(first, stop, dtype=np.int64)
        indices = np.column_stack(np.unravel_index(flat_indices, self.shape))
        # Query the SAME rounded physical centers later used by the fitted balls.
        # Normalizing before world-coordinate reconstruction differs at large offsets.
        physical_centers = trimesh.transform_points(indices, self.transform())
        return np.asarray(
            (physical_centers - self.normalization_origin_m) / self.normalization_scale_m,
            dtype=np.float32,
        )


def _grid_layout(triangle_mesh: trimesh.Trimesh, configuration: FitConfig) -> GridLayout:
    voxel_size_m = (
        configuration.voxel_size or float(triangle_mesh.extents.max()) / configuration.resolution
    )
    if np.any(triangle_mesh.extents <= voxel_size_m / 2):
        raise ValueError(
            "Mesh has no interior grid samples; it may be thinner than the voxel pitch. "
            "Increase --resolution, decrease --voxel-size, or use --voxelizer subdivide / "
            "--method slab"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        extent_ratios = triangle_mesh.extents / voxel_size_m
    if not np.all(np.isfinite(extent_ratios)):
        raise ValueError("Voxel size is too small to represent the grid; increase --voxel-size")
    shape = tuple(math.ceil(float(ratio)) + 2 for ratio in extent_ratios)
    grid_cells = math.prod(shape)
    if grid_cells > configuration.max_grid_cells:
        raise ValueError(
            f"Distance grid needs {grid_cells:,} cells (limit {configuration.max_grid_cells:,}); "
            "increase --voxel-size or reduce --resolution"
        )
    coordinate_ulp_m = abs(np.spacing(float(np.max(np.abs(triangle_mesh.vertices)))))
    if voxel_size_m <= 4 * coordinate_ulp_m:
        raise ValueError(
            "Voxel pitch is below reliable world-coordinate precision; recenter the mesh "
            "near the origin or increase --voxel-size"
        )
    normalization_origin_m = triangle_mesh.bounds[0] + triangle_mesh.extents / 2
    return GridLayout(
        shape,
        voxel_size_m,
        triangle_mesh.bounds[0] - voxel_size_m / 2,
        normalization_origin_m,
        float(triangle_mesh.extents.max()),
    )


def _closed_components(triangle_mesh: trimesh.Trimesh, solid_policy: str) -> list[trimesh.Trimesh]:
    if not triangle_mesh.is_watertight:
        raise ValueError("Distance-grid backends require a closed triangle surface")
    if triangle_mesh.body_count == 1:
        return [triangle_mesh]
    if solid_policy == "single":
        raise ValueError(
            "Multiple closed surface bodies have ambiguous cavity/union semantics; use "
            "--voxel-solid-policy union-components to treat EACH shell as a solid, "
            "or keep --voxelizer subdivide"
        )
    components = list(triangle_mesh.split(only_watertight=False, repair=False))
    if any(not component.is_watertight for component in components):
        raise ValueError("Every component must be closed for union-components distance queries")
    return components


class _Open3DDistanceQuery:
    def __init__(self, vertices: np.ndarray, faces: np.ndarray, device: str):
        self.scene = None
        try:
            self.open3d = importlib.import_module("open3d")
        except ImportError as exc:
            raise ValueError(
                "Open3D voxelizer needs the optional open3d>=0.19,<0.20 package. "
                "From the repository root run: python -m pip install -e '.[open3d]'"
            ) from exc
        try:
            # RaycastingScene uses CPU/SYCL, not Open3D's CUDA tensor device.
            self.scene = self.open3d.t.geometry.RaycastingScene(nthreads=1)
            self.scene.add_triangles(
                self.open3d.core.Tensor(np.ascontiguousarray(vertices, dtype=np.float32)),
                self.open3d.core.Tensor(np.ascontiguousarray(faces, dtype=np.uint32)),
            )
        except BaseException:
            self.close()
            raise

    def signed_distances(self, normalized_points: np.ndarray) -> np.ndarray:
        if self.scene is None:
            raise RuntimeError("Open3D distance query is closed")
        queries = self.open3d.core.Tensor(normalized_points)
        return self.scene.compute_signed_distance(queries, nthreads=1, nsamples=3).numpy()

    def close(self) -> None:
        self.scene = None

    def __enter__(self):
        if self.scene is None:
            raise RuntimeError("Open3D distance query is closed")
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        self.close()


def _distance_query(backend: str, vertices: np.ndarray, faces: np.ndarray, device: str):
    if backend == "open3d":
        return _Open3DDistanceQuery(vertices, faces, device)
    try:
        module = importlib.import_module(".warp_queries", package=__package__)
    except ImportError as exc:
        raise ValueError(
            "Warp voxelizer needs the optional warp-lang>=1.19,<1.20 package. "
            "From the repository root run: python -m pip install -e '.[warp]'"
        ) from exc
    return module.WarpDistanceQuery(vertices, faces, device)


def _merge_component_clearance(
    layout: GridLayout,
    distance_query,
    interior_clearance: np.ndarray,
    numerical_allowance_m: float,
) -> tuple[int, int]:
    """Merge one component into the shared clearance lower bound in batch order."""
    cell_count = len(interior_clearance)
    interior_count = 0
    best_clearance_m = 0.0
    representative_index = -1
    for first in range(0, cell_count, _QUERY_BATCH_SIZE):
        stop = min(first + _QUERY_BATCH_SIZE, cell_count)
        signed_distances = distance_query.signed_distances(layout.normalized_points(first, stop))
        if not np.all(np.isfinite(signed_distances)):
            raise ValueError("Mesh distance query returned a non-finite result")
        if np.any(np.abs(signed_distances) >= 1e20):
            raise ValueError("Mesh distance query missed its BVH; check mesh and voxel scale")
        component_clearance = np.maximum(
            -signed_distances.astype(np.float64) * layout.normalization_scale_m
            - numerical_allowance_m,
            0,
        )
        interior_count += int(np.count_nonzero(component_clearance))
        best_local_index = int(np.argmax(component_clearance))
        if component_clearance[best_local_index] > best_clearance_m:
            best_clearance_m = component_clearance[best_local_index]
            representative_index = first + best_local_index
        # Each ball fits in an individual solid. The largest component distance
        # is a lower bound for clearance inside the components' solid union.
        interior_clearance[first:stop] = np.maximum(
            interior_clearance[first:stop], component_clearance
        )
    return interior_count, representative_index


def _sample_closed_components(
    components: list[trimesh.Trimesh],
    layout: GridLayout,
    configuration: FitConfig,
    numerical_allowance_m: float,
) -> tuple[np.ndarray, list[int], list[int]]:
    interior_clearance = np.zeros(math.prod(layout.shape), dtype=np.float64)
    component_interior_counts = []
    representative_flat_indices = []
    for component_index, component in enumerate(components):
        normalized_vertices = np.asarray(
            (component.vertices - layout.normalization_origin_m) / layout.normalization_scale_m,
            dtype=np.float32,
        )
        with closing(
            _distance_query(
                configuration.voxelizer,
                normalized_vertices,
                component.faces,
                configuration.device,
            )
        ) as distance_query:
            interior_count, representative_index = _merge_component_clearance(
                layout, distance_query, interior_clearance, numerical_allowance_m
            )
        if not interior_count:
            raise ValueError(
                f"Closed component {component_index} has no interior grid samples; "
                "it may be thinner than the voxel pitch. Increase --resolution, decrease "
                "--voxel-size, or use --voxelizer subdivide / --method slab"
            )
        component_interior_counts.append(interior_count)
        representative_flat_indices.append(representative_index)
    return interior_clearance, component_interior_counts, representative_flat_indices


def _distance_grid_details(
    configuration: FitConfig,
    layout: GridLayout,
    numerical_allowance_m: float,
    coordinate_ulp_m: float,
    component_interior_counts: list[int],
    representative_flat_indices: list[int],
) -> dict[str, Any]:
    representative_grid_indices = np.column_stack(
        np.unravel_index(representative_flat_indices, layout.shape)
    )
    representative_centers = trimesh.transform_points(
        representative_grid_indices, layout.transform()
    )
    details = {
        "voxelizer": configuration.voxelizer,
        "voxel_solid_policy": configuration.voxel_solid_policy,
        "geometry_query_device": "cpu"
        if configuration.voxelizer == "open3d"
        else configuration.device,
        "voxel_sampling": "strict interior cell centers; no surface-shell filling",
        "query_geometry_precision": "float32, centered and normalized",
        "query_sign_samples": 3,
        "radius_numerical_allowance_m": numerical_allowance_m,
        "world_coordinate_ulp_m": coordinate_ulp_m,
        "closed_component_count": len(component_interior_counts),
        "component_interior_sample_counts": component_interior_counts,
        "component_representative_grid_indices": representative_grid_indices.tolist(),
        "component_representative_centers_m": representative_centers.tolist(),
        "self_intersection_checked": False,
        "radius_model": "original mesh distance minus numerical allowance; union lower bound",
        "voxelizer_warnings": [
            "Center occupancy can miss sub-voxel features; empty component sampling is rejected",
            "Closed, non-self-intersecting surfaces are assumed; this is not a certified exact SDF",
        ],
    }
    if (
        configuration.voxel_solid_policy == "union-components"
        and len(component_interior_counts) > 1
    ):
        details["voxelizer_warnings"].append(
            "Each closed shell is solid; nested cavities are filled"
        )
    return details


def _prepare_distance_grid(
    triangle_mesh: trimesh.Trimesh, configuration: FitConfig
) -> PreparedDistanceGrid:
    layout = _grid_layout(triangle_mesh, configuration)
    components = _closed_components(triangle_mesh, configuration.voxel_solid_policy)
    coordinate_ulp_m = abs(np.spacing(float(np.max(np.abs(triangle_mesh.vertices)))))
    numerical_allowance_m = (
        layout.normalization_scale_m * 64 * np.finfo(np.float32).eps
        + 4 * np.sqrt(3) * coordinate_ulp_m
    )
    interior_clearance, component_interior_counts, representative_flat_indices = (
        _sample_closed_components(components, layout, configuration, numerical_allowance_m)
    )
    clearance_grid = interior_clearance.reshape(layout.shape)
    voxel_grid = trimesh.voxel.VoxelGrid(clearance_grid > 0, transform=layout.transform())
    details = _distance_grid_details(
        configuration,
        layout,
        numerical_allowance_m,
        coordinate_ulp_m,
        component_interior_counts,
        representative_flat_indices,
    )
    return PreparedDistanceGrid(voxel_grid, clearance_grid, details)


def prepare_distance_grid(
    triangle_mesh: trimesh.Trimesh,
    configuration: FitConfig,
) -> PreparedDistanceGrid:
    """Prepare interior samples and metric radii with explicit shell semantics.

    Warp preparations are serialized through allocation, sampling and synchronized
    teardown because that runtime has shared state. Open3D preparations remain
    independent. Float32 query allowances reduce precision risk but are not a
    certified error bound; components with no interior samples fail explicitly.
    """
    if configuration.voxelizer not in {"open3d", "warp"}:
        raise ValueError("prepare_distance_grid requires open3d or warp voxelizer")
    preparation_context = (
        _WARP_PREPARATION_LOCK if configuration.voxelizer == "warp" else nullcontext()
    )
    with preparation_context:
        return _prepare_distance_grid(triangle_mesh, configuration)
