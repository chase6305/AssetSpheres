"""Optional, memory-bounded PyTorch kernels for CPU or CUDA.

Imports of this module do not import torch. All geometry calculations use float64:
float32 can turn an exterior point into an interior point near a sphere boundary.
The working-matrix budget excludes the linear input/output tensors and PyTorch's
runtime/allocator. It is not a bound on the process's total resident memory.
"""

from __future__ import annotations

import importlib
import math

import numpy as np

_SPHERE_BLOCK_SIZE = 128
_PAIR_WORK_BYTES = 40


def _positive_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _inputs(points, centers, radii) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = tuple(np.asarray(value, dtype=np.float64) for value in (points, centers, radii))
    points, centers, radii = arrays
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("centers must have shape (M, 3)")
    if radii.shape != (len(centers),):
        raise ValueError("radii must have shape (M,) matching centers")
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("points, centers, and radii must be finite")
    if np.any(radii <= 0):
        raise ValueError("radii must be positive")
    return tuple(np.array(array, order="C", copy=True) for array in arrays)


def _runtime(device: str):
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise ImportError("Tensor operations require an optional compatible PyTorch build") from exc
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was explicitly requested but PyTorch has no available CUDA device")
    return torch


def _squared_distances(centers, points):
    # Separate multiply/add operations fix the reduction order and avoid fused
    # norm accumulation changing lattice-boundary decisions across CPU/CUDA.
    delta = centers[:, None, 0] - points[None, :, 0]
    squared = delta.square()
    for axis in (1, 2):
        delta = centers[:, None, axis] - points[None, :, axis]
        squared += delta.square()
    return squared


def point_sphere_gaps(
    points,
    centers,
    radii,
    *,
    device: str = "cpu",
    chunk_size: int = 2048,
) -> np.ndarray:
    """Return ``min(norm(point-center)-radius)`` using float64 tensor blocks.

    The result is exact exterior distance and negative inside the ball union.
    A block contains at most ``chunk_size * min(M, 128)`` point-sphere pairs.
    It stores three float64 offsets per pair, squared and summed in place.
    Input validation, NumPy copies, and CPU/device transfers are included.
    """
    _positive_integer(chunk_size, "chunk_size")
    points, centers, radii = _inputs(points, centers, radii)
    torch = _runtime(device)
    if not len(centers):
        return np.full(len(points), np.inf)
    if not len(points):
        return np.empty(0, dtype=np.float64)
    with torch.no_grad():
        point_tensor, center_tensor, radius_tensor = [
            torch.from_numpy(array).to(device=device) for array in (points, centers, radii)
        ]
        minimum_gaps = torch.full((len(points),), float("inf"), dtype=torch.float64, device=device)
        for point_start in range(0, len(points), chunk_size):
            point_slice = slice(point_start, point_start + chunk_size)
            for sphere_start in range(0, len(centers), _SPHERE_BLOCK_SIZE):
                sphere_slice = slice(sphere_start, sphere_start + _SPHERE_BLOCK_SIZE)
                delta = point_tensor[point_slice, None, :] - center_tensor[None, sphere_slice, :]
                delta.square_()
                gaps = delta[:, :, 0]
                gaps.add_(delta[:, :, 1]).add_(delta[:, :, 2]).sqrt_()
                gaps -= radius_tensor[None, sphere_slice]
                minimum_gaps[point_slice] = torch.minimum(
                    minimum_gaps[point_slice], gaps.amin(dim=1)
                )
                del delta, gaps
        return minimum_gaps.cpu().numpy()


def _membership(points, centers, radii):
    # KD-tree radius queries compare squared distances; taking a square root
    # first can round an exterior point onto the boundary. Match that predicate.
    return _squared_distances(centers, points) <= radii[:, None].square()


def _scaled_membership(points, centers, radii):
    # Keep valid extreme radii from squaring to zero or infinity. This rare
    # path is chosen once from the NumPy input, without device synchronization.
    delta = (points[None, :, 0] - centers[:, None, 0]) / radii[:, None]
    squared = delta.square()
    for axis in (1, 2):
        delta = (points[None, :, axis] - centers[:, None, axis]) / radii[:, None]
        squared += delta.square()
    return squared <= 1


def greedy_cover(
    points,
    centers,
    radii,
    *,
    count: int,
    device: str = "cpu",
    max_matrix_bytes: int = 64 * 1024**2,
) -> np.ndarray:
    """Select at most ``count`` candidate balls by exact sampled set coverage.

    Ties select the lowest candidate index. No-gain candidates are not returned.
    Membership uses squared distances with explicit coordinate accumulation,
    matching KD-tree radius queries at ordinary float64 lattice boundaries.
    When a dense boolean coverage matrix and its reduction workspace fit the
    budget they are cached; otherwise coverage is recomputed in bounded tiles
    at every greedy step. Both modes implement the same greedy objective.

    ``max_matrix_bytes`` bounds planned working matrices, including boolean
    coverage/reduction temporaries and conservative distance-tile storage. It
    excludes the input tensors, the returned indices, and runtime overhead.
    """
    _positive_integer(count, "count")
    _positive_integer(max_matrix_bytes, "max_matrix_bytes")
    points, centers, radii = _inputs(points, centers, radii)
    torch = _runtime(device)
    point_count, candidate_count = len(points), len(centers)
    if not point_count or not candidate_count:
        return np.empty(0, dtype=np.int64)
    limits = np.finfo(np.float64)
    membership = (
        _scaled_membership
        if np.any((radii < math.sqrt(limits.tiny)) | (radii > math.sqrt(limits.max)))
        else _membership
    )
    # Linear state: covered/newly-covered masks, scores and reduction results.
    fixed_bytes = 3 * point_count + 24 * candidate_count
    if max_matrix_bytes < fixed_bytes + _PAIR_WORK_BYTES:
        raise ValueError(
            f"max_matrix_bytes needs at least {fixed_bytes + _PAIR_WORK_BYTES:,} bytes "
            "for this input"
        )
    matrix_bytes = point_count * candidate_count  # torch.bool is one byte.
    cache_coverage = (
        2 * matrix_bytes + fixed_bytes <= max_matrix_bytes
        and matrix_bytes + fixed_bytes + 10 * point_count <= max_matrix_bytes
    )
    working_bytes = max_matrix_bytes - fixed_bytes - (matrix_bytes if cache_coverage else 0)
    # Allow for offset, square, accumulation, and membership temporaries.
    max_tile_pairs = max(1, working_bytes // _PAIR_WORK_BYTES)
    candidate_block_size = min(candidate_count, max(1, math.isqrt(max_tile_pairs)))
    point_block_size = min(point_count, max(1, max_tile_pairs // candidate_block_size))
    # torch's bool.sum(dtype=int64) can materialize an int64 copy internally.
    # Bound that cast and the boolean mask too, not only the stored matrix.
    reduction_block_size = min(candidate_count, max(1, working_bytes // (10 * point_count)))
    selected_indices = []
    with torch.no_grad():
        point_tensor, center_tensor, radius_tensor = [
            torch.from_numpy(array).to(device=device) for array in (points, centers, radii)
        ]
        uncovered_points = torch.ones(point_count, dtype=torch.bool, device=device)
        available_candidates = torch.ones(candidate_count, dtype=torch.bool, device=device)
        coverage = None
        if cache_coverage:
            coverage = torch.empty((candidate_count, point_count), dtype=torch.bool, device=device)
            for candidate_start in range(0, candidate_count, candidate_block_size):
                candidate_slice = slice(candidate_start, candidate_start + candidate_block_size)
                for point_start in range(0, point_count, point_block_size):
                    point_slice = slice(point_start, point_start + point_block_size)
                    coverage[candidate_slice, point_slice] = membership(
                        point_tensor[point_slice],
                        center_tensor[candidate_slice],
                        radius_tensor[candidate_slice],
                    )
            coverage_gains = torch.empty(candidate_count, dtype=torch.int64, device=device)
            for candidate_start in range(0, candidate_count, reduction_block_size):
                candidate_slice = slice(candidate_start, candidate_start + reduction_block_size)
                coverage_gains[candidate_slice] = coverage[candidate_slice].sum(
                    dim=1, dtype=torch.int64
                )
        for _ in range(min(count, candidate_count)):
            if coverage is None:
                coverage_gains = torch.zeros(candidate_count, dtype=torch.int64, device=device)
                for candidate_start in range(0, candidate_count, candidate_block_size):
                    candidate_slice = slice(candidate_start, candidate_start + candidate_block_size)
                    for point_start in range(0, point_count, point_block_size):
                        point_slice = slice(point_start, point_start + point_block_size)
                        block = membership(
                            point_tensor[point_slice],
                            center_tensor[candidate_slice],
                            radius_tensor[candidate_slice],
                        )
                        block &= uncovered_points[None, point_slice]
                        coverage_gains[candidate_slice] += block.sum(dim=1, dtype=torch.int64)
                coverage_gains.masked_fill_(~available_candidates, -1)
            best_index = int(coverage_gains.argmax().item())
            if coverage_gains[best_index].item() <= 0:
                break
            selected_indices.append(best_index)
            available_candidates[best_index] = False
            if coverage is None:
                for point_start in range(0, point_count, point_block_size):
                    point_slice = slice(point_start, point_start + point_block_size)
                    block = membership(
                        point_tensor[point_slice],
                        center_tensor[best_index : best_index + 1],
                        radius_tensor[best_index : best_index + 1],
                    )[0]
                    uncovered_points[point_slice] &= ~block
            else:
                newly_covered = coverage[best_index] & uncovered_points
                for candidate_start in range(0, candidate_count, reduction_block_size):
                    candidate_slice = slice(candidate_start, candidate_start + reduction_block_size)
                    coverage_gains[candidate_slice] -= (
                        coverage[candidate_slice] & newly_covered[None, :]
                    ).sum(dim=1, dtype=torch.int64)
                uncovered_points &= ~newly_covered
                coverage_gains.masked_fill_(~available_candidates, -1)
    return np.asarray(selected_indices, dtype=np.int64)
