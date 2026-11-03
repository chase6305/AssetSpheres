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


def _positive_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _inputs(points, centers, radii) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = tuple(np.asarray(value, dtype=np.float64) for value in (points, centers, radii))
    p, c, r = arrays
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if c.ndim != 2 or c.shape[1] != 3:
        raise ValueError("centers must have shape (M, 3)")
    if r.shape != (len(c),):
        raise ValueError("radii must have shape (M,) matching centers")
    if any(not np.all(np.isfinite(a)) for a in arrays):
        raise ValueError("points, centers, and radii must be finite")
    if np.any(r <= 0):
        raise ValueError("radii must be positive")
    return tuple(np.array(a, order="C", copy=True) for a in arrays)


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
    The largest distance block is ``chunk_size * min(M, 128)`` pairs; each
    difference vector has three float64 coordinates. NumPy inputs/outputs and
    all CPU↔device transfers are part of this call.
    """
    _positive_integer(chunk_size, "chunk_size")
    p, c, r = _inputs(points, centers, radii)
    torch = _runtime(device)
    if not len(c):
        return np.full(len(p), np.inf)
    if not len(p):
        return np.empty(0, dtype=np.float64)
    with torch.no_grad():
        pt, ct, rt = [torch.from_numpy(a).to(device=device) for a in (p, c, r)]
        result = torch.full((len(p),), float("inf"), dtype=torch.float64, device=device)
        for first in range(0, len(p), chunk_size):
            stop = first + chunk_size
            for sphere in range(0, len(c), 128):
                delta = pt[first:stop, None, :] - ct[None, sphere : sphere + 128, :]
                gaps = torch.linalg.vector_norm(delta, dim=2) - rt[None, sphere : sphere + 128]
                result[first:stop] = torch.minimum(result[first:stop], gaps.amin(dim=1))
        return result.cpu().numpy()


def _membership(torch, points, centers, radii):
    delta = points[None, :, :] - centers[:, None, :]
    distance = torch.linalg.vector_norm(delta, dim=2)
    del delta
    return distance <= radii[:, None]


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
    When a dense boolean coverage matrix and its reduction workspace fit the
    budget they are cached; otherwise coverage is recomputed in bounded tiles
    at every greedy step. Both modes implement the same greedy objective.

    ``max_matrix_bytes`` bounds planned working matrices, including boolean
    coverage/reduction temporaries and conservative distance-tile storage. It
    excludes the input tensors, the returned indices, and runtime overhead.
    """
    _positive_integer(count, "count")
    _positive_integer(max_matrix_bytes, "max_matrix_bytes")
    p, c, r = _inputs(points, centers, radii)
    torch = _runtime(device)
    if not len(p) or not len(c):
        return np.empty(0, dtype=np.int64)
    # Linear state: covered/newly-covered masks, scores and reduction results.
    fixed_bytes = 3 * len(p) + 24 * len(c)
    if max_matrix_bytes < fixed_bytes + 40:
        raise ValueError(
            f"max_matrix_bytes needs at least {fixed_bytes + 40:,} bytes for this input"
        )
    matrix_bytes = len(p) * len(c)  # torch.bool is one byte.
    dense = (
        2 * matrix_bytes + fixed_bytes <= max_matrix_bytes
        and matrix_bytes + fixed_bytes + 10 * len(p) <= max_matrix_bytes
    )
    working_bytes = max_matrix_bytes - fixed_bytes - (matrix_bytes if dense else 0)
    # delta coordinates + distances + mask; 40 bytes/pair is conservative here.
    pairs = max(1, working_bytes // 40)
    rows = min(len(c), max(1, math.isqrt(pairs)))
    columns = min(len(p), max(1, pairs // rows))
    # torch's bool.sum(dtype=int64) can materialize an int64 copy internally.
    # Bound that cast and the boolean mask too, not only the stored matrix.
    reduction_rows = min(len(c), max(1, working_bytes // (10 * len(p))))
    selected = []
    with torch.no_grad():
        pt, ct, rt = [torch.from_numpy(a).to(device=device) for a in (p, c, r)]
        uncovered = torch.ones(len(p), dtype=torch.bool, device=device)
        available = torch.ones(len(c), dtype=torch.bool, device=device)
        coverage = None
        if dense:
            coverage = torch.empty((len(c), len(p)), dtype=torch.bool, device=device)
            for row in range(0, len(c), rows):
                for column in range(0, len(p), columns):
                    coverage[row : row + rows, column : column + columns] = _membership(
                        torch,
                        pt[column : column + columns],
                        ct[row : row + rows],
                        rt[row : row + rows],
                    )
            scores = torch.empty(len(c), dtype=torch.int64, device=device)
            for row in range(0, len(c), reduction_rows):
                scores[row : row + reduction_rows] = coverage[row : row + reduction_rows].sum(
                    dim=1, dtype=torch.int64
                )
        for _ in range(min(count, len(c))):
            if coverage is None:
                scores = torch.zeros(len(c), dtype=torch.int64, device=device)
                for row in range(0, len(c), rows):
                    for column in range(0, len(p), columns):
                        block = _membership(
                            torch,
                            pt[column : column + columns],
                            ct[row : row + rows],
                            rt[row : row + rows],
                        )
                        block &= uncovered[None, column : column + columns]
                        scores[row : row + rows] += block.sum(dim=1, dtype=torch.int64)
                scores.masked_fill_(~available, -1)
            best = int(scores.argmax().item())
            if scores[best].item() <= 0:
                break
            selected.append(best)
            available[best] = False
            if coverage is None:
                for column in range(0, len(p), columns):
                    block = _membership(
                        torch,
                        pt[column : column + columns],
                        ct[best : best + 1],
                        rt[best : best + 1],
                    )[0]
                    uncovered[column : column + columns] &= ~block
            else:
                newly = coverage[best] & uncovered
                for row in range(0, len(c), reduction_rows):
                    scores[row : row + reduction_rows] -= (
                        coverage[row : row + reduction_rows] & newly[None, :]
                    ).sum(dim=1, dtype=torch.int64)
                uncovered &= ~newly
                scores.masked_fill_(~available, -1)
    return np.asarray(selected, dtype=np.int64)
