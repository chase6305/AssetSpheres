"""Ordered, opt-in parallel fitting of independent robot links.

The process executor uses ``spawn`` so it never forks a parent's CUDA runtime.
Use a single GPU owner for CUDA work; this module is intended for CPU link jobs.
"""

from __future__ import annotations

import multiprocessing
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import trimesh

    from .model import FitConfig, Sphere

    FitTask = tuple[str, trimesh.Trimesh, FitConfig]
    LinkResult = tuple[str, list[Sphere], dict[str, Any]]


def _fit_link(task: FitTask) -> LinkResult:
    # Keep this worker at module scope for spawn pickling, and defer imports to
    # avoid an import cycle when fit_asset delegates to fit_links.
    from .fit import fit_mesh

    name, mesh, config = task
    try:
        spheres, report = fit_mesh(mesh, config)
    except Exception as exc:
        raise ValueError(f"Link {name!r}: {exc}") from exc
    return name, spheres, report


def fit_links(
    tasks: Iterable[FitTask],
    *,
    workers: int = 1,
    executor: str = "process",
) -> list[LinkResult]:
    """Fit independent link meshes and return results in input order.

    ``workers=1`` runs in the calling thread and does not start a pool. More
    workers must be requested explicitly; the pool is capped by the link count.
    Threads avoid mesh serialization but only overlap stages that release the
    GIL. Spawned processes can overlap Python work at the cost of worker import,
    mesh pickling, and independent per-process memory. No pool is retained across
    calls. Applications invoking process mode must use a normal importable entry
    point guarded by ``if __name__ == '__main__':``.

    Each task is ``(link_name, mesh_in_link_frame, FitConfig)``. A worker failure
    raises ``ValueError`` containing the link name. In serial/thread mode its
    cause is the original exception; process mode preserves a remote traceback.
    Input meshes are not modified by ``fit_mesh``.
    """
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if executor not in {"process", "thread"}:
        raise ValueError("executor must be 'process' or 'thread'")
    prepared = list(tasks)
    if not prepared:
        return []
    names = [task[0] for task in prepared]
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Link task names must be nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError("Link task names must be unique")
    workers = min(workers, len(prepared))
    if workers == 1:
        return [_fit_link(task) for task in prepared]
    if any(config.device == "cuda" for _, _, config in prepared):
        raise ValueError("CUDA fitting uses one GPU owner; use workers=1")
    if executor == "thread":
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="assetspheres")
    else:
        pool = ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        )
    try:
        # Executor.map yields in input order, independently of completion order.
        return list(pool.map(_fit_link, prepared))
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
