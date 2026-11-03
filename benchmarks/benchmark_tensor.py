"""Compare optional tensor operations with the existing CPU formulations.

All operation timings include NumPy validation/copies and device transfers.
CUDA is synchronized before stopping each timer. The two stage benchmarks exclude
mesh loading, voxelization and EDT; --full-fit additionally times complete voxel
fit_mesh calls, including voxelization, EDT, greedy selection and quality metrics.

Run from the repository root:
    .venv/bin/python benchmarks/benchmark_tensor.py --full-fit \
        --output outputs/benchmark-tensor.json
"""

from __future__ import annotations

import argparse
import heapq
import importlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from assetspheres import FitConfig, Sphere, fit_mesh
from assetspheres.fit import sphere_gaps
from assetspheres.tensor_backend import greedy_cover, point_sphere_gaps


def data(n_points: int, n_spheres: int):
    rng = np.random.default_rng(12)
    return (
        rng.random((n_points, 3)),
        rng.random((n_spheres, 3)),
        rng.uniform(0.06, 0.20, n_spheres),
    )


def scipy_greedy(points, centers, radii, count):
    tree = cKDTree(points)
    neighbors = [np.asarray(tree.query_ball_point(c, r), dtype=int) for c, r in zip(centers, radii)]
    covered = np.zeros(len(points), dtype=bool)
    heap = [(-len(ids), i) for i, ids in enumerate(neighbors)]
    heapq.heapify(heap)
    selected = []
    while heap and len(selected) < count:
        _, i = heapq.heappop(heap)
        score = int(np.count_nonzero(~covered[neighbors[i]]))
        if score == 0:
            continue
        # Include the tie-breaker in the lazy comparison, matching lowest-index ties.
        if heap and (-score, i) > heap[0]:
            heapq.heappush(heap, (-score, i))
            continue
        selected.append(i)
        covered[neighbors[i]] = True
    return np.asarray(selected)


def timed(call, *, device: str, torch):
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    value = call()
    if device == "cuda":
        torch.cuda.synchronize()
    return value, (time.perf_counter() - start) * 1000


def measurements(call, *, device: str, torch, repeat: int):
    result, first = timed(call, device=device, torch=torch)
    values = [timed(call, device=device, torch=torch)[1] for _ in range(repeat)]
    return result, {
        "first_call_ms": first,
        "hot_median_ms": statistics.median(values),
        "hot_min_ms": min(values),
    }


def full_fit(device: str, resolution: int):
    import trimesh

    mesh = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    return fit_mesh(
        mesh,
        FitConfig(
            method="voxel",
            device=device,
            resolution=resolution,
            count=16 if resolution <= 24 else 32,
            padding=0.002,
        ),
    )


def cold_probe(device: str, fit_resolution: int | None = None) -> None:
    # The parent times this whole fresh process, including Python/import/context startup.
    if fit_resolution is not None:
        full_fit(device, fit_resolution)
    else:
        points, centers, radii = data(6000, 32)
        point_sphere_gaps(points, centers, radii, device=device)
    if device == "cuda":
        sys.modules["torch"].cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--cold-probe", choices=["cpu", "cuda"])
    parser.add_argument("--cold-fit-resolution", type=int)
    parser.add_argument(
        "--full-fit", action="store_true", help="Also benchmark complete voxel fits"
    )
    parser.add_argument("--quick", action="store_true", help="Skip the larger stage benchmarks")
    args = parser.parse_args()
    if args.cold_probe:
        cold_probe(args.cold_probe, args.cold_fit_resolution)
        return
    if args.repeat < 1 or args.cpu_threads < 1:
        parser.error("repeat and cpu-threads must be positive")
    start = time.perf_counter()
    torch = importlib.import_module("torch")
    import_ms = (time.perf_counter() - start) * 1000
    # This is a standalone benchmark process, not a library-level global thread setting.
    torch.set_num_threads(args.cpu_threads)
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    output = {
        "scope": "stage timings include transfers; optional fits include full voxel fit_mesh",
        "dtype": "float64",
        "torch_version": torch.__version__,
        "torch_import_ms": import_ms,
        "torch_cpu_threads": torch.get_num_threads(),
        "cuda_device": torch.cuda.get_device_name(0) if "cuda" in devices else None,
        "cold_process_ms": {},
        "gaps": [],
        "greedy": [],
    }
    for device in devices:
        start = time.perf_counter()
        process = subprocess.run(
            [sys.executable, __file__, "--cold-probe", device], capture_output=True, text=True
        )
        if process.returncode:
            raise RuntimeError(process.stderr)
        output["cold_process_ms"][device] = (time.perf_counter() - start) * 1000
    gap_cases = [(6000, 32)] + ([] if args.quick else [(50_000, 128)])
    for n_points, n_spheres in gap_cases:
        points, centers, radii = data(n_points, n_spheres)
        spheres = [Sphere(tuple(c), float(r)) for c, r in zip(centers, radii)]
        reference, baseline = measurements(
            lambda: sphere_gaps(points, spheres), device="cpu", torch=torch, repeat=args.repeat
        )
        row = {"points": n_points, "spheres": n_spheres, "numpy": baseline}
        for device in devices:
            actual, row[device] = measurements(
                lambda device=device: point_sphere_gaps(points, centers, radii, device=device),
                device=device,
                torch=torch,
                repeat=args.repeat,
            )
            np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-12)
        if "cuda" in devices:
            _, h2d = timed(
                lambda: [torch.from_numpy(a).to("cuda") for a in (points, centers, radii)],
                device="cuda",
                torch=torch,
            )
            sample = torch.empty(n_points, dtype=torch.float64, device="cuda")
            _, d2h = timed(lambda: sample.cpu().numpy(), device="cuda", torch=torch)
            del sample
            row["standalone_transfer_ms"] = {"H2D": h2d, "D2H": d2h}
        output["gaps"].append(row)
    cover_cases = [(6000, 768, 32)] + ([] if args.quick else [(24_000, 1024, 48)])
    for n_points, n_spheres, count in cover_cases:
        points, centers, radii = data(n_points, n_spheres)
        reference, baseline = measurements(
            lambda: scipy_greedy(points, centers, radii, count),
            device="cpu",
            torch=torch,
            repeat=args.repeat,
        )
        row = {
            "points": n_points,
            "candidates": n_spheres,
            "count": count,
            "scipy_lazy": baseline,
            "matrix_budget_bytes": 128 * 1024**2,
        }
        for device in devices:
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                before = torch.cuda.memory_allocated()
            actual, row[device] = measurements(
                lambda device=device: greedy_cover(
                    points,
                    centers,
                    radii,
                    count=count,
                    device=device,
                    max_matrix_bytes=128 * 1024**2,
                ),
                device=device,
                torch=torch,
                repeat=args.repeat,
            )
            assert actual.tolist() == reference.tolist(), (actual, reference)
            if device == "cuda":
                row[device]["extra_peak_allocated_bytes"] = (
                    torch.cuda.max_memory_allocated() - before
                )
        output["greedy"].append(row)
    if args.full_fit:
        output["fits"] = []
        for resolution in (24, 80):
            row = {
                "shape": "box",
                "extents_m": [0.2, 0.2, 0.2],
                "resolution": resolution,
                "count": 16 if resolution <= 24 else 32,
                "cold_process_ms": {},
            }
            reference = None
            for device in devices:
                process_start = time.perf_counter()
                process = subprocess.run(
                    [
                        sys.executable,
                        __file__,
                        "--cold-probe",
                        device,
                        "--cold-fit-resolution",
                        str(resolution),
                    ],
                    capture_output=True,
                    text=True,
                )
                if process.returncode:
                    raise RuntimeError(process.stderr)
                row["cold_process_ms"][device] = (time.perf_counter() - process_start) * 1000
                result, row[device] = measurements(
                    lambda device=device: full_fit(device, resolution),
                    device=device,
                    torch=torch,
                    repeat=args.repeat,
                )
                spheres, report = result
                row["voxel_count"] = report["voxel_count"]
                if reference is None:
                    reference = spheres
                else:
                    row["cpu_cuda_spheres_identical"] = spheres == reference
                    if spheres != reference:
                        raise AssertionError("CPU/CUDA voxel sphere selection differs")
                row[device]["surface_coverage"] = report["surface_coverage"]
                row[device]["voxel_center_coverage"] = report["voxel_center_coverage"]
            output["fits"].append(row)
    serialized = json.dumps(output, indent=2)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
