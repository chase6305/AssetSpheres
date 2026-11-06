"""Compare complete CPU process fits with a previous src/assetspheres directory.

Each paired observation runs in a fresh parent process. Its timer includes task
dispatch, spawn, worker imports, fitting, reports and pool shutdown. Parent imports,
input creation/loading, a serial reference and output checks are outside the timer.
There is no pool reuse. Serialized task sizes are estimates, not pipe traffic or RSS.
Texture fixtures need Pillow, provided by the plots extra.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

CASES = ("bare", "cached", "attributes", "texture512", "texture4096", "robot")


def fingerprint(result):
    if hasattr(result, "to_dict"):
        payload = {"link_order": list(result.links), "result": result.to_dict()}
    else:
        payload = [
            (name, [sphere.to_dict() for sphere in spheres], report)
            for name, spheres, report in result
        ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def geometry_digest(tasks):
    digest = hashlib.sha256()
    for name, mesh, _ in tasks:
        digest.update(name.encode())
        for array in (mesh.vertices, mesh.faces):
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        digest.update(pickle.dumps(mesh.metadata, protocol=pickle.HIGHEST_PROTOCOL))
    return digest.hexdigest()


def fixture(args):
    import numpy as np
    import trimesh

    from assetspheres import FitConfig, load_asset

    configuration = FitConfig(method="cover", count=args.count, samples=args.samples, padding=0.003)
    if args.run_case == "robot":
        asset = load_asset(args.input, geometry=args.geometry)
        tasks = [(name, link.mesh, configuration) for name, link in asset.links.items()]
        return asset, tasks, configuration
    tasks = []
    for index in range(4):
        level = 4 if args.run_case in {"cached", "attributes"} else 2
        mesh = trimesh.creation.icosphere(subdivisions=level, radius=0.1 + index * 0.01)
        if args.run_case == "cached":
            _ = mesh.area, mesh.edges_unique, mesh.is_watertight, mesh.triangles
        if args.run_case == "attributes":
            mesh.vertex_attributes["features"] = np.ones(
                (len(mesh.vertices), 128), dtype=np.float32
            )
        if args.run_case.startswith("texture"):
            from PIL import Image

            width = int(args.run_case.removeprefix("texture"))
            material = trimesh.visual.material.PBRMaterial(
                baseColorTexture=Image.new("RGBA", (width, width), (60, 120, 180, 255))
            )
            mesh.visual = trimesh.visual.texture.TextureVisuals(
                uv=np.zeros((len(mesh.vertices), 2)), material=material
            )
        mesh.metadata["fixture"] = {"index": index, "label": "geometry transport"}
        tasks.append((f"part-{index}", mesh, configuration))
    return None, tasks, configuration


def captured_task_bytes(call):
    """Inspect the actual dispatch path with a recording pool, outside measurement."""
    from multiprocessing.reduction import ForkingPickler
    from unittest.mock import patch

    import assetspheres.parallel as parallel

    captured = []

    class RecordingPool:
        def __init__(self, **kwargs):
            assert kwargs["mp_context"].get_start_method() == "spawn"

        def map(self, function, tasks):
            captured.extend(tasks)
            return iter(())

        def shutdown(self, **kwargs):
            pass

    with patch.object(parallel, "ProcessPoolExecutor", RecordingPool):
        call()
    assert captured, "The case must create a process pool"
    return sum(len(ForkingPickler.dumps(task)) for task in captured)


def run_case(args):
    import assetspheres
    from assetspheres import fit_asset
    from assetspheres.parallel import fit_links

    assert Path(assetspheres.__file__).resolve() == args.package.resolve() / "__init__.py"
    asset, tasks, configuration = fixture(args)
    if len(tasks) < 2:
        raise ValueError("A process case needs at least two loaded links")
    before = geometry_digest(tasks)
    source_task_bytes = sum(
        len(pickle.dumps(task, protocol=pickle.DEFAULT_PROTOCOL)) for task in tasks
    )
    # A separate copy keeps reference fitting from warming the measured input's caches.
    if asset is None:
        reference_tasks = pickle.loads(pickle.dumps(tasks, protocol=pickle.HIGHEST_PROTOCOL))
        expected = fingerprint(fit_links(reference_tasks))
        del reference_tasks

        def call():
            return fit_links(tasks, workers=args.workers, executor="process")

    else:
        reference_asset = pickle.loads(pickle.dumps(asset, protocol=pickle.HIGHEST_PROTOCOL))
        expected = fingerprint(fit_asset(reference_asset, configuration))
        del reference_asset

        def call():
            return fit_asset(asset, configuration, workers=args.workers, executor="process")

    task_bytes = captured_task_bytes(call)
    gc.collect()
    started = time.perf_counter()
    result = call()
    elapsed = time.perf_counter() - started
    assert fingerprint(result) == expected
    assert geometry_digest(tasks) == before
    record = {
        "package": str(Path(assetspheres.__file__).resolve()),
        "elapsed_s": elapsed,
        "links": len(tasks),
        "workers_effective": min(args.workers, len(tasks)),
        "vertices": sum(len(mesh.vertices) for _, mesh, _ in tasks),
        "triangles": sum(len(mesh.faces) for _, mesh, _ in tasks),
        "source_geometry_sha256": before,
        "source_geometry_unchanged": True,
        "source_task_pickle_estimate_bytes": source_task_bytes,
        "submitted_task_pickle_estimate_bytes": task_bytes,
        "full_results_sha256": expected,
        "exact_serial_result": True,
    }
    args.record.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def environment():
    packages = {}
    for name in ("numpy", "scipy", "trimesh", "Pillow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": packages,
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "threads": {
            name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, help="previous src/assetspheres directory")
    parser.add_argument("--case", action="append", choices=CASES)
    parser.add_argument("--input", type=Path, help="local multi-link URDF for the robot case")
    parser.add_argument("--geometry", choices=("collision", "visual"), default="collision")
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/benchmark-process-transport.json")
    )
    parser.add_argument("--run-case", choices=CASES, help=argparse.SUPPRESS)
    parser.add_argument("--package", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--record", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.workers < 2 or args.repeats < 1:
        parser.error("workers must be at least two; repeats must be positive")
    if args.run_case:
        if args.package is None or args.record is None:
            parser.error("internal case mode needs a package and record path")
        run_case(args)
        return
    if args.reference is None or not (args.reference / "parallel.py").is_file():
        parser.error("--reference must contain a previous parallel.py")
    cases = args.case or [*CASES[:-1], *(["robot"] if args.input else [])]
    if "robot" in cases and args.input is None:
        parser.error("the robot case needs --input")
    current = Path(__file__).resolve().parents[1] / "src/assetspheres"
    packages = {"reference": args.reference.resolve(), "optimized": current}
    records = {
        **environment(),
        "source_sha256": {
            name: {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(package.glob("*.py"))
            }
            for name, package in packages.items()
        },
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "method": "cover",
        "count": args.count,
        "samples": args.samples,
        "padding_m": 0.003,
        "workers": args.workers,
        "repeats": args.repeats,
        "input": str(args.input.resolve()) if args.input else None,
        "geometry": args.geometry,
        "timing_scope": (
            "fit_links/fit_asset including spawn, serialization, fit, reports and shutdown"
        ),
        "excluded": "parent imports, load/create inputs, serial reference, GC and output checks",
        "payload_scope": (
            "ForkingPickler task estimates; excludes function/message framing; not RSS"
        ),
        "cases": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="assetspheres-transport-") as directory:
        work = Path(directory)
        for case in dict.fromkeys(cases):
            expected = None
            rows = {name: [] for name in packages}
            for repeat in range(args.repeats):
                for name in list(packages)[:: 1 if repeat % 2 == 0 else -1]:
                    package = packages[name]
                    child_record = work / f"{case}-{name}-{repeat}.json"
                    arguments = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--run-case",
                        case,
                        "--package",
                        str(package),
                        "--record",
                        str(child_record),
                        "--workers",
                        str(args.workers),
                        "--count",
                        str(args.count),
                        "--samples",
                        str(args.samples),
                        "--geometry",
                        args.geometry,
                    ]
                    if args.input:
                        arguments.extend(("--input", str(args.input.resolve())))
                    child_environment = dict(os.environ, PYTHONPATH=str(package.parent))
                    response = subprocess.run(
                        arguments,
                        cwd=work,
                        env=child_environment,
                        capture_output=True,
                        text=True,
                        timeout=180,
                    )
                    if response.returncode:
                        raise RuntimeError(f"{case}/{name}: {response.stdout}\n{response.stderr}")
                    row = json.loads(child_record.read_text())
                    if expected is None:
                        expected = row["full_results_sha256"]
                    assert row["full_results_sha256"] == expected, (case, name, repeat)
                    rows[name].append(row)
            records["cases"][case] = {
                "exact_full_results": True,
                "full_results_sha256": expected,
                **{
                    name: {
                        "median_s": statistics.median(row["elapsed_s"] for row in samples),
                        "observations": samples,
                    }
                    for name, samples in rows.items()
                },
            }
            args.output.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
            print(
                case,
                {
                    name: round(statistics.median(row["elapsed_s"] for row in samples), 4)
                    for name, samples in rows.items()
                },
                flush=True,
            )
    print(args.output)


if __name__ == "__main__":
    main()
