"""Compare warm robot preview construction and GLB serialization.

Loads collision geometry and fits cover spheres once before timing. The previous
export module shares the current model and asset helpers; only make_scene differs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import trimesh

import assetspheres.export as exporting
from assetspheres import FitConfig, fit_asset, load_asset


def compare_scenes(expected, actual):
    assert set(expected.graph.nodes_geometry) == set(actual.graph.nodes_geometry)
    for node in expected.graph.nodes_geometry:
        left_transform, left_name = expected.graph[node]
        right_transform, right_name = actual.graph[node]
        assert np.array_equal(left_transform, right_transform), node
        left, right = expected.geometry[left_name], actual.geometry[right_name]
        assert np.array_equal(left.vertices, right.vertices), node
        assert np.array_equal(left.faces, right.faces), node
        assert np.array_equal(left.visual.face_colors, right.visual.face_colors), node


def geometry_digest(asset):
    digest = hashlib.sha256()
    for name, link in asset.links.items():
        digest.update(name.encode())
        digest.update(link.mesh.vertices.tobytes())
        digest.update(link.mesh.faces.tobytes())
        digest.update(asset.zero_pose_transforms[name].tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="previous export.py")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark-robot-preview.json"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if not args.reference.is_file():
        parser.error("--reference must point to a previous export.py")
    spec = importlib.util.spec_from_file_location("assetspheres._preview_reference", args.reference)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    functions = {"reference": reference.make_scene, "optimized": exporting.make_scene}
    asset = load_asset(args.input, geometry="collision")
    before = geometry_digest(asset)
    result = fit_asset(
        asset, FitConfig(method="cover", count=args.count, padding=0.003, samples=1024)
    )
    expected, actual = [function(asset, result) for function in functions.values()]
    compare_scenes(expected, actual)
    reloaded = [
        trimesh.load_scene(io.BytesIO(scene.export(file_type="glb")), file_type="glb")
        for scene in (expected, actual)
    ]
    compare_scenes(*reloaded)
    records = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {
            name: importlib.metadata.version(name) for name in ("numpy", "scipy", "trimesh")
        },
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "threads": {
            name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
        },
        "source_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in (
                ("reference", args.reference),
                ("optimized", Path(exporting.__file__)),
            )
        },
        "input": str(args.input.resolve()),
        "input_geometry_sha256": before,
        "links": len(asset.links),
        "spheres": sum(map(len, result.links.values())),
        "config": result.to_dict()["config"],
        "timing_scope": "warm construction and optional GLB serialization; no loading/fitting/writing",
        "exact_scenes_and_reloaded_glbs": True,
        "repeats": args.repeats,
        "cases": {},
    }
    for scope in ("construction", "construction_and_glb"):
        times = {name: [] for name in functions}
        for repeat in range(args.repeats):
            for name in list(functions)[:: 1 if repeat % 2 == 0 else -1]:
                started = time.perf_counter()
                scene = functions[name](asset, result)
                if scope == "construction_and_glb":
                    scene.export(file_type="glb")
                times[name].append(time.perf_counter() - started)
        records["cases"][scope] = {
            name: {"times_s": values, "median_s": statistics.median(values)}
            for name, values in times.items()
        }
    assert geometry_digest(asset) == before
    records["source_geometry_unchanged"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
