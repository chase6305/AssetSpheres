"""Compare preview construction and GLB serialization with a previous export.py.

Synthetic inputs, textures and sphere parameters are prepared before timing.
Requires Pillow (included with the plots extra). No fitting or asset loading is timed.
"""

from __future__ import annotations

import argparse
import copy
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
from unittest.mock import patch


def compare_scenes(expected, actual, np):
    assert set(expected.graph.nodes_geometry) == set(actual.graph.nodes_geometry)
    for node in expected.graph.nodes_geometry:
        expected_transform, expected_name = expected.graph[node]
        actual_transform, actual_name = actual.graph[node]
        assert np.array_equal(expected_transform, actual_transform), node
        left, right = expected.geometry[expected_name], actual.geometry[actual_name]
        for attribute in ("vertices", "faces"):
            assert np.array_equal(getattr(left, attribute), getattr(right, attribute)), node
        assert np.array_equal(left.visual.face_colors, right.visual.face_colors), node
        for attribute in ("vertex_attributes", "face_attributes"):
            left_values, right_values = getattr(left, attribute), getattr(right, attribute)
            assert left_values.keys() == right_values.keys(), node
            for name in left_values:
                assert np.array_equal(left_values[name], right_values[name]), (node, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="previous export.py")
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark-preview.json"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if not args.reference.is_file():
        parser.error("--reference must point to a previous export.py")

    import numpy as np
    import trimesh
    from PIL import Image

    import assetspheres.export as exporting
    from assetspheres import Asset, FitConfig, FitResult, LinkGeometry, Sphere

    spec = importlib.util.spec_from_file_location("assetspheres._preview_reference", args.reference)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    functions = {"reference": reference.make_scene, "optimized": exporting.make_scene}
    records = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {
            name: importlib.metadata.version(name) for name in ("numpy", "trimesh", "Pillow")
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
        "timing_scope": "warm preloaded synthetic inputs; no fitting or loading",
        "copy_count_scope": (
            "PIL.Image.Image.copy calls only; SimpleMaterial deepcopy uses image serialization"
        ),
        "fixture": {"vertices": 642, "faces": 1280, "spheres": 24, "vertex_feature_columns": 3},
        "repeats": args.repeats,
        "cases": {},
    }
    rng = np.random.default_rng(591101)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for width in (0, 512, 2048, 4096):
        for material_kind in ("none",) if not width else ("simple", "pbr"):
            mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.1)
            image = None
            if width:
                image = Image.new("RGBA", (width, width), (60, 120, 180, 255))
                material = (
                    trimesh.visual.material.PBRMaterial(baseColorTexture=image)
                    if material_kind == "pbr"
                    else trimesh.visual.material.SimpleMaterial(image=image)
                )
                mesh.visual = trimesh.visual.texture.TextureVisuals(
                    uv=rng.random((len(mesh.vertices), 2)), material=material
                )
            mesh.vertex_attributes["_FEATURES"] = rng.normal(size=(len(mesh.vertices), 3)).astype(
                np.float32
            )
            mesh.face_attributes["label"] = np.arange(len(mesh.faces))
            source_vertices = mesh.vertices.copy()
            source_triangles = mesh.faces.copy()
            source_attributes = copy.deepcopy(mesh.vertex_attributes)
            source_faces = copy.deepcopy(mesh.face_attributes)
            image_digest = hashlib.sha256(image.tobytes()).digest() if image is not None else None
            transform = trimesh.transformations.euler_matrix(0.2, -0.4, 0.7)
            transform[:3, 3] = [0.1, 0.2, 0.3]
            asset = Asset(
                Path("synthetic.glb"), {"part": LinkGeometry("part", mesh)}, {"part": transform}
            )
            result = FitResult(
                "synthetic.glb",
                FitConfig(method="cover"),
                {"part": [Sphere((index * 0.001, 0, 0), 0.01) for index in range(24)]},
                {},
                transforms={"part": transform.tolist()},
            )
            expected, actual = [function(asset, result) for function in functions.values()]
            compare_scenes(expected, actual, np)
            payloads = [scene.export(file_type="glb") for scene in (expected, actual)]
            assert payloads[0] == payloads[1], "GLB bytes differ"
            # Read back the GLBs as well, checking the consumer's view of the scene.
            reloaded = [
                trimesh.load_scene(io.BytesIO(payload), file_type="glb") for payload in payloads
            ]
            compare_scenes(*reloaded, np)
            for scene in (expected, actual):
                _, name = scene.graph["mesh/0"]
                duplicate = scene.geometry[name]
                assert not np.shares_memory(duplicate.vertices, mesh.vertices)
                assert not np.shares_memory(
                    duplicate.vertex_attributes["_FEATURES"], mesh.vertex_attributes["_FEATURES"]
                )
                assert not np.shares_memory(
                    duplicate.face_attributes["label"], mesh.face_attributes["label"]
                )
            copies = {}
            original_copy = Image.Image.copy
            for name, function in functions.items():
                count = [0]

                def tracked_copy(source, *arguments, **keywords):
                    count[0] += 1
                    return original_copy(source, *arguments, **keywords)

                with patch.object(Image.Image, "copy", tracked_copy):
                    function(asset, result)
                copies[name] = count[0]
            timings = {}
            for scope in ("construction", "construction_and_glb"):
                times = {name: [] for name in functions}
                for repeat in range(args.repeats):
                    for name in list(functions)[:: 1 if repeat % 2 == 0 else -1]:
                        started = time.perf_counter()
                        scene = functions[name](asset, result)
                        if scope == "construction_and_glb":
                            scene.export(file_type="glb")
                        times[name].append(time.perf_counter() - started)
                timings[scope] = {
                    name: {"times_s": values, "median_s": statistics.median(values)}
                    for name, values in times.items()
                }
            assert np.array_equal(mesh.vertices, source_vertices)
            assert np.array_equal(mesh.faces, source_triangles)
            for name, values in source_attributes.items():
                assert np.array_equal(mesh.vertex_attributes[name], values)
            for name, values in source_faces.items():
                assert np.array_equal(mesh.face_attributes[name], values)
            if image is not None:
                assert hashlib.sha256(image.tobytes()).digest() == image_digest
            label = f"{material_kind}/image{width}"
            records["cases"][label] = {
                "exact_scenes_and_reloaded_glbs": True,
                "exact_glb_bytes": True,
                "glb_sha256": hashlib.sha256(payloads[0]).hexdigest(),
                "source_unchanged": True,
                "independent_geometry_attributes": True,
                "image_copy_calls": copies,
                **timings,
            }
            args.output.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
            print(label, copies, flush=True)
    print(args.output)


if __name__ == "__main__":
    main()
