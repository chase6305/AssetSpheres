"""Audit complete fits against a frozen reference package, one asset at a time.

Example:
    python benchmarks/validate_workspaces.py \
        --reference /path/to/reference/src/assetspheres \
        --input /path/to/robot.urdf --geometry collision \
        --output outputs/workspace-validation.json

Launch through the workspace's shared resource gate and lock. Set
OPENBLAS_NUM_THREADS, OMP_NUM_THREADS, MKL_NUM_THREADS and NUMEXPR_NUM_THREADS
to 1 before starting Python, and wait when CPU or memory headroom is insufficient.
Keep both source trees frozen during the audit. Each invocation runs serial CPU
fits for one asset/geometry; it does not create a worker pool.

The defaults compare six methods, each with (count=8, seed=0) and
(count=24, seed=17), resolution=32, samples=1024, refine_samples=512 and
iterations=30. Open meshes explicitly fall back to cover. These are bounded
correctness checks of complete spheres and reports, including matching rejected
inputs. The script measures neither time nor memory and makes no performance claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import scipy
import trimesh

import assetspheres
from assetspheres import FitConfig, fit_mesh, load_asset

METHODS = ("surface", "cover", "voxel", "refined", "refined-cover", "slab")
COUNT_SEED_PAIRS = ((8, 0), (24, 17))


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest(package_directory: Path) -> dict[str, str]:
    return {
        path.relative_to(package_directory).as_posix(): file_digest(path)
        for path in sorted(package_directory.rglob("*.py"))
    }


def mesh_digest(mesh: trimesh.Trimesh) -> str:
    digest = hashlib.sha256()
    digest.update(mesh.vertices.tobytes())
    digest.update(mesh.faces.tobytes())
    return digest.hexdigest()


def load_reference(package_directory: Path) -> ModuleType:
    package_name = "_assetspheres_workspaces_reference"
    specification = importlib.util.spec_from_file_location(
        package_name,
        package_directory / "__init__.py",
        submodule_search_locations=[str(package_directory)],
    )
    if specification is None or specification.loader is None:
        raise ValueError(f"Cannot load the reference package: {package_directory}")
    reference = importlib.util.module_from_spec(specification)
    sys.modules[package_name] = reference
    specification.loader.exec_module(reference)
    return reference


def fit_outcome(fit_function, mesh: trimesh.Trimesh, configuration) -> dict[str, Any]:
    try:
        spheres, report = fit_function(mesh, configuration)
    except ValueError as error:
        return {"kind": "rejection", "type": type(error).__name__, "message": str(error)}
    # Serialization failures are audit errors, not rejected fitting inputs.
    payload = {"spheres": [sphere.to_dict() for sphere in spheres], "report": report}
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")
    return {
        "kind": "success",
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "effective_method": report["effective_method"],
        "voxel_count": report.get("voxel_count"),
        "sphere_count": len(spheres),
    }


def write_record(path: Path, record: dict[str, Any]) -> None:
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", type=Path, required=True, help="Frozen assetspheres package directory"
    )
    parser.add_argument("--input", type=Path, required=True, help="URDF or standalone mesh")
    parser.add_argument("--geometry", choices=("auto", "collision", "visual"), default="collision")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--output", type=Path, required=True, help="Correctness audit JSON")
    arguments = parser.parse_args()
    reference_directory = arguments.reference.expanduser().resolve()
    input_path = arguments.input.expanduser().resolve()
    output_path = arguments.output.expanduser().resolve()
    current_directory = Path(assetspheres.__file__).resolve().parent
    if not (reference_directory / "__init__.py").is_file():
        parser.error("--reference must be an assetspheres package directory with __init__.py")
    if reference_directory == current_directory:
        parser.error("--reference must be an independent previous package directory")
    if not input_path.is_file():
        parser.error("--input must be an existing asset file")
    if output_path == input_path or any(
        output_path in directory.rglob("*.py")
        for directory in (reference_directory, current_directory)
    ):
        parser.error("--output must not overwrite the input asset or audited Python source")

    source_manifests_before = {
        "reference": source_manifest(reference_directory),
        "current": source_manifest(current_directory),
    }
    input_digest_before = file_digest(input_path)
    reference = load_reference(reference_directory)
    asset = load_asset(input_path, geometry=arguments.geometry)
    geometry_manifest_before = {name: mesh_digest(link.mesh) for name, link in asset.links.items()}
    rows: list[dict[str, Any]] = []
    record = {
        "scope": "bounded complete-fit correctness audit; no performance measurements",
        "python": platform.python_version(),
        "dependencies": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "trimesh": trimesh.__version__,
        },
        "input": str(input_path),
        "geometry": arguments.geometry,
        "methods": arguments.methods,
        "reference_package": str(reference_directory),
        "current_package": str(current_directory),
        "audit_script_sha256": file_digest(Path(__file__)),
        "source_manifests_before": source_manifests_before,
        "geometry_manifest_before": geometry_manifest_before,
        "input_sha256_before": input_digest_before,
        "threads": {
            name: os.environ.get(name)
            for name in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "rows": rows,
        "completed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_record(output_path, record)
    try:
        for link_name, link in asset.links.items():
            for method in arguments.methods:
                for count, seed in COUNT_SEED_PAIRS:
                    configuration = FitConfig(
                        method=method,
                        count=count,
                        seed=seed,
                        resolution=32,
                        samples=1024,
                        refine_samples=512,
                        iterations=30,
                        non_watertight="cover",
                    )
                    configuration_values = asdict(configuration)
                    outcomes = []
                    for fit_function, fit_configuration in (
                        (reference.fit_mesh, reference.FitConfig(**configuration_values)),
                        (fit_mesh, configuration),
                    ):
                        outcomes.append(fit_outcome(fit_function, link.mesh, fit_configuration))
                        assert mesh_digest(link.mesh) == geometry_manifest_before[link_name], (
                            "Input geometry changed",
                            link_name,
                            method,
                            count,
                        )
                    matched = outcomes[0] == outcomes[1]
                    row = {
                        "link": link_name,
                        "mesh_sha256": geometry_manifest_before[link_name],
                        "configuration": configuration_values,
                        "exact_match": matched,
                        "result": outcomes[1],
                    }
                    if not matched:
                        row["reference_result"] = outcomes[0]
                    rows.append(row)
                    write_record(output_path, record)
                    assert matched, (link_name, method, count, seed, outcomes)
                    print(
                        input_path.parent.name,
                        arguments.geometry,
                        link_name,
                        method,
                        count,
                        outcomes[1]["kind"],
                        flush=True,
                    )
        record["completed"] = True
    finally:
        record["case_count"] = len(rows)
        record["source_manifests_after"] = {
            "reference": source_manifest(reference_directory),
            "current": source_manifest(current_directory),
        }
        record["geometry_manifest_after"] = {
            name: mesh_digest(link.mesh) for name, link in asset.links.items()
        }
        record["input_sha256_after"] = file_digest(input_path)
        record["source_manifests_unchanged"] = (
            record["source_manifests_after"] == source_manifests_before
        )
        record["input_geometry_unchanged"] = (
            record["geometry_manifest_after"] == geometry_manifest_before
        )
        record["input_file_unchanged"] = record["input_sha256_after"] == input_digest_before
        manifests_unchanged = all(
            record[key]
            for key in (
                "source_manifests_unchanged",
                "input_geometry_unchanged",
                "input_file_unchanged",
            )
        )
        record["completed"] = record["completed"] and manifests_unchanged
        write_record(output_path, record)
    assert manifests_unchanged, "Input geometry, asset file or audited source changed during audit"


if __name__ == "__main__":
    main()
