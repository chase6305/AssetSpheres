"""Compare fresh interpreter startup against a previous src/assetspheres directory.

Each timed call starts a new process. Filesystem and bytecode caches are warm;
times include interpreter startup and output capture. Paired outputs and exit
statuses must match, including the invalid-argument case.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark-startup.json"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if not (args.reference / "__init__.py").is_file():
        parser.error("--reference must point to a previous src/assetspheres directory")
    current = Path(__file__).resolve().parents[1] / "src/assetspheres"
    packages = {"reference": args.reference.resolve(), "optimized": current}
    environments = {}
    for name, package in packages.items():
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(package.parent)
        environments[name] = environment
    files = ("__init__.py", "cli.py", "export.py", "model.py", "result_io.py")
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
            name: {
                filename: hashlib.sha256((package / filename).read_bytes()).hexdigest()
                if (package / filename).is_file()
                else None
                for filename in files
            }
            for name, package in packages.items()
        },
        "timing_scope": "new process per call, filesystem/bytecode caches warm; output captured",
        "repeats": args.repeats,
        "cases": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="assetspheres-startup-") as temporary:
        directory = Path(temporary)
        for name, package in packages.items():
            probe = subprocess.run(
                [sys.executable, "-c", "import assetspheres; print(assetspheres.__file__)"],
                env=environments[name],
                cwd=directory,
                capture_output=True,
                text=True,
                check=True,
            )
            assert Path(probe.stdout.strip()).resolve() == package / "__init__.py"
        fixture = directory / "spheres.json"
        fixture.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "units": "metre",
                    "sphere_frame": "link_local",
                    "source": "missing-source.urdf",
                    "config": {"method": "cover", "count": 1},
                    "links": {"part": [{"center": [0, 0, 0], "radius": 0.1}]},
                    "reports": {"part": {"surface_coverage": 1.0, "sphere_count": 1}},
                }
            )
        )
        commands = {
            "import-package": ["-c", "import assetspheres"],
            "import-sphere": ["-c", "from assetspheres import Sphere"],
            "import-fit-api": ["-c", "from assetspheres import fit_mesh"],
            "help": ["-m", "assetspheres", "--help"],
            "version": ["-m", "assetspheres", "--version"],
            "methods-json": ["-m", "assetspheres", "methods", "--json"],
            "report-json": ["-m", "assetspheres", "report", str(fixture), "--json"],
            "invalid-method": ["-m", "assetspheres", "fit", "unused.urdf", "--method", "bad"],
        }
        for label, command in commands.items():

            def run(name):
                started = time.perf_counter()
                result = subprocess.run(
                    [sys.executable, *command],
                    env=environments[name],
                    cwd=directory,
                    capture_output=True,
                    timeout=30,
                )
                elapsed = time.perf_counter() - started
                return (result.returncode, result.stdout, result.stderr), elapsed

            expected, _ = run("reference")
            actual, _ = run("optimized")
            assert actual == expected, label
            assert expected[0] == (2 if label == "invalid-method" else 0), label
            times = {name: [] for name in packages}
            for repeat in range(args.repeats):
                for name in list(packages)[:: 1 if repeat % 2 == 0 else -1]:
                    result, elapsed = run(name)
                    assert result == expected, (label, name)
                    times[name].append(elapsed)
            records["cases"][label] = {
                "exact_status_stdout_stderr": True,
                "returncode": expected[0],
                "stdout_sha256": hashlib.sha256(expected[1]).hexdigest(),
                "stderr_sha256": hashlib.sha256(expected[2]).hexdigest(),
                **{
                    name: {"times_s": values, "median_s": statistics.median(values)}
                    for name, values in times.items()
                },
            }
            args.output.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
            print(
                label, {name: round(statistics.median(values), 4) for name, values in times.items()}
            )
    print(args.output)


if __name__ == "__main__":
    main()
