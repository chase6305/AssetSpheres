"""Fresh processes keep lightweight queries independent of geometry runtimes."""

import json
import subprocess
import sys
import textwrap

import pytest


def run_python(source):
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )


def test_package_discovery_and_missing_attributes_do_not_load_numeric_backends():
    run_python("""
        import sys
        import assetspheres
        assert set(assetspheres.__all__) <= set(dir(assetspheres))
        assert getattr(assetspheres, 'missing_attribute', None) is None
        assert not any(name in sys.modules for name in ('numpy', 'trimesh', 'scipy', 'torch'))
    """)


@pytest.mark.parametrize("arguments", [["--help"], ["--version"], ["fit", "--help"], ["methods"]])
def test_cli_queries_do_not_load_numeric_or_export_backends(arguments):
    run_python(f"""
        import sys
        from assetspheres.cli import main
        try:
            status = main({arguments!r})
        except SystemExit as outcome:
            status = outcome.code
        assert status == 0
        backends = ('numpy', 'trimesh', 'scipy', 'yaml', 'torch')
        assert not any(name in sys.modules for name in backends)
    """)


def test_report_reads_saved_data_without_loading_geometry_or_yaml(tmp_path):
    from assetspheres import FitConfig, FitResult, Sphere

    report = {"part": {"surface_coverage": 1.0, "sphere_count": 1}}
    result = FitResult(
        "missing-source.urdf", FitConfig(method="cover"), {"part": [Sphere((0, 0, 0), 0.1)]}, report
    )
    path = tmp_path / "spheres.json"
    path.write_text(json.dumps(result.to_dict()))
    output = run_python(f"""
        import sys
        from assetspheres.cli import main
        assert main(['report', {str(path)!r}, '--json']) == 0
        assert not any(name in sys.modules for name in ('trimesh', 'scipy', 'yaml', 'torch'))
    """)
    assert json.loads(output.stdout) == report


def test_public_exports_keep_object_identity_star_imports_and_pickle_paths():
    run_python("""
        import importlib
        import pickle
        import assetspheres
        namespace = {}
        exec('from assetspheres import *', namespace)
        for name in assetspheres.__all__:
            assert namespace[name] is getattr(assetspheres, name)
        for name in ('asset', 'fit', 'model', 'geometry', 'surface_fit', 'surface_grid',
                     'grid_distance', 'voxel_fit'):
            assert getattr(assetspheres, name) is importlib.import_module('assetspheres.' + name)
        from assetspheres.model import Sphere, FitConfig
        assert assetspheres.Sphere is Sphere
        assert assetspheres.FitConfig is FitConfig
        for value in (Sphere((0, 0, 0), 0.1), FitConfig(method='cover')):
            restored = pickle.loads(pickle.dumps(value))
            assert restored == value and type(restored) is type(value)
        from assetspheres.export import read_result as legacy_reader
        from assetspheres.result_io import read_result
        assert legacy_reader is read_result
        assert pickle.loads(b'cassetspheres.export\\nread_result\\n.') is read_result
    """)


def test_concurrent_first_access_returns_the_original_export_objects():
    run_python("""
        from concurrent.futures import ThreadPoolExecutor
        import assetspheres
        names = assetspheres.__all__ * 4
        with ThreadPoolExecutor(max_workers=4) as pool:
            values = list(pool.map(lambda name: getattr(assetspheres, name), names))
        assert all(value is getattr(assetspheres, name) for name, value in zip(names, values))
    """)
