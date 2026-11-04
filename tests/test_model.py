"""Results are shared files: reject malformed data with useful validation errors."""

import json

import numpy as np
import pytest

from assetspheres import FitConfig, FitResult, Sphere, fit_asset, load_asset
from assetspheres.cli import main
from assetspheres.export import load_result_asset, read_result, write_outputs


def result_data():
    return FitResult("part.stl", FitConfig(), {"part": [Sphere((0, 0, 0), 0.1)]}, {}).to_dict()


def test_sphere_owns_immutable_native_float_coordinates():
    center = np.array([1, 2, 3], dtype=np.float32)
    sphere = Sphere(center, np.float32(0.1))
    center[0] = 100
    assert sphere.center == (1.0, 2.0, 3.0)
    assert hash(sphere)
    assert json.loads(json.dumps(sphere.to_dict()))["center"] == [1, 2, 3]


@pytest.mark.parametrize("value", [True, "0.1", None, float("nan"), float("inf"), 10**1000])
def test_scalar_parameters_reject_non_numbers_and_nonfinite_values(value):
    with pytest.raises(ValueError):
        Sphere((0, 0, 0), value)
    with pytest.raises(ValueError):
        FitConfig(padding=value)


def test_numpy_config_scalars_roundtrip_through_json():
    data = result_data()
    data["config"] = FitConfig(padding=np.float32(0.01)).__dict__
    assert FitResult.from_dict(json.loads(json.dumps(data))).config.padding > 0
    with pytest.raises(ValueError, match="seed"):
        FitConfig(seed=True)


@pytest.mark.parametrize("transform", [np.eye(4), np.eye(4).astype(str).tolist()])
def test_validated_numpy_fields_are_native_and_independent_of_input(transform, tmp_path):
    data = result_data()
    data["zero_pose_transforms"] = {"part": transform}
    data["asset_options"] = {"scale": np.float32(1)}
    data["reports"] = {
        "part": {"surface_coverage": np.float32(1), "surface_gap_p95_m": np.float32(0.01)}
    }
    loaded = FitResult.from_dict(data)
    path = tmp_path / "spheres.json"
    path.write_text(json.dumps(loaded.to_dict(), allow_nan=False))
    restored = read_result(path)
    assert restored.to_dict() == loaded.to_dict()
    assert type(loaded.transforms["part"][0][0]) is float
    assert type(loaded.asset_options["scale"]) is float
    assert type(loaded.reports["part"]["surface_coverage"]) is float
    assert isinstance(data["asset_options"]["scale"], np.float32)
    assert isinstance(data["reports"]["part"]["surface_coverage"], np.float32)
    transform[0][0] = 2
    assert loaded.transforms["part"][0][0] == 1


def test_numpy_asset_scale_exports_and_reloads(tmp_path):
    import trimesh

    source = tmp_path / "part.stl"
    trimesh.creation.box(extents=(1, 2, 3)).export(source)
    scale = np.float32(0.001)
    asset = load_asset(source, scale=scale)
    assert type(asset.mesh_scale) is float
    assert asset.mesh_scale == float(scale)
    result = fit_asset(asset, FitConfig(method="surface", count=2, samples=32))
    paths = write_outputs(asset, result, tmp_path / "output", include_glb=False)
    restored = read_result(paths[0])
    reloaded_asset = load_result_asset(restored)
    np.testing.assert_array_equal(
        reloaded_asset.links["part"].mesh.vertices, asset.links["part"].mesh.vertices
    )


@pytest.mark.parametrize("scale", [True, np.array([0.001]), np.longdouble("1e400")])
def test_asset_scale_rejects_invalid_scalar_values(scale, tmp_path):
    import trimesh

    source = tmp_path / "part.stl"
    trimesh.creation.box().export(source)
    with pytest.raises(ValueError, match="scale"):
        load_asset(source, scale=scale)


@pytest.mark.parametrize("literal", ["1e999", "-1e999", "NaN", "Infinity", "-Infinity"])
def test_result_reader_rejects_nonfinite_numbers_in_extension_reports(tmp_path, literal):
    data = result_data()
    data["reports"] = {"part": {"extra_score": 0.0}}
    text = json.dumps(data).replace('"extra_score": 0.0', f'"extra_score": {literal}')
    path = tmp_path / "result.json"
    path.write_text(text)
    with pytest.raises(ValueError, match="non-finite number"):
        read_result(path)


def test_result_reader_keeps_large_finite_extension_numbers(tmp_path):
    data = result_data()
    data["reports"] = {"part": {"extra_score": 1e308}}
    path = tmp_path / "result.json"
    path.write_text(json.dumps(data, allow_nan=False))
    loaded = read_result(path)
    assert loaded.reports["part"]["extra_score"] == 1e308
    assert json.loads(json.dumps(loaded.to_dict(), allow_nan=False)) == data


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("source", None),
        ("config", []),
        ("config", {"unknown_setting": 1}),
        ("config", {"method": []}),
        ("links", []),
        ("links", {}),
        ("links", {"part": []}),
        ("links", {"part": [None]}),
        ("links", {"part": [{"center": [0, 0, 0]}]}),
        ("links", {"part": [{"center": [0, 0, "bad"], "radius": 1}]}),
        ("zero_pose_transforms", []),
        ("zero_pose_transforms", {"part": [[1], [2]]}),
        ("zero_pose_transforms", {"part": np.diag([2, 1, 1, 1]).tolist()}),
        ("zero_pose_transforms", {"part": np.diag([-1, 1, 1, 1]).tolist()}),
        ("zero_pose_transforms", {"part": np.eye(4, dtype=complex) + 1j}),
        ("reports", []),
        ("reports", {"part": []}),
        ("reports", {"part": {"surface_coverage": "full"}}),
        ("reports", {"part": {"surface_coverage": 1.1}}),
        ("reports", {"part": {"surface_gap_p95_m": -0.1}}),
        ("reports", {"part": {"effective_method": []}}),
        ("reports", {"part": {"quality_metrics_stale": "false"}}),
        ("reports", {"part": {"warnings": None}}),
        ("asset_options", []),
        ("asset_options", {"link_names": ["part"]}),
        ("asset_options", {"geometry": []}),
        ("asset_options", {"scale": 0}),
        ("asset_options", {"package_roots": {"robot": None}}),
    ],
)
def test_invalid_result_fields_are_value_errors(field, value):
    data = result_data()
    data[field] = value
    with pytest.raises(ValueError):
        FitResult.from_dict(data)


@pytest.mark.parametrize("data", [None, [], "result", 12])
def test_non_object_result_is_rejected(data):
    with pytest.raises(ValueError, match="JSON object"):
        FitResult.from_dict(data)


def test_legacy_config_defaults_and_optional_metadata():
    data = result_data()
    data["config"] = {"method": "cover", "count": 4}
    for optional in ("reports", "asset_options", "zero_pose_transforms"):
        data.pop(optional)
    loaded = FitResult.from_dict(data)
    assert loaded.config.voxelizer == "subdivide"
    assert loaded.config.count == 4
    assert FitResult.from_dict(loaded.to_dict()).links == loaded.links


@pytest.mark.parametrize("text", ["[]", '{"schema_version":1}', '{"unexpected":NaN}'])
def test_cli_reports_bad_results_without_tracebacks(tmp_path, capsys, text):
    path = tmp_path / "bad.json"
    path.write_text(text)
    assert main(["report", str(path)]) == 2
    output = capsys.readouterr()
    assert "assetspheres:" in output.err and "Traceback" not in output.err
    with pytest.raises(ValueError):
        read_result(path)
