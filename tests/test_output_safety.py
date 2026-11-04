"""Exports must never replace the source assets, including with force enabled."""

from pathlib import Path

import pytest
import trimesh

from assetspheres import FitConfig, fit_asset, load_asset
from assetspheres.cli import main
from assetspheres.export import write_outputs


@pytest.mark.parametrize("source_name", ["spheres.urdf", "preview.glb"])
def test_force_cannot_overwrite_standalone_source(tmp_path, source_name):
    source = tmp_path / source_name
    if source.suffix == ".urdf":
        source.write_text(
            '<robot name="source"><link name="body"><collision><geometry>'
            '<box size="1 1 1"/></geometry></collision></link></robot>'
        )
    else:
        trimesh.creation.box().export(source)
    before = source.read_bytes()
    asset = load_asset(source)
    result = fit_asset(asset, FitConfig(method="surface", count=2, samples=32))
    with pytest.raises(ValueError, match="source asset"):
        write_outputs(asset, result, tmp_path, force=True)
    assert source.read_bytes() == before
    assert not (tmp_path / "spheres.json").exists()


def test_force_cannot_overwrite_mesh_referenced_only_by_replaced_collisions(tmp_path):
    mesh_path = tmp_path / "preview.glb"
    trimesh.creation.box().export(mesh_path)
    source = tmp_path / "robot.urdf"
    source.write_text(
        '<robot name="source"><link name="body"><collision><geometry>'
        '<mesh filename="preview.glb"/></geometry></collision></link></robot>'
    )
    before = mesh_path.read_bytes()
    asset = load_asset(source)
    result = fit_asset(asset, FitConfig(method="surface", count=2, samples=32))
    with pytest.raises(ValueError, match="source asset"):
        write_outputs(asset, result, tmp_path, force=True)
    assert mesh_path.read_bytes() == before
    assert not (tmp_path / "spheres.json").exists()


def test_cli_rejects_existing_outputs_before_fitting(monkeypatch, tmp_path, capsys):
    from assetspheres import cli

    source = Path(__file__).resolve().parents[1] / "examples" / "two_link.urdf"
    (tmp_path / "spheres.json").write_text("keep me")
    monkeypatch.setattr(cli, "fit_asset", lambda *args, **kwargs: pytest.fail("Must fail first"))
    assert main(["fit", str(source), "-o", str(tmp_path)]) == 2
    assert "already exist" in capsys.readouterr().err
    assert (tmp_path / "spheres.json").read_text() == "keep me"


def test_retained_urdf_texture_paths_are_rebased(tmp_path):
    texture = tmp_path / "texture.png"
    texture.write_bytes(b"texture content is not read by URDF primitive geometry")
    source = tmp_path / "robot.urdf"
    source.write_text(
        '<robot name="textured"><material name="coat"><texture filename="texture.png"/>'
        '</material><link name="body"><visual><geometry><box size="1 1 1"/></geometry>'
        '<material name="coat"/></visual></link></robot>'
    )
    asset = load_asset(source)
    result = fit_asset(asset, FitConfig(method="surface", count=2, samples=32))
    output = tmp_path / "nested" / "out"
    write_outputs(asset, result, output, include_glb=False)
    from xml.etree import ElementTree as ET

    exported = ET.parse(output / "spheres.urdf")
    filename = exported.find(".//texture").get("filename")
    assert (output / filename).resolve() == texture


def test_unrelated_extension_mesh_tags_are_preserved(tmp_path):
    from xml.etree import ElementTree as ET

    source = tmp_path / "robot.urdf"
    source.write_text(
        '<robot name="extended"><link name="body"><collision><geometry>'
        '<box size="1 1 1"/></geometry></collision></link>'
        '<gazebo><plugin name="custom"><mesh>plugin://model</mesh></plugin></gazebo></robot>'
    )
    asset = load_asset(source)
    result = fit_asset(asset, FitConfig(method="surface", count=2, samples=32))
    write_outputs(asset, result, tmp_path / "out", include_glb=False)
    exported = ET.parse(tmp_path / "out" / "spheres.urdf")
    assert exported.find("gazebo/plugin/mesh").text == "plugin://model"


def test_viewer_save_cannot_overwrite_source_before_starting_server(monkeypatch, tmp_path):
    viser = pytest.importorskip("viser")
    from assetspheres.viewer import create_viewer

    source = tmp_path / "part.stl"
    trimesh.creation.box().export(source)
    asset = load_asset(source)
    result = fit_asset(asset, FitConfig(method="surface", count=2, samples=32))
    monkeypatch.setattr(viser, "ViserServer", lambda **kwargs: pytest.fail("Must fail first"))
    with pytest.raises(ValueError, match="source asset"):
        create_viewer(asset, result, save_path=source)
