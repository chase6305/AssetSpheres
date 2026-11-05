from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest
import trimesh
import yaml

from assetspheres import FitConfig, fit_asset, load_asset
from assetspheres.asset import load_mesh, resolve_mesh
from assetspheres.cli import main
from assetspheres.export import load_result_asset, make_scene, read_result, write_outputs

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "two_link.urdf"


def canonical(element):
    return ET.canonicalize(ET.tostring(element, encoding="unicode"), strip_text=True)


def test_scale_then_geometry_origin_then_joint_transform(tmp_path):
    package = tmp_path / "my_robot"
    (package / "meshes").mkdir(parents=True)
    trimesh.creation.box(extents=(1, 2, 3)).export(package / "meshes" / "box.stl")
    path = tmp_path / "robot.urdf"
    path.write_text("""<robot name="test">
      <link name="root"/>
      <link name="tip"><collision><origin xyz="1 2 3" rpy="0 0 1.5707963267948966"/>
      <geometry><mesh filename="package://my_robot/meshes/box.stl" scale="2 3 4"/>
      </geometry></collision></link>
      <joint name="j" type="fixed"><parent link="root"/><child link="tip"/>
      <origin xyz="10 0 0"/></joint></robot>""")
    asset = load_asset(path, package_roots={"my_robot": package})
    assert np.allclose(asset.links["tip"].mesh.centroid, [1, 2, 3])
    assert np.allclose(asset.links["tip"].mesh.extents, [6, 2, 12])
    assert np.allclose(asset.zero_pose_transforms["tip"][:3, 3], [10, 0, 0])
    result = fit_asset(asset, FitConfig(method="cover", count=4, samples=128))
    centers = np.array([s.center for s in result.links["tip"]])
    assert np.max(centers[:, 0]) < 5  # joint origin must not be baked into sphere centers


def test_scene_node_transforms_and_repeated_instances(tmp_path):
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.creation.box(), geom_name="box", node_name="left")
    transform = np.eye(4)
    transform[0, 3] = 4
    scene.graph.update(frame_to="right", matrix=transform, geometry="box")
    path = tmp_path / "instances.glb"
    scene.export(path)
    mesh = load_mesh(path)
    assert np.allclose(mesh.bounds, [[-0.5, -0.5, -0.5], [4.5, 0.5, 0.5]])
    assert len(mesh.faces) == 24


@pytest.mark.parametrize("face", ["1 2 99", "-10 -2 -1"])
def test_invalid_obj_face_indices_report_the_source_path(tmp_path, face):
    source = tmp_path / "invalid.obj"
    source.write_text(f"v 0 0 0\nv 1 0 0\nv 0 1 0\nf {face}\n")
    with pytest.raises(ValueError, match="Invalid triangle mesh") as outcome:
        load_asset(source)
    assert str(source) in str(outcome.value)
    assert isinstance(outcome.value.__cause__, IndexError)


def test_exports_roundtrip_and_preserve_urdf(tmp_path):
    asset = load_asset(EXAMPLE)
    result = fit_asset(
        asset, FitConfig(method="cover", count=8, samples=128), link_counts={"arm": 5}
    )
    paths = write_outputs(asset, result, tmp_path / "output")
    assert len(paths) == 5
    loaded = read_result(paths[0])
    assert loaded.links == result.links
    assert load_result_asset(loaded).links.keys() == asset.links.keys()
    exported = ET.parse(tmp_path / "output" / "spheres.urdf").getroot()
    original = asset.urdf.getroot()
    assert canonical(exported.find("joint")) == canonical(original.find("joint"))
    for link in exported.findall("link"):
        name = link.get("name")
        assert len(link.findall("collision")) == len(result.links[name])
        for collision, sphere in zip(link.findall("collision"), result.links[name]):
            assert np.allclose(
                np.fromstring(collision.find("origin").get("xyz"), sep=" "), sphere.center
            )
            assert float(collision.find("geometry/sphere").get("radius")) == sphere.radius
        original_link = original.find(f"link[@name='{name}']")
        assert canonical(link.find("visual")) == canonical(original_link.find("visual"))
    curobo = yaml.safe_load((tmp_path / "output" / "curobo_spheres.yml").read_text())
    lula = yaml.safe_load((tmp_path / "output" / "lula_spheres.yml").read_text())
    assert curobo["collision_spheres"]["arm"] == [s.to_dict() for s in result.links["arm"]]
    assert lula["collision_spheres"][1]["arm"] == curobo["collision_spheres"]["arm"]
    assert load_asset(tmp_path / "output" / "spheres.urdf").links.keys() == asset.links.keys()
    preview = trimesh.load(tmp_path / "output" / "preview.glb", force="scene")
    assert len(preview.graph.nodes_geometry) == sum(map(len, result.links.values())) + len(
        asset.links
    )
    # Shared GLB buffers must preserve the position and radius of every instance.
    for link_index, (name, spheres) in enumerate(result.links.items()):
        for sphere_index, sphere in enumerate(spheres):
            transform, _ = preview.graph[f"spheres/{link_index}/{sphere_index}"]
            expected = np.asarray(result.transforms[name])
            assert np.allclose(
                transform[:3, 3], trimesh.transform_points([sphere.center], expected)[0]
            )
            assert np.allclose(np.linalg.norm(transform[:3, :3], axis=0), sphere.radius)
    with pytest.raises(ValueError, match="already exist"):
        write_outputs(asset, result, tmp_path / "output")


@pytest.mark.parametrize("textured", [False, True])
def test_preview_keeps_independent_attributes_and_skips_discarded_textures(monkeypatch, textured):
    asset = load_asset(EXAMPLE, link_names=["base"])
    result = fit_asset(asset, FitConfig(method="cover", count=2, samples=64))
    source = asset.links["base"].mesh
    vertices = source.vertices.copy()
    features = np.arange(len(vertices) * 3, dtype=np.float32).reshape(-1, 3)
    labels = np.arange(len(source.faces))
    source.vertex_attributes["_FEATURES"] = features.copy()
    source.face_attributes["labels"] = labels.copy()
    if textured:
        image_module = pytest.importorskip("PIL.Image")
        image = image_module.new("RGBA", (16, 16), (60, 120, 180, 255))
        source.visual = trimesh.visual.texture.TextureVisuals(
            uv=np.zeros((len(vertices), 2)),
            material=trimesh.visual.material.PBRMaterial(baseColorTexture=image),
        )
        monkeypatch.setattr(
            image, "copy", lambda: pytest.fail("Preview discards the source texture")
        )
    visual = source.visual

    scene = make_scene(asset, result)
    _, geometry_name = scene.graph["mesh/0"]
    preview = scene.geometry[geometry_name]
    np.testing.assert_array_equal(preview.vertices, vertices)
    np.testing.assert_array_equal(preview.vertex_attributes["_FEATURES"], features)
    np.testing.assert_array_equal(preview.face_attributes["labels"], labels)
    assert isinstance(preview.visual, trimesh.visual.ColorVisuals)
    np.testing.assert_array_equal(preview.visual.face_colors[0], [165, 178, 191, 120])
    restored = trimesh.load(BytesIO(scene.export(file_type="glb")), file_type="glb", force="scene")
    assert set(restored.graph.nodes_geometry) == set(scene.graph.nodes_geometry)

    preview.vertices[0, 0] += 1
    preview.vertex_attributes["_FEATURES"][0, 0] = -1
    preview.face_attributes["labels"][0] = -1
    np.testing.assert_array_equal(source.vertices, vertices)
    np.testing.assert_array_equal(source.vertex_attributes["_FEATURES"], features)
    np.testing.assert_array_equal(source.face_attributes["labels"], labels)
    assert source.visual is visual
    if textured:
        assert image.getpixel((0, 0)) == (60, 120, 180, 255)


def test_preview_link_geometry_and_colors_remain_independent():
    asset = load_asset(EXAMPLE)
    result = fit_asset(asset, FitConfig(method="cover", count=2, samples=64))
    scene = make_scene(asset, result)
    first = scene.geometry["sphere_template/0"]
    second = scene.geometry["sphere_template/1"]
    vertices, faces, colors = (
        second.vertices.copy(),
        second.faces.copy(),
        second.visual.face_colors.copy(),
    )
    assert not np.array_equal(first.visual.face_colors, colors)
    first.vertices[0] += 1
    first.faces[0] = 0
    first.visual.face_colors[0] = [0, 0, 0, 0]
    np.testing.assert_array_equal(second.vertices, vertices)
    np.testing.assert_array_equal(second.faces, faces)
    np.testing.assert_array_equal(second.visual.face_colors, colors)
    fresh_scene = make_scene(asset, result)
    np.testing.assert_array_equal(fresh_scene.geometry["sphere_template/1"].vertices, vertices)


def test_mesh_scaling_survives_export_and_view_reload(tmp_path):
    source = tmp_path / "object.stl"
    trimesh.creation.box(extents=(100, 200, 300)).export(source)
    asset = load_asset(source, scale=0.001)
    result = fit_asset(asset, FitConfig(method="cover", count=4, samples=128))
    out = tmp_path / "nested" / "output"
    write_outputs(asset, result, out, include_glb=False)
    reloaded = load_asset(out / "spheres.urdf", geometry="visual")
    assert np.allclose(reloaded.links["object"].mesh.extents, [0.1, 0.2, 0.3])
    assert np.allclose(
        load_result_asset(read_result(out / "spheres.json")).links["object"].mesh.extents,
        [0.1, 0.2, 0.3],
    )
    filename = ET.parse(out / "spheres.urdf").find(".//mesh").get("filename")
    assert not Path(filename).is_absolute()
    assert (out / filename).resolve() == source


def test_selected_links_only_replace_their_collisions(tmp_path):
    asset = load_asset(EXAMPLE, link_names=["arm"])
    result = fit_asset(asset, FitConfig(method="surface", count=3, samples=128))
    write_outputs(asset, result, tmp_path, include_glb=False)
    exported = ET.parse(tmp_path / "spheres.urdf")
    assert exported.find(".//link[@name='base']/collision/geometry/box") is not None
    assert len(exported.findall(".//link[@name='arm']/collision")) == 3


def test_package_uri_resolution_and_useful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="--package-root"):
        resolve_mesh("package://missing/meshes/x.stl", tmp_path, {})
    with pytest.raises(ValueError, match="local"):
        resolve_mesh("https://example.com/model.stl", tmp_path, {})


@pytest.mark.parametrize(
    "joints",
    [
        '<joint name="j" type="fixed"><parent link="a"/><child link="a"/></joint>',
        '<joint name="j" type="fixed"><parent link="a"/><child link="missing"/></joint>',
        "",
    ],
)
def test_reject_invalid_urdf_trees(tmp_path, joints):
    source = tmp_path / "invalid.urdf"
    source.write_text(f'<robot name="bad"><link name="a"/><link name="b"/>{joints}</robot>')
    with pytest.raises(ValueError):
        load_asset(source)


def test_cli_fit_report_and_reexport(tmp_path, capsys):
    out = tmp_path / "fit"
    assert (
        main(
            [
                "fit",
                str(EXAMPLE),
                "--method",
                "cover",
                "--count",
                "5",
                "--samples",
                "128",
                "--link-count",
                "arm=3",
                "--no-glb",
                "-o",
                str(out),
            ]
        )
        == 0
    )
    assert main(["report", str(out / "spheres.json"), "--json"]) == 0
    assert (
        main(["export", str(out / "spheres.json"), "--no-glb", "-o", str(tmp_path / "export")]) == 0
    )
    assert main(["fit", "missing.urdf", "-o", str(out)]) == 2
    assert "missing.urdf" in capsys.readouterr().err


def test_auto_visual_fallback_is_reported(tmp_path):
    source = tmp_path / "visual.urdf"
    source.write_text("""<robot name="visual"><link name="a"><visual><geometry>
    <box size="1 1 1"/></geometry></visual></link></robot>""")
    asset = load_asset(source)
    result = fit_asset(asset, FitConfig(method="surface", count=3, samples=128))
    assert result.reports["a"]["geometry_source"] == "visual"
    with pytest.raises(ValueError, match="No collision"):
        load_asset(source, geometry="collision")


def test_result_source_override_survives_reexport(tmp_path):
    source = tmp_path / "original.stl"
    trimesh.creation.box().export(source)
    asset = load_asset(source)
    result = fit_asset(asset, FitConfig(method="surface", count=3, samples=128))
    renamed = tmp_path / "renamed.stl"
    source.rename(renamed)
    moved_asset = load_result_asset(result, path=renamed)
    assert list(moved_asset.links) == ["original"]
    write_outputs(moved_asset, result, tmp_path / "export", include_glb=False)
    reloaded = read_result(tmp_path / "export" / "spheres.json")
    assert reloaded.source == str(renamed)
    assert list(load_result_asset(reloaded).links) == ["original"]


def test_malformed_urdf_has_cli_error_without_traceback(tmp_path, capsys):
    source = tmp_path / "bad.urdf"
    source.write_text('<robot name="unfinished"><link>')
    assert main(["fit", str(source), "-o", str(tmp_path / "out")]) == 2
    assert "Invalid URDF XML" in capsys.readouterr().err


def test_urdf_entity_rejection_has_cli_error_without_traceback(tmp_path, capsys):
    source = tmp_path / "entity.urdf"
    source.write_text(
        '<!DOCTYPE robot [<!ENTITY injected "entity">]>'
        '<robot name="&injected;"><link name="body"/></robot>'
    )
    assert main(["inspect", str(source)]) == 2
    error = capsys.readouterr().err
    assert "Invalid URDF XML" in error and "Traceback" not in error
