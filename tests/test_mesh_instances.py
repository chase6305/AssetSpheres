"""Shared mesh parsing must preserve independent link-local instances."""

import numpy as np
import trimesh

from assetspheres import load_asset


def test_repeated_mesh_is_parsed_once_and_transformed_independently(monkeypatch, tmp_path):
    import assetspheres.asset as module

    mesh_path = tmp_path / "part.stl"
    trimesh.creation.box(extents=(0.2, 0.1, 0.3)).export(mesh_path)
    robot_path = tmp_path / "robot.urdf"
    robot_path.write_text(
        '<robot name="instances">'
        '<link name="base"><collision><geometry><mesh filename="part.stl"/>'
        "</geometry></collision></link>"
        '<link name="tip"><collision><origin xyz="0.4 0.2 0"/>'
        '<geometry><mesh filename="part.stl" scale="2 3 1"/></geometry></collision>'
        '<visual><geometry><mesh filename="unused-missing.stl"/></geometry></visual></link>'
        '<joint name="fixed" type="fixed"><parent link="base"/><child link="tip"/>'
        '<origin xyz="0 0 2"/></joint></robot>'
    )
    read_mesh = module.load_mesh
    parsed_paths = []

    def track_read(path):
        parsed_paths.append(path)
        return read_mesh(path)

    monkeypatch.setattr(module, "load_mesh", track_read)
    asset = load_asset(robot_path)
    assert parsed_paths == [mesh_path]
    np.testing.assert_allclose(
        asset.links["base"].mesh.bounds, [[-0.1, -0.05, -0.15], [0.1, 0.05, 0.15]]
    )
    np.testing.assert_allclose(
        asset.links["tip"].mesh.bounds, [[0.2, 0.05, -0.15], [0.6, 0.35, 0.15]]
    )
    np.testing.assert_array_equal(asset.zero_pose_transforms["tip"][:3, 3], [0, 0, 2])
    asset.links["tip"].mesh.apply_translation((10, 0, 0))
    assert asset.links["base"].mesh.bounds[1, 0] < 1

    # A later load reads the source again instead of reusing a process-wide cache.
    trimesh.creation.box(extents=(0.4, 0.1, 0.3)).export(mesh_path)
    reloaded = load_asset(robot_path)
    assert parsed_paths == [mesh_path, mesh_path]
    np.testing.assert_allclose(reloaded.links["base"].mesh.extents, [0.4, 0.1, 0.3])


def test_selected_links_only_read_their_own_meshes(monkeypatch, tmp_path):
    import assetspheres.asset as module

    trimesh.creation.box().export(tmp_path / "part.stl")
    robot_path = tmp_path / "robot.urdf"
    robot_path.write_text(
        '<robot name="selection"><link name="base"><collision><geometry>'
        '<mesh filename="part.stl"/></geometry></collision></link>'
        '<link name="unused"><collision><geometry><mesh filename="missing.stl"/>'
        '</geometry></collision></link><joint name="fixed" type="fixed">'
        '<parent link="base"/><child link="unused"/></joint></robot>'
    )
    parsed_paths = []
    read_mesh = module.load_mesh

    def track_read(path):
        parsed_paths.append(path)
        return read_mesh(path)

    monkeypatch.setattr(module, "load_mesh", track_read)
    asset = load_asset(robot_path, link_names=["base"])
    assert list(asset.links) == ["base"]
    assert parsed_paths == [tmp_path / "part.stl"]
