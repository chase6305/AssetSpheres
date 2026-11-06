from dataclasses import replace

import numpy as np
import pytest
import trimesh

from assetspheres import FitConfig
from assetspheres.parallel import fit_links


class _UnpicklableVisual(trimesh.visual.ColorVisuals):
    def __reduce_ex__(self, protocol):
        raise AssertionError("Fitting must not send source visuals to a worker")


class _OffsetCopyMesh(trimesh.Trimesh):
    def copy(self, include_cache=False, include_visual=True):
        mesh = super().copy(include_cache=include_cache, include_visual=include_visual)
        mesh.apply_translation([2, 0, 0])
        return mesh


class _UncopyableMetadata:
    def __deepcopy__(self, memo):
        raise RuntimeError("metadata cannot be copied")


def tasks(method="surface"):
    config = FitConfig(method=method, count=5, samples=128, resolution=16)
    return [
        ("third", trimesh.creation.box(extents=(0.3, 0.1, 0.2)), config),
        ("first", trimesh.creation.icosphere(subdivisions=1, radius=0.1), replace(config, count=3)),
        (
            "second",
            trimesh.creation.cylinder(radius=0.05, height=0.3, sections=12),
            replace(config, count=4),
        ),
    ]


@pytest.mark.parametrize("executor", ["thread", "process"])
@pytest.mark.parametrize("method", ["surface", "voxel", "cover"])
def test_parallel_matches_serial_order_parameters_and_input_meshes(executor, method):
    inputs = tasks(method)
    vertices = [mesh.vertices.copy() for _, mesh, _ in inputs]
    serial = fit_links(inputs)
    concurrent = fit_links(iter(inputs), workers=2, executor=executor)
    assert concurrent == serial
    assert [name for name, _, _ in concurrent] == ["third", "first", "second"]
    assert [len(spheres) for _, spheres, _ in concurrent] == [5, 3, 4]
    for (_, mesh, _), original in zip(inputs, vertices):
        assert np.array_equal(mesh.vertices, original)


@pytest.mark.parametrize("executor", ["thread", "process"])
def test_parallel_failure_names_link(executor):
    inputs = tasks()
    name, mesh, config = inputs[1]
    # An open mesh cannot use the volume method unless fallback is explicit.
    mesh.update_faces(np.arange(len(mesh.faces) - 1))
    inputs[1] = (name, mesh, replace(config, method="voxel"))
    with pytest.raises(ValueError, match="Link 'first'.*watertight"):
        fit_links(inputs, workers=2, executor=executor)


def test_single_worker_bypasses_pool_and_preserves_error_context(monkeypatch):
    import assetspheres.parallel as module

    def unexpected_pool(*args, **kwargs):
        pytest.fail("Single-worker fitting must not create an executor")

    monkeypatch.setattr(module, "ThreadPoolExecutor", unexpected_pool)
    monkeypatch.setattr(module, "ProcessPoolExecutor", unexpected_pool)
    assert len(fit_links(tasks(), workers=1)) == 3
    assert len(fit_links(tasks()[:1], workers=8)) == 1
    name, mesh, config = tasks()[0]
    mesh.update_faces(np.arange(len(mesh.faces) - 1))
    with pytest.raises(ValueError, match="Link 'third'.*watertight"):
        fit_links([(name, mesh, replace(config, method="voxel"))])


@pytest.mark.parametrize("workers", [0, -1, 1.5, True, "2"])
def test_reject_invalid_worker_count(workers):
    with pytest.raises(ValueError, match="workers"):
        fit_links([], workers=workers)


def test_validate_executor_and_link_names():
    with pytest.raises(ValueError, match="executor"):
        fit_links([], executor="invalid")
    inputs = tasks()
    with pytest.raises(ValueError, match="unique"):
        fit_links([inputs[0], inputs[0]])
    with pytest.raises(ValueError, match="nonempty"):
        fit_links([("", *inputs[0][1:])])
    assert fit_links([], workers=2) == []


@pytest.mark.parametrize("executor", ["thread", "process"])
def test_direct_parallel_api_enforces_one_cuda_owner(monkeypatch, executor):
    import assetspheres.parallel as module

    def forbidden(*args, **kwargs):
        pytest.fail("CUDA multi-worker tasks must fail before starting a pool")

    monkeypatch.setattr(module, "ThreadPoolExecutor", forbidden)
    monkeypatch.setattr(module, "ProcessPoolExecutor", forbidden)
    inputs = [
        (name, mesh, replace(config, method="voxel", device="cuda"))
        for name, mesh, config in tasks()
    ]
    with pytest.raises(ValueError, match="one GPU owner"):
        fit_links(inputs, workers=2, executor=executor)


def test_process_fitting_ignores_unpicklable_visuals_and_custom_attributes():
    inputs = tasks("cover")
    source = inputs[0][1]
    colors = np.tile([20, 40, 60, 255], (len(source.faces), 1)).astype(np.uint8)
    source.visual = _UnpicklableVisual(mesh=source, face_colors=colors)

    def unused():
        return None

    source.vertex_attributes["unused"] = unused
    source.face_attributes["unused"] = unused
    vertices, faces = source.vertices.copy(), source.faces.copy()
    visual = source.visual

    expected = fit_links(inputs)
    assert fit_links(inputs, workers=2, executor="process") == expected
    np.testing.assert_array_equal(source.vertices, vertices)
    np.testing.assert_array_equal(source.faces, faces)
    np.testing.assert_array_equal(source.visual.face_colors, colors)
    assert source.visual is visual
    assert source.vertex_attributes["unused"] is unused
    assert source.face_attributes["unused"] is unused


def test_process_preserves_custom_mesh_copy_behavior():
    inputs = tasks("cover")
    name, source, config = inputs[0]
    custom = _OffsetCopyMesh(vertices=source.vertices, faces=source.faces, process=False)
    inputs[0] = (name, custom, config)
    expected = fit_links(inputs)
    assert all(sphere.center[0] > 1 for sphere in expected[0][1])
    assert fit_links(inputs, workers=2, executor="process") == expected
    np.testing.assert_array_equal(custom.vertices, source.vertices)


def test_process_preserves_metadata_copy_failures_and_link_context():
    inputs = tasks("cover")
    inputs[0][1].metadata["value"] = _UncopyableMetadata()
    for workers in (1, 2):
        with pytest.raises(ValueError, match="Link 'third': metadata cannot be copied"):
            fit_links(inputs, workers=workers, executor="process")


@pytest.mark.parametrize("kind", ["none", "array", "empty", "bad_faces"])
def test_process_invalid_mesh_errors_match_serial(kind):
    inputs = tasks("cover")
    invalid = {
        "none": None,
        "array": np.zeros((3, 3)),
        "empty": trimesh.Trimesh(),
        "bad_faces": trimesh.Trimesh(
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 5]], process=False
        ),
    }[kind]
    name, _, config = inputs[0]
    inputs[0] = (name, invalid, config)
    with pytest.raises(ValueError) as expected:
        fit_links(inputs)
    with pytest.raises(ValueError) as actual:
        fit_links(inputs, workers=2, executor="process")
    assert str(actual.value) == str(expected.value)
