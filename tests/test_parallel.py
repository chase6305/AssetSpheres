from dataclasses import replace

import numpy as np
import pytest
import trimesh

from assetspheres import FitConfig
from assetspheres.parallel import fit_links


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
