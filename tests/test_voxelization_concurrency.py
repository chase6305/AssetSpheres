"""Query lifetime checks and real CPU link-parallelism regression tests."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import trimesh

from assetspheres import Asset, FitConfig, LinkGeometry, fit_asset, voxelization


def _prepare_pair(configuration):
    start_barrier = threading.Barrier(2)

    def prepare():
        start_barrier.wait()
        return voxelization.prepare_distance_grid(trimesh.creation.box(), configuration)

    with ThreadPoolExecutor(max_workers=2) as executor:
        return list(executor.map(lambda _: prepare(), range(2)))


def test_warp_preparation_serializes_construction_sampling_and_close(monkeypatch):
    state_lock = threading.Lock()
    state = {"active": 0, "peak": 0, "closed": 0}

    class TrackedQuery:
        def __init__(self):
            with state_lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            time.sleep(0.01)

        def signed_distances(self, points):
            time.sleep(0.01)
            return np.full(len(points), -0.1, dtype=np.float32)

        def close(self):
            time.sleep(0.01)
            with state_lock:
                state["active"] -= 1
                state["closed"] += 1

    monkeypatch.setattr(voxelization, "_distance_query", lambda *args: TrackedQuery())
    _prepare_pair(FitConfig(voxelizer="warp", resolution=8))
    assert state == {"active": 0, "peak": 1, "closed": 2}


def test_open3d_preparations_keep_independent_queries(monkeypatch):
    state_lock = threading.Lock()
    both_queries_active = threading.Event()
    state = {"active": 0, "peak": 0, "closed": 0}

    class IndependentQuery:
        def __init__(self):
            with state_lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
                if state["active"] == 2:
                    both_queries_active.set()

        def signed_distances(self, points):
            assert both_queries_active.wait(timeout=3), "Open3D queries were serialized"
            return np.full(len(points), -0.1, dtype=np.float32)

        def close(self):
            with state_lock:
                state["active"] -= 1
                state["closed"] += 1

    monkeypatch.setattr(voxelization, "_distance_query", lambda *args: IndependentQuery())
    _prepare_pair(FitConfig(voxelizer="open3d", resolution=8))
    assert state == {"active": 0, "peak": 2, "closed": 2}


def test_query_failure_closes_resources_and_releases_warp_lock(monkeypatch):
    class FailingQuery:
        closed = False

        def signed_distances(self, points):
            raise RuntimeError("Injected distance-query failure")

        def close(self):
            self.closed = True

    query = FailingQuery()
    monkeypatch.setattr(voxelization, "_distance_query", lambda *args: query)
    with pytest.raises(RuntimeError, match="Injected distance-query failure"):
        voxelization.prepare_distance_grid(
            trimesh.creation.box(), FitConfig(voxelizer="warp", resolution=8)
        )
    assert query.closed
    released = threading.Event()

    def acquire_from_another_thread():
        with voxelization._WARP_PREPARATION_LOCK:
            released.set()

    probe = threading.Thread(target=acquire_from_another_thread, daemon=True)
    probe.start()
    probe.join(timeout=3)
    assert released.is_set(), "A failed preparation retained the Warp lifecycle lock"


@pytest.mark.parametrize("backend", ["open3d", "warp"])
def test_real_cpu_distance_backend_threads_match_serial_twice(backend):
    pytest.importorskip(backend)
    asset = Asset(
        path=Path("synthetic.urdf"),
        links={
            "box": LinkGeometry("box", trimesh.creation.box(extents=(0.3, 0.12, 0.18))),
            "cylinder": LinkGeometry(
                "cylinder", trimesh.creation.cylinder(radius=0.04, height=0.25, sections=16)
            ),
        },
        zero_pose_transforms={"box": np.eye(4), "cylinder": np.eye(4)},
    )
    configuration = FitConfig(voxelizer=backend, device="cpu", resolution=12, count=6, samples=128)
    serial = fit_asset(asset, configuration)
    for _ in range(2):
        threaded = fit_asset(asset, configuration, workers=2, executor="thread")
        assert threaded.links == serial.links
        assert threaded.reports == serial.reports


@pytest.mark.parametrize("backend", ["open3d", "warp"])
def test_cpu_query_close_is_idempotent_and_rejects_reuse(backend):
    pytest.importorskip(backend)
    box = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    with voxelization._distance_query(backend, box.vertices, box.faces, "cpu") as query:
        distances = query.signed_distances(
            np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32)
        )
    assert np.allclose(distances, [-0.1, 0.1], atol=2e-6)
    if backend == "warp":
        assert distances.flags.owndata
        assert query.mesh is query.vertex_buffer is query.index_buffer is None
    query.close()
    with pytest.raises(RuntimeError, match="closed"):
        query.signed_distances(np.zeros((1, 3), dtype=np.float32))


def test_warp_constructor_failure_releases_partial_buffers(monkeypatch):
    pytest.importorskip("warp")
    from assetspheres import warp_queries

    closed_states = []
    original_close = warp_queries.WarpDistanceQuery.close

    def recorded_close(query):
        original_close(query)
        closed_states.append((query.mesh, query.vertex_buffer, query.index_buffer))

    def fail_mesh_construction(*args, **kwargs):
        raise RuntimeError("Injected BVH construction failure")

    monkeypatch.setattr(warp_queries.WarpDistanceQuery, "close", recorded_close)
    monkeypatch.setattr(warp_queries.wp, "Mesh", fail_mesh_construction)
    box = trimesh.creation.box()
    with pytest.raises(RuntimeError, match="Injected BVH construction failure"):
        warp_queries.WarpDistanceQuery(box.vertices, box.faces, "cpu")
    assert closed_states == [(None, None, None)]


def test_warp_failed_launch_synchronizes_before_release(monkeypatch):
    pytest.importorskip("warp")
    from assetspheres import warp_queries

    box = trimesh.creation.box()
    query = warp_queries.WarpDistanceQuery(box.vertices, box.faces, "cpu")
    synchronized_devices = []
    original_synchronize = warp_queries.wp.synchronize_device

    def synchronize(device):
        synchronized_devices.append(device)
        original_synchronize(device)

    def fail_launch(*args, **kwargs):
        raise RuntimeError("Injected kernel launch failure")

    monkeypatch.setattr(warp_queries.wp, "synchronize_device", synchronize)
    monkeypatch.setattr(warp_queries.wp, "launch", fail_launch)
    with pytest.raises(RuntimeError, match="Injected kernel launch failure"), query:
        query.signed_distances(np.zeros((1, 3), dtype=np.float32))
    assert synchronized_devices
    assert query.mesh is query.vertex_buffer is query.index_buffer is None
    synchronization_count = len(synchronized_devices)
    query.close()
    assert len(synchronized_devices) == synchronization_count


@pytest.mark.parametrize("backend", ["open3d", "warp"])
def test_excessive_pitch_fails_before_optional_query_construction(monkeypatch, backend):
    monkeypatch.setattr(
        voxelization, "_distance_query", lambda *args: pytest.fail("Should reject the pitch first")
    )
    with pytest.raises(ValueError, match="no interior grid samples.*thinner"):
        voxelization.prepare_distance_grid(
            trimesh.creation.box(), FitConfig(voxelizer=backend, voxel_size=1e308)
        )


@pytest.mark.parametrize("backend,extra", [("open3d", "open3d"), ("warp", "warp")])
def test_optional_dependency_errors_give_local_checkout_install_command(
    monkeypatch, backend, extra
):
    original_import = voxelization.importlib.import_module

    def missing_optional_module(module_name, *args, **kwargs):
        if module_name in {"open3d", ".warp_queries"}:
            raise ImportError("Injected missing dependency")
        return original_import(module_name, *args, **kwargs)

    monkeypatch.setattr(voxelization.importlib, "import_module", missing_optional_module)
    box = trimesh.creation.box()
    with pytest.raises(ValueError) as failure:
        voxelization._distance_query(backend, box.vertices, box.faces, "cpu")
    assert f"python -m pip install -e '.[{extra}]'" in str(failure.value)
