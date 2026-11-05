"""Offline GUI contract checks; these do not start a browser or listen on a socket."""

import asyncio
import inspect
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from assetspheres import FitConfig, fit_asset, load_asset
from assetspheres.export import read_result
from assetspheres.viewer import create_viewer

viser = pytest.importorskip("viser")


class Widget:
    def __init__(self, value=None):
        self.value = value
        self.callbacks = []
        self.removed = False

    def remove(self):
        self.removed = True

    def on_update(self, callback):
        self.callbacks.append(callback)
        return callback

    on_click = on_update

    def emit(self, *, client=object()):
        for callback in self.callbacks:
            output = callback(SimpleNamespace(client=client, target=self))
            if inspect.isawaitable(output):
                asyncio.run(output)


class OfflineGui:
    def __init__(self):
        self.widgets = {}

    def __getattr__(self, method):
        def add(*args, **kwargs):
            # Bind against the installed Viser API to catch invalid method arguments.
            bound = inspect.signature(getattr(viser.GuiApi, method)).bind(self, *args, **kwargs)
            arguments = bound.arguments
            value = arguments.get("initial_value")
            if method == "add_dropdown" and value is None:
                value = arguments["options"][0]
            label = arguments.get("label", "status")
            self.widgets[label] = Widget(value)
            self.widgets[label].order = arguments.get("order", len(self.widgets))
            return self.widgets[label]

        return add


class SceneHandle(SimpleNamespace):
    def __setattr__(self, name, value):
        if "updated_fields" in self.__dict__:
            self.updated_fields.add(name)
        super().__setattr__(name, value)


class OfflineScene:
    def __init__(self):
        self.calls = []
        self.handles = {}

    def __getattr__(self, method):
        def add(*args, **kwargs):
            inspect.signature(getattr(viser.SceneApi, method)).bind(self, *args, **kwargs)
            self.calls.append((method, args, kwargs))
            handle = SceneHandle(
                remove=lambda: None,
                position=kwargs.get("position", (0, 0, 0)),
                updated_fields=set(),
            )
            self.handles[args[0]] = handle
            return handle

        return add


@pytest.mark.parametrize("method", ["cover", "slab", "refined-cover"])
def test_viewer_frames_manual_edit_and_save(monkeypatch, tmp_path, method):
    asset = load_asset(Path(__file__).resolve().parents[1] / "examples" / "two_link.urdf")
    result = fit_asset(asset, FitConfig(method=method, count=4, samples=128))
    if method == "refined-cover":
        assert result.reports["arm"]["surface_repair_certification"]
    server = SimpleNamespace(
        gui=OfflineGui(),
        scene=OfflineScene(),
        atomic=nullcontext,
        on_client_connect=lambda callback: callback,
    )
    monkeypatch.setattr(viser, "ViserServer", lambda **kwargs: server)
    save_path = tmp_path / "edited.json"
    assert create_viewer(asset, result, save_path=save_path) is server
    frames = [call for call in server.scene.calls if call[0] == "add_frame"]
    assert np.allclose(frames[1][2]["position"], [0, 0, 0.12])
    assert np.allclose(frames[1][2]["wxyz"], [np.cos(0.55 / 2), 0, np.sin(0.55 / 2), 0])
    widgets = server.gui.widgets
    registered_geometry_count = len(server.scene.calls)
    previous_center = widgets["Center in link (m)"]
    previous_radius = widgets["Radius (m)"]
    for handle in server.scene.handles.values():
        handle.updated_fields.clear()
    widgets["Link"].value = "arm"
    widgets["Link"].emit()
    assert {path for path, handle in server.scene.handles.items() if handle.updated_fields} == {
        "/asset/0/sphere/0",
        "/asset/1/sphere/0",
    }
    for handle in server.scene.handles.values():
        handle.updated_fields.clear()
    widgets["Sphere index"].value = 1
    widgets["Sphere index"].emit()
    assert {path for path, handle in server.scene.handles.items() if handle.updated_fields} == {
        "/asset/1/sphere/0",
        "/asset/1/sphere/1",
    }
    widgets["Sphere index"].value = 0
    widgets["Sphere index"].emit()
    original = result.links["arm"][0]
    assert previous_center.removed and previous_radius.removed
    assert widgets["Center in link (m)"].order == previous_center.order
    assert widgets["Radius (m)"].order == previous_radius.order
    # An event already queued for the old selection cannot edit the new one.
    previous_radius.value = 0.123
    previous_radius.emit()
    assert result.links["arm"][0] == original
    assert server.gui.widgets["status"].content == (
        "Selection changed. Edit the current sphere again."
    )
    # Changing only the center must preserve the radius in the model, even
    # while the radius widget still contains an unsynchronized value.
    widgets["Radius (m)"].value = 0.234
    widgets["Center in link (m)"].value = (0.01, 0.02, 0.03)
    widgets["Center in link (m)"].emit(client=None)
    assert result.links["arm"][0] == original  # widget sync must not count as editing
    widgets["Center in link (m)"].emit()
    assert result.links["arm"][0].center == (0.01, 0.02, 0.03)
    assert result.links["arm"][0].radius == original.radius
    assert result.reports["arm"]["quality_metrics_stale"] is True
    assert "triangle_surface_coverage_certified" not in result.reports["arm"]
    assert "bounded_solid_coverage_conditional" not in result.reports["arm"]
    assert "surface_repair_certification" not in result.reports["arm"]
    assert "quality_metrics_stale" not in result.reports["base"]
    for handle in server.scene.handles.values():
        handle.updated_fields.clear()
    widgets["Center in link (m)"].value = (1.0, 2.0, 3.0)
    widgets["Radius (m)"].value = 0.045
    widgets["Radius (m)"].emit()
    assert result.links["arm"][0].center == (0.01, 0.02, 0.03)
    assert {path for path, handle in server.scene.handles.items() if handle.updated_fields} == {
        "/asset/1/sphere/0"
    }
    assert len(server.scene.calls) == registered_geometry_count
    handle = server.scene.handles["/asset/1/sphere/0"]
    assert handle.position == (0.01, 0.02, 0.03)
    assert handle.radius == 0.045
    widgets["Only selected link"].value = True
    widgets["Only selected link"].emit()
    assert server.scene.handles["/asset/0"].visible is False
    assert server.scene.handles["/asset/1"].visible is True
    assert server.scene.handles["/asset/0/mesh"].visible is False
    assert server.scene.handles["/asset/0/sphere/0"].visible is False
    assert server.scene.handles["/asset/1/sphere/0"].visible is True
    widgets["Show spheres"].value = False
    widgets["Show spheres"].emit()
    assert handle.visible is False
    widgets["Save edited JSON"].emit()
    saved = read_result(save_path)
    assert saved.links["arm"][0].radius == 0.045
    assert saved.links["arm"][0].center == (0.01, 0.02, 0.03)
    saved_bytes = save_path.read_bytes()
    result.reports["arm"]["custom_score"] = np.float32(0.5)
    widgets["Save edited JSON"].emit()
    assert save_path.read_bytes() == saved_bytes
    assert widgets["status"].content.startswith("Save failed:")
    widgets["Show spheres"].value = True
    widgets["Show spheres"].emit()
    widgets["Link"].value = "base"
    widgets["Link"].emit()
    assert server.scene.handles["/asset/0"].visible is True
    assert server.scene.handles["/asset/1"].visible is False
    assert server.scene.handles["/asset/0/sphere/0"].visible is True
    assert server.scene.handles["/asset/1/sphere/0"].visible is False


def test_unchanged_editor_input_keeps_quality_metrics(monkeypatch, tmp_path):
    asset = load_asset(Path(__file__).resolve().parents[1] / "examples" / "two_link.urdf")
    result = fit_asset(asset, FitConfig(method="slab", count=4, samples=32))
    server = SimpleNamespace(
        gui=OfflineGui(),
        scene=OfflineScene(),
        atomic=nullcontext,
        on_client_connect=lambda callback: callback,
    )
    monkeypatch.setattr(viser, "ViserServer", lambda **kwargs: server)
    create_viewer(asset, result, save_path=tmp_path / "edited.json")
    server.gui.widgets["Radius (m)"].emit()
    assert "quality_metrics_stale" not in result.reports["base"]


def test_edit_precedes_later_save_with_queued_sync_callbacks(monkeypatch, tmp_path):
    asset = load_asset(Path(__file__).resolve().parents[1] / "examples" / "two_link.urdf")
    result = fit_asset(asset, FitConfig(method="slab", count=4, samples=32))
    server = SimpleNamespace(
        gui=OfflineGui(),
        scene=OfflineScene(),
        atomic=nullcontext,
        on_client_connect=lambda callback: callback,
    )
    monkeypatch.setattr(viser, "ViserServer", lambda **kwargs: server)
    save_path = tmp_path / "edited.json"
    create_viewer(asset, result, save_path=save_path)
    widgets = server.gui.widgets
    queued_callbacks = []

    async def dispatch(widget):
        event = SimpleNamespace(client=object(), target=widget)
        for callback in widget.callbacks:
            # Viser awaits async callbacks; regular callbacks run in a thread pool.
            if inspect.iscoroutinefunction(callback):
                await callback(event)
            else:
                queued_callbacks.append((callback, event))

    async def run_queued_callbacks():
        # The pool may execute Save before an earlier regular callback.
        for callback, event in reversed(queued_callbacks):
            callback(event)

    async def edit_then_save():
        widgets["Radius (m)"].value = 0.045
        await asyncio.gather(
            asyncio.create_task(dispatch(widgets["Radius (m)"])),
            asyncio.create_task(dispatch(widgets["Save edited JSON"])),
            asyncio.create_task(run_queued_callbacks()),
        )

    asyncio.run(edit_then_save())
    assert result.links["base"][0].radius == 0.045
    assert read_result(save_path).links["base"][0].radius == 0.045
    assert widgets["status"].content.startswith("Saved ")


@pytest.mark.parametrize(
    "control_label,pending_value,expected_selection",
    [("Link", "arm", ("arm", 0)), ("Sphere index", 1, ("base", 1))],
)
def test_pending_selection_rejects_previous_editor_input(
    monkeypatch, tmp_path, control_label, pending_value, expected_selection
):
    asset = load_asset(Path(__file__).resolve().parents[1] / "examples" / "two_link.urdf")
    result = fit_asset(asset, FitConfig(method="slab", count=4, samples=32))
    server = SimpleNamespace(
        gui=OfflineGui(),
        scene=OfflineScene(),
        atomic=nullcontext,
        on_client_connect=lambda callback: callback,
    )
    monkeypatch.setattr(viser, "ViserServer", lambda **kwargs: server)
    create_viewer(asset, result, save_path=tmp_path / "edited.json")
    original_links = {name: list(spheres) for name, spheres in result.links.items()}
    original_reports = deepcopy(result.reports)
    widgets = server.gui.widgets
    previous_radius = widgets["Radius (m)"]
    # Match Viser's IO update before its selection callback runs in the pool.
    widgets[control_label].value = pending_value
    previous_radius.value = 0.123
    previous_radius.emit()
    assert result.links == original_links
    assert result.reports == original_reports
    assert widgets["status"].content == "Selection changed. Edit the current sphere again."

    widgets[control_label].emit()
    assert previous_radius.removed
    widgets["Radius (m)"].value = 0.045
    widgets["Radius (m)"].emit()
    link_name, ball_index = expected_selection
    assert result.links[link_name][ball_index].radius == 0.045
    assert result.reports[link_name]["quality_metrics_stale"] is True


def test_pending_link_with_fewer_spheres_rejects_unclamped_index(monkeypatch, tmp_path):
    asset = load_asset(Path(__file__).resolve().parents[1] / "examples" / "two_link.urdf")
    result = fit_asset(asset, FitConfig(method="slab", count=4, samples=32), link_counts={"arm": 1})
    server = SimpleNamespace(
        gui=OfflineGui(),
        scene=OfflineScene(),
        atomic=nullcontext,
        on_client_connect=lambda callback: callback,
    )
    monkeypatch.setattr(viser, "ViserServer", lambda **kwargs: server)
    create_viewer(asset, result, save_path=tmp_path / "edited.json")
    widgets = server.gui.widgets
    widgets["Sphere index"].value = 3
    widgets["Sphere index"].emit()
    original_links = {name: list(spheres) for name, spheres in result.links.items()}
    original_reports = deepcopy(result.reports)
    previous_radius = widgets["Radius (m)"]
    widgets["Link"].value = "arm"
    previous_radius.value = 0.123
    previous_radius.emit()
    assert result.links == original_links
    assert result.reports == original_reports

    widgets["Link"].emit()
    assert widgets["Sphere index"].value == 0
    widgets["Radius (m)"].value = 0.045
    widgets["Radius (m)"].emit()
    assert result.links["arm"][0].radius == 0.045


def test_camera_frames_world_bounds_and_selected_link(monkeypatch, tmp_path):
    from assetspheres.viewer import _scene_bounds

    asset = load_asset(Path(__file__).resolve().parents[1] / "examples" / "two_link.urdf")
    result = fit_asset(asset, FitConfig(method="slab", count=4, samples=32))
    callbacks = []
    server = SimpleNamespace(
        gui=OfflineGui(),
        scene=OfflineScene(),
        atomic=nullcontext,
        on_client_connect=callbacks.append,
    )
    monkeypatch.setattr(viser, "ViserServer", lambda **kwargs: server)
    create_viewer(asset, result, save_path=tmp_path / "edited.json")
    camera = SimpleNamespace(fov=np.deg2rad(75), aspect=0.75, min_orbit_distance=0.01)
    client = SimpleNamespace(camera=camera, atomic=nullcontext)
    callbacks[0](client)
    bounds = _scene_bounds(asset, result, list(result.links))
    np.testing.assert_allclose(camera.look_at, bounds.mean(axis=0))
    distance = np.linalg.norm(camera.position - camera.look_at)
    radius = np.linalg.norm(bounds[1] - bounds[0]) / 2
    assert distance > radius
    assert camera.near < distance - radius and camera.far > distance + radius
    widgets = server.gui.widgets
    widgets["Link"].value = "arm"
    widgets["Link"].emit()
    widgets["Only selected link"].value = True
    widgets["Only selected link"].emit()
    widgets["Frame model"].emit(client=client)
    expected = _scene_bounds(asset, result, ["arm"]).mean(axis=0)
    np.testing.assert_allclose(camera.look_at, expected)


def test_sub_tolerance_center_edits_reach_visible_geometry(monkeypatch, tmp_path):
    class RoundingHandle:
        def __init__(self, position):
            self._position = np.asarray(position)

        @property
        def position(self):
            return self._position

        @position.setter
        def position(self, value):
            # Match Viser's pose setter, which ignores np.allclose changes.
            if not np.allclose(value, self._position):
                self._position = np.asarray(value)

        def remove(self):
            pass

    class RoundingScene(OfflineScene):
        def __getattr__(self, method):
            original = super().__getattr__(method)

            def add(*args, **kwargs):
                handle = original(*args, **kwargs)
                if method == "add_icosphere":
                    handle = self.handles[args[0]] = RoundingHandle(kwargs["position"])
                return handle

            return add

    asset = load_asset(Path(__file__).resolve().parents[1] / "examples" / "two_link.urdf")
    result = fit_asset(asset, FitConfig(method="slab", count=4, samples=32))
    server = SimpleNamespace(
        gui=OfflineGui(),
        scene=RoundingScene(),
        atomic=nullcontext,
        on_client_connect=lambda callback: callback,
    )
    monkeypatch.setattr(viser, "ViserServer", lambda **kwargs: server)
    create_viewer(asset, result, save_path=tmp_path / "edited.json")
    original = result.links["base"][0].center
    edited = (original[0] + 1e-7, original[1], original[2])
    assert np.allclose(original, edited) and original != edited
    registered_count = len(server.scene.calls)
    widget = server.gui.widgets["Center in link (m)"]
    widget.value = edited
    widget.emit()
    assert result.links["base"][0].center == edited
    np.testing.assert_array_equal(server.scene.handles["/asset/0/sphere/0"].position, edited)
    assert len(server.scene.calls) == registered_count + 1
