"""Offline GUI contract checks; these do not start a browser or listen on a socket."""

import inspect
from contextlib import nullcontext
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

    def on_update(self, callback):
        self.callbacks.append(callback)
        return callback

    on_click = on_update

    def emit(self, *, client=object()):
        for callback in self.callbacks:
            callback(SimpleNamespace(client=client))


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
            return self.widgets[label]

        return add


class OfflineScene:
    def __init__(self):
        self.calls = []

    def __getattr__(self, method):
        def add(*args, **kwargs):
            inspect.signature(getattr(viser.SceneApi, method)).bind(self, *args, **kwargs)
            self.calls.append((method, args, kwargs))
            return SimpleNamespace(remove=lambda: None)

        return add


@pytest.mark.parametrize("method", ["cover", "slab"])
def test_viewer_frames_manual_edit_and_save(monkeypatch, tmp_path, method):
    asset = load_asset(Path(__file__).resolve().parents[1] / "examples" / "two_link.urdf")
    result = fit_asset(asset, FitConfig(method=method, count=4, samples=128))
    server = SimpleNamespace(gui=OfflineGui(), scene=OfflineScene(), atomic=nullcontext)
    monkeypatch.setattr(viser, "ViserServer", lambda **kwargs: server)
    save_path = tmp_path / "edited.json"
    assert create_viewer(asset, result, save_path=save_path) is server
    frames = [call for call in server.scene.calls if call[0] == "add_frame"]
    assert np.allclose(frames[1][2]["position"], [0, 0, 0.12])
    assert np.allclose(frames[1][2]["wxyz"], [np.cos(0.55 / 2), 0, np.sin(0.55 / 2), 0])
    widgets = server.gui.widgets
    widgets["Link"].value = "arm"
    widgets["Link"].emit()
    original = result.links["arm"][0]
    widgets["Center in link (m)"].value = (0.01, 0.02, 0.03)
    widgets["Center in link (m)"].emit(client=None)
    assert result.links["arm"][0] == original  # widget sync must not count as editing
    widgets["Center in link (m)"].emit()
    assert result.links["arm"][0].center == (0.01, 0.02, 0.03)
    assert result.reports["arm"]["quality_metrics_stale"] is True
    assert "triangle_surface_coverage_certified" not in result.reports["arm"]
    assert "bounded_solid_coverage_conditional" not in result.reports["arm"]
    assert "quality_metrics_stale" not in result.reports["base"]
    widgets["Radius (m)"].value = 0.045
    widgets["Radius (m)"].emit()
    widgets["Save edited JSON"].emit()
    saved = read_result(save_path)
    assert saved.links["arm"][0].radius == 0.045
    assert saved.links["arm"][0].center == (0.01, 0.02, 0.03)
