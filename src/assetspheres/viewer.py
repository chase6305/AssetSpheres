from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import trimesh

from .asset import Asset
from .model import FitResult, Sphere


def create_viewer(
    asset: Asset,
    result: FitResult,
    *,
    host: str = "127.0.0.1",
    port: int = 8890,
    save_path: str | Path = "spheres.edited.json",
):
    """Build a local Viser inspector. The original input file is never edited."""
    try:
        import viser
    except ImportError as exc:
        raise ImportError("Install the viewer extra: pip install 'assetspheres[viewer]'") from exc
    if not result.links or any(not spheres for spheres in result.links.values()):
        raise ValueError("Viewer requires at least one sphere on each loaded link")
    result.source = str(asset.path)
    result.asset_options = {
        "geometry": asset.geometry,
        "scale": asset.mesh_scale,
        "package_roots": {name: str(path) for name, path in asset.package_roots.items()},
    }
    server = viser.ViserServer(host=host, port=port, label="AssetSpheres")
    server.scene.set_up_direction("+z")
    names = list(result.links)
    selected = server.gui.add_dropdown("Link", names)
    solo = server.gui.add_checkbox("Only selected link", False)
    show_mesh = server.gui.add_checkbox("Show source geometry", True)
    show_spheres = server.gui.add_checkbox("Show spheres", True)
    opacity = server.gui.add_slider(
        "Sphere opacity", min=0.05, max=1.0, step=0.05, initial_value=0.5
    )
    wireframe = server.gui.add_checkbox("Wireframe spheres", False)
    index = server.gui.add_number(
        "Sphere index", 0, min=0, max=len(result.links[names[0]]) - 1, step=1
    )
    first = result.links[names[0]][0]
    center = server.gui.add_vector3("Center in link (m)", first.center, step=0.001)
    radius = server.gui.add_number("Radius (m)", first.radius, min=1e-8, step=0.001)
    save = server.gui.add_button("Save edited JSON")
    status = server.gui.add_markdown("Link-local coordinates · Joint positions are zero.")
    handles = []
    lock = threading.RLock()
    colors = [(66, 170, 218), (255, 176, 72), (132, 196, 105), (208, 130, 190)]

    def draw() -> None:
        with lock, server.atomic():
            for handle in handles:
                handle.remove()
            handles.clear()
            for i, (name, spheres) in enumerate(result.links.items()):
                if solo.value and name != selected.value:
                    continue
                transform = np.asarray(result.transforms.get(name, np.eye(4)), dtype=float)
                quaternion = trimesh.transformations.quaternion_from_matrix(transform)
                handles.append(
                    server.scene.add_frame(
                        f"/asset/{i}", wxyz=quaternion, position=transform[:3, 3], show_axes=False
                    )
                )
                if show_mesh.value:
                    mesh = asset.links[name].mesh
                    handles.append(
                        server.scene.add_mesh_simple(
                            f"/asset/{i}/mesh",
                            mesh.vertices,
                            mesh.faces,
                            color=(175, 185, 195),
                            opacity=0.35,
                            side="double",
                        )
                    )
                if show_spheres.value:
                    for j, sphere in enumerate(spheres):
                        color = (
                            (250, 90, 90)
                            if name == selected.value and j == index.value
                            else (colors[i % len(colors)])
                        )
                        handles.append(
                            server.scene.add_icosphere(
                                f"/asset/{i}/sphere/{j}",
                                sphere.radius,
                                color,
                                position=sphere.center,
                                opacity=opacity.value,
                                wireframe=wireframe.value,
                                subdivisions=2,
                            )
                        )

    def sync_editor() -> None:
        spheres = result.links[selected.value]
        index.max = len(spheres) - 1
        index.value = min(int(index.value), len(spheres) - 1)
        sphere = spheres[int(index.value)]
        center.value, radius.value = sphere.center, sphere.radius

    @selected.on_update
    def _select(event) -> None:
        with lock:
            sync_editor()
            draw()

    @index.on_update
    def _select_sphere(event) -> None:
        with lock:
            sync_editor()
            draw()

    def edit(event) -> None:
        # Programmatic widget synchronization must not be interpreted as a user edit.
        if event.client is None:
            return
        with lock:
            name = selected.value
            try:
                sphere = Sphere(tuple(center.value), float(radius.value))
            except ValueError as exc:
                status.content = str(exc)
                return
            result.links[name][int(index.value)] = sphere
            report = result.reports.setdefault(name, {})
            report["quality_metrics_stale"] = True
            report.pop("triangle_surface_coverage_certified", None)
            status.content = (
                "Edited. Saved coverage metrics are now stale. Save JSON to keep changes."
            )
            draw()

    center.on_update(edit)
    radius.on_update(edit)
    for control in (solo, show_mesh, show_spheres, opacity, wireframe):
        control.on_update(lambda event: draw())

    @save.on_click
    def _save(event) -> None:
        with lock:
            path = Path(save_path).expanduser().resolve()
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                data = json.dumps(result.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)
                temporary = path.with_name(path.name + ".tmp")
                temporary.write_text(data, encoding="utf-8")
                temporary.replace(path)
                status.content = (
                    f"Saved `{path.name}`. Run `assetspheres export` to update planner files."
                )
            except (ValueError, OSError) as exc:
                status.content = f"Save failed: {exc}"

    draw()
    return server


def serve(asset: Asset, result: FitResult, **kwargs) -> None:
    server = create_viewer(asset, result, **kwargs)
    print(f"Viewer: http://{kwargs.get('host', '127.0.0.1')}:{server.get_port()}", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
