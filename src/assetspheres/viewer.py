from __future__ import annotations

import json
import tempfile
import threading
from pathlib import Path

import numpy as np
import trimesh

from .asset import Asset, source_paths
from .model import FitResult, Sphere


def _scene_bounds(asset: Asset, result: FitResult, link_names: list[str]) -> np.ndarray:
    bounds = []
    for link_name in link_names:
        transform = np.asarray(result.transforms.get(link_name, np.eye(4)), dtype=float)
        points = trimesh.transform_points(asset.links[link_name].mesh.vertices, transform)
        bounds.append([points.min(axis=0), points.max(axis=0)])
        spheres = result.links[link_name]
        centers = trimesh.transform_points([sphere.center for sphere in spheres], transform)
        radii = np.array([sphere.radius for sphere in spheres])[:, None]
        bounds.append([(centers - radii).min(axis=0), (centers + radii).max(axis=0)])
    bounds = np.asarray(bounds)
    return np.array([bounds[:, 0].min(axis=0), bounds[:, 1].max(axis=0)])


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
    if set(result.links) != set(asset.links):
        raise ValueError("Result links must match loaded asset links")
    save_path = Path(save_path).expanduser().resolve()
    if save_path in source_paths(asset):
        raise ValueError(f"Edited JSON would overwrite a source asset: {save_path}")
    result.source = str(asset.path)
    result.asset_options = {
        "geometry": asset.geometry,
        "scale": asset.mesh_scale,
        "package_roots": {name: str(path) for name, path in asset.package_roots.items()},
    }
    server = viser.ViserServer(host=host, port=port, label="AssetSpheres")
    server.scene.set_up_direction("+z")
    link_names = list(result.links)
    frame_button = server.gui.add_button("Frame model")
    selected_link = server.gui.add_dropdown("Link", link_names)
    solo_link = server.gui.add_checkbox("Only selected link", False)
    show_mesh = server.gui.add_checkbox("Show source geometry", True)
    show_spheres = server.gui.add_checkbox("Show spheres", True)
    opacity_input = server.gui.add_slider(
        "Sphere opacity", min=0.05, max=1.0, step=0.05, initial_value=0.5
    )
    wireframe_input = server.gui.add_checkbox("Wireframe spheres", False)
    sphere_index_input = server.gui.add_number(
        "Sphere index", 0, min=0, max=len(result.links[link_names[0]]) - 1, step=1
    )
    first_sphere = result.links[link_names[0]][0]
    center_input = server.gui.add_vector3("Center in link (m)", first_sphere.center, step=0.001)
    radius_input = server.gui.add_number("Radius (m)", first_sphere.radius, min=1e-8, step=0.001)
    center_order, radius_order = center_input.order, radius_input.order
    save_button = server.gui.add_button("Save edited JSON")
    status = server.gui.add_markdown("Link-local coordinates · Joint positions are zero.")
    lock = threading.RLock()
    colors = [(66, 170, 218), (255, 176, 72), (132, 196, 105), (208, 130, 190)]
    link_frames, mesh_handles, sphere_handles = {}, {}, {}

    def add_ball(link_index: int, sphere_index: int, sphere: Sphere):
        return server.scene.add_icosphere(
            f"/asset/{link_index}/sphere/{sphere_index}",
            sphere.radius,
            colors[link_index % len(colors)],
            position=sphere.center,
            opacity=opacity_input.value,
            wireframe=wireframe_input.value,
            subdivisions=2,
        )

    def frame_camera(client) -> None:
        with lock:
            bounds = _scene_bounds(
                asset, result, [selected_link.value] if solo_link.value else link_names
            )
        target = bounds[0] + (bounds[1] - bounds[0]) / 2
        scene_radius = max(float(np.linalg.norm(bounds[1] - bounds[0])) / 2, 1e-6)
        camera = client.camera
        half_angle = min(camera.fov / 2, np.arctan(np.tan(camera.fov / 2) * camera.aspect))
        distance = 1.35 * scene_radius / np.sin(half_angle)
        direction = np.array([1.0, -1.4, 0.9])
        direction /= np.linalg.norm(direction)
        with client.atomic():
            camera.max_orbit_distance = max(scene_radius * 100, camera.min_orbit_distance)
            camera.min_orbit_distance = scene_radius * 0.01
            camera.position = target + direction * distance
            camera.look_at = target
            camera.up_direction = (0.0, 0.0, 1.0)
            camera.near = scene_radius * 0.001
            camera.far = max(distance + scene_radius * 20, 10.0)

    server.on_client_connect(frame_camera)

    @frame_button.on_click
    def _frame_model(event) -> None:
        if event.client is not None:
            frame_camera(event.client)

    # Keep geometry registered: an editor action should send property updates,
    # not remove/retransmit every mesh and ball in a large robot.
    with server.atomic():
        for link_index, (link_name, spheres) in enumerate(result.links.items()):
            transform = np.asarray(result.transforms.get(link_name, np.eye(4)), dtype=float)
            quaternion = trimesh.transformations.quaternion_from_matrix(transform)
            link_frames[link_name] = server.scene.add_frame(
                f"/asset/{link_index}", wxyz=quaternion, position=transform[:3, 3], show_axes=False
            )
            mesh = asset.links[link_name].mesh
            mesh_handles[link_name] = server.scene.add_mesh_simple(
                f"/asset/{link_index}/mesh",
                mesh.vertices,
                mesh.faces,
                color=(175, 185, 195),
                opacity=0.35,
                side="double",
            )
            sphere_handles[link_name] = [
                add_ball(link_index, ball_index, sphere)
                for ball_index, sphere in enumerate(spheres)
            ]

    def draw_ball(link_index: int, link_name: str, ball_index: int) -> None:
        sphere = result.links[link_name][ball_index]
        handle = sphere_handles[link_name][ball_index]
        if not np.array_equal(handle.position, sphere.center):
            handle.position = sphere.center
            if not np.array_equal(handle.position, sphere.center):
                # Viser 1.1 skips numerically close position writes. Recreate
                # only this ball so even sub-tolerance edits remain visible.
                handle.remove()
                handle = sphere_handles[link_name][ball_index] = add_ball(
                    link_index, ball_index, sphere
                )
        handle.visible = (
            not solo_link.value or link_name == selected_link.value
        ) and show_spheres.value
        handle.color = (
            (250, 90, 90)
            if (link_name, ball_index) == editor_selection
            else colors[link_index % len(colors)]
        )
        handle.radius = sphere.radius
        handle.opacity = opacity_input.value
        handle.wireframe = wireframe_input.value

    def draw() -> None:
        with lock, server.atomic():
            for link_index, (link_name, spheres) in enumerate(result.links.items()):
                link_visible = not solo_link.value or link_name == selected_link.value
                link_frames[link_name].visible = link_visible
                mesh_handles[link_name].visible = link_visible and show_mesh.value
                for ball_index in range(len(spheres)):
                    draw_ball(link_index, link_name, ball_index)

    def sync_editor() -> tuple[str, int]:
        nonlocal center_input, radius_input
        link_name = selected_link.value
        spheres = result.links[link_name]
        ball_index = max(0, min(int(sphere_index_input.value), len(spheres) - 1))
        sphere_index_input.max = len(spheres) - 1
        sphere_index_input.value = ball_index
        sphere = spheres[ball_index]
        # New control IDs isolate each selection from delayed updates belonging
        # to its predecessor. Scene geometry remains registered.
        with server.atomic():
            center_input.remove()
            radius_input.remove()
            center_input = server.gui.add_vector3(
                "Center in link (m)", sphere.center, step=0.001, order=center_order
            )
            radius_input = server.gui.add_number(
                "Radius (m)", sphere.radius, min=1e-8, step=0.001, order=radius_order
            )
            center_input.on_update(edit)
            radius_input.on_update(edit)
        return link_name, ball_index

    editor_selection = (link_names[0], 0)

    def _select(event) -> None:
        nonlocal editor_selection
        if event.client is None:
            return
        with lock:
            previous_selection = editor_selection
            editor_selection = sync_editor()
            if solo_link.value:
                draw()
            else:
                with server.atomic():
                    for link_name, ball_index in {previous_selection, editor_selection}:
                        draw_ball(link_names.index(link_name), link_name, ball_index)

    async def edit(event) -> None:
        # Finish on Viser's event loop without yielding before later Save dispatch.
        # Programmatic widget synchronization must not be interpreted as a user edit.
        if event.client is None:
            return
        with lock:
            # Selection callbacks may still be pending after Viser updates control values.
            live_selection = (selected_link.value, int(sphere_index_input.value))
            if (
                event.target is not center_input and event.target is not radius_input
            ) or live_selection != editor_selection:
                status.content = "Selection changed. Edit the current sphere again."
                return
            link_name, ball_index = editor_selection
            current_sphere = result.links[link_name][ball_index]
            try:
                # A delayed update to the other widget must not change a field
                # that the user did not edit.
                sphere = (
                    Sphere(tuple(center_input.value), current_sphere.radius)
                    if event.target is center_input
                    else Sphere(current_sphere.center, float(radius_input.value))
                )
            except ValueError as exc:
                status.content = str(exc)
                return
            if sphere == current_sphere:
                return
            result.links[link_name][ball_index] = sphere
            report = result.reports.setdefault(link_name, {})
            report["quality_metrics_stale"] = True
            report.pop("triangle_surface_coverage_certified", None)
            report.pop("bounded_solid_coverage_conditional", None)
            report.pop("surface_certification", None)
            status.content = (
                "Edited. Saved coverage metrics are now stale. Save JSON to keep changes."
            )
            with server.atomic():
                draw_ball(link_names.index(link_name), link_name, ball_index)

    selected_link.on_update(_select)
    sphere_index_input.on_update(_select)
    center_input.on_update(edit)
    radius_input.on_update(edit)
    for control in (solo_link, show_mesh, show_spheres, opacity_input, wireframe_input):
        control.on_update(lambda event: draw())

    @save_button.on_click
    def _save(event) -> None:
        with lock:
            path = save_path
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                data = json.dumps(result.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)
                with tempfile.TemporaryDirectory(prefix=".assetspheres-", dir=path.parent) as temp:
                    temporary = Path(temp) / "edited.json"
                    temporary.write_text(data, encoding="utf-8")
                    temporary.replace(path)
                status.content = (
                    f"Saved `{path.name}`. Run `assetspheres export` to update planner files."
                )
            except (ValueError, TypeError, OSError) as exc:
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
