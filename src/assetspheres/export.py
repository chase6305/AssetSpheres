from __future__ import annotations

import copy
import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import trimesh
import yaml

from .asset import Asset, load_asset, resolve_mesh, source_paths, urdf_file_references
from .model import FitResult
from .result_io import read_result as read_result


def load_result_asset(
    result: FitResult,
    *,
    path: str | Path | None = None,
    package_roots: dict[str, str | Path] | None = None,
) -> Asset:
    options = dict(result.asset_options)
    options["package_roots"] = {**options.get("package_roots", {}), **(package_roots or {})}
    source = Path(path or result.source)
    if source.suffix.lower() != ".urdf" and len(result.links) == 1:
        asset = load_asset(source, **options)
        name = next(iter(result.links))
        link = next(iter(asset.links.values()))
        link.name = name
        asset.links = {name: link}
        asset.zero_pose_transforms = {name: np.eye(4)}
        return asset
    return load_asset(source, link_names=list(result.links), **options)


def _urdf(asset: Asset, result: FitResult, output: Path) -> bytes:
    if asset.urdf is not None:
        root = copy.deepcopy(asset.urdf.getroot())
    else:
        root = ET.Element("robot", name=asset.path.stem)
        link_element = ET.SubElement(root, "link", name=next(iter(asset.links)))
        geometry = ET.SubElement(ET.SubElement(link_element, "visual"), "geometry")
        ET.SubElement(
            geometry,
            "mesh",
            filename=str(asset.path),
            scale=" ".join([format(asset.mesh_scale, ".17g")] * 3),
        )
    for link_element in root.findall("link"):
        link_name = link_element.get("name")
        if link_name not in result.links:
            continue
        for collision in link_element.findall("collision"):
            link_element.remove(collision)
        for sphere_index, sphere in enumerate(result.links[link_name]):
            collision = ET.SubElement(
                link_element, "collision", name=f"assetspheres_{sphere_index}"
            )
            ET.SubElement(
                collision,
                "origin",
                xyz=" ".join(format(x, ".17g") for x in sphere.center),
                rpy="0 0 0",
            )
            ET.SubElement(
                ET.SubElement(collision, "geometry"), "sphere", radius=format(sphere.radius, ".17g")
            )
    # Output location changes; rebase retained meshes and URDF material textures.
    for reference in urdf_file_references(root):
        filename = reference.get("filename")
        if not filename:
            raise ValueError(f"Retained URDF {reference.tag} is missing filename")
        resolved = resolve_mesh(filename, asset.path.parent, asset.package_roots)
        reference.set("filename", Path(os.path.relpath(resolved, output.parent)).as_posix())
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def make_scene(asset: Asset, result: FitResult) -> trimesh.Scene:
    scene = trimesh.Scene()
    for link_index, (link_name, spheres) in enumerate(result.links.items()):
        link_transform = np.asarray(result.transforms.get(link_name, np.eye(4)), dtype=float)
        source_mesh = asset.links[link_name].mesh
        # The preview replaces source materials; preserve independent attributes instead.
        mesh = source_mesh.copy(include_visual=False)
        mesh.vertex_attributes.update(
            {key: copy.deepcopy(value) for key, value in source_mesh.vertex_attributes.items()}
        )
        mesh.face_attributes.update(
            {key: copy.deepcopy(value) for key, value in source_mesh.face_attributes.items()}
        )
        mesh.visual = trimesh.visual.ColorVisuals(mesh, face_colors=[165, 178, 191, 120])
        scene.add_geometry(mesh, node_name=f"mesh/{link_index}", transform=link_transform)
        sphere_color = [
            int(c)
            for c in trimesh.visual.color.hsv_to_rgba([[(link_index * 0.19) % 1, 0.65, 0.95]])[0]
        ]
        sphere_color[3] = 160
        # GLB instances share unit sphere buffers per link. Node transforms map
        # unit sphere -> link -> scene: link_transform @ sphere_transform.
        unit_sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        unit_sphere.visual.face_colors = sphere_color
        geometry_name = f"sphere_template/{link_index}"
        for sphere_index, sphere in enumerate(spheres):
            sphere_transform = np.diag([sphere.radius, sphere.radius, sphere.radius, 1.0])
            sphere_transform[:3, 3] = sphere.center
            node_name = f"spheres/{link_index}/{sphere_index}"
            if sphere_index == 0:
                scene.add_geometry(
                    unit_sphere,
                    geom_name=geometry_name,
                    node_name=node_name,
                    transform=link_transform @ sphere_transform,
                )
            else:
                scene.graph.update(
                    frame_to=node_name,
                    matrix=link_transform @ sphere_transform,
                    geometry=geometry_name,
                )
    return scene


def check_output_paths(
    asset: Asset,
    directory: str | Path,
    *,
    force: bool = False,
    include_glb: bool = True,
) -> list[Path]:
    """Validate export destinations without fitting, creating, or replacing files."""
    directory = Path(directory).expanduser().resolve()
    if directory.exists() and not directory.is_dir():
        raise ValueError(f"Output directory is not a directory: {directory}")
    names = ["spheres.json", "curobo_spheres.yml", "lula_spheres.yml", "spheres.urdf"]
    if include_glb:
        names.append("preview.glb")
    paths = [directory / name for name in names]
    protected = source_paths(asset)
    for path in paths:
        if path.resolve() in protected:
            raise ValueError(
                f"Output would overwrite a source asset: {path}; choose another directory"
            )
        if path.is_dir():
            raise ValueError(f"Output path is a directory: {path}")
    existing = [path.name for path in paths if path.exists() or path.is_symlink()]
    if existing and not force:
        raise ValueError(f"Outputs already exist: {existing}; use --force to replace them")
    return paths


def write_outputs(
    asset: Asset,
    result: FitResult,
    directory: str | Path,
    *,
    force: bool = False,
    include_glb: bool = True,
) -> list[Path]:
    """Export JSON, planner sphere fragments, URDF, and an optional GLB preview.

    Existing outputs require force; source assets are always protected. Compute
    all payloads before replacing files. Retained URDF mesh/texture references
    are relative to the output location, not copied.
    """
    paths = check_output_paths(asset, directory, force=force, include_glb=include_glb)
    directory = paths[0].parent
    if set(result.links) != set(asset.links):
        raise ValueError("Result links must match loaded asset links")
    # An explicit source-path override must survive a subsequent result reload.
    result = replace(
        result,
        source=str(asset.path),
        asset_options={
            "geometry": asset.geometry,
            "scale": asset.mesh_scale,
            "package_roots": {name: str(path) for name, path in asset.package_roots.items()},
        },
    )
    mapping = {name: [s.to_dict() for s in spheres] for name, spheres in result.links.items()}
    payloads = {
        "spheres.json": json.dumps(result.to_dict(), ensure_ascii=False, indent=2, allow_nan=False),
        "curobo_spheres.yml": yaml.safe_dump({"collision_spheres": mapping}, sort_keys=False),
        "lula_spheres.yml": yaml.safe_dump(
            {"collision_spheres": [{name: value} for name, value in mapping.items()]},
            sort_keys=False,
        ),
        "spheres.urdf": _urdf(asset, result, directory / "spheres.urdf"),
    }
    if include_glb:
        payloads["preview.glb"] = make_scene(asset, result).export(file_type="glb")
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".assetspheres-", dir=directory) as temp:
        for name, content in payloads.items():
            data = content.encode("utf-8") if isinstance(content, str) else content
            (Path(temp) / name).write_bytes(data)
        for name in payloads:
            os.replace(Path(temp) / name, directory / name)
    return paths
