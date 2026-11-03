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

from .asset import Asset, load_asset, resolve_mesh
from .model import FitResult


def read_result(path: str | Path) -> FitResult:
    return FitResult.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


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
        link = ET.SubElement(root, "link", name=next(iter(asset.links)))
        geometry = ET.SubElement(ET.SubElement(link, "visual"), "geometry")
        ET.SubElement(
            geometry,
            "mesh",
            filename=str(asset.path),
            scale=" ".join([format(asset.mesh_scale, ".17g")] * 3),
        )
    for link in root.findall("link"):
        name = link.get("name")
        if name not in result.links:
            continue
        for collision in link.findall("collision"):
            link.remove(collision)
        for i, sphere in enumerate(result.links[name]):
            collision = ET.SubElement(link, "collision", name=f"assetspheres_{i}")
            ET.SubElement(
                collision,
                "origin",
                xyz=" ".join(format(x, ".17g") for x in sphere.center),
                rpy="0 0 0",
            )
            ET.SubElement(
                ET.SubElement(collision, "geometry"), "sphere", radius=format(sphere.radius, ".17g")
            )
    # Output location changes; rebase every retained visual/collision mesh URI.
    for mesh in root.findall(".//geometry/mesh"):
        filename = mesh.get("filename")
        if not filename:
            raise ValueError("Retained URDF mesh is missing filename")
        resolved = resolve_mesh(filename, asset.path.parent, asset.package_roots)
        mesh.set("filename", Path(os.path.relpath(resolved, output.parent)).as_posix())
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def make_scene(asset: Asset, result: FitResult) -> trimesh.Scene:
    scene = trimesh.Scene()
    for index, (name, spheres) in enumerate(result.links.items()):
        transform = np.asarray(result.transforms.get(name, np.eye(4)), dtype=float)
        mesh = asset.links[name].mesh.copy()
        mesh.visual = trimesh.visual.ColorVisuals(mesh, face_colors=[165, 178, 191, 120])
        scene.add_geometry(mesh, node_name=f"mesh/{index}", transform=transform)
        color = [
            int(c) for c in trimesh.visual.color.hsv_to_rgba([[(index * 0.19) % 1, 0.65, 0.95]])[0]
        ]
        color[3] = 160
        for i, sphere in enumerate(spheres):
            ball = trimesh.creation.icosphere(subdivisions=2, radius=sphere.radius)
            ball.visual.face_colors = color
            local = np.eye(4)
            local[:3, 3] = sphere.center
            scene.add_geometry(ball, node_name=f"spheres/{index}/{i}", transform=transform @ local)
    return scene


def write_outputs(
    asset: Asset,
    result: FitResult,
    directory: str | Path,
    *,
    force: bool = False,
    include_glb: bool = True,
) -> list[Path]:
    """Export a JSON result, planner sphere fragments, URDF, and optional GLB preview.

    Existing named outputs require force. Compute all payloads before replacing files.
    Mesh references in the URDF are relative to its output location, not copied.
    """
    directory = Path(directory).expanduser().resolve()
    names = ["spheres.json", "curobo_spheres.yml", "lula_spheres.yml", "spheres.urdf"]
    if include_glb:
        names.append("preview.glb")
    existing = [name for name in names if (directory / name).exists()]
    if existing and not force:
        raise ValueError(f"Outputs already exist: {existing}; use --force to replace them")
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
    return [directory / name for name in payloads]
