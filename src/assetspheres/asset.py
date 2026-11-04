from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlparse
from xml.etree import ElementTree as ET

import numpy as np
import trimesh
from defusedxml.common import DefusedXmlException
from defusedxml.ElementTree import ParseError as DefusedParseError
from defusedxml.ElementTree import parse

from .model import _finite_number


@dataclass
class LinkGeometry:
    name: str
    mesh: trimesh.Trimesh
    geometry_source: str = "mesh"
    geometry_types: tuple[str, ...] = ()


@dataclass
class Asset:
    path: Path
    links: dict[str, LinkGeometry]
    zero_pose_transforms: dict[str, np.ndarray]
    urdf: ET.ElementTree | None = None
    package_roots: dict[str, Path] = field(default_factory=dict)
    geometry: str = "auto"
    mesh_scale: float = 1.0


def _vector(text: str | None, default: tuple, label: str) -> np.ndarray:
    try:
        result = np.array([float(component) for component in text.split()] if text else default)
    except ValueError as exc:
        raise ValueError(f"Invalid {label}: {text!r}") from exc
    if result.shape != (len(default),) or not np.all(np.isfinite(result)):
        raise ValueError(f"Expected {len(default)} finite values for {label}: {text!r}")
    return result


def origin_matrix(element: ET.Element | None) -> np.ndarray:
    if element is None:
        return np.eye(4)
    xyz = _vector(element.get("xyz"), (0, 0, 0), "origin xyz")
    rpy = _vector(element.get("rpy"), (0, 0, 0), "origin rpy")
    transform = trimesh.transformations.euler_matrix(*rpy, axes="sxyz")
    transform[:3, 3] = xyz
    return transform


def resolve_mesh(filename: str, base: Path, packages: dict[str, Path]) -> Path:
    """Resolve local, file:// and ROS package:// URIs without searching arbitrarily."""
    if filename.startswith("package://"):
        package, separator, relative = filename[len("package://") :].partition("/")
        if not package or not separator or not relative:
            raise ValueError(f"Invalid package mesh URI: {filename}")
        roots = []
        if package in packages:
            roots.append(Path(packages[package]))
        for directory in os.environ.get("ROS_PACKAGE_PATH", "").split(os.pathsep):
            if directory:
                root = Path(directory)
                roots.append(root if root.name == package else root / package)
        # The URDF may live inside its own package; inspect named ancestors only.
        roots.extend(parent for parent in (base, *base.parents) if parent.name == package)
        candidates = [root / unquote(relative) for root in roots]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(
            f"Cannot resolve {filename}; pass --package-root {package}=/path/to/{package}"
        )
    if filename.startswith("file://"):
        parsed = urlparse(filename)
        if parsed.netloc not in {"", "localhost"}:
            raise ValueError(f"Remote file URI is unsupported: {filename}")
        path = Path(unquote(parsed.path))
    elif "://" in filename:
        raise ValueError(f"Only local mesh assets are supported: {filename}")
    else:
        path = Path(unquote(filename)).expanduser()
    if not path.is_absolute():
        path = base / path
    if not path.is_file():
        raise FileNotFoundError(f"Mesh does not exist: {path}")
    return path.resolve()


def urdf_file_references(root: ET.Element) -> list[ET.Element]:
    """Known URDF file references; extension/plugin elements are left alone."""
    return [
        *root.findall("link/visual/geometry/mesh"),
        *root.findall("link/collision/geometry/mesh"),
        *root.findall("material/texture"),
        *root.findall("link/visual/material/texture"),
    ]


def source_paths(asset: Asset) -> set[Path]:
    """Source asset and known URDF mesh/texture references, even replaced geometry."""
    paths = {asset.path.resolve()}
    if asset.urdf is not None:
        for reference in urdf_file_references(asset.urdf.getroot()):
            filename = reference.get("filename")
            if filename:
                paths.add(resolve_mesh(filename, asset.path.parent, asset.package_roots))
    return paths


def load_mesh(path: Path) -> trimesh.Trimesh:
    # to_mesh bakes scene node transforms, including repeated geometry instances.
    try:
        loaded = trimesh.load(path, force="scene", process=True)
        mesh = loaded.to_mesh()
    except IndexError as exc:
        raise ValueError(f"Invalid triangle mesh in {path}: {exc}") from exc
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError(f"No triangle geometry in {path}")
    if not np.all(np.isfinite(mesh.vertices)):
        raise ValueError(f"Non-finite vertices in {path}")
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if not len(mesh.faces) or mesh.area <= 0:
        raise ValueError(f"Mesh has no nondegenerate triangles: {path}")
    return mesh


def _positive(element: ET.Element, attribute: str) -> float:
    value = float(element.get(attribute, "nan"))
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{element.tag} {attribute} must be finite and positive")
    return value


def _geometry(
    element: ET.Element,
    base: Path,
    packages: dict[str, Path],
    shared_meshes: dict[str, trimesh.Trimesh | None],
) -> trimesh.Trimesh:
    geometry = element.find("geometry")
    if geometry is None or len(geometry) != 1:
        raise ValueError("Each URDF visual/collision must have exactly one geometry")
    shape = geometry[0]
    if shape.tag == "mesh":
        filename = shape.get("filename")
        if not filename:
            raise ValueError("URDF mesh is missing filename")
        if filename in shared_meshes:
            cached_mesh = shared_meshes[filename]
            if cached_mesh is None:
                cached_mesh = load_mesh(resolve_mesh(filename, base, packages))
                shared_meshes[filename] = cached_mesh
            # Origins and scales belong to each instance, never to the cached mesh.
            mesh = cached_mesh.copy()
        else:
            mesh = load_mesh(resolve_mesh(filename, base, packages))
        scale = _vector(shape.get("scale"), (1, 1, 1), "mesh scale")
        if np.any(scale == 0):
            raise ValueError("URDF mesh scale must be nonzero")
        mesh.apply_scale(scale)
    elif shape.tag == "box":
        size = _vector(shape.get("size"), (0, 0, 0), "box size")
        if np.any(size <= 0):
            raise ValueError("URDF box size must be positive")
        mesh = trimesh.creation.box(extents=size)
    elif shape.tag == "cylinder":
        mesh = trimesh.creation.cylinder(
            radius=_positive(shape, "radius"), height=_positive(shape, "length"), sections=48
        )
    elif shape.tag == "sphere":
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=_positive(shape, "radius"))
    else:
        raise ValueError(f"Unsupported URDF geometry: {shape.tag}")
    # Scale in mesh frame, then apply the geometry's URDF origin into link frame.
    mesh.apply_transform(origin_matrix(element.find("origin")))
    return mesh


def _zero_pose(root: ET.Element, names: list[str]) -> dict[str, np.ndarray]:
    children: dict[str, list[tuple[str, np.ndarray]]] = {name: [] for name in names}
    parents: dict[str, str] = {}
    for joint in root.findall("joint"):
        parent_element, child_element = joint.find("parent"), joint.find("child")
        if parent_element is None or child_element is None:
            raise ValueError(f"Joint {joint.get('name')} is missing parent/child")
        parent_name, child_name = parent_element.get("link"), child_element.get("link")
        if parent_name not in children or child_name not in children or child_name in parents:
            raise ValueError(f"Invalid parent/child tree at joint {joint.get('name')}")
        parents[child_name] = parent_name
        children[parent_name].append((child_name, origin_matrix(joint.find("origin"))))
    roots = [name for name in names if name not in parents]
    if len(roots) != 1:
        raise ValueError(f"URDF must have one root link; found {roots}")
    transforms: dict[str, np.ndarray] = {}
    pending = [(roots[0], np.eye(4))]
    while pending:
        link_name, transform = pending.pop()
        if link_name in transforms:
            raise ValueError("URDF joint graph contains a cycle")
        transforms[link_name] = transform
        pending.extend(
            (child_name, transform @ joint_transform)
            for child_name, joint_transform in children[link_name]
        )
    if len(transforms) != len(names):
        raise ValueError("URDF joint graph contains a disconnected cycle")
    return transforms


def load_asset(
    path: str | Path,
    *,
    geometry: str = "auto",
    package_roots: dict[str, str | Path] | None = None,
    link_names: list[str] | None = None,
    scale: float = 1.0,
) -> Asset:
    """Load geometry in link-local coordinates. Standalone mesh units default to metres."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if geometry not in {"auto", "collision", "visual"}:
        raise ValueError("geometry must be auto, collision, or visual")
    scale = _finite_number(scale, "scale")
    if scale <= 0:
        raise ValueError("scale must be finite and positive")
    packages = {
        name: Path(package_path).expanduser().resolve()
        for name, package_path in (package_roots or {}).items()
    }
    if path.suffix.lower() == ".xacro":
        raise ValueError("Expand Xacro to URDF first: xacro robot.urdf.xacro > robot.urdf")
    if path.suffix.lower() != ".urdf":
        mesh = load_mesh(path)
        mesh.apply_scale(scale)
        name = path.stem
        if link_names and set(link_names) != {name}:
            raise ValueError(f"Standalone mesh link name is {name!r}")
        return Asset(path, {name: LinkGeometry(name, mesh)}, {name: np.eye(4)}, mesh_scale=scale)
    if scale != 1:
        raise ValueError("--scale applies to standalone meshes; URDF uses mesh scale and SI units")
    try:
        tree = parse(path)
    except (ET.ParseError, DefusedParseError, DefusedXmlException) as exc:
        raise ValueError(f"Invalid URDF XML in {path}: {exc}") from exc
    root = tree.getroot()
    if root.tag != "robot":
        raise ValueError("URDF root must be <robot>")
    link_elements = root.findall("link")
    all_link_names = [element.get("name", "") for element in link_elements]
    if (
        not all_link_names
        or any(not name for name in all_link_names)
        or len(set(all_link_names)) != len(all_link_names)
    ):
        raise ValueError("URDF must contain uniquely named links")
    if link_names and set(link_names) - set(all_link_names):
        raise ValueError(f"Unknown links: {sorted(set(link_names) - set(all_link_names))}")
    transforms = _zero_pose(root, all_link_names)
    geometry_groups = []
    for element, link_name in zip(link_elements, all_link_names):
        if link_names and link_name not in link_names:
            continue
        geometry_source = geometry
        if geometry_source == "auto":
            geometry_source = "collision" if element.findall("collision") else "visual"
        geometry_elements = element.findall(geometry_source)
        if not geometry_elements:
            if link_names:
                raise ValueError(f"Link {link_name!r} has no {geometry_source} geometry")
            continue
        geometry_groups.append((link_name, geometry_source, geometry_elements))
    mesh_references = Counter(
        mesh.get("filename")
        for _, _, geometry_elements in geometry_groups
        for element in geometry_elements
        for mesh in element.findall("geometry/mesh")
    )
    # Cache only repeated selected references, for this load_asset call.
    # Do not retain extra mesh copies for unique references.
    shared_meshes: dict[str, trimesh.Trimesh | None] = {
        filename: None for filename, count in mesh_references.items() if filename and count > 1
    }
    links = {}
    for link_name, geometry_source, geometry_elements in geometry_groups:
        try:
            meshes = [
                _geometry(geometry_element, path.parent, packages, shared_meshes)
                for geometry_element in geometry_elements
            ]
        except (ValueError, FileNotFoundError) as exc:
            raise ValueError(f"Link {link_name!r} ({geometry_source}): {exc}") from exc
        geometry_types = tuple(
            geometry_element.find("geometry")[0].tag for geometry_element in geometry_elements
        )
        links[link_name] = LinkGeometry(
            link_name, trimesh.util.concatenate(meshes), geometry_source, geometry_types
        )
    if not links:
        raise ValueError(f"No {geometry} geometry found in {path}")
    return Asset(path, links, transforms, tree, packages, geometry)
