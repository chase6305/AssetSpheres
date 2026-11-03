from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlparse
from xml.etree import ElementTree as ET

import numpy as np
import trimesh
from defusedxml.ElementTree import parse


@dataclass
class LinkGeometry:
    name: str
    mesh: trimesh.Trimesh
    geometry_source: str = "mesh"


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
        result = np.array([float(x) for x in text.split()] if text else default)
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
        package, sep, relative = filename[len("package://") :].partition("/")
        if not package or not sep or not relative:
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


def load_mesh(path: Path) -> trimesh.Trimesh:
    # to_mesh bakes scene node transforms, including repeated geometry instances.
    loaded = trimesh.load(path, force="scene", process=True)
    mesh = loaded.to_mesh()
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


def _geometry(element: ET.Element, base: Path, packages: dict[str, Path]) -> trimesh.Trimesh:
    geometry = element.find("geometry")
    if geometry is None or len(geometry) != 1:
        raise ValueError("Each URDF visual/collision must have exactly one geometry")
    shape = geometry[0]
    if shape.tag == "mesh":
        filename = shape.get("filename")
        if not filename:
            raise ValueError("URDF mesh is missing filename")
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
        parent, child = joint.find("parent"), joint.find("child")
        if parent is None or child is None:
            raise ValueError(f"Joint {joint.get('name')} is missing parent/child")
        p, c = parent.get("link"), child.get("link")
        if p not in children or c not in children or c in parents:
            raise ValueError(f"Invalid parent/child tree at joint {joint.get('name')}")
        parents[c] = p
        children[p].append((c, origin_matrix(joint.find("origin"))))
    roots = [name for name in names if name not in parents]
    if len(roots) != 1:
        raise ValueError(f"URDF must have one root link; found {roots}")
    transforms: dict[str, np.ndarray] = {}
    pending = [(roots[0], np.eye(4))]
    while pending:
        name, transform = pending.pop()
        if name in transforms:
            raise ValueError("URDF joint graph contains a cycle")
        transforms[name] = transform
        pending.extend((child, transform @ local) for child, local in children[name])
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
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    packages = {name: Path(p).expanduser().resolve() for name, p in (package_roots or {}).items()}
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
    except ET.ParseError as exc:
        raise ValueError(f"Invalid URDF XML in {path}: {exc}") from exc
    root = tree.getroot()
    if root.tag != "robot":
        raise ValueError("URDF root must be <robot>")
    elements = root.findall("link")
    names = [e.get("name", "") for e in elements]
    if not names or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("URDF must contain uniquely named links")
    if link_names and set(link_names) - set(names):
        raise ValueError(f"Unknown links: {sorted(set(link_names) - set(names))}")
    transforms = _zero_pose(root, names)
    links = {}
    for element, name in zip(elements, names):
        if link_names and name not in link_names:
            continue
        selected = geometry
        if selected == "auto":
            selected = "collision" if element.findall("collision") else "visual"
        geometries = element.findall(selected)
        if not geometries:
            if link_names:
                raise ValueError(f"Link {name!r} has no {selected} geometry")
            continue
        try:
            meshes = [_geometry(e, path.parent, packages) for e in geometries]
        except (ValueError, FileNotFoundError) as exc:
            raise ValueError(f"Link {name!r} ({selected}): {exc}") from exc
        links[name] = LinkGeometry(name, trimesh.util.concatenate(meshes), selected)
    if not links:
        raise ValueError(f"No {geometry} geometry found in {path}")
    return Asset(path, links, transforms, tree, packages, geometry)
