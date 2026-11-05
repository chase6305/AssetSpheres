from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from numbers import Real
from typing import Any

import numpy as np


def _finite_number(value: Any, name: str) -> float:
    # Native scalars avoid the numbers.Real registry check in the common JSON path.
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, Real)):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


@dataclass(frozen=True)
class Sphere:
    """A sphere in the owning link's local frame, in metres."""

    center: tuple[float, float, float]
    radius: float

    def __post_init__(self) -> None:
        if np.shape(self.center) != (3,):
            raise ValueError("Sphere center must contain three finite numbers")
        center = tuple(_finite_number(x, "Sphere center") for x in self.center)
        radius = _finite_number(self.radius, "Sphere radius")
        if radius <= 0:
            raise ValueError("Sphere radius must be finite and positive")
        # Frozen spheres must not retain a caller's mutable list/array. Native
        # float scalars also make NumPy inputs safe to serialize as JSON/YAML.
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "radius", radius)

    def to_dict(self) -> dict[str, Any]:
        return {"center": list(self.center), "radius": self.radius}


@dataclass(frozen=True)
class FitConfig:
    """Per-link fitting settings; lengths are in metres and count is an upper bound."""

    method: str = "voxel"
    count: int = 32
    resolution: int = 48
    voxel_size: float | None = None
    padding: float = 0.0
    surface_radius: float = 0.01
    samples: int = 6000
    seed: int = 0
    max_grid_cells: int = 2_000_000
    max_candidates: int = 768
    non_watertight: str = "error"
    device: str = "cpu"
    iterations: int = 120
    refine_samples: int = 2048
    outside_weight: float = 10.0
    refine_reference: str = "voxel"
    voxelizer: str = "subdivide"
    voxel_solid_policy: str = "single"

    def __post_init__(self) -> None:
        if self.method not in {
            "voxel",
            "surface",
            "cover",
            "cover-tight",
            "cover-opt",
            "slab",
            "refined",
            "refined-cover",
            "lula",
        }:
            raise ValueError(f"Unknown fitting method: {self.method}")
        for name in (
            "count",
            "resolution",
            "samples",
            "max_grid_cells",
            "max_candidates",
            "iterations",
            "refine_samples",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.resolution < 4 or self.samples < 32:
            raise ValueError("resolution must be >= 4 and samples must be >= 32")
        if self.refine_samples < 32:
            raise ValueError("refine_samples must be >= 32")
        if self.refine_reference not in {"voxel", "mesh"}:
            raise ValueError("refine_reference must be voxel or mesh")
        if self.voxelizer not in {"subdivide", "open3d", "warp"}:
            raise ValueError("voxelizer must be subdivide, open3d, or warp")
        if self.voxel_solid_policy not in {"single", "union-components"}:
            raise ValueError("voxel_solid_policy must be single or union-components")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.device == "cuda" and self.method not in {"voxel", "refined", "refined-cover"}:
            raise ValueError(
                "CUDA execution supports only voxel, refined and refined-cover methods"
            )
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name in ("padding", "surface_radius", "outside_weight", "voxel_size"):
            value = getattr(self, name)
            if name == "voxel_size" and value is None:
                continue
            value = _finite_number(value, name)
            if value < 0 or (name != "padding" and value == 0):
                limit = "nonnegative" if name == "padding" else "positive"
                raise ValueError(f"{name} must be finite and {limit}")
            object.__setattr__(self, name, value)
        if self.non_watertight not in {"error", "surface", "cover"}:
            raise ValueError("non_watertight must be error, surface, or cover")


@dataclass
class FitResult:
    """Link-local spheres, quality reports, and zero-pose preview transforms."""

    source: str
    config: FitConfig
    links: dict[str, list[Sphere]]
    reports: dict[str, dict[str, Any]]
    transforms: dict[str, list[list[float]]] = field(default_factory=dict)
    asset_options: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "units": "metre",
            "sphere_frame": "link_local",
            "source": self.source,
            "config": asdict(self.config),
            "links": {name: [s.to_dict() for s in spheres] for name, spheres in self.links.items()},
            "reports": self.reports,
            "zero_pose_transforms": self.transforms,
            "asset_options": self.asset_options,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FitResult:
        if not isinstance(data, dict):
            raise ValueError("AssetSpheres result must be a JSON object")
        if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
            raise ValueError("Unsupported AssetSpheres schema_version (expected 1)")
        if data.get("units") != "metre" or data.get("sphere_frame") != "link_local":
            raise ValueError("Expected metre units and link_local sphere coordinates")
        if not isinstance(data.get("source"), str) or not data["source"]:
            raise ValueError("Result source must be a nonempty path string")
        if not isinstance(data.get("config"), dict):
            raise ValueError("Result config must be an object")
        try:
            config = FitConfig(**data["config"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid result config: {exc}") from exc
        raw_links = data.get("links")
        if not isinstance(raw_links, dict) or not raw_links:
            raise ValueError("Result links must be a nonempty object")
        links = {}
        for name, spheres in raw_links.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Result link names must be nonempty strings")
            if not isinstance(spheres, list) or not spheres:
                raise ValueError(f"Result link {name!r} must contain a nonempty sphere list")
            links[name] = []
            for index, sphere in enumerate(spheres):
                try:
                    if not isinstance(sphere, dict):
                        raise ValueError("Sphere must be an object with center and radius")
                    links[name].append(Sphere(sphere["center"], sphere["radius"]))
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid sphere {index} on link {name!r}: {exc}") from exc
        raw_transforms = data.get("zero_pose_transforms", {})
        if not isinstance(raw_transforms, dict):
            raise ValueError("zero_pose_transforms must be an object")
        transforms = {}
        for name, transform in raw_transforms.items():
            try:
                if np.iscomplexobj(transform):
                    raise ValueError("Transform coordinates must be real")
                matrix = np.asarray(transform, dtype=float)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"Invalid zero-pose transform for {name}") from exc
            if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
                raise ValueError(f"Invalid zero-pose transform for {name}")
            if not (
                np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-8)
                and np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), rtol=0, atol=1e-8)
                and np.isclose(np.linalg.det(matrix[:3, :3]), 1, rtol=0, atol=1e-8)
            ):
                raise ValueError(f"Expected a rigid zero-pose transform for {name}")
            transforms[name] = matrix.tolist()
        raw_reports = data.get("reports", {})
        if not isinstance(raw_reports, dict) or any(
            not isinstance(report, dict) for report in raw_reports.values()
        ):
            raise ValueError("Result reports must be an object of link report objects")
        reports = {name: dict(report) for name, report in raw_reports.items()}
        for name, report in reports.items():
            for metric in ("surface_coverage", "surface_gap_p95_m"):
                if report.get(metric) is not None:
                    value = _finite_number(report[metric], f"Report {name!r} {metric}")
                    if value < 0 or (metric == "surface_coverage" and value > 1):
                        raise ValueError(f"Report {name!r} {metric} is outside its valid range")
                    report[metric] = value
            if "effective_method" in report and not isinstance(report["effective_method"], str):
                raise ValueError(f"Report {name!r} effective_method must be a string")
            if "quality_metrics_stale" in report and not isinstance(
                report["quality_metrics_stale"], bool
            ):
                raise ValueError(f"Report {name!r} quality_metrics_stale must be a boolean")
            warnings = report.get("warnings", [])
            if not isinstance(warnings, list) or any(not isinstance(w, str) for w in warnings):
                raise ValueError(f"Report {name!r} warnings must be a list of strings")
        options = data.get("asset_options", {})
        if not isinstance(options, dict) or set(options) - {"geometry", "scale", "package_roots"}:
            raise ValueError(
                "Invalid result asset_options; expected geometry, scale, package_roots"
            )
        options = dict(options)
        geometry = options.get("geometry", "auto")
        if not isinstance(geometry, str) or geometry not in {"auto", "collision", "visual"}:
            raise ValueError("Invalid asset_options geometry")
        scale = _finite_number(options.get("scale", 1), "asset_options scale")
        if scale <= 0:
            raise ValueError("asset_options scale must be positive")
        if "scale" in options:
            options["scale"] = scale
        packages = options.get("package_roots", {})
        if not isinstance(packages, dict) or any(
            not isinstance(name, str) or not name or not isinstance(path, str) or not path
            for name, path in packages.items()
        ):
            raise ValueError("asset_options package_roots must map package names to path strings")
        return cls(
            source=data["source"],
            config=config,
            links=links,
            reports=reports,
            transforms=transforms,
            asset_options=options,
        )
