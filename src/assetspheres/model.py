from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Sphere:
    """A sphere in the owning link's local frame, in metres."""

    center: tuple[float, float, float]
    radius: float

    def __post_init__(self) -> None:
        if np.shape(self.center) != (3,) or not np.all(np.isfinite(self.center)):
            raise ValueError("Sphere center must contain three finite numbers")
        if not np.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("Sphere radius must be finite and positive")

    def to_dict(self) -> dict[str, Any]:
        return {"center": [float(x) for x in self.center], "radius": float(self.radius)}


@dataclass(frozen=True)
class FitConfig:
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

    def __post_init__(self) -> None:
        if self.method not in {"voxel", "surface", "cover", "lula"}:
            raise ValueError(f"Unknown fitting method: {self.method}")
        for name in ("count", "resolution", "samples", "max_grid_cells", "max_candidates"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.resolution < 4 or self.samples < 32:
            raise ValueError("resolution must be >= 4 and samples must be >= 32")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not np.isfinite(self.padding) or self.padding < 0:
            raise ValueError("padding must be finite and nonnegative")
        if not np.isfinite(self.surface_radius) or self.surface_radius <= 0:
            raise ValueError("surface_radius must be finite and positive")
        if self.voxel_size is not None and (
            not np.isfinite(self.voxel_size) or self.voxel_size <= 0
        ):
            raise ValueError("voxel_size must be finite and positive")
        if self.non_watertight not in {"error", "surface", "cover"}:
            raise ValueError("non_watertight must be error, surface, or cover")


@dataclass
class FitResult:
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
        if data.get("schema_version") != 1:
            raise ValueError("Unsupported AssetSpheres schema_version (expected 1)")
        if data.get("units") != "metre" or data.get("sphere_frame") != "link_local":
            raise ValueError("Expected metre units and link_local sphere coordinates")
        transforms = data.get("zero_pose_transforms", {})
        for name, transform in transforms.items():
            if np.shape(transform) != (4, 4) or not np.all(np.isfinite(transform)):
                raise ValueError(f"Invalid zero-pose transform for {name}")
        return cls(
            source=data["source"],
            config=FitConfig(**data["config"]),
            links={
                name: [Sphere(tuple(s["center"]), s["radius"]) for s in spheres]
                for name, spheres in data["links"].items()
            },
            reports=data.get("reports", {}),
            transforms=transforms,
            asset_options=data.get("asset_options", {}),
        )
