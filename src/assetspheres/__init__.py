"""URDF/mesh to collision spheres, with no simulator dependency."""

from importlib import import_module as _import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .asset import Asset, LinkGeometry, load_asset
    from .fit import fit_asset, fit_mesh
    from .model import FitConfig, FitResult, Sphere

_EXPORT_MODULES = {
    "Asset": "asset",
    "LinkGeometry": "asset",
    "FitConfig": "model",
    "FitResult": "model",
    "Sphere": "model",
    "load_asset": "asset",
    "fit_asset": "fit",
    "fit_mesh": "fit",
}
# Preserve direct access to the submodules previously loaded by package import.
_SUBMODULES = {
    "asset",
    "fit",
    "geometry",
    "grid_distance",
    "model",
    "surface_fit",
    "surface_grid",
    "voxel_fit",
}

__all__ = [
    "Asset",
    "LinkGeometry",
    "FitConfig",
    "FitResult",
    "Sphere",
    "load_asset",
    "fit_asset",
    "fit_mesh",
]
__version__ = "0.3.1"


def __getattr__(name: str):
    if name in _SUBMODULES:
        value = _import_module(f".{name}", __name__)
    elif name in _EXPORT_MODULES:
        module = _import_module(f".{_EXPORT_MODULES[name]}", __name__)
        value = getattr(module, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | _SUBMODULES)
