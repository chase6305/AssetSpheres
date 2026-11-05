"""Render actual fits of the bundled URDF or a supplied URDF/mesh.

From the repository root, after installing the plots extra:
    python examples/render_comparison.py
"""

import argparse
import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import trimesh
from matplotlib.colors import LightSource
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from assetspheres import FitConfig, fit_asset, load_asset
from assetspheres.geometry import sample_surface, sphere_gaps

METHODS = ("voxel", "cover", "cover-tight", "slab", "refined")
LINK_COLORS = ("#1496b8", "#ed9b32", "#8672bc", "#4aa78b", "#ce7486", "#5793c5", "#92a64f")
BACKGROUND = "#f6f8fc"
TEXT_COLOR = "#17344b"
VALIDATION_SAMPLES = 10_000
VALIDATION_SEED = 950071


def scene_bounds(asset, results):
    """Use one shared physical frame, including every fitted ball's extent."""
    bounds = []
    for link_name, link in asset.links.items():
        transform = asset.zero_pose_transforms[link_name]
        vertices = trimesh.transform_points(link.mesh.vertices, transform)
        bounds.append([vertices.min(axis=0), vertices.max(axis=0)])
        for result in results.values():
            spheres = result.links[link_name]
            centers = trimesh.transform_points([sphere.center for sphere in spheres], transform)
            radii = np.array([sphere.radius for sphere in spheres])[:, None]
            bounds.append([(centers - radii).min(axis=0), (centers + radii).max(axis=0)])
    bounds = np.asarray(bounds)
    minimum, maximum = bounds[:, 0].min(axis=0), bounds[:, 1].max(axis=0)
    margin = (maximum - minimum).max() * 0.06
    return minimum - margin, maximum + margin


def draw_model(axis, asset, result, bounds, zoom=1.0):
    for link_index, (link_name, link) in enumerate(asset.links.items()):
        transform = asset.zero_pose_transforms[link_name]
        source_vertices = trimesh.transform_points(link.mesh.vertices, transform)
        axis.add_collection3d(
            Poly3DCollection(
                source_vertices[link.mesh.faces],
                facecolors="#899bab",
                edgecolors="#62788a" if result is None else "#7d8c99",
                linewidths=0.08,
                alpha=1.0 if result is None else 0.12,
                shade=True,
                lightsource=LightSource(azdeg=315, altdeg=45),
                clip_on=False,
            )
        )
        if result is None:
            continue
        sphere_meshes = []
        for sphere in result.links[link_name]:
            ball = trimesh.creation.icosphere(subdivisions=2, radius=sphere.radius)
            ball.apply_translation(sphere.center)
            sphere_meshes.append(ball)
        balls = trimesh.util.concatenate(sphere_meshes)
        sphere_vertices = trimesh.transform_points(balls.vertices, transform)
        axis.add_collection3d(
            Poly3DCollection(
                sphere_vertices[balls.faces],
                facecolors=LINK_COLORS[link_index % len(LINK_COLORS)],
                linewidths=0,
                alpha=0.78,
                shade=True,
                lightsource=LightSource(azdeg=315, altdeg=45),
                clip_on=False,
            )
        )

    minimum, maximum = bounds
    axis.set(
        xlim=(minimum[0], maximum[0]), ylim=(minimum[1], maximum[1]), zlim=(minimum[2], maximum[2])
    )
    axis.set_box_aspect(maximum - minimum, zoom=zoom)
    axis.set_proj_type("ortho")
    axis.view_init(elev=18, azim=-64)
    axis.set_axis_off()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "examples/two_link.urdf")
    parser.add_argument(
        "--output", type=Path, help="PNG destination; matching JSON is written beside it"
    )
    parser.add_argument("--title", default="Collision sphere fitting")
    parser.add_argument("--columns", type=int, choices=(2, 3), default=3)
    parser.add_argument("--zoom", type=float, default=1.0, help="Shared display magnification")
    parser.add_argument("--geometry", choices=("auto", "collision", "visual"), default="auto")
    arguments = parser.parse_args()
    if not np.isfinite(arguments.zoom) or arguments.zoom <= 0:
        parser.error("--zoom must be finite and positive")
    output = arguments.output or root / "docs/comparison.png"
    data_output = (
        output.with_suffix(".json") if arguments.output else root / "docs/visual-examples.json"
    )
    asset = load_asset(arguments.input, geometry=arguments.geometry)
    results = {}
    records = {
        "scope": "Actual CPU fits; illustration and held-out surface checks, not a speed benchmark",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "trimesh": trimesh.__version__,
        "matplotlib": matplotlib.__version__,
        "input": f"{asset.path.parent.name}/{asset.path.name}",
        "title": arguments.title,
        "input_sha256": hashlib.sha256(asset.path.read_bytes()).hexdigest(),
        "display_pose": "all joint positions zero"
        if asset.urdf is not None
        else "source mesh frame",
        "view": {
            "columns": arguments.columns,
            "zoom": arguments.zoom,
            "elevation": 18,
            "azimuth": -64,
        },
        "geometry": asset.geometry,
        "links": {
            name: {
                "geometry_source": link.geometry_source,
                "triangle_count": len(link.mesh.faces),
                "watertight": bool(link.mesh.is_watertight),
                "mesh_sha256": hashlib.sha256(
                    np.asarray(link.mesh.vertices).tobytes() + np.asarray(link.mesh.faces).tobytes()
                ).hexdigest(),
            }
            for name, link in asset.links.items()
        },
        "validation": {
            "samples_per_link": VALIDATION_SAMPLES,
            "seed": VALIDATION_SEED,
            "coverage_tolerance": "mesh longest extent * 1e-9",
            "aggregate": "arithmetic mean of the link coverage fractions",
        },
        "methods": {},
    }
    validation_points = {
        name: sample_surface(link.mesh, VALIDATION_SAMPLES, VALIDATION_SEED)
        for name, link in asset.links.items()
    }
    for method in METHODS:
        configuration = FitConfig(method=method, count=24, padding=0.003, seed=0)
        result = fit_asset(asset, configuration)
        results[method] = result
        coverage = {
            name: float(
                np.mean(
                    sphere_gaps(validation_points[name], spheres)
                    <= asset.links[name].mesh.extents.max() * 1e-9
                )
            )
            for name, spheres in result.links.items()
        }
        records["methods"][method] = {
            "config": asdict(configuration),
            "sphere_count": sum(len(spheres) for spheres in result.links.values()),
            "coverage_by_link": coverage,
            "mean_surface_coverage": float(np.mean(list(coverage.values()))),
            "spheres": {
                name: [sphere.to_dict() for sphere in spheres]
                for name, spheres in result.links.items()
            },
        }

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": TEXT_COLOR})
    figure = plt.figure(figsize=(12, 10), facecolor=BACKGROUND)
    bounds = scene_bounds(asset, results)
    panels = [("Source geometry", None), *[(method, results[method]) for method in METHODS]]
    rows = len(panels) // arguments.columns
    for panel_index, (label, result) in enumerate(panels):
        axis = figure.add_subplot(
            rows, arguments.columns, panel_index + 1, projection="3d", facecolor=BACKGROUND
        )
        draw_model(axis, asset, result, bounds, zoom=arguments.zoom)
        axis.set_title(label, fontsize=16, weight="bold", pad=0, color=TEXT_COLOR)
        if result is None:
            caption = (
                f"{len(asset.links)} fitted links  |  zero joint pose"
                if asset.urdf is not None
                else "Standalone mesh  |  source frame"
            )
        else:
            record = records["methods"][label]
            caption = (
                f"{record['sphere_count']} balls"
                f"  |  {record['mean_surface_coverage']:.1%} sampled coverage"
            )
        axis.text2D(
            0.5,
            0.01,
            caption,
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            fontsize=10,
            color="#536b7e",
        )

    figure.suptitle(arguments.title, x=0.5, y=0.98, fontsize=23, weight="bold")
    figure.text(
        0.5,
        0.94,
        "CPU  /  up to 24 balls per link  /  3 mm padding  /  shared scale & view",
        ha="center",
        fontsize=11,
        color="#536b7e",
    )
    figure.legend(
        handles=[
            Patch(facecolor=LINK_COLORS[index % len(LINK_COLORS)], label=name)
            for index, name in enumerate(asset.links)
        ]
        + [Patch(facecolor="#899bab", label="source mesh")],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.917),
        ncol=min(4, len(asset.links) + 1),
        frameon=False,
        fontsize=9,
    )
    figure.text(
        0.5,
        0.025,
        "Coverage: independent 10,000 surface points per link; mean across links.\n"
        "Sample coverage is not a continuous-volume certificate. Slab enclosure is conditional.",
        ha="center",
        fontsize=10,
        color="#536b7e",
        linespacing=1.6,
    )
    figure.subplots_adjust(
        left=0.01,
        right=0.99,
        bottom=0.075,
        top=0.85,
        wspace=0.01,
        hspace=0.18 if rows == 3 else 0.08,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, facecolor=BACKGROUND)
    plt.close(figure)
    data_output.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(output)
    print(data_output)


if __name__ == "__main__":
    main()
