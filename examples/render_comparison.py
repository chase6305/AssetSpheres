"""Render a reproducible, headless comparison of the bundled URDF.

Install the plots extra, then run from the repository root:
    python examples/render_comparison.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from assetspheres import FitConfig, fit_asset, load_asset

root = Path(__file__).resolve().parents[1]
asset = load_asset(root / "examples" / "two_link.urdf")
voxel = fit_asset(asset, FitConfig(method="voxel", count=24, padding=0.003))
cover = fit_asset(asset, FitConfig(method="cover", count=24, padding=0.003))
slab = fit_asset(asset, FitConfig(method="slab", count=24, padding=0.003))
refined = fit_asset(asset, FitConfig(method="refined", count=24, padding=0.003))
fig = plt.figure(figsize=(16, 5), facecolor="#f5f7fa")
colors = ["#2995b8", "#ee9c36"]

for column, (label, result) in enumerate(
    [
        ("Source geometry", None),
        ("Voxel", voxel),
        ("Triangle cover", cover),
        ("Parallel slabs", slab),
        ("Refined", refined),
    ]
):
    ax = fig.add_subplot(1, 5, column + 1, projection="3d", facecolor="#f5f7fa")
    for i, (name, link) in enumerate(asset.links.items()):
        transform = asset.zero_pose_transforms[name]
        vertices = trimesh.transform_points(link.mesh.vertices, transform)
        source = Poly3DCollection(
            vertices[link.mesh.faces],
            linewidths=0,
            facecolor="#7c8c9b",
            alpha=0.85 if result is None else 0.10,
        )
        ax.add_collection3d(source)
        if result is not None:
            for sphere in result.links[name]:
                ball = trimesh.creation.icosphere(subdivisions=2, radius=sphere.radius)
                ball.apply_translation(sphere.center)
                vertices = trimesh.transform_points(ball.vertices, transform)
                ax.add_collection3d(
                    Poly3DCollection(
                        vertices[ball.faces],
                        linewidths=0.08,
                        edgecolor=colors[i],
                        facecolor=colors[i],
                        alpha=0.5,
                    )
                )
    ax.set(xlim=(-0.15, 0.4), ylim=(-0.275, 0.275), zlim=(-0.05, 0.5))
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=17, azim=-66)
    ax.set_axis_off()
    fig.text((column + 0.5) / 5, 0.76, label, ha="center", fontsize=12, color="#263b4d")

fig.suptitle("AssetSpheres", fontsize=24, fontweight="bold", color="#163449", y=0.96)
fig.text(0.5, 0.86, "Up to 24 spheres per link · 3 mm padding", ha="center", color="#526779")
fig.text(
    0.5,
    0.07,
    "Slab enclosure assumes a bounded solid boundary. "
    "Refinement trades coverage against excess volume.",
    ha="center",
    color="#526779",
    fontsize=9,
)
fig.subplots_adjust(left=0.02, right=0.98, top=0.74, bottom=0.08, wspace=0.01)
output = root / "docs" / "comparison.png"
fig.savefig(output, dpi=160, facecolor=fig.get_facecolor())
print(output)
