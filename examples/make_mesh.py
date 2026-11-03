"""Generate a small mesh with a handle; no third-party assets needed."""

from pathlib import Path

import trimesh

body = trimesh.creation.box(extents=(0.12, 0.09, 0.18))
handle = trimesh.creation.torus(major_radius=0.055, minor_radius=0.012)
handle.apply_transform(trimesh.transformations.rotation_matrix(1.5707963267948966, (1, 0, 0)))
handle.apply_translation((0.085, 0, 0.02))
mesh = trimesh.util.concatenate([body, handle])
output = Path(__file__).with_name("handled_object.stl")
mesh.export(output)
print(output)
