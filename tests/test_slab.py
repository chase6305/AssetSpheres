import numpy as np
import pytest
import trimesh

from assetspheres import FitConfig
from assetspheres.fit import sample_surface, sphere_gaps
from assetspheres.slab import slab_spheres


def config(count=8, padding=0.0):
    return FitConfig(method="cover", count=count, padding=padding, samples=128)


@pytest.mark.parametrize("kind", ["cylinder", "box", "torus"])
def test_independent_dense_surface_samples_are_enclosed(kind):
    if kind == "cylinder":
        mesh = trimesh.creation.cylinder(radius=0.04, height=0.8, sections=48)
    elif kind == "box":
        mesh = trimesh.creation.box(extents=(0.3, 0.2, 0.8))
    else:
        mesh = trimesh.creation.torus(major_radius=0.2, minor_radius=0.05)
    original_vertices, original_faces = mesh.vertices.copy(), mesh.faces.copy()
    spheres, details = slab_spheres(mesh, config())
    assert 0 < len(spheres) <= 8
    samples = sample_surface(mesh, 15_000, 194)
    assert np.max(sphere_gaps(samples, spheres)) <= 1e-12
    assert np.max(sphere_gaps(mesh.vertices, spheres)) <= 1e-12
    assert np.allclose(np.linalg.norm(details["slab_axis"]), 1.0)
    assert details["slab_count_nonempty"] == len(spheres)
    assert len(details["slab_point_counts"]) == len(spheres)
    assert "clipped triangle" in details["surface_certification"]
    assert "solid_volume_guarantee" not in details
    assert np.array_equal(mesh.vertices, original_vertices)
    assert np.array_equal(mesh.faces, original_faces)


def test_box_interior_samples_are_enclosed():
    extents = np.array([0.3, 0.2, 0.8])
    mesh = trimesh.creation.box(extents=extents)
    spheres, _ = slab_spheres(mesh, config(count=6))
    points = (np.random.default_rng(99).random((20_000, 3)) - 0.5) * extents
    assert np.max(sphere_gaps(points, spheres)) <= 0


def test_single_triangle_crossing_many_slabs():
    mesh = trimesh.Trimesh(
        vertices=[[-1.0, -0.2, 0.0], [1.0, -0.2, 0.0], [0.0, 0.4, 0.0]],
        faces=[[0, 1, 2]],
        process=False,
    )
    spheres, details = slab_spheres(mesh, config(count=16))
    assert len(spheres) == 16
    assert all(count > 0 for count in details["slab_point_counts"])
    assert np.max(sphere_gaps(sample_surface(mesh, 20_000, 72), spheres)) <= 1e-12


def test_duplicate_vertices_and_coplanar_faces():
    vertices = np.array([[-1.0, -0.2, 0.0], [1.0, -0.2, 0.0], [1.0, 0.2, 0.0], [-1.0, 0.2, 0.0]])
    mesh = trimesh.Trimesh(
        vertices=np.tile(vertices, (3, 1)), faces=[[0, 1, 2], [7, 4, 6], [8, 9, 10]], process=False
    )
    spheres, _ = slab_spheres(mesh, config(count=12))
    assert np.max(sphere_gaps(sample_surface(mesh, 12_000, 73), spheres)) <= 1e-12


@pytest.mark.parametrize(
    "scale, translation",
    [
        (1e-9, [0.0, 0.0, 0.0]),
        (1e3, [0.0, 0.0, 0.0]),
        (1.0, [1e12, -1e12, 1e12]),
        (0.001, [1e10, -1e10, 1e10]),
    ],
)
def test_scale_translation_and_near_coplanarity(scale, translation):
    mesh = trimesh.creation.box(extents=np.array([0.3, 0.02, 1e-9]) * scale)
    mesh.apply_translation(translation)
    spheres, details = slab_spheres(mesh, config(count=5))
    # Barycentric sampling in a large translated frame can itself move samples
    # by one ULP; the slab's coordinate-aware margin must still cover them.
    points = sample_surface(mesh, 5000, 112)
    assert np.max(sphere_gaps(points, spheres)) <= 0
    assert details["radius_roundoff_margin_m"] > 0


def test_padding_is_added_after_clipping_and_tightening():
    mesh = trimesh.creation.cylinder(radius=0.03, height=0.4)
    original, _ = slab_spheres(mesh, config(count=4))
    padded, _ = slab_spheres(mesh, config(count=4, padding=0.005))
    assert [s.center for s in original] == [s.center for s in padded]
    assert all(b.radius - a.radius == pytest.approx(0.005) for a, b in zip(original, padded))


def test_rotation_preserves_surface_enclosure():
    mesh = trimesh.creation.cylinder(radius=0.04, height=0.8, sections=40)
    mesh.apply_transform(trimesh.transformations.euler_matrix(0.7, -0.3, 0.8))
    spheres, _ = slab_spheres(mesh, config(count=8))
    assert np.max(sphere_gaps(sample_surface(mesh, 10_000, 100), spheres)) <= 1e-12


@pytest.mark.parametrize(
    "mesh",
    [
        trimesh.Trimesh(),
        trimesh.Trimesh(
            vertices=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            faces=[[0, 1, 2]],
            process=False,
        ),
    ],
)
def test_reject_empty_or_degenerate_mesh(mesh):
    with pytest.raises(ValueError):
        slab_spheres(mesh, config())
