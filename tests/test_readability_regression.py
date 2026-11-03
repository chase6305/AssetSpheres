"""Preserve the public keyword interface while internal fitting stages change."""

import numpy as np
import trimesh

from assetspheres import FitConfig, Sphere, fit_mesh
from assetspheres.fit import sample_surface, sphere_gaps
from assetspheres.refine import refine_spheres


def test_public_sampling_and_fitting_keyword_arguments():
    mesh = trimesh.creation.box(extents=(0.2, 0.1, 0.3))
    points = sample_surface(mesh=mesh, count=64, seed=11)
    spheres, report = fit_mesh(mesh=mesh, config=FitConfig(method="cover", count=4, samples=64))
    assert np.all(sphere_gaps(points=points, spheres=spheres) <= 1e-12)
    assert report["triangle_surface_coverage_certified"] is True


def test_refine_preserves_existing_keyword_interface():
    mesh = trimesh.creation.box(extents=(0.2, 0.1, 0.3))
    initial = [Sphere((0.0, 0.0, 0.0), 0.03)]
    occupied = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.0, 0.02, 0.0], [0.0, 0.0, 0.02]])
    configuration = FitConfig(method="refined", count=1, iterations=1, refine_samples=32)
    spheres, report = refine_spheres(
        mesh=mesh, initial=initial, occupied=occupied, pitch=0.02, cfg=configuration
    )
    assert len(spheres) == 1
    assert report["optimization_final_loss"] <= report["optimization_initial_loss"]
    assert initial == [Sphere((0.0, 0.0, 0.0), 0.03)]
