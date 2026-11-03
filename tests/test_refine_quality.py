from dataclasses import replace

import numpy as np
import pytest
import trimesh
from scipy.optimize import check_grad

from assetspheres import FitConfig, Sphere, fit_asset, fit_mesh, load_asset
from assetspheres.quality import compare_sphere_models
from assetspheres.refine import _loss_gradient


def test_refinement_analytic_gradient_matches_finite_difference():
    rng = np.random.default_rng(3)
    points = rng.normal(size=(23, 3))
    parameters = np.array([[0.0, 0.0, 0.0, 0.8], [1.0, 0.0, 0.0, 0.6]])
    positive = np.arange(len(points)) % 2 == 0
    weights = np.full(len(points), 1 / len(points))

    def objective(flat):
        return _loss_gradient(flat.reshape(-1, 4), points, positive, weights, 0.01)[0]

    def gradient(flat):
        return _loss_gradient(flat.reshape(-1, 4), points, positive, weights, 0.01)[1].ravel()

    assert check_grad(objective, gradient, parameters.ravel()) < 1e-5


def test_numpy_clamped_norm_has_zero_center_gradient():
    _, gradient = _loss_gradient(
        np.array([[0.0, 0.0, 0.0, 0.1]]),
        np.array([[1e-13, 0.0, 0.0]]),
        np.array([False]),
        np.ones(1),
        0.001,
    )
    assert np.array_equal(gradient[:, :3], np.zeros((1, 3)))


def test_refined_reduces_training_loss_and_obeys_budget_padding():
    mesh = trimesh.creation.box(extents=(0.3, 0.1, 0.2))
    cfg = FitConfig(
        method="refined", count=8, resolution=20, samples=256, refine_samples=128, iterations=30
    )
    spheres, report = fit_mesh(mesh, cfg)
    assert 0 < len(spheres) <= 8
    assert report["optimization_final_loss"] < report["optimization_initial_loss"]
    padded, _ = fit_mesh(mesh, replace(cfg, padding=0.003))
    for first, second in zip(spheres, padded):
        assert first.center == second.center
        assert second.radius == pytest.approx(first.radius + 0.003)
    assert "triangle_surface_coverage_certified" not in report


def test_mesh_reference_training_is_explicit_and_single_body_only():
    pytest.importorskip("rtree")
    mesh = trimesh.creation.box(extents=(0.3, 0.1, 0.2))
    config = FitConfig(
        method="refined",
        refine_reference="mesh",
        count=6,
        resolution=20,
        samples=128,
        refine_samples=64,
        iterations=10,
    )
    spheres, report = fit_mesh(mesh, config)
    assert len(spheres) <= 6
    assert "original mesh.contains" in report["optimization_reference"]
    assert report["optimization_final_loss"] <= report["optimization_initial_loss"]
    second = mesh.copy()
    second.apply_translation((1, 0, 0))
    with pytest.raises(ValueError, match="one closed surface body"):
        fit_mesh(trimesh.util.concatenate((mesh, second)), config)


def test_common_domain_captures_exterior_volume_outside_mesh_bbox():
    pytest.importorskip("rtree")
    mesh = trimesh.creation.box()
    models = {
        "small": [Sphere((0.0, 0.0, 0.0), 0.5)],
        "large": [Sphere((0.0, 0.0, 0.0), 2.0)],
    }
    reports = compare_sphere_models(mesh, models, volume_samples=16000, surface_samples=128)
    assert reports["small"]["evaluation_domain_m"] == reports["large"]["evaluation_domain_m"]
    assert reports["large"]["volume_coverage"] == 1
    assert reports["large"]["outside_volume_over_mesh"] > 20
    assert reports["large"]["volume_iou"] < 0.1
    assert reports["small"]["volume_iou"] > 0.4


def test_quality_open_mesh_does_not_claim_solid_metrics():
    mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]])
    result = compare_sphere_models(
        mesh, {"model": [Sphere((0.5, 0.5, 0), 1.0)]}, surface_samples=64, volume_samples=64
    )["model"]
    assert result["surface_coverage"] == 1
    assert "volume_iou" not in result


def test_quality_requires_explicit_multibody_solid_semantics():
    pytest.importorskip("rtree")
    first = trimesh.creation.box()
    second = first.copy()
    second.apply_translation((0.4, 0, 0))
    mesh = trimesh.util.concatenate((first, second))
    models = {"model": [Sphere((0.2, 0.0, 0.0), 1.0)]}
    default = compare_sphere_models(mesh, models, surface_samples=64, volume_samples=4000)["model"]
    assert "volume_iou" not in default
    union = compare_sphere_models(
        mesh, models, surface_samples=64, volume_samples=12000, solid_policy="union-components"
    )["model"]
    parity = compare_sphere_models(
        mesh, models, surface_samples=64, volume_samples=12000, solid_policy="parity"
    )["model"]
    assert union["validation_interior_points"] > parity["validation_interior_points"] * 1.2
    assert union["solid_policy"] == "union-components"


def test_cover_tight_preserves_surface_and_never_enlarges_partition_balls():
    mesh = trimesh.creation.cylinder(radius=0.05, height=0.3, sections=32)
    config = FitConfig(method="cover", count=16, samples=512)
    baseline, _ = fit_mesh(mesh, config)
    tighter, report = fit_mesh(mesh, replace(config, method="cover-tight"))
    assert len(tighter) == len(baseline)
    assert all(new.radius <= old.radius * (1 + 1e-14) for old, new in zip(baseline, tighter))
    assert report["surface_coverage"] == 1
    assert report["triangle_surface_coverage_certified"] is True


def test_fit_asset_parallel_integration_and_cuda_worker_guard():
    from pathlib import Path

    asset = load_asset(Path(__file__).resolve().parents[1] / "examples" / "two_link.urdf")
    cfg = FitConfig(method="surface", count=3, samples=128)
    serial = fit_asset(asset, cfg)
    threaded = fit_asset(asset, cfg, workers=2)
    assert threaded.links == serial.links
    assert threaded.reports == serial.reports
    with pytest.raises(ValueError, match="one GPU owner"):
        fit_asset(asset, FitConfig(device="cuda"), workers=2)


def test_cuda_refinement_tracks_numpy_objective():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("No CUDA runtime")
    from assetspheres.refine import _cuda_adam, _numpy_adam

    rng = np.random.default_rng(42)
    points = rng.uniform(-0.5, 0.5, (128, 3))
    parameters = np.array([[-0.2, 0.0, 0.0, 0.18], [0.2, 0.0, 0.0, 0.18]])
    positive = np.linalg.norm(points, axis=1) < 0.4
    weights = np.full(len(points), 1 / len(points))
    args = (
        parameters.copy(),
        points,
        positive,
        weights,
        0.002,
        np.array([-0.5, -0.5, -0.5, 0.01]),
        np.array([0.5, 0.5, 0.5, 1.0]),
        20,
    )
    cpu, initial_cpu, final_cpu = _numpy_adam(*args)
    gpu, initial_gpu, final_gpu = _cuda_adam(*args)
    assert initial_cpu == pytest.approx(initial_gpu, abs=1e-12)
    assert final_cpu == pytest.approx(final_gpu, abs=1e-10)
    assert np.allclose(cpu, gpu, atol=1e-9)
