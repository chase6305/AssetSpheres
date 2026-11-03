import subprocess
import sys

import numpy as np
import pytest

from assetspheres.tensor_backend import greedy_cover, point_sphere_gaps

torch = pytest.importorskip("torch")
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def numpy_greedy(points, centers, radii, count):
    coverage = np.linalg.norm(points[None] - centers[:, None], axis=2) <= radii[:, None]
    uncovered = np.ones(len(points), dtype=bool)
    available = np.ones(len(centers), dtype=bool)
    selected = []
    for _ in range(min(count, len(centers))):
        scores = (coverage & uncovered).sum(axis=1)
        scores[~available] = -1
        best = int(scores.argmax())
        if scores[best] <= 0:
            break
        selected.append(best)
        available[best] = False
        uncovered &= ~coverage[best]
    return selected


@pytest.mark.parametrize("device", DEVICES)
def test_gaps_match_numpy_across_chunks(device):
    rng = np.random.default_rng(123)
    points, centers = rng.normal(size=(103, 3)), rng.normal(size=(137, 3))
    radii = rng.uniform(0.1, 1.0, len(centers))
    expected = (np.linalg.norm(points[:, None] - centers[None], axis=2) - radii).min(axis=1)
    actual = point_sphere_gaps(points, centers, radii, device=device, chunk_size=17)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("budget", [2048, 1024**2])
def test_greedy_matches_reference_dense_and_streaming(device, budget):
    rng = np.random.default_rng(18)
    points, centers = rng.normal(size=(67, 3)), rng.normal(size=(23, 3))
    radii = rng.uniform(0.1, 1.5, len(centers))
    expected = numpy_greedy(points, centers, radii, 7)
    actual = greedy_cover(points, centers, radii, count=7, device=device, max_matrix_bytes=budget)
    assert actual.tolist() == expected


@pytest.mark.parametrize("device", DEVICES)
def test_boundary_precision_uses_float64(device):
    points = np.array([[1.0, 0.0, 0.0], [1.0 + 1e-8, 0.0, 0.0], [1.0 - 1e-8, 0.0, 0.0]])
    centers, radii = np.zeros((1, 3)), np.ones(1)
    gaps = point_sphere_gaps(points, centers, radii, device=device)
    assert gaps[0] == 0 and gaps[1] > 0 and gaps[2] < 0
    # A float32 conversion would incorrectly cover the exterior point as well.
    assert greedy_cover(points[1:2], centers, radii, count=1, device=device).size == 0


@pytest.mark.parametrize("device", DEVICES)
def test_ties_choose_lowest_index_and_stop_without_gain(device):
    points = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    centers = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    assert greedy_cover(points, centers, np.ones(3), count=3, device=device).tolist() == [0, 2]


@pytest.mark.parametrize("device", DEVICES)
def test_empty_inputs(device):
    assert point_sphere_gaps(np.empty((0, 3)), np.zeros((1, 3)), [1], device=device).size == 0
    assert np.isinf(point_sphere_gaps(np.zeros((2, 3)), np.empty((0, 3)), [], device=device)).all()
    assert greedy_cover(np.zeros((2, 3)), np.empty((0, 3)), [], count=1, device=device).size == 0


def test_validation_and_working_memory_limit():
    with pytest.raises(ValueError, match="positive integer"):
        greedy_cover([[0, 0, 0]], [[0, 0, 0]], [1], count=0)
    with pytest.raises(ValueError, match="at least"):
        greedy_cover(
            np.zeros((100, 3)), np.zeros((20, 3)), np.ones(20), count=1, max_matrix_bytes=512
        )
    with pytest.raises(ValueError, match="matching"):
        point_sphere_gaps([[0, 0, 0]], [[0, 0, 0]], [])
    with pytest.raises(ValueError, match="finite"):
        point_sphere_gaps([[float("nan"), 0, 0]], [[0, 0, 0]], [1])
    with pytest.raises(ValueError, match="device"):
        point_sphere_gaps([[0, 0, 0]], [[0, 0, 0]], [1], device="automatic")


def test_cuda_request_does_not_silently_fall_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="explicitly requested"):
        point_sphere_gaps([[0, 0, 0]], [[0, 0, 0]], [1], device="cuda")


def test_import_does_not_import_torch():
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import assetspheres.tensor_backend; assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_cast_and_reduction_temporaries_respect_planned_budget():
    rng = np.random.default_rng(11)
    points, centers = rng.random((5000, 3)), rng.random((256, 3))
    radii = np.full(256, 0.2)
    budget = 4 * 1024**2
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    greedy_cover(points, centers, radii, count=5, device="cuda", max_matrix_bytes=budget)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - baseline
    # Exclude linear inputs and permit small allocator rounding / runtime metadata.
    inputs = points.nbytes + centers.nbytes + radii.nbytes
    assert peak < budget + inputs + 256 * 1024


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_complete_voxel_fit_has_same_cpu_cuda_spheres():
    import trimesh

    from assetspheres import FitConfig, fit_mesh

    mesh = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    cpu, cpu_report = fit_mesh(mesh, FitConfig(count=8, resolution=16, samples=512, padding=0.002))
    cuda, cuda_report = fit_mesh(
        mesh, FitConfig(count=8, resolution=16, samples=512, padding=0.002, device="cuda")
    )
    assert cpu == cuda
    assert cpu_report["surface_coverage"] == cuda_report["surface_coverage"]
    assert cpu_report["voxel_center_coverage"] == cuda_report["voxel_center_coverage"]
