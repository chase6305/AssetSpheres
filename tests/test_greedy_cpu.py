"""Greedy coverage preserves scalar queries, heap ties and mixed index representations."""

import heapq

import numpy as np
import pytest
from scipy.spatial import cKDTree

from assetspheres import voxel_fit


@pytest.fixture
def small_packed_threshold(monkeypatch):
    monkeypatch.setattr(voxel_fit, "_PACKED_COVER_MIN_SAMPLES", 256)


def reference_greedy(samples, centers, radii, count):
    tree = cKDTree(samples)
    neighborhoods = [
        np.asarray(tree.query_ball_point(center, radius), dtype=int)
        for center, radius in zip(centers, radii)
    ]
    covered = np.zeros(len(samples), dtype=bool)
    priorities = [(-len(indices), index) for index, indices in enumerate(neighborhoods)]
    heapq.heapify(priorities)
    selected = []
    while priorities and len(selected) < count:
        _, index = heapq.heappop(priorities)
        indices = neighborhoods[index]
        uncovered = int(np.count_nonzero(~covered[indices]))
        if uncovered == 0:
            continue
        if priorities and (-uncovered, index) > priorities[0]:
            heapq.heappush(priorities, (-uncovered, index))
            continue
        selected.append(index)
        covered[indices] = True
    return selected


@pytest.mark.usefixtures("small_packed_threshold")
@pytest.mark.parametrize("budget", [-1, 0, 1, 2, 99])
def test_greedy_keeps_selection_inputs_and_global_rng(budget):
    rng = np.random.default_rng(202611125)
    samples = rng.random((514, 3))[::2]
    centers = np.asfortranarray(rng.random((17, 3)))
    radii = rng.uniform(0, 1.2, 17)
    before = [value.tobytes() for value in (samples, centers, radii)]
    samples.flags.writeable = False
    rng_before = np.random.get_state()
    expected = reference_greedy(samples, centers, radii, budget)
    assert (
        voxel_fit._greedy_cover_cpu(
            point_samples=samples,
            candidate_centers=centers,
            candidate_radii=radii,
            sphere_count=budget,
        )
        == expected
    )
    assert before == [value.tobytes() for value in (samples, centers, radii)]
    rng_after = np.random.get_state()
    assert rng_before[0] == rng_after[0] and rng_before[2:] == rng_after[2:]
    np.testing.assert_array_equal(rng_before[1], rng_after[1])


@pytest.mark.usefixtures("small_packed_threshold")
@pytest.mark.parametrize("last_set,expected", [([3, 4], [0, 2]), ([1, 4], [0, 1])])
def test_stale_heap_counts_and_ties_across_bit_boundaries(last_set, expected):
    samples = np.full((257, 5), 10000.0)
    samples[[63, 64, 127, 128, 256]] = np.eye(5)
    centers = np.zeros((3, 5))
    centers[0, [0, 1, 2]] = 1
    centers[1, [0, 1, 3]] = 1
    centers[2, last_set] = 1
    radii = np.sqrt([3.0, 3.0, 2.0])
    assert reference_greedy(samples, centers, radii, 2) == expected
    assert voxel_fit._greedy_cover_cpu(samples, centers, radii, 2) == expected


def test_mixed_neighborhoods_sync_both_coverage_representations(monkeypatch):
    if not voxel_fit._CPYTHON_INT_SIZES:
        pytest.skip("The integer storage optimization targets CPython")
    samples = np.empty((6001, 1))
    samples[:100, 0] = -1
    samples[100:200, 0] = 0
    samples[200, 0] = 1
    samples[201, 0] = -0.5
    samples[202:, 0] = np.arange(len(samples) - 202) + 10
    centers = np.array([[-0.5], [1.0], [0.5], [-0.5]])
    radii = np.array([0.5, 0.0, 0.5, 0.0])
    # Exercise representation synchronization independently of activation policy.
    monkeypatch.setattr(voxel_fit, "_can_pack_cover_neighborhoods", lambda *args: True)
    assert reference_greedy(samples, centers, radii, 4) == [0, 1]
    assert voxel_fit._greedy_cover_cpu(samples, centers, radii, 4) == [0, 1]


def fallback_arguments(kind):
    rng = np.random.default_rng(202611127)
    samples, centers = rng.random((257, 3)), rng.random((17, 3))
    radii = rng.uniform(0, 1.2, 17)
    if kind == "generator":
        return samples, (center for center in centers), (radius for radius in radii), 2
    if kind in {"float32", "longdouble", "object", "nonnative"}:
        dtype = {
            "float32": np.float32,
            "longdouble": np.longdouble,
            "object": object,
            "nonnative": np.dtype(np.float64).newbyteorder("S"),
        }[kind]
        return samples.astype(dtype), centers.astype(dtype), radii.astype(dtype), 2
    if kind == "zip-unused-nan":
        centers[-1, 0] = np.nan
        return samples, centers, radii[:-1], 2
    if kind == "invalid-shape":
        return samples, centers[:, :2], radii, 2
    if kind == "nan-empty-candidates":
        samples[0, 0] = np.nan
        return samples, np.empty((0, 3)), np.empty(0), 2
    if kind == "nan-zero-budget":
        centers[-1, 0] = np.nan
        return samples, centers, radii, 0
    return samples.tolist(), centers.tolist(), radii.tolist(), 2


@pytest.mark.parametrize(
    "kind",
    [
        "list",
        "generator",
        "float32",
        "longdouble",
        "object",
        "nonnative",
        "zip-unused-nan",
        "invalid-shape",
        "nan-empty-candidates",
        "nan-zero-budget",
    ],
)
@pytest.mark.usefixtures("small_packed_threshold")
def test_scalar_fallback_preserves_iterables_dtypes_and_rejections(kind):
    try:
        expected = reference_greedy(*fallback_arguments(kind))
    except (ValueError, TypeError) as error:
        with pytest.raises(type(error)) as caught:
            voxel_fit._greedy_cover_cpu(*fallback_arguments(kind))
        assert str(caught.value) == str(error)
    else:
        assert voxel_fit._greedy_cover_cpu(*fallback_arguments(kind)) == expected


def test_unknown_integer_storage_and_budget_use_the_array_reference(monkeypatch):
    rng = np.random.default_rng(202611128)
    samples, centers = rng.random((32768, 3)), rng.random((17, 3))
    radii = rng.uniform(0, 1.2, 17)
    expected = reference_greedy(samples, centers, radii, 2)

    def forbidden(*args):
        raise AssertionError("integer storage sizing must not run")

    with monkeypatch.context() as guarded:
        guarded.setattr(voxel_fit, "_CPYTHON_INT_SIZES", False)
        guarded.setattr(voxel_fit, "_packed_cover_state_bytes", forbidden)
        assert voxel_fit._greedy_cover_cpu(samples, centers, radii, 2) == expected
    original_array = voxel_fit._greedy_cover_array_cpu
    calls = []

    def tracked_array(*args):
        calls.append(True)
        return original_array(*args)

    monkeypatch.setattr(voxel_fit, "_PACKED_COVER_LIMIT_BYTES", 0)
    monkeypatch.setattr(voxel_fit, "_greedy_cover_array_cpu", tracked_array)
    assert voxel_fit._greedy_cover_cpu(samples, centers, radii, 2) == expected
    assert calls == [True]


@pytest.mark.parametrize("sample_count", [6000, 32767, 32768])
def test_public_dispatch_keeps_default_samples_and_checks_large_sample_boundary(
    monkeypatch, sample_count
):
    if not voxel_fit._CPYTHON_INT_SIZES:
        pytest.skip("Packed dispatch is enabled only on CPython")
    samples = np.arange(sample_count, dtype=np.float64)[:, None]
    centers = samples[[0, sample_count // 2, -1]].copy()
    radii = np.zeros(3)
    original = voxel_fit._greedy_cover_array_cpu
    calls = []

    def tracked_array(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(voxel_fit, "_greedy_cover_array_cpu", tracked_array)
    assert voxel_fit._greedy_cover_cpu(samples, centers, radii, 2) == [0, 1]
    assert bool(calls) == (sample_count < 32768)


def test_large_one_point_neighborhoods_do_not_allocate_full_packing_scratch(monkeypatch):
    if not voxel_fit._CPYTHON_INT_SIZES:
        pytest.skip("Packed dispatch is enabled only on CPython")
    samples = np.arange(32768, dtype=np.float64)[:, None]
    centers = samples[[0, 8192, -1]].copy()
    radii = np.zeros(3)
    expected = reference_greedy(samples, centers, radii, 4)
    original_empty = np.empty

    def tracked_empty(shape, *args, **kwargs):
        dtype = kwargs.get("dtype", args[0] if args else None)
        if np.dtype(dtype) == np.dtype(np.uint8):
            size = int(np.prod(shape)) if isinstance(shape, tuple) else int(shape)
            assert size < len(samples), "sparse coverage must not allocate full packing scratch"
        return original_empty(shape, *args, **kwargs)

    def forbidden(*args):
        raise AssertionError("large sparse input should use hybrid arrays without packing")

    monkeypatch.setattr(voxel_fit.np, "empty", tracked_empty)
    monkeypatch.setattr(voxel_fit, "_greedy_cover_array_cpu", forbidden)
    monkeypatch.setattr(voxel_fit, "_pack_cover_indices", forbidden)
    assert voxel_fit._greedy_cover_cpu(samples, centers, radii, 4) == expected


@pytest.mark.parametrize("budget", [0, 2])
def test_optimized_path_still_queries_each_candidate_as_a_scalar(monkeypatch, budget):
    samples = np.arange(32768, dtype=np.float64)[:, None]
    centers = samples[np.arange(17) * 1000].copy()
    radii = np.zeros(17)
    expected = reference_greedy(samples, centers, radii, budget)
    queried = []

    class ScalarTree:
        def __init__(self, points):
            self.tree = cKDTree(points)

        def query_ball_point(self, center, radius, **kwargs):
            assert center.ndim == 1 and kwargs == {}
            queried.append(center.copy())
            return self.tree.query_ball_point(center, radius)

    monkeypatch.setattr(voxel_fit, "cKDTree", ScalarTree)
    assert voxel_fit._greedy_cover_cpu(samples, centers, radii, budget) == expected
    np.testing.assert_array_equal(np.asarray(queried), centers)
