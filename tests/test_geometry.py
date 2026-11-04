"""Sampling invariants shared by surface fitting and triangle clustering."""

import numpy as np

from assetspheres.geometry import farthest_point_indices


def test_farthest_points_break_ties_by_input_order_and_cap_count():
    points = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
    original = points.copy()
    indices = farthest_point_indices(points, count=10, seed=0)
    np.testing.assert_array_equal(indices, [4, 3, 1, 2, 0])
    np.testing.assert_array_equal(points, original)


def test_farthest_points_do_not_repeat_coincident_indices():
    points = np.zeros((12, 3))[::2]
    indices = farthest_point_indices(points, count=6, seed=0)
    np.testing.assert_array_equal(indices, [5, 0, 1, 2, 3, 4])
