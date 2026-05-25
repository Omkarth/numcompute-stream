"""Tests for numcompute_stream.utils."""

from __future__ import annotations

import numpy as np
import pytest

from numcompute_stream.utils import (
    any_nan,
    check_array,
    check_X_y,
    nan_col_counts,
    nan_mask,
    nan_row_mask,
    safe_divide,
)


# check_array

class TestCheckArray:
    def test_valid_2d(self):
        X = np.ones((5, 3))
        result = check_array(X)
        assert result is X

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            check_array(np.ones(5))  # 1-D, default ndim=2

    def test_non_ndarray_raises(self):
        with pytest.raises(TypeError, match="ndarray"):
            check_array([[1, 2], [3, 4]])

    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError, match="sample"):
            check_array(np.ones((0, 3)), min_samples=1)

    def test_too_few_features_raises(self):
        with pytest.raises(ValueError, match="feature"):
            check_array(np.ones((5, 0)), min_features=1)

    def test_nan_disallowed_raises(self):
        X = np.array([[1.0, np.nan], [2.0, 3.0]])
        with pytest.raises(ValueError, match="NaN"):
            check_array(X, allow_nan=False)

    def test_nan_allowed_passes(self):
        X = np.array([[1.0, np.nan]])
        result = check_array(X, allow_nan=True)
        assert result.shape == (1, 2)

    def test_dtype_cast(self):
        X = np.ones((3, 2), dtype=np.int32)
        result = check_array(X, dtype=np.float64)
        assert result.dtype == np.float64

    def test_custom_name_in_error(self):
        with pytest.raises(ValueError, match="MyData"):
            check_array(np.ones(3), name="MyData")


# check_X_y

class TestCheckXy:
    def test_valid(self):
        X = np.ones((10, 4))
        y = np.zeros(10)
        X_out, y_out = check_X_y(X, y)
        assert X_out.shape == (10, 4)
        assert y_out.shape == (10,)

    def test_mismatched_samples_raises(self):
        with pytest.raises(ValueError, match="same number of samples"):
            check_X_y(np.ones((10, 4)), np.zeros(9))

    def test_nan_in_y_raises(self):
        X = np.ones((5, 2))
        y = np.array([0.0, 1.0, np.nan, 0.0, 1.0])
        with pytest.raises(ValueError, match="NaN"):
            check_X_y(X, y)

    def test_nan_in_X_allowed(self):
        X = np.array([[np.nan, 1.0], [2.0, 3.0]])
        y = np.array([0.0, 1.0])
        X_out, y_out = check_X_y(X, y, allow_nan=True)
        assert X_out.shape == (2, 2)


# NaN helpers

class TestNanHelpers:
    def test_any_nan_true(self):
        assert any_nan(np.array([1.0, np.nan, 3.0])) is True

    def test_any_nan_false(self):
        assert any_nan(np.array([1.0, 2.0, 3.0])) is False

    def test_nan_mask_shape(self):
        X = np.array([[1.0, np.nan], [np.nan, 4.0]])
        mask = nan_mask(X)
        assert mask.shape == (2, 2)
        assert mask[0, 1] and mask[1, 0]
        assert not mask[0, 0] and not mask[1, 1]

    def test_nan_col_counts(self):
        X = np.array([
            [1.0, np.nan, 3.0],
            [np.nan, 2.0, np.nan],
            [3.0, np.nan, 5.0],
        ])
        counts = nan_col_counts(X)
        np.testing.assert_array_equal(counts, [1, 2, 1])

    def test_nan_col_counts_1d_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            nan_col_counts(np.array([1.0, np.nan]))

    def test_nan_row_mask(self):
        X = np.array([
            [1.0, 2.0],
            [np.nan, 2.0],
            [3.0, 4.0],
        ])
        mask = nan_row_mask(X)
        np.testing.assert_array_equal(mask, [False, True, False])

    def test_nan_row_mask_1d_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            nan_row_mask(np.array([1.0, np.nan]))


# safe_divide

class TestSafeDivide:
    def test_normal_division(self):
        a = np.array([6.0, 8.0])
        b = np.array([2.0, 4.0])
        np.testing.assert_allclose(safe_divide(a, b), [3.0, 2.0])

    def test_zero_denominator_fills(self):
        a = np.array([5.0, 0.0])
        b = np.array([0.0, 0.0])
        result = safe_divide(a, b, fill=-1.0)
        np.testing.assert_array_equal(result, [-1.0, -1.0])

    def test_mixed(self):
        a = np.array([4.0, 3.0])
        b = np.array([2.0, 0.0])
        result = safe_divide(a, b, fill=0.0)
        np.testing.assert_allclose(result, [2.0, 0.0])
