"""Tests for numcompute_stream.stats — RunningStats (Welford)."""

from __future__ import annotations

import numpy as np
import pytest

from numcompute_stream.stats import RunningStats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rs(*arrays: np.ndarray) -> RunningStats:
    rs = RunningStats()
    for arr in arrays:
        rs.update(arr)
    return rs


# ---------------------------------------------------------------------------
# Basic single-batch accuracy
# ---------------------------------------------------------------------------

class TestRunningStatsSingleBatch:
    def test_mean_1d(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rs = _make_rs(data)
        np.testing.assert_allclose(rs.mean, [3.0])

    def test_mean_2d(self):
        X = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
        rs = _make_rs(X)
        np.testing.assert_allclose(rs.mean, [3.0, 20.0])

    def test_var_matches_numpy(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, 4))
        rs = _make_rs(X)
        np.testing.assert_allclose(rs.var, X.var(axis=0), rtol=1e-10)

    def test_std_matches_numpy(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((100, 3))
        rs = _make_rs(X)
        np.testing.assert_allclose(rs.std, X.std(axis=0), rtol=1e-10)

    def test_unbiased_var_matches_numpy(self):
        rng = np.random.default_rng(2)
        X = rng.standard_normal((50, 2))
        rs = _make_rs(X)
        np.testing.assert_allclose(rs.var_unbiased, X.var(axis=0, ddof=1), rtol=1e-10)

    def test_count_after_single_batch(self):
        X = np.ones((7, 3))
        rs = _make_rs(X)
        assert rs.count == 7


# ---------------------------------------------------------------------------
# Incremental updates must match one-shot
# ---------------------------------------------------------------------------

class TestRunningStatsIncremental:
    def test_chunked_mean_matches_oneshot(self):
        rng = np.random.default_rng(3)
        X = rng.standard_normal((300, 5))
        rs_full = _make_rs(X)

        rs_chunked = RunningStats()
        for start in range(0, 300, 50):
            rs_chunked.update(X[start:start + 50])

        np.testing.assert_allclose(rs_full.mean, rs_chunked.mean, rtol=1e-12)
        np.testing.assert_allclose(rs_full.var, rs_chunked.var, rtol=1e-12)

    def test_count_accumulates(self):
        rs = RunningStats()
        for _ in range(5):
            rs.update(np.ones((20, 2)))
        assert rs.count == 100


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestRunningStatsEdgeCases:
    def test_no_data_mean_raises(self):
        with pytest.raises(ValueError, match="no data"):
            RunningStats().mean

    def test_no_data_var_raises(self):
        with pytest.raises(ValueError, match="no data"):
            RunningStats().var

    def test_single_sample_var_is_zero(self):
        rs = _make_rs(np.array([[5.0, 7.0]]))
        np.testing.assert_array_equal(rs.var, [0.0, 0.0])

    def test_single_sample_unbiased_var_raises(self):
        rs = _make_rs(np.array([[1.0]]))
        with pytest.raises(ValueError, match="fewer than 2"):
            rs.var_unbiased

    def test_constant_stream_var_zero(self):
        X = np.full((50, 3), fill_value=7.0)
        rs = _make_rs(X)
        np.testing.assert_allclose(rs.var, [0.0, 0.0, 0.0], atol=1e-12)

    def test_nan_in_input_raises(self):
        X = np.array([[1.0, np.nan], [2.0, 3.0]])
        with pytest.raises(ValueError, match="NaN"):
            RunningStats().update(X)

    def test_feature_mismatch_raises(self):
        rs = _make_rs(np.ones((5, 3)))
        with pytest.raises(ValueError, match="feature"):
            rs.update(np.ones((5, 4)))

    def test_reset_clears_state(self):
        rs = _make_rs(np.ones((10, 2)))
        rs.reset()
        assert rs.count == 0
        with pytest.raises(ValueError):
            rs.mean

    def test_reset_then_reuse(self):
        rs = _make_rs(np.ones((5, 2)) * 99.0)
        rs.reset()
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        rs.update(X)
        np.testing.assert_allclose(rs.mean, [2.0, 3.0])


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

class TestRunningStatsMerge:
    def test_merge_mean_equals_combined(self):
        rng = np.random.default_rng(7)
        X1 = rng.standard_normal((80, 4))
        X2 = rng.standard_normal((120, 4))

        rs1 = _make_rs(X1)
        rs2 = _make_rs(X2)
        merged = rs1.merge(rs2)

        X_all = np.vstack([X1, X2])
        np.testing.assert_allclose(merged.mean, X_all.mean(axis=0), rtol=1e-12)
        np.testing.assert_allclose(merged.var, X_all.var(axis=0), rtol=1e-12)
        assert merged.count == 200

    def test_merge_with_empty_left(self):
        rs_empty = RunningStats()
        X = np.ones((5, 2))
        rs_full = _make_rs(X)
        merged = rs_empty.merge(rs_full)
        np.testing.assert_allclose(merged.mean, rs_full.mean)

    def test_merge_feature_mismatch_raises(self):
        rs1 = _make_rs(np.ones((5, 2)))
        rs2 = _make_rs(np.ones((5, 3)))
        with pytest.raises(ValueError, match="n_features"):
            rs1.merge(rs2)
