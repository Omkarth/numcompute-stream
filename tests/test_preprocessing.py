"""Tests for numcompute_stream.preprocessing."""

from __future__ import annotations

import numpy as np
import pytest

from numcompute_stream.preprocessing import Imputer, OneHotEncoder, StandardScaler


# Helpers

def _make_X(seed: int = 0, n: int = 50, p: int = 3) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, p))


# ===========================================================================
# StandardScaler
# ===========================================================================

class TestStandardScaler:
    def test_basic_fit_transform(self):
        X = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
        sc = StandardScaler()
        sc.partial_fit(X)
        Xt = sc.transform(X)
        np.testing.assert_allclose(Xt.mean(axis=0), 0.0, atol=1e-10)

    def test_multi_chunk_matches_single_pass(self):
        """Fitting in two chunks must yield the same mean/scale as one pass."""
        rng = np.random.default_rng(7)
        X = rng.standard_normal((100, 4))
        sc_single = StandardScaler().partial_fit(X)
        sc_chunks = StandardScaler()
        sc_chunks.partial_fit(X[:50])
        sc_chunks.partial_fit(X[50:])
        np.testing.assert_allclose(sc_single.mean_, sc_chunks.mean_, atol=1e-10)
        np.testing.assert_allclose(sc_single.scale_, sc_chunks.scale_, atol=1e-10)

    def test_single_row_chunk(self):
        """Partial fitting one row at a time must not raise."""
        sc = StandardScaler()
        for i in range(5):
            sc.partial_fit(np.array([[float(i), float(i * 2)]]))
        Xt = sc.transform(np.array([[1.0, 2.0]]))
        assert Xt.shape == (1, 2)

    def test_nan_passthrough_in_transform(self):
        """NaN values in X must remain NaN after transform."""
        sc = StandardScaler().partial_fit(np.array([[1.0, 2.0], [3.0, 4.0]]))
        X = np.array([[np.nan, 2.0], [1.0, np.nan]])
        Xt = sc.transform(X)
        assert np.isnan(Xt[0, 0])
        assert np.isnan(Xt[1, 1])
        assert not np.isnan(Xt[0, 1])
        assert not np.isnan(Xt[1, 0])

    def test_all_nan_column(self):
        """A column that is always NaN during fit must output NaN in transform."""
        X_fit = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])
        sc = StandardScaler().partial_fit(X_fit)
        X_test = np.array([[2.0, 5.0]])
        Xt = sc.transform(X_test)
        assert not np.isnan(Xt[0, 0])   # first column has stats
        assert np.isnan(Xt[0, 1])        # second column never fitted

    def test_constant_column_outputs_zero(self):
        """A constant column (std=0) should map to 0.0, not NaN or inf."""
        X = np.array([[3.0], [3.0], [3.0]])
        sc = StandardScaler().partial_fit(X)
        Xt = sc.transform(X)
        np.testing.assert_array_equal(Xt, np.zeros((3, 1)))

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError, match="not been fitted"):
            StandardScaler().transform(np.array([[1.0]]))

    def test_copy_false_modifies_in_place(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        sc = StandardScaler(copy=False).partial_fit(X.copy())
        original = X.copy()
        Xt = sc.transform(X)
        assert Xt is X                         # same object
        assert not np.array_equal(Xt, original)  # values changed


# ===========================================================================
# Imputer — mean strategy
# ===========================================================================

class TestImputer:
    def test_mean_basic(self):
        X_fit = np.array([[1.0, 10.0], [3.0, 20.0]])
        imp = Imputer(strategy="mean").partial_fit(X_fit)
        X = np.array([[np.nan, 10.0], [3.0, np.nan]])
        Xt = imp.transform(X)
        assert not np.any(np.isnan(Xt))
        assert Xt[0, 0] == pytest.approx(2.0)   # mean of [1, 3]
        assert Xt[1, 1] == pytest.approx(15.0)  # mean of [10, 20]

    def test_mean_multi_chunk(self):
        """Running mean must be consistent across two-chunk fit."""
        rng = np.random.default_rng(3)
        X = rng.standard_normal((60, 2))
        imp_single = Imputer("mean").partial_fit(X)
        imp_chunks = Imputer("mean")
        imp_chunks.partial_fit(X[:30])
        imp_chunks.partial_fit(X[30:])
        np.testing.assert_allclose(
            imp_single.statistics_, imp_chunks.statistics_, atol=1e-10
        )

    def test_mean_single_row_chunk(self):
        imp = Imputer("mean")
        for i in range(1, 6):
            imp.partial_fit(np.array([[float(i)]]))
        X = np.array([[np.nan]])
        Xt = imp.transform(X)
        assert Xt[0, 0] == pytest.approx(3.0)   # mean of [1,2,3,4,5]

    def test_mean_all_nan_column_fills_zero(self):
        """Column that was always NaN during fit gets fill value 0.0."""
        X_fit = np.array([[1.0, np.nan], [2.0, np.nan]])
        imp = Imputer("mean").partial_fit(X_fit)
        X = np.array([[np.nan, np.nan]])
        Xt = imp.transform(X)
        assert Xt[0, 0] == pytest.approx(1.5)
        assert Xt[0, 1] == pytest.approx(0.0)

    def test_mean_no_nans_idempotent(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        imp = Imputer("mean").partial_fit(X)
        Xt = imp.transform(X.copy())
        np.testing.assert_array_equal(Xt, X)

    def test_most_frequent_basic(self):
        X_fit = np.array([[1.0], [2.0], [1.0], [3.0], [1.0]])
        imp = Imputer("most_frequent").partial_fit(X_fit)
        X = np.array([[np.nan], [2.0]])
        Xt = imp.transform(X)
        assert Xt[0, 0] == pytest.approx(1.0)   # mode = 1.0
        assert Xt[1, 0] == pytest.approx(2.0)   # no NaN, unchanged

    def test_most_frequent_multi_chunk(self):
        """Mode must accumulate correctly across chunks."""
        imp = Imputer("most_frequent")
        imp.partial_fit(np.array([[1.0], [2.0], [1.0]]))  # 1→2, 2→1
        imp.partial_fit(np.array([[2.0], [2.0]]))          # 1→2, 2→3
        Xt = imp.transform(np.array([[np.nan]]))
        assert Xt[0, 0] == pytest.approx(2.0)

    def test_most_frequent_all_nan_column_stays_nan(self):
        """Column seen only as NaN: fill value is undefined, leave NaN."""
        X_fit = np.array([[1.0, np.nan], [2.0, np.nan]])
        imp = Imputer("most_frequent").partial_fit(X_fit)
        X = np.array([[np.nan, np.nan]])
        Xt = imp.transform(X)
        assert Xt[0, 0] == pytest.approx(1.5) or Xt[0, 0] in (1.0, 2.0)
        assert np.isnan(Xt[0, 1])

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="strategy"):
            Imputer(strategy="median")

    def test_imputer_unfitted_raises(self):
        with pytest.raises(RuntimeError, match="not been fitted"):
            Imputer().transform(np.array([[1.0]]))


# ===========================================================================
# OneHotEncoder
# ===========================================================================

class TestOneHotEncoder:
    def test_basic_single_column(self):
        X = np.array([[0.0], [1.0], [2.0]])
        enc = OneHotEncoder().partial_fit(X)
        Xt = enc.transform(X)
        assert Xt.shape == (3, 3)
        np.testing.assert_array_equal(Xt[0], [1, 0, 0])
        np.testing.assert_array_equal(Xt[1], [0, 1, 0])
        np.testing.assert_array_equal(Xt[2], [0, 0, 1])

    def test_multi_column(self):
        X = np.array([[0.0, 10.0], [1.0, 20.0]])
        enc = OneHotEncoder().partial_fit(X)
        Xt = enc.transform(X)
        # 2 cats in col0 + 2 cats in col1 = 4 columns
        assert Xt.shape == (2, 4)
        assert enc.n_features_out_ == 4

    def test_growing_categories_widens_output(self):
        """
        After a second partial_fit introduces a new category, the next
        transform must return a wider array than the previous one.
        """
        enc = OneHotEncoder()
        X1 = np.array([[0.0], [1.0]])
        enc.partial_fit(X1)
        Xt1 = enc.transform(X1)
        assert Xt1.shape == (2, 2)

        X2 = np.array([[2.0]])           # new category
        enc.partial_fit(X2)
        Xt2 = enc.transform(np.array([[0.0], [1.0], [2.0]]))
        assert Xt2.shape == (3, 3)       # output is now wider

    def test_n_features_out_grows(self):
        enc = OneHotEncoder()
        enc.partial_fit(np.array([[0.0], [1.0]]))
        assert enc.n_features_out_ == 2
        enc.partial_fit(np.array([[2.0], [3.0]]))
        assert enc.n_features_out_ == 4

    def test_never_seen_category_all_zeros(self):
        """A value not present during partial_fit → all-zero indicator row."""
        enc = OneHotEncoder().partial_fit(np.array([[0.0], [1.0]]))
        Xt = enc.transform(np.array([[99.0]]))
        np.testing.assert_array_equal(Xt[0], [0, 0])

    def test_nan_in_transform_all_zeros(self):
        """NaN in transform produces an all-zero indicator row for that col."""
        enc = OneHotEncoder().partial_fit(np.array([[0.0], [1.0], [2.0]]))
        Xt = enc.transform(np.array([[np.nan]]))
        np.testing.assert_array_equal(Xt[0], [0, 0, 0])

    def test_single_row_chunk(self):
        """Fitting one row at a time must accumulate categories correctly."""
        enc = OneHotEncoder()
        for v in [0.0, 1.0, 2.0]:
            enc.partial_fit(np.array([[v]]))
        assert enc.n_features_out_ == 3
        Xt = enc.transform(np.array([[1.0]]))
        np.testing.assert_array_equal(Xt[0], [0, 1, 0])

    def test_nan_in_partial_fit_ignored(self):
        """NaN values during partial_fit must not appear as a category."""
        enc = OneHotEncoder()
        enc.partial_fit(np.array([[0.0], [np.nan], [1.0]]))
        assert enc.categories_[0] == [0.0, 1.0]  # NaN not included

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError, match="not been fitted"):
            OneHotEncoder().transform(np.array([[1.0]]))

    def test_categories_sorted(self):
        """Categories must always be sorted regardless of insertion order."""
        enc = OneHotEncoder()
        enc.partial_fit(np.array([[3.0], [1.0]]))
        enc.partial_fit(np.array([[2.0]]))
        assert enc.categories_[0] == [1.0, 2.0, 3.0]
