"""Tests for numcompute_stream.tree — structure, leaf statistics, and split logic."""

from __future__ import annotations

import numpy as np
import pytest

from numcompute_stream.tree import (
    HoeffdingTreeClassifier,
    LeafStats,
    Node,
    _entropy,
    _gini,
    _hoeffding_bound,
)


# Impurity functions

class TestGini:
    def test_pure_node(self):
        counts = np.array([10.0, 0.0])
        assert _gini(counts) == pytest.approx(0.0)

    def test_uniform_binary(self):
        counts = np.array([5.0, 5.0])
        assert _gini(counts) == pytest.approx(0.5)

    def test_uniform_three_class(self):
        counts = np.array([4.0, 4.0, 4.0])
        # 1 - 3*(1/3)^2 = 1 - 1/3 = 2/3
        assert _gini(counts) == pytest.approx(2 / 3)

    def test_empty_counts(self):
        assert _gini(np.array([0.0, 0.0])) == pytest.approx(0.0)


class TestEntropy:
    def test_pure_node(self):
        counts = np.array([10.0, 0.0])
        assert _entropy(counts) == pytest.approx(0.0)

    def test_uniform_binary(self):
        counts = np.array([5.0, 5.0])
        assert _entropy(counts) == pytest.approx(1.0)

    def test_uniform_three_class(self):
        counts = np.array([3.0, 3.0, 3.0])
        # -3 * (1/3)*log2(1/3) = log2(3)
        assert _entropy(counts) == pytest.approx(np.log2(3))

    def test_empty_counts(self):
        assert _entropy(np.array([0.0, 0.0])) == pytest.approx(0.0)


# Node

class TestNode:
    def test_new_node_is_leaf(self):
        stats = LeafStats(n_features=3)
        node = Node(leaf_stats=stats, depth=0)
        assert node.is_leaf
        assert node.leaf_stats is stats
        assert node.split_feature is None

    def test_to_internal_flips_is_leaf(self):
        stats = LeafStats(n_features=2)
        node = Node(leaf_stats=stats, depth=0)

        left = Node(leaf_stats=LeafStats(n_features=2), depth=1)
        right = Node(leaf_stats=LeafStats(n_features=2), depth=1)

        node.to_internal(
            split_feature=1,
            split_threshold=0.5,
            left_child=left,
            right_child=right,
        )

        assert not node.is_leaf
        assert node.split_feature == 1
        assert node.split_threshold == pytest.approx(0.5)
        assert node.leaf_stats is None
        assert node.left_child is left
        assert node.right_child is right

    def test_depth_stored(self):
        node = Node(depth=3)
        assert node.depth == 3


# LeafStats

class TestLeafStats:
    def test_empty_majority_class_is_none(self):
        ls = LeafStats(n_features=2)
        assert ls.majority_class() is None

    def test_accumulation_single_class(self):
        ls = LeafStats(n_features=2)
        x = np.array([1.0, 2.0])
        for _ in range(5):
            ls.update(x, label=0)
        assert ls.n_samples == 5
        assert ls.class_counts == {0: 5}
        assert ls.majority_class() == 0

    def test_accumulation_two_classes(self):
        ls = LeafStats(n_features=2)
        for _ in range(3):
            ls.update(np.array([1.0, 0.5]), label=0)
        for _ in range(7):
            ls.update(np.array([2.0, 1.5]), label=1)
        assert ls.n_samples == 10
        assert ls.class_counts == {0: 3, 1: 7}
        assert ls.majority_class() == 1

    def test_feature_stats_populated(self):
        ls = LeafStats(n_features=2)
        xs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        for x in xs:
            ls.update(x, label=0)
        rs = ls.get_feature_stats(0, 0)
        assert rs is not None
        assert rs.count == 3
        assert rs.mean == pytest.approx(np.array([3.0]))

    def test_nan_skips_feature_stats(self):
        ls = LeafStats(n_features=2)
        ls.update(np.array([np.nan, 2.0]), label=0)
        # NaN feature 0 → no RunningStats for (0, 0)
        assert ls.get_feature_stats(0, 0) is None
        # Feature 1 should still be tracked
        assert ls.get_feature_stats(1, 0) is not None
        assert ls.n_samples == 1
        assert ls.class_counts == {0: 1}

    def test_nan_sample_counted(self):
        ls = LeafStats(n_features=3)
        ls.update(np.array([np.nan, np.nan, np.nan]), label=1)
        assert ls.n_samples == 1
        assert ls.class_counts == {1: 1}

    def test_split_eligible_threshold(self):
        ls = LeafStats(n_features=1, grace_period=5)
        x = np.array([1.0])
        assert not ls.split_eligible
        for _ in range(5):
            ls.update(x, label=0)
        assert ls.split_eligible

    def test_class_counts_array_aligned(self):
        ls = LeafStats(n_features=1)
        for _ in range(3):
            ls.update(np.array([0.0]), label=0)
        for _ in range(7):
            ls.update(np.array([1.0]), label=2)
        classes = np.array([0, 1, 2])
        arr = ls.class_counts_array(classes)
        np.testing.assert_array_equal(arr, [3.0, 0.0, 7.0])


# HoeffdingTreeClassifier — structure and partial_fit stub

class TestHoeffdingTreeClassifier:
    def test_init_defaults(self):
        clf = HoeffdingTreeClassifier()
        assert clf.criterion == "gini"
        assert clf.delta == pytest.approx(1e-7)
        assert clf.max_depth is None

    def test_invalid_criterion_raises(self):
        with pytest.raises(ValueError, match="criterion"):
            HoeffdingTreeClassifier(criterion="mse")

    def test_predict_before_fit_raises(self):
        clf = HoeffdingTreeClassifier()
        with pytest.raises(ValueError, match="partial_fit"):
            clf.predict(np.array([[1.0, 2.0]]))

    def test_predict_root_leaf_returns_majority(self):
        clf = HoeffdingTreeClassifier()
        X = np.array([[1.0, 2.0]] * 7 + [[3.0, 4.0]] * 3)
        y = np.array([0] * 7 + [1] * 3)
        clf.partial_fit(X, y)
        preds = clf.predict(X)
        # Only one leaf — everyone gets majority class 0
        assert np.all(preds == 0)

    def test_predict_minority_samples_still_majority(self):
        clf = HoeffdingTreeClassifier()
        X = np.ones((10, 2))
        y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])  # class 0 is majority
        clf.partial_fit(X, y)
        preds = clf.predict(np.ones((5, 2)))
        assert np.all(preds == 0)

    def test_partial_fit_updates_leaf_counts(self):
        clf = HoeffdingTreeClassifier()
        X = np.random.default_rng(0).standard_normal((20, 3))
        y = np.array([0] * 12 + [1] * 8)
        clf.partial_fit(X, y)
        leaf = clf._root
        assert leaf.is_leaf
        assert leaf.leaf_stats.n_samples == 20
        counts = leaf.leaf_stats.class_counts
        assert counts[0] == 12
        assert counts[1] == 8

    def test_partial_fit_two_chunks(self):
        clf = HoeffdingTreeClassifier()
        X1 = np.ones((5, 2))
        y1 = np.zeros(5, dtype=int)
        X2 = np.ones((5, 2)) * 2
        y2 = np.ones(5, dtype=int)
        clf.partial_fit(X1, y1)
        clf.partial_fit(X2, y2)
        assert clf._root.leaf_stats.n_samples == 10
        counts = clf._root.leaf_stats.class_counts
        assert counts[0] == 5
        assert counts[1] == 5

    def test_empty_chunk_is_no_op(self):
        clf = HoeffdingTreeClassifier()
        X = np.ones((4, 2))
        y = np.array([0, 0, 1, 1])
        clf.partial_fit(X, y)
        # Empty chunk should not raise or change state
        clf.partial_fit(np.empty((0, 2)), np.empty(0, dtype=int))
        assert clf._root.leaf_stats.n_samples == 4

    def test_classes_provided_on_first_call(self):
        clf = HoeffdingTreeClassifier()
        X = np.ones((3, 2))
        y = np.array([0, 0, 1])
        clf.partial_fit(X, y, classes=np.array([0, 1, 2]))
        assert 2 in clf._classes

    def test_predict_entropy_criterion(self):
        clf = HoeffdingTreeClassifier(criterion="entropy")
        X = np.ones((6, 2))
        y = np.array([0, 0, 0, 0, 1, 1])
        clf.partial_fit(X, y)
        preds = clf.predict(X)
        assert np.all(preds == 0)

    def test_nan_in_predict_routes_left(self):
        clf = HoeffdingTreeClassifier()
        # Build a tree manually with a root internal node
        clf.partial_fit(np.ones((4, 2)), np.array([0, 0, 1, 1]))
        # Manually create a split: root becomes internal, two leaf children
        left = Node(leaf_stats=LeafStats(n_features=2), depth=1)
        right = Node(leaf_stats=LeafStats(n_features=2), depth=1)
        for _ in range(3):
            left.leaf_stats.update(np.array([0.0, 0.0]), label=0)
        for _ in range(3):
            right.leaf_stats.update(np.array([1.0, 1.0]), label=1)
        clf._root.to_internal(
            split_feature=0,
            split_threshold=0.5,
            left_child=left,
            right_child=right,
        )
        # NaN on split feature should go left → class 0
        x_nan = np.array([[np.nan, 0.5]])
        assert clf.predict(x_nan)[0] == 0


# Hoeffding bound

class TestHoeffdingBound:
    def test_gini_formula(self):
        # ε = sqrt(R² · ln(1/δ) / 2n), R=1 for gini
        n, delta = 100, 1e-7
        eps = _hoeffding_bound(n, delta, "gini", 2)
        expected = float(np.sqrt(np.log(1.0 / delta) / (2.0 * n)))
        assert eps == pytest.approx(expected, rel=1e-9)

    def test_entropy_formula(self):
        # R = log2(n_classes)
        n, delta, n_cls = 200, 1e-7, 3
        eps = _hoeffding_bound(n, delta, "entropy", n_cls)
        R = np.log2(3)
        expected = float(np.sqrt(R ** 2 * np.log(1.0 / delta) / (2.0 * n)))
        assert eps == pytest.approx(expected, rel=1e-9)

    def test_larger_n_gives_smaller_bound(self):
        eps_small = _hoeffding_bound(100, 1e-7, "gini", 2)
        eps_large = _hoeffding_bound(10_000, 1e-7, "gini", 2)
        assert eps_large < eps_small

    def test_entropy_bound_larger_than_gini_for_same_n(self):
        # log2(3) > 1 so entropy bound > gini bound
        eps_gini = _hoeffding_bound(500, 1e-7, "gini", 3)
        eps_ent = _hoeffding_bound(500, 1e-7, "entropy", 3)
        assert eps_ent > eps_gini


# Split decision tests

class TestSplitDecision:
    def _separable_data(self, rng, n=2000):
        """2-class data: feature 0 is informative, features 1-3 are noise."""
        half = n // 2
        X0 = np.column_stack([
            rng.normal(-3.0, 0.3, half),
            rng.normal(0.0, 1.0, (half, 3)),
        ])
        X1 = np.column_stack([
            rng.normal(3.0, 0.3, half),
            rng.normal(0.0, 1.0, (half, 3)),
        ])
        X = np.vstack([X0, X1])
        y = np.array([0] * half + [1] * half)
        return X, y

    def test_split_fires_when_clearly_separable(self):
        rng = np.random.default_rng(42)
        X, y = self._separable_data(rng, n=2000)
        clf = HoeffdingTreeClassifier(
            grace_period=200, min_samples_split=200, delta=1e-7, tau=0.05
        )
        clf.partial_fit(X, y)
        assert not clf._root.is_leaf, "Expected a split on clearly separable data"

    def test_split_fires_on_informative_feature(self):
        rng = np.random.default_rng(7)
        X, y = self._separable_data(rng, n=3000)
        clf = HoeffdingTreeClassifier(
            grace_period=200, min_samples_split=200, delta=1e-7, tau=0.05
        )
        clf.partial_fit(X, y)
        assert not clf._root.is_leaf
        # Feature 0 is the informative one; split should be on it
        assert clf._root.split_feature == 0

    def test_split_waits_when_eps_large(self):
        # Only 50 samples, default delta → eps ≈ 0.4; random data → gains ≈ 0
        rng = np.random.default_rng(0)
        n = 50
        X = rng.standard_normal((n, 4))
        y = rng.integers(0, 2, n)
        clf = HoeffdingTreeClassifier(
            grace_period=50, min_samples_split=50, delta=1e-7, tau=0.001
        )
        clf.partial_fit(X, y)
        # With tiny tau (0.001) and large eps, neither condition should fire
        assert clf._root.is_leaf, "Should not split when eps is large and gains are random"

    def test_tau_breaking_fires_when_eps_small(self):
        # Enough data to make eps < tau, any positive gain triggers split
        rng = np.random.default_rng(1)
        X, y = self._separable_data(rng, n=3000)
        # Very large tau so tau-breaking is easy to trigger
        clf = HoeffdingTreeClassifier(
            grace_period=200, min_samples_split=200, delta=1e-7, tau=0.99
        )
        clf.partial_fit(X, y)
        assert not clf._root.is_leaf

    def test_max_depth_respected(self):
        rng = np.random.default_rng(3)
        X, y = self._separable_data(rng, n=3000)
        clf = HoeffdingTreeClassifier(
            max_depth=0, grace_period=200, min_samples_split=200
        )
        clf.partial_fit(X, y)
        assert clf._root.is_leaf, "max_depth=0 means root must stay a leaf"

    def test_min_samples_split_respected(self):
        rng = np.random.default_rng(4)
        X, y = self._separable_data(rng, n=400)
        # min_samples_split larger than total data
        clf = HoeffdingTreeClassifier(
            min_samples_split=10_000, grace_period=200
        )
        clf.partial_fit(X, y)
        assert clf._root.is_leaf, "Should not split before min_samples_split is reached"

    def test_end_to_end_streaming_accuracy_improves(self):
        rng = np.random.default_rng(99)
        n_total = 3000
        half = n_total // 2
        X = np.column_stack([
            np.concatenate([rng.normal(-2.0, 0.5, half), rng.normal(2.0, 0.5, half)]),
            rng.normal(0.0, 1.0, n_total),
            rng.normal(0.0, 1.0, n_total),
            rng.normal(0.0, 1.0, n_total),
        ])
        y = np.array([0] * half + [1] * half, dtype=np.int64)

        # Shuffle
        idx = rng.permutation(n_total)
        X, y = X[idx], y[idx]

        clf = HoeffdingTreeClassifier(
            grace_period=200, min_samples_split=200, delta=1e-7
        )
        chunk = 500
        for start in range(0, n_total, chunk):
            clf.partial_fit(X[start:start + chunk], y[start:start + chunk])

        preds = clf.predict(X)
        accuracy = float((preds == y).mean())
        assert accuracy > 0.80, f"Expected accuracy > 0.80; got {accuracy:.3f}"
        assert not clf._root.is_leaf, "Tree should have split at least once"

    def test_nan_inputs_during_partial_fit_no_crash(self):
        rng = np.random.default_rng(5)
        X = rng.standard_normal((300, 4))
        y = (X[:, 0] > 0).astype(int)

        # Inject NaNs into random positions
        nan_mask = rng.random(X.shape) < 0.15
        X[nan_mask] = np.nan

        clf = HoeffdingTreeClassifier(grace_period=100, min_samples_split=100)
        clf.partial_fit(X, y)   # must not raise
        preds = clf.predict(X)
        assert preds.shape == (300,)


# predict_proba

class TestPredictProba:
    def test_shape_and_prob_sum(self):
        clf = HoeffdingTreeClassifier()
        X = np.ones((10, 3))
        y = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
        clf.partial_fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (10, 2)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(10))
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_root_leaf_only(self):
        clf = HoeffdingTreeClassifier()
        X = np.ones((5, 2))
        y = np.array([0, 0, 0, 1, 1])
        clf.partial_fit(X, y)
        assert clf._root.is_leaf
        proba = clf.predict_proba(X)
        # All samples land at the same leaf: 3/5 class-0, 2/5 class-1
        np.testing.assert_allclose(proba[:, 0], 0.6)
        np.testing.assert_allclose(proba[:, 1], 0.4)

    def test_different_proba_after_split(self):
        rng = np.random.default_rng(42)
        half = 1000
        X0 = np.column_stack([rng.normal(-3.0, 0.3, half), rng.normal(0.0, 1.0, (half, 2))])
        X1 = np.column_stack([rng.normal(3.0, 0.3, half), rng.normal(0.0, 1.0, (half, 2))])
        X = np.vstack([X0, X1])
        y = np.array([0] * half + [1] * half)
        clf = HoeffdingTreeClassifier(grace_period=200, min_samples_split=200, delta=1e-7)
        clf.partial_fit(X, y)
        assert not clf._root.is_leaf, "Expected a split"
        proba = clf.predict_proba(X)
        # Class-0 samples should have higher P(class=0) than class-1 samples
        p0_for_class0 = proba[:half, 0].mean()
        p0_for_class1 = proba[half:, 0].mean()
        assert p0_for_class0 > p0_for_class1

    def test_predict_proba_before_fit_raises(self):
        clf = HoeffdingTreeClassifier()
        with pytest.raises(ValueError, match="partial_fit"):
            clf.predict_proba(np.array([[1.0, 2.0]]))


# Tree introspection properties

class TestTreeIntrospection:
    def test_unfit_tree_all_zero(self):
        clf = HoeffdingTreeClassifier()
        assert clf.tree_depth == 0
        assert clf.n_leaves == 0
        assert clf.n_nodes == 0

    def test_root_only_tree(self):
        clf = HoeffdingTreeClassifier()
        clf.partial_fit(np.ones((5, 2)), np.array([0, 0, 1, 1, 0]))
        assert clf._root.is_leaf
        assert clf.tree_depth == 0
        assert clf.n_leaves == 1
        assert clf.n_nodes == 1

    def test_binary_tree_node_identity_after_split(self):
        rng = np.random.default_rng(7)
        half = 1000
        X = np.vstack([
            np.column_stack([rng.normal(-3.0, 0.3, half), rng.normal(0.0, 1.0, (half, 2))]),
            np.column_stack([rng.normal(3.0, 0.3, half), rng.normal(0.0, 1.0, (half, 2))]),
        ])
        y = np.array([0] * half + [1] * half)
        clf = HoeffdingTreeClassifier(grace_period=200, min_samples_split=200, delta=1e-7)
        clf.partial_fit(X, y)
        assert not clf._root.is_leaf
        # For any binary tree: n_nodes == 2 * n_leaves - 1
        assert clf.n_nodes == 2 * clf.n_leaves - 1
        assert clf.tree_depth >= 1

    def test_single_class_chunk(self):
        clf = HoeffdingTreeClassifier()
        X = np.ones((10, 3))
        y = np.zeros(10, dtype=int)
        clf.partial_fit(X, y)
        preds = clf.predict(X)
        assert np.all(preds == 0)

    def test_all_nan_feature_column_no_crash(self):
        rng = np.random.default_rng(11)
        X = rng.standard_normal((200, 3))
        X[:, 1] = np.nan  # entire column is NaN
        y = (X[:, 0] > 0).astype(int)
        clf = HoeffdingTreeClassifier(grace_period=100, min_samples_split=100)
        clf.partial_fit(X, y)   # must not raise
        preds = clf.predict(X)
        assert preds.shape == (200,)
