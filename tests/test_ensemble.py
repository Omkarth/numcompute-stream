"""Tests for numcompute_stream.ensemble — OnlineRandomForestClassifier."""

from __future__ import annotations

import math

import numpy as np
import pytest

from numcompute_stream.ensemble import OnlineRandomForestClassifier


# Helpers

def _make_separable(n=400, n_features=6, seed=0):
    """Two linearly separable Gaussian blobs."""
    rng = np.random.default_rng(seed)
    half = n // 2
    X0 = rng.standard_normal((half, n_features)) - 2.0
    X1 = rng.standard_normal((half, n_features)) + 2.0
    X = np.vstack([X0, X1])
    y = np.array([0] * half + [1] * half, dtype=np.int64)
    idx = rng.permutation(n)
    return X[idx], y[idx]


def _stream_fit(clf, X, y, chunk=50):
    for start in range(0, len(X), chunk):
        clf.partial_fit(X[start : start + chunk], y[start : start + chunk])
    return clf


# Construction

class TestConstruction:
    def test_defaults(self):
        clf = OnlineRandomForestClassifier()
        assert clf.n_estimators == 10
        assert clf.max_features == "sqrt"
        assert clf.voting == "hard"
        assert clf.random_state is None
        assert clf._states is None

    def test_invalid_voting(self):
        with pytest.raises(ValueError, match="voting"):
            OnlineRandomForestClassifier(voting="plurality")

    def test_invalid_max_features(self):
        clf = OnlineRandomForestClassifier(max_features="bad", n_estimators=2)
        X = np.random.default_rng(0).standard_normal((10, 4))
        y = np.zeros(10, dtype=np.int64)
        with pytest.raises(ValueError, match="max_features"):
            clf.partial_fit(X, y)


# Feature subsets

class TestFeatureSubsets:
    def test_sqrt_subset_size(self):
        n_features = 16
        clf = OnlineRandomForestClassifier(n_estimators=5, max_features="sqrt", random_state=1)
        X = np.random.default_rng(1).standard_normal((20, n_features))
        y = np.zeros(20, dtype=np.int64)
        clf.partial_fit(X, y)
        expected_k = int(math.sqrt(n_features))
        for subset in clf.feature_subsets_:
            assert len(subset) == expected_k

    def test_log2_subset_size(self):
        n_features = 16
        clf = OnlineRandomForestClassifier(n_estimators=5, max_features="log2", random_state=2)
        X = np.random.default_rng(2).standard_normal((20, n_features))
        y = np.zeros(20, dtype=np.int64)
        clf.partial_fit(X, y)
        expected_k = int(math.log2(n_features))
        for subset in clf.feature_subsets_:
            assert len(subset) == expected_k

    def test_int_subset_size(self):
        clf = OnlineRandomForestClassifier(n_estimators=4, max_features=3, random_state=3)
        X = np.random.default_rng(3).standard_normal((20, 10))
        y = np.zeros(20, dtype=np.int64)
        clf.partial_fit(X, y)
        for subset in clf.feature_subsets_:
            assert len(subset) == 3

    def test_subsets_are_valid_indices(self):
        n_features = 8
        clf = OnlineRandomForestClassifier(n_estimators=6, max_features="sqrt", random_state=7)
        X = np.random.default_rng(7).standard_normal((20, n_features))
        y = np.zeros(20, dtype=np.int64)
        clf.partial_fit(X, y)
        for subset in clf.feature_subsets_:
            assert subset.min() >= 0
            assert subset.max() < n_features
            assert len(np.unique(subset)) == len(subset), "duplicate indices in subset"


# Partial fit

class TestPartialFit:
    def test_first_call_no_raise(self):
        clf = OnlineRandomForestClassifier(n_estimators=3, random_state=0)
        X = np.random.default_rng(0).standard_normal((30, 4))
        y = (np.arange(30) % 2).astype(np.int64)
        clf.partial_fit(X, y)  # should not raise

    def test_wrong_n_features_raises(self):
        clf = OnlineRandomForestClassifier(n_estimators=2, random_state=0)
        X4 = np.random.default_rng(0).standard_normal((10, 4))
        X5 = np.random.default_rng(1).standard_normal((10, 5))
        y = np.zeros(10, dtype=np.int64)
        clf.partial_fit(X4, y)
        with pytest.raises(ValueError, match="features"):
            clf.partial_fit(X5, y)

    def test_returns_self(self):
        clf = OnlineRandomForestClassifier(n_estimators=2, random_state=0)
        X = np.random.default_rng(0).standard_normal((10, 4))
        y = np.zeros(10, dtype=np.int64)
        assert clf.partial_fit(X, y) is clf


# Predict

class TestPredict:
    def test_predict_shape(self):
        clf = OnlineRandomForestClassifier(n_estimators=5, random_state=0)
        X, y = _make_separable(200)
        _stream_fit(clf, X, y)
        preds = clf.predict(X[:10])
        assert preds.shape == (10,)

    def test_predict_labels_in_classes(self):
        clf = OnlineRandomForestClassifier(n_estimators=5, random_state=0)
        X, y = _make_separable(200)
        _stream_fit(clf, X, y)
        preds = clf.predict(X)
        assert set(preds.tolist()).issubset({0, 1})

    def test_predict_before_fit_raises(self):
        clf = OnlineRandomForestClassifier(n_estimators=3)
        X = np.random.default_rng(0).standard_normal((5, 4))
        with pytest.raises(ValueError):
            clf.predict(X)


# predict_proba

class TestPredictProba:
    def test_proba_sums_to_one(self):
        clf = OnlineRandomForestClassifier(n_estimators=5, voting="soft", random_state=0)
        X, y = _make_separable(200)
        _stream_fit(clf, X, y)
        proba = clf.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)

    def test_proba_shape(self):
        clf = OnlineRandomForestClassifier(n_estimators=5, random_state=0)
        X, y = _make_separable(200)
        _stream_fit(clf, X, y)
        proba = clf.predict_proba(X[:15])
        assert proba.shape == (15, 2)

    def test_proba_in_unit_interval(self):
        clf = OnlineRandomForestClassifier(n_estimators=5, random_state=0)
        X, y = _make_separable(200)
        _stream_fit(clf, X, y)
        proba = clf.predict_proba(X)
        assert (proba >= 0.0).all() and (proba <= 1.0).all()


# Voting consistency

class TestVoting:
    def test_hard_voting_majority(self):
        """On toy data where all trees agree, hard vote must match unanimous label."""
        rng = np.random.default_rng(42)
        # 3-feature, very separable
        X0 = rng.standard_normal((100, 3)) - 5.0
        X1 = rng.standard_normal((100, 3)) + 5.0
        X = np.vstack([X0, X1])
        y = np.array([0] * 100 + [1] * 100, dtype=np.int64)

        clf = OnlineRandomForestClassifier(
            n_estimators=7, max_features=2, random_state=0,
            grace_period=50, min_samples_split=5,
        )
        _stream_fit(clf, X, y, chunk=40)

        # collect per-tree predictions on training data
        tree_preds = np.column_stack([
            state.tree.predict(X[:, state.feature_subset])
            for state in clf._states
            if state.tree._root is not None
        ])
        # majority class per sample (no scipy needed)
        classes = np.unique(y)
        votes = np.zeros((len(X), len(classes)), dtype=np.int64)
        class_to_idx = {int(c): i for i, c in enumerate(classes)}
        for col in range(tree_preds.shape[1]):
            for j, p in enumerate(tree_preds[:, col]):
                votes[j, class_to_idx[int(p)]] += 1
        majority = classes[np.argmax(votes, axis=1)]
        np.testing.assert_array_equal(clf.predict(X), majority)

    def test_soft_voting_agree_with_proba_argmax(self):
        clf = OnlineRandomForestClassifier(n_estimators=5, voting="soft", random_state=1)
        X, y = _make_separable(200)
        _stream_fit(clf, X, y)
        proba = clf.predict_proba(X[:20])
        expected = clf._classes[np.argmax(proba, axis=1)]
        np.testing.assert_array_equal(clf.predict(X[:20]), expected)


# End-to-end accuracy

class TestEndToEnd:
    def test_ensemble_beats_or_ties_single_tree(self):
        """Ensemble accuracy should be >= single HoeffdingTree on separable data."""
        from numcompute_stream.tree import HoeffdingTreeClassifier
        from numcompute_stream.metrics import Accuracy

        X, y = _make_separable(600, seed=99)

        # Single tree
        single = HoeffdingTreeClassifier(grace_period=50, min_samples_split=5)
        _stream_fit(single, X, y, chunk=60)
        single_acc = Accuracy()
        single_acc.update(y, single.predict(X))

        # Ensemble
        ens = OnlineRandomForestClassifier(
            n_estimators=10, max_features="sqrt", random_state=42,
            grace_period=50, min_samples_split=5,
        )
        _stream_fit(ens, X, y, chunk=60)
        ens_acc = Accuracy()
        ens_acc.update(y, ens.predict(X))

        assert ens_acc.result() >= single_acc.result() - 0.02  # allow tiny tolerance


# OOB accuracy

class TestOOBAccuracy:
    def test_oob_list_length(self):
        clf = OnlineRandomForestClassifier(n_estimators=5, random_state=0)
        X, y = _make_separable(200)
        _stream_fit(clf, X, y)
        assert len(clf.oob_accuracy_) == 5

    def test_oob_non_trivial(self):
        """On random binary data OOB accuracy should be between 0.3 and 0.7."""
        rng = np.random.default_rng(7)
        X = rng.standard_normal((500, 6))
        y = rng.integers(0, 2, size=500).astype(np.int64)
        clf = OnlineRandomForestClassifier(n_estimators=8, random_state=5)
        _stream_fit(clf, X, y, chunk=50)
        for acc in clf.oob_accuracy_:
            assert 0.2 <= acc <= 0.8, f"OOB accuracy {acc} looks degenerate"

    def test_oob_zero_before_fit(self):
        clf = OnlineRandomForestClassifier(n_estimators=3)
        assert clf.oob_accuracy_ == []


# Reproducibility

class TestReproducibility:
    def test_same_seed_same_subsets(self):
        X, y = _make_separable(100)
        clf1 = OnlineRandomForestClassifier(n_estimators=6, random_state=42)
        clf2 = OnlineRandomForestClassifier(n_estimators=6, random_state=42)
        clf1.partial_fit(X, y)
        clf2.partial_fit(X, y)
        for s1, s2 in zip(clf1.feature_subsets_, clf2.feature_subsets_):
            np.testing.assert_array_equal(s1, s2)

    def test_same_seed_same_predictions(self):
        X, y = _make_separable(200)
        clf1 = OnlineRandomForestClassifier(n_estimators=5, random_state=0)
        clf2 = OnlineRandomForestClassifier(n_estimators=5, random_state=0)
        _stream_fit(clf1, X, y)
        _stream_fit(clf2, X, y)
        np.testing.assert_array_equal(clf1.predict(X), clf2.predict(X))

    def test_different_seeds_different_subsets(self):
        X, y = _make_separable(100)
        clf1 = OnlineRandomForestClassifier(n_estimators=6, random_state=0)
        clf2 = OnlineRandomForestClassifier(n_estimators=6, random_state=99)
        clf1.partial_fit(X, y)
        clf2.partial_fit(X, y)
        # At least one subset should differ
        all_same = all(
            np.array_equal(s1, s2)
            for s1, s2 in zip(clf1.feature_subsets_, clf2.feature_subsets_)
        )
        assert not all_same


# ARF drift detection (day 11)

def _make_abrupt_drift_stream(n_per_phase=1500, n_features=6, seed=0):
    """Return two phases: phase 1 label = X[:,0]>0, phase 2 label = X[:,0]<=0."""
    rng = np.random.default_rng(seed)
    X1 = rng.standard_normal((n_per_phase, n_features))
    y1 = (X1[:, 0] > 0).astype(np.int64)
    X2 = rng.standard_normal((n_per_phase, n_features))
    y2 = (X2[:, 0] <= 0).astype(np.int64)
    return X1, y1, X2, y2


class TestARFDriftDetection:
    def test_adwins_created_when_drift_detection_true(self):
        from numcompute_stream.drift import ADWIN

        clf = OnlineRandomForestClassifier(n_estimators=3, drift_detection=True, random_state=0)
        X, y = _make_separable(50)
        clf.partial_fit(X, y)
        for state in clf._states:
            assert state.warning_adwin is not None
            assert state.drift_adwin is not None
            assert isinstance(state.warning_adwin, ADWIN)
            assert isinstance(state.drift_adwin, ADWIN)

    def test_no_adwins_when_drift_detection_false(self):
        clf = OnlineRandomForestClassifier(n_estimators=3, drift_detection=False, random_state=0)
        X, y = _make_separable(50)
        clf.partial_fit(X, y)
        for state in clf._states:
            assert state.warning_adwin is None
            assert state.drift_adwin is None

    def test_candidate_none_at_start(self):
        clf = OnlineRandomForestClassifier(n_estimators=4, drift_detection=True, random_state=0)
        X, y = _make_separable(50)
        clf.partial_fit(X, y)
        for state in clf._states:
            assert state.candidate is None

    def test_stable_stream_zero_drift(self):
        """Shuffled separable data should produce zero drift events."""
        clf = OnlineRandomForestClassifier(
            n_estimators=5, drift_detection=True, random_state=42,
            grace_period=50, min_samples_split=5,
        )
        X, y = _make_separable(500, seed=42)
        _stream_fit(clf, X, y, chunk=50)
        assert clf.drift_count_ == 0

    def test_abrupt_drift_detected(self):
        """Clear concept flip must trigger at least one confirmed drift event."""
        X1, y1, X2, y2 = _make_abrupt_drift_stream(n_per_phase=1500, seed=7)
        clf = OnlineRandomForestClassifier(
            n_estimators=5, drift_detection=True, random_state=0,
            grace_period=20, min_samples_split=5,
        )
        _stream_fit(clf, X1, y1, chunk=100)
        _stream_fit(clf, X2, y2, chunk=100)
        assert clf.drift_count_ >= 1, (
            f"Expected drift but got drift_count_={clf.drift_count_}"
        )

    def test_warning_fires_before_drift(self):
        """Warning ADWIN must fire at least as many times as confirmed drift."""
        X1, y1, X2, y2 = _make_abrupt_drift_stream(n_per_phase=1500, seed=3)
        clf = OnlineRandomForestClassifier(
            n_estimators=5, drift_detection=True, random_state=0,
            grace_period=20, min_samples_split=5,
        )
        _stream_fit(clf, X1, y1, chunk=100)
        _stream_fit(clf, X2, y2, chunk=100)
        # warning_count_ >= drift_count_ because warning always precedes drift
        # (or fires without leading to drift in the same window)
        assert clf.warning_count_ >= clf.drift_count_
        assert clf.warning_count_ >= 1, (
            f"Expected warning events but got warning_count_={clf.warning_count_}"
        )

    def test_arf_better_accuracy_on_drifting_stream(self):
        """ARF should adapt faster than a fixed ensemble after concept drift."""
        from numcompute_stream.metrics import Accuracy

        X1, y1, X2, y2 = _make_abrupt_drift_stream(n_per_phase=1500, seed=5)

        def run(drift_detection):
            clf = OnlineRandomForestClassifier(
                n_estimators=5, drift_detection=drift_detection, random_state=0,
                grace_period=20, min_samples_split=5,
            )
            _stream_fit(clf, X1, y1, chunk=100)
            _stream_fit(clf, X2, y2, chunk=100)
            return clf

        arf = run(True)
        fixed = run(False)

        # Evaluate both on the final 300 samples of phase 2 (new concept)
        X_test, y_test = X2[-300:], y2[-300:]
        acc_arf = Accuracy()
        acc_arf.update(y_test, arf.predict(X_test))
        acc_fixed = Accuracy()
        acc_fixed.update(y_test, fixed.predict(X_test))

        assert acc_arf.result() >= acc_fixed.result() - 0.05, (
            f"ARF acc={acc_arf.result():.3f} should be >= fixed acc={acc_fixed.result():.3f} "
            f"(within 5% tolerance)"
        )

    def test_reproducibility_same_drift_count(self):
        """Same random_state must yield identical drift_count_ across runs."""
        X1, y1, X2, y2 = _make_abrupt_drift_stream(n_per_phase=1500, seed=11)

        def run():
            clf = OnlineRandomForestClassifier(
                n_estimators=5, drift_detection=True, random_state=42,
                grace_period=20, min_samples_split=5,
            )
            _stream_fit(clf, X1, y1, chunk=100)
            _stream_fit(clf, X2, y2, chunk=100)
            return clf.drift_count_

        assert run() == run()
