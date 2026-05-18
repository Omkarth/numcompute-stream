"""Tests for numcompute_stream.metrics."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from numcompute_stream.metrics import (
    Accuracy,
    PrecisionRecallF1,
    RollingMetric,
    StreamingAUC,
    StreamingConfusionMatrix,
)


# ---------------------------------------------------------------------------
# StreamingConfusionMatrix
# ---------------------------------------------------------------------------

class TestStreamingConfusionMatrix:
    def test_binary_basic(self):
        cm = StreamingConfusionMatrix(n_classes=2)
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        cm.update(y_true, y_pred)
        mat = cm.result()
        assert mat.shape == (2, 2)
        assert mat[0, 0] == 1   # TN
        assert mat[0, 1] == 1   # FP
        assert mat[1, 0] == 1   # FN
        assert mat[1, 1] == 1   # TP

    def test_multiclass_fixed(self):
        cm = StreamingConfusionMatrix(n_classes=3)
        y_true = np.array([0, 1, 2, 0])
        y_pred = np.array([0, 2, 2, 1])
        cm.update(y_true, y_pred)
        mat = cm.result()
        assert mat.shape == (3, 3)
        assert mat[0, 0] == 1
        assert mat[2, 2] == 1
        assert mat.sum() == 4

    def test_auto_grows(self):
        cm = StreamingConfusionMatrix()
        cm.update(np.array([0, 1]), np.array([0, 1]))
        assert cm.n_classes == 2
        cm.update(np.array([2]), np.array([2]))
        assert cm.n_classes == 3
        assert cm.result().shape == (3, 3)
        assert cm.result().sum() == 3

    def test_first_seen_class_in_pred(self):
        cm = StreamingConfusionMatrix()
        cm.update(np.array([0]), np.array([1]))
        assert cm.n_classes == 2

    def test_empty_chunk_noop(self):
        cm = StreamingConfusionMatrix(n_classes=2)
        cm.update(np.array([0, 1]), np.array([0, 1]))
        before = cm.result().copy()
        cm.update(np.array([]), np.array([]))
        np.testing.assert_array_equal(cm.result(), before)

    def test_mismatched_shapes_raises(self):
        cm = StreamingConfusionMatrix()
        with pytest.raises(ValueError, match="same shape"):
            cm.update(np.array([0, 1, 2]), np.array([0, 1]))

    def test_out_of_range_label_raises(self):
        cm = StreamingConfusionMatrix(n_classes=2)
        with pytest.raises(ValueError, match="outside"):
            cm.update(np.array([0, 5]), np.array([0, 1]))

    def test_reset_clears_state(self):
        cm = StreamingConfusionMatrix(n_classes=2)
        cm.update(np.array([0, 1]), np.array([1, 0]))
        cm.reset()
        np.testing.assert_array_equal(cm.result(), np.zeros((2, 2)))

    def test_reset_auto_clears(self):
        cm = StreamingConfusionMatrix()
        cm.update(np.array([0, 1, 2]), np.array([0, 1, 2]))
        cm.reset()
        assert cm.n_classes == 0
        assert cm.result().shape == (0, 0)

    def test_classes_property(self):
        cm = StreamingConfusionMatrix()
        cm.update(np.array([2, 0]), np.array([0, 2]))
        assert cm.classes_ == [0, 2]

    def test_cumulative_multi_chunk(self):
        cm = StreamingConfusionMatrix(n_classes=2)
        cm.update(np.array([0, 0]), np.array([0, 0]))
        cm.update(np.array([1, 1]), np.array([1, 1]))
        mat = cm.result()
        assert mat[0, 0] == 2
        assert mat[1, 1] == 2
        assert mat.sum() == 4


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

class TestAccuracy:
    def test_perfect_accuracy(self):
        acc = Accuracy()
        acc.update(np.array([0, 1, 2]), np.array([0, 1, 2]))
        assert acc.result() == pytest.approx(1.0)

    def test_partial_accuracy(self):
        acc = Accuracy()
        acc.update(np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1]))
        assert acc.result() == pytest.approx(0.5)

    def test_empty_returns_zero(self):
        acc = Accuracy()
        assert acc.result() == 0.0

    def test_empty_chunk_noop(self):
        acc = Accuracy()
        acc.update(np.array([0, 1]), np.array([0, 1]))
        before = acc.result()
        acc.update(np.array([]), np.array([]))
        assert acc.result() == pytest.approx(before)

    def test_single_class_chunk(self):
        acc = Accuracy()
        acc.update(np.array([0, 0, 0]), np.array([0, 0, 0]))
        assert acc.result() == pytest.approx(1.0)

    def test_reset(self):
        acc = Accuracy()
        acc.update(np.array([0, 1]), np.array([0, 0]))
        acc.reset()
        assert acc.result() == 0.0

    def test_mismatched_shapes_raises(self):
        acc = Accuracy()
        with pytest.raises(ValueError, match="same shape"):
            acc.update(np.array([0, 1, 2]), np.array([0, 1]))


# ---------------------------------------------------------------------------
# PrecisionRecallF1
# ---------------------------------------------------------------------------

class TestPrecisionRecallF1:
    def test_perfect_binary(self):
        m = PrecisionRecallF1(warn_zero_division=False)
        m.update(np.array([0, 1, 0, 1]), np.array([0, 1, 0, 1]))
        assert m.macro_precision() == pytest.approx(1.0)
        assert m.macro_recall() == pytest.approx(1.0)
        assert m.macro_f1() == pytest.approx(1.0)

    def test_macro_f1_multiclass(self):
        m = PrecisionRecallF1(n_classes=3, warn_zero_division=False)
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        m.update(y_true, y_pred)
        assert m.macro_f1() == pytest.approx(1.0)

    def test_zero_division_no_exception(self):
        m = PrecisionRecallF1(n_classes=3, warn_zero_division=False)
        # class 2 never predicted → precision undefined for class 2
        m.update(np.array([0, 1, 2]), np.array([0, 1, 0]))
        p = m.precision()
        assert not np.any(np.isnan(p))
        assert p[2] == 0.0

    def test_zero_division_emits_warning(self):
        m = PrecisionRecallF1(n_classes=2, warn_zero_division=True)
        # class 1 never predicted
        m.update(np.array([0, 1]), np.array([0, 0]))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = m.precision()
        assert any("undefined" in str(warning.message).lower() for warning in w)

    def test_empty_returns_zero(self):
        m = PrecisionRecallF1()
        assert m.result() == 0.0
        assert m.macro_precision() == 0.0
        assert m.macro_recall() == 0.0

    def test_result_is_macro_f1(self):
        m = PrecisionRecallF1(warn_zero_division=False)
        m.update(np.array([0, 1]), np.array([0, 1]))
        assert m.result() == pytest.approx(m.macro_f1())

    def test_per_class_f1_shape(self):
        m = PrecisionRecallF1(n_classes=3, warn_zero_division=False)
        m.update(np.array([0, 1, 2]), np.array([0, 1, 2]))
        assert m.f1().shape == (3,)

    def test_reset(self):
        m = PrecisionRecallF1()
        m.update(np.array([0, 1]), np.array([1, 0]))
        m.reset()
        assert m.result() == 0.0


# ---------------------------------------------------------------------------
# RollingMetric
# ---------------------------------------------------------------------------

class TestRollingMetric:
    def test_window_limits_chunks(self):
        acc = Accuracy()
        rm = RollingMetric(acc, window_size=2)
        rm.update(np.array([0, 0]), np.array([0, 0]))   # perfect
        rm.update(np.array([1, 1]), np.array([1, 1]))   # perfect
        rm.update(np.array([0, 0]), np.array([1, 1]))   # all wrong
        # window has last 2 chunks: chunk2 (perfect) and chunk3 (wrong) → 50%
        assert rm.result() == pytest.approx(0.5)

    def test_old_data_evicted(self):
        acc = Accuracy()
        rm = RollingMetric(acc, window_size=1)
        rm.update(np.array([0, 0]), np.array([1, 1]))   # 0% accuracy
        rm.update(np.array([1, 1]), np.array([1, 1]))   # 100% accuracy
        # Only second chunk retained
        assert rm.result() == pytest.approx(1.0)

    def test_empty_window_result(self):
        acc = Accuracy()
        rm = RollingMetric(acc, window_size=3)
        assert rm.result() == 0.0   # no data → accuracy returns 0.0

    def test_reset_clears_window(self):
        acc = Accuracy()
        rm = RollingMetric(acc, window_size=5)
        rm.update(np.array([0, 1]), np.array([0, 1]))
        rm.reset()
        assert rm.result() == 0.0

    def test_empty_chunk_noop(self):
        acc = Accuracy()
        rm = RollingMetric(acc, window_size=3)
        rm.update(np.array([0, 1]), np.array([0, 1]))
        before = rm.result()
        rm.update(np.array([]), np.array([]))
        assert rm.result() == pytest.approx(before)

    def test_invalid_window_size(self):
        with pytest.raises(ValueError, match="window_size"):
            RollingMetric(Accuracy(), window_size=0)


# ---------------------------------------------------------------------------
# StreamingAUC
# ---------------------------------------------------------------------------

class TestStreamingAUC:
    def test_perfect_auc(self):
        auc = StreamingAUC()
        scores = np.array([0.9, 0.8, 0.3, 0.2])
        labels = np.array([1, 1, 0, 0])
        auc.update(labels, scores)
        assert auc.result() == pytest.approx(1.0)

    def test_chance_level_random(self):
        rng = np.random.default_rng(42)
        auc = StreamingAUC(window_size=2000)
        labels = rng.integers(0, 2, size=2000)
        scores = rng.random(size=2000)
        auc.update(labels, scores)
        assert 0.45 <= auc.result() <= 0.55

    def test_single_class_returns_half(self):
        auc = StreamingAUC()
        auc.update(np.array([0, 0, 0]), np.array([0.1, 0.4, 0.2]))
        assert auc.result() == pytest.approx(0.5)

    def test_empty_returns_half(self):
        auc = StreamingAUC()
        assert auc.result() == pytest.approx(0.5)

    def test_window_slides(self):
        auc = StreamingAUC(window_size=4)
        # First 4 samples: perfect separation
        auc.update(np.array([1, 1]), np.array([0.9, 0.8]))
        auc.update(np.array([0, 0]), np.array([0.2, 0.1]))
        assert auc.result() == pytest.approx(1.0)
        # Push 4 new samples that are all wrong → old ones evicted
        auc.update(np.array([0, 0]), np.array([0.9, 0.8]))
        auc.update(np.array([1, 1]), np.array([0.1, 0.2]))
        assert auc.result() == pytest.approx(0.0)

    def test_non_binary_labels_raises(self):
        auc = StreamingAUC()
        with pytest.raises(ValueError, match="binary-only"):
            auc.update(np.array([0, 1, 2]), np.array([0.1, 0.5, 0.9]))

    def test_mismatched_shapes_raises(self):
        auc = StreamingAUC()
        with pytest.raises(ValueError, match="same shape"):
            auc.update(np.array([0, 1]), np.array([0.5]))

    def test_reset(self):
        auc = StreamingAUC()
        auc.update(np.array([0, 1]), np.array([0.1, 0.9]))
        auc.reset()
        assert auc.result() == pytest.approx(0.5)

    def test_empty_chunk_noop(self):
        auc = StreamingAUC()
        auc.update(np.array([0, 1]), np.array([0.1, 0.9]))
        before = auc.result()
        auc.update(np.array([]), np.array([]))
        assert auc.result() == pytest.approx(before)
