"""Streaming classification metrics."""

from __future__ import annotations

import collections
import warnings

import numpy as np

from .base import BaseMetric


def _validate(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape; "
            f"got {y_true.shape} and {y_pred.shape}"
        )
    return y_true.ravel(), y_pred.ravel()


class StreamingConfusionMatrix(BaseMetric):
    """Incrementally updated confusion matrix for multi-class classification.

    Rows correspond to true classes, columns to predicted classes.  Classes are
    mapped to integer indices in the order first encountered, or to
    ``{0 … n_classes-1}`` when *n_classes* is given at construction.

    Parameters
    ----------
    n_classes : int | None
        When given the matrix is pre-allocated and all labels must lie in
        ``[0, n_classes)``.  When *None* the matrix grows as new labels appear.
    """

    def __init__(self, n_classes: int | None = None) -> None:
        """Initialise the confusion matrix; see class docstring for parameter details."""
        if n_classes is not None and n_classes < 2:
            raise ValueError(f"n_classes must be ≥ 2; got {n_classes}")
        self._fixed_n = n_classes
        self._reset_state()

    # ------------------------------------------------------------------
    # BaseMetric
    # ------------------------------------------------------------------

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> "StreamingConfusionMatrix":
        """Increment matrix cells for one chunk of true/predicted label pairs."""
        y_true, y_pred = _validate(y_true, y_pred)
        if y_true.size == 0:
            return self

        all_labels = np.unique(np.concatenate([y_true, y_pred]))
        new_labels = [int(lbl) for lbl in all_labels if int(lbl) not in self._label_to_idx]

        if new_labels:
            if self._fixed_n is not None:
                for lbl in new_labels:
                    if not (0 <= lbl < self._fixed_n):
                        raise ValueError(
                            f"Label {lbl} outside [0, {self._fixed_n}) "
                            f"for n_classes={self._fixed_n}"
                        )
                    self._label_to_idx[lbl] = lbl
            else:
                for lbl in sorted(new_labels):
                    self._label_to_idx[lbl] = len(self._label_to_idx)
                n = len(self._label_to_idx)
                old_n = self._matrix.shape[0]
                grown = np.zeros((n, n), dtype=np.int64)
                grown[:old_n, :old_n] = self._matrix
                self._matrix = grown

        true_idx = np.array([self._label_to_idx[int(l)] for l in y_true], dtype=np.intp)
        pred_idx = np.array([self._label_to_idx[int(l)] for l in y_pred], dtype=np.intp)
        np.add.at(self._matrix, (true_idx, pred_idx), 1)
        return self

    def result(self) -> np.ndarray:  # type: ignore[override]
        """Return the accumulated confusion matrix (rows=true, cols=predicted)."""
        return self._matrix.copy()

    def reset(self) -> "StreamingConfusionMatrix":
        """Clear all accumulated counts; returns self."""
        self._reset_state()
        return self

    # ------------------------------------------------------------------
    # Extra properties
    # ------------------------------------------------------------------

    @property
    def classes_(self) -> list[int]:
        """Sorted list of all class labels tracked so far."""
        return sorted(self._label_to_idx.keys())

    @property
    def n_classes(self) -> int:
        """Number of classes currently tracked."""
        return self._matrix.shape[0]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        n = self._fixed_n
        if n is not None:
            self._matrix: np.ndarray = np.zeros((n, n), dtype=np.int64)
            self._label_to_idx: dict[int, int] = {i: i for i in range(n)}
        else:
            self._matrix = np.zeros((0, 0), dtype=np.int64)
            self._label_to_idx = {}

    def __repr__(self) -> str:
        return f"StreamingConfusionMatrix(n_classes={self.n_classes})"


class Accuracy(BaseMetric):
    """Streaming accuracy derived from a confusion matrix.

    Parameters
    ----------
    n_classes : int | None
        Forwarded to the internal :class:`StreamingConfusionMatrix`.
    """

    def __init__(self, n_classes: int | None = None) -> None:
        """Initialise accuracy metric backed by a StreamingConfusionMatrix."""
        self._cm = StreamingConfusionMatrix(n_classes=n_classes)

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> "Accuracy":
        """Incorporate one chunk of true/predicted labels into the confusion matrix."""
        self._cm.update(y_true, y_pred)
        return self

    def result(self) -> float:
        """Accuracy in ``[0, 1]``, or ``0.0`` if no samples have been seen."""
        mat = self._cm.result()
        total = int(mat.sum())
        if total == 0:
            return 0.0
        return float(np.trace(mat) / total)

    def reset(self) -> "Accuracy":
        """Clear all accumulated state; returns self."""
        self._cm.reset()
        return self

    def __repr__(self) -> str:
        return f"Accuracy(result={self.result():.4f})"


class PrecisionRecallF1(BaseMetric):
    """Per-class and macro-averaged precision, recall, and F1.

    Derived from :class:`StreamingConfusionMatrix`.  Classes with zero
    predicted or zero true samples contribute ``0.0`` to macro averages rather
    than raising an exception or producing NaN.

    Parameters
    ----------
    n_classes : int | None
        Forwarded to the internal :class:`StreamingConfusionMatrix`.
    warn_zero_division : bool
        Emit a :class:`UserWarning` when a per-class denominator is zero.
        Default ``True``.
    """

    def __init__(
        self,
        n_classes: int | None = None,
        warn_zero_division: bool = True,
    ) -> None:
        """Initialise PrecisionRecallF1; see class docstring for parameter details."""
        self._cm = StreamingConfusionMatrix(n_classes=n_classes)
        self._warn = warn_zero_division

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> "PrecisionRecallF1":
        """Incorporate one chunk of true/predicted labels into the confusion matrix."""
        self._cm.update(y_true, y_pred)
        return self

    def result(self) -> float:
        """Return macro-averaged F1, or ``0.0`` if no samples seen."""
        return self.macro_f1()

    def reset(self) -> "PrecisionRecallF1":
        """Clear all accumulated state; returns self."""
        self._cm.reset()
        return self

    # ------------------------------------------------------------------
    # Per-class statistics
    # ------------------------------------------------------------------

    def precision(self) -> np.ndarray:
        """Per-class precision, indexed by :attr:`~StreamingConfusionMatrix.classes_`."""
        mat = self._cm.result()
        if mat.size == 0:
            return np.array([], dtype=float)
        col_sums = mat.sum(axis=0).astype(float)
        diag = np.diag(mat).astype(float)
        zero_mask = col_sums == 0
        if self._warn and np.any(zero_mask):
            bad = [self._cm.classes_[i] for i, z in enumerate(zero_mask) if z]
            warnings.warn(
                f"Precision undefined for class(es) {bad} "
                f"(no predicted samples); returning 0.0",
                UserWarning,
                stacklevel=2,
            )
        return np.where(zero_mask, 0.0, diag / np.where(zero_mask, 1.0, col_sums))

    def recall(self) -> np.ndarray:
        """Per-class recall, indexed by :attr:`~StreamingConfusionMatrix.classes_`."""
        mat = self._cm.result()
        if mat.size == 0:
            return np.array([], dtype=float)
        row_sums = mat.sum(axis=1).astype(float)
        diag = np.diag(mat).astype(float)
        zero_mask = row_sums == 0
        if self._warn and np.any(zero_mask):
            bad = [self._cm.classes_[i] for i, z in enumerate(zero_mask) if z]
            warnings.warn(
                f"Recall undefined for class(es) {bad} "
                f"(no true samples); returning 0.0",
                UserWarning,
                stacklevel=2,
            )
        return np.where(zero_mask, 0.0, diag / np.where(zero_mask, 1.0, row_sums))

    def f1(self) -> np.ndarray:
        """Per-class F1 score."""
        p = self.precision()
        r = self.recall()
        denom = p + r
        return np.where(denom > 0, 2.0 * p * r / denom, 0.0)

    # ------------------------------------------------------------------
    # Macro averages
    # ------------------------------------------------------------------

    def macro_precision(self) -> float:
        """Macro-averaged precision."""
        if self._cm.n_classes == 0:
            return 0.0
        return float(np.mean(self.precision()))

    def macro_recall(self) -> float:
        """Macro-averaged recall."""
        if self._cm.n_classes == 0:
            return 0.0
        return float(np.mean(self.recall()))

    def macro_f1(self) -> float:
        """Macro-averaged F1."""
        if self._cm.n_classes == 0:
            return 0.0
        return float(np.mean(self.f1()))

    def __repr__(self) -> str:
        return f"PrecisionRecallF1(macro_f1={self.macro_f1():.4f})"


class RollingMetric(BaseMetric):
    """Sliding-window wrapper around any :class:`BaseMetric`.

    Retains the last *window_size* ``(y_true, y_pred)`` chunks (by call count,
    not sample count) and recomputes the wrapped metric over only those chunks
    on each call to :meth:`result`.

    Parameters
    ----------
    base_metric : BaseMetric
        A constructed metric instance.  It is reset and reused internally;
        do not share the same instance with other callers.
    window_size : int
        Maximum number of update calls to retain.
    """

    def __init__(self, base_metric: BaseMetric, window_size: int) -> None:
        """Initialise the rolling wrapper; see class docstring for parameter details."""
        if window_size < 1:
            raise ValueError(f"window_size must be ≥ 1; got {window_size}")
        self._metric = base_metric
        self._window_size = window_size
        self._window: collections.deque[tuple[np.ndarray, np.ndarray]] = collections.deque()

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> "RollingMetric":
        """Buffer one chunk; evict the oldest chunk when window_size is exceeded."""
        y_true, y_pred = _validate(y_true, y_pred)
        if y_true.size == 0:
            return self
        self._window.append((y_true.copy(), y_pred.copy()))
        while len(self._window) > self._window_size:
            self._window.popleft()
        return self

    def result(self) -> float:
        """Recompute the wrapped metric over the current window."""
        self._metric.reset()
        for yt, yp in self._window:
            self._metric.update(yt, yp)
        return self._metric.result()

    def reset(self) -> "RollingMetric":
        """Clear the buffer and reset the wrapped metric; returns self."""
        self._window.clear()
        self._metric.reset()
        return self

    def __repr__(self) -> str:
        return (
            f"RollingMetric(metric={type(self._metric).__name__}, "
            f"window_size={self._window_size}, buffered={len(self._window)})"
        )


class StreamingAUC(BaseMetric):
    """Sliding-window ROC-AUC for binary classification.

    Maintains the last *window_size* individual ``(score, label)`` pairs and
    computes the area under the ROC curve via the trapezoidal rule.

    Parameters
    ----------
    window_size : int
        Maximum number of *samples* (not chunks) to retain.  Default 10 000.
    """

    def __init__(self, window_size: int = 10_000) -> None:
        """Initialise the AUC accumulator; see class docstring for parameter details."""
        if window_size < 1:
            raise ValueError(f"window_size must be ≥ 1; got {window_size}")
        self._window_size = window_size
        self._scores: collections.deque[float] = collections.deque()
        self._labels: collections.deque[int] = collections.deque()

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> "StreamingAUC":
        """Incorporate a chunk of binary predictions.

        Parameters
        ----------
        y_true : array-like of int
            Binary ground-truth labels (0 or 1).
        y_pred : array-like of float
            Predicted scores or probabilities for the positive class.
        """
        y_true, y_pred = _validate(y_true, y_pred)
        if y_true.size == 0:
            return self

        labels = y_true.astype(int)
        unique = set(labels.tolist())
        invalid = unique - {0, 1}
        if invalid:
            raise ValueError(
                f"StreamingAUC is binary-only; unexpected label(s): {sorted(invalid)}"
            )

        for lbl, score in zip(labels.tolist(), y_pred.tolist()):
            if len(self._scores) >= self._window_size:
                self._scores.popleft()
                self._labels.popleft()
            self._scores.append(float(score))
            self._labels.append(int(lbl))

        return self

    def result(self) -> float:
        """ROC-AUC over the current window.

        Returns ``0.5`` (chance level) when fewer than 2 samples are buffered
        or only one class is present.
        """
        if len(self._scores) < 2:
            return 0.5

        scores = np.array(self._scores)
        labels = np.array(self._labels, dtype=int)

        n_pos = int(labels.sum())
        n_neg = len(labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            return 0.5

        order = np.argsort(scores)[::-1]
        sorted_labels = labels[order]

        tps = np.cumsum(sorted_labels)
        fps = np.cumsum(1 - sorted_labels)

        tpr = np.concatenate([[0.0], tps / n_pos])
        fpr = np.concatenate([[0.0], fps / n_neg])

        return float(np.trapezoid(tpr, fpr))

    def reset(self) -> "StreamingAUC":
        """Clear the score and label buffers; returns self."""
        self._scores.clear()
        self._labels.clear()
        return self

    def __repr__(self) -> str:
        return (
            f"StreamingAUC(window_size={self._window_size}, "
            f"buffered={len(self._scores)})"
        )
