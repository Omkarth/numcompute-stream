"""Hoeffding Tree classifier — structure and leaf statistics."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .base import BaseEstimator
from .stats import RunningStats


# ---------------------------------------------------------------------------
# Impurity functions
# ---------------------------------------------------------------------------

def _gini(class_counts: np.ndarray) -> float:
    """Gini impurity for a node with the given class-count vector."""
    total = float(class_counts.sum())
    if total == 0.0:
        return 0.0
    p = class_counts / total
    return float(1.0 - (p ** 2).sum())


def _entropy(class_counts: np.ndarray) -> float:
    """Shannon entropy (base-2) for a node with the given class-count vector."""
    total = float(class_counts.sum())
    if total == 0.0:
        return 0.0
    mask = class_counts > 0
    p = class_counts[mask] / total
    return float(-np.sum(p * np.log2(p)))


# ---------------------------------------------------------------------------
# Leaf statistics
# ---------------------------------------------------------------------------

class LeafStats:
    """Sufficient statistics maintained at each tree leaf.

    Tracks per-class sample counts and per-(feature, class) Gaussian summaries
    (via :class:`~numcompute_stream.stats.RunningStats`) needed to evaluate
    split candidates later.

    Parameters
    ----------
    n_features : int
    grace_period : int
        Minimum samples before ``split_eligible`` becomes ``True``.
    """

    def __init__(self, n_features: int, grace_period: int = 200) -> None:
        self._n_features = n_features
        self._grace_period = grace_period
        self._n_samples: int = 0
        self._class_counts: Dict[int, int] = {}
        # (feature_idx, class_label) → RunningStats for that feature's values
        self._feature_stats: Dict[Tuple[int, int], RunningStats] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def update(self, x: np.ndarray, label: int) -> None:
        """Incorporate one sample *x* (shape ``(n_features,)``) with its label."""
        self._n_samples += 1
        self._class_counts[label] = self._class_counts.get(label, 0) + 1

        for feat_idx in range(self._n_features):
            val = x[feat_idx]
            if np.isnan(val):
                continue
            key = (feat_idx, label)
            if key not in self._feature_stats:
                self._feature_stats[key] = RunningStats(n_features=1)
            self._feature_stats[key].update(np.array([[val]]))

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def n_samples(self) -> int:
        """Total samples routed to this leaf."""
        return self._n_samples

    @property
    def class_counts(self) -> Dict[int, int]:
        return dict(self._class_counts)

    @property
    def split_eligible(self) -> bool:
        """``True`` once the leaf has seen at least ``grace_period`` samples."""
        return self._n_samples >= self._grace_period

    def majority_class(self) -> Optional[int]:
        """Label with the highest count, or ``None`` if leaf is empty."""
        if not self._class_counts:
            return None
        return max(self._class_counts, key=lambda k: self._class_counts[k])

    def class_counts_array(self, classes: np.ndarray) -> np.ndarray:
        """Counts aligned to *classes* (length ``len(classes)``)."""
        return np.array(
            [self._class_counts.get(int(c), 0) for c in classes],
            dtype=np.float64,
        )

    def get_feature_stats(self, feat_idx: int, label: int) -> Optional[RunningStats]:
        return self._feature_stats.get((feat_idx, label))

    def __repr__(self) -> str:
        return (
            f"LeafStats(n_samples={self._n_samples}, "
            f"classes={sorted(self._class_counts)})"
        )


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------

class Node:
    """One node in the Hoeffding Tree.

    A node is either:

    * a **leaf** — ``is_leaf`` is ``True``, ``leaf_stats`` is set, no children
    * an **internal node** — ``is_leaf`` is ``False``, children defined by
      ``split_feature`` and ``split_threshold`` (left ≤ threshold,
      right > threshold)
    """

    def __init__(
        self,
        leaf_stats: Optional[LeafStats] = None,
        depth: int = 0,
    ) -> None:
        self.leaf_stats: Optional[LeafStats] = leaf_stats
        self.depth: int = depth
        self.split_feature: Optional[int] = None
        self.split_threshold: Optional[float] = None
        self.left_child: Optional[Node] = None
        self.right_child: Optional[Node] = None

    @property
    def is_leaf(self) -> bool:
        return self.split_feature is None

    def to_internal(
        self,
        split_feature: int,
        split_threshold: float,
        left_child: Node,
        right_child: Node,
    ) -> None:
        """Convert this leaf node into an internal node in-place."""
        self.leaf_stats = None
        self.split_feature = split_feature
        self.split_threshold = split_threshold
        self.left_child = left_child
        self.right_child = right_child

    def __repr__(self) -> str:
        if self.is_leaf:
            return f"Node(leaf, depth={self.depth}, stats={self.leaf_stats})"
        return (
            f"Node(internal, depth={self.depth}, "
            f"feat={self.split_feature}, "
            f"thresh={self.split_threshold:.4g})"
        )


# ---------------------------------------------------------------------------
# Hoeffding Tree classifier
# ---------------------------------------------------------------------------

class HoeffdingTreeClassifier(BaseEstimator):
    """Very Fast Decision Tree (VFDT) classifier.

    Learns an axis-aligned binary decision tree from a stream of labelled
    examples.  This implementation covers the tree *structure* and
    per-leaf statistic accumulation; split decisions are handled separately.

    Parameters
    ----------
    criterion : {'gini', 'entropy'}
        Impurity measure for split-candidate evaluation.
    max_depth : int | None
        Maximum tree depth.  ``None`` means unbounded.
    min_samples_split : int
        Minimum samples at a leaf before it may be split.
    grace_period : int
        Samples between consecutive split evaluations at a leaf.
    delta : float
        Confidence parameter for the Hoeffding bound (δ).
    """

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 20,
        grace_period: int = 200,
        delta: float = 1e-7,
    ) -> None:
        if criterion not in ("gini", "entropy"):
            raise ValueError(f"criterion must be 'gini' or 'entropy'; got {criterion!r}")
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.grace_period = grace_period
        self.delta = delta

        self._root: Optional[Node] = None
        self._classes: Optional[np.ndarray] = None
        self._n_features: Optional[int] = None
        self._impurity_fn = _gini if criterion == "gini" else _entropy

    # ------------------------------------------------------------------
    # BaseEstimator interface
    # ------------------------------------------------------------------

    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classes: Optional[np.ndarray] = None,
    ) -> "HoeffdingTreeClassifier":
        """Route samples to leaves and update leaf statistics.

        Splitting is *not* triggered here; this method only accumulates
        sufficient statistics at each leaf.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)

        if X.ndim == 1:
            X = X.reshape(1, -1)
        if y.ndim == 0:
            y = y.reshape(1)

        if X.shape[0] == 0:
            return self

        n_samples, n_features = X.shape

        if self._root is None:
            self._n_features = n_features
            self._classes = (
                np.sort(np.asarray(classes, dtype=np.int64))
                if classes is not None
                else np.unique(y)
            )
            self._root = Node(
                leaf_stats=LeafStats(n_features, grace_period=self.grace_period),
                depth=0,
            )
        else:
            if n_features != self._n_features:
                raise ValueError(
                    f"Expected {self._n_features} features; got {n_features}"
                )
            if classes is not None:
                extra = np.sort(np.asarray(classes, dtype=np.int64))
                self._classes = np.unique(np.concatenate([self._classes, extra]))

        for i in range(n_samples):
            leaf = self._route_to_leaf(X[i])
            leaf.leaf_stats.update(X[i], int(y[i]))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the majority-class label for each sample in *X*."""
        if self._root is None:
            raise ValueError("Call partial_fit before predict.")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        out = np.empty(X.shape[0], dtype=np.int64)
        for i in range(X.shape[0]):
            leaf = self._route_to_leaf(X[i])
            mc = leaf.leaf_stats.majority_class()
            out[i] = mc if mc is not None else int(self._classes[0])
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _route_to_leaf(self, x: np.ndarray) -> Node:
        """Walk *x* down the tree and return its leaf node."""
        node = self._root
        while not node.is_leaf:
            val = x[node.split_feature]
            # NaN goes left (conservative default)
            node = node.left_child if (np.isnan(val) or val <= node.split_threshold) else node.right_child
        return node

    def __repr__(self) -> str:
        return (
            f"HoeffdingTreeClassifier("
            f"criterion={self.criterion!r}, "
            f"max_depth={self.max_depth}, "
            f"delta={self.delta})"
        )
