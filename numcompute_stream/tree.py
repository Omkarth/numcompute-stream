"""Hoeffding Tree classifier — structure, leaf statistics, and split logic."""

from __future__ import annotations

import math as _math
from typing import Dict, Optional, Tuple

import numpy as np

from .base import BaseEstimator
from .stats import RunningStats


# ---------------------------------------------------------------------------
# Impurity functions (scalar — used for parent node)
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
# Vectorised impurity (over candidate thresholds)
# ---------------------------------------------------------------------------

def _gini_vec(counts: np.ndarray) -> np.ndarray:
    """Gini impurity for each row of *counts* (shape ``(n_thresh, n_classes)``)."""
    totals = counts.sum(axis=1, keepdims=True)
    safe = np.where(totals > 0, totals, 1.0)
    p = counts / safe
    result = 1.0 - (p ** 2).sum(axis=1)
    result[totals.ravel() == 0] = 0.0
    return result


def _entropy_vec(counts: np.ndarray) -> np.ndarray:
    """Shannon entropy for each row of *counts* (shape ``(n_thresh, n_classes)``)."""
    totals = counts.sum(axis=1, keepdims=True)
    safe = np.where(totals > 0, totals, 1.0)
    p = counts / safe
    log_p = np.where(p > 0, np.log2(np.where(p > 0, p, 1.0)), 0.0)
    result = -(p * log_p).sum(axis=1)
    result[totals.ravel() == 0] = 0.0
    return result


# ---------------------------------------------------------------------------
# Hoeffding bound
# ---------------------------------------------------------------------------

def _hoeffding_bound(n: int, delta: float, criterion: str, n_classes: int) -> float:
    """Hoeffding bound ε = sqrt(R² · ln(1/δ) / (2n)).

    For *gini* R = 1; for *entropy* R = log₂(n_classes).
    """
    if criterion == "gini":
        R = 1.0
    else:
        R = np.log2(max(n_classes, 2))
    return float(np.sqrt(R ** 2 * np.log(1.0 / delta) / (2.0 * n)))


# ---------------------------------------------------------------------------
# Gaussian CDF (no scipy dependency)
# ---------------------------------------------------------------------------

_verf = np.frompyfunc(_math.erf, 1, 1)


def _normal_cdf(z: np.ndarray) -> np.ndarray:
    """Standard-normal CDF: Φ(z) = (1 + erf(z / √2)) / 2 (no scipy needed)."""
    return 0.5 * (1.0 + _verf(z / _math.sqrt(2.0))).astype(np.float64)


# ---------------------------------------------------------------------------
# Candidate threshold generation
# ---------------------------------------------------------------------------

def _candidate_thresholds(
    leaf_stats: "LeafStats",
    feat_idx: int,
    classes: np.ndarray,
) -> np.ndarray:
    """Generate ≈10 candidate split thresholds for *feat_idx* from Gaussian summaries.

    Strategy:
    * Midpoints between consecutive class-conditional means (sorted).
    * Global mean ± {0.5, 1.0} × global std as quantile-based candidates.
    """
    valid_rs = []
    means = []
    for c in classes:
        rs = leaf_stats.get_feature_stats(feat_idx, int(c))
        if rs is not None and rs.count > 0:
            mu = float(rs.mean[0])
            cnt = leaf_stats._class_counts.get(int(c), 0)
            valid_rs.append((rs, mu, cnt))
            means.append(mu)

    if not means:
        return np.array([])

    thresholds: list[float] = []

    # midpoints between sorted class means
    sorted_means = np.sort(np.array(means))
    mids = (sorted_means[:-1] + sorted_means[1:]) / 2.0
    thresholds.extend(mids.tolist())

    # global weighted mean and std for quantile candidates
    total_cnt = sum(cnt for _, _, cnt in valid_rs)
    if total_cnt > 0:
        g_mean = sum(mu * cnt for _, mu, cnt in valid_rs) / total_cnt
        # law of total variance
        g_var = sum(
            cnt * (rs.var[0] + (mu - g_mean) ** 2)
            for rs, mu, cnt in valid_rs
        ) / total_cnt
        g_std = float(np.sqrt(max(g_var, 0.0)))
        if g_std > 1e-10:
            for mult in (-1.0, -0.5, 0.5, 1.0):
                thresholds.append(g_mean + mult * g_std)

    return np.unique(thresholds) if thresholds else np.array([])


# ---------------------------------------------------------------------------
# Info-gain vectorised over candidate thresholds
# ---------------------------------------------------------------------------

def _info_gains_for_feature(
    leaf_stats: "LeafStats",
    feat_idx: int,
    thresholds: np.ndarray,
    classes: np.ndarray,
    parent_counts: np.ndarray,
    parent_impurity: float,
    criterion: str,
) -> np.ndarray:
    """Return info-gain for each threshold in *thresholds* (shape ``(n_thresh,)``).

    Uses Gaussian CDF over per-class ``RunningStats`` to estimate how many
    samples fall left/right of each candidate threshold — no raw samples stored.
    """
    n_classes = len(classes)

    per_class_count = parent_counts  # shape (n_classes,)

    # Per-class Gaussian parameters; default sigma large ≡ phi ≈ 0.5 (unknown)
    mus = np.zeros(n_classes)
    sigmas = np.full(n_classes, 1e10)
    for i, c in enumerate(classes):
        rs = leaf_stats.get_feature_stats(feat_idx, int(c))
        if rs is not None and rs.count > 0:
            mus[i] = float(rs.mean[0])
            s = float(rs.std[0])
            sigmas[i] = max(s, 1e-10)

    # z: (n_thresh, n_classes); phi: (n_thresh, n_classes)
    z = (thresholds[:, np.newaxis] - mus[np.newaxis, :]) / sigmas[np.newaxis, :]
    phi = _normal_cdf(z)

    n_left = phi * per_class_count[np.newaxis, :]   # (n_thresh, n_classes)
    n_right = per_class_count[np.newaxis, :] - n_left

    total = float(per_class_count.sum())

    n_l = n_left.sum(axis=1)   # (n_thresh,)
    n_r = n_right.sum(axis=1)

    if criterion == "gini":
        imp_left = _gini_vec(n_left)
        imp_right = _gini_vec(n_right)
    else:
        imp_left = _entropy_vec(n_left)
        imp_right = _entropy_vec(n_right)

    weighted_child = (n_l / total) * imp_left + (n_r / total) * imp_right
    gains = parent_impurity - weighted_child

    # Zero out degenerate splits (empty child)
    valid = (n_l > 0) & (n_r > 0)
    return np.where(valid, gains, 0.0)


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
    examples using the Hoeffding bound to decide when a split is statistically
    warranted.

    Split decision rule (Domingos & Hulten 2000):
    Let G₁, G₂ be the best and second-best info gains across all candidate
    splits at a leaf, and ε the Hoeffding bound.

    * If G₁ − G₂ > ε  → split on the best (feature, threshold).
    * Else if ε < tau  → split anyway (tie-breaking rule: bound is tight
      enough that further waiting is unlikely to change the decision).
    * Otherwise        → wait for more data.

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
    tau : float
        Tie-breaking threshold: when ε < tau, the leading split is chosen
        even if its margin over the second-best gain is smaller than ε.
        Default ``0.05``.
    """

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 20,
        grace_period: int = 200,
        delta: float = 1e-7,
        tau: float = 0.05,
    ) -> None:
        if criterion not in ("gini", "entropy"):
            raise ValueError(f"criterion must be 'gini' or 'entropy'; got {criterion!r}")
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.grace_period = grace_period
        self.delta = delta
        self.tau = tau

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
        """Route samples to leaves, update statistics, and attempt splits.

        A split is evaluated at a leaf every *grace_period* samples routed to
        that leaf.  Whether the split fires depends on the Hoeffding bound.
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
            ns = leaf.leaf_stats.n_samples
            if ns % self.grace_period == 0:
                self._try_split(leaf)

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

    def _find_best_split(
        self, leaf_stats: LeafStats
    ) -> Optional[Tuple[float, float, int, float]]:
        """Search all features for the best and second-best split.

        Returns ``(best_gain, second_best_gain, best_feat, best_thresh)``
        or ``None`` if no candidate thresholds exist anywhere.
        """
        classes = self._classes
        parent_counts = leaf_stats.class_counts_array(classes)
        parent_impurity = self._impurity_fn(parent_counts)

        best_gain: float = -np.inf
        second_best_gain: float = 0.0   # "null split" baseline
        best_feat: Optional[int] = None
        best_thresh: Optional[float] = None

        for feat_idx in range(self._n_features):
            thresholds = _candidate_thresholds(leaf_stats, feat_idx, classes)
            if len(thresholds) == 0:
                continue

            gains = _info_gains_for_feature(
                leaf_stats, feat_idx, thresholds, classes,
                parent_counts, parent_impurity, self.criterion,
            )

            feat_best_idx = int(np.argmax(gains))
            feat_best_gain = float(gains[feat_best_idx])

            if feat_best_gain > best_gain:
                second_best_gain = best_gain if best_gain > 0.0 else second_best_gain
                best_gain = feat_best_gain
                best_feat = feat_idx
                best_thresh = float(thresholds[feat_best_idx])
            elif feat_best_gain > second_best_gain:
                second_best_gain = feat_best_gain

        if best_feat is None:
            return None
        return best_gain, second_best_gain, best_feat, best_thresh

    def _try_split(self, node: Node) -> None:
        """Evaluate the Hoeffding bound and split *node* if warranted."""
        leaf_stats = node.leaf_stats

        if leaf_stats.n_samples < self.min_samples_split:
            return
        if self.max_depth is not None and node.depth >= self.max_depth:
            return

        parent_counts = leaf_stats.class_counts_array(self._classes)
        if self._impurity_fn(parent_counts) == 0.0:
            return  # pure node — nothing to gain from splitting

        result = self._find_best_split(leaf_stats)
        if result is None:
            return

        best_gain, second_best_gain, best_feat, best_thresh = result

        # Number of distinct classes observed at this leaf
        n_seen = int((parent_counts > 0).sum())
        eps = _hoeffding_bound(leaf_stats.n_samples, self.delta, self.criterion, n_seen)

        should_split = (best_gain - second_best_gain > eps) or (eps < self.tau and best_gain > 0.0)
        if not should_split:
            return

        left = Node(
            leaf_stats=LeafStats(self._n_features, grace_period=self.grace_period),
            depth=node.depth + 1,
        )
        right = Node(
            leaf_stats=LeafStats(self._n_features, grace_period=self.grace_period),
            depth=node.depth + 1,
        )
        node.to_internal(best_feat, best_thresh, left, right)

    def __repr__(self) -> str:
        return (
            f"HoeffdingTreeClassifier("
            f"criterion={self.criterion!r}, "
            f"max_depth={self.max_depth}, "
            f"delta={self.delta}, "
            f"tau={self.tau})"
        )
