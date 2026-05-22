"""Online Random Forest ensemble classifier."""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from .base import BaseEstimator
from .metrics import Accuracy
from .tree import HoeffdingTreeClassifier


def _feature_count(n_features: int, max_features) -> int:
    if isinstance(max_features, int):
        return max(1, min(max_features, n_features))
    if max_features == "sqrt":
        return max(1, int(math.sqrt(n_features)))
    if max_features == "log2":
        return max(1, int(math.log2(max(n_features, 2))))
    raise ValueError(
        f"max_features must be int, 'sqrt', or 'log2'; got {max_features!r}"
    )


class _TreeState:
    """Bundle of per-tree mutable state."""

    __slots__ = ("tree", "feature_subset", "oob_metric", "rng")

    def __init__(
        self,
        tree: HoeffdingTreeClassifier,
        feature_subset: np.ndarray,
        oob_metric: Accuracy,
        rng: np.random.Generator,
    ) -> None:
        self.tree = tree
        self.feature_subset = feature_subset
        self.oob_metric = oob_metric
        self.rng = rng


class OnlineRandomForestClassifier(BaseEstimator):
    """Online Random Forest for streaming classification.

    Each tree in the ensemble sees a random feature subset and a Poisson-weighted
    bootstrap of every chunk.  Samples with Poisson weight 0 (OOB) are used to
    track per-tree out-of-bag accuracy without fitting.

    Parameters
    ----------
    n_estimators : int
        Number of trees.
    max_features : int | 'sqrt' | 'log2'
        Features considered per tree.
    random_state : int | None
        Seed for reproducibility.
    criterion, max_depth, min_samples_split, grace_period, delta, tau
        Passed through to each :class:`~numcompute_stream.tree.HoeffdingTreeClassifier`.
    voting : {'hard', 'soft'}
        Aggregation strategy for :meth:`predict` / :meth:`predict_proba`.
    """

    def __init__(
        self,
        n_estimators: int = 10,
        max_features="sqrt",
        random_state: Optional[int] = None,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 20,
        grace_period: int = 200,
        delta: float = 1e-7,
        tau: float = 0.05,
        voting: str = "hard",
    ) -> None:
        if voting not in ("hard", "soft"):
            raise ValueError(f"voting must be 'hard' or 'soft'; got {voting!r}")
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.grace_period = grace_period
        self.delta = delta
        self.tau = tau
        self.voting = voting

        self._states: Optional[List[_TreeState]] = None
        self._classes: Optional[np.ndarray] = None
        self._n_features: Optional[int] = None
        self._global_rng: Optional[np.random.Generator] = None
        self._chunk_counter: int = 0

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_forest(self, n_features: int, classes: np.ndarray) -> None:
        self._n_features = n_features
        self._classes = classes
        seed = self.random_state if self.random_state is not None else None
        self._global_rng = np.random.default_rng(seed)

        k = _feature_count(n_features, self.max_features)
        all_features = np.arange(n_features)
        self._states = []

        for i in range(self.n_estimators):
            # Derive a reproducible seed per tree from the global RNG
            tree_seed = int(self._global_rng.integers(0, 2**31))
            tree_rng = np.random.default_rng(tree_seed)
            subset = np.sort(tree_rng.choice(all_features, size=k, replace=False))

            tree = HoeffdingTreeClassifier(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                grace_period=self.grace_period,
                delta=self.delta,
                tau=self.tau,
            )
            self._states.append(
                _TreeState(tree=tree, feature_subset=subset, oob_metric=Accuracy(), rng=tree_rng)
            )

    # ------------------------------------------------------------------
    # BaseEstimator interface
    # ------------------------------------------------------------------

    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classes: Optional[np.ndarray] = None,
    ) -> "OnlineRandomForestClassifier":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if y.ndim == 0:
            y = y.reshape(1)
        if X.shape[0] == 0:
            return self

        n_samples, n_features = X.shape

        if self._states is None:
            observed = np.unique(y) if classes is None else np.sort(np.asarray(classes, dtype=np.int64))
            self._init_forest(n_features, observed)
        else:
            if n_features != self._n_features:
                raise ValueError(
                    f"Expected {self._n_features} features; got {n_features}"
                )
            if classes is not None:
                extra = np.sort(np.asarray(classes, dtype=np.int64))
                self._classes = np.unique(np.concatenate([self._classes, extra]))

        self._chunk_counter += 1

        for state in self._states:
            # Per-tree, per-chunk Poisson weights — seeded reproducibly
            weights = state.rng.poisson(1.0, size=n_samples)

            oob_mask = weights == 0
            fit_mask = weights >= 1

            # OOB scoring (tree must be initialised before scoring)
            if state.tree._root is not None and oob_mask.any():
                X_oob = X[oob_mask][:, state.feature_subset]
                y_oob = y[oob_mask]
                y_pred_oob = state.tree.predict(X_oob)
                state.oob_metric.update(y_oob, y_pred_oob)

            # Fit on in-bag samples, replicating by Poisson weight
            if fit_mask.any():
                X_inbag_rows = []
                y_inbag_rows = []
                for idx in np.where(fit_mask)[0]:
                    w = int(weights[idx])
                    X_inbag_rows.append(np.tile(X[idx : idx + 1, :], (w, 1)))
                    y_inbag_rows.append(np.full(w, y[idx], dtype=np.int64))

                X_inbag = np.concatenate(X_inbag_rows, axis=0)
                y_inbag = np.concatenate(y_inbag_rows, axis=0)

                X_sub = X_inbag[:, state.feature_subset]
                state.tree.partial_fit(X_sub, y_inbag, classes=self._classes)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._states is None:
            raise ValueError("Call partial_fit before predict.")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples = X.shape[0]

        if self.voting == "soft":
            proba = self.predict_proba(X)
            return self._classes[np.argmax(proba, axis=1)]

        # hard voting: collect per-tree predictions then majority vote
        votes = np.zeros((n_samples, len(self._classes)), dtype=np.int64)
        class_to_idx = {int(c): i for i, c in enumerate(self._classes)}

        for state in self._states:
            if state.tree._root is None:
                continue
            preds = state.tree.predict(X[:, state.feature_subset])
            for j, p in enumerate(preds):
                idx = class_to_idx.get(int(p))
                if idx is not None:
                    votes[j, idx] += 1

        # ties → lowest class label (argmax returns first maximum)
        return self._classes[np.argmax(votes, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._states is None:
            raise ValueError("Call partial_fit before predict_proba.")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples = X.shape[0]
        n_classes = len(self._classes)
        accum = np.zeros((n_samples, n_classes), dtype=np.float64)
        n_fit = 0

        for state in self._states:
            if state.tree._root is None:
                continue
            # Tree may have seen a subset of classes; align to global class order
            tree_classes = state.tree._classes
            proba_tree = state.tree.predict_proba(X[:, state.feature_subset])
            # Map tree columns → global columns
            for ti, tc in enumerate(tree_classes):
                gi = np.searchsorted(self._classes, tc)
                if gi < n_classes and self._classes[gi] == tc:
                    accum[:, gi] += proba_tree[:, ti]
            n_fit += 1

        if n_fit == 0:
            return np.full((n_samples, n_classes), 1.0 / n_classes)

        accum /= n_fit
        # Re-normalise rows that don't sum to 1 (can happen when tree classes ≠ global)
        row_sums = accum.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return accum / row_sums

    # ------------------------------------------------------------------
    # Introspection properties
    # ------------------------------------------------------------------

    @property
    def oob_accuracy_(self) -> List[float]:
        """OOB accuracy for each tree (0.0 if no OOB samples seen yet)."""
        if self._states is None:
            return []
        return [state.oob_metric.result() for state in self._states]

    @property
    def feature_subsets_(self) -> List[np.ndarray]:
        """Feature index subset for each tree."""
        if self._states is None:
            return []
        return [state.feature_subset.copy() for state in self._states]

    def __repr__(self) -> str:
        return (
            f"OnlineRandomForestClassifier("
            f"n_estimators={self.n_estimators}, "
            f"max_features={self.max_features!r}, "
            f"voting={self.voting!r})"
        )
