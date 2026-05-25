"""Streaming preprocessors: StandardScaler, Imputer, OneHotEncoder."""

from __future__ import annotations

from typing import Literal

import numpy as np

from .base import BaseTransformer
from .utils import check_array


class StandardScaler(BaseTransformer):
    """Standardise features to zero mean and unit variance.

    Fits incrementally using a numerically stable batch Welford (Chan's
    parallel) update.  NaN values are skipped during fitting; NaN values
    passed to ``transform`` are propagated unchanged.

    For columns that have only ever seen NaN values (no valid observations),
    ``transform`` outputs NaN for every row of that column.

    Parameters
    ----------
    copy : bool
        When True (default), ``transform`` returns a new array.

    Examples
    --------
    >>> sc = StandardScaler()
    >>> sc.partial_fit(np.array([[1.0, 2.0], [3.0, 4.0]]))
    >>> sc.transform(np.array([[1.0, 2.0]]))
    array([[-1.,  -1.]])
    """

    def __init__(self, copy: bool = True) -> None:
        """Initialise StandardScaler; see class docstring for parameter details."""
        self._copy = copy
        self._n_features: int | None = None
        self._n_seen: np.ndarray | None = None   # per-column valid-value count
        self._mean: np.ndarray | None = None
        self._M2: np.ndarray | None = None       # sum of squared deviations

    # BaseTransformer

    def partial_fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "StandardScaler":
        """Update running mean and variance with one chunk.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            NaN values are silently skipped; they do not contribute to the
            running statistics.
        """
        X = np.asarray(X, dtype=np.float64)
        check_array(X, ndim=2, allow_nan=True)
        _, n_features = X.shape

        if self._mean is None:
            self._n_features = n_features
            self._n_seen = np.zeros(n_features, dtype=np.float64)
            self._mean = np.zeros(n_features, dtype=np.float64)
            self._M2 = np.zeros(n_features, dtype=np.float64)
        elif n_features != self._n_features:
            raise ValueError(
                f"Expected {self._n_features} feature(s); got {n_features}"
            )

        mask = ~np.isnan(X)                                    # (n, p)
        n_new = mask.sum(axis=0).astype(np.float64)           # valid per column
        safe_X = np.where(mask, X, 0.0)
        batch_mean = safe_X.sum(axis=0) / np.maximum(n_new, 1.0)
        diff = np.where(mask, X - batch_mean, 0.0)
        batch_M2 = (diff ** 2).sum(axis=0)

        n_old = self._n_seen
        n_combined = n_old + n_new
        safe_comb = np.maximum(n_combined, 1.0)
        delta = batch_mean - self._mean
        self._mean = np.where(
            n_combined > 0,
            self._mean + delta * (n_new / safe_comb),
            self._mean,
        )
        self._M2 += batch_M2 + delta ** 2 * (n_old * n_new / safe_comb)
        self._n_seen = n_combined
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Standardise X using the running mean and std.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            NaN values are allowed; they pass through as NaN in the output.

        Returns
        -------
        X_scaled : ndarray of same shape.

        Raises
        ------
        RuntimeError : if ``partial_fit`` has not been called yet.
        """
        if self._mean is None:
            raise RuntimeError("StandardScaler has not been fitted yet.")
        X = np.asarray(X, dtype=np.float64)
        check_array(X, ndim=2, allow_nan=True)
        if X.shape[1] != self._n_features:
            raise ValueError(
                f"Expected {self._n_features} feature(s); got {X.shape[1]}"
            )
        if self._copy:
            X = X.copy()

        std = np.sqrt(self._M2 / np.maximum(self._n_seen, 1.0))
        safe_std = np.where(std == 0.0, 1.0, std)   # constant col → output 0
        X -= self._mean
        X /= safe_std
        # Columns with zero valid observations have no statistics → NaN out.
        never_seen = self._n_seen == 0
        if never_seen.any():
            X[:, never_seen] = np.nan
        return X

    # Properties

    @property
    def mean_(self) -> np.ndarray:
        """Per-feature running mean, shape ``(n_features,)``."""
        if self._mean is None:
            raise RuntimeError("StandardScaler has not been fitted yet.")
        return self._mean.copy()

    @property
    def scale_(self) -> np.ndarray:
        """Per-feature running std (population), shape ``(n_features,)``."""
        if self._mean is None:
            raise RuntimeError("StandardScaler has not been fitted yet.")
        return np.sqrt(self._M2 / np.maximum(self._n_seen, 1.0))

    def __repr__(self) -> str:
        return (
            f"StandardScaler(n_features={self._n_features}, "
            f"n_seen={self._n_seen})"
        )


class Imputer(BaseTransformer):
    """Fill missing values (NaN) with a running per-column statistic.

    Parameters
    ----------
    strategy : {'mean', 'most_frequent'}
        ``'mean'``          — replace NaN with the running per-column mean of
                             all non-NaN values seen so far.  For a column that
                             has only seen NaN values, the fill value is 0.0.
        ``'most_frequent'`` — replace NaN with the most frequently seen value
                             per column.  Suitable for integer-coded categories.
                             For a column that has only seen NaN values, NaN is
                             left in place.

    Examples
    --------
    >>> imp = Imputer(strategy='mean')
    >>> imp.partial_fit(np.array([[1.0, 2.0], [3.0, 4.0]]))
    >>> imp.transform(np.array([[np.nan, 2.0]]))
    array([[2., 2.]])
    """

    def __init__(self, strategy: Literal["mean", "most_frequent"] = "mean") -> None:
        """Initialise Imputer; see class docstring for parameter details."""
        if strategy not in ("mean", "most_frequent"):
            raise ValueError(
                f"strategy must be 'mean' or 'most_frequent'; got {strategy!r}"
            )
        self._strategy = strategy
        self._n_features: int | None = None
        self._n_seen: np.ndarray | None = None   # used by 'mean' only
        self._mean: np.ndarray | None = None      # used by 'mean' only
        self._freq: list[dict] | None = None      # used by 'most_frequent' only

    # BaseTransformer

    def partial_fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "Imputer":
        """Update running statistics with one chunk.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            NaN values are ignored; they do not contribute to any statistic.
        """
        X = np.asarray(X, dtype=np.float64)
        check_array(X, ndim=2, allow_nan=True)
        _, n_features = X.shape

        if self._n_features is None:
            self._n_features = n_features
            if self._strategy == "mean":
                self._n_seen = np.zeros(n_features, dtype=np.float64)
                self._mean = np.zeros(n_features, dtype=np.float64)
            else:
                self._freq = [{} for _ in range(n_features)]
        elif n_features != self._n_features:
            raise ValueError(
                f"Expected {self._n_features} feature(s); got {n_features}"
            )

        mask = ~np.isnan(X)

        if self._strategy == "mean":
            n_new = mask.sum(axis=0).astype(np.float64)
            safe_X = np.where(mask, X, 0.0)
            batch_sum = safe_X.sum(axis=0)
            n_combined = self._n_seen + n_new
            old_sum = self._mean * self._n_seen
            self._mean = np.where(
                n_combined > 0,
                (old_sum + batch_sum) / np.maximum(n_combined, 1.0),
                self._mean,
            )
            self._n_seen = n_combined
        else:
            for j in range(n_features):
                valid = X[:, j][mask[:, j]]
                if valid.size == 0:
                    continue
                vals, counts = np.unique(valid, return_counts=True)
                freq_j = self._freq[j]
                for v, c in zip(vals.tolist(), counts.tolist()):
                    freq_j[v] = freq_j.get(v, 0) + c

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Fill NaN values in X using the learned statistics.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        X_imputed : ndarray of same shape with NaN replaced where possible.
            For ``strategy='mean'`` and an all-NaN column: fill value is 0.0.
            For ``strategy='most_frequent'`` and an all-NaN column: NaN remains.

        Raises
        ------
        RuntimeError : if ``partial_fit`` has not been called yet.
        """
        if self._n_features is None:
            raise RuntimeError("Imputer has not been fitted yet.")
        X = np.asarray(X, dtype=np.float64)
        check_array(X, ndim=2, allow_nan=True)
        if X.shape[1] != self._n_features:
            raise ValueError(
                f"Expected {self._n_features} feature(s); got {X.shape[1]}"
            )
        X = X.copy()
        nan_locs = np.isnan(X)
        if not nan_locs.any():
            return X

        if self._strategy == "mean":
            # Vectorised: broadcast fill values across rows where NaN
            fill = self._mean                            # (n_features,)
            X[nan_locs] = np.broadcast_to(fill, X.shape)[nan_locs]
        else:
            for j in range(self._n_features):
                col_nan = nan_locs[:, j]
                if not col_nan.any():
                    continue
                freq_j = self._freq[j]
                if not freq_j:
                    continue                              # no data → leave NaN
                mode_val = max(freq_j, key=lambda k: freq_j[k])
                X[col_nan, j] = mode_val

        return X

    # Properties

    @property
    def strategy(self) -> str:
        """The imputation strategy: ``'mean'`` or ``'most_frequent'``."""
        return self._strategy

    @property
    def statistics_(self) -> np.ndarray:
        """Per-column fill values as a 1-D float array.

        For ``strategy='most_frequent'``, columns with no data return NaN.
        """
        if self._n_features is None:
            raise RuntimeError("Imputer has not been fitted yet.")
        if self._strategy == "mean":
            return self._mean.copy()
        result = np.full(self._n_features, np.nan)
        for j, freq_j in enumerate(self._freq):
            if freq_j:
                result[j] = max(freq_j, key=lambda k: freq_j[k])
        return result

    def __repr__(self) -> str:
        return (
            f"Imputer(strategy={self._strategy!r}, "
            f"n_features={self._n_features})"
        )


class OneHotEncoder(BaseTransformer):
    """Encode categorical columns as binary indicator arrays.

    **Streaming semantics**: The set of known categories grows monotonically
    as ``partial_fit`` observes new values.  The output column count equals
    the total number of unique categories seen so far across all input
    columns.  Calling ``partial_fit`` after a previous ``transform`` may
    widen the output — consumers must re-check ``n_features_out_`` after
    each ``partial_fit`` call.

    Values encountered in ``transform`` that were never seen in
    ``partial_fit`` produce an all-zero indicator row for that input column
    (unknown-category handling).

    NaN values in ``X`` during ``partial_fit`` are silently ignored (they
    are not treated as a category).  NaN values in ``X`` during ``transform``
    produce an all-zero indicator row for that input column.

    Parameters
    ----------
    dtype : numpy dtype, default float64
        Data type of the indicator output.

    Examples
    --------
    >>> enc = OneHotEncoder()
    >>> enc.partial_fit(np.array([[0.0], [1.0], [2.0]]))
    >>> enc.transform(np.array([[0.0], [3.0]]))  # 3 never seen → all zeros
    array([[1., 0., 0.],
           [0., 0., 0.]])
    >>> enc.partial_fit(np.array([[3.0]]))        # new category discovered
    >>> enc.n_features_out_                       # output is now wider
    4
    """

    def __init__(self, dtype: np.dtype = np.float64) -> None:
        """Initialise OneHotEncoder; see class docstring for parameter details."""
        self._dtype = np.dtype(dtype)
        self._n_features: int | None = None
        # One sorted list of known category values per input column.
        self._categories: list[list[float]] | None = None

    # BaseTransformer

    def partial_fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "OneHotEncoder":
        """Update the known category sets from one chunk.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            NaN values are ignored and do not become categories.
        """
        X = np.asarray(X, dtype=np.float64)
        check_array(X, ndim=2, allow_nan=True)
        _, n_features = X.shape

        if self._categories is None:
            self._n_features = n_features
            self._categories = [[] for _ in range(n_features)]
        elif n_features != self._n_features:
            raise ValueError(
                f"Expected {self._n_features} feature(s); got {n_features}"
            )

        for j in range(n_features):
            col = X[:, j]
            valid = col[~np.isnan(col)]
            if valid.size == 0:
                continue
            new_vals = np.unique(valid)
            known = set(self._categories[j])
            added = [v for v in new_vals.tolist() if v not in known]
            if added:
                self._categories[j].extend(added)
                self._categories[j].sort()

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """One-hot encode X using all categories known at call time.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            NaN values produce an all-zero indicator row for that column.

        Returns
        -------
        X_ohe : ndarray of shape (n_samples, n_features_out_)
            Output width equals ``n_features_out_`` at the time of the call.
            This width may grow between calls if ``partial_fit`` has been
            called in between.

        Raises
        ------
        RuntimeError : if ``partial_fit`` has not been called yet.
        """
        if self._categories is None:
            raise RuntimeError("OneHotEncoder has not been fitted yet.")
        X = np.asarray(X, dtype=np.float64)
        check_array(X, ndim=2, allow_nan=True)
        if X.shape[1] != self._n_features:
            raise ValueError(
                f"Expected {self._n_features} feature(s); got {X.shape[1]}"
            )
        n_samples = X.shape[0]
        parts: list[np.ndarray] = []

        for j in range(self._n_features):
            cats = self._categories[j]
            if not cats:
                continue
            cats_arr = np.array(cats, dtype=np.float64)   # (n_cats,)
            col = X[:, j].reshape(-1, 1)                  # (n_samples, 1)
            # NaN == anything is False → NaN rows produce all-zero rows
            indicator = (col == cats_arr).astype(self._dtype)
            parts.append(indicator)

        if not parts:
            return np.empty((n_samples, 0), dtype=self._dtype)
        return np.concatenate(parts, axis=1)

    # Properties

    @property
    def categories_(self) -> list[list[float]]:
        """Per-column list of known categories (sorted ascending).

        Length equals n_features; each sub-list may grow after ``partial_fit``.
        """
        if self._categories is None:
            raise RuntimeError("OneHotEncoder has not been fitted yet.")
        return [list(c) for c in self._categories]

    @property
    def n_features_out_(self) -> int:
        """Total output columns — sum of category counts across all input columns.

        May increase after each ``partial_fit`` call.
        """
        if self._categories is None:
            raise RuntimeError("OneHotEncoder has not been fitted yet.")
        return sum(len(c) for c in self._categories)

    def __repr__(self) -> str:
        n_cats = (
            [len(c) for c in self._categories]
            if self._categories is not None
            else None
        )
        return (
            f"OneHotEncoder(n_features={self._n_features}, "
            f"n_cats_per_col={n_cats})"
        )
