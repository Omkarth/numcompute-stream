"""Streaming statistics: Welford online mean/variance and related utilities."""

from __future__ import annotations

import numpy as np

from .utils import check_array


class RunningStats:
    """Online mean and variance via Welford's one-pass algorithm.

    Accepts batches of shape ``(n_samples,)`` (scalar stream) or
    ``(n_samples, n_features)`` (multivariate stream).  All per-feature
    statistics are maintained independently with no Python loops.

    Parameters
    ----------
    n_features : int | None
        Number of features.  If None, inferred from the first call to
        ``update``.  Must be provided when creating from scratch if you want to
        call ``mean`` / ``var`` before any data arrives.

    Examples
    --------
    >>> rs = RunningStats()
    >>> rs.update(np.array([[1.0, 2.0], [3.0, 4.0]]))
    >>> rs.mean
    array([2., 3.])
    """

    def __init__(self, n_features: int | None = None) -> None:
        if n_features is not None and n_features < 1:
            raise ValueError(f"n_features must be ≥ 1; got {n_features}")
        self._n_features = n_features
        self._count: int = 0
        self._mean: np.ndarray | None = None
        self._M2: np.ndarray | None = None   # sum of squared deviations from mean

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, X: np.ndarray) -> "RunningStats":
        """Incorporate a new batch of observations.

        Parameters
        ----------
        X : ndarray of shape ``(n_samples,)`` or ``(n_samples, n_features)``
            NaN values are **not** allowed; impute before calling.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError(f"X must be 1-D or 2-D; got shape {X.shape}")

        check_array(X, ndim=2, allow_nan=False)

        n_new, n_feat = X.shape

        if self._mean is None:
            # First batch ever — initialise accumulators.
            if self._n_features is not None and n_feat != self._n_features:
                raise ValueError(
                    f"Expected {self._n_features} feature(s); got {n_feat}"
                )
            self._n_features = n_feat
            self._mean = np.zeros(n_feat)
            self._M2 = np.zeros(n_feat)
        elif n_feat != self._n_features:
            raise ValueError(
                f"Expected {self._n_features} feature(s); got {n_feat}"
            )

        # Chan's parallel / batch Welford update (numerically stable).
        # mean_new = mean of incoming batch
        # delta    = difference between batch mean and running mean
        batch_mean = X.mean(axis=0)
        batch_M2 = ((X - batch_mean) ** 2).sum(axis=0)

        n_old = self._count
        n_combined = n_old + n_new

        delta = batch_mean - self._mean
        self._mean = self._mean + delta * (n_new / n_combined)
        self._M2 = (
            self._M2
            + batch_M2
            + delta ** 2 * (n_old * n_new / n_combined)
        )
        self._count = n_combined

        return self

    def reset(self) -> "RunningStats":
        """Clear all accumulated state while keeping ``n_features``."""
        self._count = 0
        self._mean = None
        self._M2 = None
        return self

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Total number of samples seen so far."""
        return self._count

    @property
    def mean(self) -> np.ndarray:
        """Per-feature running mean.  Shape ``(n_features,)``."""
        self._require_data("mean")
        return self._mean.copy()

    @property
    def var(self) -> np.ndarray:
        """Per-feature *population* variance.  Shape ``(n_features,)``.

        Returns zero for a single-sample stream (undefined but numerically safe).
        """
        self._require_data("var")
        if self._count < 2:
            return np.zeros(self._n_features)
        return self._M2 / self._count

    @property
    def var_unbiased(self) -> np.ndarray:
        """Per-feature *sample* (unbiased) variance.  Shape ``(n_features,)``.

        Raises
        ------
        ValueError : if fewer than 2 samples have been seen.
        """
        self._require_data("var_unbiased")
        if self._count < 2:
            raise ValueError(
                "Unbiased variance is undefined with fewer than 2 samples."
            )
        return self._M2 / (self._count - 1)

    @property
    def std(self) -> np.ndarray:
        """Per-feature population standard deviation.  Shape ``(n_features,)``."""
        return np.sqrt(self.var)

    @property
    def std_unbiased(self) -> np.ndarray:
        """Per-feature sample (unbiased) standard deviation."""
        return np.sqrt(self.var_unbiased)

    # ------------------------------------------------------------------
    # Merging
    # ------------------------------------------------------------------

    def merge(self, other: "RunningStats") -> "RunningStats":
        """Return a new RunningStats combining self and other (non-destructive).

        Both must have seen at least one sample and have the same n_features.
        """
        if self._count == 0:
            return other._copy()
        if other._count == 0:
            return self._copy()

        if self._n_features != other._n_features:
            raise ValueError(
                f"Cannot merge RunningStats with different n_features: "
                f"{self._n_features} vs {other._n_features}"
            )

        merged = RunningStats(n_features=self._n_features)
        n_a, n_b = self._count, other._count
        n_combined = n_a + n_b

        delta = other._mean - self._mean
        merged._mean = self._mean + delta * (n_b / n_combined)
        merged._M2 = (
            self._M2
            + other._M2
            + delta ** 2 * (n_a * n_b / n_combined)
        )
        merged._count = n_combined
        return merged

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_data(self, name: str) -> None:
        if self._count == 0:
            raise ValueError(
                f"Cannot compute {name} — no data has been observed yet."
            )

    def _copy(self) -> "RunningStats":
        rs = RunningStats(n_features=self._n_features)
        rs._count = self._count
        rs._mean = None if self._mean is None else self._mean.copy()
        rs._M2 = None if self._M2 is None else self._M2.copy()
        return rs

    def __repr__(self) -> str:
        return (
            f"RunningStats(n_features={self._n_features}, count={self._count})"
        )
