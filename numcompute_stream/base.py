"""Abstract base classes for all numcompute_stream components."""

from __future__ import annotations

import abc
import numpy as np


class BaseTransformer(abc.ABC):
    """Streaming transformer: fit incrementally, transform in-place."""

    @abc.abstractmethod
    def partial_fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "BaseTransformer":
        """Update internal state with one chunk.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ignored by most transformers, accepted for pipeline compatibility

        Returns
        -------
        self
        """

    @abc.abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply learned transformation.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_transformed_features)
        """

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        return self.partial_fit(X, y).transform(X)


class BaseEstimator(abc.ABC):
    """Streaming estimator: learns incrementally, predicts on demand."""

    @abc.abstractmethod
    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classes: np.ndarray | None = None,
    ) -> "BaseEstimator":
        """Update model with one chunk.

        Parameters
        ----------
        X      : ndarray of shape (n_samples, n_features)
        y      : ndarray of shape (n_samples,)  — integer class labels
        classes: 1-D array of all possible class labels; required on first call
                 for multi-class models that need to know the label space upfront

        Returns
        -------
        self
        """

    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,)
        """

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_classes)

        Raises
        ------
        NotImplementedError
            If the estimator does not support probability estimates.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement predict_proba."
        )


class BaseMetric(abc.ABC):
    """Streaming metric: accumulates state across chunks, emits a scalar."""

    @abc.abstractmethod
    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> "BaseMetric":
        """Incorporate one chunk of predictions.

        Parameters
        ----------
        y_true : ndarray of shape (n_samples,)
        y_pred : ndarray of shape (n_samples,)

        Returns
        -------
        self
        """

    @abc.abstractmethod
    def result(self) -> float:
        """Return the current metric value."""

    @abc.abstractmethod
    def reset(self) -> "BaseMetric":
        """Clear all accumulated state.

        Returns
        -------
        self
        """


class BaseDriftDetector(abc.ABC):
    """Streaming drift detector: signals concept drift from a 1-D value stream."""

    @abc.abstractmethod
    def update(self, value: float) -> "BaseDriftDetector":
        """Incorporate one observation.

        Parameters
        ----------
        value : scalar observation (e.g. per-chunk accuracy, 0/1 error)

        Returns
        -------
        self
        """

    @property
    @abc.abstractmethod
    def drift_detected(self) -> bool:
        """True if drift was detected on the most recent update."""

    @property
    @abc.abstractmethod
    def warning_detected(self) -> bool:
        """True if a warning zone was entered on the most recent update."""

    @abc.abstractmethod
    def reset(self) -> "BaseDriftDetector":
        """Reset detector to its initial state.

        Returns
        -------
        self
        """
