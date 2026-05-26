"""Streaming pipeline: chained transformers followed by an estimator."""

from __future__ import annotations

from typing import Union

import numpy as np

from .base import BaseEstimator


class Pipeline(BaseEstimator):
    """Chain of transformers followed by a final estimator.

    Parameters
    ----------
    steps : list of (name, object) tuples
        All but the last must be transformers (``partial_fit`` + ``transform``).
        The last must be an estimator (``partial_fit`` + ``predict``).
        Names must be unique strings.

    Raises
    ------
    ValueError
        If *steps* is empty or names are duplicated.
    TypeError
        If a non-final step lacks ``transform``, or the final step lacks ``predict``.
    """

    def __init__(self, steps: list[tuple[str, object]]) -> None:
        """Initialise Pipeline; validates step names and interfaces."""
        if not steps:
            raise ValueError("Pipeline requires at least one step.")

        names = [name for name, _ in steps]
        if len(names) != len(set(names)):
            seen: set[str] = set()
            for n in names:
                if n in seen:
                    raise ValueError(f"Duplicate step name: {n!r}")
                seen.add(n)

        for i, (name, obj) in enumerate(steps[:-1]):
            if not (hasattr(obj, "partial_fit") and hasattr(obj, "transform")):
                raise TypeError(
                    f"Step {name!r} at index {i} must be a transformer "
                    f"(needs partial_fit and transform)."
                )

        final_name, final_obj = steps[-1]
        if not (hasattr(final_obj, "partial_fit") and hasattr(final_obj, "predict")):
            raise TypeError(
                f"Final step {final_name!r} must be an estimator "
                f"(needs partial_fit and predict)."
            )

        self._steps = list(steps)
        self._fitted = False

    # BaseEstimator

    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classes: np.ndarray | None = None,
    ) -> "Pipeline":
        """Fit each transformer then the final estimator on one chunk."""
        X = np.asarray(X, dtype=np.float64)
        for _, transformer in self._steps[:-1]:
            transformer.partial_fit(X, y)
            X = transformer.transform(X)
        final = self._steps[-1][1]
        if classes is not None:
            final.partial_fit(X, y, classes=classes)
        else:
            final.partial_fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Transform X through all steps, then call the final estimator's predict."""
        if not self._fitted:
            raise ValueError("Pipeline not fitted")
        X = self.transform(X)
        return self._steps[-1][1].predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Transform X through all steps, then call the final estimator's predict_proba."""
        if not self._fitted:
            raise ValueError("Pipeline not fitted")
        X = self.transform(X)
        return self._steps[-1][1].predict_proba(X)

    # Transformer interface

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply all transformer steps in order (does not include the final estimator)."""
        if not self._fitted:
            raise ValueError("Pipeline not fitted")
        X = np.asarray(X, dtype=np.float64)
        for _, transformer in self._steps[:-1]:
            X = transformer.transform(X)
        return X

    # Access helpers

    def __getitem__(self, name_or_index: Union[str, int]) -> object:
        """Return the object for a step by name (str) or position (int)."""
        if isinstance(name_or_index, int):
            return self._steps[name_or_index][1]
        for n, obj in self._steps:
            if n == name_or_index:
                return obj
        raise KeyError(f"No step named {name_or_index!r}")

    @property
    def named_steps(self) -> dict[str, object]:
        """Mapping of step name → step object."""
        return {n: obj for n, obj in self._steps}

    def __repr__(self) -> str:
        parts = ", ".join(f"({n!r}, {type(obj).__name__})" for n, obj in self._steps)
        return f"Pipeline([{parts}])"
