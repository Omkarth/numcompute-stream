"""StreamTrainer: prequential evaluation over a chunk iterator."""

from __future__ import annotations

import tracemalloc
from typing import Iterator

import numpy as np

from .base import BaseMetric
from .pipeline import Pipeline


class StreamTrainer:
    """Evaluate and train a Pipeline using prequential (test-then-train) evaluation.

    Parameters
    ----------
    pipeline : Pipeline
        A Pipeline instance (need not be fitted yet).
    metrics : dict of str → BaseMetric, optional
        Named metrics to track per chunk.
    log_memory : bool
        When True, record peak memory (bytes) from tracemalloc per chunk.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        metrics: dict[str, BaseMetric] | None = None,
        log_memory: bool = False,
    ) -> None:
        """Initialise StreamTrainer; see class docstring for parameter details."""
        self.pipeline = pipeline
        self.metrics: dict[str, BaseMetric] = metrics or {}
        self.log_memory = log_memory
        self._fitted = False

    def fit_stream(
        self,
        chunks_iterator: Iterator[tuple[np.ndarray, np.ndarray]],
    ) -> dict:
        """Run prequential evaluation over all chunks.

        For each chunk: predict (or skip if untrained), update metrics, then fit.

        Parameters
        ----------
        chunks_iterator : iterable of (X, y) pairs

        Returns
        -------
        dict with keys:
            ``per_chunk_<name>`` for each metric,
            ``memory_log`` (list of peak bytes, or empty),
            ``n_chunks`` (int),
            ``total_samples`` (int).
        """
        per_chunk: dict[str, list[float]] = {name: [] for name in self.metrics}
        memory_log: list[int] = []
        n_chunks = 0
        total_samples = 0

        for X, y in chunks_iterator:
            X = np.asarray(X, dtype=np.float64)
            y = np.asarray(y)
            n = len(y)
            n_chunks += 1
            total_samples += n

            if self.log_memory:
                tracemalloc.start()

            if self._fitted:
                y_pred = self.pipeline.predict(X)
                for name, metric in self.metrics.items():
                    metric.update(y, y_pred)
                    per_chunk[name].append(metric.result())
            else:
                y_pred = np.full(n, int(np.bincount(y.astype(int)).argmax())) if n > 0 else y
                for name, metric in self.metrics.items():
                    per_chunk[name].append(metric.result())

            self.pipeline.partial_fit(X, y)
            self._fitted = True

            if self.log_memory:
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                memory_log.append(peak)

        result: dict = {
            f"per_chunk_{name}": vals for name, vals in per_chunk.items()
        }
        result["memory_log"] = memory_log
        result["n_chunks"] = n_chunks
        result["total_samples"] = total_samples
        return result

    def fit_chunk(self, X: np.ndarray, y: np.ndarray) -> None:
        """Prequential single-chunk: predict (if fitted), update metrics, then fit.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,)
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        if self._fitted:
            y_pred = self.pipeline.predict(X)
            for metric in self.metrics.values():
                metric.update(y, y_pred)

        self.pipeline.partial_fit(X, y)
        self._fitted = True

    def score_chunk(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Score one chunk WITHOUT updating the pipeline or metric state.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,)

        Returns
        -------
        dict of metric_name → score value for this chunk only.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        y_pred = self.pipeline.predict(X)

        scores: dict[str, float] = {}
        for name, metric in self.metrics.items():
            from copy import deepcopy
            tmp = deepcopy(metric)
            tmp.update(y, y_pred)
            scores[name] = tmp.result()
        return scores

    def __repr__(self) -> str:
        return (
            f"StreamTrainer(pipeline={self.pipeline!r}, "
            f"metrics={list(self.metrics)}, log_memory={self.log_memory})"
        )
