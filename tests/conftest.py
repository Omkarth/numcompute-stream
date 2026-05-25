"""Shared pytest fixtures — synthetic data generators reused across all test modules."""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pytest


# RNG

@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


# Small in-memory arrays

@pytest.fixture
def small_binary(rng) -> tuple[np.ndarray, np.ndarray]:
    """100-sample, 4-feature binary classification dataset."""
    X = rng.standard_normal((100, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(float)
    return X, y


@pytest.fixture
def small_multiclass(rng) -> tuple[np.ndarray, np.ndarray]:
    """150-sample, 4-feature 3-class dataset (like iris)."""
    X = rng.standard_normal((150, 4))
    y = np.array([0] * 50 + [1] * 50 + [2] * 50, dtype=float)
    return X, y


@pytest.fixture
def single_class(rng) -> tuple[np.ndarray, np.ndarray]:
    """Edge case: all samples belong to class 0."""
    X = rng.standard_normal((30, 3))
    y = np.zeros(30)
    return X, y


@pytest.fixture
def chunk_stream(small_binary) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pre-split small_binary into 10-sample chunks."""
    X, y = small_binary
    chunks = []
    for start in range(0, len(y), 10):
        chunks.append((X[start:start + 10], y[start:start + 10]))
    return chunks


# Temporary CSV files

@pytest.fixture
def tmp_csv(tmp_path, rng) -> pathlib.Path:
    """A small CSV with header; last column is the binary label."""
    X = rng.standard_normal((60, 3))
    y = (X[:, 0] > 0).astype(float).reshape(-1, 1)
    data = np.hstack([X, y])
    p = tmp_path / "sample.csv"
    header = "feat0,feat1,feat2,label"
    np.savetxt(str(p), data, delimiter=",", header=header, comments="")
    return p


@pytest.fixture
def tmp_csv_no_header(tmp_path, rng) -> pathlib.Path:
    """Same layout but no header row."""
    X = rng.standard_normal((40, 2))
    y = (X[:, 0] > 0).astype(float).reshape(-1, 1)
    data = np.hstack([X, y])
    p = tmp_path / "no_header.csv"
    np.savetxt(str(p), data, delimiter=",")
    return p


@pytest.fixture
def tmp_csv_with_nans(tmp_path) -> pathlib.Path:
    """CSV that contains NaN values in feature columns."""
    lines = [
        "a,b,label",
        "1.0,2.0,0",
        ",3.0,1",       # NaN in col a
        "2.0,,0",       # NaN in col b
        "3.0,4.0,1",
    ]
    p = tmp_path / "nans.csv"
    p.write_text("\n".join(lines))
    return p


# Synthetic concept-drift stream

def make_drift_stream(
    n_pre: int = 200,
    n_post: int = 200,
    n_features: int = 4,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-phase stream where the decision boundary flips after n_pre samples.

    The label is determined by the sign of feature 0 pre-drift, and the sign of
    feature 1 post-drift — a clean concept drift.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_pre + n_post, n_features))
    y = np.empty(n_pre + n_post)
    y[:n_pre] = (X[:n_pre, 0] > 0).astype(float)
    y[n_pre:] = (X[n_pre:, 1] > 0).astype(float)
    return X, y


@pytest.fixture
def drift_stream() -> tuple[np.ndarray, np.ndarray]:
    return make_drift_stream()
