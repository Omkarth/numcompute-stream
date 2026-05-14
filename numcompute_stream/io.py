"""I/O utilities: CSV loading, train/test split, chunk iteration."""

from __future__ import annotations

import csv
import pathlib
from collections.abc import Generator, Iterator

import numpy as np


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_csv(
    path: str | pathlib.Path,
    target_col: int = -1,
    delimiter: str = ",",
    skip_header: bool = True,
    dtype: np.dtype = np.float64,
) -> tuple[np.ndarray, np.ndarray]:
    """Load an entire CSV file into (X, y) arrays.

    Parameters
    ----------
    path       : path to the CSV file
    target_col : column index of the label (default -1 = last column);
                 negative indices are resolved after loading
    delimiter  : field delimiter (default ',')
    skip_header: whether to skip the first row
    dtype      : numpy dtype for the feature matrix X

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)

    Raises
    ------
    FileNotFoundError : if path does not exist
    ValueError        : if the file is empty or has inconsistent column counts
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    rows: list[list[str]] = []
    with path.open(newline="") as fh:
        reader = csv.reader(fh, delimiter=delimiter)
        if skip_header:
            next(reader, None)
        for row in reader:
            if row:
                rows.append(row)

    if not rows:
        raise ValueError(f"No data rows found in {path}")

    n_cols = len(rows[0])
    if any(len(r) != n_cols for r in rows):
        raise ValueError(f"Inconsistent column counts in {path}")

    data = np.array(rows, dtype=dtype)

    col = target_col if target_col >= 0 else data.shape[1] + target_col
    if not (0 <= col < data.shape[1]):
        raise ValueError(
            f"target_col {target_col} out of range for {data.shape[1]} columns"
        )

    feature_cols = list(range(data.shape[1]))
    feature_cols.pop(col)

    X = data[:, feature_cols]
    y = data[:, col]
    return X, y


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    shuffle: bool = True,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays into train and test sets.

    Parameters
    ----------
    X            : ndarray of shape (n_samples, n_features)
    y            : ndarray of shape (n_samples,)
    test_size    : fraction of samples for the test set (0 < test_size < 1)
    shuffle      : whether to shuffle before splitting
    random_state : seed for reproducibility

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have the same number of samples; "
            f"got {X.shape[0]} and {y.shape[0]}"
        )
    if not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size must be in (0, 1); got {test_size}")

    n = X.shape[0]
    indices = np.arange(n)

    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    split = int(np.floor(n * (1.0 - test_size)))
    train_idx, test_idx = indices[:split], indices[split:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# ---------------------------------------------------------------------------
# Streaming chunk iterator
# ---------------------------------------------------------------------------

def stream_chunks(
    source: str | pathlib.Path | tuple[np.ndarray, np.ndarray],
    chunk_size: int = 256,
    target_col: int = -1,
    delimiter: str = ",",
    skip_header: bool = True,
    dtype: np.dtype = np.float64,
    loop: bool = False,
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """Yield (X_chunk, y_chunk) pairs from a CSV file or in-memory arrays.

    Parameters
    ----------
    source     : CSV file path  **or**  a (X, y) tuple of pre-loaded arrays
    chunk_size : number of samples per chunk (last chunk may be smaller)
    target_col : label column index — only used when source is a file path
    delimiter  : CSV delimiter — only used when source is a file path
    skip_header: skip first row — only used when source is a file path
    dtype      : numpy dtype — only used when source is a file path
    loop       : if True, cycle over the data indefinitely

    Yields
    ------
    X_chunk : ndarray of shape (≤chunk_size, n_features)
    y_chunk : ndarray of shape (≤chunk_size,)

    Raises
    ------
    ValueError : if chunk_size < 1
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be ≥ 1; got {chunk_size}")

    if isinstance(source, (str, pathlib.Path)):
        X, y = load_csv(source, target_col=target_col, delimiter=delimiter,
                        skip_header=skip_header, dtype=dtype)
    else:
        X, y = source

    n = X.shape[0]
    while True:
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            yield X[start:end], y[start:end]
        if not loop:
            break


# ---------------------------------------------------------------------------
# Line-by-line streaming (memory-efficient, no full load)
# ---------------------------------------------------------------------------

def stream_csv_rows(
    path: str | pathlib.Path,
    target_col: int = -1,
    delimiter: str = ",",
    skip_header: bool = True,
    dtype: np.dtype = np.float64,
) -> Iterator[tuple[np.ndarray, float]]:
    """Yield one (x_row, y_scalar) at a time from a CSV file.

    Useful when the full dataset does not fit in memory.  For most uses,
    prefer `stream_chunks` which batches rows for vectorised processing.

    Yields
    ------
    x : ndarray of shape (n_features,)
    y : scalar label
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with path.open(newline="") as fh:
        reader = csv.reader(fh, delimiter=delimiter)
        if skip_header:
            header = next(reader, None)
            if header is None:
                return
            n_cols = len(header)
            col = target_col if target_col >= 0 else n_cols + target_col
        else:
            first = next(reader, None)
            if first is None:
                return
            n_cols = len(first)
            col = target_col if target_col >= 0 else n_cols + target_col
            row = np.array(first, dtype=dtype)
            feature_cols = [i for i in range(n_cols) if i != col]
            yield row[feature_cols], row[col]

        feature_cols = [i for i in range(n_cols) if i != col]
        for raw in reader:
            if not raw:
                continue
            row = np.array(raw, dtype=dtype)
            yield row[feature_cols], row[col]
