"""Shape validators and NaN helpers shared across the package."""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Array validation
# ---------------------------------------------------------------------------

def check_array(
    X: np.ndarray,
    *,
    ndim: int = 2,
    min_samples: int = 1,
    min_features: int = 1,
    allow_nan: bool = False,
    dtype: type | np.dtype | None = None,
    name: str = "X",
) -> np.ndarray:
    """Validate a NumPy array and return it (possibly cast).

    Parameters
    ----------
    X            : array to validate
    ndim         : expected number of dimensions (default 2)
    min_samples  : minimum number of rows required
    min_features : minimum number of columns required (only checked for ndim=2)
    allow_nan    : if False, raises ValueError when NaN values are present
    dtype        : if given, cast X to this dtype
    name         : name used in error messages

    Returns
    -------
    X : validated (and possibly cast) array

    Raises
    ------
    TypeError  : if X is not an ndarray
    ValueError : if shape constraints or NaN constraint are violated
    """
    if not isinstance(X, np.ndarray):
        raise TypeError(f"{name} must be a numpy ndarray; got {type(X).__name__}")

    if dtype is not None:
        X = X.astype(dtype)

    if X.ndim != ndim:
        raise ValueError(
            f"{name} must be {ndim}-D; got shape {X.shape}"
        )

    if X.shape[0] < min_samples:
        raise ValueError(
            f"{name} must have at least {min_samples} sample(s); "
            f"got {X.shape[0]}"
        )

    if ndim == 2 and X.shape[1] < min_features:
        raise ValueError(
            f"{name} must have at least {min_features} feature(s); "
            f"got {X.shape[1]}"
        )

    if not allow_nan and np.any(np.isnan(X)):
        raise ValueError(f"{name} contains NaN values")

    return X


def check_X_y(
    X: np.ndarray,
    y: np.ndarray,
    *,
    allow_nan: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate that X and y are compatible 2-D / 1-D arrays.

    Parameters
    ----------
    X         : feature matrix of shape (n_samples, n_features)
    y         : label vector of shape (n_samples,)
    allow_nan : passed through to check_array for X

    Returns
    -------
    X, y : validated arrays

    Raises
    ------
    ValueError : if shapes are incompatible
    """
    X = check_array(X, ndim=2, allow_nan=allow_nan)
    y = check_array(y, ndim=1, name="y", allow_nan=False)

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have the same number of samples; "
            f"got X.shape={X.shape}, y.shape={y.shape}"
        )

    return X, y


# ---------------------------------------------------------------------------
# NaN helpers
# ---------------------------------------------------------------------------

def any_nan(X: np.ndarray) -> bool:
    """Return True if X contains at least one NaN."""
    return bool(np.any(np.isnan(X)))


def nan_mask(X: np.ndarray) -> np.ndarray:
    """Return a boolean array of the same shape as X; True where NaN."""
    return np.isnan(X)


def nan_col_counts(X: np.ndarray) -> np.ndarray:
    """Return number of NaN values per column.

    Parameters
    ----------
    X : 2-D array of shape (n_samples, n_features)

    Returns
    -------
    counts : 1-D int array of shape (n_features,)
    """
    if X.ndim != 2:
        raise ValueError(f"nan_col_counts expects a 2-D array; got shape {X.shape}")
    return np.sum(np.isnan(X), axis=0)


def nan_row_mask(X: np.ndarray) -> np.ndarray:
    """Return a 1-D boolean mask; True for rows that contain any NaN.

    Parameters
    ----------
    X : 2-D array of shape (n_samples, n_features)

    Returns
    -------
    mask : bool array of shape (n_samples,)
    """
    if X.ndim != 2:
        raise ValueError(f"nan_row_mask expects a 2-D array; got shape {X.shape}")
    return np.any(np.isnan(X), axis=1)


def safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
    fill: float = 0.0,
) -> np.ndarray:
    """Element-wise division; replaces divide-by-zero results with `fill`.

    Parameters
    ----------
    numerator   : array
    denominator : array, same shape as numerator
    fill        : value to use when denominator is zero (default 0.0)

    Returns
    -------
    result : array of same shape
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(
            denominator == 0,
            fill,
            numerator / np.where(denominator == 0, 1, denominator),
        )
    return result
