"""Benchmark: StandardScaler transform — loop-based vs NumPy broadcasting.

Measures wall-clock time to transform a 100 000 × 8 matrix using pre-fitted
mean and std vectors.

Loop baseline:  Python for-loop over each row, applying (x - mean) / std.
Vectorised:     NumPy broadcasting over the whole matrix at once.
"""

import time

import numpy as np

from numcompute_stream.preprocessing import StandardScaler

# ── Configuration ──────────────────────────────────────────────────────────────

N_SAMPLES = 100_000
N_FEATURES = 8
N_REPEATS = 5
RNG = np.random.default_rng(1)


# ── Loop-based implementation ──────────────────────────────────────────────────

def transform_loop(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    out = np.empty_like(X)
    for i in range(X.shape[0]):
        out[i] = (X[i] - mean) / std
    return out


# ── Vectorised implementation ──────────────────────────────────────────────────

def transform_vectorised(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(X)


# ── Timing helper ──────────────────────────────────────────────────────────────

def _min_ms(fn, *args) -> float:
    best = float("inf")
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best * 1_000


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    X = RNG.standard_normal((N_SAMPLES, N_FEATURES))
    X_ref = X * 3.0 + 2.0  # non-trivial mean/std

    scaler = StandardScaler()
    scaler.partial_fit(X_ref)

    mean = scaler.mean_
    std = scaler.scale_
    # Replace zero std with 1 to match scaler's safe_std logic
    safe_std = np.where(std == 0.0, 1.0, std)

    # Correctness check
    out_loop = transform_loop(X, mean, safe_std)
    out_vec = transform_vectorised(X, scaler)
    assert np.allclose(out_loop, out_vec, rtol=1e-10), "output mismatch"

    t_loop = _min_ms(transform_loop, X, mean, safe_std)
    t_vec = _min_ms(transform_vectorised, X, scaler)
    speedup = t_loop / t_vec

    shape_str = f"{N_SAMPLES:,} × {N_FEATURES}"
    header = f"{'Implementation':<26} {'Time (ms)':>12} {'Speedup':>10}"
    sep = "-" * len(header)
    print(f"\nbench_scaler — StandardScaler transform ({shape_str}, min of {N_REPEATS} runs)\n")
    print(header)
    print(sep)
    print(f"{'loop (row-by-row)':<26} {t_loop:>12.3f} {'1.00×':>10}")
    print(f"{'vectorised (broadcast)':<26} {t_vec:>12.3f} {speedup:>9.2f}×")
    print(sep)


if __name__ == "__main__":
    main()
