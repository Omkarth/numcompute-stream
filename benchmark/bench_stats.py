"""Benchmark: RunningStats Welford update — loop-based vs NumPy vectorised.

Measures wall-clock time for processing a 100 000-sample chunk into a
single-feature RunningStats accumulator.

Loop baseline:  scalar Welford update, one sample at a time.
Vectorised:     Chan's parallel batch update (the actual implementation).
"""

import time

import numpy as np

from numcompute_stream.stats import RunningStats

# ── Configuration ──────────────────────────────────────────────────────────────

N_SAMPLES = 100_000
N_REPEATS = 5
RNG = np.random.default_rng(0)


# ── Loop-based implementation ──────────────────────────────────────────────────

def _welford_loop(X: np.ndarray) -> tuple[float, float, int]:
    """Scalar Welford update, one sample at a time.  Returns (mean, M2, count)."""
    mean = 0.0
    M2 = 0.0
    count = 0
    for x in X:
        count += 1
        delta = x - mean
        mean += delta / count
        delta2 = x - mean
        M2 += delta * delta2
    return mean, M2, count


def run_loop(data: np.ndarray) -> tuple[float, float]:
    mean, M2, _ = _welford_loop(data)
    return mean, M2


# ── Vectorised implementation ──────────────────────────────────────────────────

def run_vectorised(data: np.ndarray) -> tuple[float, float]:
    rs = RunningStats()
    rs.update(data)
    return float(rs.mean[0]), float(rs._M2[0])


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
    data = RNG.standard_normal(N_SAMPLES)

    # Correctness check
    loop_mean, loop_M2 = run_loop(data)
    vec_mean, vec_M2 = run_vectorised(data)
    assert np.isclose(loop_mean, vec_mean, rtol=1e-10), "mean mismatch"
    assert np.isclose(loop_M2, vec_M2, rtol=1e-10), "M2 mismatch"

    t_loop = _min_ms(run_loop, data)
    t_vec = _min_ms(run_vectorised, data)
    speedup = t_loop / t_vec

    header = f"{'Implementation':<22} {'Time (ms)':>12} {'Speedup':>10}"
    sep = "-" * len(header)
    print(f"\nbench_stats — RunningStats Welford update ({N_SAMPLES:,} samples, min of {N_REPEATS} runs)\n")
    print(header)
    print(sep)
    print(f"{'loop (scalar Welford)':<22} {t_loop:>12.3f} {'1.00×':>10}")
    print(f"{'vectorised (Chan)':<22} {t_vec:>12.3f} {speedup:>9.2f}×")
    print(sep)


if __name__ == "__main__":
    main()
