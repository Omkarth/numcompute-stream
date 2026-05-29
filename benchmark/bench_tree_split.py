"""Benchmark: Info-gain over candidate thresholds — loop-based vs vectorised.

Measures wall-clock time to evaluate info-gain for 30 candidate thresholds at
a leaf node with 1 000 samples spread across 3 classes.

Loop baseline:  Python for-loop over each threshold; computes child impurity
                with scalar _gini / _entropy for each threshold individually.
Vectorised:     _info_gains_for_feature from tree.py, which processes all
                thresholds in a single matrix operation.
"""

import math
import time

import numpy as np

from numcompute_stream.stats import RunningStats
from numcompute_stream.tree import (
    LeafStats,
    _gini,
    _gini_vec,
    _info_gains_for_feature,
    _normal_cdf,
)

# ── Configuration ──────────────────────────────────────────────────────────────

N_SAMPLES = 1_000
N_CLASSES = 3
N_THRESHOLDS = 30
N_REPEATS = 5
RNG = np.random.default_rng(2)


# ── Build a realistic LeafStats fixture ───────────────────────────────────────

def _make_leaf_stats() -> tuple[LeafStats, np.ndarray, np.ndarray, float, np.ndarray]:
    """Return (leaf_stats, classes, parent_counts, parent_impurity, thresholds)."""
    classes = np.arange(N_CLASSES, dtype=np.int64)
    ls = LeafStats(n_features=1, grace_period=1)

    for label in range(N_CLASSES):
        n = N_SAMPLES // N_CLASSES
        vals = RNG.normal(loc=float(label), scale=0.5, size=n)
        for v in vals:
            ls.update(np.array([v]), label)

    parent_counts = ls.class_counts_array(classes)
    parent_impurity = _gini(parent_counts)
    thresholds = np.linspace(-1.0, 3.0, N_THRESHOLDS)
    return ls, classes, parent_counts, parent_impurity, thresholds


# ── Loop-based implementation ──────────────────────────────────────────────────

def _gini_scalar(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total == 0.0:
        return 0.0
    p = counts / total
    return float(1.0 - (p ** 2).sum())


def info_gains_loop(
    leaf_stats: LeafStats,
    feat_idx: int,
    thresholds: np.ndarray,
    classes: np.ndarray,
    parent_counts: np.ndarray,
    parent_impurity: float,
) -> np.ndarray:
    n_classes = len(classes)
    mus = np.zeros(n_classes)
    sigmas = np.full(n_classes, 1e10)
    for i, c in enumerate(classes):
        rs = leaf_stats.get_feature_stats(feat_idx, int(c))
        if rs is not None and rs.count > 0:
            mus[i] = float(rs.mean[0])
            s = float(rs.std[0])
            sigmas[i] = max(s, 1e-10)

    total = float(parent_counts.sum())
    gains = np.empty(len(thresholds))

    for t_idx, thresh in enumerate(thresholds):
        n_left = np.empty(n_classes)
        for i in range(n_classes):
            z = (thresh - mus[i]) / sigmas[i]
            phi = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
            n_left[i] = phi * parent_counts[i]
        n_right = parent_counts - n_left

        nl = n_left.sum()
        nr = n_right.sum()
        if nl <= 0 or nr <= 0:
            gains[t_idx] = 0.0
            continue

        imp_l = _gini_scalar(n_left)
        imp_r = _gini_scalar(n_right)
        gains[t_idx] = parent_impurity - (nl / total) * imp_l - (nr / total) * imp_r

    return gains


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
    ls, classes, parent_counts, parent_impurity, thresholds = _make_leaf_stats()
    feat_idx = 0

    # Correctness check
    gains_loop = info_gains_loop(
        ls, feat_idx, thresholds, classes, parent_counts, parent_impurity
    )
    gains_vec = _info_gains_for_feature(
        ls, feat_idx, thresholds, classes, parent_counts, parent_impurity, "gini"
    )
    assert np.allclose(gains_loop, gains_vec, rtol=1e-8, atol=1e-10), (
        f"output mismatch\nloop: {gains_loop}\nvec:  {gains_vec}"
    )

    t_loop = _min_ms(
        info_gains_loop,
        ls, feat_idx, thresholds, classes, parent_counts, parent_impurity,
    )
    t_vec = _min_ms(
        _info_gains_for_feature,
        ls, feat_idx, thresholds, classes, parent_counts, parent_impurity, "gini",
    )
    speedup = t_loop / t_vec

    desc = f"{N_THRESHOLDS} thresholds, {N_CLASSES} classes, {N_SAMPLES} samples"
    header = f"{'Implementation':<30} {'Time (ms)':>12} {'Speedup':>10}"
    sep = "-" * len(header)
    print(f"\nbench_tree_split — info-gain ({desc}, min of {N_REPEATS} runs)\n")
    print(header)
    print(sep)
    print(f"{'loop (per-threshold)':<30} {t_loop:>12.4f} {'1.00×':>10}")
    print(f"{'vectorised (_info_gains_for_feature)':<30} {t_vec:>12.4f} {speedup:>9.2f}×")
    print(sep)


if __name__ == "__main__":
    main()
