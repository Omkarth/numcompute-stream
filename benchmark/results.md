# Benchmark Results

All timings are the **minimum** wall-clock time over 5 runs (`time.perf_counter`).
Machine: Apple Silicon (darwin 25.5.0), Python 3.14, NumPy 2.x.

---

## 1. RunningStats Welford update (`bench_stats.py`)

**Setup:** 100 000-sample 1-D chunk fed into a fresh accumulator.

| Implementation       | Time (ms) | Speedup |
|----------------------|----------:|--------:|
| loop (scalar Welford) | 15.363   | 1.00×   |
| vectorised (Chan)     |  0.059   | 260.76× |

Chan's parallel update computes the batch mean and M2 with `X.mean(axis=0)` and
`((X − batch_mean) ** 2).sum(axis=0)` — both BLAS/LAPACK-routed calls that operate
on contiguous memory.  The scalar loop pays a Python interpreter dispatch cost for
every sample and cannot exploit SIMD or cache prefetching across elements.

---

## 2. StandardScaler transform (`bench_scaler.py`)

**Setup:** 100 000 × 8 matrix; mean and std pre-fitted; transform only.

| Implementation          | Time (ms) | Speedup |
|-------------------------|----------:|--------:|
| loop (row-by-row)       | 44.260    | 1.00×   |
| vectorised (broadcast)  |  0.803    | 55.13×  |

Broadcasting `(X − mean) / std` materialises a single subtraction and a single
division over the full matrix in C, keeping all 800 000 doubles in L2/L3 cache
through one linear scan.  The row loop forces 100 000 Python frame invocations
and breaks spatial locality by striding across columns on each iteration.

---

## 3. Info-gain over candidate thresholds (`bench_tree_split.py`)

**Setup:** 30 candidate thresholds, 3 classes, 1 000 samples per leaf.

| Implementation                      | Time (ms) | Speedup |
|-------------------------------------|----------:|--------:|
| loop (per-threshold)                | 0.1475    | 1.00×   |
| vectorised (`_info_gains_for_feature`) | 0.0268 | 5.50×   |

`_info_gains_for_feature` stacks all thresholds into a `(n_thresh, n_classes)`
matrix, computes the normal CDF and child impurities in one vectorised pass, and
masks degenerate splits without any Python branching.  The loop version calls
`math.erf` and `_gini_scalar` separately for each threshold, incurring function-call
overhead proportional to `n_thresh`.  The speedup is lower than in the other two
benchmarks because `n_thresh` (30) is small; it would grow super-linearly with more
thresholds.
