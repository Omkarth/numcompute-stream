# numcompute-stream: Technical Report

University of Adelaide — NumCompute Assignment 2.2

---

## 1. Design Decisions

### 1.1 Hoeffding Tree (VFDT)

`HoeffdingTreeClassifier` implements the Very Fast Decision Tree algorithm introduced by
Domingos & Hulten (2000).  The central insight of VFDT is that a streaming learner never
needs to store raw samples: each leaf maintains only *sufficient statistics* that can be
updated in O(1) per sample and from which the best split can be evaluated at any time.

Our implementation stores, per leaf, a class-count vector and a `RunningStats` object for
each (feature, class) pair.  `RunningStats` uses Welford's one-pass algorithm to accumulate
the mean and sum of squared deviations without materialising the sample array.  When split
candidates are evaluated, per-class Gaussian parameters (mean and variance) are read from
these accumulators and fed into the vectorised `_info_gains_for_feature` function, which
computes information gain across all candidate thresholds in a single NumPy pass.

Split commitment is governed by the Hoeffding bound: after observing *n* samples at a leaf,
a split on feature *A* is committed when the gap between the best and second-best impurity
gain exceeds ε = √(R² ln(1/δ) / 2n).  The range parameter R depends on the criterion: for
Gini, R = 1 (impurity is bounded in [0, 1]); for entropy, R = log₂(K) where K is the
number of classes, because entropy is bounded in [0, log₂(K)].  We use δ = 1 × 10⁻⁷ as the
default split confidence.  A tie-breaking threshold τ (default 0.05) allows early splitting
when the bound is already tight and the two top candidates are nearly indistinguishable —
this prevents a leaf from hoarding samples when additional data are unlikely to change the
outcome.

The tree grows strictly top-down: internal nodes store only the split feature, threshold, and
child pointers; all statistics live in leaves.  After a split, the two fresh child leaves
inherit their parent's class distribution as a prior so that the first prediction at a new
leaf is not arbitrary.

### 1.2 Online Random Forest

`OnlineRandomForestClassifier` builds an ensemble of `HoeffdingTreeClassifier` instances
following the online bagging scheme of Oza & Russell (2001).  For each incoming sample, each
tree receives a Poisson(λ = 1) weight drawn independently.  Samples with weight zero are
never passed to `partial_fit`; instead they are used as out-of-bag (OOB) observations for
that tree, tracking its accuracy without any separate hold-out set.  Since E[Poisson(1)] = 1,
each sample contributes on average exactly once per tree — matching the expectation of batch
bootstrap — but the scheme is fully online: no sample buffer is required.

Diversity is introduced in two ways.  First, each tree is assigned a random feature subset of
size `max_features` at construction time, sampled without replacement from the full feature
set.  All subsequent lookups for that tree are remapped through this subset.  Second, the
Poisson weights ensure that the trees see different replications of each chunk.

Prediction aggregates trees by hard-vote (default) or probability-averaged soft-vote,
controlled by the `voting` parameter.  OOB accuracy is exposed per tree via the `oob_scores_`
property after the first chunk.

### 1.3 ADWIN + Adaptive Random Forest

The Adaptive Random Forest (ARF) variant of `OnlineRandomForestClassifier` adds concept-drift
detection following Gomes et al. (2017).  Each tree maintains two independent `ADWIN`
detectors, both fed the binary OOB correctness signal (1 for correct, 0 for incorrect).

The *warning* detector uses `warning_delta = 0.01`, a looser threshold that fires on early
signs of degradation.  When it fires, a background candidate tree is instantiated and begins
fitting in parallel on all subsequent samples, but does not yet contribute to the ensemble
vote.  The *drift* detector uses `drift_delta = 0.001`, a stricter threshold requiring
stronger statistical evidence.  When the drift detector fires, the original tree is replaced:
if a candidate exists, it becomes the new tree; otherwise a fresh tree is created.  This
two-threshold state machine — *stable → warning → drift* — was proposed in Gomes et al.
(2017) to prevent unnecessary resets triggered by transient noise.

Both detectors enforce a minimum window of 100 OOB samples before evaluating any cut point.
This guard prevents spurious drift detection on the leading edge of a stream, where the OOB
window is too short for the Hoeffding criterion to be meaningful.

`ADWIN` itself (Bifet & Gavaldà, 2007) maintains a sliding window of bounded size
(`max_window_size = 5000`), tests all candidate cut points for a significant shift in the
window mean, and drops the older sub-window if drift is confirmed.  Our implementation uses a
plain Python deque rather than exponential histograms, giving O(W²) worst-case update cost
(discussed further in Section 4.3).

### 1.4 NumPy-only Compliance

The assignment prohibits pandas and scipy.  Two places required non-trivial workarounds.

CSV ingestion uses the standard `csv` module rather than `pandas.read_csv`.  The `load_csv`
function reads each row as a list of strings, converts columns to `float64` with explicit
`np.nan` propagation for missing cells, and concatenates into a NumPy array.  This is
slightly more verbose than a one-liner pandas call, but is entirely dependency-free.

The normal CDF required for Gaussian split evaluation uses `math.erf` wrapped via
`np.frompyfunc(_math.erf, 1, 1)` to create a ufunc that accepts NumPy arrays without a
scalar Python loop in calling code:

```python
_verf = np.frompyfunc(_math.erf, 1, 1)

def _normal_cdf(z):
    return 0.5 * (1.0 + _verf(z / math.sqrt(2.0))).astype(np.float64)
```

This replaces `scipy.special.erf` while still vectorising over threshold arrays.  Similarly,
majority-class prediction uses `np.argmax` on the class-count vector rather than
`scipy.stats.mode`, and streaming variance uses Welford accumulators rather than
`scipy.stats` descriptive functions.

### 1.5 Uniform Streaming Interface

All components share a common protocol enforced by abstract base classes in `base.py`:

- `BaseTransformer` requires `partial_fit(X)`, `transform(X)`, and `reset()`.
- `BaseEstimator` requires `partial_fit(X, y)`, `predict(X)`, and `reset()`.
- `BaseMetric` requires `update(y_true, y_pred)`, `result()`, and `reset()`.
- `BaseDriftDetector` requires `update(value: float) -> bool` and `reset()`.

This uniformity makes `Pipeline` trivial to implement: it simply iterates transformers in
order calling `partial_fit` then `transform`, then delegates to the estimator.  `StreamTrainer`
can drive any `Pipeline` through any chunk iterator without knowing which models or
preprocessors are inside.  New components can be dropped in without changing the evaluation
harness.

---

## 2. Testing and Edge Cases

### 2.1 Per-module Test Coverage

The full test suite consists of 267 tests across 10 test files.  Coverage was measured with
`pytest --cov=numcompute_stream`.

| Module | Stmts | Miss | Coverage | Tests |
|---|---:|---:|---:|---:|
| `__init__.py` | 10 | 0 | 100% | — |
| `base.py` | 35 | 2 | 94% | — |
| `drift.py` | 66 | 5 | 92% | 12 |
| `ensemble.py` | 210 | 16 | 92% | 33 |
| `io.py` | 83 | 12 | 86% | 20 |
| `metrics.py` | 201 | 12 | 94% | 41 |
| `pipeline.py` | 64 | 8 | 88% | 15 |
| `preprocessing.py` | 204 | 25 | 88% | 28 |
| `stats.py` | 163 | 16 | 90% | 27 |
| `stream.py` | 66 | 1 | 98% | — |
| `tree.py` | 301 | 21 | 93% | 51 |
| `utils.py` | 38 | 0 | 100% | 23 |
| `visualise.py` | 82 | 0 | 100% | 17 |
| **TOTAL** | **1 523** | **118** | **92%** | **267** |

The uncovered lines are predominantly error branches in `io.py` and argument-validation paths
in `preprocessing.py` that are reachable only through specific file-encoding edge cases; the
core algorithmic paths are fully exercised.

### 2.2 Categories of Edge Cases

Tests are grouped into five categories that reflect the failure modes specific to streaming
learning.

**Numerical stability.** `RunningStats` is tested against `np.var` on normally distributed,
constant, and single-sample sequences.  A constant stream must produce variance exactly zero;
a single sample must produce biased variance zero and raise on unbiased variance (denominator
would be zero).  `StreamingHistogram` quantile estimates are tested against `np.quantile` on
random data to confirm they stay within a tolerance of the true quantile.

**Missing data.** `Imputer` is tested with columns that are entirely `np.nan` on the first
chunk, which must not raise and must produce a valid imputed output once sufficient data
arrive.  The Hoeffding Tree skips NaN values when updating per-feature statistics but still
counts the sample in the class distribution; NaN-containing rows at predict time are routed
left by convention.

**Shape mismatches.** `check_X_y` in `utils.py` is tested with mismatched row counts, 1-D
arrays, and zero-row inputs.  All transformers and estimators are tested to raise
`ValueError` on the first call if the feature count does not match what was fitted.

**Streaming semantics.** `StandardScaler` is tested to confirm that statistics accumulate
correctly across multiple `partial_fit` calls and that `transform` uses the statistics from
all calls seen so far, not just the most recent chunk.  `Accuracy` and `RollingMetric` are
tested to confirm `reset()` returns the object to its initial state.

**Drift detection.** `ADWIN` is tested with a hand-crafted step-function stream (ones
followed by zeros) where drift must be detected, and with a stationary stream where it must
not fire.  `OnlineRandomForestClassifier` is tested to confirm that drift events increment
`drift_event_count_` and that tree replacement produces a fresh tree with no split history.

### 2.3 Mathematical Correctness Tests

Four tests verify the underlying mathematics directly rather than end-to-end behaviour.

`test_var_matches_numpy` feeds identical data to `RunningStats.partial_fit` and `np.var`,
then asserts agreement to 10 decimal places.  This guards against off-by-one errors in the
Welford M2 accumulator (using `n−1` instead of `n` for the correction term, for example).

`test_hoeffding_bound_gini` and `test_hoeffding_bound_entropy` assert the closed-form result
of `_hoeffding_bound` for known (n, δ) pairs: for `criterion='gini'` the formula reduces to
√(ln(1/δ) / (2n)), and for `criterion='entropy'` R is substituted as log₂(K).  This test
caught a real bug during development where R was incorrectly set to 1 for both criteria,
causing entropy-based trees to split too aggressively.

`test_histogram_quantile_approx` asserts that `StreamingHistogram.quantile(q)` stays within
0.05 of `np.quantile(data, q)` on a 10 000-sample uniform random array, confirming that the
CDF interpolation is directionally correct even with a finite bin count.

`test_binary_tree_node_identity_after_split` asserts that after any sequence of splits,
`clf.n_nodes == 2 * clf.n_leaves − 1`.  This is the structural invariant of a full binary
tree (every internal node has exactly two children) and will fail if a split accidentally
creates an orphaned node or fails to remove a leaf from the leaf registry.

### 2.4 Test-driven Development Reflections

Two bugs were caught purely by tests before the buggy code reached any end-to-end run.

The first was the Hoeffding bound R-parameter bug noted above.  When `criterion='entropy'` was
added as an option, the `_hoeffding_bound` function was initially not updated to substitute
R = log₂(K).  The unit test for the entropy case failed immediately with a value that was too
small, pointing directly to the missing substitution.  Without this test, the bug would have
been invisible at the integration level: the tree would still produce predictions, just with
splits committed earlier than the statistical guarantee warranted.

The second was a Welford accumulator reset bug: after calling `reset()`, the internal `_n`
counter was reset but `_mean` and `_M2` were not.  A test that fits data, resets, then fits
different data and checks the variance caught this immediately.  Because `RunningStats` is
used inside every leaf of every tree in the forest, this bug would have corrupted per-leaf
statistics silently.

---

## 3. Benchmark Results

### 3.1 Methodology

All benchmarks use `time.perf_counter` and report the minimum wall-clock time over five runs.
Each run operates on identical pre-generated input data.  The minimum-of-five convention is
standard practice for microbenchmarks: it excludes OS scheduling jitter and cache-cold first
runs, leaving a time that is representative of steady-state execution.  Benchmarks were run
on an Apple Silicon machine (darwin 25.5.0, Python 3.14.3, NumPy 2.x).

Loop baselines are written in idiomatic Python with no deliberate pessimisation: scalar
Welford uses the textbook recurrence; row-by-row scaling uses a simple `for row in X` loop;
per-threshold info-gain uses `math.erf` directly.  The vectorised implementations are the
actual production code paths from `numcompute_stream`.

### 3.2 RunningStats Welford Update

**Setup:** 100 000-sample one-dimensional chunk fed into a fresh `RunningStats` accumulator.

| Implementation | Time (ms) | Speedup |
|---|---:|---:|
| loop (scalar Welford) | 15.96 | 1.00× |
| vectorised (Chan parallel merge) | 0.062 | 258.98× |

The vectorised path uses Chan's parallel merge formula: the batch mean is computed as
`X.mean(axis=0)` and the batch M2 as `((X − batch_mean) ** 2).sum(axis=0)`, then these are
merged into the running accumulator with the parallel-combination update.  Both operations are
BLAS/LAPACK-routed calls that operate on contiguous `float64` memory and exploit SIMD
instructions.  The scalar loop pays a Python interpreter dispatch cost for every one of the
100 000 samples and cannot exploit SIMD or cache prefetching across elements.  At 258.98×
speedup, this is the single largest win in the codebase.

### 3.3 StandardScaler Transform

**Setup:** 100 000 × 8 matrix; mean and standard deviation pre-fitted; transform only.

| Implementation | Time (ms) | Speedup |
|---|---:|---:|
| loop (row-by-row) | 44.84 | 1.00× |
| vectorised (broadcast) | 0.833 | 53.80× |

The vectorised implementation computes `(X − mean) / std` as a single broadcasting operation
over the full matrix.  NumPy lays out the 100 000 × 8 array in row-major (C) order, so the
subtraction and division traverse memory linearly, keeping 800 000 `float64` values in L2/L3
cache through one linear scan.  The row loop forces 100 000 Python frame invocations and
breaks spatial locality: each iteration accesses a non-contiguous stride across columns before
moving to the next row.  The 53.80× speedup is lower than the Welford case because the
operation touches more memory (a matrix rather than a vector), but is still dominated by the
elimination of Python dispatch overhead.

### 3.4 Info-Gain Across Candidate Thresholds

**Setup:** 30 candidate thresholds, 3 classes, 1 000 samples per leaf.

| Implementation | Time (ms) | Speedup |
|---|---:|---:|
| loop (per-threshold) | 0.1436 | 1.00× |
| vectorised (`_info_gains_for_feature`) | 0.0262 | 5.49× |

`_info_gains_for_feature` stacks all threshold-conditional class counts into a
`(n_thresh, n_classes)` matrix, computes the Gaussian CDF and child impurities in one
vectorised pass, and masks degenerate splits without any Python branching.  The loop version
calls `math.erf` (via `_verf`) and evaluates `_gini_scalar` separately for each of the 30
thresholds, incurring function-call overhead proportional to `n_thresh`.

The 5.49× speedup is substantially lower than the other two benchmarks.  This is explained by
the operation's algorithmic cost: `_normal_cdf` is not a simple arithmetic expression —
`math.erf` is itself a transcendental function computed by a minimax polynomial approximation.
When the per-element operation is expensive, Python dispatch overhead becomes a smaller
fraction of total time and the relative advantage of vectorisation shrinks.

### 3.5 Summary

The three benchmarks illustrate a consistent pattern: vectorisation matters most when the
inner per-element operation is simple arithmetic (258.98× for running mean/variance, 53.80×
for a subtract-and-divide).  Once the per-element operation is algorithmically expensive —
such as evaluating a transcendental function in the CDF case — the dispatch overhead is
already a small fraction of total cost, and the speedup is more modest (5.49×).  The practical
implication for streaming-learning code is that the highest-frequency code paths
(per-sample statistics updates, per-chunk scaling) benefit disproportionately from vectorised
formulations, while lower-frequency paths (split evaluation, which runs at most once per leaf)
yield smaller but still meaningful improvements.

---

## 4. Reflections

### 4.1 What Went Well

Committing to a true VFDT from the outset — rather than a mini-batch approximation that
accumulates samples until a batch is large enough — proved to be the right call.  The
sufficient-statistics design (Welford accumulators per leaf, no stored samples) kept memory
bounded by construction and forced clean separation between statistics accumulation and split
evaluation.  This made the tree code easier to test in isolation: a leaf's statistics are a
deterministic function of the samples seen, independent of when or whether a split occurs.

The uniform `partial_fit` / `transform` / `predict` / `update` / `result` / `reset`
interface, enforced by abstract base classes, paid dividends throughout the project.
`Pipeline` and `StreamTrainer` were written once and work with any combination of components
without modification.  Adding the `OnlineRandomForestClassifier` did not require changes to
the evaluation harness; it dropped in as a `BaseEstimator` like any other.

Test-driven development caught at least two bugs that would have been silent at the
integration level: the Hoeffding bound R-parameter error (described in Section 2.4) and the
Welford reset bug.  Both were caught within minutes of writing the relevant production code.
The discipline of writing the unit test before or immediately after writing the function —
rather than after the full system was assembled — made the test genuinely diagnostic rather
than a post-hoc sanity check.

### 4.2 What Was Harder Than Expected

ADWIN's O(W²) cut search was more of a practical constraint than initially appreciated.  The
canonical implementation (Bifet & Gavaldà, 2007) uses exponential histograms that reduce the
update cost to O(log W).  Implementing exponential histograms in pure Python is non-trivial,
and we chose instead to cap `max_window_size` at 5 000.  On streams with slow drift, the
window fills quickly and every update triggers a full O(W²) scan of 5 000 × 5 000 ÷ 2 = 12.5
million pair evaluations in the worst case.  This is acceptable for the demo-scale streams in
this project but would be a hard limit at production scale.

The ARF state machine was more complex to implement correctly than the description in Gomes
et al. (2017) suggests.  Three state variables interact per tree (warning detector, drift
detector, candidate tree), and the transitions between *stable*, *warning*, and *drift* states
must handle edge cases: what happens if the drift detector fires before a candidate was
created?  What if the warning detector fires twice in a row?  Getting these transitions
correct required careful reading of the original paper and several test iterations.

An unexpected empirical finding during the demo notebook was that `OnlineRandomForestClassifier`
with `max_features='sqrt'` underperformed a single `HoeffdingTreeClassifier` on the
8-feature Electricity dataset.  With `max_features='sqrt'`, each tree sees only 2 features,
which is insufficient diversity for such a low-dimensional dataset.  Setting `max_features=6`
(leaving out only 2 features per tree) restored the expected ensemble advantage.  This is a
well-known sensitivity of Random Forest to feature-subset size, but it was a concrete reminder
that the default `max_features='sqrt'` is calibrated for higher-dimensional datasets.

### 4.3 Limitations Honestly Stated

**O(W²) ADWIN.** As noted above, the plain-deque window implementation has O(W²) update cost
in the worst case.  The canonical O(log W) solution requires exponential histograms, which
were out of scope here.  The cap at `max_window_size = 5 000` is a pragmatic workaround, not
a principled solution.

**Binary-only AUC.** `StreamingAUC` computes the area under the ROC curve using the
trapezoidal approximation on a running score histogram, but it only supports binary
classification (y ∈ {0, 1}).  Multiclass AUC would require either a one-vs-rest decomposition
or a direct multiclass formulation (Hand & Till, 2001), neither of which was implemented.

**Axis-aligned splits only.** `HoeffdingTreeClassifier` considers only univariate threshold
splits.  Oblique or multi-feature splits can represent axis-aligned combinations of features
more compactly but require substantially more complex split-finding logic.

**No model serialisation.** All model state (leaf statistics, tree structure, scaler means
and variances) lives in Python objects in memory.  There is no mechanism to save a fitted
model to disk and resume fitting or prediction in a later process.  This is a significant
practical limitation for any deployment scenario.

### 4.4 What I Would Do Differently

Building a synthetic drift generator earlier would have accelerated development considerably.
A parameterised stream generator (adjustable drift timing, drift type — abrupt vs. gradual,
number of classes) would have made the ARF unit tests more rigorous and the demo notebook more
informative.  Instead, drift scenarios were constructed ad hoc in test fixtures, making them
harder to reuse.

Investing in visualisation tooling earlier — specifically a per-tree accuracy plot over time
— would have made the ADWIN tuning (choice of `warning_delta` and `drift_delta`) more
transparent.  The current `visualise.py` module was written near the end of the project;
having a live metric-over-time plot earlier would have surfaced the Online RF / single-tree
gap immediately and saved time diagnosing the `max_features` issue.

### 4.5 Closing

The project delivered a working streaming machine learning library in 3 566 lines of Python
and 267 tests across 31 commits, achieving 92% line coverage with no external dependencies
beyond NumPy.  The core algorithms — VFDT, online bagging, and ADWIN-based adaptive drift
detection — behave as the cited papers describe, and the vectorised implementations produce
meaningful performance improvements over their loop equivalents.  The limitations are real and
honestly bounded; the framework is not production-ready, but it is a faithful implementation
of the theoretical foundations it was designed to demonstrate.

---

## References

- Bifet, A., & Gavaldà, R. (2007). Learning from time-changing data with adaptive windowing. *Proceedings of the 2007 SIAM International Conference on Data Mining (SDM)*, 443–448.
- Domingos, P., & Hulten, G. (2000). Mining very fast data streams. *Proceedings of the 6th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 71–80.
- Gomes, H. M., Bifet, A., Read, J., Barddal, J. P., Enembreck, F., Pfharinger, B., Holmes, G., & Abdessalem, T. (2017). Adaptive random forests for evolving data stream classification. *Machine Learning*, 106(9–10), 1469–1495.
- Oza, N. C., & Russell, S. (2001). Online bagging and boosting. *Proceedings of the 8th International Workshop on Artificial Intelligence and Statistics (AISTATS)*, 105–112.
