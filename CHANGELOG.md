# Changelog

All notable changes to numcompute-stream are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- `feat(pipeline)`: `Pipeline` — chained transformers + estimator with `partial_fit`/`predict`/`transform`
- `feat(pipeline)`: `StreamTrainer` — prequential (test-then-train) evaluation with per-chunk metric logging and optional memory tracking
- `feat(visualise)`: `plot_metric_over_time`, `compare_models`, `plot_predictions_vs_ground_truth`, `plot_confusion_matrix` plotting utilities
- `demo`: `stream_demo.ipynb` comparing single Hoeffding Tree, Online RF, and Adaptive RF on a synthetic concept-drift stream
- `bench`: loop-vs-vectorised performance comparisons for `RunningStats`, `StandardScaler`, and info-gain (260×, 55×, 5.5× speedups)

### Fixed
- `fix(demo)`: set `max_features=6` in the demo notebook so the Online RF outperforms a single tree on the Electricity dataset

### Changed
- `style`: simplified section-separator comments in source and test files
- `docs`: expanded README with corrected quickstart, full module table, Benchmarks section, and tightened Design Highlights
- `docs`: promoted one-liner docstrings to Args/Returns format on `HoeffdingTreeClassifier.partial_fit`, `HoeffdingTreeClassifier.predict`, and `StreamingConfusionMatrix.update`
- `chore`: gitignore all `.venv*` variants to keep accidentally-created environments out of the repo

---

### Historical

- `chore`: init repo structure, `pyproject.toml`, `.gitignore`
- `feat(base)`: abstract base classes — `BaseTransformer`, `BaseEstimator`, `BaseMetric`, `BaseDriftDetector`
- `feat(io)`: `CSVLoader`, `train_test_split`, `stream_chunks` chunk iterator
- `test(io)`: I/O module test suite and shared `conftest.py` fixtures
- `feat(utils)`: shape validators (`check_array`, `check_X_y`) and NaN helpers
- `feat(stats)`: `RunningStats` — Welford online mean/variance with Chan's parallel merge
- `feat(stats)`: `StreamingHistogram` with CDF-based quantile approximation
- `feat(preprocessing)`: `StandardScaler`, `Imputer` (mean / most-frequent), `OneHotEncoder` — all streaming
- `feat(metrics)`: `Accuracy`, `PrecisionRecallF1`, `StreamingAUC`, `RollingMetric`, `StreamingConfusionMatrix`
- `feat(tree)`: `HoeffdingTreeClassifier` — leaf statistics and tree structure
- `feat(tree)`: Hoeffding bound and split decision logic (Gini / entropy gain)
- `feat(tree)`: `predict_proba`, `tree_depth`, `n_leaves`, `n_nodes` properties
- `feat(ensemble)`: `OnlineRandomForestClassifier` with Poisson bootstrap and OOB tracking
- `feat(drift)`: `ADWIN` adaptive windowing drift detector
- `feat(ensemble)`: ADWIN drift detection wired into `OnlineRandomForestClassifier` (warning + confirmed-drift tree replacement)
- `docs`: README skeleton and CHANGELOG initialised
