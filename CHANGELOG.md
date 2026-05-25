# Changelog

All notable changes to numcompute-stream are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
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
