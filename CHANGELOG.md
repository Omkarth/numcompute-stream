# Changelog

All notable changes to numcompute-stream are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Repository scaffolding: package structure, pyproject.toml, .gitignore
- `base.py`: abstract base classes (`BaseTransformer`, `BaseEstimator`, `BaseMetric`, `BaseDriftDetector`)
- `io.py`: `CSVLoader`, `train_test_split`, `stream_chunks` iterator
- `tests/conftest.py`: shared synthetic data fixtures
- `tests/test_io.py`: I/O module test suite (5 tests)
- `utils.py`: `check_array`, `check_X_y`, `any_nan`, `nan_mask`, `nan_col_counts`, `nan_row_mask`, `safe_divide`
- `stats.py`: `RunningStats` — Welford online mean/variance with Chan's parallel merge
- `tests/test_utils.py`: shape-validator and NaN-helper tests (23 tests)
- `tests/test_stats.py`: RunningStats tests — accuracy, incremental, edge cases, merge (20 tests)
