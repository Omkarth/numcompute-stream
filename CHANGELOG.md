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
