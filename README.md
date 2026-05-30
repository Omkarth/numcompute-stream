# numcompute-stream

A streaming machine learning framework built from scratch with NumPy.
University of Adelaide — NumCompute Assignment 2.2.

## Features

- **Streaming / online learning** — every component supports incremental `partial_fit` as data arrives in chunks
- **Hoeffding Tree (VFDT)** — online decision tree with per-leaf sufficient statistics and Hoeffding-bound splits
- **Online Random Forest** — ensemble with Poisson bootstrap, OOB tracking, and ADWIN drift detection per tree
- **ADWIN drift detector** — adaptive sliding-window change detector for concept drift
- **Full preprocessing** — `StandardScaler`, `Imputer`, `OneHotEncoder`, all streaming
- **Streaming metrics** — accuracy, precision, recall, F1, AUC, confusion matrix with rolling-window variants
- **Pipeline & StreamTrainer** — chained preprocessing + estimator with prequential evaluation
- **Plotting utilities** — metric-over-time, model comparison, predictions vs ground truth, confusion-matrix heatmap

## Installation

```bash
git clone <repo-url>
cd numcompute-stream
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quickstart

The example below uses synthetic data so it runs without any dataset file.
To stream a real CSV instead, replace the tuple `(X_all, y_all)` with a file path:
`stream_chunks("data/elec2.csv", chunk_size=200, target_col=-1)`.

```python
import numpy as np
from numcompute_stream import StandardScaler, OnlineRandomForestClassifier, Accuracy
from numcompute_stream.io import stream_chunks

rng = np.random.default_rng(42)
X_all = rng.standard_normal((5000, 6))
y_all = (X_all[:, 0] + X_all[:, 1] > 0).astype(int)

scaler = StandardScaler()
model = OnlineRandomForestClassifier(n_estimators=10, random_state=42)
acc = Accuracy()

for X_chunk, y_chunk in stream_chunks((X_all, y_all), chunk_size=200):
    scaler.partial_fit(X_chunk)
    X_scaled = scaler.transform(X_chunk)
    model.partial_fit(X_scaled, y_chunk)
    acc.update(y_chunk, model.predict(X_scaled))

print(f"Final accuracy: {acc.result():.4f}")
```

## Module Overview

| File | Description |
|------|-------------|
| `base.py` | Abstract base classes: `BaseTransformer`, `BaseEstimator`, `BaseMetric`, `BaseDriftDetector` |
| `io.py` | `stream_chunks` chunk iterator, `load_csv`, and `train_test_split` for streaming data |
| `utils.py` | Shape validators (`check_array`, `check_X_y`) and NaN helpers |
| `stats.py` | `RunningStats` (Welford online mean/variance) and `StreamingHistogram` |
| `preprocessing.py` | `StandardScaler`, `Imputer`, `OneHotEncoder` — all streaming |
| `metrics.py` | `Accuracy`, `PrecisionRecallF1`, `StreamingAUC`, `RollingMetric`, `StreamingConfusionMatrix` |
| `tree.py` | `HoeffdingTreeClassifier` — Hoeffding Tree (VFDT) implementation |
| `ensemble.py` | `OnlineRandomForestClassifier` — Online Random Forest with ADWIN drift detection |
| `drift.py` | `ADWIN` — ADaptive WINdowing drift detector |
| `pipeline.py` | `Pipeline` — chain of transformers followed by an estimator |
| `stream.py` | `StreamTrainer` — prequential (test-then-train) evaluation over a chunk iterator |
| `visualise.py` | `plot_metric_over_time`, `compare_models`, `plot_predictions_vs_ground_truth`, `plot_confusion_matrix` |

## Running Tests

```bash
pytest                          # all 267 tests with coverage
pytest -v tests/test_tree.py    # single module
pytest --cov-report=html        # HTML coverage report → htmlcov/index.html
```

## How to Run the Demo

Clone, install, then launch the notebook:

```bash
git clone <repo-url>
cd numcompute-stream
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install jupyter
jupyter notebook demo/stream_demo.ipynb
```

The notebook generates all data synthetically and does not require any external CSV files.
Cells run top-to-bottom and compare a single Hoeffding Tree, an Online Random Forest, and
an Adaptive Random Forest on a synthetic concept-drift stream.

## Design Highlights

### Hoeffding Tree (VFDT)

`HoeffdingTreeClassifier` implements the Very Fast Decision Tree algorithm (Domingos & Hulten, 2000).
Each leaf accumulates per-class sample counts and per-(feature, class) Gaussian summaries via
Welford's algorithm — no raw samples are stored, so memory is bounded regardless of stream length.
Split decisions use the Hoeffding bound: when the information-gain gap between the best and
second-best split candidate exceeds ε = √(R² ln(1/δ) / 2n), the tree commits to that split.
A tie-breaking rule (parameter `tau`) allows early splitting when the bound is already tight
enough that additional samples are unlikely to reverse the decision.

### Online Random Forest

`OnlineRandomForestClassifier` builds an ensemble of Hoeffding Trees using two ideas from
batch Random Forests: random feature subsets (sampled once per tree at construction) and
Poisson-weighted bootstrap sampling (each sample is replicated by its Poisson(1) draw;
samples with draw 0 become out-of-bag without storing the training set).  OOB accuracy
is tracked per tree and used as the signal for ADWIN drift detection.  Both hard and soft
(probability-averaged) voting are supported.

### ADWIN Drift Detection

Each tree in the forest maintains two `ADWIN` detectors on its OOB correctness signal.
The looser warning detector (higher `delta`) spins up a background candidate tree when it fires;
the stricter confirmed-drift detector (lower `delta`) promotes the candidate — or falls back to
a fresh tree if none exists — when strong evidence of drift accumulates.  Both detectors require
a minimum window of 100 OOB samples before evaluating any cut, preventing spurious resets on
small or ordered datasets.  This two-threshold scheme is the core of the Adaptive Random Forest
(Gomes et al., 2017).

## Benchmarks

Vectorised NumPy operations dominate their loop baselines across all three measured hotpaths.
Full methodology and analysis: [`benchmark/results.md`](benchmark/results.md).

| Benchmark | Loop time | Vectorised time | Speedup |
|-----------|----------:|----------------:|--------:|
| `RunningStats` Welford update (100 k samples, 1-D) | 15.363 ms | 0.059 ms | **260×** |
| `StandardScaler` transform (100 k × 8 matrix) | 44.260 ms | 0.803 ms | **55×** |
| Info-gain over 30 candidate thresholds | 0.148 ms | 0.027 ms | **5.5×** |

## Limitations / Known Issues

- **O(W²) ADWIN** — uses a plain list window rather than exponential histograms; capped at `max_window_size=5000`.
- **Axis-aligned splits only** — `HoeffdingTreeClassifier` considers only univariate threshold splits.
- **Numeric features only** — categorical features must be encoded (e.g. `OneHotEncoder`) before fitting.
- **Binary-only AUC** — `StreamingAUC` supports only binary classification (`y ∈ {0, 1}`).
- **No serialisation** — fitted models cannot be saved to disk; all state lives in memory.

## License

MIT
