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

## Installation

```bash
git clone <repo-url>
cd numcompute-stream
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quickstart

```python
import numpy as np
from numcompute_stream.io import stream_chunks
from numcompute_stream.preprocessing import StandardScaler
from numcompute_stream.ensemble import OnlineRandomForestClassifier
from numcompute_stream.metrics import Accuracy

scaler = StandardScaler()
model = OnlineRandomForestClassifier(n_estimators=10, random_state=42)
acc = Accuracy()

for X_chunk, y_chunk in stream_chunks("data/elec2.csv", chunk_size=200):
    scaler.partial_fit(X_chunk)
    X_scaled = scaler.transform(X_chunk)
    model.partial_fit(X_scaled, y_chunk.astype(int))
    acc.update(y_chunk.astype(int), model.predict(X_scaled))

print(f"Final accuracy: {acc.result():.4f}")
```

## Module Overview

| File | Description |
|------|-------------|
| `base.py` | Abstract base classes: `BaseTransformer`, `BaseEstimator`, `BaseMetric`, `BaseDriftDetector` |
| `io.py` | `stream_chunks` chunk iterator and `train_test_split` for streaming data |
| `utils.py` | Shape validators (`check_array`, `check_X_y`) and NaN helpers |
| `stats.py` | `RunningStats` (Welford online mean/variance) and `StreamingHistogram` |
| `preprocessing.py` | `StandardScaler`, `Imputer`, `OneHotEncoder` — all streaming |
| `metrics.py` | `Accuracy`, `PrecisionRecallF1`, `StreamingAUC`, `RollingMetric`, `StreamingConfusionMatrix` |
| `tree.py` | `HoeffdingTreeClassifier` — Hoeffding Tree (VFDT) implementation |
| `ensemble.py` | `OnlineRandomForestClassifier` — Online Random Forest with ADWIN drift detection |
| `drift.py` | `ADWIN` — ADaptive WINdowing drift detector |

## Running Tests

```bash
pytest                          # all 235 tests with coverage
pytest -v tests/test_tree.py    # single module
pytest --cov-report=html        # HTML coverage report → htmlcov/index.html
```

## Running the Demo Notebook

```bash
pip install jupyter
jupyter notebook demo/stream_demo.ipynb
```

## Design Highlights

### Hoeffding Tree (VFDT)

The `HoeffdingTreeClassifier` implements the Very Fast Decision Tree algorithm (Domingos & Hulten, 2000). Each leaf accumulates per-class sample counts and per-(feature, class) Gaussian summaries via Welford's algorithm. Split decisions are made using the Hoeffding bound: if the information-gain gap between the best and second-best split candidate exceeds the bound, the tree splits — otherwise it waits for more data. A tie-breaking rule (controlled by `tau`) allows splitting early when the bound is already tight enough that additional samples are unlikely to change the outcome.

### Online Random Forest

`OnlineRandomForestClassifier` extends the Hoeffding Tree into an ensemble using two ideas borrowed from batch Random Forests: random feature subsets per tree and Poisson-weighted bootstrap sampling (each chunk sample is duplicated by its Poisson(1) draw, and samples with draw 0 become out-of-bag). OOB accuracy is tracked per tree without extra data. Hard or soft voting is supported for prediction.

### ADWIN Drift Detection

Each tree in the forest maintains two `ADWIN` detectors on its OOB correctness signal. The looser warning detector (higher `delta`) spins up a background candidate tree when it fires; the stricter drift detector (lower `delta`) either promotes the candidate or resets to a fresh tree when confirmed drift is detected. Both detectors require a minimum window of 100 OOB samples before evaluating any cut, preventing spurious resets on small or ordered datasets.

## Limitations / Known Issues

- **O(W²) ADWIN** — the current implementation uses a plain list window rather than exponential histograms, giving O(W²) worst-case cost per update. Capped at `max_window_size=5000` this is acceptable for coursework throughput.
- **Axis-aligned splits only** — `HoeffdingTreeClassifier` considers only univariate threshold splits; oblique or multivariate splits are not implemented.
- **Numeric features only** — all preprocessors and tree statistics assume real-valued input. Categorical features must be encoded (e.g. `OneHotEncoder`) before fitting.
- **Binary-only AUC** — `StreamingAUC` supports only binary classification (`y ∈ {0, 1}`).
- **No serialisation** — fitted models cannot currently be saved to disk; all state lives in memory.

## License

MIT
