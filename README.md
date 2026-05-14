# numcompute-stream

A streaming machine learning framework built from scratch with NumPy and matplotlib.
University of Adelaide — NumCompute Assignment 2.2.

## Features

- **Streaming / online learning** — every component supports incremental `partial_fit` as data arrives in chunks
- **Hoeffding Tree (VFDT)** — proper online decision tree with per-leaf sufficient statistics and Hoeffding-bound splits
- **Adaptive Random Forest (ARF)** — online ensemble with ADWIN drift detection, warning trees, and automatic replacement
- **ADWIN drift detector** — adaptive sliding-window change detector for concept drift
- **Full preprocessing pipeline** — StandardScaler, Imputer, OneHotEncoder, all streaming
- **Streaming metrics** — accuracy, precision, recall, F1, AUC, confusion matrix with rolling-window variants
- **Visualisation** — matplotlib-based streaming metric plots with drift markers

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```python
from numcompute_stream.io import stream_chunks
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import StandardScaler
from numcompute_stream.ensemble import AdaptiveRandomForest
from numcompute_stream.stream import StreamTrainer

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", AdaptiveRandomForest(n_estimators=10)),
])

trainer = StreamTrainer(pipeline)
for X_chunk, y_chunk in stream_chunks("data/elec2.csv", chunk_size=200):
    trainer.partial_fit(X_chunk, y_chunk)

trainer.plot_metrics()
```

## Project Structure

```
numcompute_stream/
  base.py          # Abstract base classes
  io.py            # CSV loader, train/test split, chunk iterator
  utils.py         # Shape validators, helpers
  stats.py         # Welford streaming stats, histograms, quantiles
  preprocessing.py # StandardScaler, Imputer, OneHotEncoder
  metrics.py       # Streaming classification metrics
  tree.py          # Hoeffding Tree (VFDT)
  ensemble.py      # Online Random Forest
  drift.py         # ADWIN drift detector
  pipeline.py      # Transformer + estimator chaining
  stream.py        # StreamTrainer orchestrator
  visualise.py     # matplotlib plot functions

tests/             # ≥30 unit tests
demo/              # stream_demo.ipynb
benchmark/         # Loop vs. vectorised comparisons
data/              # Datasets (not committed — see below)
```

## Datasets

Download the Electricity (elec2) dataset:

```bash
# From UCI / OpenML
curl -L "https://www.openml.org/data/get_csv/2419/electricity-normalized.arff" -o data/elec2_raw.csv
# Or use the benchmark script: python benchmark/download_data.py
```

## Running Tests

```bash
pytest                        # all tests with coverage
pytest -v tests/test_io.py    # single module
pytest --cov-report=html      # HTML coverage report → htmlcov/index.html
```

## Benchmarks

```bash
python benchmark/bench_stats.py
python benchmark/bench_tree.py
python benchmark/bench_pipeline.py
```

## License

Academic use only — University of Adelaide NumCompute coursework.
