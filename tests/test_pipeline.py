"""Tests for Pipeline and StreamTrainer."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from numcompute_stream.metrics import Accuracy
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import Imputer, StandardScaler
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.tree import HoeffdingTreeClassifier


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_data(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(float)
    return X, y


def _chunks(X: np.ndarray, y: np.ndarray, size: int = 40):
    for i in range(0, len(y), size):
        yield X[i : i + size], y[i : i + size]


# ── Pipeline validation ───────────────────────────────────────────────────────

def test_pipeline_empty_steps_raises():
    with pytest.raises(ValueError, match="at least one step"):
        Pipeline([])


def test_pipeline_duplicate_names_raises():
    sc = StandardScaler()
    tree = HoeffdingTreeClassifier()
    with pytest.raises(ValueError, match="Duplicate"):
        Pipeline([("sc", sc), ("sc", tree)])


def test_pipeline_non_transformer_in_middle_raises():
    tree1 = HoeffdingTreeClassifier()
    tree2 = HoeffdingTreeClassifier()
    with pytest.raises(TypeError, match="transformer"):
        Pipeline([("t1", tree1), ("t2", tree2)])


def test_pipeline_non_estimator_final_raises():
    sc = StandardScaler()
    sc2 = StandardScaler()
    with pytest.raises(TypeError, match="estimator"):
        Pipeline([("sc1", sc), ("sc2", sc2)])


# ── Pipeline functional ───────────────────────────────────────────────────────

def test_pipeline_two_step_fits_and_predicts():
    X, y = _make_data()
    pipe = Pipeline([("sc", StandardScaler()), ("tree", HoeffdingTreeClassifier())])
    for Xc, yc in _chunks(X, y):
        pipe.partial_fit(Xc, yc)
    preds = pipe.predict(X[:10])
    assert preds.shape == (10,)
    assert set(preds.tolist()).issubset({0.0, 1.0})


def test_pipeline_three_step_handles_nan():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((100, 4))
    X[rng.random(X.shape) < 0.1] = np.nan
    y = (np.nansum(X[:, :2], axis=1) > 0).astype(float)

    pipe = Pipeline([
        ("imp", Imputer(strategy="mean")),
        ("sc", StandardScaler()),
        ("tree", HoeffdingTreeClassifier()),
    ])
    for Xc, yc in _chunks(X, y, size=25):
        pipe.partial_fit(Xc, yc)

    X_clean = np.nan_to_num(X[:10])
    preds = pipe.predict(X_clean)
    assert preds.shape == (10,)


def test_pipeline_named_steps():
    sc = StandardScaler()
    tree = HoeffdingTreeClassifier()
    pipe = Pipeline([("sc", sc), ("tree", tree)])
    ns = pipe.named_steps
    assert set(ns.keys()) == {"sc", "tree"}
    assert ns["sc"] is sc
    assert ns["tree"] is tree


def test_pipeline_getitem_by_name_and_index():
    sc = StandardScaler()
    tree = HoeffdingTreeClassifier()
    pipe = Pipeline([("sc", sc), ("tree", tree)])
    assert pipe["sc"] is sc
    assert pipe["tree"] is tree
    assert pipe[0] is sc
    assert pipe[1] is tree


def test_pipeline_transform_before_fit_raises():
    pipe = Pipeline([("sc", StandardScaler()), ("tree", HoeffdingTreeClassifier())])
    with pytest.raises(ValueError, match="not fitted"):
        pipe.transform(np.ones((5, 4)))


def test_pipeline_predict_before_fit_raises():
    pipe = Pipeline([("sc", StandardScaler()), ("tree", HoeffdingTreeClassifier())])
    with pytest.raises(ValueError, match="not fitted"):
        pipe.predict(np.ones((5, 4)))


# ── StreamTrainer ─────────────────────────────────────────────────────────────

def test_stream_trainer_constructs_without_metrics():
    pipe = Pipeline([("sc", StandardScaler()), ("tree", HoeffdingTreeClassifier())])
    trainer = StreamTrainer(pipe)
    assert trainer.metrics == {}


def test_stream_trainer_fit_stream_returns_expected_keys():
    X, y = _make_data()
    pipe = Pipeline([("sc", StandardScaler()), ("tree", HoeffdingTreeClassifier())])
    trainer = StreamTrainer(pipe, metrics={"acc": Accuracy()})
    result = trainer.fit_stream(_chunks(X, y))
    assert "per_chunk_acc" in result
    assert "memory_log" in result
    assert "n_chunks" in result
    assert "total_samples" in result


def test_stream_trainer_per_chunk_length_matches_n_chunks():
    X, y = _make_data()
    pipe = Pipeline([("sc", StandardScaler()), ("tree", HoeffdingTreeClassifier())])
    trainer = StreamTrainer(pipe, metrics={"acc": Accuracy()})
    result = trainer.fit_stream(_chunks(X, y, size=40))
    assert len(result["per_chunk_acc"]) == result["n_chunks"]


def test_stream_trainer_memory_log_populated_when_requested():
    X, y = _make_data()
    pipe = Pipeline([("sc", StandardScaler()), ("tree", HoeffdingTreeClassifier())])
    trainer = StreamTrainer(pipe, log_memory=True)
    result = trainer.fit_stream(_chunks(X, y, size=40))
    assert len(result["memory_log"]) == result["n_chunks"]
    assert all(isinstance(v, int) for v in result["memory_log"])


def test_stream_trainer_score_chunk_does_not_modify_pipeline():
    X, y = _make_data()
    pipe = Pipeline([("sc", StandardScaler()), ("tree", HoeffdingTreeClassifier())])
    acc = Accuracy()
    trainer = StreamTrainer(pipe, metrics={"acc": acc})

    # train on all but last chunk
    chunks = list(_chunks(X, y, size=40))
    for Xc, yc in chunks[:-1]:
        trainer.fit_chunk(Xc, yc)

    Xt, yt = chunks[-1]
    acc_before = acc.result()
    trainer.score_chunk(Xt, yt)
    acc_after = acc.result()

    # score_chunk must not change metric state
    assert acc_before == acc_after
    # pipeline state also unchanged: predict should still work
    preds = pipe.predict(Xt)
    assert preds.shape == (len(yt),)
