"""Tests for numcompute_stream.visualise plotting functions."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
import matplotlib.figure

from numcompute_stream.visualise import (
    compare_models,
    plot_confusion_matrix,
    plot_metric_over_time,
    plot_predictions_vs_ground_truth,
)


# plot_metric_over_time

def test_metric_over_time_returns_figure():
    fig = plot_metric_over_time([0.7, 0.8, 0.85], title="Accuracy", ylabel="Acc")
    assert isinstance(fig, matplotlib.figure.Figure)


def test_metric_over_time_single_axes():
    fig = plot_metric_over_time([0.5, 0.6], title="T", ylabel="Y")
    assert len(fig.axes) == 1


def test_metric_over_time_empty_no_error():
    fig = plot_metric_over_time([], title="Empty", ylabel="Y")
    assert isinstance(fig, matplotlib.figure.Figure)
    ax = fig.axes[0]
    assert len(ax.lines) == 0


def test_metric_over_time_drift_events_add_lines():
    values = [0.5, 0.6, 0.7, 0.8, 0.9]
    drift_events = [1, 3]
    fig = plot_metric_over_time(values, title="T", ylabel="Y", drift_events=drift_events)
    ax = fig.axes[0]
    # 1 data line + 2 vertical lines
    assert len(ax.lines) == 1 + len(drift_events)


def test_metric_over_time_save_path(tmp_path):
    out = tmp_path / "metric.png"
    fig = plot_metric_over_time([0.5, 0.6], title="T", ylabel="Y", save_path=str(out))
    assert out.exists()
    assert isinstance(fig, matplotlib.figure.Figure)


# compare_models

def test_compare_models_returns_figure():
    metric_dict = {"A": [0.7, 0.8], "B": [0.6, 0.75]}
    fig = compare_models(metric_dict, title="Compare", ylabel="Acc")
    assert isinstance(fig, matplotlib.figure.Figure)


def test_compare_models_three_models_three_lines():
    metric_dict = {"A": [0.7, 0.8, 0.9], "B": [0.6, 0.7, 0.8], "C": [0.5, 0.6, 0.7]}
    fig = compare_models(metric_dict, title="Compare", ylabel="Acc")
    ax = fig.axes[0]
    assert len(ax.lines) == 3


def test_compare_models_save_path(tmp_path):
    out = tmp_path / "compare.png"
    metric_dict = {"M1": [0.8, 0.85]}
    fig = compare_models(metric_dict, title="T", ylabel="Y", save_path=str(out))
    assert out.exists()
    assert isinstance(fig, matplotlib.figure.Figure)


# plot_predictions_vs_ground_truth

def test_predictions_vs_truth_returns_figure():
    fig = plot_predictions_vs_ground_truth([0, 1, 0, 1], [0, 0, 1, 1])
    assert isinstance(fig, matplotlib.figure.Figure)


def test_predictions_vs_truth_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        plot_predictions_vs_ground_truth([0, 1, 0], [0, 1])


def test_predictions_vs_truth_n_samples_slices():
    y_true = list(range(10))
    y_pred = list(range(10))
    fig = plot_predictions_vs_ground_truth(y_true, y_pred, n_samples=3)
    ax = fig.axes[0]
    # two step lines for 3 samples each
    for line in ax.lines:
        assert len(line.get_xdata()) == 3


def test_predictions_vs_truth_save_path(tmp_path):
    out = tmp_path / "pred.png"
    fig = plot_predictions_vs_ground_truth([0, 1], [1, 0], save_path=str(out))
    assert out.exists()


# plot_confusion_matrix

def test_confusion_matrix_returns_figure():
    cm = np.array([[5, 1], [2, 8]])
    fig = plot_confusion_matrix(cm)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_confusion_matrix_single_axes():
    cm = np.array([[5, 1], [2, 8]])
    fig = plot_confusion_matrix(cm)
    assert len(fig.axes) >= 1


def test_confusion_matrix_normalize_values_in_01():
    cm = np.array([[10, 2], [3, 15]])
    fig = plot_confusion_matrix(cm, normalize=True)
    ax = fig.axes[0]
    texts = [t.get_text() for t in ax.texts]
    for txt in texts:
        val = float(txt)
        assert 0.0 <= val <= 1.0


def test_confusion_matrix_save_path(tmp_path):
    out = tmp_path / "cm.png"
    cm = np.array([[5, 1], [2, 8]])
    fig = plot_confusion_matrix(cm, save_path=str(out))
    assert out.exists()


def test_confusion_matrix_custom_labels():
    cm = np.array([[5, 1], [2, 8]])
    fig = plot_confusion_matrix(cm, class_labels=["cat", "dog"])
    ax = fig.axes[0]
    xtick_labels = [t.get_text() for t in ax.get_xticklabels()]
    assert "cat" in xtick_labels
    assert "dog" in xtick_labels
