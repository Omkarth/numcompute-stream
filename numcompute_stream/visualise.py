"""Plotting utilities for streaming metrics."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from typing import Optional


def plot_metric_over_time(
    metric_values,
    title: str,
    ylabel: str,
    save_path=None,
    drift_events=None,
    figsize=(10, 5),
) -> plt.Figure:
    """Line plot of a metric across chunks with optional drift event markers.

    Parameters
    ----------
    metric_values:
        1-D array-like of per-chunk metric values.
    title:
        Plot title.
    ylabel:
        Label for the y-axis.
    save_path:
        File path to save the figure; if None the figure is returned without saving.
    drift_events:
        Optional list of chunk indices where drift was detected — draws a
        vertical red dashed line at each index.
    figsize:
        Figure size passed to ``plt.subplots``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    values = np.asarray(metric_values)
    fig, ax = plt.subplots(figsize=figsize)
    if values.size > 0:
        ax.plot(np.arange(len(values)), values)
    if drift_events:
        for idx in drift_events:
            ax.axvline(x=idx, color="red", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Chunk")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    if save_path is not None:
        fig.savefig(save_path)
    return fig


def compare_models(
    metric_dict: dict,
    title: str,
    ylabel: str,
    save_path=None,
    figsize=(10, 5),
) -> plt.Figure:
    """Plot per-chunk metrics for multiple models on the same axes.

    Parameters
    ----------
    metric_dict:
        Mapping of model name to a sequence of per-chunk values.
    title:
        Plot title.
    ylabel:
        Label for the y-axis.
    save_path:
        File path to save the figure; if None the figure is returned without saving.
    figsize:
        Figure size passed to ``plt.subplots``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    for name, values in metric_dict.items():
        arr = np.asarray(values)
        ax.plot(np.arange(len(arr)), arr, label=name)
    ax.set_title(title)
    ax.set_xlabel("Chunk")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()
    if save_path is not None:
        fig.savefig(save_path)
    return fig


def plot_predictions_vs_ground_truth(
    y_true,
    y_pred,
    n_samples: Optional[int] = None,
    save_path=None,
    figsize=(10, 5),
) -> plt.Figure:
    """Step plot of true and predicted labels over sample index.

    Parameters
    ----------
    y_true:
        Ground-truth labels.
    y_pred:
        Predicted labels.
    n_samples:
        If provided, only the last ``n_samples`` entries are plotted.
    save_path:
        File path to save the figure; if None the figure is returned without saving.
    figsize:
        Figure size passed to ``plt.subplots``.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If ``y_true`` and ``y_pred`` have different lengths.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"y_true and y_pred must have the same length, "
            f"got {y_true.shape[0]} and {y_pred.shape[0]}"
        )
    if n_samples is not None:
        y_true = y_true[-n_samples:]
        y_pred = y_pred[-n_samples:]
    indices = np.arange(len(y_true))
    fig, ax = plt.subplots(figsize=figsize)
    ax.step(indices, y_true, where="mid", label="Ground truth")
    ax.step(indices, y_pred, where="mid", label="Predicted", linestyle="--")
    ax.set_title("Predictions vs Ground Truth")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Label")
    ax.grid(True)
    ax.legend()
    if save_path is not None:
        fig.savefig(save_path)
    return fig


def plot_confusion_matrix(
    cm_matrix,
    class_labels=None,
    normalize: bool = False,
    title: str = "Confusion Matrix",
    save_path=None,
    figsize=(6, 5),
) -> plt.Figure:
    """Heatmap of a confusion matrix.

    Parameters
    ----------
    cm_matrix:
        2-D array-like confusion matrix of shape ``(n_classes, n_classes)``.
    class_labels:
        Optional sequence of class label strings for axis ticks.
    normalize:
        If True, normalize each row so values are in ``[0, 1]``.
    title:
        Plot title.
    save_path:
        File path to save the figure; if None the figure is returned without saving.
    figsize:
        Figure size passed to ``plt.subplots``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    cm = np.asarray(cm_matrix, dtype=float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        # avoid divide-by-zero for empty rows
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm = cm / row_sums
    n = cm.shape[0]
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ticks = np.arange(n)
    tick_labels = class_labels if class_labels is not None else [str(i) for i in range(n)]
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels)
    fmt = ".2f" if normalize else ".0f"
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
    return fig
