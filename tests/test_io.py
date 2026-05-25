"""Tests for numcompute_stream.io."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from numcompute_stream.io import (
    load_csv,
    stream_chunks,
    stream_csv_rows,
    train_test_split,
)


# load_csv

class TestLoadCsv:
    def test_shape(self, tmp_csv):
        X, y = load_csv(tmp_csv)
        assert X.shape == (60, 3)
        assert y.shape == (60,)

    def test_label_column_is_excluded_from_X(self, tmp_csv):
        X, y = load_csv(tmp_csv, target_col=-1)
        # last col was label; X must not contain it
        X_full = np.loadtxt(str(tmp_csv), delimiter=",", skiprows=1)
        np.testing.assert_array_equal(X, X_full[:, :3])
        np.testing.assert_array_equal(y, X_full[:, 3])

    def test_no_header(self, tmp_csv_no_header):
        X, y = load_csv(tmp_csv_no_header, skip_header=False)
        assert X.shape == (40, 2)
        assert y.shape == (40,)

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_csv(tmp_path / "nonexistent.csv")

    def test_invalid_target_col(self, tmp_csv):
        with pytest.raises(ValueError, match="target_col"):
            load_csv(tmp_csv, target_col=99)


# train_test_split

class TestTrainTestSplit:
    def test_sizes(self, small_binary):
        X, y = small_binary
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
        assert X_tr.shape[0] == 80
        assert X_te.shape[0] == 20
        assert y_tr.shape[0] == 80
        assert y_te.shape[0] == 20

    def test_no_overlap(self, small_binary):
        X, y = small_binary
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=7)
        combined = np.vstack([X_tr, X_te])
        # all original rows must appear exactly once
        assert combined.shape[0] == X.shape[0]

    def test_shuffle_false_preserves_order(self, small_binary):
        X, y = small_binary
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)
        np.testing.assert_array_equal(X_tr, X[:80])
        np.testing.assert_array_equal(X_te, X[80:])

    def test_invalid_test_size(self, small_binary):
        X, y = small_binary
        with pytest.raises(ValueError):
            train_test_split(X, y, test_size=1.5)

    def test_mismatched_lengths(self):
        X = np.zeros((10, 2))
        y = np.zeros(8)
        with pytest.raises(ValueError):
            train_test_split(X, y)


# stream_chunks — from in-memory arrays

class TestStreamChunksFromArrays:
    def test_chunk_count_and_sizes(self, small_binary):
        X, y = small_binary          # 100 samples
        chunks = list(stream_chunks((X, y), chunk_size=30))
        sizes = [xc.shape[0] for xc, _ in chunks]
        assert sizes == [30, 30, 30, 10]  # last chunk is remainder

    def test_reconstruction(self, small_binary):
        X, y = small_binary
        X_rec = np.vstack([xc for xc, _ in stream_chunks((X, y), chunk_size=20)])
        np.testing.assert_array_equal(X_rec, X)

    def test_chunk_size_larger_than_data(self, small_binary):
        X, y = small_binary
        chunks = list(stream_chunks((X, y), chunk_size=200))
        assert len(chunks) == 1
        assert chunks[0][0].shape[0] == 100

    def test_invalid_chunk_size(self, small_binary):
        X, y = small_binary
        with pytest.raises(ValueError):
            list(stream_chunks((X, y), chunk_size=0))

    def test_loop_yields_multiple_passes(self, small_binary):
        X, y = small_binary
        gen = stream_chunks((X, y), chunk_size=100, loop=True)
        chunks = [next(gen) for _ in range(3)]
        assert all(c[0].shape == (100, 4) for c in chunks)


# stream_chunks — from CSV file

class TestStreamChunksFromFile:
    def test_from_csv_reconstructs_data(self, tmp_csv):
        X_full, y_full = load_csv(tmp_csv)
        X_rec = np.vstack([xc for xc, _ in stream_chunks(tmp_csv, chunk_size=20)])
        np.testing.assert_array_almost_equal(X_rec, X_full)

    def test_from_csv_chunk_count(self, tmp_csv):
        chunks = list(stream_chunks(tmp_csv, chunk_size=25))
        # 60 rows → 3 chunks of 25, 25, 10
        assert len(chunks) == 3


# stream_csv_rows

class TestStreamCsvRows:
    def test_row_count(self, tmp_csv):
        rows = list(stream_csv_rows(tmp_csv))
        assert len(rows) == 60

    def test_row_shape(self, tmp_csv):
        x, y = next(stream_csv_rows(tmp_csv))
        assert x.shape == (3,)
        assert isinstance(float(y), float)

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            list(stream_csv_rows(tmp_path / "missing.csv"))
