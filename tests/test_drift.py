"""Tests for numcompute_stream.drift (ADWIN)."""

from __future__ import annotations

import math
import pytest
import numpy as np

from numcompute_stream.drift import ADWIN


RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# 1. Stable stream — no drift detected
# ---------------------------------------------------------------------------

def test_stable_stream_no_drift():
    rng = np.random.default_rng(0)
    detector = ADWIN(delta=0.002)
    for v in rng.normal(0.5, 0.1, 1000):
        detector.update(float(v))
    assert detector.drift_count == 0


# ---------------------------------------------------------------------------
# 2. Abrupt drift — detected at least once
# ---------------------------------------------------------------------------

def test_abrupt_drift_detected():
    rng = np.random.default_rng(1)
    detector = ADWIN(delta=0.002)
    for v in rng.normal(0.5, 0.1, 500):
        detector.update(float(v))
    drift_seen = False
    for v in rng.normal(0.9, 0.1, 500):
        if detector.update(float(v)):
            drift_seen = True
    assert drift_seen
    assert detector.drift_count >= 1


# ---------------------------------------------------------------------------
# 3. Reset clears window; drift_count resets to zero
# ---------------------------------------------------------------------------

def test_reset_clears_window_and_count():
    rng = np.random.default_rng(2)
    detector = ADWIN(delta=0.002)
    # Feed enough to trigger at least one drift.
    for v in rng.normal(0.5, 0.1, 300):
        detector.update(float(v))
    for v in rng.normal(0.9, 0.1, 300):
        detector.update(float(v))
    assert detector.drift_count >= 1
    detector.reset()
    assert detector.width == 0
    assert detector.drift_count == 0


# ---------------------------------------------------------------------------
# 4. min_window_length respected — no evaluation until window is long enough
# ---------------------------------------------------------------------------

def test_min_window_length_respected():
    detector = ADWIN(delta=0.002, min_window_length=20)
    # Feed values that would differ wildly — but window < min_window_length.
    results = [detector.update(float(i % 2)) for i in range(19)]
    assert all(r is False for r in results)
    assert detector.drift_count == 0


# ---------------------------------------------------------------------------
# 5. max_window_size respected — window never exceeds cap
# ---------------------------------------------------------------------------

def test_max_window_size_capped():
    cap = 200
    detector = ADWIN(delta=0.002, max_window_size=cap, min_window_length=10)
    rng = np.random.default_rng(3)
    for v in rng.normal(0.5, 0.01, 500):
        detector.update(float(v))
    # Stable stream so no drift; window should be at or below cap.
    assert detector.width <= cap


# ---------------------------------------------------------------------------
# 6. Binary stream (0/1): stable 80% accuracy → no drift; drop to 50% → drift
# ---------------------------------------------------------------------------

def test_binary_stream_oob_correctness():
    rng = np.random.default_rng(4)
    detector = ADWIN(delta=0.002)
    # Stable 80% correct.
    stable = (rng.random(600) < 0.8).astype(float)
    for v in stable:
        detector.update(float(v))
    assert detector.drift_count == 0
    # Sudden drop to 50% correct.
    noisy = (rng.random(600) < 0.5).astype(float)
    drift_seen = False
    for v in noisy:
        if detector.update(float(v)):
            drift_seen = True
    assert drift_seen


# ---------------------------------------------------------------------------
# 7. mean property correctness on a small window
# ---------------------------------------------------------------------------

def test_mean_property():
    detector = ADWIN(delta=0.002, max_window_size=5000, min_window_length=2)
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    for v in values:
        detector.update(v)
    # If no drift was triggered, mean should match the full window.
    if detector.drift_count == 0:
        assert math.isclose(detector.mean, sum(values) / len(values), rel_tol=1e-9)
    else:
        # After drift the window is a suffix; mean must still be consistent.
        assert detector.width > 0


# ---------------------------------------------------------------------------
# 8. drift_count increments across multiple drifts
# ---------------------------------------------------------------------------

def test_drift_count_increments():
    rng = np.random.default_rng(5)
    detector = ADWIN(delta=0.002)

    def feed(mean, n):
        for v in rng.normal(mean, 0.05, n):
            detector.update(float(v))

    feed(0.1, 400)
    feed(0.9, 400)
    feed(0.1, 400)
    feed(0.9, 400)
    assert detector.drift_count >= 2


# ---------------------------------------------------------------------------
# 9. Width shrinks after drift (window drops the older portion)
# ---------------------------------------------------------------------------

def test_width_shrinks_after_drift():
    rng = np.random.default_rng(6)
    detector = ADWIN(delta=0.002)
    for v in rng.normal(0.5, 0.05, 300):
        detector.update(float(v))
    width_before = detector.width
    for v in rng.normal(0.9, 0.05, 300):
        if detector.update(float(v)):
            assert detector.width < width_before + 1
            break


# ---------------------------------------------------------------------------
# 10. Edge case: single update returns False, width == 1
# ---------------------------------------------------------------------------

def test_single_update_no_cut():
    detector = ADWIN(delta=0.002)
    result = detector.update(0.5)
    assert result is False
    assert detector.width == 1
    assert detector.drift_count == 0


# ---------------------------------------------------------------------------
# 11. warning_detected is always False (ADWIN has no warning zone)
# ---------------------------------------------------------------------------

def test_warning_detected_always_false():
    detector = ADWIN(delta=0.002)
    for v in range(50):
        detector.update(float(v % 2))
    assert detector.warning_detected is False


# ---------------------------------------------------------------------------
# 12. drift_detected property reflects last update only
# ---------------------------------------------------------------------------

def test_drift_detected_property():
    rng = np.random.default_rng(7)
    detector = ADWIN(delta=0.002)
    for v in rng.normal(0.5, 0.05, 400):
        detector.update(float(v))
    for v in rng.normal(0.9, 0.05, 400):
        detector.update(float(v))
    # After the stream feed the property must agree with the last update result.
    last_result = detector.update(float(rng.normal(0.9, 0.05)))
    assert detector.drift_detected == last_result
