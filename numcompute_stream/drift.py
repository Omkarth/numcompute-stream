"""Concept drift detectors for streaming data.

Implements ADWIN (ADaptive WINdowing) as described in:
    Bifet & Gavalda, "Learning from Time-Changing Data with Adaptive Windowing", SDM 2007.

The canonical implementation uses exponential histograms for O(log W) update cost.
This implementation uses a plain list window, giving O(W²) worst-case per update.
For windows capped at max_window_size=5000 this is fast enough for our scope.
"""

from __future__ import annotations

import math
from collections import deque

from .base import BaseDriftDetector


class ADWIN(BaseDriftDetector):
    """ADaptive WINdowing drift detector.

    Maintains a sliding window of recent observations and tests all candidate
    cut points for a statistically significant difference in means between the
    two sub-windows. When drift is detected the older sub-window is dropped,
    shrinking the window to the more recent distribution.

    Parameters
    ----------
    delta : float, default 0.002
        Confidence parameter controlling the Hoeffding cut threshold.
        Smaller values require stronger evidence before signalling drift.
        **Not** the same as the Hoeffding Tree split confidence (also
        sometimes called delta); these are independent hyperparameters.
    max_window_size : int, default 5000
        Maximum number of observations retained. Oldest values are evicted
        once this cap is reached, bounding memory on stable streams.
    min_window_length : int, default 10
        Minimum window size before any cut evaluation is attempted.
        Also the minimum length required for each sub-window at a candidate
        cut point.
    """

    def __init__(
        self,
        delta: float = 0.002,
        max_window_size: int = 5000,
        min_window_length: int = 10,
    ) -> None:
        """Initialise the ADWIN detector; see class docstring for parameter details."""
        if delta <= 0 or delta >= 1:
            raise ValueError(f"delta must be in (0, 1); got {delta!r}")
        if max_window_size < 2 * min_window_length:
            raise ValueError(
                "max_window_size must be at least 2 * min_window_length"
            )

        self._delta = delta
        self._max_window_size = max_window_size
        self._min_window_length = min_window_length

        self._window: deque[float] = deque()
        self._drift_count: int = 0
        self._drift_detected: bool = False

    # ------------------------------------------------------------------
    # BaseDriftDetector interface
    # ------------------------------------------------------------------

    def update(self, value: float) -> bool:
        """Append *value* to the window and test for drift.

        Parameters
        ----------
        value : float
            Scalar observation (e.g. per-sample accuracy indicator 0/1,
            or any bounded real-valued signal).

        Returns
        -------
        bool
            True if drift was detected (window was cut), False otherwise.
        """
        self._window.append(float(value))
        self._drift_detected = False

        # Enforce memory cap before evaluating cuts.
        while len(self._window) > self._max_window_size:
            self._window.popleft()

        n = len(self._window)
        if n < self._min_window_length:
            return False

        # Convert to list once for indexed slicing.
        w = list(self._window)
        total_sum = sum(w)

        # Precompute prefix sums for O(1) sub-mean calculation.
        prefix = [0.0] * (n + 1)
        for i, v in enumerate(w):
            prefix[i + 1] = prefix[i] + v

        # Scan cut points: each sub-window must be >= min_window_length.
        for cut in range(self._min_window_length, n - self._min_window_length + 1):
            n0 = cut
            n1 = n - cut

            sum0 = prefix[cut]
            sum1 = total_sum - sum0
            mean0 = sum0 / n0
            mean1 = sum1 / n1

            # Harmonic mean of sub-window sizes.
            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            eps_cut = math.sqrt((1.0 / (2.0 * m)) * math.log(4.0 * n / self._delta))

            if abs(mean0 - mean1) > eps_cut:
                # Drop the older sub-window (W0), keep the newer one (W1).
                self._window = deque(w[cut:])
                self._drift_count += 1
                self._drift_detected = True
                return True

        return False

    @property
    def drift_detected(self) -> bool:
        """True if drift was detected on the most recent :meth:`update` call."""
        return self._drift_detected

    @property
    def warning_detected(self) -> bool:
        """Always False — ADWIN does not distinguish warning zones."""
        return False

    def reset(self) -> "ADWIN":
        """Clear the window; drift_count is also reset to zero.

        Returns
        -------
        self
        """
        self._window = deque()
        self._drift_detected = False
        self._drift_count = 0
        return self

    # ------------------------------------------------------------------
    # Additional properties
    # ------------------------------------------------------------------

    @property
    def width(self) -> int:
        """Current number of observations in the window."""
        return len(self._window)

    @property
    def mean(self) -> float:
        """Mean of all observations currently in the window.

        Returns
        -------
        float
            0.0 if the window is empty.
        """
        if not self._window:
            return 0.0
        return sum(self._window) / len(self._window)

    @property
    def drift_count(self) -> int:
        """Total number of drift events detected over the detector's lifetime."""
        return self._drift_count
