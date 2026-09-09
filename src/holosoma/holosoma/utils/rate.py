"""Rate limiting utilities for simulation loops.

This module provides high-precision rate limiting functionality using time.perf_counter()
for nanosecond-level accuracy, essential for high-frequency simulation loops.

Adapted from holosoma_inference's PreciseRateLimiter implementation.
"""

from __future__ import annotations

import time

from loguru import logger


class RateLimiter:
    """High-precision rate limiter for controlling simulation loop frequency.

    Uses time.perf_counter() for nanosecond-level precision and absolute target
    time tracking to prevent drift accumulation. This is critical for high-frequency
    simulations (e.g., 1000 Hz) where time.time()'s ~1ms resolution is insufficient.

    Key features:
    - Uses time.perf_counter() for highest precision time measurement
    - Monotonic timing (never goes backward)
    - Automatic drift tracking and compensation
    - Absolute target time approach prevents accumulation errors
    """

    def __init__(self, frequency: float):
        """Initialize rate limiter.

        Parameters
        ----------
        frequency : float
            Target frequency in Hz (e.g., 1000.0 for 1000 Hz).
        """
        self.frequency: float = frequency
        self.dt: float = 1.0 / frequency
        self.next_target_time: float | None = None

        # Drift tracking for diagnostics
        self.total_drift: float = 0.0
        self.max_drift: float = 0.0
        self.drift_count: int = 0

        logger.info(f"Rate limiter initialized: {frequency} Hz ({self.dt * 1000:.2f} ms period)")

    def sleep(self) -> float:
        """Sleep to maintain target frequency using high-precision timing.

        This method uses time.perf_counter() for nanosecond-precision timing
        and an absolute target time approach:
        1. Calculate when the next iteration should start
        2. Sleep until that target time
        3. Advance target by one period

        Returns
        -------
        float
            Actual sleep time in seconds (0.0 if already behind schedule).
        """
        current_time = time.perf_counter()

        if self.next_target_time is None:
            # First iteration - set target for next iteration
            self.next_target_time = current_time + self.dt
            return 0.0

        # Calculate sleep time to reach next target
        sleep_time = self.next_target_time - current_time

        if sleep_time > 0:
            # We're ahead of schedule, sleep until target time
            time.sleep(sleep_time)
            actual_sleep = sleep_time
        else:
            # We're behind schedule - track drift but continue
            drift = -sleep_time
            self.total_drift += drift
            self.max_drift = max(self.max_drift, drift)
            self.drift_count += 1
            actual_sleep = 0.0

        # Advance target time by one period for next iteration
        self.next_target_time += self.dt

        return actual_sleep

    def reset(self) -> None:
        """Reset the rate limiter timing and statistics."""
        self.next_target_time = None
        self.total_drift = 0.0
        self.max_drift = 0.0
        self.drift_count = 0

    def get_stats(self) -> dict[str, float]:
        """Get drift statistics for diagnostics.

        Returns
        -------
        dict[str, float]
            Dictionary containing drift statistics including average and max drift.
        """
        avg_drift = self.total_drift / max(self.drift_count, 1)
        return {
            "frequency": self.frequency,
            "period_ms": self.dt * 1000,
            "total_drift_ms": self.total_drift * 1000,
            "max_drift_ms": self.max_drift * 1000,
            "avg_drift_ms": avg_drift * 1000,
            "drift_count": self.drift_count,
        }
