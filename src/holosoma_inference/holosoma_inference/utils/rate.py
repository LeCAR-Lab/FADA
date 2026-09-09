import threading
import time


class PreciseRateLimiter:
    """
    High-precision rate limiter using time.perf_counter() for nanosecond accuracy

    Features:
    - Uses time.perf_counter() for highest precision time measurement
    - Automatic drift compensation
    - Thread-safe
    - Supports dynamic frequency adjustment
    """

    def __init__(self, frequency: float, max_sleep_time: float = 0.1):
        """
        Initialize rate limiter

        Args:
            frequency: Target frequency (Hz)
            max_sleep_time: Maximum single sleep time (seconds), used to avoid long blocking
        """
        self.frequency = frequency
        self.period = 1.0 / frequency
        self.max_sleep_time = max_sleep_time

        # Time control variables
        self._next_time = None
        self._last_sleep_time = 0.0
        self._drift_compensation = 0.0
        self._lock = threading.Lock()

        # Statistics
        self._total_sleeps = 0
        self._total_drift = 0.0
        self._max_drift = 0.0

    def sleep(self) -> float:
        """
        Sleep until next cycle

        Returns:
            Actual sleep time (seconds)
        """
        with self._lock:
            current_time = time.perf_counter()

            # Initialize next target time
            if self._next_time is None:
                self._next_time = current_time + self.period

            # Calculate required sleep time
            sleep_time = self._next_time - current_time

            # If already timed out, record drift and return immediately
            if sleep_time <= 0:
                drift = -sleep_time
                self._total_drift += drift
                self._max_drift = max(self._max_drift, drift)
                self._next_time = current_time + self.period
                return 0.0

            # Segmented sleep for higher precision
            actual_sleep_time = 0.0
            while sleep_time > 0:
                # Calculate this sleep time
                this_sleep = min(sleep_time, self.max_sleep_time)

                # Sleep
                time.sleep(this_sleep)

                # Update actual sleep time
                actual_sleep_time += this_sleep
                sleep_time -= this_sleep

                # Check if interrupted
                if sleep_time > 0:
                    current_time = time.perf_counter()
                    sleep_time = self._next_time - current_time

            # Update next target time
            self._next_time += self.period

            # Record statistics
            self._total_sleeps += 1
            self._last_sleep_time = actual_sleep_time

            return actual_sleep_time

    def set_frequency(self, frequency: float):
        """
        Dynamically set frequency

        Args:
            frequency: New target frequency (Hz)
        """
        with self._lock:
            self.frequency = frequency
            self.period = 1.0 / frequency

    def get_stats(self) -> dict:
        """
        Get statistics

        Returns:
            Dictionary containing statistics
        """
        with self._lock:
            avg_drift = self._total_drift / max(self._total_sleeps, 1)
            return {
                "frequency": self.frequency,
                "period": self.period,
                "total_sleeps": self._total_sleeps,
                "total_drift": self._total_drift,
                "max_drift": self._max_drift,
                "avg_drift": avg_drift,
                "last_sleep_time": self._last_sleep_time,
            }

    def reset_stats(self):
        """Reset statistics"""
        with self._lock:
            self._total_sleeps = 0
            self._total_drift = 0.0
            self._max_drift = 0.0


# For backward compatibility, provide RateLimiter alias
RateLimiter = PreciseRateLimiter
