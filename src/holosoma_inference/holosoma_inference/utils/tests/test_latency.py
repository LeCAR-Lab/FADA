#!/usr/bin/env python3
"""Unit tests for the latency tracking implementation."""

import time
import unittest

from holosoma_inference.utils.latency import LatencyStats, LatencyTracker


class TestLatencyTracker(unittest.TestCase):
    """Test cases for LatencyTracker functionality."""

    def test_latency_tracker_basic(self):
        """Test basic latency tracker functionality."""
        tracker = LatencyTracker(window_size=10)

        # Test measurement context manager
        with tracker.measure("test_stage"):
            time.sleep(0.001)  # 1ms sleep

        # Check that measurement was recorded
        assert "test_stage" in tracker.measurements
        assert len(tracker.measurements["test_stage"]) == 1
        assert tracker.measurements["test_stage"][0] >= 1.0  # Should be at least 1ms

    def test_latency_tracker_cycle(self):
        """Test cycle measurement functionality."""
        tracker = LatencyTracker()

        tracker.start_cycle()

        with tracker.measure("stage1"):
            time.sleep(0.001)

        with tracker.measure("stage2"):
            time.sleep(0.002)

        cycle_results = tracker.end_cycle()

        # Check cycle results
        assert "stage1" in cycle_results
        assert "stage2" in cycle_results
        assert "total" in cycle_results
        assert cycle_results["stage1"] >= 1.0
        assert cycle_results["stage2"] >= 2.0
        assert cycle_results["total"] >= 3.0

    def test_latency_stats(self):
        """Test statistics calculation."""
        tracker = LatencyTracker()

        # Add multiple measurements
        for i in range(5):
            with tracker.measure("test_stage"):
                time.sleep(0.001 * (i + 1))  # Variable sleep times

        stats = tracker.get_stats(["test_stage"])

        assert "test_stage" in stats
        stat = stats["test_stage"]
        assert isinstance(stat, LatencyStats)
        assert stat.count == 5
        assert stat.mean_ms > 0
        assert stat.min_ms > 0
        assert stat.max_ms > stat.min_ms

    def test_fps_tracking(self):
        """Test FPS tracking functionality."""
        tracker = LatencyTracker()

        # Simulate multiple cycles
        for _ in range(3):
            tracker.start_cycle()
            time.sleep(0.01)  # ~100 FPS
            tracker.end_cycle()

        fps = tracker.get_fps()
        assert fps > 0
        # Should be roughly around 100 FPS (allowing for timing variations)
        assert fps < 200  # Upper bound check

    def test_get_stats_str(self):
        """Test formatted statistics string output."""
        tracker = LatencyTracker()

        # Add some measurements
        with tracker.measure("read_state"):
            time.sleep(0.001)
        with tracker.measure("inference"):
            time.sleep(0.002)

        stats_str = tracker.get_stats_str()

        # Should contain stage names and measurements
        assert "read_state:" in stats_str
        assert "inference:" in stats_str
        assert "ms" in stats_str
        assert "|" in stats_str  # Pipe separator

    def test_reset_functionality(self):
        """Test reset functionality."""
        tracker = LatencyTracker()

        # Add some measurements
        with tracker.measure("test_stage"):
            time.sleep(0.001)

        # Verify measurements exist
        assert "test_stage" in tracker.measurements

        # Reset and verify clean state
        tracker.reset()
        assert len(tracker.measurements) == 0
        assert len(tracker.current_cycle) == 0


if __name__ == "__main__":
    unittest.main()
