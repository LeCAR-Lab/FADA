"""Latency measurement utility."""

from __future__ import annotations

import statistics
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class LatencyStats:
    """Statistics for a stage over multiple measurements."""

    stage: str
    count: int = 0
    mean_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0


class LatencyTracker:
    """Minimal latency measurement system."""

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.measurements: dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.current_cycle: dict[str, float] = {}
        self.cycle_start_time: float | None = None
        self.last_cycle_start_time: float | None = None
        self.fps_measurements: deque = deque(maxlen=window_size)

    @contextmanager
    def measure(self, stage: str):
        """Context manager for measuring a single stage."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            self.measurements[stage].append(duration_ms)
            self.current_cycle[stage] = duration_ms

    def start_cycle(self):
        """Start a new measurement cycle."""
        current_time = time.perf_counter()

        # Calculate FPS if we have a previous cycle
        if self.last_cycle_start_time is not None:
            cycle_duration = current_time - self.last_cycle_start_time
            fps = 1.0 / cycle_duration if cycle_duration > 0 else 0.0
            self.fps_measurements.append(fps)

        self.last_cycle_start_time = current_time
        self.cycle_start_time = current_time
        self.current_cycle.clear()

    def end_cycle(self) -> dict[str, float]:
        """End current cycle and return measurements."""
        if self.cycle_start_time:
            total_time = (time.perf_counter() - self.cycle_start_time) * 1000
            self.current_cycle["total"] = total_time
            self.measurements["total"].append(total_time)

        return self.current_cycle.copy()

    def get_stats(self, stages: list[str] | None = None) -> dict[str, LatencyStats]:
        """Get statistics for specified stages or all stages."""
        if stages is None:
            stages = list(self.measurements.keys())

        stats = {}
        for stage in stages:
            if self.measurements.get(stage):
                data = list(self.measurements[stage])
                stats[stage] = LatencyStats(
                    stage=stage,
                    count=len(data),
                    mean_ms=statistics.mean(data),
                    std_ms=statistics.stdev(data) if len(data) > 1 else 0.0,
                    min_ms=min(data),
                    max_ms=max(data),
                )
            else:
                raise ValueError(f"No {stage=} in {stages=}!")
        return stats

    def get_stats_str(self) -> str:
        """Get formatted one-line latency statistics string."""
        stats = self.get_stats()
        if not stats:
            return ""

        # Create one-line latency report
        latency_parts = []
        stage_order = ["read_state", "preprocessing", "inference", "postprocessing", "action_pub", "total"]
        for stage in stage_order:
            if stage in stats:
                stat = stats[stage]
                latency_parts.append(f"{stage}: {stat.mean_ms:.3f}±{stat.std_ms:.3f}ms")

        if latency_parts:
            return " | ".join(latency_parts)
        return ""

    def get_fps(self) -> float:
        """Get current FPS (frames per second) based on cycle timing."""
        if not self.fps_measurements:
            return 0.0
        return statistics.mean(self.fps_measurements)

    def reset(self):
        """Reset all measurements."""
        self.measurements.clear()
        self.current_cycle.clear()
