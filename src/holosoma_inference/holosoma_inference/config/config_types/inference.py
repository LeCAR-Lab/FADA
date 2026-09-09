"""Top-level inference configuration types for holosoma_inference."""

from __future__ import annotations

from typing import Annotated

import tyro
from pydantic.dataclasses import dataclass

from .observation import ObservationConfig
from .robot import RobotConfig
from .task import TaskConfig


@dataclass(frozen=True)
class InferenceConfig:
    """Top-level configuration for policy inference.

    Combines robot, observation, and task configurations
    for running policies on real robots or in simulation.
    """

    robot: RobotConfig
    """Robot hardware and control configuration."""

    observation: ObservationConfig
    """Observation space configuration."""

    task: TaskConfig
    """Task execution configuration."""

    secondary: Annotated[InferenceConfig | None, tyro.conf.Suppress] = None
    """Secondary policy config for dual-mode (X-button switch).
    Suppressed from Tyro CLI to prevent recursive expansion; handled via
    --secondary-preset / --secondary.* in run_policy.py's pre-parser."""
