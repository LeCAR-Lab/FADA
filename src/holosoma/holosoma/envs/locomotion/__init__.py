"""Locomotion task classes built on the manager workflow.

``LeggedRobotLocomotion`` is kept as a backward-compatible alias of
``LeggedRobotLocomotionManager``.
"""

from .locomotion_manager import LeggedRobotLocomotionManager

__all__ = ["LeggedRobotLocomotion", "LeggedRobotLocomotionManager"]
