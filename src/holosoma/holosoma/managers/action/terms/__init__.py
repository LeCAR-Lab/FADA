"""Action terms for common action processing patterns.

This module contains concrete implementations of action terms,
such as joint position control, velocity control, etc.
"""

from .joint_control import JointPositionActionTerm

__all__ = ["JointPositionActionTerm"]
