"""Simulator package for holosoma.

This module exports shared utilities for all simulator backends.
"""

from __future__ import annotations

# Export field requirement decorator (zero dependencies, works in all environments)
from .shared.field_decorators import mujoco_required_field

__all__ = ["mujoco_required_field"]
