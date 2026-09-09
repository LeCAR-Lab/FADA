"""Tyro-based configuration system for holosoma_inference.

This module provides a type-safe configuration system using Pydantic dataclasses and Tyro CLI.
"""

from . import config_types, config_values

__all__ = [
    "config_types",
    "config_values",
]
