"""Shared utilities for draw adapters across different simulators."""

from __future__ import annotations

import numpy as np

from holosoma.utils.safe_torch_import import torch


def convert_to_numpy(pos: list[float] | tuple[float, float, float] | np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert position input to numpy array.

    Args:
        pos: Position as list, tuple, numpy array, or torch tensor

    Returns:
        Position as numpy array with dtype float64
    """
    if torch is not None and isinstance(pos, torch.Tensor):
        return pos.cpu().numpy().astype(np.float64)
    if isinstance(pos, (list, tuple)):
        return np.array(pos, dtype=np.float64)
    return np.array(pos, dtype=np.float64)


def convert_to_list(pos: list[float] | tuple[float, float, float] | np.ndarray | torch.Tensor) -> list[float]:
    """Convert position input to list of floats.

    Args:
        pos: Position as list, tuple, numpy array, or torch tensor

    Returns:
        Position as list of floats
    """
    if torch is not None and isinstance(pos, torch.Tensor):
        return pos.cpu().numpy().astype(np.float64).tolist()
    if isinstance(pos, np.ndarray):
        return pos.astype(np.float64).tolist()
    if isinstance(pos, (list, tuple)):
        return list(pos)
    return list(pos)


def convert_to_tuple(
    pos: list[float] | tuple[float, float, float] | np.ndarray | torch.Tensor,
) -> tuple[float, ...]:
    """Convert position input to tuple of floats.

    Args:
        pos: Position as list, tuple, numpy array, or torch tensor

    Returns:
        Position as tuple of 3 floats
    """
    converted = convert_to_list(pos)
    if len(converted) != 3:
        raise ValueError(f"Expected 3D position, got {len(converted)} dimensions")
    return tuple(converted)
