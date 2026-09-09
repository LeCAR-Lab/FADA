"""MuJoCo backend module with optional Warp support.

This module provides two backends for MuJoCo simulation:
- ClassicBackend: CPU-based single-environment simulation (always available)
- WarpBackend: GPU-accelerated multi-environment simulation (optional)

WarpBackend requires additional dependencies:
- warp-lang: GPU kernel compilation framework
- mujoco-warp: MuJoCo integration with Warp

To install with GPU support:
    bash scripts/setup_mujoco.sh --with-warp

Or install dependencies manually:
    pip install warp-lang mujoco-warp
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import IMujocoBackend
from .classic_backend import ClassicBackend

# Try to import WarpBackend - gracefully handle if warp not installed
WARP_AVAILABLE = False
WarpBackend: type[IMujocoBackend] | None = None

if TYPE_CHECKING:
    from .warp_backend import WarpBackend as WarpBackendType
else:
    try:
        from .warp_backend import WarpBackend

        WARP_AVAILABLE = True
    except ImportError:
        # Warp dependencies not available - this is expected for CPU-only installs
        # WarpBackend will remain None and WARP_AVAILABLE will be False
        pass

__all__ = ["WARP_AVAILABLE", "ClassicBackend", "IMujocoBackend", "WarpBackend"]
