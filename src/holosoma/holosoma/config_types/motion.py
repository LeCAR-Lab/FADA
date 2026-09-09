from __future__ import annotations

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class MotionLibConfig:
    """RPL-style motion library configuration."""

    motion_file: str
    """Path to a converted motion file or a directory of converted motion files."""

    step_dt: float = 1.0 / 50.0
    """Environment step dt used to align motion playback."""

    standardize_motion_length: bool = False
    """If True, normalize every motion's effective duration to ``standardize_motion_length_value`` seconds."""

    standardize_motion_length_value: float = 10.0
    """Target duration in seconds when ``standardize_motion_length`` is enabled."""
