"""Base classes for randomization manager terms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

    from holosoma.config_types.randomization import RandomizationTermCfg
    from holosoma.managers.randomization.manager import RandomizationManager


class RandomizationTermBase(ABC):
    """Base class for stateful randomization hooks."""

    def __init__(self, cfg: RandomizationTermCfg, env: Any):
        self.cfg = cfg
        self.env = env
        self.manager: RandomizationManager | None = None

    @abstractmethod
    def setup(self) -> None:
        """Called once during environment setup."""

    @abstractmethod
    def reset(self, env_ids: Tensor | None) -> None:
        """Called when specific environments reset."""

    @abstractmethod
    def step(self) -> None:
        """Called every simulation step (if configured)."""
