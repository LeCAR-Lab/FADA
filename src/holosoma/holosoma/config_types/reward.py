"""Configuration types for reward manager."""

from __future__ import annotations

from dataclasses import field
from typing import Any

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class RewardTermCfg:
    """Configuration for a single reward term."""

    func: str
    """Import path to the reward function or class"""
    """(e.g. ``holosoma.managers.reward.terms.locomotion:tracking_lin_vel``)."""

    params: dict[str, Any] = field(default_factory=dict)
    """Additional parameters forwarded to the reward term."""

    weight: float = 1.0
    """Weight applied to the reward term (manager multiplies by ``dt``)."""

    tags: list[str] = field(default_factory=list)
    """Tags for categorizing reward terms (e.g., ["penalty", "tracking"])."""


@dataclass(frozen=True)
class RewardManagerCfg:
    """Configuration for the reward manager."""

    terms: dict[str, RewardTermCfg] = field(default_factory=dict)
    """Mapping of reward term name to configuration."""

    only_positive_rewards: bool = False
    """If ``True``, clip the total reward to be non-negative."""


@dataclass(frozen=True)
class MultiAgentRewardTermCfg(RewardTermCfg):
    """Reward term with routing for decoupled (multi-policy) PPO."""

    ma_reward_group: str | None = None
    """``lower_body``, ``upper_body``, or ``shared`` (both agents).
    ``None`` is treated as ``shared`` when using :class:`MultiAgentRewardManagerCfg`."""


@dataclass(frozen=True)
class MultiAgentRewardManagerCfg(RewardManagerCfg):
    """Reward manager config that fills per-body buffers for PPO-MA."""

    multi_agent_body_keys: tuple[str, ...] | None = None
    """E.g. ``(\"lower_body\", \"upper_body\")``; requires :class:`MultiAgentRewardManager`."""
