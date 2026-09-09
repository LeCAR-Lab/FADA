"""Oracle checkpoints naming the old algo class still load.

An oracle checkpoint stores the dotted path of the algo class that produced it,
and DAgger rebuilds the expert by resolving that string. This project's PPO
subclass is named `PPO_Deploy`; checkpoints written under the earlier name
`PPO_DeLA` resolve to the same class.
"""

from __future__ import annotations

import pytest

from holosoma.agents.ppo.ppo import PPO_Deploy
from holosoma.fada.common.utils import resolve_algo_class


def test_legacy_ppo_target_resolves_to_the_renamed_class():
    resolved = resolve_algo_class("holosoma.agents.ppo.ppo.PPO_DeLA")
    assert resolved is PPO_Deploy


def test_current_ppo_target_still_resolves():
    assert resolve_algo_class("holosoma.agents.ppo.ppo.PPO_Deploy") is PPO_Deploy


def test_unknown_target_still_raises():
    """A genuinely missing class still raises."""
    with pytest.raises(Exception):
        resolve_algo_class("holosoma.agents.ppo.ppo.NoSuchAlgoClass")


def test_trainer_uses_the_resolver_not_get_class():
    """The trainer resolves the algo target through `resolve_algo_class`, not bare
    `get_class`."""
    from pathlib import Path

    import holosoma.fada.planner_idm.trainer as trainer_mod

    trainer_src = Path(trainer_mod.__file__).read_text()
    assert "get_class(eval_cfg.algo._target_)" not in trainer_src
    assert "resolve_algo_class(eval_cfg.algo._target_)" in trainer_src
