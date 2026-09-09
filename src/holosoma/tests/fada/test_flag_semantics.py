"""Agreement between the three entry points' flags and their `--help` text.

Each test states the flag's behaviour and requires the help text (or the guard) to
match it. Where a claim is about a unit or an interaction rather than about wording,
the premise is read from the source rather than asserted about.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest
from holosoma.fada.planner_idm import config as fada_config
from holosoma.fada.planner_idm import eval_checkpoint as fada_eval
from holosoma.fada.planner_idm.config import FADAConfig, build_arg_parser, config_from_args
from holosoma.fada.planner_idm.trainer import FADATrainer

_EXPERT = "/nonexistent/oracle_run/20250101_120000_t1_oracle/model_24999.pt"


def _train_cfg(*extra: str) -> FADAConfig:
    args = build_arg_parser().parse_args(["--expert-checkpoint", _EXPERT, *extra])
    return config_from_args(args)


def _help_for(field: str) -> str:
    matches = [entry[3] for entry in fada_config._OVERRIDABLE_FIELDS if entry[0] == field]
    assert len(matches) == 1, f"{field}: expected exactly one CLI entry, got {len(matches)}"
    return matches[0].lower()


def _eval_help_for(field: str) -> str:
    matches = [entry[3] for entry in fada_eval._OVERRIDABLE_DEFAULTS if entry[0] == field]
    assert len(matches) == 1, f"{field}: expected exactly one CLI entry, got {len(matches)}"
    return matches[0].lower()


# ---------------------------------------------------------------------------
# `--warmup-episodes` counts parallel rollouts, not episodes.
# ---------------------------------------------------------------------------


def test_warmup_episodes_really_counts_parallel_rollouts_not_episodes() -> None:
    """The collection target is rollouts * num_envs * horizon."""
    cfg = FADAConfig(expert_checkpoint=_EXPERT, warmup_episodes=1, warmup_collect_steps=None)
    fake = types.SimpleNamespace(cfg=cfg)
    target = FADATrainer._target_optimal_offline_steps(fake, num_envs=1024, horizon=500)
    assert target == 1 * 1024 * 500

    # The defaults the help text quotes: 1024 envs, 500 steps per episode.
    assert FADAConfig(expert_checkpoint=_EXPERT).num_envs == 1024
    assert FADAConfig(expert_checkpoint=_EXPERT).warmup_max_steps_per_episode == 500


def test_the_warmup_episodes_help_states_the_real_unit() -> None:
    help_text = _help_for("warmup_episodes")
    assert "rollout" in help_text, help_text
    assert "num_envs" in help_text, "the help must give the multiplication, not just a caveat"
    assert "warmup_max_steps_per_episode" in help_text, help_text


def test_the_warmup_collect_steps_help_states_the_overshoot() -> None:
    """It is a floor: collection runs whole parallel rollouts and stops once the total is
    reached, so the result exceeds the budget by up to one rollout."""
    help_text = _help_for("warmup_collect_steps")
    assert "floor" in help_text or "exceed" in help_text, help_text
    assert "rollout" in help_text, help_text


# ---------------------------------------------------------------------------
# `--resume-online-buffer` requires `--resume-checkpoint`.
# ---------------------------------------------------------------------------


def test_resume_online_buffer_alone_is_rejected(tmp_path: Path) -> None:
    buffer = tmp_path / "online_buffer.npz"
    buffer.write_bytes(b"")

    with pytest.raises(ValueError, match="--resume-online-buffer requires --resume-checkpoint"):
        _train_cfg("--resume-online-buffer", str(buffer))


def test_a_resume_online_buffer_path_that_does_not_exist_is_rejected(tmp_path: Path) -> None:
    """A `--resume-online-buffer` path that does not exist raises rather than starting
    with an empty buffer."""
    with pytest.raises(ValueError, match="does not exist"):
        _train_cfg(
            "--resume-checkpoint",
            str(tmp_path / "student.pt"),
            "--resume-online-buffer",
            str(tmp_path / "typo.npz"),
        )


def test_the_valid_resume_pair_is_still_accepted(tmp_path: Path) -> None:
    """The valid `--resume-checkpoint` + `--resume-online-buffer` pair is accepted."""
    buffer = tmp_path / "online_buffer.npz"
    buffer.write_bytes(b"")
    cfg = _train_cfg(
        "--resume-checkpoint",
        str(tmp_path / "student.pt"),
        "--resume-online-buffer",
        str(buffer),
    )
    assert cfg.resume_online_buffer == str(buffer)


def test_resume_checkpoint_alone_is_still_accepted(tmp_path: Path) -> None:
    """The derived fallbacks (`online_buffer_latest.npz`, then `online_buffer.npz`) are
    guesses the code makes on the user's behalf; their absence is not an error."""
    cfg = _train_cfg("--resume-checkpoint", str(tmp_path / "student.pt"))
    assert cfg.resume_online_buffer is None


def test_the_resume_online_buffer_help_says_it_needs_its_companion() -> None:
    help_text = _help_for("resume_online_buffer")
    assert "--resume-checkpoint" in help_text, help_text


# ---------------------------------------------------------------------------
# `--load-warmup-ckpt` and `--force-warmup-train` cancel each other at the defaults.
# ---------------------------------------------------------------------------


def test_both_warmup_reuse_flags_default_true_so_the_load_never_happens() -> None:
    """The premise: `load_warmup_ckpt and not force_warmup_train` with both true."""
    cfg = FADAConfig(expert_checkpoint=_EXPERT)
    assert cfg.load_warmup_ckpt is True
    assert cfg.force_warmup_train is True
    assert not (cfg.load_warmup_ckpt and not cfg.force_warmup_train), (
        "if this ever becomes reachable at the defaults, the help below is wrong again"
    )


def test_the_warmup_reuse_help_admits_the_default_never_loads() -> None:
    load_help = _help_for("load_warmup_ckpt")
    assert "never" in load_help, load_help
    assert "--force-warmup-train" in load_help, "the help must name the flag that cancels it"

    force_help = _help_for("force_warmup_train")
    # `--force-warmup-train` suppresses the load; the two cannot both take effect.
    assert "even when a cached checkpoint loads" not in force_help, force_help
    assert "--load-warmup-ckpt" in force_help, force_help


# ---------------------------------------------------------------------------
# `--future-obs-mask-step` is one-based and is validated at parse time.
# ---------------------------------------------------------------------------


def _eval_defaults(*extra: str):
    args = fada_eval._build_arg_parser().parse_args(["--checkpoint", "/nonexistent/model_final.pt", *extra])
    return fada_eval.resolve_eval_defaults(args)


def test_future_obs_mask_step_zero_is_rejected_at_the_cli() -> None:
    """`--future-obs-mask-step 0` is rejected while parsing, before the checkpoint is
    loaded and the simulator is brought up."""
    with pytest.raises(ValueError, match="--future-obs-mask-step"):
        _eval_defaults("--future-obs-mask-step", "0")


def test_the_mask_modes_that_need_a_step_are_rejected_without_one() -> None:
    for mode in ("drop", "only", "prefix"):
        with pytest.raises(ValueError, match="--future-obs-mask-step"):
            _eval_defaults("--future-obs-mask-mode", mode)


def test_the_mask_modes_that_do_not_need_a_step_are_still_accepted() -> None:
    for mode in ("none", "full"):
        resolved = _eval_defaults("--future-obs-mask-mode", mode)
        assert resolved.future_obs_mask_mode == mode
        assert resolved.future_obs_mask_step is None


def test_a_valid_mask_mode_and_step_pair_is_still_accepted() -> None:
    resolved = _eval_defaults("--future-obs-mask-mode", "drop", "--future-obs-mask-step", "1")
    assert resolved.future_obs_mask_step == 1


def test_the_mask_step_help_states_that_it_is_one_based() -> None:
    help_text = _eval_help_for("future_obs_mask_step")
    assert "one-based" in help_text, help_text
    assert "pred_horizon" in help_text, "the help must give the upper bound, not just the lower one"
