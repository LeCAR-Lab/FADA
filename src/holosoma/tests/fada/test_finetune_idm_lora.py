from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from holosoma.fada.common.dataset import ReplayBuffer
from holosoma.fada.planner_idm.finetune_idm_lora import (
    _BufferSampler,
    _FinetuneDefaults,
    _trainable_param_signature,
    resolve_finetune_recipe,
)
from holosoma.fada.planner_idm.finetune_idm_lora import _build_arg_parser as build_finetune_arg_parser
from holosoma.fada.planner_idm.finetune_idm_lora import main as finetune_idm_main
from holosoma.fada.planner_idm.model import PlannerIDMPolicy
from holosoma.utils.safe_torch_import import torch


def test_buffer_source_sampler_uses_configurable_mixed_offline_ratio() -> None:
    sampler = object.__new__(_BufferSampler)
    sampler._IDM_SUBOPTIMAL_RATIO = 0.25
    sampler._SUBOPTIMAL_EXPERT_RATIO = 0.5
    sampler._ONLINE_TRAJECTORY_RATIO = 0.5
    sampler._MIXED_OFFLINE_RATIO = 0.4

    counts = sampler._batch_counts(100)

    assert counts == {
        "optimal": 30,
        "suboptimal": 13,
        "suboptimal_expert": 12,
        "online": 23,
        "online_trajectory": 22,
    }


# ---------------------------------------------------------------------------
# Buffer batch sizing when no suboptimal NPZ is supplied.
#
# `_BufferSampler` is exercised directly rather than via the CLI: the release's
# small-scale (target-H5-only) path never constructs one. See
# `_FinetuneDefaults`' docstring "NPZ buffer fields note".
# ---------------------------------------------------------------------------


def _write_replay_buffer_npz(path: Path, *, base: float, num_steps: int = 40) -> None:
    """Build a dual-target ReplayBuffer NPZ in the format `_BufferSampler` consumes
    (obs_dim=4, act_dim=2, cmd_dim=3, history_len=3, pred_horizon=2 -- matching the
    tiny checkpoint fixtures used elsewhere in this test package).
    """
    buf = ReplayBuffer(
        capacity=num_steps,
        obs_dim=4,
        act_dim=2,
        cmd_dim=3,
        history_len=3,
        pred_horizon=2,
        require_future_obs_targets=True,
        growable=True,
    )
    obs = np.full((num_steps, 4), base, dtype=np.float32)
    current_command = np.full((num_steps, 3), base + 0.1, dtype=np.float32)
    executed_act = np.full((num_steps, 2), base + 0.2, dtype=np.float32)
    expert_act = np.full((num_steps, 2), base + 0.3, dtype=np.float32)
    expert_chunk = np.full((num_steps, 2, 2), base + 0.3, dtype=np.float32)
    expert_future_obs_chunk = np.full((num_steps, 2, 4), base + 0.4, dtype=np.float32)
    strict_label_valid = np.ones((num_steps,), dtype=np.bool_)
    reward = np.full((num_steps,), base + 0.5, dtype=np.float32)
    done = np.zeros((num_steps,), dtype=np.bool_)
    done[-1] = True
    env_id = np.zeros((num_steps,), dtype=np.int32)
    episode_id = np.zeros((num_steps,), dtype=np.int64)
    buf.add_batch(
        obs=obs,
        current_command=current_command,
        executed_act=executed_act,
        expert_act=expert_act,
        expert_chunk=expert_chunk,
        expert_future_obs_chunk=expert_future_obs_chunk,
        strict_label_valid=strict_label_valid,
        reward=reward,
        done=done,
        env_id=env_id,
        episode_id=episode_id,
    )
    buf.save_npz(path)


def test_buffer_sampler_no_subopt_mode_does_not_silently_shrink_batch(tmp_path: Path) -> None:
    """`idm_suboptimal_ratio=0.0` must realize the full requested batch size when no
    suboptimal buffer is supplied.

    At the non-zero default (0.375) the sampler allocates that fraction of every
    buffer batch to a source that does not exist and drops those rows, so the
    realized batch is smaller than requested and no error is raised.
    """
    optimal_path = tmp_path / "optimal.npz"
    online_path = tmp_path / "online.npz"
    _write_replay_buffer_npz(optimal_path, base=1.0)
    _write_replay_buffer_npz(online_path, base=2.0)

    common = {
        "optimal_npz": str(optimal_path),
        "suboptimal_npz": None,
        "online_npz": str(online_path),
        "obs_dim": 4,
        "act_dim": 2,
        "cmd_dim": 3,
        "history_len": 3,
        "pred_horizon": 2,
    }
    requested_bs = 512

    buggy_sampler = _BufferSampler(**common, idm_suboptimal_ratio=0.375)
    buggy_batches = buggy_sampler.sample_source_batches(requested_bs)
    buggy_realized = sum(v["history_obs"].shape[0] for v in buggy_batches.values())
    assert buggy_realized < requested_bs, (
        "sanity check: the pre-fix default ratio with no suboptimal buffer must reproduce "
        "the shrink this test guards against"
    )

    fixed_sampler = _BufferSampler(**common, idm_suboptimal_ratio=0.0)
    fixed_batches = fixed_sampler.sample_source_batches(requested_bs)
    fixed_realized = sum(v["history_obs"].shape[0] for v in fixed_batches.values())
    assert fixed_realized == requested_bs, (
        f"idm_suboptimal_ratio=0.0 with no suboptimal NPZ must realize the full requested "
        f"batch ({requested_bs}), got {fixed_realized}"
    )


def test_finetune_idm_lora_cli_exposes_trim_flags(tmp_path: Path) -> None:
    """`--trim-head-steps`/`--trim-tail-steps` are CLI args whose defaults track
    `_FinetuneDefaults` and which can be overridden on the command line.

    The NPZ dual-target buffer paths and the four buffer-mix ratios are not CLI flags
    (see `_FinetuneDefaults`' docstring "NPZ buffer fields note"); they are
    `_FinetuneDefaults` constants, not `argparse.Namespace` attributes, so this test
    does not assert on them.
    """
    parser = build_finetune_arg_parser()
    dataset_path = tmp_path / "dataset.h5"
    dataset_path.touch()
    args = parser.parse_args(
        [
            "--checkpoint",
            str(tmp_path / "ckpt.pt"),
            "--target-datasets",
            str(dataset_path),
            "--run-name",
            "t1_loco_sft",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert args.trim_head_steps == _FinetuneDefaults.trim_head_steps
    assert args.trim_tail_steps == _FinetuneDefaults.trim_tail_steps
    assert not hasattr(args, "npz_optimal")
    assert not hasattr(args, "idm_suboptimal_ratio")

    overridden = parser.parse_args(
        [
            "--checkpoint",
            str(tmp_path / "ckpt.pt"),
            "--target-datasets",
            str(dataset_path),
            "--run-name",
            "t1_loco_sft",
            "--output-dir",
            str(tmp_path / "out"),
            "--trim-head-steps",
            "5",
            "--trim-tail-steps",
            "3",
        ]
    )
    assert overridden.trim_head_steps == 5
    assert overridden.trim_tail_steps == 3


def _build_tiny_planner_idm_model() -> PlannerIDMPolicy:
    return PlannerIDMPolicy(
        obs_dim=4,
        act_dim=2,
        cmd_dim=3,
        history_len=3,
        pred_horizon=2,
        planner_d_model=8,
        planner_nhead=2,
        planner_num_layers=1,
        planner_dim_feedforward=16,
        planner_dropout=0.0,
        planner_use_action_history=True,
        idm_d_model=8,
        idm_nhead=2,
        idm_encoder_num_layers=1,
        idm_decoder_num_layers=1,
        idm_dim_feedforward=16,
        idm_dropout=0.0,
        idm_use_current_command_for_history=True,
    )


def test_finetune_idm_main_honors_overridden_trim_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The trim window used to load H5 data comes from the CLI override rather than
    `_FinetuneDefaults`.

    Drives a real (tiny) load with a non-default trim and asserts `summary.json`
    reports the overridden values.
    """
    checkpoint_path = tmp_path / "planner_idm.pt"
    model = _build_tiny_planner_idm_model()
    payload = {
        "model_state_dict": model.state_dict(),
        "planner_state_dict": model.planner_state_dict(),
        "idm_state_dict": model.idm_state_dict(),
        "cfg": {
            "history_len": 3,
            "pred_horizon": 2,
            "planner_d_model": 8,
            "planner_nhead": 2,
            "planner_num_layers": 1,
            "planner_dim_feedforward": 16,
            "planner_dropout": 0.0,
            "planner_use_learned_positional_encoding": False,
            "planner_use_action_history": True,
            "idm_d_model": 8,
            "idm_nhead": 2,
            "idm_encoder_num_layers": 1,
            "idm_decoder_num_layers": 1,
            "idm_dim_feedforward": 16,
            "idm_dropout": 0.0,
            "idm_use_learned_positional_encoding": False,
            "idm_use_current_command_for_history": True,
            "fdm_enabled": False,
        },
        "obs_dim": 4,
        "act_dim": 2,
        "cmd_dim": 3,
        "compact_obs": {
            "term_order": ["base_ang_vel", "dof_pos", "dof_vel", "projected_gravity"],
            "term_scale": dict.fromkeys(("base_ang_vel", "dof_pos", "dof_vel", "projected_gravity"), 1.0),
            "term_noise": dict.fromkeys(("base_ang_vel", "dof_pos", "dof_vel", "projected_gravity"), 0.0),
            "add_noise": False,
        },
    }
    torch.save(payload, checkpoint_path)

    h5py = pytest.importorskip("h5py")
    dataset_path = tmp_path / "dataset.h5"
    timesteps = 20
    with h5py.File(dataset_path, "w") as handle:
        episodes = handle.create_group("episodes")
        episode = episodes.create_group("episode_0000")
        episode.create_dataset("dynamics_obs", data=np.arange(timesteps * 4, dtype=np.float32).reshape(timesteps, 1, 4))
        episode.create_dataset(
            "actions", data=(np.arange(timesteps * 2, dtype=np.float32).reshape(timesteps, 1, 2) + 1.0)
        )
        episode.create_dataset(
            "current_command", data=(np.arange(timesteps * 3, dtype=np.float32).reshape(timesteps, 1, 3) + 10.0)
        )
        episode.create_dataset("dones", data=np.zeros((timesteps, 1, 1), dtype=np.bool_))

    output_root = tmp_path / "finetune_out"
    run_name = "20260101_000000_trim_override"
    overrides = {
        "obs_key": "dynamics_obs",
        "batch_size": 4,
        "max_train_steps": 1,
        "device": "cpu",
        "use_wandb": False,
        "export_onnx": False,
        "save_checkpoints": False,
        "save_adapter_only": True,
        "show_progress": False,
    }
    for key, value in overrides.items():
        monkeypatch.setattr(_FinetuneDefaults, key, value)

    old_argv = sys.argv
    sys.argv = [
        "finetune_idm_lora",
        "--checkpoint",
        str(checkpoint_path),
        "--target-datasets",
        str(dataset_path),
        "--run-name",
        run_name,
        "--output-dir",
        str(output_root),
        "--trim-head-steps",
        "2",
        "--trim-tail-steps",
        "3",
    ]
    try:
        finetune_idm_main()
    finally:
        sys.argv = old_argv

    summary = json.loads((output_root / run_name / "summary.json").read_text(encoding="utf-8"))
    assert summary["trim_head_steps"] == 2
    assert summary["trim_tail_steps"] == 3
    # `_FinetuneDefaults` itself is unchanged; only the resolved/used value moves.
    assert _FinetuneDefaults.trim_head_steps == 0
    assert _FinetuneDefaults.trim_tail_steps == 0


# ---------------------------------------------------------------------------
# A DAgger checkpoint carries its own training-time
# lr/weight_decay/grad_clip in `cfg` -- typically cfg["weight_decay"] == 0.01 and
# cfg["grad_clip"] == 1.0, which differ from the finetune recipe's 1e-3 / 0.5.
# If `_FinetuneDefaults.lr`/`weight_decay`/`grad_clip` are `None`,
# `_resolve_training_hparam`'s "cli" tier never fires for these three fields and
# the checkpoint's own values win instead.
#
# A cfg that omits these keys cannot discriminate the tiers: it falls through the
# "checkpoint" tier to "fallback" whether or not the "cli" tier is wired up. Both
# tests below therefore use a `cfg` dict that explicitly sets
# weight_decay=0.01/grad_clip=1.0, so they pass only if the "cli" tier (now
# `_FinetuneDefaults`) wins.
# ---------------------------------------------------------------------------


def _realistic_dagger_train_cfg() -> dict:
    """A `cfg` dict shaped like a real DAgger checkpoint's `payload["cfg"]`: its own
    training-time lr/weight_decay/grad_clip plus a handful of other FADAConfig fields.
    """
    return {
        "lr": 3e-4,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "batch_size": 64,
        "history_len": 30,
        "pred_horizon": 6,
        "idm_action_loss_horizon_gamma": 1.0,
        "seed": 42,
    }


def test_resolve_finetune_recipe_prefers_recipe_over_realistic_checkpoint_cfg() -> None:
    """`resolve_finetune_recipe()` resolves lr/weight_decay/grad_clip from the recipe,
    not from a real-shaped checkpoint cfg.

    The fixture cfg differs from the recipe on all three (3e-4/0.01/1.0 versus
    1e-4/1e-3/0.5), so each assertion discriminates the tiers; the `*_source` fields
    must read `"cli"`, not `"checkpoint"`.
    """
    recipe = resolve_finetune_recipe(train_cfg=_realistic_dagger_train_cfg())

    assert recipe.lr == pytest.approx(1e-4)
    assert recipe.weight_decay == pytest.approx(1e-3)
    assert recipe.grad_clip == pytest.approx(0.5)
    assert recipe.lr_source == "cli"
    assert recipe.weight_decay_source == "cli"
    assert recipe.grad_clip_source == "cli"


def test_finetune_idm_main_full_run_uses_recipe_hparams_and_produces_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Small end-to-end `main()` run asserting, in one pass:

    - hparams: the resolved lr/weight_decay/grad_clip in `summary.json` come from
      the finetune recipe (1e-3/0.5), not from the checkpoint's own cfg (0.01/1.0),
      loaded through the `payload.get("cfg")` path `main()` uses.
    - trim: --trim-head-steps/--trim-tail-steps are honored (summary.json).
    - data: every loaded trajectory trains; there is no val split.
    - step-driven stopping: training halts at max_train_steps.
    - artifacts: a checkpoint .pt and an ONNX file are written to disk.

    Small by construction (max_train_steps=3, 4 tiny synthetic trajectories): a
    wiring check, not a training-quality benchmark.
    """
    checkpoint_path = tmp_path / "planner_idm.pt"
    model = _build_tiny_planner_idm_model()
    train_cfg = _realistic_dagger_train_cfg()
    train_cfg.update(
        {
            "planner_d_model": 8,
            "planner_nhead": 2,
            "planner_num_layers": 1,
            "planner_dim_feedforward": 16,
            "planner_dropout": 0.0,
            "planner_use_learned_positional_encoding": False,
            "planner_use_action_history": True,
            "idm_d_model": 8,
            "idm_nhead": 2,
            "idm_encoder_num_layers": 1,
            "idm_decoder_num_layers": 1,
            "idm_dim_feedforward": 16,
            "idm_dropout": 0.0,
            "idm_use_learned_positional_encoding": False,
            "idm_use_current_command_for_history": True,
            "fdm_enabled": False,
            "history_len": 3,
            "pred_horizon": 2,
        }
    )
    payload = {
        "model_state_dict": model.state_dict(),
        "planner_state_dict": model.planner_state_dict(),
        "idm_state_dict": model.idm_state_dict(),
        "cfg": train_cfg,
        "obs_dim": 4,
        "act_dim": 2,
        "cmd_dim": 3,
        "compact_obs": {
            "term_order": ["base_ang_vel", "dof_pos", "dof_vel", "projected_gravity"],
            "term_scale": dict.fromkeys(("base_ang_vel", "dof_pos", "dof_vel", "projected_gravity"), 1.0),
            "term_noise": dict.fromkeys(("base_ang_vel", "dof_pos", "dof_vel", "projected_gravity"), 0.0),
            "add_noise": False,
        },
    }
    torch.save(payload, checkpoint_path)

    h5py = pytest.importorskip("h5py")
    dataset_path = tmp_path / "dataset.h5"
    timesteps = 20
    num_envs = 4
    with h5py.File(dataset_path, "w") as handle:
        episodes = handle.create_group("episodes")
        episode = episodes.create_group("episode_0000")
        episode.create_dataset(
            "dynamics_obs",
            data=np.random.default_rng(0).standard_normal((timesteps, num_envs, 4)).astype(np.float32),
        )
        episode.create_dataset(
            "actions",
            data=np.random.default_rng(1).standard_normal((timesteps, num_envs, 2)).astype(np.float32),
        )
        episode.create_dataset(
            "current_command",
            data=np.random.default_rng(2).standard_normal((timesteps, num_envs, 3)).astype(np.float32),
        )

    output_root = tmp_path / "finetune_out"
    run_name = "20260101_000000_full_run"
    overrides = {
        "obs_key": "dynamics_obs",
        "batch_size": 4,
        "max_train_steps": 3,
        "device": "cpu",
        "use_wandb": False,
        "export_onnx": True,
        "export_onnx_every_epoch": False,
        "save_checkpoints": True,
        "save_adapter_only": False,
        "show_progress": False,
    }
    for key, value in overrides.items():
        monkeypatch.setattr(_FinetuneDefaults, key, value)

    old_argv = sys.argv
    sys.argv = [
        "finetune_idm_lora",
        "--checkpoint",
        str(checkpoint_path),
        "--target-datasets",
        str(dataset_path),
        "--run-name",
        run_name,
        "--output-dir",
        str(output_root),
        "--trim-head-steps",
        "1",
        "--trim-tail-steps",
        "1",
    ]
    try:
        finetune_idm_main()
    finally:
        sys.argv = old_argv

    summary = json.loads((output_root / run_name / "summary.json").read_text(encoding="utf-8"))

    # Recipe values, not the checkpoint's own cfg values.
    assert summary["lr"] == pytest.approx(1e-4)
    assert summary["weight_decay"] == pytest.approx(1e-3)
    assert summary["grad_clip"] == pytest.approx(0.5)
    assert summary["lr_source"] == "cli"
    assert summary["weight_decay_source"] == "cli"
    assert summary["grad_clip_source"] == "cli"

    # Trim.
    assert summary["trim_head_steps"] == 1
    assert summary["trim_tail_steps"] == 1

    # Every loaded trajectory trains (there is no val split).
    assert summary["train_trajectories"] == num_envs
    assert summary["train_samples"] > 0

    # Step-driven stopping: halts at max_train_steps.
    assert summary["total_steps_done"] == 3
    assert summary["max_train_steps"] == 3

    # Artifacts written to disk.
    assert summary["last_checkpoint"] is not None
    assert Path(summary["last_checkpoint"]).is_file()
    assert summary["last_onnx_output_path"] is not None
    assert Path(summary["last_onnx_output_path"]).is_file()


# ---------------------------------------------------------------------------
# The milestone precheck near the top of main() checks `export_onnx_every_epoch` as
# well as `export_onnx`: the former gates milestone writes further down
# (`export_onnx_each_epoch = export_onnx and bool(D.export_onnx_every_epoch)`).
# `export_onnx=True` (the default) with `export_onnx_every_epoch=False` (also the
# default) yields a milestone record with `checkpoint=None, onnx=None` once training
# reaches the requested step, since neither `save_checkpoints` nor
# `export_onnx_each_epoch` is true at that point.
# ---------------------------------------------------------------------------


def _build_minimal_checkpoint_and_dataset(tmp_path: Path) -> tuple[Path, Path]:
    checkpoint_path = tmp_path / "planner_idm.pt"
    model = _build_tiny_planner_idm_model()
    payload = {
        "model_state_dict": model.state_dict(),
        "planner_state_dict": model.planner_state_dict(),
        "idm_state_dict": model.idm_state_dict(),
        "cfg": {
            "history_len": 3,
            "pred_horizon": 2,
            "planner_d_model": 8,
            "planner_nhead": 2,
            "planner_num_layers": 1,
            "planner_dim_feedforward": 16,
            "planner_dropout": 0.0,
            "planner_use_learned_positional_encoding": False,
            "planner_use_action_history": True,
            "idm_d_model": 8,
            "idm_nhead": 2,
            "idm_encoder_num_layers": 1,
            "idm_decoder_num_layers": 1,
            "idm_dim_feedforward": 16,
            "idm_dropout": 0.0,
            "idm_use_learned_positional_encoding": False,
            "idm_use_current_command_for_history": True,
            "fdm_enabled": False,
        },
        "obs_dim": 4,
        "act_dim": 2,
        "cmd_dim": 3,
        "compact_obs": {
            "term_order": ["base_ang_vel", "dof_pos", "dof_vel", "projected_gravity"],
            "term_scale": dict.fromkeys(("base_ang_vel", "dof_pos", "dof_vel", "projected_gravity"), 1.0),
            "term_noise": dict.fromkeys(("base_ang_vel", "dof_pos", "dof_vel", "projected_gravity"), 0.0),
            "add_noise": False,
        },
    }
    torch.save(payload, checkpoint_path)

    h5py = pytest.importorskip("h5py")
    dataset_path = tmp_path / "dataset.h5"
    timesteps = 20
    with h5py.File(dataset_path, "w") as handle:
        episodes = handle.create_group("episodes")
        episode = episodes.create_group("episode_0000")
        episode.create_dataset("dynamics_obs", data=np.arange(timesteps * 4, dtype=np.float32).reshape(timesteps, 1, 4))
        episode.create_dataset(
            "actions", data=(np.arange(timesteps * 2, dtype=np.float32).reshape(timesteps, 1, 2) + 1.0)
        )
        episode.create_dataset(
            "current_command", data=(np.arange(timesteps * 3, dtype=np.float32).reshape(timesteps, 1, 3) + 10.0)
        )
        episode.create_dataset("dones", data=np.zeros((timesteps, 1, 1), dtype=np.bool_))
    return checkpoint_path, dataset_path


def test_finetune_idm_main_milestone_precheck_rejects_export_onnx_without_every_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_path, dataset_path = _build_minimal_checkpoint_and_dataset(tmp_path)

    overrides = {
        "obs_key": "dynamics_obs",
        "batch_size": 4,
        "max_train_steps": 2,
        "device": "cpu",
        "use_wandb": False,
        "export_onnx": True,
        "export_onnx_every_epoch": False,  # `export_onnx` alone must not satisfy the precheck
        "save_checkpoints": False,
        "save_adapter_only": True,
        "show_progress": False,
        "save_step_milestones": "1",
    }
    for key, value in overrides.items():
        monkeypatch.setattr(_FinetuneDefaults, key, value)

    old_argv = sys.argv
    sys.argv = [
        "finetune_idm_lora",
        "--checkpoint",
        str(checkpoint_path),
        "--target-datasets",
        str(dataset_path),
        "--run-name",
        "20260101_000000_milestone_precheck_reject",
        "--output-dir",
        str(tmp_path / "out"),
    ]
    try:
        with pytest.raises(ValueError, match="save-step-milestones"):
            finetune_idm_main()
    finally:
        sys.argv = old_argv


def test_finetune_idm_main_milestone_with_export_onnx_every_epoch_writes_real_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With `export_onnx_every_epoch=True`, a step milestone produces a real ONNX
    artifact rather than a `checkpoint=None, onnx=None` record.
    """
    checkpoint_path, dataset_path = _build_minimal_checkpoint_and_dataset(tmp_path)

    output_root = tmp_path / "out"
    run_name = "20260101_000000_milestone_precheck_ok"
    overrides = {
        "obs_key": "dynamics_obs",
        "batch_size": 4,
        "max_train_steps": 2,
        "device": "cpu",
        "use_wandb": False,
        "export_onnx": True,
        "export_onnx_every_epoch": True,
        "save_checkpoints": False,
        "save_adapter_only": True,
        "show_progress": False,
        "save_step_milestones": "1",
    }
    for key, value in overrides.items():
        monkeypatch.setattr(_FinetuneDefaults, key, value)

    old_argv = sys.argv
    sys.argv = [
        "finetune_idm_lora",
        "--checkpoint",
        str(checkpoint_path),
        "--target-datasets",
        str(dataset_path),
        "--run-name",
        run_name,
        "--output-dir",
        str(output_root),
    ]
    try:
        finetune_idm_main()
    finally:
        sys.argv = old_argv

    summary = json.loads((output_root / run_name / "summary.json").read_text(encoding="utf-8"))
    epoch_artifacts = summary["epoch_artifacts"]
    assert epoch_artifacts, "expected at least one step-milestone record"
    for record in epoch_artifacts:
        assert record["checkpoint"] is not None or record["onnx"] is not None
        if record["onnx"] is not None:
            assert Path(record["onnx"]).is_file()


# ---------------------------------------------------------------------------
# The tests below assert the consequence of each behavior rather than the value
# recorded in summary.json: the window count trim produces, the trajectory set
# `main()` trained on, and an onnxruntime forward pass through the exported graph.
#
# There is no train/val split, hence no train/val disjointness, no `best_val`
# selection and no `_best` artifact to assert on. The provenance test below pins
# what `main()` trained on.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# `main()` records the trajectories it trained on -- read back off `train_dataset`
# itself, not off the list it was built from -- into
# `<run_dir>/training_data_provenance.json`, into `summary.json`, and as its return
# value. The end-to-end test below asserts that every loaded trajectory reaches the
# training set and that the returned, on-disk and summary-embedded copies agree.
# ---------------------------------------------------------------------------

_PROVENANCE_TIMESTEPS = 20
_PROVENANCE_NUM_ENVS = 4
# TrajectoryWindowDataset window count per trajectory = timesteps - pred_horizon_k.
_PROVENANCE_WINDOWS_PER_TRAJ = _PROVENANCE_TIMESTEPS - 2


def _write_multi_trajectory_dataset(tmp_path: Path) -> Path:
    """A target H5 with `_PROVENANCE_NUM_ENVS` parallel envs in one episode, so
    `load_trajectories_from_h5` yields that many distinct `Trajectory.source` ids.

    `_build_minimal_checkpoint_and_dataset`'s single-env fixture yields one
    trajectory, which cannot distinguish "recorded every trajectory it trained on"
    from "recorded the only one there was".
    """
    h5py = pytest.importorskip("h5py")
    dataset_path = tmp_path / "multi_traj_dataset.h5"
    steps = _PROVENANCE_TIMESTEPS
    envs = _PROVENANCE_NUM_ENVS
    rng = np.random.default_rng(7)
    with h5py.File(dataset_path, "w") as handle:
        episodes = handle.create_group("episodes")
        episode = episodes.create_group("episode_0000")
        episode.create_dataset("dynamics_obs", data=rng.standard_normal((steps, envs, 4)).astype(np.float32))
        episode.create_dataset("actions", data=rng.standard_normal((steps, envs, 2)).astype(np.float32))
        episode.create_dataset("current_command", data=rng.standard_normal((steps, envs, 3)).astype(np.float32))
        episode.create_dataset("dones", data=np.zeros((steps, envs, 1), dtype=np.bool_))
    return dataset_path


def _run_finetune_main_for_data_provenance(
    *,
    checkpoint_path: Path,
    dataset_path: Path,
    output_root: Path,
    run_name: str,
    extra_overrides: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> dict:
    overrides = {
        "obs_key": "dynamics_obs",
        "batch_size": 4,
        "max_train_steps": 2,
        "device": "cpu",
        "use_wandb": False,
        "export_onnx": False,
        "save_checkpoints": False,
        "save_adapter_only": True,
        "show_progress": False,
    }
    overrides.update(extra_overrides)
    for key, value in overrides.items():
        monkeypatch.setattr(_FinetuneDefaults, key, value)

    old_argv = sys.argv
    sys.argv = [
        "finetune_idm_lora",
        "--checkpoint",
        str(checkpoint_path),
        "--target-datasets",
        str(dataset_path),
        "--run-name",
        run_name,
        "--output-dir",
        str(output_root),
    ]
    try:
        return finetune_idm_main()
    finally:
        sys.argv = old_argv


def test_finetune_idm_main_records_full_training_data_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every loaded trajectory trains, and the provenance record says exactly that.

    There is no `train_val_split`/`target_dataset_count` to override; this is the
    only data path.
    """
    assert not hasattr(_FinetuneDefaults, "train_val_split")
    assert not hasattr(_FinetuneDefaults, "target_dataset_count")

    checkpoint_path, _ = _build_minimal_checkpoint_and_dataset(tmp_path)
    dataset_path = _write_multi_trajectory_dataset(tmp_path)
    output_root = tmp_path / "finetune_out_default"
    run_name = "20260101_000000_data_provenance"

    provenance = _run_finetune_main_for_data_provenance(
        checkpoint_path=checkpoint_path,
        dataset_path=dataset_path,
        output_root=output_root,
        run_name=run_name,
        extra_overrides={},
        monkeypatch=monkeypatch,
    )

    expected_windows = _PROVENANCE_NUM_ENVS * _PROVENANCE_WINDOWS_PER_TRAJ
    assert provenance["num_loaded_trajectories"] == _PROVENANCE_NUM_ENVS
    assert provenance["num_train_trajectories"] == _PROVENANCE_NUM_ENVS
    assert provenance["num_train_windows"] == expected_windows
    assert set(provenance["train_trajectory_sources"]) == set(provenance["all_trajectory_sources"])

    run_dir = output_root / run_name
    on_disk = json.loads((run_dir / "training_data_provenance.json").read_text(encoding="utf-8"))
    assert on_disk["num_train_windows"] == expected_windows
    assert on_disk["train_trajectory_sources"] == provenance["train_trajectory_sources"]

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["training_data_provenance_path"] == str(run_dir / "training_data_provenance.json")
    assert summary["training_data_provenance"]["num_train_windows"] == expected_windows
    assert summary["train_trajectories"] == _PROVENANCE_NUM_ENVS
    assert summary["train_samples"] == expected_windows


def test_finetune_idm_lora_cli_has_no_train_val_split_surface() -> None:
    """The split flags are absent from the CLI and their fields absent from the
    resolved `FinetuneRecipe`.
    """
    parser = build_finetune_arg_parser()
    flags = {opt for action in parser._actions for opt in action.option_strings}
    for removed in ("--train-val-split", "--target-dataset-count", "--val-every-steps", "--val-batches"):
        assert removed not in flags, f"{removed} must stay removed"

    removed_fields = {"train_val_split", "target_dataset_count", "val_every_steps", "val_batches"}
    recipe_fields = {field.name for field in dataclasses.fields(resolve_finetune_recipe(train_cfg={}))}
    assert recipe_fields.isdisjoint(removed_fields)


def test_trainable_param_signature_changes_only_for_requires_grad_params() -> None:
    """Unit coverage of the helper `main()`'s post-training-loop self-check uses:
    the signature changes when a `requires_grad=True` parameter's values change, and
    does not change when a frozen (`requires_grad=False`) parameter's values change.
    """
    model = _build_tiny_planner_idm_model()
    for p in model.parameters():
        p.requires_grad = False
    # Mark exactly one parameter trainable, as LoRA/full-IDM finetuning unfreezes
    # only a subset of the model.
    trainable_param = next(iter(model.parameters()))
    trainable_param.requires_grad = True
    frozen_param = None
    for p in model.parameters():
        if p is not trainable_param:
            frozen_param = p
            break
    assert frozen_param is not None

    sig_before = _trainable_param_signature(model)

    with torch.no_grad():
        frozen_param.add_(1.0)
    assert _trainable_param_signature(model) == sig_before, "mutating a frozen param must not move the signature"

    with torch.no_grad():
        trainable_param.add_(1.0)
    assert _trainable_param_signature(model) != sig_before, "mutating the trainable param must move the signature"


def test_finetune_idm_main_trim_actually_reduces_window_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The number of training windows equals the count trim implies.

    `dones` is all-zero in the fixture dataset, so the loader's per-env cutoff is the
    full episode length (see `load_trajectories_from_h5`):
    effective_len = T - trim_head - trim_tail, windows = effective_len - pred_horizon_k.
    """
    checkpoint_path, dataset_path = _build_minimal_checkpoint_and_dataset(tmp_path)
    timesteps = 20
    trim_head = 2
    trim_tail = 3
    pred_horizon = 2
    expected_windows = (timesteps - trim_head - trim_tail) - pred_horizon
    assert expected_windows > 0  # sanity check on the fixture arithmetic itself

    overrides = {
        "obs_key": "dynamics_obs",
        "batch_size": 4,
        "max_train_steps": 1,
        "device": "cpu",
        "use_wandb": False,
        "export_onnx": False,
        "save_checkpoints": False,
        "save_adapter_only": True,
        "show_progress": False,
    }
    for key, value in overrides.items():
        monkeypatch.setattr(_FinetuneDefaults, key, value)

    output_root = tmp_path / "finetune_out"
    run_name = "20260101_000000_trim_real_windows"
    old_argv = sys.argv
    sys.argv = [
        "finetune_idm_lora",
        "--checkpoint",
        str(checkpoint_path),
        "--target-datasets",
        str(dataset_path),
        "--run-name",
        run_name,
        "--output-dir",
        str(output_root),
        "--trim-head-steps",
        str(trim_head),
        "--trim-tail-steps",
        str(trim_tail),
    ]
    try:
        finetune_idm_main()
    finally:
        sys.argv = old_argv

    summary = json.loads((output_root / run_name / "summary.json").read_text(encoding="utf-8"))
    assert summary["train_samples"] == expected_windows, (
        f"trim_head={trim_head}, trim_tail={trim_tail} on a {timesteps}-step episode with "
        f"pred_horizon={pred_horizon} must produce exactly {expected_windows} training windows, "
        f"got {summary['train_samples']} (a short-circuited trim would instead give "
        f"{timesteps - pred_horizon})"
    )


def test_finetune_idm_main_export_onnx_actually_loads_and_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Validates the exported graph with `onnx.checker`, loads it with onnxruntime,
    runs one forward pass, and asserts the input/output contract (names, dtypes,
    shapes) declared by `_export_planner_idm_onnx`'s `input_names` in
    `holosoma/fada/planner_idm/eval_checkpoint.py`.

    That list is **five** inputs, all unconditional: `history_obs`, `history_act`,
    `current_command`, `history_valid_mask` and `teacher_future_obs`. The fifth is a
    graph input, and ONNX Runtime requires a feed for every graph input on every run,
    which is why `LocomotionPolicy_FADA` passes zeros for it on the policy path. The
    assertion below is an equality against the whole set, so an export that drops or
    adds one fails here.
    """
    onnx = pytest.importorskip("onnx")
    onnxruntime = pytest.importorskip("onnxruntime")

    checkpoint_path, dataset_path = _build_minimal_checkpoint_and_dataset(tmp_path)
    obs_dim, act_dim, cmd_dim, history_len, pred_horizon = 4, 2, 3, 3, 2

    overrides = {
        "obs_key": "dynamics_obs",
        "batch_size": 4,
        "max_train_steps": 1,
        "device": "cpu",
        "use_wandb": False,
        "export_onnx": True,
        "export_onnx_every_epoch": False,
        "save_checkpoints": False,
        "save_adapter_only": False,
        "show_progress": False,
    }
    for key, value in overrides.items():
        monkeypatch.setattr(_FinetuneDefaults, key, value)

    output_root = tmp_path / "finetune_out"
    run_name = "20260101_000000_onnx_real_forward"
    old_argv = sys.argv
    sys.argv = [
        "finetune_idm_lora",
        "--checkpoint",
        str(checkpoint_path),
        "--target-datasets",
        str(dataset_path),
        "--run-name",
        run_name,
        "--output-dir",
        str(output_root),
    ]
    try:
        finetune_idm_main()
    finally:
        sys.argv = old_argv

    summary = json.loads((output_root / run_name / "summary.json").read_text(encoding="utf-8"))
    onnx_path = Path(summary["last_onnx_output_path"])
    assert onnx_path.is_file()

    model_proto = onnx.load(str(onnx_path))
    onnx.checker.check_model(model_proto)

    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_names = {inp.name for inp in session.get_inputs()}
    assert input_names == {
        "history_obs",
        "history_act",
        "current_command",
        "history_valid_mask",
        "teacher_future_obs",
    }
    output_names = [out.name for out in session.get_outputs()]
    assert output_names == ["actions", "pred_future_obs", "idm_teacher_actions"]

    # batch=1: the graph is traced at batch=1 (the deployment shape -- one sim/robot
    # instance per ONNX session), and `nn.MultiheadAttention`'s internal reshape does
    # not generalize to other batch sizes under the TorchScript-based exporter this
    # repo uses (`dynamo=False` in `_export_planner_idm_onnx`).
    batch = 1
    feeds = {
        "history_obs": np.zeros((batch, history_len, obs_dim), dtype=np.float32),
        "history_act": np.zeros((batch, history_len, act_dim), dtype=np.float32),
        "current_command": np.zeros((batch, cmd_dim), dtype=np.float32),
        "history_valid_mask": np.ones((batch, history_len), dtype=np.bool_),
        "teacher_future_obs": np.zeros((batch, pred_horizon, obs_dim), dtype=np.float32),
    }
    actions, pred_future_obs, idm_teacher_actions = session.run(output_names, feeds)

    assert actions.shape[0] == batch
    assert actions.shape[-1] == act_dim
    assert pred_future_obs.shape == (batch, pred_horizon, obs_dim)
    assert idm_teacher_actions.shape[0] == batch
    assert idm_teacher_actions.shape[-1] == act_dim
    assert np.isfinite(actions).all()
    assert np.isfinite(pred_future_obs).all()
    assert np.isfinite(idm_teacher_actions).all()


def test_finetune_idm_main_writes_only_the_last_checkpoint_and_onnx(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no validation there is no best-vs-last selection: the run writes exactly
    one checkpoint and one ONNX -- the final training step's -- and no `_best`
    sibling, with `summary.json` recording `selected_checkpoint == "last"` and none
    of the `best_*` keys.
    """
    checkpoint_path, dataset_path = _build_minimal_checkpoint_and_dataset(tmp_path)

    overrides = {
        "obs_key": "dynamics_obs",
        "batch_size": 4,
        "max_train_steps": 2,
        "device": "cpu",
        "use_wandb": False,
        "export_onnx": True,
        "save_checkpoints": True,
        "save_adapter_only": False,
        "show_progress": False,
    }
    for key, value in overrides.items():
        monkeypatch.setattr(_FinetuneDefaults, key, value)

    output_root = tmp_path / "finetune_out"
    run_name = "20260101_000000_last_only_artifacts"
    old_argv = sys.argv
    sys.argv = [
        "finetune_idm_lora",
        "--checkpoint",
        str(checkpoint_path),
        "--target-datasets",
        str(dataset_path),
        "--run-name",
        run_name,
        "--output-dir",
        str(output_root),
    ]
    try:
        finetune_idm_main()
    finally:
        sys.argv = old_argv

    run_dir = output_root / run_name
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    assert summary["selected_checkpoint"] == "last"
    for removed_key in ("best_checkpoint", "best_onnx_output_path", "best_val_loss", "best_is_fallback_no_val"):
        assert removed_key not in summary, f"summary.json must not carry {removed_key!r} any more"

    assert Path(summary["last_checkpoint"]).is_file()
    assert Path(summary["last_onnx_output_path"]).is_file()
    assert not list(run_dir.glob("*_best.pt")), "no `_best` checkpoint may be written"
    assert not list(run_dir.glob("*_best.onnx")), "no `_best` ONNX may be exported"

    # The checkpoint's own record agrees with summary.json about what shipped.
    last_payload = torch.load(summary["last_checkpoint"], map_location="cpu", weights_only=False)
    assert last_payload["extra"]["finetune_idm_lora"]["selected_checkpoint"] == "last"
