from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from holosoma.fada.planner_idm.finetune_idm_lora import _FinetuneDefaults
from holosoma.fada.planner_idm.finetune_idm_lora import main as finetune_idm_main
from holosoma.fada.planner_idm.model import PlannerIDMPolicy
from holosoma.utils.safe_torch_import import torch


def _patch_finetune_defaults(monkeypatch: pytest.MonkeyPatch, **overrides: object) -> None:
    """finetune_idm_lora.py's CLI is --checkpoint/--target-datasets/--run-name/
    --output-dir; every other knob lives on _FinetuneDefaults and must be patched
    directly rather than passed on argv."""
    for key, value in overrides.items():
        monkeypatch.setattr(_FinetuneDefaults, key, value)


def _build_tiny_checkpoint_model() -> PlannerIDMPolicy:
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


def _write_planner_idm_checkpoint(path: Path, *, include_norm: bool = False) -> None:
    model = _build_tiny_checkpoint_model()
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
            "term_scale": {
                "base_ang_vel": 1.0,
                "dof_pos": 1.0,
                "dof_vel": 1.0,
                "projected_gravity": 1.0,
            },
            "term_noise": {
                "base_ang_vel": 0.0,
                "dof_pos": 0.0,
                "dof_vel": 0.0,
                "projected_gravity": 0.0,
            },
            "add_noise": False,
        },
    }
    if include_norm:
        payload["normalization"] = {"io_enabled": True}
        payload["obs_norm_stats"] = {
            "obs_mean": [0.0, 0.1, 0.2, 0.3],
            "obs_std": [1.0, 1.1, 1.2, 1.3],
            "eps": 1e-6,
        }
        payload["action_norm_stats"] = {
            "action_mean": [0.0, 0.1],
            "action_std": [1.0, 1.1],
            "eps": 1e-6,
        }
        payload["command_norm_stats"] = {
            "command_mean": [0.0, 0.1, 0.2],
            "command_std": [1.0, 1.1, 1.2],
            "eps": 1e-6,
        }
    torch.save(payload, path)


def _write_planner_idm_h5(path: Path, *, timesteps: int = 8) -> None:
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as handle:
        episodes = handle.create_group("episodes")
        episode = episodes.create_group("episode_0000")
        dynamics_obs = np.arange(timesteps * 4, dtype=np.float32).reshape(timesteps, 1, 4)
        actions = np.arange(timesteps * 2, dtype=np.float32).reshape(timesteps, 1, 2) + 1.0
        command = np.arange(timesteps * 3, dtype=np.float32).reshape(timesteps, 1, 3) + 10.0
        dones = np.zeros((timesteps, 1, 1), dtype=np.bool_)
        episode.create_dataset("dynamics_obs", data=dynamics_obs)
        episode.create_dataset("actions", data=actions)
        episode.create_dataset("current_command", data=command)
        episode.create_dataset("dones", data=dones)


def test_finetune_idm_main_saves_step_milestone_checkpoints_and_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finetuning is step-driven only: checkpoints are written at
    `--save-step-milestones` steps, not once per epoch -- there is no epoch loop.
    See `_FinetuneDefaults`' docstring "Step-driven-only note"."""
    checkpoint_path = tmp_path / "planner_idm.pt"
    dataset_path = tmp_path / "dataset.h5"
    _write_planner_idm_checkpoint(checkpoint_path)
    _write_planner_idm_h5(dataset_path)

    output_root = tmp_path / "finetune_outputs"
    run_name = "20260101_000000_idm_artifacts"
    _patch_finetune_defaults(
        monkeypatch,
        obs_key="dynamics_obs",
        trim_head_steps=0,
        trim_tail_steps=0,
        batch_size=2,
        max_train_steps=2,
        save_step_milestones="1,2",
        device="cpu",
        use_wandb=False,
        export_onnx=False,
        save_checkpoints=True,
        save_adapter_only=False,
        show_progress=False,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "finetune_idm_lora",
            "--checkpoint",
            str(checkpoint_path),
            "--target-datasets",
            str(dataset_path),
            "--output-dir",
            str(output_root),
            "--run-name",
            run_name,
        ],
    )
    finetune_idm_main()

    run_dir = output_root / run_name
    assert (run_dir / "checkpoints" / "step_000001.pt").is_file()
    assert (run_dir / "checkpoints" / "step_000002.pt").is_file()
    assert (run_dir / "model_finetuned_idm_lora.pt").is_file()
    # No `_best` sibling -- there is no validation and so no best-vs-last selection;
    # the one checkpoint written is the final training step's.
    assert not (run_dir / "model_finetuned_idm_lora_best.pt").exists()

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["command_input_semantics"] == "current_command_broadcast_to_history_tokens"
    assert summary["idm_use_current_command_for_history"] is True
    assert summary["max_train_steps"] == 2
    assert summary["total_steps_done"] == 2
    assert len(summary["epoch_artifacts"]) == 2
    assert summary["epoch_artifacts"][0]["checkpoint"] is not None
    assert summary["epoch_artifacts"][0]["onnx"] is None


def test_finetune_idm_save_step_milestones_filters_to_declared_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only steps named in `--save-step-milestones` get a checkpoint -- there is no
    "save every step" fallback, so a single declared milestone produces exactly one
    checkpoint file even though training runs for more steps than that."""
    checkpoint_path = tmp_path / "planner_idm.pt"
    dataset_path = tmp_path / "dataset.h5"
    _write_planner_idm_checkpoint(checkpoint_path)
    _write_planner_idm_h5(dataset_path)

    output_root = tmp_path / "finetune_milestones_ckpt"
    run_name = "20260103_000000_idm_milestones"
    _patch_finetune_defaults(
        monkeypatch,
        obs_key="dynamics_obs",
        trim_head_steps=0,
        trim_tail_steps=0,
        batch_size=2,
        max_train_steps=3,
        save_step_milestones="2",
        device="cpu",
        use_wandb=False,
        export_onnx=False,
        save_checkpoints=True,
        save_adapter_only=False,
        show_progress=False,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "finetune_idm_lora",
            "--checkpoint",
            str(checkpoint_path),
            "--target-datasets",
            str(dataset_path),
            "--output-dir",
            str(output_root),
            "--run-name",
            run_name,
        ],
    )
    finetune_idm_main()

    run_dir = output_root / run_name
    assert not (run_dir / "checkpoints" / "step_000001.pt").is_file()
    assert (run_dir / "checkpoints" / "step_000002.pt").is_file()
    assert not (run_dir / "checkpoints" / "step_000003.pt").is_file()
    assert (run_dir / "model_finetuned_idm_lora.pt").is_file()
    # No `_best` sibling -- there is no validation and so no best-vs-last selection;
    # the one checkpoint written is the final training step's.
    assert not (run_dir / "model_finetuned_idm_lora_best.pt").exists()

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["max_train_steps"] == 3
    assert summary["total_steps_done"] == 3
    assert summary["last_checkpoint"] is not None
    assert len(summary["epoch_artifacts"]) == 1
    assert summary["epoch_artifacts"][0]["checkpoint"] is not None


def test_finetune_idm_default_skips_checkpoints_writes_final_onnx(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("onnx")
    checkpoint_path = tmp_path / "planner_idm.pt"
    dataset_path = tmp_path / "dataset.h5"
    _write_planner_idm_checkpoint(checkpoint_path)
    _write_planner_idm_h5(dataset_path)

    output_root = tmp_path / "finetune_onnx_only"
    run_name = "20260102_000000_idm_onnx_only"
    _patch_finetune_defaults(
        monkeypatch,
        obs_key="dynamics_obs",
        trim_head_steps=0,
        trim_tail_steps=0,
        batch_size=2,
        max_train_steps=1,
        device="cpu",
        use_wandb=False,
        show_progress=False,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "finetune_idm_lora",
            "--checkpoint",
            str(checkpoint_path),
            "--target-datasets",
            str(dataset_path),
            "--output-dir",
            str(output_root),
            "--run-name",
            run_name,
        ],
    )
    finetune_idm_main()

    run_dir = output_root / run_name
    assert not (run_dir / "checkpoints").exists()
    assert not (run_dir / "model_finetuned_idm_lora.pt").is_file()
    assert (run_dir / "planner_idm_policy.onnx").is_file()
    # The exported ONNX is the last (and only) one -- no `_best` variant.
    assert not (run_dir / "planner_idm_policy_best.onnx").exists()

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["last_checkpoint"] is None
    assert summary["selected_checkpoint"] == "last"
    assert summary["last_onnx_output_path"] is not None
    # No --save-step-milestones set, so no per-step artifacts are recorded (there is
    # no "save every step" fallback -- see the milestones-filter test above).
    assert summary["epoch_artifacts"] == []
