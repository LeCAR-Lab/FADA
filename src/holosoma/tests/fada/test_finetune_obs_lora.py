from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from holosoma.fada.common.backbone import MLPPolicy, TransformerPolicy
from holosoma.fada.common.lora_utils import (
    LoRAMultiheadAttention,
    RawObsPreprocess,
    Trajectory,
    TrajectoryObsWindowDataset,
    _run_epoch,
    build_merged_state_dict,
    configure_policy_finetune_method,
    inject_lora_into_policy_backbone,
    inject_lora_into_transformer_backbone,
    load_trajectories_from_h5,
)
from holosoma.utils.safe_torch_import import optim, torch


def _build_tiny_model() -> TransformerPolicy:
    return TransformerPolicy(
        obs_dim=4,
        act_dim=1,
        cmd_dim=1,
        history_len=3,
        pred_horizon=2,
        d_model=8,
        nhead=1,
        num_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        use_learned_positional_encoding=False,
        predict_future_obs=True,
    )


def _build_tiny_mlp_model() -> MLPPolicy:
    return MLPPolicy(
        obs_dim=4,
        act_dim=1,
        cmd_dim=1,
        history_len=3,
        pred_horizon=2,
        hidden_dims=(16, 8),
        dropout=0.0,
        predict_future_obs=True,
    )


def test_lora_injection_only_trains_adapters_and_merge_loads() -> None:
    model = _build_tiny_model()
    base_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    replaced = inject_lora_into_transformer_backbone(
        model,
        r=2,
        alpha=4.0,
        dropout=0.0,
    )
    assert replaced
    assert not any(name in {"obs_embed", "act_embed", "cmd_embed"} for name in replaced)
    assert not any(name.endswith(".in_proj_weight") for name in replaced)

    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    assert trainable_names
    assert all(name.endswith(".A.weight") or name.endswith(".B.weight") for name in trainable_names)

    frozen_names = {name for name, param in model.named_parameters() if not param.requires_grad}
    assert any(name.startswith("obs_embed.") for name in frozen_names)
    assert any(name.startswith("act_embed.") for name in frozen_names)
    assert any(name.startswith("cmd_embed.") for name in frozen_names)
    assert any(name.startswith("action_head.") for name in frozen_names)
    assert any(name.startswith("obs_head.") for name in frozen_names)

    merged_state = build_merged_state_dict(model)
    assert not any(key.endswith(".A.weight") or key.endswith(".B.weight") for key in merged_state)

    reloaded = _build_tiny_model()
    reloaded.load_compatible_state_dict(merged_state)

    # Freshly injected adapters have B=0, so merged weights should match original state.
    for key, value in base_state.items():
        assert torch.allclose(merged_state[key], value)


def test_lora_injection_with_qkv_merges_attention_in_proj_weight() -> None:
    model = _build_tiny_model()
    base_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    replaced = inject_lora_into_transformer_backbone(
        model,
        r=2,
        alpha=4.0,
        dropout=0.0,
        target_scope="encoder_only_qkv",
    )

    assert replaced
    assert any(name.endswith(".in_proj_weight") for name in replaced)
    assert any(isinstance(module, LoRAMultiheadAttention) for module in model.modules())

    target_module = next(module for module in model.modules() if isinstance(module, LoRAMultiheadAttention))
    with torch.no_grad():
        target_module.B.weight.fill_(0.25)

    merged_state = build_merged_state_dict(model)
    assert not torch.allclose(
        merged_state["encoder.layers.0.self_attn.in_proj_weight"],
        base_state["encoder.layers.0.self_attn.in_proj_weight"],
    )

    reloaded = _build_tiny_model()
    reloaded.load_compatible_state_dict(merged_state)


def test_mlp_lora_injection_targets_backbone_only_and_merges() -> None:
    model = _build_tiny_mlp_model()
    base_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    replaced, resolved_scope = inject_lora_into_policy_backbone(
        model,
        r=2,
        alpha=4.0,
        dropout=0.0,
        target_scope="backbone_only",
    )

    assert resolved_scope == "backbone_only"
    assert replaced
    assert all(name.startswith("backbone.") for name in replaced)
    assert not any(name.startswith("action_head.") for name in replaced)
    assert not any(name.startswith("obs_head.") for name in replaced)

    merged_state = build_merged_state_dict(model)
    reloaded = _build_tiny_mlp_model()
    reloaded.load_compatible_state_dict(merged_state)

    for key, value in base_state.items():
        assert torch.allclose(merged_state[key], value)


def test_mlp_last_layer_finetune_only_trains_final_backbone_linear() -> None:
    model = _build_tiny_mlp_model()
    base_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    replaced, resolved_method, resolved_scope = configure_policy_finetune_method(
        model,
        finetune_method="last_layer",
        lora_r=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
        lora_target_scope="backbone_only",
    )

    assert resolved_method == "last_layer"
    assert resolved_scope is None
    assert replaced == ["backbone.2"]

    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    assert trainable_names == {"backbone.2.weight", "backbone.2.bias"}

    frozen_names = {name for name, param in model.named_parameters() if not param.requires_grad}
    assert any(name.startswith("backbone.0.") for name in frozen_names)
    assert any(name.startswith("action_head.") for name in frozen_names)
    assert any(name.startswith("obs_head.") for name in frozen_names)

    merged_state = build_merged_state_dict(model)
    reloaded = _build_tiny_mlp_model()
    reloaded.load_compatible_state_dict(merged_state)

    for key, value in base_state.items():
        assert torch.allclose(merged_state[key], value)


def test_transformer_full_finetune_enables_all_model_parameters() -> None:
    model = _build_tiny_model()

    replaced, resolved_method, resolved_scope = configure_policy_finetune_method(
        model,
        finetune_method="full",
        lora_r=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
        lora_target_scope="encoder_only",
    )

    assert resolved_method == "full"
    assert resolved_scope is None
    assert replaced == ["model"]

    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    assert trainable_names
    assert any(name.startswith("obs_embed.") for name in trainable_names)
    assert any(name.startswith("encoder.layers.0.") for name in trainable_names)
    assert any(name.startswith("action_head.") for name in trainable_names)
    assert any(name.startswith("obs_head.") for name in trainable_names)


def _write_h5_episode(
    path: Path,
    *,
    include_command: bool = True,
    include_raw_obs: bool = True,
    include_dynamics_obs: bool = False,
    timesteps: int = 6,
) -> None:
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as f:
        episodes = f.create_group("episodes")
        ep = episodes.create_group("episode_0000")

        num_envs = 1
        raw_obs = np.arange(timesteps * num_envs * 4, dtype=np.float32).reshape(timesteps, num_envs, 4)
        actions = np.arange(timesteps * num_envs, dtype=np.float32).reshape(timesteps, num_envs, 1) + 1.0
        current_command = np.arange(timesteps * num_envs, dtype=np.float32).reshape(timesteps, num_envs, 1) + 100.0
        dones = np.zeros((timesteps, num_envs, 1), dtype=np.bool_)

        if include_raw_obs:
            ep.create_dataset("raw_dynamics_obs", data=raw_obs)
        if include_dynamics_obs:
            ep.create_dataset("dynamics_obs", data=raw_obs.copy())
        ep.create_dataset("actions", data=actions)
        ep.create_dataset("dones", data=dones)
        if include_command:
            ep.create_dataset("current_command", data=current_command)


def test_trajectory_windows_and_future_obs_supervision_from_h5(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset_ok.h5"
    _write_h5_episode(dataset_path, include_command=True)

    preprocess = RawObsPreprocess(
        term_order=("base_ang_vel", "dof_pos"),
        term_scale={"base_ang_vel": 2.0, "dof_pos": 3.0},
        act_dim=1,
        obs_dim=4,
    )

    trajectories, stats = load_trajectories_from_h5(
        [dataset_path],
        obs_key="auto",
        command_key="current_command",
        obs_dim=4,
        act_dim=1,
        cmd_dim=1,
        raw_obs_preprocess=preprocess,
        pred_horizon_k=2,
    )

    assert stats["obs_key_usage"]["raw_dynamics_obs"] == 1
    assert len(trajectories) == 1

    traj = trajectories[0]
    raw_obs = np.arange(6 * 4, dtype=np.float32).reshape(6, 4)
    expected_obs = raw_obs.copy()
    expected_obs[:, :3] *= 2.0
    expected_obs[:, 3:] *= 3.0
    assert np.allclose(traj.obs, expected_obs)

    dataset = TrajectoryObsWindowDataset([traj], history_len=3, pred_horizon_k=2)
    assert len(dataset) == 4  # T - K = 6 - 2

    sample0 = dataset[0]
    assert sample0["history_valid_mask"].tolist() == [False, False, True]
    assert np.allclose(sample0["history_obs"].numpy()[-1], expected_obs[0])
    assert np.allclose(sample0["history_act"].numpy(), np.zeros((3, 1), dtype=np.float32))
    assert np.allclose(sample0["target_next_observations"].numpy(), expected_obs[1:3])

    sample2 = dataset[2]
    assert sample2["history_valid_mask"].tolist() == [True, True, True]
    assert np.allclose(sample2["history_obs"].numpy(), expected_obs[0:3])
    assert np.allclose(sample2["history_act"].numpy(), np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32))
    assert np.allclose(sample2["target_next_observations"].numpy(), expected_obs[3:5])


def test_trajectory_window_dataset_returns_anchor_current_command() -> None:
    traj = Trajectory(
        obs=np.zeros((5, 4), dtype=np.float32),
        actions=np.arange(5, dtype=np.float32).reshape(5, 1),
        current_command=np.arange(5, dtype=np.float32).reshape(5, 1) + 10.0,
        source="synthetic",
    )
    dataset = TrajectoryObsWindowDataset(
        [traj],
        history_len=3,
        pred_horizon_k=1,
    )

    sample = dataset[2]
    assert sample["history_valid_mask"].tolist() == [True, True, True]
    assert np.allclose(sample["current_command"].numpy(), np.asarray([12.0], dtype=np.float32))

    sample0 = dataset[0]
    assert sample0["history_valid_mask"].tolist() == [False, False, True]
    assert np.allclose(sample0["current_command"].numpy(), np.asarray([10.0], dtype=np.float32))


def test_load_h5_trim_head_tail_steps(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset_trim.h5"
    _write_h5_episode(
        dataset_path,
        include_command=True,
        include_raw_obs=True,
        timesteps=260,
    )

    preprocess = RawObsPreprocess(
        term_order=("base_ang_vel", "dof_pos"),
        term_scale={"base_ang_vel": 1.0, "dof_pos": 1.0},
        act_dim=1,
        obs_dim=4,
    )

    trajectories, _ = load_trajectories_from_h5(
        [dataset_path],
        obs_key="auto",
        command_key="current_command",
        obs_dim=4,
        act_dim=1,
        cmd_dim=1,
        raw_obs_preprocess=preprocess,
        pred_horizon_k=2,
        trim_head_steps=100,
        trim_tail_steps=100,
    )

    assert len(trajectories) == 1
    traj = trajectories[0]
    assert traj.obs.shape[0] == 60
    assert traj.actions.shape[0] == 60
    assert traj.current_command.shape[0] == 60
    assert np.allclose(traj.obs[0], np.asarray([400.0, 401.0, 402.0, 403.0], dtype=np.float32))


def test_training_smoke_updates_only_lora_and_reduces_obs_loss() -> None:
    torch.manual_seed(7)
    np.random.seed(7)

    model = TransformerPolicy(
        obs_dim=4,
        act_dim=1,
        cmd_dim=1,
        history_len=2,
        pred_horizon=1,
        d_model=8,
        nhead=1,
        num_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        use_learned_positional_encoding=False,
        predict_future_obs=True,
    )
    inject_lora_into_transformer_backbone(model, r=2, alpha=4.0, dropout=0.0)

    non_lora_before = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if not (name.endswith(".A.weight") or name.endswith(".B.weight"))
    }

    traj = Trajectory(
        obs=np.zeros((48, 4), dtype=np.float32),
        actions=np.zeros((48, 1), dtype=np.float32),
        current_command=np.zeros((48, 1), dtype=np.float32),
        source="synthetic",
    )
    dataset = TrajectoryObsWindowDataset([traj], history_len=2, pred_horizon_k=1)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-2, weight_decay=0.0)

    losses: list[float] = []
    for epoch in range(6):
        metrics = _run_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            device=torch.device("cpu"),
            io_norm_enabled=False,
            obs_stats=None,
            action_stats=None,
            command_stats=None,
            pred_horizon_k=1,
            grad_clip=1.0,
            show_progress=False,
            epoch_desc=f"train-{epoch}",
        )
        losses.append(float(metrics["loss_obs"]))
        assert metrics["loss_action"] == pytest.approx(0.0)

    assert losses[-1] < losses[0]

    for name, param in model.named_parameters():
        if name in non_lora_before:
            assert torch.allclose(param.detach(), non_lora_before[name], atol=0.0, rtol=0.0)


def test_mlp_training_smoke_updates_only_lora_and_reduces_obs_loss() -> None:
    torch.manual_seed(11)
    np.random.seed(11)

    model = MLPPolicy(
        obs_dim=4,
        act_dim=1,
        cmd_dim=1,
        history_len=2,
        pred_horizon=1,
        hidden_dims=(8, 4),
        dropout=0.0,
        predict_future_obs=True,
    )
    inject_lora_into_policy_backbone(model, r=2, alpha=4.0, dropout=0.0, target_scope="backbone_only")

    non_lora_before = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if not (name.endswith(".A.weight") or name.endswith(".B.weight"))
    }

    traj = Trajectory(
        obs=np.linspace(0.0, 1.0, num=48 * 4, dtype=np.float32).reshape(48, 4),
        actions=np.linspace(0.0, 0.5, num=48, dtype=np.float32).reshape(48, 1),
        current_command=np.linspace(-0.2, 0.2, num=48, dtype=np.float32).reshape(48, 1),
        source="synthetic_mlp",
    )
    dataset = TrajectoryObsWindowDataset([traj], history_len=2, pred_horizon_k=1)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-2, weight_decay=0.0)

    losses: list[float] = []
    for epoch in range(6):
        metrics = _run_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            device=torch.device("cpu"),
            io_norm_enabled=False,
            obs_stats=None,
            action_stats=None,
            command_stats=None,
            pred_horizon_k=1,
            grad_clip=1.0,
            show_progress=False,
            epoch_desc=f"mlp-train-{epoch}",
        )
        losses.append(float(metrics["loss_obs"]))
        assert metrics["loss_action"] == pytest.approx(0.0)

    assert losses[-1] < losses[0]

    for name, param in model.named_parameters():
        if name in non_lora_before:
            assert torch.allclose(param.detach(), non_lora_before[name], atol=0.0, rtol=0.0)


def test_transformer_policy_uses_rank2_current_command() -> None:
    model = TransformerPolicy(
        obs_dim=4,
        act_dim=2,
        cmd_dim=3,
        history_len=2,
        pred_horizon=1,
        d_model=8,
        nhead=1,
        num_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        use_learned_positional_encoding=False,
        predict_future_obs=False,
    )
    history_obs = torch.zeros((2, 2, 4), dtype=torch.float32)
    history_act = torch.zeros((2, 2, 2), dtype=torch.float32)
    current_command = torch.zeros((2, 3), dtype=torch.float32)
    history_valid = torch.ones((2, 2), dtype=torch.bool)

    pred = model(history_obs, history_act, current_command, history_valid_mask=history_valid)
    assert pred.shape == (2, 1, 2)

    with pytest.raises(ValueError, match="rank-2"):
        model(
            history_obs,
            history_act,
            torch.zeros((2, 2, 3), dtype=torch.float32),
            history_valid_mask=history_valid,
        )
