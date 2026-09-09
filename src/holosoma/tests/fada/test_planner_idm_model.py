from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from holosoma.fada.common.current_command import (
    COMMAND_PROFILE_LOCOMOTION,
    command_components_for_profile,
    command_dim_for_profile,
    extract_current_command_torch,
)
from holosoma.fada.common.lora_utils import build_merged_state_dict
from holosoma.fada.planner_idm.eval_checkpoint import _build_model_from_checkpoint
from holosoma.fada.planner_idm.finetune_idm_lora import (
    TrajectoryWindowDataset,
    configure_idm_finetune_method,
    inject_lora_into_idm_backbone,
)
from holosoma.fada.planner_idm.model import (
    IDM,
    Planner,
    PlannerIDMPolicy,
)
from holosoma.utils.safe_torch_import import torch


def _fake_command_env(commands: torch.Tensor) -> SimpleNamespace:
    num_envs = int(commands.shape[0])
    gait_state = SimpleNamespace(phase=torch.zeros((num_envs, 2), dtype=torch.float32, device=commands.device))
    command_manager = SimpleNamespace(
        commands=commands,
        get_state=lambda name: gait_state if name == "locomotion_gait" else None,
    )
    return SimpleNamespace(
        command_manager=command_manager,
        device=commands.device,
        num_envs=num_envs,
    )


def _build_tiny_policy(*, planner_use_action_history: bool = False) -> PlannerIDMPolicy:
    return PlannerIDMPolicy(
        obs_dim=4,
        act_dim=2,
        cmd_dim=3,
        history_len=3,
        pred_horizon=2,
        planner_d_model=8,
        planner_nhead=1,
        planner_num_layers=1,
        planner_dim_feedforward=16,
        planner_dropout=0.0,
        planner_use_action_history=planner_use_action_history,
        idm_d_model=8,
        idm_nhead=1,
        idm_encoder_num_layers=1,
        idm_decoder_num_layers=1,
        idm_dim_feedforward=16,
        idm_dropout=0.0,
    )


def _build_tiny_policy_with_idm_command() -> PlannerIDMPolicy:
    return PlannerIDMPolicy(
        obs_dim=4,
        act_dim=2,
        cmd_dim=3,
        history_len=3,
        pred_horizon=2,
        planner_d_model=8,
        planner_nhead=1,
        planner_num_layers=1,
        planner_dim_feedforward=16,
        planner_dropout=0.0,
        planner_use_action_history=True,
        idm_d_model=8,
        idm_nhead=1,
        idm_encoder_num_layers=1,
        idm_decoder_num_layers=1,
        idm_dim_feedforward=16,
        idm_dropout=0.0,
        idm_use_current_command_for_history=True,
    )


def test_locomotion_command_profile_stays_7d() -> None:
    commands = torch.zeros((3, 9), dtype=torch.float32)
    commands[:, 0] = 0.4
    commands[:, 2] = 0.2
    out = extract_current_command_torch(_fake_command_env(commands), profile=COMMAND_PROFILE_LOCOMOTION)

    assert out.shape == (3, 7)
    assert command_dim_for_profile(COMMAND_PROFILE_LOCOMOTION) == 7
    assert command_components_for_profile(COMMAND_PROFILE_LOCOMOTION) == (
        "command_lin_vel",
        "command_ang_vel",
        "sin_phase",
        "cos_phase",
    )


def test_planner_forward_shapes_with_optional_action_history() -> None:
    history_obs = torch.randn(2, 3, 4)
    current_command = torch.randn(2, 3)
    history_act = torch.randn(2, 3, 2)
    history_valid = torch.ones(2, 3, dtype=torch.bool)

    planner_no_act = Planner(
        obs_dim=4,
        act_dim=2,
        cmd_dim=3,
        history_len=3,
        pred_horizon=2,
        d_model=8,
        nhead=1,
        num_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        use_action_history=False,
    )
    pred_no_act = planner_no_act(history_obs, current_command, history_valid_mask=history_valid)
    assert pred_no_act.shape == (2, 2, 4)

    planner_with_act = Planner(
        obs_dim=4,
        act_dim=2,
        cmd_dim=3,
        history_len=3,
        pred_horizon=2,
        d_model=8,
        nhead=1,
        num_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        use_action_history=True,
    )
    pred_with_act = planner_with_act(
        history_obs,
        current_command,
        history_action=history_act,
        history_valid_mask=history_valid,
    )
    assert pred_with_act.shape == (2, 2, 4)


def test_idm_cross_attention_supports_history_and_future_lengths() -> None:
    model = IDM(
        obs_dim=4,
        act_dim=2,
        history_len=5,
        pred_horizon=2,
        d_model=8,
        nhead=1,
        encoder_num_layers=1,
        decoder_num_layers=1,
        dim_feedforward=16,
        dropout=0.0,
    )
    history_obs = torch.randn(3, 5, 4)
    history_act = torch.randn(3, 5, 2)
    future_obs = torch.randn(3, 2, 4)
    history_valid = torch.ones(3, 5, dtype=torch.bool)

    pred_actions = model(
        history_obs,
        history_act,
        None,
        future_obs,
        history_valid_mask=history_valid,
    )
    assert pred_actions.shape == (3, 2, 2)


def test_idm_cross_attention_requires_current_command_when_enabled() -> None:
    model = IDM(
        obs_dim=4,
        act_dim=2,
        cmd_dim=3,
        history_len=3,
        pred_horizon=2,
        d_model=8,
        nhead=1,
        encoder_num_layers=1,
        decoder_num_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        use_command_history=True,
    )
    history_obs = torch.randn(2, 3, 4)
    history_act = torch.randn(2, 3, 2)
    current_command = torch.randn(2, 3)
    future_obs = torch.randn(2, 2, 4)
    history_valid = torch.ones(2, 3, dtype=torch.bool)

    pred_actions = model(
        history_obs,
        history_act,
        current_command,
        future_obs,
        history_valid_mask=history_valid,
    )
    assert pred_actions.shape == (2, 2, 2)

    try:
        model(
            history_obs,
            history_act,
            None,
            future_obs,
            history_valid_mask=history_valid,
        )
    except ValueError as exc:
        assert "current_command" in str(exc)
    else:
        raise AssertionError("Expected IDM current-command path to require current_command")


def test_planner_idm_policy_forward_supports_teacher_override() -> None:
    model = _build_tiny_policy(planner_use_action_history=True)
    history_obs = torch.randn(2, 3, 4)
    history_act = torch.randn(2, 3, 2)
    current_command = torch.randn(2, 3)
    history_valid = torch.ones(2, 3, dtype=torch.bool)
    teacher_future_obs = torch.randn(2, 2, 4)

    pred_actions_teacher, pred_future_obs = model(
        history_obs,
        history_act,
        current_command,
        history_valid_mask=history_valid,
        return_obs=True,
        future_obs_override=teacher_future_obs,
    )
    pred_actions_planner, pred_future_obs_planner = model(
        history_obs,
        history_act,
        current_command,
        history_valid_mask=history_valid,
        return_obs=True,
    )

    assert pred_actions_teacher.shape == (2, 2, 2)
    assert pred_actions_planner.shape == (2, 2, 2)
    assert pred_future_obs.shape == (2, 2, 4)
    assert torch.allclose(pred_future_obs, pred_future_obs_planner)


def test_planner_idm_policy_forward_supports_idm_current_command_flag() -> None:
    model = _build_tiny_policy_with_idm_command()
    history_obs = torch.randn(2, 3, 4)
    history_act = torch.randn(2, 3, 2)
    current_command = torch.randn(2, 3)
    history_valid = torch.tensor([[False, True, True], [True, True, True]], dtype=torch.bool)

    pred_actions, pred_future_obs = model(
        history_obs,
        history_act,
        current_command,
        history_valid_mask=history_valid,
        return_obs=True,
    )

    assert pred_actions.shape == (2, 2, 2)
    assert pred_future_obs.shape == (2, 2, 4)


def test_planner_idm_arch_checkpoint_rebuild_preserves_architecture() -> None:
    model = _build_tiny_policy()
    cfg = {
        "policy_arch": "planner_idm",
        "history_len": 3,
        "pred_horizon": 2,
        "planner_d_model": 8,
        "planner_nhead": 1,
        "planner_num_layers": 1,
        "planner_dim_feedforward": 16,
        "planner_dropout": 0.0,
        "planner_use_learned_positional_encoding": False,
        "planner_use_action_history": False,
        "idm_use_current_command_for_history": False,
        "idm_d_model": 8,
        "idm_nhead": 1,
        "idm_encoder_num_layers": 1,
        "idm_decoder_num_layers": 1,
        "idm_dim_feedforward": 16,
        "idm_dropout": 0.0,
        "idm_use_learned_positional_encoding": False,
        "planner_predict_delta": False,
    }
    payload = {
        "cfg": cfg,
        "obs_dim": 4,
        "act_dim": 2,
        "cmd_dim": 3,
        "model_state_dict": model.state_dict(),
    }

    rebuilt, dims = _build_model_from_checkpoint(payload, device=torch.device("cpu"))

    assert isinstance(rebuilt, PlannerIDMPolicy)
    assert dims["obs_dim"] == 4
    assert dims["act_dim"] == 2
    assert dims["cmd_dim"] == 3


def test_planner_idm_lora_final_scope_injects() -> None:
    model = _build_tiny_policy()

    replaced, resolved_method, resolved_scope = configure_idm_finetune_method(
        model,
        finetune_method="lora",
        lora_r=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
        lora_target_scope="encoder_decoder_qkv",
    )

    assert replaced
    assert resolved_method == "lora"
    assert resolved_scope == "encoder_decoder_qkv"
    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    assert trainable_names
    assert all(name.endswith(".A.weight") or name.endswith(".B.weight") for name in trainable_names)
    assert all(name.startswith("idm.") for name in trainable_names)


def test_idm_lora_injection_only_hits_idm_backbone_and_merged_state_reloads() -> None:
    model = _build_tiny_policy(planner_use_action_history=False)
    base_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    replaced = inject_lora_into_idm_backbone(
        model,
        r=2,
        alpha=4.0,
        dropout=0.0,
    )
    assert replaced
    assert all(name.startswith("idm.history_encoder.") or name.startswith("idm.decoder.") for name in replaced)

    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    assert trainable_names
    assert all(name.endswith(".A.weight") or name.endswith(".B.weight") for name in trainable_names)
    assert not any(name.startswith("planner.") for name in trainable_names)

    merged_state = build_merged_state_dict(model)
    reloaded = _build_tiny_policy(planner_use_action_history=False)
    reloaded.load_compatible_state_dict(merged_state)
    for key, value in base_state.items():
        assert torch.allclose(merged_state[key], value)


def test_idm_lora_injection_decoder_only_skips_history_encoder() -> None:
    model = _build_tiny_policy(planner_use_action_history=False)

    replaced = inject_lora_into_idm_backbone(
        model,
        r=2,
        alpha=4.0,
        dropout=0.0,
        target_scope="decoder_only",
    )

    assert replaced
    assert all(name.startswith("idm.decoder.") for name in replaced)
    assert not any(name.startswith("idm.history_encoder.") for name in replaced)

    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    assert trainable_names
    assert all(name.endswith(".A.weight") or name.endswith(".B.weight") for name in trainable_names)


def test_idm_full_finetune_only_trains_idm_parameters() -> None:
    model = _build_tiny_policy(planner_use_action_history=False)

    replaced, resolved_method, resolved_scope = configure_idm_finetune_method(
        model,
        finetune_method="full",
        lora_r=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
        lora_target_scope="encoder_decoder",
    )

    assert replaced == ["idm"]
    assert resolved_method == "full"
    assert resolved_scope is None

    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    assert trainable_names
    assert all(name.startswith("idm.") for name in trainable_names)
    assert not any(name.startswith("planner.") for name in trainable_names)


def test_idm_lora_injection_decoder_only_qkv_adds_decoder_attention_qkv_only() -> None:
    model = _build_tiny_policy(planner_use_action_history=False)

    replaced = inject_lora_into_idm_backbone(
        model,
        r=2,
        alpha=4.0,
        dropout=0.0,
        target_scope="decoder_only_qkv",
    )

    assert replaced
    assert any(name.endswith(".in_proj_weight") for name in replaced)
    assert any(name.startswith("idm.decoder.") and name.endswith(".in_proj_weight") for name in replaced)
    assert not any(name.startswith("idm.history_encoder.") and name.endswith(".in_proj_weight") for name in replaced)


def test_idm_lora_injection_encoder_decoder_qkv_merges_history_encoder_attention() -> None:
    model = _build_tiny_policy(planner_use_action_history=False)
    base_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    replaced = inject_lora_into_idm_backbone(
        model,
        r=2,
        alpha=4.0,
        dropout=0.0,
        target_scope="encoder_decoder_qkv",
    )

    assert any(name.startswith("idm.decoder.") and name.endswith(".in_proj_weight") for name in replaced)
    assert any(name.startswith("idm.history_encoder.") and name.endswith(".in_proj_weight") for name in replaced)

    qkv_module = next(
        module
        for module_name, module in model.named_modules()
        if module_name == "idm.history_encoder.layers.0.self_attn" and hasattr(module, "B")
    )
    with torch.no_grad():
        qkv_module.B.weight.fill_(0.25)

    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    assert trainable_names
    assert all(name.endswith(".A.weight") or name.endswith(".B.weight") for name in trainable_names)

    merged_state = build_merged_state_dict(model)
    assert not torch.allclose(
        merged_state["idm.history_encoder.layers.0.self_attn.in_proj_weight"],
        base_state["idm.history_encoder.layers.0.self_attn.in_proj_weight"],
    )
    assert any(name.startswith("idm.decoder.") for name in trainable_names)
    assert any(name.startswith("idm.history_encoder.") for name in trainable_names)


def test_action_window_dataset_aligns_future_obs_and_actions() -> None:
    obs = np.arange(6 * 4, dtype=np.float32).reshape(6, 4)
    actions = (np.arange(6 * 2, dtype=np.float32).reshape(6, 2) + 10.0)
    current_command = np.arange(6 * 3, dtype=np.float32).reshape(6, 3) + 100.0
    traj = type("Traj", (), {"obs": obs, "actions": actions, "current_command": current_command, "source": "test"})()
    dataset = TrajectoryWindowDataset([traj], history_len=3, pred_horizon_k=2)

    sample0 = dataset[0]
    assert sample0["history_valid_mask"].tolist() == [False, False, True]
    assert np.allclose(sample0["history_obs"].numpy()[-1], obs[0])
    assert np.allclose(sample0["history_act"].numpy(), np.zeros((3, 2), dtype=np.float32))
    assert np.allclose(sample0["future_observations"].numpy(), obs[1:3])
    assert np.allclose(sample0["target_actions"].numpy(), actions[0:2])

    sample2 = dataset[2]
    assert sample2["history_valid_mask"].tolist() == [True, True, True]
    assert np.allclose(sample2["history_obs"].numpy(), obs[0:3])
    assert np.allclose(
        sample2["history_act"].numpy(),
        np.asarray(
            [
                [0.0, 0.0],
                actions[0],
                actions[1],
            ],
            dtype=np.float32,
        ),
    )
    assert np.allclose(sample2["future_observations"].numpy(), obs[3:5])
    assert np.allclose(sample2["target_actions"].numpy(), actions[2:4])


def test_action_window_dataset_returns_anchor_current_command() -> None:
    obs = np.zeros((5, 4), dtype=np.float32)
    actions = np.zeros((5, 2), dtype=np.float32)
    current_command = np.arange(15, dtype=np.float32).reshape(5, 3) + 1.0
    traj = type("Traj", (), {"obs": obs, "actions": actions, "current_command": current_command, "source": "test"})()
    dataset = TrajectoryWindowDataset(
        [traj],
        history_len=3,
        pred_horizon_k=1,
    )

    sample = dataset[2]
    assert np.allclose(sample["current_command"].numpy(), current_command[2].astype(np.float32, copy=False))

    sample0 = dataset[0]
    assert np.allclose(sample0["current_command"].numpy(), current_command[0].astype(np.float32, copy=False))
