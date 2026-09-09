from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import numpy as np

from holosoma.fada.common.compact_obs import extract_compact_obs
from holosoma.fada.common.current_command import (
    extract_current_command_torch,
    extract_tracking_command_torch,
)
from holosoma.managers.observation.terms import locomotion as obs_terms
from holosoma.utils.safe_torch_import import torch


def _to_sum_and_count(value: Any) -> tuple[float, int]:
    if value is None:
        return 0.0, 0
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0, 0
        tensor = value.detach().float()
        return float(tensor.sum().item()), int(tensor.numel())
    if isinstance(value, (float, int)):
        return float(value), 1
    return 0.0, 0


def _accumulate_episode_metrics(
    source: dict[str, Any] | None,
    sums: dict[str, float],
    counts: dict[str, int],
) -> None:
    if not source:
        return
    for key, value in source.items():
        value_sum, value_count = _to_sum_and_count(value)
        if value_count <= 0:
            continue
        sums[key] = sums.get(key, 0.0) + float(value_sum)
        counts[key] = counts.get(key, 0) + int(value_count)


def _mean_metric(sums: dict[str, float], counts: dict[str, int], key: str) -> float:
    count = int(counts.get(key, 0))
    if count <= 0:
        return float("nan")
    return float(sums.get(key, 0.0) / count)


def _linear_tracking_metric(sums: dict[str, float], counts: dict[str, int]) -> float:
    for key in ("rew_tracking_lin_vel", "rew_track_lin_vel_xy"):
        value = _mean_metric(sums, counts, key)
        if not np.isnan(value):
            return value
    components = [_mean_metric(sums, counts, key) for key in ("rew_tracking_lin_vel_x", "rew_tracking_lin_vel_y")]
    valid = [v for v in components if not np.isnan(v)]
    return float(np.sum(valid)) if valid else 0.0


_REWARD_TERM_ALIASES = {
    "track_lin_vel_xy": "tracking_lin_vel",
    "track_ang_vel_z": "tracking_ang_vel",
}


def _accumulate_step_reward_terms(
    env: Any,
    active_envs: torch.Tensor,
    sums: dict[str, float],
) -> None:
    reward_manager = getattr(env, "reward_manager", None)
    step_terms = getattr(reward_manager, "last_scaled_term_rewards", None)
    if not step_terms:
        return
    for term_name, value in step_terms.items():
        if not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[0] != active_envs.shape[0]:
            continue
        term_sum = float(value[active_envs].detach().float().sum().item())
        sums[f"rew_{term_name}"] = sums.get(f"rew_{term_name}", 0.0) + term_sum
        alias_name = _REWARD_TERM_ALIASES.get(term_name)
        if alias_name is not None:
            sums[f"rew_{alias_name}"] = sums.get(f"rew_{alias_name}", 0.0) + term_sum


def _normalized_step_reward_metric(
    sums: dict[str, float],
    *,
    key: str,
    denom: float,
) -> float:
    if denom <= 0.0 or key not in sums:
        return float("nan")
    return float(sums[key] / denom)


def _linear_tracking_step_metric(sums: dict[str, float], *, denom: float) -> float:
    for key in ("rew_tracking_lin_vel", "rew_track_lin_vel_xy"):
        value = _normalized_step_reward_metric(sums, key=key, denom=denom)
        if not np.isnan(value):
            return value
    components = [
        _normalized_step_reward_metric(sums, key=key, denom=denom)
        for key in ("rew_tracking_lin_vel_x", "rew_tracking_lin_vel_y")
    ]
    valid = [v for v in components if not np.isnan(v)]
    return float(np.sum(valid)) if valid else 0.0


def _progress_range(
    total: int,
    *,
    desc: str,
    enabled: bool,
    leave: bool = False,
):
    if total <= 0:
        return range(0)
    if not enabled:
        return range(total)
    try:
        from tqdm.auto import tqdm  # type: ignore
    except Exception:
        return range(total)
    return tqdm(range(total), total=total, desc=desc, leave=leave, dynamic_ncols=True)


def _clip_actions(env: Any, actions: torch.Tensor) -> torch.Tensor:
    control_cfg = env.robot_config.control
    if bool(getattr(control_cfg, "clip_actions", False)):
        clip_val = float(control_cfg.action_clip_value)
        return torch.clamp(actions, -clip_val, clip_val)
    return actions


def _force_vector(value: tuple[float, float, float] | None, *, device: torch.device) -> torch.Tensor | None:
    if value is None:
        return None
    return torch.tensor([float(value[0]), float(value[1]), float(value[2])], device=device, dtype=torch.float32)


def _install_fixed_ee_force_override(
    env: Any,
    *,
    left: tuple[float, float, float] | None,
    right: tuple[float, float, float] | None,
    device: torch.device,
) -> Any | None:
    """Pin end-effector forces during eval.

    MuJoCo deployment can inject fixed wrist forces directly through simulator
    config. IsaacSim FAR eval normally samples forces through the force manager,
    so matching fixed MuJoCo stressors requires overriding that force calculation.
    """
    if left is None and right is None:
        return None
    required = (
        "_calculate_ee_forces",
        "_update_force_application_positions",
        "apply_force_tensor",
        "left_hand_idx",
        "right_hand_idx",
    )
    missing = [name for name in required if not hasattr(env, name)]
    if missing:
        raise AttributeError(f"fixed EE force requested but env is missing: {missing}")

    left_t = _force_vector(left, device=device)
    right_t = _force_vector(right, device=device)
    if left_t is None:
        left_t = torch.zeros(3, device=device, dtype=torch.float32)
    if right_t is None:
        right_t = torch.zeros(3, device=device, dtype=torch.float32)

    original = env._calculate_ee_forces

    def _calculate_fixed_ee_forces(self: Any) -> None:
        self._update_force_application_positions()
        left_force = left_t.view(1, 3).expand(int(self.num_envs), -1).clone()
        right_force = right_t.view(1, 3).expand(int(self.num_envs), -1).clone()
        self.left_ee_apply_force = left_force.clone()
        self.right_ee_apply_force = right_force.clone()
        self.apply_force_tensor.zero_()
        self.apply_force_tensor[:, self.left_hand_idx, :] = left_force
        self.apply_force_tensor[:, self.right_hand_idx, :] = right_force

    env._calculate_ee_forces = types.MethodType(_calculate_fixed_ee_forces, env)
    env._calculate_ee_forces()
    return original


def _extract_compact_obs(
    env: Any,
    *,
    compact_obs_term_scale: dict[str, float] | None,
) -> torch.Tensor:
    return extract_compact_obs(
        env,
        term_scale=compact_obs_term_scale,
    )


def _extract_current_command(env: Any, *, command_profile: str | None = None) -> torch.Tensor:
    return extract_current_command_torch(env, profile=command_profile)


def _extract_command_for_logging(env: Any) -> torch.Tensor:
    return extract_tracking_command_torch(env)


def _configure_eval_command_sampling(env: Any, *, interval_steps: int) -> tuple[Any | None, bool | None]:
    command_manager = getattr(env, "command_manager", None)
    if command_manager is None:
        return None, None

    if interval_steps > 0 and hasattr(env, "dt"):
        command_cfg = getattr(command_manager, "command_cfg", None)
        if command_cfg is not None:
            setattr(command_cfg, "locomotion_command_resampling_time", float(interval_steps) * float(env.dt))

    command_term = None
    prev_allow_eval_randomization: bool | None = None
    if hasattr(command_manager, "get_state"):
        command_term = command_manager.get_state("locomotion_command")
    if command_term is not None and hasattr(command_term, "allow_eval_randomization"):
        prev_allow_eval_randomization = bool(command_term.allow_eval_randomization)
        command_term.allow_eval_randomization = True
    return command_term, prev_allow_eval_randomization


def _make_idm_teacher_window(
    *,
    history_obs: torch.Tensor,
    history_act: torch.Tensor,
    current_command: torch.Tensor | None,
    history_valid_mask: torch.Tensor | None,
    eligible_mask: torch.Tensor,
    planner_future_obs: torch.Tensor | None = None,
) -> dict[str, Any]:
    return {
        "history_obs": history_obs.detach().clone(),
        "history_act": history_act.detach().clone(),
        "current_command": (None if current_command is None else current_command.detach().clone()),
        "history_valid_mask": (None if history_valid_mask is None else history_valid_mask.detach().clone()),
        "eligible_mask": eligible_mask.detach().clone(),
        "planner_future_obs": (None if planner_future_obs is None else planner_future_obs.detach().clone()),
        "future_obs": [],
        "future_actions": [],
    }


def _apply_future_obs_mask(
    future_obs: torch.Tensor,
    *,
    history_obs: torch.Tensor,
    mode: str,
    step: int | None,
    fill: str,
) -> torch.Tensor:
    """Mask planner future-observation tokens for eval-only step-importance ablations.

    ``step`` is 1-indexed. ``history_obs`` and ``future_obs`` must already be in the
    same model space (raw or normalized).
    """
    mode = str(mode).lower()
    fill = str(fill).lower()
    if mode in {"none", "full", ""}:
        return future_obs
    if future_obs.ndim != 3:
        raise ValueError("future_obs mask expects rank-3 tensor [B, K, D]")
    pred_horizon = int(future_obs.shape[1])
    if step is None:
        raise ValueError(f"future_obs_mask_mode={mode!r} requires --future-obs-mask-step")
    step_idx = int(step) - 1
    if step_idx < 0 or step_idx >= pred_horizon:
        raise ValueError(f"future_obs_mask_step must be in [1, {pred_horizon}], got {step}")

    visible = torch.zeros((pred_horizon,), dtype=torch.bool, device=future_obs.device)
    if mode == "drop":
        visible.fill_(True)
        visible[step_idx] = False
    elif mode == "only":
        visible[step_idx] = True
    elif mode == "prefix":
        visible[: step_idx + 1] = True
    else:
        raise ValueError(f"Unknown future_obs_mask_mode: {mode!r}")

    if torch.all(visible):
        return future_obs

    masked = future_obs.clone()
    current_obs = history_obs[:, -1:, :]
    if fill == "current":
        fill_values = current_obs.expand(-1, pred_horizon, -1)
    elif fill == "nearest":
        fill_values = current_obs.expand(-1, pred_horizon, -1).clone()
        visible_indices = torch.nonzero(visible, as_tuple=False).flatten()
        if visible_indices.numel() > 0:
            for idx in range(pred_horizon):
                nearest_pos = torch.argmin(torch.abs(visible_indices - idx))
                nearest_idx = int(visible_indices[nearest_pos].item())
                fill_values[:, idx, :] = future_obs[:, nearest_idx, :]
    else:
        raise ValueError(f"Unknown future_obs_mask_fill: {fill!r}")

    masked[:, ~visible, :] = fill_values[:, ~visible, :]
    return masked


def _update_idm_teacher_windows(
    *,
    pending_windows: list[dict[str, Any]],
    actual_future_obs: torch.Tensor,
    actual_action: torch.Tensor,
    dones: torch.Tensor,
    pred_horizon: int,
    model: Any,
    io_normalization: bool,
    action_mean: torch.Tensor | None,
    action_std: torch.Tensor | None,
    obs_mean: torch.Tensor | None = None,
    obs_std: torch.Tensor | None = None,
) -> tuple[list[dict[str, Any]], float, int, float, int, float, int]:
    """Returns (remaining_windows, idm_sq_sum, idm_count, planner_obs_sq_sum, planner_obs_count, action_sens_sq_sum, action_sens_count)."""
    sq_sum = 0.0
    value_count = 0
    planner_obs_sq_sum = 0.0
    planner_obs_count = 0
    action_sens_sq_sum = 0.0
    action_sens_count = 0
    remaining_windows: list[dict[str, Any]] = []

    for window in pending_windows:
        current_mask = window["eligible_mask"]
        if not torch.any(current_mask):
            continue

        window["future_obs"].append(actual_future_obs.detach().clone())
        window["future_actions"].append(actual_action.detach().clone())

        if len(window["future_obs"]) >= int(pred_horizon):
            teacher_future_obs = torch.stack(window["future_obs"], dim=1)
            target_actions = torch.stack(window["future_actions"], dim=1)
            teacher_pred_actions = model.predict_idm_actions(
                window["history_obs"],
                window["history_act"],
                window["current_command"],
                teacher_future_obs,
                history_valid_mask=window["history_valid_mask"],
            )
            if io_normalization:
                assert action_mean is not None and action_std is not None
                teacher_pred_actions = (
                    teacher_pred_actions * action_std.view(1, 1, -1) + action_mean.view(1, 1, -1)
                )
            diff = teacher_pred_actions[current_mask] - target_actions[current_mask]
            sq_sum += float(torch.sum(diff * diff).item())
            value_count += int(diff.numel())

            # Planner obs MSE and action sensitivity
            planner_future = window.get("planner_future_obs")
            if planner_future is not None:
                # Denormalize for fair comparison if needed
                actual_denorm = teacher_future_obs
                planner_denorm = planner_future
                if io_normalization and obs_mean is not None and obs_std is not None:
                    actual_denorm = teacher_future_obs * obs_std.view(1, 1, -1) + obs_mean.view(1, 1, -1)
                    planner_denorm = planner_future * obs_std.view(1, 1, -1) + obs_mean.view(1, 1, -1)
                obs_diff = planner_denorm[current_mask] - actual_denorm[current_mask]
                planner_obs_sq_sum += float(torch.sum(obs_diff * obs_diff).item())
                planner_obs_count += int(obs_diff.numel())

                # Action sensitivity: IDM(planner_future_obs) vs IDM(actual_future_obs)
                planner_pred_actions = model.predict_idm_actions(
                    window["history_obs"],
                    window["history_act"],
                    window["current_command"],
                    planner_future,
                    history_valid_mask=window["history_valid_mask"],
                )
                if io_normalization:
                    assert action_mean is not None and action_std is not None
                    planner_pred_actions = (
                        planner_pred_actions * action_std.view(1, 1, -1) + action_mean.view(1, 1, -1)
                    )
                sens_diff = planner_pred_actions[current_mask] - teacher_pred_actions[current_mask]
                action_sens_sq_sum += float(torch.sum(sens_diff * sens_diff).item())
                action_sens_count += int(sens_diff.numel())
            continue

        next_mask = current_mask & (~dones)
        if torch.any(next_mask):
            window["eligible_mask"] = next_mask
            remaining_windows.append(window)

    return remaining_windows, sq_sum, value_count, planner_obs_sq_sum, planner_obs_count, action_sens_sq_sum, action_sens_count


@torch.no_grad()
def evaluate_policy(
    env: Any,
    model: Any,
    *,
    history_len: int,
    pred_horizon: int,
    act_dim: int,
    cmd_dim: int,
    command_profile: str | None,
    predict_future_obs: bool,
    idm_use_current_command_for_history: bool,
    policy_arch: str = "planner_idm",
    obs_norm_stats: dict[str, torch.Tensor] | None,
    action_norm_stats: dict[str, torch.Tensor] | None,
    command_norm_stats: dict[str, torch.Tensor] | None,
    io_normalization: bool,
    num_episodes: int,
    max_steps: int,
    command_resample_interval: int,
    open_loop_deploy_steps: int,
    output_dir: Path,
    device: torch.device,
    compact_obs_term_scale: dict[str, float] | None,
    force_scale: float | None = None,
    fixed_ee_force_left: tuple[float, float, float] | None = None,
    fixed_ee_force_right: tuple[float, float, float] | None = None,
    show_progress: bool = True,
    future_obs_mask_mode: str = "none",
    future_obs_mask_step: int | None = None,
    future_obs_mask_fill: str = "current",
) -> Path:
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if open_loop_deploy_steps <= 0:
        raise ValueError("open_loop_deploy_steps must be positive")
    if history_len <= 0:
        raise ValueError("history_len must be positive")
    if pred_horizon <= 0:
        raise ValueError("pred_horizon must be positive")
    if open_loop_deploy_steps > pred_horizon:
        raise ValueError(
            f"open_loop_deploy_steps ({open_loop_deploy_steps}) cannot exceed pred_horizon ({pred_horizon})"
        )
    if policy_arch != "planner_idm":
        raise ValueError(f"Unknown policy_arch: {policy_arch!r}")
    if str(getattr(model, "idm_model_type", "transformer")) == "hybrid" and open_loop_deploy_steps != 1:
        raise ValueError(
            "idm_model_type='hybrid' outputs only the first action step; "
            f"open_loop_deploy_steps must be 1 (got {open_loop_deploy_steps})"
        )
    if io_normalization and obs_norm_stats is None:
        raise ValueError("io_normalization=True requires obs_norm_stats")
    if io_normalization and action_norm_stats is None:
        raise ValueError("io_normalization=True requires action_norm_stats")
    if io_normalization and command_norm_stats is None:
        raise ValueError("io_normalization=True requires command_norm_stats")
    future_obs_mask_mode = str(future_obs_mask_mode).lower()
    future_obs_mask_fill = str(future_obs_mask_fill).lower()
    if future_obs_mask_mode not in {"none", "full", "drop", "only", "prefix"}:
        raise ValueError(f"Unknown future_obs_mask_mode: {future_obs_mask_mode!r}")
    if future_obs_mask_fill not in {"current", "nearest"}:
        raise ValueError(f"Unknown future_obs_mask_fill: {future_obs_mask_fill!r}")
    if future_obs_mask_mode in {"drop", "only", "prefix"}:
        if future_obs_mask_step is None:
            raise ValueError(f"future_obs_mask_mode={future_obs_mask_mode!r} requires future_obs_mask_step")
        if int(future_obs_mask_step) < 1 or int(future_obs_mask_step) > int(pred_horizon):
            raise ValueError(
                f"future_obs_mask_step must be in [1, {int(pred_horizon)}], got {future_obs_mask_step}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    num_envs = int(env.num_envs)

    obs_dim = int(
        _extract_compact_obs(
            env,
            compact_obs_term_scale=compact_obs_term_scale,
        ).shape[1]
    )

    state_log = {
        "command_x": np.zeros((num_episodes, num_envs, max_steps), dtype=np.float32),
        "command_y": np.zeros((num_episodes, num_envs, max_steps), dtype=np.float32),
        "command_yaw": np.zeros((num_episodes, num_envs, max_steps), dtype=np.float32),
        "base_vel_x": np.zeros((num_episodes, num_envs, max_steps), dtype=np.float32),
        "base_vel_y": np.zeros((num_episodes, num_envs, max_steps), dtype=np.float32),
        "base_vel_yaw": np.zeros((num_episodes, num_envs, max_steps), dtype=np.float32),
        "env_done_status": np.zeros((num_episodes, num_envs, max_steps), dtype=np.bool_),
        "per_episode_step": np.zeros((num_episodes, num_envs, 1), dtype=np.int64),
        "rew_tracking_lin_vel": np.zeros((num_episodes,), dtype=np.float32),
        "rew_tracking_ang_vel": np.zeros((num_episodes,), dtype=np.float32),
        "total_reward": np.zeros((num_episodes,), dtype=np.float32),
        "idm_inverse_action_mse": np.full((num_episodes,), np.nan, dtype=np.float32),
        "idm_inverse_action_rmse": np.full((num_episodes,), np.nan, dtype=np.float32),
        "planner_future_obs_mse": np.full((num_episodes,), np.nan, dtype=np.float32),
        "action_sensitivity_mse": np.full((num_episodes,), np.nan, dtype=np.float32),
        "future_obs_mask_mode": np.asarray([future_obs_mask_mode]),
        "future_obs_mask_step": np.asarray([-1 if future_obs_mask_step is None else int(future_obs_mask_step)], dtype=np.int64),
        "future_obs_mask_fill": np.asarray([future_obs_mask_fill]),
    }

    obs_mean: torch.Tensor | None = None
    obs_std: torch.Tensor | None = None
    action_mean: torch.Tensor | None = None
    action_std: torch.Tensor | None = None
    command_mean: torch.Tensor | None = None
    command_std: torch.Tensor | None = None
    if io_normalization:
        obs_mean = obs_norm_stats["obs_mean"].to(device=device, dtype=torch.float32).view(1, -1)
        obs_std = obs_norm_stats["obs_std"].to(device=device, dtype=torch.float32).view(1, -1)
        action_mean = action_norm_stats["action_mean"].to(device=device, dtype=torch.float32).view(1, -1)
        action_std = action_norm_stats["action_std"].to(device=device, dtype=torch.float32).view(1, -1)
        command_mean = command_norm_stats["command_mean"].to(device=device, dtype=torch.float32).view(1, -1)
        command_std = command_norm_stats["command_std"].to(device=device, dtype=torch.float32).view(1, -1)

    was_training = model.training
    model.eval()
    command_term = None
    prev_allow_eval_randomization: bool | None = None
    original_calculate_ee_forces = _install_fixed_ee_force_override(
        env,
        left=fixed_ee_force_left,
        right=fixed_ee_force_right,
        device=device,
    )
    termination_manager = getattr(env, "termination_manager", None)
    termination_term_names = list(getattr(termination_manager, "_term_names", [])) if termination_manager is not None else []
    termination_done_counts_per_episode: list[dict[str, int]] = []
    termination_primary_counts_per_episode: list[dict[str, int]] = []
    termination_overlap_per_episode: list[int] = []

    try:
        command_term, prev_allow_eval_randomization = _configure_eval_command_sampling(
            env,
            interval_steps=int(command_resample_interval),
        )
        for epi in _progress_range(
            num_episodes,
            desc="Eval Episodes",
            enabled=bool(show_progress),
            leave=True,
        ):
            _ = env.reset_all()
            if force_scale is not None:
                if not hasattr(env, "apply_force_scale"):
                    raise AttributeError("force_scale was requested but env has no apply_force_scale attribute")
                env.apply_force_scale.fill_(float(force_scale))

            hist_obs = torch.zeros((num_envs, history_len, obs_dim), device=device, dtype=torch.float32)
            hist_act = torch.zeros((num_envs, history_len, act_dim), device=device, dtype=torch.float32)
            prev_action = torch.zeros((num_envs, act_dim), device=device, dtype=torch.float32)
            hist_valid = torch.zeros((num_envs, history_len), dtype=torch.bool, device=device)

            done_status = torch.zeros((num_envs,), dtype=torch.bool, device=device)
            episode_steps = torch.zeros((num_envs,), dtype=torch.long, device=device)
            running_episode_return = torch.zeros((num_envs,), dtype=torch.float32, device=device)
            first_done_recorded = torch.zeros((num_envs,), dtype=torch.bool, device=device)
            first_episode_returns = torch.zeros((num_envs,), dtype=torch.float32, device=device)
            first_episode_lengths = torch.zeros((num_envs,), dtype=torch.long, device=device)
            episode_metric_sums: dict[str, float] = {}
            episode_metric_counts: dict[str, int] = {}
            episode_step_reward_sums: dict[str, float] = {}
            episode_termination_done_counts: dict[str, int] = {}
            episode_termination_primary_counts: dict[str, int] = {}
            episode_termination_overlap_events = 0

            idm_inverse_sq_sum = 0.0
            idm_inverse_count = 0
            planner_obs_sq_sum = 0.0
            planner_obs_count = 0
            action_sens_sq_sum = 0.0
            action_sens_count = 0
            pending_idm_windows: list[dict[str, Any]] = []

            cached_plan_actions: torch.Tensor | None = None
            deploy_step_in_plan = int(open_loop_deploy_steps)

            for step in _progress_range(
                max_steps,
                desc=f"Eval Steps [{int(epi) + 1}/{num_episodes}]",
                enabled=bool(show_progress),
                leave=False,
            ):
                obs_curr = _extract_compact_obs(
                    env,
                    compact_obs_term_scale=compact_obs_term_scale,
                )
                current_command = _extract_current_command(env, command_profile=command_profile)
                cmd_for_log = _extract_command_for_logging(env)
                base_lin = obs_terms.get_base_lin_vel(env)
                base_ang = obs_terms.get_base_ang_vel(env)

                hist_obs = torch.roll(hist_obs, shifts=-1, dims=1)
                hist_act = torch.roll(hist_act, shifts=-1, dims=1)
                hist_valid = torch.roll(hist_valid, shifts=-1, dims=1)
                hist_obs[:, -1, :] = obs_curr
                hist_act[:, -1, :] = prev_action
                hist_valid[:, -1] = True

                model_hist_obs = hist_obs
                model_hist_act = hist_act
                model_current_command = current_command
                if io_normalization:
                    assert obs_mean is not None and obs_std is not None
                    assert action_mean is not None and action_std is not None
                    assert command_mean is not None and command_std is not None
                    obs_mean_seq = obs_mean.view(1, 1, -1)
                    obs_std_seq = obs_std.view(1, 1, -1)
                    action_mean_seq = action_mean.view(1, 1, -1)
                    action_std_seq = action_std.view(1, 1, -1)
                    command_mean_seq = command_mean.view(1, -1)
                    command_std_seq = command_std.view(1, -1)
                    model_hist_obs = (hist_obs - obs_mean_seq) / obs_std_seq
                    model_hist_act = (hist_act - action_mean_seq) / action_std_seq
                    model_current_command = (current_command - command_mean_seq) / command_std_seq

                teacher_hist_obs = model_hist_obs.detach().clone()
                teacher_hist_act = model_hist_act.detach().clone()
                teacher_current_command = model._resolve_idm_current_command(
                    model_current_command,
                    idm_use_current_command_for_history=bool(idm_use_current_command_for_history),
                )
                teacher_hist_valid = hist_valid.detach().clone()

                did_replan = cached_plan_actions is None or deploy_step_in_plan >= int(open_loop_deploy_steps)
                cached_planner_future_obs: torch.Tensor | None = None
                if did_replan:
                    pred_actions, planner_future_obs_raw = model(
                        model_hist_obs,
                        model_hist_act,
                        model_current_command,
                        history_valid_mask=hist_valid,
                        return_obs=True,
                    )
                    # planner_future_obs_raw is in the model's internal space (normalized
                    # if io_normalization, raw otherwise) — same space as teacher_future_obs
                    # in _update_idm_teacher_windows, so they're directly comparable.
                    if future_obs_mask_mode in {"drop", "only", "prefix"}:
                        planner_future_obs_raw = _apply_future_obs_mask(
                            planner_future_obs_raw,
                            history_obs=model_hist_obs,
                            mode=future_obs_mask_mode,
                            step=future_obs_mask_step,
                            fill=future_obs_mask_fill,
                        )
                        idm_current_command = model._resolve_idm_current_command(
                            model_current_command,
                            idm_use_current_command_for_history=bool(idm_use_current_command_for_history),
                        )
                        pred_actions = model.predict_idm_actions(
                            model_hist_obs,
                            model_hist_act,
                            idm_current_command,
                            planner_future_obs_raw,
                            history_valid_mask=hist_valid,
                        )
                    cached_planner_future_obs = planner_future_obs_raw.detach().clone()
                    if io_normalization:
                        assert action_mean is not None and action_std is not None
                        action_mean_seq = action_mean.view(1, 1, -1)
                        action_std_seq = action_std.view(1, 1, -1)
                        pred_actions = pred_actions * action_std_seq + action_mean_seq
                    cached_plan_actions = pred_actions
                    deploy_step_in_plan = 0

                assert cached_plan_actions is not None
                active_envs = ~done_status
                pending_idm_windows.append(
                    _make_idm_teacher_window(
                        history_obs=teacher_hist_obs,
                        history_act=teacher_hist_act,
                        current_command=teacher_current_command,
                        history_valid_mask=teacher_hist_valid,
                        eligible_mask=active_envs,
                        planner_future_obs=cached_planner_future_obs,
                    )
                )
                action_exec = _clip_actions(env, cached_plan_actions[:, deploy_step_in_plan, :])
                action_exec = torch.where(
                    done_status.unsqueeze(-1),
                    torch.zeros_like(action_exec),
                    action_exec,
                )
                deploy_step_in_plan += 1

                _, rewards, dones, infos = env.step({"actions": action_exec})
                obs_next = _extract_compact_obs(
                    env,
                    compact_obs_term_scale=compact_obs_term_scale,
                )
                dones = dones.bool()
                rewards_f = rewards.float()
                teacher_obs_next = obs_next
                if io_normalization:
                    assert obs_mean is not None and obs_std is not None
                    teacher_obs_next = (obs_next - obs_mean.view(1, -1)) / obs_std.view(1, -1)
                (
                    pending_idm_windows,
                    idm_sq_delta, idm_count_delta,
                    planner_obs_sq_delta, planner_obs_count_delta,
                    action_sens_sq_delta, action_sens_count_delta,
                ) = _update_idm_teacher_windows(
                    pending_windows=pending_idm_windows,
                    actual_future_obs=teacher_obs_next,
                    actual_action=action_exec,
                    dones=dones,
                    pred_horizon=pred_horizon,
                    model=model,
                    io_normalization=io_normalization,
                    action_mean=action_mean,
                    action_std=action_std,
                    obs_mean=obs_mean,
                    obs_std=obs_std,
                )
                idm_inverse_sq_sum += float(idm_sq_delta)
                idm_inverse_count += int(idm_count_delta)
                planner_obs_sq_sum += float(planner_obs_sq_delta)
                planner_obs_count += int(planner_obs_count_delta)
                action_sens_sq_sum += float(action_sens_sq_delta)
                action_sens_count += int(action_sens_count_delta)

                running_episode_return[active_envs] += rewards_f[active_envs]
                _accumulate_step_reward_terms(env, active_envs, episode_step_reward_sums)
                episode_steps[active_envs] += 1
                newly_done = dones & active_envs
                if torch.any(newly_done):
                    new_done_ids = torch.nonzero(newly_done, as_tuple=False).squeeze(-1)
                    if new_done_ids.dim() == 0:
                        new_done_ids = new_done_ids.unsqueeze(0)
                    first_episode_returns[new_done_ids] = running_episode_return[new_done_ids]
                    first_episode_lengths[new_done_ids] = episode_steps[new_done_ids]
                    first_done_recorded[new_done_ids] = True
                    summarize = getattr(termination_manager, "summarize_newly_done", None)
                    if callable(summarize):
                        term_summary = summarize(newly_done)
                        for term_name, count in term_summary.get("term_done_counts", {}).items():
                            episode_termination_done_counts[term_name] = (
                                episode_termination_done_counts.get(term_name, 0) + int(count)
                            )
                        for term_name, count in term_summary.get("primary_done_counts", {}).items():
                            episode_termination_primary_counts[term_name] = (
                                episode_termination_primary_counts.get(term_name, 0) + int(count)
                            )
                        episode_termination_overlap_events += int(term_summary.get("overlap_events", 0))
                    episode_info = infos.get("episode") if isinstance(infos, dict) else None
                    keep_mask_in_done = active_envs[new_done_ids]
                    if episode_info and torch.any(keep_mask_in_done):
                        filtered_episode: dict[str, Any] = {}
                        for key, value in episode_info.items():
                            if isinstance(value, torch.Tensor):
                                tensor = value
                                if tensor.ndim >= 1 and tensor.shape[0] == new_done_ids.numel():
                                    filtered_episode[key] = tensor[keep_mask_in_done]
                                else:
                                    filtered_episode[key] = tensor
                            else:
                                filtered_episode[key] = value
                        _accumulate_episode_metrics(filtered_episode, episode_metric_sums, episode_metric_counts)

                done_status = done_status | dones

                state_log["command_x"][epi, :, step] = (cmd_for_log[:, 0] * active_envs.float()).detach().cpu().numpy()
                state_log["command_y"][epi, :, step] = (cmd_for_log[:, 1] * active_envs.float()).detach().cpu().numpy()
                state_log["command_yaw"][epi, :, step] = (cmd_for_log[:, 2] * active_envs.float()).detach().cpu().numpy()
                state_log["base_vel_x"][epi, :, step] = (base_lin[:, 0] * active_envs.float()).detach().cpu().numpy()
                state_log["base_vel_y"][epi, :, step] = (base_lin[:, 1] * active_envs.float()).detach().cpu().numpy()
                state_log["base_vel_yaw"][epi, :, step] = (base_ang[:, 2] * active_envs.float()).detach().cpu().numpy()
                state_log["env_done_status"][epi, :, step] = done_status.detach().cpu().numpy()

                prev_action = action_exec.detach()
                if torch.any(dones):
                    prev_action[dones] = 0.0
                    hist_obs[dones] = 0.0
                    hist_act[dones] = 0.0
                    hist_valid[dones] = False
                    running_episode_return[dones] = 0.0

                if done_status.all():
                    break

            state_log["per_episode_step"][epi, :, 0] = episode_steps.detach().cpu().numpy()

            ongoing_mask = (~first_done_recorded) & (episode_steps > 0)
            ongoing_env_ids = torch.nonzero(ongoing_mask, as_tuple=False).squeeze(-1)
            if ongoing_env_ids.dim() == 0:
                ongoing_env_ids = ongoing_env_ids.unsqueeze(0)
            if ongoing_env_ids.numel() > 0:
                first_episode_returns[ongoing_env_ids] = running_episode_return[ongoing_env_ids]
                first_episode_lengths[ongoing_env_ids] = episode_steps[ongoing_env_ids]
            if ongoing_env_ids.numel() > 0 and getattr(env, "reward_manager", None) is not None:
                ongoing_rates = env.reward_manager.get_episode_rates(ongoing_env_ids)
                _accumulate_episode_metrics(ongoing_rates, episode_metric_sums, episode_metric_counts)

            valid_first_mask = first_episode_lengths > 0
            if torch.any(valid_first_mask):
                reward_metric_denom = float(valid_first_mask.float().sum().item()) * max(
                    float(getattr(env, "max_episode_length_s", 0.0)),
                    float(getattr(env, "dt", 1.0)),
                )
                state_log["rew_tracking_lin_vel"][epi] = _linear_tracking_step_metric(
                    episode_step_reward_sums,
                    denom=reward_metric_denom,
                )
                state_log["rew_tracking_ang_vel"][epi] = _normalized_step_reward_metric(
                    episode_step_reward_sums,
                    key="rew_tracking_ang_vel",
                    denom=reward_metric_denom,
                )
                state_log["total_reward"][epi] = float(first_episode_returns[valid_first_mask].float().mean().item())
            else:
                state_log["total_reward"][epi] = float("nan")

            if idm_inverse_count > 0:
                mse = float(idm_inverse_sq_sum / float(idm_inverse_count))
                state_log["idm_inverse_action_mse"][epi] = mse
                state_log["idm_inverse_action_rmse"][epi] = float(np.sqrt(mse))
            if planner_obs_count > 0:
                state_log["planner_future_obs_mse"][epi] = float(planner_obs_sq_sum / float(planner_obs_count))
            if action_sens_count > 0:
                state_log["action_sensitivity_mse"][epi] = float(action_sens_sq_sum / float(action_sens_count))

            termination_done_counts_per_episode.append(episode_termination_done_counts)
            termination_primary_counts_per_episode.append(episode_termination_primary_counts)
            termination_overlap_per_episode.append(int(episode_termination_overlap_events))

        state_log_path = output_dir / "state_log.npz"
        all_done_terms = set(termination_term_names)
        for per_epi in termination_done_counts_per_episode:
            all_done_terms.update(per_epi.keys())
        for term_name in sorted(all_done_terms):
            arr = np.zeros((num_episodes,), dtype=np.float32)
            for epi, per_epi in enumerate(termination_done_counts_per_episode):
                arr[epi] = float(per_epi.get(term_name, 0))
            state_log[f"termination_done_count__{term_name}"] = arr

        all_primary_terms: set[str] = set()
        for per_epi in termination_primary_counts_per_episode:
            all_primary_terms.update(per_epi.keys())
        for term_name in sorted(all_primary_terms):
            arr = np.zeros((num_episodes,), dtype=np.float32)
            for epi, per_epi in enumerate(termination_primary_counts_per_episode):
                arr[epi] = float(per_epi.get(term_name, 0))
            state_log[f"termination_done_primary_count__{term_name}"] = arr

        if termination_overlap_per_episode:
            state_log["termination_overlap_events"] = np.asarray(termination_overlap_per_episode, dtype=np.float32)
        if termination_primary_counts_per_episode:
            totals = np.zeros((num_episodes,), dtype=np.float32)
            for epi, per_epi in enumerate(termination_primary_counts_per_episode):
                totals[epi] = float(sum(per_epi.values()))
            state_log["termination_done_total"] = totals
        if force_scale is not None:
            state_log["force_scale"] = np.asarray([float(force_scale)], dtype=np.float32)
        if fixed_ee_force_left is not None:
            state_log["fixed_ee_force_left"] = np.asarray(fixed_ee_force_left, dtype=np.float32)
        if fixed_ee_force_right is not None:
            state_log["fixed_ee_force_right"] = np.asarray(fixed_ee_force_right, dtype=np.float32)
        np.savez_compressed(state_log_path, **state_log)
        return state_log_path
    finally:
        if original_calculate_ee_forces is not None:
            env._calculate_ee_forces = original_calculate_ee_forces
        if command_term is not None and prev_allow_eval_randomization is not None:
            command_term.allow_eval_randomization = bool(prev_allow_eval_randomization)
        if was_training:
            model.train()


__all__ = ["evaluate_policy"]
