from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any

import numpy as np

from holosoma.fada.common.backbone import HistoryPolicy
from holosoma.fada.common.compact_obs import extract_compact_obs
from holosoma.fada.common.current_command import (
    extract_current_command_torch,
    extract_tracking_command_torch,
)
from holosoma.managers.observation.terms import locomotion as obs_terms
from holosoma.utils.safe_torch_import import torch


def _to_sum_and_count(value: Any) -> tuple[float, int]:
    """Reduce a metric value to (sum, count) for aggregation."""
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


def _extract_current_command(env: Any) -> torch.Tensor:
    return extract_current_command_torch(env, profile=getattr(env, "transformer_command_profile", None))


def _extract_command_for_logging(env: Any) -> torch.Tensor:
    return extract_tracking_command_torch(env)


def _configure_eval_command_sampling(env: Any, *, interval_steps: int) -> tuple[Any | None, bool | None]:
    """Use command-manager callback sampling during eval.

    Converts the step interval to `locomotion_command_resampling_time` and lets the
    command callbacks handle updates, matching PPO/train-time behavior. Command term
    randomization stays enabled.
    """
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


@torch.no_grad()
def evaluate_policy(
    env: Any,
    model: HistoryPolicy,
    *,
    history_len: int,
    pred_horizon: int,
    act_dim: int,
    cmd_dim: int,
    predict_future_obs: bool,
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
    compact_obs_term_noise: dict[str, float] | None = None,
    force_scale: float | None = None,
    fixed_ee_force_left: tuple[float, float, float] | None = None,
    fixed_ee_force_right: tuple[float, float, float] | None = None,
    show_progress: bool = True,
) -> Path:
    """Run evaluation and save state_log.npz for plotting.

    Log-only: speed tracking errors and episode length are derived by downstream plot
    scripts.
    """
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
            "open_loop_deploy_steps cannot exceed pred_horizon: "
            f"{open_loop_deploy_steps} > {pred_horizon}"
        )
    if io_normalization and obs_norm_stats is None:
        raise ValueError(
            "io_normalization=True requires obs_norm_stats from checkpoint, but none was provided."
        )
    if io_normalization and action_norm_stats is None:
        raise ValueError(
            "io_normalization=True requires action_norm_stats from checkpoint, but none was provided."
        )
    if io_normalization and command_norm_stats is None:
        raise ValueError(
            "io_normalization=True requires command_norm_stats from checkpoint, but none was provided."
        )
    if predict_future_obs and not bool(getattr(model, "predict_future_obs", False)):
        raise ValueError(
            "predict_future_obs=True requested for eval, but checkpoint model was built without obs_head."
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
        # Reward metrics (same logic as train/rollout for comparability)
        "rew_tracking_lin_vel": np.zeros((num_episodes,), dtype=np.float32),
        "rew_tracking_ang_vel": np.zeros((num_episodes,), dtype=np.float32),
        "total_reward": np.zeros((num_episodes,), dtype=np.float32),
        "reconstruction_loss": np.full((num_episodes,), np.nan, dtype=np.float32),
        # Per-horizon observation reconstruction loss (h=1..pred_horizon), averaged over obs dims.
        "reconstruction_loss_per_horizon": np.full((num_episodes, pred_horizon), np.nan, dtype=np.float32),
        # Compatibility placeholders to avoid warnings in existing plot scripts.
        "predicted_target": np.zeros((num_episodes, num_envs, max_steps, 1), dtype=np.float32),
        "actual_target": np.zeros((num_episodes, num_envs, max_steps, 1), dtype=np.float32),
    }

    obs_mean: torch.Tensor | None = None
    obs_std: torch.Tensor | None = None
    if io_normalization and obs_norm_stats is not None:
        obs_mean = obs_norm_stats["obs_mean"].to(device=device, dtype=torch.float32).view(1, -1)
        obs_std = obs_norm_stats["obs_std"].to(device=device, dtype=torch.float32).view(1, -1)
    action_mean: torch.Tensor | None = None
    action_std: torch.Tensor | None = None
    if io_normalization and action_norm_stats is not None:
        action_mean = action_norm_stats["action_mean"].to(device=device, dtype=torch.float32).view(1, -1)
        action_std = action_norm_stats["action_std"].to(device=device, dtype=torch.float32).view(1, -1)
    command_mean: torch.Tensor | None = None
    command_std: torch.Tensor | None = None
    if io_normalization and command_norm_stats is not None:
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
            obs_recon_sq_sum_by_h = np.zeros((pred_horizon,), dtype=np.float64)
            obs_recon_count_by_h = np.zeros((pred_horizon,), dtype=np.int64)
            # target_step(int) -> [(horizon_idx, pred_obs_tensor[B, obs_dim]), ...]
            pending_obs_predictions: dict[int, list[tuple[int, torch.Tensor]]] = {}
            cached_plan_actions: torch.Tensor | None = None
            deploy_step_in_plan = int(open_loop_deploy_steps)  # force planning at first step
            episode_metric_sums: dict[str, float] = {}
            episode_metric_counts: dict[str, int] = {}
            episode_step_reward_sums: dict[str, float] = {}
            episode_termination_done_counts: dict[str, int] = {}
            episode_termination_primary_counts: dict[str, int] = {}
            episode_termination_overlap_events = 0

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
                current_command = _extract_current_command(env)
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

                did_replan = cached_plan_actions is None or deploy_step_in_plan >= int(open_loop_deploy_steps)
                if did_replan:
                    pred_next_obs: torch.Tensor | None = None
                    if predict_future_obs:
                        pred_actions, pred_next_obs = model(
                            model_hist_obs,
                            model_hist_act,
                            model_current_command,
                            history_valid_mask=hist_valid,
                            return_obs=True,
                        )
                    else:
                        pred_actions = model(
                            model_hist_obs,
                            model_hist_act,
                            model_current_command,
                            history_valid_mask=hist_valid,
                        )
                    if io_normalization:
                        assert action_mean is not None and action_std is not None
                        action_mean_seq = action_mean.view(1, 1, -1)
                        action_std_seq = action_std.view(1, 1, -1)
                        pred_actions = pred_actions * action_std_seq + action_mean_seq
                        if pred_next_obs is not None:
                            assert obs_mean is not None and obs_std is not None
                            obs_mean_seq = obs_mean.view(1, 1, -1)
                            obs_std_seq = obs_std.view(1, 1, -1)
                            pred_next_obs = pred_next_obs * obs_std_seq + obs_mean_seq
                    cached_plan_actions = pred_actions
                    deploy_step_in_plan = 0

                    # Register future-observation predictions against their absolute target step index.
                    if pred_next_obs is not None:
                        for h_idx in range(min(pred_horizon, int(pred_next_obs.shape[1]))):
                            target_step = int(step) + h_idx + 1
                            pending_obs_predictions.setdefault(target_step, []).append(
                                (h_idx, pred_next_obs[:, h_idx, :].detach())
                            )

                assert cached_plan_actions is not None
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
                active_envs = ~done_status
                _accumulate_step_reward_terms(env, active_envs, episode_step_reward_sums)
                newly_done = dones & active_envs
                valid_obs_envs = active_envs & (~dones)

                due_preds = pending_obs_predictions.pop(int(step) + 1, [])
                if due_preds and torch.any(valid_obs_envs):
                    target_next = obs_next
                    for h_idx, pred_h in due_preds:
                        diff = pred_h[valid_obs_envs] - target_next[valid_obs_envs]
                        obs_recon_sq_sum_by_h[h_idx] += float(torch.sum(diff * diff).item())
                        obs_recon_count_by_h[h_idx] += int(diff.numel())

                # First-episode-only accumulation for each env in this rollout.
                running_episode_return[active_envs] += rewards_f[active_envs]

                active_mask = active_envs.float()
                episode_steps[active_envs] += 1

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

                    episode_info = infos.get("episode")
                    done_ids = torch.nonzero(dones, as_tuple=False).squeeze(-1)
                    if done_ids.dim() == 0:
                        done_ids = done_ids.unsqueeze(0)
                    keep_mask_in_done = active_envs[done_ids]
                    if episode_info and torch.any(keep_mask_in_done):
                        filtered_episode: dict[str, Any] = {}
                        for key, value in episode_info.items():
                            if isinstance(value, torch.Tensor):
                                tensor = value
                                if tensor.ndim >= 1 and tensor.shape[0] == done_ids.numel():
                                    filtered_episode[key] = tensor[keep_mask_in_done]
                                else:
                                    filtered_episode[key] = tensor
                            else:
                                filtered_episode[key] = value
                        _accumulate_episode_metrics(filtered_episode, episode_metric_sums, episode_metric_counts)

                done_status = done_status | dones

                state_log["command_x"][epi, :, step] = (cmd_for_log[:, 0] * active_mask).detach().cpu().numpy()
                state_log["command_y"][epi, :, step] = (cmd_for_log[:, 1] * active_mask).detach().cpu().numpy()
                state_log["command_yaw"][epi, :, step] = (cmd_for_log[:, 2] * active_mask).detach().cpu().numpy()
                state_log["base_vel_x"][epi, :, step] = (base_lin[:, 0] * active_mask).detach().cpu().numpy()
                state_log["base_vel_y"][epi, :, step] = (base_lin[:, 1] * active_mask).detach().cpu().numpy()
                state_log["base_vel_yaw"][epi, :, step] = (base_ang[:, 2] * active_mask).detach().cpu().numpy()
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

            # Include first-episode ongoing (truncated) envs in means.
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
            # Per-episode means (same semantics as train Episode/rew_* and Train/mean_reward)
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
                state_log["total_reward"][epi] = float(
                    first_episode_returns[valid_first_mask].float().mean().item()
                )
            else:
                state_log["total_reward"][epi] = float("nan")
            valid_horizon_mask = obs_recon_count_by_h > 0
            if np.any(valid_horizon_mask):
                horizon_losses = np.full((pred_horizon,), np.nan, dtype=np.float32)
                horizon_losses[valid_horizon_mask] = (
                    obs_recon_sq_sum_by_h[valid_horizon_mask] / obs_recon_count_by_h[valid_horizon_mask]
                ).astype(np.float32)
                state_log["reconstruction_loss_per_horizon"][epi, :] = horizon_losses
                # Report a chunk-level loss: mean over forecast horizons (not only h=1).
                state_log["reconstruction_loss"][epi] = float(np.nanmean(horizon_losses))

            termination_done_counts_per_episode.append(episode_termination_done_counts)
            termination_primary_counts_per_episode.append(episode_termination_primary_counts)
            termination_overlap_per_episode.append(int(episode_termination_overlap_events))
    finally:
        if command_term is not None and prev_allow_eval_randomization is not None:
            command_term.allow_eval_randomization = bool(prev_allow_eval_randomization)

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

    state_log_path = output_dir / "state_log.npz"
    np.savez_compressed(state_log_path, **state_log)

    recon_arr = np.atleast_1d(state_log["reconstruction_loss"].squeeze())
    recon_mean = float(np.nanmean(recon_arr)) if recon_arr.size else float("nan")
    summary = {
        "num_episodes": num_episodes,
        "num_envs": num_envs,
        "max_steps": max_steps,
        "open_loop_deploy_steps": int(open_loop_deploy_steps),
        "compact_obs_term_scale": compact_obs_term_scale,
        "compact_obs_term_noise": compact_obs_term_noise,
        "compact_obs_add_noise": False,
        "history_len": history_len,
        "pred_horizon": pred_horizon,
        "act_dim": act_dim,
        "cmd_dim": cmd_dim,
        "predict_future_obs": bool(predict_future_obs),
        "command_input_name": "current_command",
        "command_input_semantics": "current_command_broadcast_to_history_tokens",
        "io_normalization": bool(io_normalization),
        "reconstruction_error_space": "scaled_compact_obs",
        "state_log_path": str(state_log_path),
        "reconstruction_loss_mean": recon_mean,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if was_training:
        model.train()
    if original_calculate_ee_forces is not None:
        env._calculate_ee_forces = original_calculate_ee_forces

    return state_log_path
