from __future__ import annotations

import copy
import itertools
import random
from pathlib import Path
from typing import Any

import numpy as np

from holosoma.fada.common.dataset import (
    ReplayBuffer,
    ReplayCacheLegacyMetadataMissingError,
    ReplayCacheMetadataMismatchError,
)
from holosoma.fada.planner_idm.offline_data_utils import (
    OfflineRolloutData,
    OracleRelabeledChunks,
    flatten_offline_rollout_to_replay_batch,
    flatten_offline_rollout_with_oracle_chunks,
    list_model_checkpoints,
    select_optimal_checkpoint,
    select_suboptimal_checkpoints,
    target_suboptimal_steps,
)
from holosoma.fada.planner_idm.trainer import (
    _accumulate_prefixed_metrics,
    _clip_actions,
    _extract_command_for_logging,
    _extract_compact_obs,
    _extract_current_command,
    _finalize_prefixed_metrics,
    _nested_to_device,
    _progress_range,
    _set_command_resampling_interval_from_steps,
)
from holosoma.managers.observation.terms import locomotion as obs_terms
from holosoma.utils.safe_torch_import import F, optim, torch


class _RolloutMixin:
    def _extract_compact_obs_for_env(
        self,
        env: Any,
    ) -> torch.Tensor:
        return _extract_compact_obs(
            env,
            compact_obs_term_scale=self.compact_obs_term_scale,
        )

    @staticmethod
    def _copy_nested_obs(dst: Any, src: Any) -> Any:
        if isinstance(dst, torch.Tensor) and isinstance(src, torch.Tensor):
            if dst.shape == src.shape:
                dst.copy_(src)
            return dst
        if isinstance(dst, dict) and isinstance(src, dict):
            for key, src_val in src.items():
                if key not in dst:
                    dst[key] = copy.deepcopy(src_val)
                    continue
                dst[key] = _RolloutMixin._copy_nested_obs(dst[key], src_val)
            return dst
        return copy.deepcopy(src)

    def _clone_nested(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().clone()
        if isinstance(value, dict):
            return {k: self._clone_nested(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._clone_nested(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._clone_nested(v) for v in value)
        return copy.deepcopy(value)

    def _capture_rng_snapshot(self) -> dict[str, Any]:
        snapshot: dict[str, Any] = {
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_cpu_rng_state": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            snapshot["torch_cuda_rng_states"] = [state.detach().clone() for state in torch.cuda.get_rng_state_all()]
        return snapshot

    def _restore_rng_snapshot(self, snapshot: dict[str, Any]) -> None:
        if "python_random_state" in snapshot:
            random.setstate(snapshot["python_random_state"])
        if "numpy_random_state" in snapshot:
            np.random.set_state(snapshot["numpy_random_state"])
        if "torch_cpu_rng_state" in snapshot:
            torch.random.set_rng_state(snapshot["torch_cpu_rng_state"])
        if torch.cuda.is_available() and "torch_cuda_rng_states" in snapshot:
            torch.cuda.set_rng_state_all(snapshot["torch_cuda_rng_states"])

    @staticmethod
    def _copy_tensor_from_snapshot(obj: Any, attr_name: str, snapshot: dict[str, Any]) -> None:
        if attr_name not in snapshot:
            return
        if not hasattr(obj, attr_name):
            return
        dst_val = getattr(obj, attr_name)
        src_val = snapshot[attr_name]
        if isinstance(dst_val, torch.Tensor) and isinstance(src_val, torch.Tensor) and dst_val.shape == src_val.shape:
            dst_val.copy_(src_val.to(device=dst_val.device))

    @staticmethod
    def _clone_root_states_tensor(root_states: Any) -> torch.Tensor:
        """Clone root states from tensor or proxy wrappers."""
        if isinstance(root_states, torch.Tensor):
            return root_states.detach().clone()
        if hasattr(root_states, "clone"):
            cloned = root_states.clone()
            if isinstance(cloned, torch.Tensor):
                return cloned.detach().clone()
        raise TypeError(f"Unsupported all_root_states type for snapshot: {type(root_states)}")

    @staticmethod
    def _restore_root_states_tensor(root_states: Any, value: torch.Tensor, env_device: torch.device) -> None:
        """Restore root states into tensor or proxy wrappers."""
        root_states_device = getattr(root_states, "device", env_device)
        value = value.to(device=root_states_device)
        if isinstance(root_states, torch.Tensor):
            root_states.copy_(value)
            return
        if hasattr(root_states, "__setitem__"):
            all_actor_ids = torch.arange(value.shape[0], device=root_states_device, dtype=torch.long)
            root_states[all_actor_ids, :13] = value
            return
        raise TypeError(f"Unsupported all_root_states type for restore: {type(root_states)}")

    def _capture_env_snapshot(self, env: Any) -> dict[str, Any]:
        snapshot: dict[str, Any] = {}
        simulator = env.simulator
        snapshot["sim_all_root_states"] = self._clone_root_states_tensor(simulator.all_root_states)
        snapshot["sim_dof_state"] = simulator.dof_state.detach().clone()
        if hasattr(simulator, "contact_forces_history") and isinstance(simulator.contact_forces_history, torch.Tensor):
            snapshot["sim_contact_forces_history"] = simulator.contact_forces_history.detach().clone()
        if hasattr(simulator, "commands") and isinstance(simulator.commands, torch.Tensor):
            snapshot["sim_commands"] = simulator.commands.detach().clone()

        for attr in (
            "episode_length_buf",
            "reset_buf",
            "time_out_buf",
            "rew_buf",
            "_pending_episode_lengths",
            "_pending_episode_update_mask",
            "base_quat",
            "need_to_refresh_envs",
            "push_robot_vel_buf",
            "record_push_robot_vel_buf",
            "action_delay_idx",
            "_max_push_vel",
        ):
            if hasattr(env, attr):
                val = getattr(env, attr)
                if isinstance(val, torch.Tensor):
                    snapshot[attr] = val.detach().clone()

        for attr in (
            "common_step_counter",
            "_randomize_push_robots",
            "_randomize_ctrl_delay",
            "_pending_torque_rfi",
            "is_evaluating",
        ):
            if hasattr(env, attr):
                snapshot[attr] = copy.deepcopy(getattr(env, attr))

        snapshot["command_tensor"] = env.command_manager.commands.detach().clone()
        command_cfg = getattr(env.command_manager, "command_cfg", None)
        if command_cfg is not None and hasattr(command_cfg, "locomotion_command_resampling_time"):
            snapshot["command_resample_time"] = float(getattr(command_cfg, "locomotion_command_resampling_time"))
        command_term = env.command_manager.get_state("locomotion_command")
        if command_term is not None and hasattr(command_term, "allow_eval_randomization"):
            snapshot["allow_eval_randomization"] = bool(command_term.allow_eval_randomization)
        gait_state = env.command_manager.get_state("locomotion_gait")
        if gait_state is not None:
            gait_payload: dict[str, Any] = {}
            for attr in ("phase_offset", "phase", "gait_freq", "phase_dt"):
                if hasattr(gait_state, attr):
                    val = getattr(gait_state, attr)
                    if isinstance(val, torch.Tensor):
                        gait_payload[attr] = val.detach().clone()
            if hasattr(gait_state, "mean_gait_freq"):
                gait_payload["mean_gait_freq"] = float(gait_state.mean_gait_freq)
            snapshot["gait_state"] = gait_payload

        push_state = env.randomization_manager.get_state("push_randomizer_state")
        if push_state is not None:
            push_payload: dict[str, Any] = {}
            for attr in ("push_interval_s", "push_robot_counter", "push_robot_plot_counter", "_max_push_vel_tensor"):
                if hasattr(push_state, attr):
                    val = getattr(push_state, attr)
                    if isinstance(val, torch.Tensor):
                        push_payload[attr] = val.detach().clone()
            for attr in ("enabled", "push_interval_range"):
                if hasattr(push_state, attr):
                    push_payload[attr] = copy.deepcopy(getattr(push_state, attr))
            snapshot["push_state"] = push_payload

        actuator_state = env.randomization_manager.get_state("actuator_randomizer_state")
        if actuator_state is not None:
            actuator_payload: dict[str, Any] = {}
            for attr in ("kp_scale", "kd_scale", "rfi_lim_scale"):
                if hasattr(actuator_state, attr):
                    val = getattr(actuator_state, attr)
                    if isinstance(val, torch.Tensor):
                        actuator_payload[attr] = val.detach().clone()
            for attr in ("enable_pd_gain", "enable_rfi_lim", "kp_range", "kd_range", "rfi_lim_range", "rfi_lim"):
                if hasattr(actuator_state, attr):
                    actuator_payload[attr] = copy.deepcopy(getattr(actuator_state, attr))
            snapshot["actuator_state"] = actuator_payload

        if hasattr(env.action_manager, "_action") and isinstance(env.action_manager._action, torch.Tensor):
            snapshot["action_manager_action"] = env.action_manager._action.detach().clone()
        if hasattr(env.action_manager, "_prev_action") and isinstance(env.action_manager._prev_action, torch.Tensor):
            snapshot["action_manager_prev_action"] = env.action_manager._prev_action.detach().clone()

        joint_term = env.action_manager.get_term("joint_control")
        joint_payload: dict[str, Any] = {}
        for attr in (
            "_raw_actions",
            "_processed_actions",
            "_actions_after_delay",
            "torques",
            "_prev_dof_vel",
            "action_queue",
            "_kp_scale",
            "_kd_scale",
            "_rfi_lim_scale",
        ):
            if hasattr(joint_term, attr):
                val = getattr(joint_term, attr)
                if isinstance(val, torch.Tensor):
                    joint_payload[attr] = val.detach().clone()
        for attr in ("_randomize_torque_rfi", "_rfi_lim"):
            if hasattr(joint_term, attr):
                joint_payload[attr] = copy.deepcopy(getattr(joint_term, attr))
        snapshot["joint_term"] = joint_payload

        history_snapshot: dict[str, dict[str, list[Any]]] = {}
        for group_name, group_buffers in env.observation_manager._history_buffers.items():
            history_snapshot[group_name] = {}
            for term_name, buffer in group_buffers.items():
                history_snapshot[group_name][term_name] = [self._clone_nested(item) for item in buffer]
        snapshot["obs_history"] = history_snapshot
        snapshot["obs_buf_dict"] = self._clone_nested(env.obs_buf_dict)
        snapshot["extras"] = self._clone_nested(env.extras)
        snapshot["log_dict"] = self._clone_nested(env.log_dict)

        reward_manager = getattr(env, "reward_manager", None)
        if reward_manager is not None:
            snapshot["reward_buf_internal"] = reward_manager._reward_buf.detach().clone()
            snapshot["reward_episode_sums"] = {
                k: v.detach().clone() for k, v in reward_manager.episode_sums.items()
            }
            snapshot["reward_episode_sums_raw"] = {
                k: v.detach().clone() for k, v in reward_manager.episode_sums_raw.items()
            }

        curriculum_states: dict[str, Any] = {}
        for term_name, term in env.curriculum_manager.iter_terms():
            if hasattr(term, "state_dict") and hasattr(term, "load_state_dict"):
                try:
                    curriculum_states[term_name] = copy.deepcopy(term.state_dict())
                except Exception:
                    continue
        snapshot["curriculum_states"] = curriculum_states
        return snapshot

    def _restore_env_snapshot(self, env: Any, snapshot: dict[str, Any]) -> None:
        snapshot = _nested_to_device(snapshot, device=env.device)
        simulator = env.simulator
        env_ids = torch.arange(int(env.num_envs), device=env.device, dtype=torch.long)

        self._restore_root_states_tensor(simulator.all_root_states, snapshot["sim_all_root_states"], env.device)
        simulator.dof_state.copy_(snapshot["sim_dof_state"])
        simulator.set_actor_root_state_tensor(env_ids, simulator.all_root_states)
        simulator.set_dof_state_tensor(env_ids, simulator.dof_state)
        if "sim_contact_forces_history" in snapshot and hasattr(simulator, "contact_forces_history"):
            if isinstance(simulator.contact_forces_history, torch.Tensor):
                simulator.contact_forces_history.copy_(snapshot["sim_contact_forces_history"])
        if "sim_commands" in snapshot and hasattr(simulator, "commands"):
            if isinstance(simulator.commands, torch.Tensor):
                simulator.commands.copy_(snapshot["sim_commands"])

        for attr in (
            "episode_length_buf",
            "reset_buf",
            "time_out_buf",
            "rew_buf",
            "_pending_episode_lengths",
            "_pending_episode_update_mask",
            "base_quat",
            "need_to_refresh_envs",
            "push_robot_vel_buf",
            "record_push_robot_vel_buf",
            "action_delay_idx",
            "_max_push_vel",
        ):
            self._copy_tensor_from_snapshot(env, attr, snapshot)

        for attr in (
            "common_step_counter",
            "_randomize_push_robots",
            "_randomize_ctrl_delay",
            "_pending_torque_rfi",
            "is_evaluating",
        ):
            if attr in snapshot:
                setattr(env, attr, copy.deepcopy(snapshot[attr]))

        env.command_manager.commands.copy_(snapshot["command_tensor"])
        command_cfg = getattr(env.command_manager, "command_cfg", None)
        if command_cfg is not None and "command_resample_time" in snapshot:
            setattr(command_cfg, "locomotion_command_resampling_time", float(snapshot["command_resample_time"]))
        command_term = env.command_manager.get_state("locomotion_command")
        if command_term is not None and "allow_eval_randomization" in snapshot:
            command_term.allow_eval_randomization = bool(snapshot["allow_eval_randomization"])
        gait_state = env.command_manager.get_state("locomotion_gait")
        gait_payload = snapshot.get("gait_state", {})
        if gait_state is not None and isinstance(gait_payload, dict):
            for attr in ("phase_offset", "phase", "gait_freq", "phase_dt"):
                if attr in gait_payload and hasattr(gait_state, attr):
                    getattr(gait_state, attr).copy_(gait_payload[attr])
            if "mean_gait_freq" in gait_payload and hasattr(gait_state, "mean_gait_freq"):
                gait_state.mean_gait_freq = float(gait_payload["mean_gait_freq"])

        push_state = env.randomization_manager.get_state("push_randomizer_state")
        push_payload = snapshot.get("push_state", {})
        if push_state is not None and isinstance(push_payload, dict):
            for attr in ("push_interval_s", "push_robot_counter", "push_robot_plot_counter", "_max_push_vel_tensor"):
                if attr in push_payload and hasattr(push_state, attr):
                    getattr(push_state, attr).copy_(push_payload[attr])
            for attr in ("enabled", "push_interval_range"):
                if attr in push_payload and hasattr(push_state, attr):
                    setattr(push_state, attr, copy.deepcopy(push_payload[attr]))

        actuator_state = env.randomization_manager.get_state("actuator_randomizer_state")
        actuator_payload = snapshot.get("actuator_state", {})
        if actuator_state is not None and isinstance(actuator_payload, dict):
            for attr in ("kp_scale", "kd_scale", "rfi_lim_scale"):
                if attr in actuator_payload and hasattr(actuator_state, attr):
                    getattr(actuator_state, attr).copy_(actuator_payload[attr])
            for attr in ("enable_pd_gain", "enable_rfi_lim", "kp_range", "kd_range", "rfi_lim_range", "rfi_lim"):
                if attr in actuator_payload and hasattr(actuator_state, attr):
                    setattr(actuator_state, attr, copy.deepcopy(actuator_payload[attr]))

        if "action_manager_action" in snapshot and hasattr(env.action_manager, "_action"):
            env.action_manager._action.copy_(snapshot["action_manager_action"])
        if "action_manager_prev_action" in snapshot and hasattr(env.action_manager, "_prev_action"):
            env.action_manager._prev_action.copy_(snapshot["action_manager_prev_action"])

        joint_term = env.action_manager.get_term("joint_control")
        joint_payload = snapshot.get("joint_term", {})
        if isinstance(joint_payload, dict):
            for attr in (
                "_raw_actions",
                "_processed_actions",
                "_actions_after_delay",
                "torques",
                "_prev_dof_vel",
                "action_queue",
                "_kp_scale",
                "_kd_scale",
                "_rfi_lim_scale",
            ):
                if attr in joint_payload and hasattr(joint_term, attr):
                    dst = getattr(joint_term, attr)
                    src = joint_payload[attr]
                    if isinstance(dst, torch.Tensor) and isinstance(src, torch.Tensor):
                        dst.copy_(src)
            for attr in ("_randomize_torque_rfi", "_rfi_lim"):
                if attr in joint_payload and hasattr(joint_term, attr):
                    setattr(joint_term, attr, copy.deepcopy(joint_payload[attr]))

        history_snapshot = snapshot.get("obs_history", {})
        dst_history = env.observation_manager._history_buffers
        for group_name, group_buffers in dst_history.items():
            for term_name, dst_buffer in group_buffers.items():
                dst_buffer.clear()
                if group_name in history_snapshot and term_name in history_snapshot[group_name]:
                    for item in history_snapshot[group_name][term_name]:
                        dst_buffer.append(_nested_to_device(item, device=env.device))

        restored_obs_buf = _nested_to_device(snapshot.get("obs_buf_dict", {}), device=env.device)
        env.obs_buf_dict = self._copy_nested_obs(env.obs_buf_dict, restored_obs_buf)
        if "extras" in snapshot:
            env.extras = _nested_to_device(snapshot["extras"], device=env.device)
        if "log_dict" in snapshot:
            env.log_dict = _nested_to_device(snapshot["log_dict"], device=env.device)

        reward_manager = getattr(env, "reward_manager", None)
        if reward_manager is not None:
            if "reward_buf_internal" in snapshot:
                reward_manager._reward_buf.copy_(snapshot["reward_buf_internal"])
            for key, value in snapshot.get("reward_episode_sums", {}).items():
                if key in reward_manager.episode_sums:
                    reward_manager.episode_sums[key].copy_(value)
            for key, value in snapshot.get("reward_episode_sums_raw", {}).items():
                if key in reward_manager.episode_sums_raw:
                    reward_manager.episode_sums_raw[key].copy_(value)

        for term_name, state in snapshot.get("curriculum_states", {}).items():
            term = env.curriculum_manager.get_term(term_name)
            if term is not None and hasattr(term, "load_state_dict"):
                try:
                    term.load_state_dict(copy.deepcopy(state))
                except Exception:
                    continue

        simulator.refresh_sim_tensors()
        env._pre_compute_observations_callback()

    @torch.no_grad()
    def _compute_strict_expert_chunk_labels(
        self,
        env: Any,
        anchor_obs_dict: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if not self._use_teacher_aligned_labels:
            return None, None, None
        if self.strict_label_worker is not None:
            train_snapshot = self._capture_env_snapshot(env)
            worker_chunk, worker_future_obs_chunk, worker_valid = self.strict_label_worker.compute(
                snapshot=train_snapshot,
                anchor_obs=anchor_obs_dict,
            )
            if worker_chunk is None or worker_future_obs_chunk is None or worker_valid is None:
                raise RuntimeError(
                    "Strict-label worker returned empty labels while teacher-aligned labels are enabled. "
                    "This indicates a strict rollout failure."
                )
            expert_chunk = worker_chunk.to(device=self.device, dtype=torch.float32)
            expert_future_obs_chunk = worker_future_obs_chunk.to(device=self.device, dtype=torch.float32)
            strict_label_valid = worker_valid.to(device=self.device, dtype=torch.bool)
            return expert_chunk, expert_future_obs_chunk, strict_label_valid
        if self.strict_label_env is None:
            raise RuntimeError("strict_label_env is required when teacher-aligned labels are enabled")
        strict_env = self.strict_label_env
        if int(strict_env.num_envs) != int(env.num_envs):
            raise RuntimeError(
                "strict_label_env num_envs mismatch: "
                f"train={int(env.num_envs)} strict={int(strict_env.num_envs)}"
            )

        train_snapshot = self._capture_env_snapshot(env)
        rng_snapshot = self._capture_rng_snapshot()
        num_envs = int(strict_env.num_envs)
        pred_horizon = int(self.cfg.pred_horizon)

        try:
            # Sync shadow env to the current training env state, then roll out strict
            # labels exclusively in shadow env so training env dynamics remain untouched.
            self._restore_env_snapshot(strict_env, train_snapshot)

            anchor_commands = strict_env.command_manager.commands.detach().clone()
            simulator_commands = (
                strict_env.simulator.commands.detach().clone()
                if hasattr(strict_env.simulator, "commands") and isinstance(strict_env.simulator.commands, torch.Tensor)
                else None
            )

            command_cfg = getattr(strict_env.command_manager, "command_cfg", None)
            if command_cfg is not None and hasattr(command_cfg, "locomotion_command_resampling_time"):
                # Keep instruction/command fixed along the expert branch rollout.
                setattr(command_cfg, "locomotion_command_resampling_time", -1.0)
            command_term = strict_env.command_manager.get_state("locomotion_command")
            if command_term is not None and hasattr(command_term, "allow_eval_randomization"):
                command_term.allow_eval_randomization = False

            expert_chunk = torch.zeros((num_envs, pred_horizon, self.act_dim), device=self.device, dtype=torch.float32)
            expert_future_obs_chunk = torch.zeros(
                (num_envs, pred_horizon, self.obs_dim),
                device=self.device,
                dtype=torch.float32,
            )
            strict_label_valid = torch.ones((num_envs,), device=self.device, dtype=torch.bool)
            inactive_envs = torch.zeros((num_envs,), device=self.device, dtype=torch.bool)
            branch_obs_dict: dict[str, torch.Tensor] = {
                key: value.detach().clone() for key, value in anchor_obs_dict.items()
            }

            for h in range(pred_horizon):
                # Ensure command tensors stay fixed during expert branch rollout.
                strict_env.command_manager.commands.copy_(anchor_commands)
                if simulator_commands is not None:
                    strict_env.simulator.commands.copy_(simulator_commands)
                expert_action_h = _clip_actions(strict_env, self.expert_policy.act(branch_obs_dict))
                if torch.any(inactive_envs):
                    expert_action_h = expert_action_h.clone()
                    expert_action_h[inactive_envs] = 0.0
                expert_chunk[:, h, :] = expert_action_h
                branch_obs_dict, _, branch_dones, _ = strict_env.step({"actions": expert_action_h})
                obs_next_h = self._extract_compact_obs_for_env(strict_env)
                if torch.any(inactive_envs):
                    obs_next_h = obs_next_h.clone()
                    obs_next_h[inactive_envs] = 0.0
                expert_future_obs_chunk[:, h, :] = obs_next_h
                branch_dones = branch_dones.bool()
                if torch.any(branch_dones):
                    strict_label_valid &= ~branch_dones
                    inactive_envs |= branch_dones
                if bool(torch.all(inactive_envs)):
                    break

            if torch.any(~strict_label_valid):
                expert_chunk = expert_chunk.clone()
                expert_chunk[~strict_label_valid] = 0.0
                expert_future_obs_chunk = expert_future_obs_chunk.clone()
                expert_future_obs_chunk[~strict_label_valid] = 0.0
            return expert_chunk, expert_future_obs_chunk, strict_label_valid
        finally:
            self._restore_rng_snapshot(rng_snapshot)

    @torch.no_grad()
    def _collect_expert_policy_rollout(
        self,
        env: Any,
        *,
        horizon: int,
        progress_desc: str = "Offline Collect (Steps)",
    ) -> OfflineRolloutData:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        obs_dict = env.reset_all()
        _set_command_resampling_interval_from_steps(env, interval_steps=int(self.cfg.command_resample_interval))

        obs_steps: list[np.ndarray] = []
        current_command_steps: list[np.ndarray] = []
        executed_act_steps: list[np.ndarray] = []
        expert_act_steps: list[np.ndarray] = []
        next_obs_steps: list[np.ndarray] = []
        reward_steps: list[np.ndarray] = []
        done_steps: list[np.ndarray] = []
        failure_steps: list[np.ndarray] = []

        for step_idx in _progress_range(
            horizon,
            desc=str(progress_desc),
            enabled=bool(self.cfg.show_progress),
            leave=False,
        ):
            obs_curr = self._extract_compact_obs_for_env(env)
            cmd_curr = self._extract_current_command(env)
            expert_action = _clip_actions(env, self.expert_policy.act(obs_dict))
            obs_dict, rewards, dones, _infos = env.step({"actions": expert_action})
            next_obs = self._extract_compact_obs_for_env(env)
            failure = dones.bool()
            is_boundary = (step_idx + 1) >= horizon
            boundary_dones = (
                torch.ones_like(failure, dtype=torch.bool)
                if is_boundary
                else torch.zeros_like(failure, dtype=torch.bool)
            )
            buffer_dones = torch.logical_or(failure, boundary_dones)

            obs_steps.append(obs_curr.detach().cpu().numpy().astype(np.float32, copy=False))
            current_command_steps.append(cmd_curr.detach().cpu().numpy().astype(np.float32, copy=False))
            executed_act_steps.append(expert_action.detach().cpu().numpy().astype(np.float32, copy=False))
            expert_act_steps.append(expert_action.detach().cpu().numpy().astype(np.float32, copy=False))
            next_obs_steps.append(next_obs.detach().cpu().numpy().astype(np.float32, copy=False))
            reward_steps.append(rewards.detach().cpu().numpy().astype(np.float32, copy=False))
            done_steps.append(buffer_dones.detach().cpu().numpy().astype(np.bool_, copy=False))
            failure_steps.append(failure.detach().cpu().numpy().astype(np.bool_, copy=False))

        return OfflineRolloutData(
            obs=np.stack(obs_steps, axis=0),
            current_command=np.stack(current_command_steps, axis=0),
            executed_act=np.stack(executed_act_steps, axis=0),
            expert_act=np.stack(expert_act_steps, axis=0),
            next_obs=np.stack(next_obs_steps, axis=0),
            reward=np.stack(reward_steps, axis=0),
            done=np.stack(done_steps, axis=0),
            failure=np.stack(failure_steps, axis=0),
        )

    @torch.no_grad()
    def _collect_suboptimal_rollout_with_oracle_labels(
        self,
        env: Any,
        *,
        horizon: int,
        progress_desc: str = "Suboptimal+Oracle Collect",
    ) -> tuple[OfflineRolloutData, OracleRelabeledChunks]:
        """Collect suboptimal rollout with per-step oracle relabeling via shadow env.

        Returns:
            rollout_data: standard rollout from the suboptimal policy (for IDM buffer).
            oracle_chunks: oracle future obs / actions at each step (for planner buffer).
        """
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        obs_dict = env.reset_all()
        _set_command_resampling_interval_from_steps(env, interval_steps=int(self.cfg.command_resample_interval))

        pred_horizon = int(self.cfg.pred_horizon)
        num_envs = int(env.num_envs)

        # Standard rollout storage.
        obs_steps: list[np.ndarray] = []
        current_command_steps: list[np.ndarray] = []
        executed_act_steps: list[np.ndarray] = []
        expert_act_steps: list[np.ndarray] = []
        next_obs_steps: list[np.ndarray] = []
        reward_steps: list[np.ndarray] = []
        done_steps: list[np.ndarray] = []
        failure_steps: list[np.ndarray] = []

        # Oracle chunks storage — pre-allocate full arrays.
        oracle_expert_chunks = np.zeros(
            (horizon, num_envs, pred_horizon, self.act_dim), dtype=np.float32,
        )
        oracle_future_obs_chunks = np.zeros(
            (horizon, num_envs, pred_horizon, self.obs_dim), dtype=np.float32,
        )
        oracle_valid = np.zeros((horizon, num_envs), dtype=np.bool_)

        # Save suboptimal checkpoint path so we can restore after oracle relabeling.
        suboptimal_checkpoint = self._currently_loaded_expert_checkpoint

        # Collect suboptimal rollout, storing env snapshots for batch oracle relabeling.
        stored_snapshots: list[dict[str, Any]] = []
        stored_obs_dicts: list[dict[str, torch.Tensor]] = []

        for step_idx in _progress_range(
            horizon,
            desc=str(progress_desc),
            enabled=bool(self.cfg.show_progress),
            leave=False,
        ):
            obs_curr = self._extract_compact_obs_for_env(env)
            cmd_curr = self._extract_current_command(env)

            # Capture state before stepping (for oracle relabeling later).
            stored_snapshots.append(self._capture_env_snapshot(env))
            stored_obs_dicts.append({k: v.detach().clone() for k, v in obs_dict.items()})

            # Step with suboptimal policy.
            expert_action = _clip_actions(env, self.expert_policy.act(obs_dict))
            obs_dict, rewards, dones, _infos = env.step({"actions": expert_action})
            next_obs = self._extract_compact_obs_for_env(env)
            failure = dones.bool()
            is_boundary = (step_idx + 1) >= horizon
            boundary_dones = (
                torch.ones_like(failure, dtype=torch.bool)
                if is_boundary
                else torch.zeros_like(failure, dtype=torch.bool)
            )
            buffer_dones = torch.logical_or(failure, boundary_dones)

            obs_steps.append(obs_curr.detach().cpu().numpy().astype(np.float32, copy=False))
            current_command_steps.append(cmd_curr.detach().cpu().numpy().astype(np.float32, copy=False))
            executed_act_steps.append(expert_action.detach().cpu().numpy().astype(np.float32, copy=False))
            expert_act_steps.append(expert_action.detach().cpu().numpy().astype(np.float32, copy=False))
            next_obs_steps.append(next_obs.detach().cpu().numpy().astype(np.float32, copy=False))
            reward_steps.append(rewards.detach().cpu().numpy().astype(np.float32, copy=False))
            done_steps.append(buffer_dones.detach().cpu().numpy().astype(np.bool_, copy=False))
            failure_steps.append(failure.detach().cpu().numpy().astype(np.bool_, copy=False))

        # --- Batch oracle relabeling ---
        # Use the stored snapshots to compute oracle future obs.
        # The worker has its own oracle policy, so no checkpoint swap needed in main process.
        # For local shadow env, swap to oracle first.
        use_worker = self.strict_label_worker is not None
        if not use_worker:
            self._maybe_load_expert_checkpoint(self.configured_expert_checkpoint)

        for step_idx in _progress_range(
            horizon,
            desc="Oracle Relabel",
            enabled=bool(self.cfg.show_progress),
            leave=False,
        ):
            snapshot = stored_snapshots[step_idx]
            anchor_obs = stored_obs_dicts[step_idx]

            if use_worker:
                chunk, future_obs, valid = self.strict_label_worker.compute(
                    snapshot=snapshot, anchor_obs=anchor_obs,
                )
                if chunk is not None and future_obs is not None and valid is not None:
                    oracle_expert_chunks[step_idx] = chunk.detach().cpu().numpy()
                    oracle_future_obs_chunks[step_idx] = future_obs.detach().cpu().numpy()
                    oracle_valid[step_idx] = valid.detach().cpu().numpy()
            else:
                # Local shadow env path: restore snapshot, roll oracle forward.
                strict_env = self.strict_label_env
                self._restore_env_snapshot(strict_env, snapshot)
                anchor_commands = strict_env.command_manager.commands.detach().clone()
                sim_cmds = (
                    strict_env.simulator.commands.detach().clone()
                    if hasattr(strict_env.simulator, "commands")
                    and isinstance(strict_env.simulator.commands, torch.Tensor)
                    else None
                )
                branch_obs = {k: v.detach().clone() for k, v in anchor_obs.items()}
                step_valid = torch.ones((num_envs,), device=self.device, dtype=torch.bool)
                inactive = torch.zeros((num_envs,), device=self.device, dtype=torch.bool)

                for h in range(pred_horizon):
                    strict_env.command_manager.commands.copy_(anchor_commands)
                    if sim_cmds is not None:
                        strict_env.simulator.commands.copy_(sim_cmds)
                    oracle_act = _clip_actions(strict_env, self.expert_policy.act(branch_obs))
                    if torch.any(inactive):
                        oracle_act = oracle_act.clone()
                        oracle_act[inactive] = 0.0
                    oracle_expert_chunks[step_idx, :, h, :] = oracle_act.detach().cpu().numpy()
                    branch_obs, _, branch_dones, _ = strict_env.step({"actions": oracle_act})
                    obs_h = self._extract_compact_obs_for_env(strict_env)
                    if torch.any(inactive):
                        obs_h = obs_h.clone()
                        obs_h[inactive] = 0.0
                    oracle_future_obs_chunks[step_idx, :, h, :] = obs_h.detach().cpu().numpy()
                    branch_dones = branch_dones.bool()
                    if torch.any(branch_dones):
                        step_valid &= ~branch_dones
                        inactive |= branch_dones
                    if bool(torch.all(inactive)):
                        break

                if torch.any(~step_valid):
                    oracle_expert_chunks[step_idx][~step_valid.cpu().numpy()] = 0.0
                    oracle_future_obs_chunks[step_idx][~step_valid.cpu().numpy()] = 0.0
                oracle_valid[step_idx] = step_valid.detach().cpu().numpy()

        # Free stored snapshots to release GPU memory.
        stored_snapshots.clear()
        stored_obs_dicts.clear()

        # Restore suboptimal checkpoint for subsequent rollouts.
        if not use_worker and suboptimal_checkpoint is not None:
            self._maybe_load_expert_checkpoint(suboptimal_checkpoint)

        rollout_data = OfflineRolloutData(
            obs=np.stack(obs_steps, axis=0),
            current_command=np.stack(current_command_steps, axis=0),
            executed_act=np.stack(executed_act_steps, axis=0),
            expert_act=np.stack(expert_act_steps, axis=0),
            next_obs=np.stack(next_obs_steps, axis=0),
            reward=np.stack(reward_steps, axis=0),
            done=np.stack(done_steps, axis=0),
            failure=np.stack(failure_steps, axis=0),
        )

        oracle_chunks = OracleRelabeledChunks(
            expert_chunk=oracle_expert_chunks,
            expert_future_obs_chunk=oracle_future_obs_chunks,
            valid=oracle_valid,
        )
        return rollout_data, oracle_chunks

    def _append_oracle_relabeled_to_buffer(
        self,
        *,
        target_buffer: ReplayBuffer,
        rollout_data: OfflineRolloutData,
        oracle_chunks: OracleRelabeledChunks,
        episode_ids: np.ndarray,
    ) -> dict[str, int]:
        """Flatten suboptimal rollout with oracle chunks and add to target buffer.

        expert_chunk / expert_future_obs_chunk store oracle-relabeled data.
        The raw trajectory fields (expert_act, obs) are preserved so IDM/FDM can
        reconstruct real-dynamics targets via trajectory_target_* keys at sample time.
        """
        filtered = flatten_offline_rollout_with_oracle_chunks(
            rollout_data,
            oracle_chunks,
            mode="suboptimal",
            pred_horizon=int(self.cfg.pred_horizon),
            episode_ids=episode_ids,
        )
        if filtered.obs.shape[0] > 0:
            target_buffer.add_batch(
                obs=filtered.obs,
                current_command=filtered.current_command,
                executed_act=filtered.executed_act,
                expert_act=filtered.expert_act,
                expert_chunk=filtered.expert_chunk,
                expert_future_obs_chunk=filtered.expert_future_obs_chunk,
                strict_label_valid=filtered.strict_label_valid,
                reward=filtered.reward,
                done=filtered.done,
                env_id=filtered.env_id,
                episode_id=filtered.episode_id,
            )
        return {
            "kept_steps": int(filtered.kept_steps),
            "kept_episodes": int(filtered.kept_episodes),
            "dropped_episodes": int(filtered.dropped_episodes),
        }

    def _append_filtered_rollout_to_buffer(
        self,
        *,
        target_buffer: ReplayBuffer,
        rollout_data: OfflineRolloutData,
        mode: str,
        episode_ids: np.ndarray,
    ) -> dict[str, int]:
        filtered = flatten_offline_rollout_to_replay_batch(
            rollout_data,
            mode=mode,  # type: ignore[arg-type]
            pred_horizon=int(self.cfg.pred_horizon),
            episode_ids=episode_ids,
        )
        if filtered.obs.shape[0] > 0:
            target_buffer.add_batch(
                obs=filtered.obs,
                current_command=filtered.current_command,
                executed_act=filtered.executed_act,
                expert_act=filtered.expert_act,
                expert_chunk=filtered.expert_chunk,
                expert_future_obs_chunk=filtered.expert_future_obs_chunk,
                strict_label_valid=filtered.strict_label_valid,
                reward=filtered.reward,
                done=filtered.done,
                env_id=filtered.env_id,
                episode_id=filtered.episode_id,
            )
        episode_ids += 1
        return {
            "kept_steps": int(filtered.kept_steps),
            "kept_episodes": int(filtered.kept_episodes),
            "dropped_episodes": int(filtered.dropped_episodes),
        }

    def _make_collection_buffer(self, *, capacity: int) -> ReplayBuffer:
        resolved_capacity = max(1, int(capacity))
        return ReplayBuffer(
            capacity=resolved_capacity,
            obs_dim=self.offline_buffer.obs_dim,
            act_dim=self.offline_buffer.act_dim,
            cmd_dim=self.offline_buffer.cmd_dim,
            history_len=self.offline_buffer.history_len,
            pred_horizon=self.offline_buffer.pred_horizon,
            require_future_obs_targets=self.offline_buffer.require_future_obs_targets,
            growable=True,
        )

    @torch.no_grad()
    def _collect_strict_optimal_rollout_buffer(
        self,
        env: Any,
        *,
        horizon: int,
    ) -> ReplayBuffer:
        num_envs = int(env.num_envs)
        temp_buffer = self._make_collection_buffer(capacity=horizon * num_envs)
        self._collect_with_policy(
            env,
            num_steps=horizon,
            sigma=0.0,
            execute_expert=True,
            target_buffer=temp_buffer,
            progress_desc="Optimal Collect (Steps)",
        )
        return temp_buffer

    def _append_complete_episodes_from_buffer(
        self,
        *,
        source_buffer: ReplayBuffer,
        target_buffer: ReplayBuffer,
        required_length: int,
    ) -> dict[str, int]:
        if required_length <= 0:
            raise ValueError("required_length must be positive")
        payload = source_buffer._as_chronological_dict()
        num_rows = int(payload["obs"].shape[0])
        if num_rows <= 0:
            return {
                "kept_steps": 0,
                "kept_episodes": 0,
                "dropped_episodes": 0,
            }

        episode_positions: dict[tuple[int, int], list[int]] = {}
        for pos, (env_id, episode_id) in enumerate(zip(payload["env_id"].tolist(), payload["episode_id"].tolist())):
            key = (int(env_id), int(episode_id))
            bucket = episode_positions.get(key)
            if bucket is None:
                bucket = []
                episode_positions[key] = bucket
            bucket.append(int(pos))

        keep_mask = np.zeros((num_rows,), dtype=np.bool_)
        kept_episodes = 0
        for positions in episode_positions.values():
            pos_arr = np.asarray(positions, dtype=np.int64)
            done_flags = payload["done"][pos_arr].astype(np.bool_, copy=False)
            is_full_horizon_episode = (
                pos_arr.size == required_length
                and bool(done_flags[-1])
                and int(done_flags.sum()) == 1
            )
            if not is_full_horizon_episode:
                continue
            keep_mask[pos_arr] = True
            kept_episodes += 1

        kept_steps = int(keep_mask.sum())
        if kept_steps > 0:
            target_buffer.add_batch(
                obs=payload["obs"][keep_mask],
                current_command=payload["current_command"][keep_mask],
                executed_act=payload["executed_act"][keep_mask],
                expert_act=payload["expert_act"][keep_mask],
                expert_chunk=payload["expert_chunk"][keep_mask],
                expert_future_obs_chunk=payload["expert_future_obs_chunk"][keep_mask],
                strict_label_valid=payload["strict_label_valid"][keep_mask],
                reward=payload["reward"][keep_mask],
                done=payload["done"][keep_mask],
                env_id=payload["env_id"][keep_mask],
                episode_id=payload["episode_id"][keep_mask],
            )
        return {
            "kept_steps": int(kept_steps),
            "kept_episodes": int(kept_episodes),
            "dropped_episodes": int(len(episode_positions) - kept_episodes),
        }

    def _target_optimal_offline_steps(self, *, num_envs: int, horizon: int) -> int:
        if self.cfg.warmup_collect_steps is not None:
            return int(self.cfg.warmup_collect_steps)
        return int(self.cfg.warmup_episodes) * int(num_envs) * int(horizon)

    def _collect_optimal_offline_data(self, env: Any) -> dict[str, Any]:
        checkpoint_dir = self.configured_expert_checkpoint.parent
        checkpoints = list_model_checkpoints(checkpoint_dir)
        if not checkpoints:
            raise FileNotFoundError(f"No model_*.pt checkpoints found in: {checkpoint_dir}")
        optimal_checkpoint = select_optimal_checkpoint(checkpoints)
        self._maybe_load_expert_checkpoint(optimal_checkpoint)

        horizon = int(self.cfg.warmup_max_steps_per_episode)
        num_envs = int(env.num_envs)
        target_steps = self._target_optimal_offline_steps(num_envs=num_envs, horizon=horizon)
        kept_steps = 0
        kept_episodes = 0
        dropped_episodes = 0
        rollouts = 0
        stagnation_rollouts = 0

        if self.cfg.warmup_collect_steps is not None:
            while kept_steps < target_steps:
                rollout_buffer = self._collect_strict_optimal_rollout_buffer(env, horizon=horizon)
                rollout_stats = self._append_complete_episodes_from_buffer(
                    source_buffer=rollout_buffer,
                    target_buffer=self.offline_buffer,
                    required_length=horizon,
                )
                kept_steps += int(rollout_stats["kept_steps"])
                kept_episodes += int(rollout_stats["kept_episodes"])
                dropped_episodes += int(rollout_stats["dropped_episodes"])
                rollouts += 1
                stagnation_rollouts = stagnation_rollouts + 1 if rollout_stats["kept_steps"] <= 0 else 0
                if stagnation_rollouts >= 10:
                    raise RuntimeError(
                        "Optimal offline collection made no progress for 10 consecutive rollouts. "
                        "The chosen checkpoint may be too unstable to generate full-horizon trajectories."
                    )
            collect_mode = "steps"
        else:
            target_rollouts = int(self.cfg.warmup_episodes)
            while rollouts < target_rollouts:
                rollout_buffer = self._collect_strict_optimal_rollout_buffer(env, horizon=horizon)
                rollout_stats = self._append_complete_episodes_from_buffer(
                    source_buffer=rollout_buffer,
                    target_buffer=self.offline_buffer,
                    required_length=horizon,
                )
                kept_steps += int(rollout_stats["kept_steps"])
                kept_episodes += int(rollout_stats["kept_episodes"])
                dropped_episodes += int(rollout_stats["dropped_episodes"])
                rollouts += 1
            collect_mode = "episodes"
            stagnation_rollouts = 0

        return {
            "collect_mode": collect_mode,
            "checkpoint": str(optimal_checkpoint),
            "label_mode": "strict_teacher_aligned",
            "target_steps": int(target_steps),
            "rollouts": int(rollouts),
            "kept_steps": int(kept_steps),
            "kept_episodes": int(kept_episodes),
            "dropped_episodes": int(dropped_episodes),
            "horizon": horizon,
        }

    def _collect_suboptimal_offline_data(self, env: Any, *, optimal_target_steps: int) -> dict[str, Any]:
        if not self.suboptimal_enabled:
            return {
                "enabled": False,
                "target_steps": 0,
                "kept_steps": 0,
                "kept_episodes": 0,
                "dropped_episodes": 0,
                "rollouts": 0,
                "selected_checkpoints": [],
            }
        target_steps = target_suboptimal_steps(
            optimal_kept_steps=int(optimal_target_steps),
            ratio=float(self.cfg.suboptimal_data_ratio),
        )
        if target_steps <= 0:
            return {
                "enabled": True,
                "target_steps": 0,
                "kept_steps": 0,
                "kept_episodes": 0,
                "dropped_episodes": 0,
                "rollouts": 0,
                "selected_checkpoints": [],
            }

        checkpoint_dir = self.configured_expert_checkpoint.parent
        checkpoints = list_model_checkpoints(checkpoint_dir)
        if len(checkpoints) <= 1:
            raise RuntimeError(
                "suboptimal_data_ratio > 0 requires at least two checkpoints in the expert checkpoint directory."
            )
        reward_map = self._load_reward_curve(checkpoint_dir)
        sampling_mode = str(self.cfg.suboptimal_checkpoint_sampling)
        selected_checkpoints = select_suboptimal_checkpoints(
            checkpoints,
            n_target=int(self.cfg.suboptimal_num_checkpoints),
            sampling_mode=sampling_mode,  # type: ignore[arg-type]
            reward_map=reward_map,
        )
        if not selected_checkpoints:
            raise RuntimeError("Failed to resolve any suboptimal checkpoints for offline warmup collection.")

        horizon = int(self.cfg.warmup_max_steps_per_episode)
        num_envs = int(env.num_envs)
        episode_ids = np.zeros((num_envs,), dtype=np.int64)
        checkpoint_cycle = itertools.cycle(selected_checkpoints)
        checkpoint_rollouts = {str(checkpoint): 0 for checkpoint in selected_checkpoints}
        checkpoint_steps = {str(checkpoint): 0 for checkpoint in selected_checkpoints}
        kept_steps = 0
        kept_episodes = 0
        dropped_episodes = 0
        rollouts = 0
        stagnation_rollouts = 0

        while kept_steps < target_steps:
            checkpoint = next(checkpoint_cycle)
            self._maybe_load_expert_checkpoint(checkpoint)

            if self.planner_suboptimal_enabled:
                # Inline oracle relabeling — same pattern as DAgger online:
                # loaded suboptimal policy drives env, worker provides oracle
                # labels per-step via _compute_strict_expert_chunk_labels.
                # Buffer stores: executed_act = suboptimal action (for IDM
                # trajectory targets), expert_chunk = oracle chunk (for planner).
                buffer_size_before = len(self.suboptimal_offline_buffer)
                collect_stats = self._collect_with_policy(
                    env,
                    num_steps=horizon,
                    sigma=0.0,
                    execute_expert=False,
                    use_loaded_policy=True,
                    target_buffer=self.suboptimal_offline_buffer,
                    progress_desc=f"Suboptimal+Oracle [{Path(str(checkpoint)).stem}]",
                )
                new_steps = len(self.suboptimal_offline_buffer) - buffer_size_before
                rollout_stats = {
                    "kept_steps": new_steps,
                    "kept_episodes": int(collect_stats.get("completed_episodes", 0)),
                    "dropped_episodes": 0,
                }
            else:
                rollout_data = self._collect_expert_policy_rollout(
                    env,
                    horizon=horizon,
                    progress_desc="Suboptimal Collect (Steps)",
                )
                rollout_stats = self._append_filtered_rollout_to_buffer(
                    target_buffer=self.suboptimal_offline_buffer,
                    rollout_data=rollout_data,
                    mode="suboptimal",
                    episode_ids=episode_ids,
                )

            checkpoint_rollouts[str(checkpoint)] += 1
            checkpoint_steps[str(checkpoint)] += int(rollout_stats["kept_steps"])
            kept_steps += int(rollout_stats["kept_steps"])
            kept_episodes += int(rollout_stats["kept_episodes"])
            dropped_episodes += int(rollout_stats["dropped_episodes"])
            rollouts += 1
            stagnation_rollouts = stagnation_rollouts + 1 if rollout_stats["kept_steps"] <= 0 else 0
            if stagnation_rollouts >= max(10, 2 * len(selected_checkpoints)):
                raise RuntimeError(
                    "Suboptimal offline collection made no progress for too many consecutive rollouts. "
                    "The sampled checkpoints may all terminate before producing usable data."
                )

        result: dict[str, Any] = {
            "enabled": True,
            "target_optimal_steps": int(optimal_target_steps),
            "target_steps": int(target_steps),
            "kept_steps": int(kept_steps),
            "kept_episodes": int(kept_episodes),
            "dropped_episodes": int(dropped_episodes),
            "rollouts": int(rollouts),
            "selected_checkpoints": [str(checkpoint) for checkpoint in selected_checkpoints],
            "checkpoint_rollouts": checkpoint_rollouts,
            "checkpoint_steps": checkpoint_steps,
            "sampling_mode": str(sampling_mode),
        }
        return result

    def _load_offline_caches(self) -> dict[str, Any]:
        stats: dict[str, Any] = {
            "offline_cache_hit": False,
            "optimal_offline_cache_hit": False,
            "suboptimal_offline_cache_hit": False,
        }
        if self.cfg.force_recollect_offline:
            return stats

        optimal_cache = Path(self.cfg.offline_cache_path)
        suboptimal_cache = Path(self.cfg.offline_suboptimal_cache_path)

        try:
            if optimal_cache.exists():
                optimal_loaded = self.offline_buffer.load_npz(
                    optimal_cache,
                    expected_compact_obs_metadata=self._offline_cache_compact_obs_metadata,
                )
                stats["optimal_offline_cache_hit"] = True
                stats["optimal_offline_loaded_transitions"] = int(optimal_loaded)
            else:
                self.offline_buffer.clear()

            if self.suboptimal_enabled and suboptimal_cache.exists():
                suboptimal_loaded = self.suboptimal_offline_buffer.load_npz(
                    suboptimal_cache,
                    expected_compact_obs_metadata=self._offline_cache_compact_obs_metadata,
                )
                stats["suboptimal_offline_cache_hit"] = True
                stats["suboptimal_offline_loaded_transitions"] = int(suboptimal_loaded)
            else:
                self.suboptimal_offline_buffer.clear()

            stats["offline_cache_hit"] = bool(stats["optimal_offline_cache_hit"]) and (
                not self.suboptimal_enabled or bool(stats["suboptimal_offline_cache_hit"])
            )
            return stats
        except ReplayCacheLegacyMetadataMissingError:
            stats["offline_cache_legacy_missing_metadata"] = True
            return stats
        except ReplayCacheMetadataMismatchError as exc:
            raise RuntimeError(
                "Offline cache compact-obs metadata mismatch. "
                "Delete/recollect the caches or enable --force-recollect-offline true.\n"
                f"optimal_cache={optimal_cache}\n"
                f"suboptimal_cache={suboptimal_cache}\n"
                f"{exc}"
            ) from exc

    @torch.no_grad()
    def _collect_with_policy(
        self,
        env: Any,
        *,
        num_steps: int,
        sigma: float,
        execute_expert: bool,
        target_buffer: ReplayBuffer,
        progress_desc: str | None = None,
        use_loaded_policy: bool = False,
    ) -> dict[str, Any]:
        if execute_expert and use_loaded_policy:
            raise ValueError(
                "execute_expert and use_loaded_policy are mutually exclusive. "
                "execute_expert=True uses oracle chunk actions to step the env; "
                "use_loaded_policy=True uses the currently-loaded expert_policy checkpoint."
            )
        if use_loaded_policy:
            if not bool(getattr(self, "_use_teacher_aligned_labels", False)):
                raise ValueError(
                    "use_loaded_policy=True requires teacher-aligned labels (strict_chunk_labels "
                    "or predict_future_obs) so oracle labels come from the strict_label_worker, "
                    "not from self.expert_policy (which holds the suboptimal checkpoint)."
                )
            if getattr(self, "strict_label_worker", None) is None:
                raise ValueError(
                    "use_loaded_policy=True requires strict_label_worker because self.expert_policy "
                    "holds the suboptimal checkpoint in this mode. The local shadow env fallback "
                    "in _compute_strict_expert_chunk_labels would produce incorrect oracle labels."
                )
        obs_dict = env.reset_all()
        num_envs = int(env.num_envs)
        _set_command_resampling_interval_from_steps(env, interval_steps=int(self.cfg.command_resample_interval))
        hist_obs, hist_act, prev_action, episode_ids = self._init_rollout_buffers(num_envs=num_envs)
        if use_loaded_policy:
            episode_counter_attr = "_suboptimal_collect_episode_ids"
        elif execute_expert:
            episode_counter_attr = "_expert_collect_episode_ids"
        else:
            episode_counter_attr = "_online_rollout_episode_ids"
        persisted_episode_ids = getattr(self, episode_counter_attr, None)
        if (
            isinstance(persisted_episode_ids, torch.Tensor)
            and persisted_episode_ids.ndim == 1
            and int(persisted_episode_ids.shape[0]) == num_envs
        ):
            episode_ids = persisted_episode_ids.to(device=self.device, dtype=torch.long).clone()
        hist_valid = torch.zeros((num_envs, self.cfg.history_len), dtype=torch.bool, device=self.device)

        self.student_model.eval()

        collected_steps = 0
        reward_sum = 0.0
        lin_err_sum = 0.0
        lin_err_sq_sum = 0.0
        ang_err_sum = 0.0
        ang_err_sq_sum = 0.0
        action_mag_sum = 0.0
        action_mag_sq_sum = 0.0
        action_values_sum = 0.0
        action_values_sq_sum = 0.0
        action_values_count = 0
        correction_energy_sum = 0.0
        correction_energy_weight = 0.0
        strict_label_valid_count = 0
        strict_label_total_count = 0

        running_episode_return = torch.zeros((num_envs,), dtype=torch.float32, device=self.device)
        running_episode_length = torch.zeros((num_envs,), dtype=torch.long, device=self.device)
        # First-episode-only stats per env in this rollout:
        # count only the first done episode (or ongoing truncated at boundary).
        first_done_recorded = torch.zeros((num_envs,), dtype=torch.bool, device=self.device)
        first_episode_returns = torch.zeros((num_envs,), dtype=torch.float32, device=self.device)
        first_episode_lengths = torch.zeros((num_envs,), dtype=torch.long, device=self.device)
        episode_metric_sums: dict[str, float] = {}
        episode_metric_counts: dict[str, int] = {}
        raw_episode_metric_sums: dict[str, float] = {}
        raw_episode_metric_counts: dict[str, int] = {}
        env_metric_sums: dict[str, float] = {}
        env_metric_counts: dict[str, int] = {}

        def _filter_done_metrics(
            source: dict[str, Any] | None,
            keep_positions: torch.Tensor,
            total_done: int,
        ) -> dict[str, Any] | None:
            if not source or total_done <= 0 or keep_positions.numel() <= 0:
                return None
            out: dict[str, Any] = {}
            for key, value in source.items():
                if isinstance(value, torch.Tensor):
                    tensor = value
                    if tensor.ndim >= 1 and tensor.shape[0] == total_done:
                        out[key] = tensor[keep_positions]
                    else:
                        out[key] = tensor
                else:
                    out[key] = value
            return out

        if progress_desc is not None:
            rollout_desc = str(progress_desc)
        elif use_loaded_policy:
            rollout_desc = "Suboptimal Collect (Loaded Policy)"
        elif execute_expert:
            rollout_desc = "Offline Collect (Steps)"
        else:
            rollout_desc = "Online Rollout"
        # Keep all envs active across a rollout. When an env is done mid-rollout,
        # we close the episode, clear its history, and continue collecting from its
        # reset state in subsequent steps.
        train_active_mask = torch.ones((num_envs,), dtype=torch.bool, device=self.device)
        for step in _progress_range(
            num_steps,
            desc=rollout_desc,
            enabled=bool(self.cfg.show_progress),
            leave=False,
        ):
            obs_curr = self._extract_compact_obs_for_env(env)
            current_command = _extract_current_command(env)
            cmd_track = _extract_command_for_logging(env)
            base_lin = obs_terms.get_base_lin_vel(env)
            base_ang = obs_terms.get_base_ang_vel(env)

            hist_obs = torch.roll(hist_obs, shifts=-1, dims=1)
            hist_act = torch.roll(hist_act, shifts=-1, dims=1)
            hist_valid = torch.roll(hist_valid, shifts=-1, dims=1)
            hist_obs[:, -1, :] = obs_curr
            hist_act[:, -1, :] = prev_action
            hist_valid[:, -1] = train_active_mask

            expert_chunk: torch.Tensor | None = None
            expert_future_obs_chunk: torch.Tensor | None = None
            strict_label_valid: torch.Tensor | None = None
            if bool(getattr(self, "_use_teacher_aligned_labels", False)):
                expert_chunk, expert_future_obs_chunk, strict_label_valid = (
                    self._compute_strict_expert_chunk_labels(env, obs_dict)
                )
                if expert_chunk is None or expert_future_obs_chunk is None or strict_label_valid is None:
                    raise RuntimeError(
                        "Missing strict labels while teacher-aligned labels are enabled. "
                        "Strict rollout must produce action and future-observation chunks."
                    )

            # --- Determine execution action and oracle label action ---
            # Compute loaded-policy action first if this mode needs it,
            # to avoid calling self.expert_policy.act() twice.
            loaded_policy_action: torch.Tensor | None = None
            if use_loaded_policy:
                loaded_policy_action = _clip_actions(env, self.expert_policy.act(obs_dict))

            # Oracle label: prefer strict chunk label from worker, fall back to
            # loaded policy action (which equals oracle when execute_expert=True
            # without strict labels).
            if expert_chunk is not None:
                expert_action = expert_chunk[:, 0, :]
            elif loaded_policy_action is not None:
                expert_action = loaded_policy_action
            else:
                expert_action = _clip_actions(env, self.expert_policy.act(obs_dict))

            # Who drives the env:
            if use_loaded_policy:
                # Suboptimal mode: loaded checkpoint drives env, oracle labels via worker.
                assert loaded_policy_action is not None
                action_exec = loaded_policy_action
            elif execute_expert:
                # Optimal mode: oracle action drives env.
                action_exec = expert_action
            else:
                # Student mode (DAgger online): student model drives env.
                model_hist_obs = hist_obs
                model_hist_act = hist_act
                model_current_command = current_command
                action_mean_seq: torch.Tensor | None = None
                action_std_seq: torch.Tensor | None = None
                if self._use_io_normalization():
                    obs_mean, obs_std = self._ensure_obs_norm_stats_for_training()
                    action_mean, action_std = self._ensure_action_norm_stats_for_training()
                    command_mean, command_std = self._ensure_command_norm_stats_for_training()
                    obs_mean_seq = obs_mean.view(1, 1, -1)
                    obs_std_seq = obs_std.view(1, 1, -1)
                    action_mean_seq = action_mean.view(1, 1, -1)
                    action_std_seq = action_std.view(1, 1, -1)
                    command_mean_seq = command_mean.view(1, -1)
                    command_std_seq = command_std.view(1, -1)
                    model_hist_obs = (model_hist_obs - obs_mean_seq) / obs_std_seq
                    model_hist_act = (model_hist_act - action_mean_seq) / action_std_seq
                    model_current_command = (model_current_command - command_mean_seq) / command_std_seq
                pred_actions = self.student_model(
                    model_hist_obs,
                    model_hist_act,
                    model_current_command,
                    history_valid_mask=hist_valid,
                )
                if self._use_io_normalization():
                    assert action_mean_seq is not None and action_std_seq is not None
                    pred_actions = pred_actions * action_std_seq + action_mean_seq
                action_exec = pred_actions[:, 0, :]
                if sigma > 0.0:
                    action_exec = action_exec + float(sigma) * torch.randn_like(action_exec)
                action_exec = _clip_actions(env, action_exec)

            obs_dict, rewards, dones, infos = env.step({"actions": action_exec})
            dones = dones.bool()
            new_failures = dones
            is_rollout_boundary = (int(step) + 1) >= int(num_steps)
            boundary_dones = (
                torch.ones_like(dones, dtype=torch.bool)
                if is_rollout_boundary
                else torch.zeros_like(dones, dtype=torch.bool)
            )
            # Always close episodes at rollout boundary so no (env_id, episode_id)
            # remains open-ended in replay.
            buffer_dones = torch.logical_or(new_failures, boundary_dones)
            _accumulate_prefixed_metrics(
                infos.get("to_log"),
                prefix="Env/",
                sums=env_metric_sums,
                counts=env_metric_counts,
            )
            if strict_label_valid is None:
                strict_label_valid_for_buffer = train_active_mask
            else:
                strict_label_valid_for_buffer = torch.logical_and(strict_label_valid, train_active_mask)

            self._push_step_to_buffer(
                target_buffer=target_buffer,
                obs_curr=obs_curr,
                current_command=current_command,
                action_exec=action_exec,
                expert_action=expert_action,
                expert_chunk=expert_chunk,
                expert_future_obs_chunk=expert_future_obs_chunk,
                strict_label_valid=strict_label_valid_for_buffer,
                rewards=rewards,
                dones=buffer_dones,
                episode_ids=episode_ids,
            )
            strict_label_valid_count += int(strict_label_valid_for_buffer.sum().item())
            strict_label_total_count += int(strict_label_valid_for_buffer.numel())

            rewards_f = rewards.float()
            lin_err = torch.linalg.norm(base_lin[:, :2] - cmd_track[:, :2], dim=1)
            ang_err = torch.abs(base_ang[:, 2] - cmd_track[:, 2])
            action_mag = torch.linalg.norm(action_exec, dim=1)
            correction_energy = torch.sum(torch.square(action_exec - expert_action), dim=1)

            collected_steps += num_envs
            reward_sum += float(rewards_f.sum().item())
            lin_err_sum += float(lin_err.sum().item())
            lin_err_sq_sum += float(torch.sum(torch.square(lin_err)).item())
            ang_err_sum += float(ang_err.sum().item())
            ang_err_sq_sum += float(torch.sum(torch.square(ang_err)).item())
            action_mag_sum += float(action_mag.sum().item())
            action_mag_sq_sum += float(torch.sum(torch.square(action_mag)).item())
            action_values_sum += float(action_exec.sum().item())
            action_values_sq_sum += float(torch.sum(torch.square(action_exec)).item())
            action_values_count += int(action_exec.numel())
            if strict_label_valid is not None:
                strict_mask_f = strict_label_valid_for_buffer.to(device=correction_energy.device, dtype=torch.float32)
                correction_energy_sum += float(torch.sum(correction_energy * strict_mask_f).item())
                correction_energy_weight += float(strict_mask_f.sum().item())
            else:
                correction_energy_sum += float(correction_energy.sum().item())
                correction_energy_weight += float(correction_energy.numel())

            running_episode_return += rewards_f
            running_episode_length += 1

            if torch.any(new_failures):
                done_ids = torch.nonzero(new_failures, as_tuple=False).squeeze(-1)
                if done_ids.dim() == 0:
                    done_ids = done_ids.unsqueeze(0)
                first_done_before = first_done_recorded[done_ids]
                new_done_positions = torch.nonzero(~first_done_before, as_tuple=False).squeeze(-1)
                if new_done_positions.dim() == 0 and new_done_positions.numel() > 0:
                    new_done_positions = new_done_positions.unsqueeze(0)
                if new_done_positions.numel() > 0:
                    new_done_ids = done_ids[new_done_positions]
                    first_episode_returns[new_done_ids] = running_episode_return[new_done_ids]
                    first_episode_lengths[new_done_ids] = running_episode_length[new_done_ids]
                    first_done_recorded[new_done_ids] = True

                    # Episode/RawEpisode stats: keep only first done per env.
                    filtered_episode = _filter_done_metrics(
                        infos.get("episode"),
                        new_done_positions,
                        total_done=int(done_ids.numel()),
                    )
                    _accumulate_prefixed_metrics(
                        filtered_episode,
                        prefix="Episode/",
                        sums=episode_metric_sums,
                        counts=episode_metric_counts,
                    )
                    filtered_raw_episode = _filter_done_metrics(
                        infos.get("raw_episode"),
                        new_done_positions,
                        total_done=int(done_ids.numel()),
                    )
                    _accumulate_prefixed_metrics(
                        filtered_raw_episode,
                        prefix="RawEpisode/",
                        sums=raw_episode_metric_sums,
                        counts=raw_episode_metric_counts,
                    )

                running_episode_return[new_failures] = 0.0
                running_episode_length[new_failures] = 0

            if torch.any(buffer_dones):
                # Keep history strictly within episodes/chunks (no cross-boundary mixing).
                hist_obs[buffer_dones] = 0.0
                hist_act[buffer_dones] = 0.0
                hist_valid[buffer_dones] = False
            # Mid-rollout dones start a new episode immediately; boundary dones also
            # close the current episode and advance id for next rollout.
            if torch.any(buffer_dones):
                episode_ids[buffer_dones] += 1

            prev_action = action_exec.detach()
            if torch.any(buffer_dones):
                prev_action[buffer_dones] = 0.0

        count_steps = float(max(1, collected_steps))
        action_mag_mean = action_mag_sum / count_steps
        action_mag_var = max(action_mag_sq_sum / count_steps - action_mag_mean * action_mag_mean, 0.0)
        action_mag_std = float(np.sqrt(action_mag_var))

        action_values_count_f = float(max(1, action_values_count))
        action_values_mean = action_values_sum / action_values_count_f
        action_values_var = max(
            action_values_sq_sum / action_values_count_f - action_values_mean * action_values_mean,
            0.0,
        )
        action_values_std = float(np.sqrt(action_values_var))

        lin_err_mean = lin_err_sum / count_steps
        lin_err_rmse = float(np.sqrt(max(lin_err_sq_sum / count_steps, 0.0)))
        ang_err_mean = ang_err_sum / count_steps
        ang_err_rmse = float(np.sqrt(max(ang_err_sq_sum / count_steps, 0.0)))
        correction_energy_mean = correction_energy_sum / float(max(1.0, correction_energy_weight))

        # First-episode-only statistics:
        # for envs that never done in this rollout, use truncated ongoing episode.
        ongoing_mask = (~first_done_recorded) & (running_episode_length > 0)
        if torch.any(ongoing_mask):
            first_episode_returns[ongoing_mask] = running_episode_return[ongoing_mask]
            first_episode_lengths[ongoing_mask] = running_episode_length[ongoing_mask]

        # Include ongoing envs in Episode/ and RawEpisode/ so the mean is over all
        # envs in first-episode stats (done first episode + truncated first episode).
        ongoing_env_ids = torch.nonzero(ongoing_mask, as_tuple=False).squeeze(-1)
        if ongoing_env_ids.dim() == 0:
            ongoing_env_ids = ongoing_env_ids.unsqueeze(0)
        if ongoing_env_ids.numel() > 0 and getattr(env, "reward_manager", None) is not None:
            ongoing_rates = env.reward_manager.get_episode_rates(ongoing_env_ids)
            _accumulate_prefixed_metrics(
                ongoing_rates,
                prefix="Episode/",
                sums=episode_metric_sums,
                counts=episode_metric_counts,
            )
            ongoing_raw = env.reward_manager.get_raw_episode_rates(ongoing_env_ids)
            _accumulate_prefixed_metrics(
                ongoing_raw,
                prefix="RawEpisode/",
                sums=raw_episode_metric_sums,
                counts=raw_episode_metric_counts,
            )

        valid_first_mask = first_episode_lengths > 0
        if torch.any(valid_first_mask):
            episode_length_mean = float(first_episode_lengths[valid_first_mask].float().mean().item())
            episode_return_mean = float(first_episode_returns[valid_first_mask].float().mean().item())
        else:
            episode_length_mean = float("nan")
            episode_return_mean = float("nan")

        episode_metric_means = _finalize_prefixed_metrics(episode_metric_sums, episode_metric_counts)
        raw_episode_metric_means = _finalize_prefixed_metrics(raw_episode_metric_sums, raw_episode_metric_counts)
        env_metric_means = _finalize_prefixed_metrics(env_metric_sums, env_metric_counts)

        # average_episode_length is a global running average (updated only on dones); use its
        # current value at end of rollout instead of the mean-over-steps from env_metric_means.
        env_average_episode_length: float | None = None
        if hasattr(env, "average_episode_length"):
            env_average_episode_length = float(env.average_episode_length)

        setattr(self, episode_counter_attr, episode_ids.detach().clone())

        return {
            "collected_steps": int(collected_steps),
            "completed_episodes": int(first_done_recorded.sum().item()),
            "step_reward_mean": float(reward_sum / count_steps),
            "episode_return_mean": episode_return_mean,
            "episode_length_mean": episode_length_mean,
            "survival_time_mean": episode_length_mean,
            "lin_vel_tracking_error_mean": float(lin_err_mean),
            "lin_vel_tracking_error_rmse": float(lin_err_rmse),
            "ang_vel_tracking_error_mean": float(ang_err_mean),
            "ang_vel_tracking_error_rmse": float(ang_err_rmse),
            "action_magnitude_mean": float(action_mag_mean),
            "action_magnitude_std": float(action_mag_std),
            "action_component_std": float(action_values_std),
            "expert_correction_energy": float(correction_energy_mean),
            "strict_label_valid_ratio": (
                float(strict_label_valid_count / strict_label_total_count)
                if strict_label_total_count > 0
                else float("nan")
            ),
            "strict_label_valid_count": int(strict_label_valid_count),
            "strict_label_total_count": int(strict_label_total_count),
            "episode_metric_means": episode_metric_means,
            "raw_episode_metric_means": raw_episode_metric_means,
            "env_metric_means": env_metric_means,
            "env_average_episode_length": env_average_episode_length,
        }

    @torch.no_grad()
    def _collect_offline_by_episodes(
        self,
        env: Any,
        *,
        num_episodes: int,
        max_steps_per_episode: int,
    ) -> dict[str, int]:
        if num_episodes <= 0:
            raise ValueError("num_episodes must be positive")
        if max_steps_per_episode <= 0:
            raise ValueError("max_steps_per_episode must be positive")

        num_envs = int(env.num_envs)
        total_rollout_steps = int(num_episodes * max_steps_per_episode)
        collected_steps = 0
        completed_episodes = 0
        env_done_events = 0

        for _ in _progress_range(
            num_episodes,
            desc="Offline Collect (Episodes)",
            enabled=bool(self.cfg.show_progress),
            leave=False,
        ):
            rollout_stats = self._collect_with_policy(
                env,
                num_steps=int(max_steps_per_episode),
                sigma=0.0,
                execute_expert=True,
                target_buffer=self.offline_buffer,
            )
            collected_steps += int(rollout_stats.get("collected_steps", 0))
            # One fixed-length rollout chunk per env is treated as one completed episode.
            completed_episodes += int(num_envs)
            # Proxy for true mid-rollout failure events: envs that terminated before boundary.
            env_done_events += int(rollout_stats.get("completed_episodes", 0))

        return {
            "collected_steps": int(collected_steps),
            "completed_episodes": int(completed_episodes),
            "env_done_events": int(env_done_events),
            "rollout_steps": int(total_rollout_steps),
        }

    @torch.no_grad()
    def _collect_offline_by_steps(
        self,
        env: Any,
        *,
        num_steps: int,
    ) -> dict[str, int]:
        rollout_stats = self._collect_with_policy(
            env,
            num_steps=int(num_steps),
            sigma=0.0,
            execute_expert=True,
            target_buffer=self.offline_buffer,
        )
        return {
            "collected_steps": int(rollout_stats["collected_steps"]),
            "completed_episodes": int(rollout_stats["completed_episodes"]),
        }

    def _push_step_to_buffer(
        self,
        *,
        target_buffer: ReplayBuffer,
        obs_curr: torch.Tensor,
        current_command: torch.Tensor,
        action_exec: torch.Tensor,
        expert_action: torch.Tensor,
        expert_chunk: torch.Tensor | None,
        expert_future_obs_chunk: torch.Tensor | None,
        strict_label_valid: torch.Tensor | None,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        episode_ids: torch.Tensor,
    ) -> None:
        num_envs = int(obs_curr.shape[0])
        env_ids = np.arange(num_envs, dtype=np.int32)

        expert_chunk_np: np.ndarray | None = None
        if expert_chunk is not None:
            expert_chunk_np = expert_chunk.detach().cpu().numpy().astype(np.float32, copy=False)
        expert_future_obs_chunk_np: np.ndarray | None = None
        if expert_future_obs_chunk is not None:
            expert_future_obs_chunk_np = expert_future_obs_chunk.detach().cpu().numpy().astype(np.float32, copy=False)
        strict_label_valid_np: np.ndarray | None = None
        if strict_label_valid is not None:
            strict_label_valid_np = strict_label_valid.detach().cpu().numpy().astype(np.bool_, copy=False)

        target_buffer.add_batch(
            obs=obs_curr.detach().cpu().numpy().astype(np.float32, copy=False),
            current_command=current_command.detach().cpu().numpy().astype(np.float32, copy=False),
            executed_act=action_exec.detach().cpu().numpy().astype(np.float32, copy=False),
            expert_act=expert_action.detach().cpu().numpy().astype(np.float32, copy=False),
            expert_chunk=expert_chunk_np,
            expert_future_obs_chunk=expert_future_obs_chunk_np,
            strict_label_valid=strict_label_valid_np,
            reward=rewards.detach().cpu().numpy().astype(np.float32, copy=False),
            done=dones.detach().cpu().numpy().astype(np.bool_, copy=False),
            env_id=env_ids,
            episode_id=episode_ids.detach().cpu().numpy().astype(np.int64, copy=False),
        )

    def _init_rollout_buffers(
        self,
        *,
        num_envs: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hist_obs = torch.zeros(
            (num_envs, self.cfg.history_len, self.obs_dim),
            device=self.device,
            dtype=torch.float32,
        )
        hist_act = torch.zeros(
            (num_envs, self.cfg.history_len, self.act_dim),
            device=self.device,
            dtype=torch.float32,
        )
        prev_action = torch.zeros((num_envs, self.act_dim), device=self.device, dtype=torch.float32)
        episode_ids = torch.zeros((num_envs,), device=self.device, dtype=torch.long)
        return hist_obs, hist_act, prev_action, episode_ids
