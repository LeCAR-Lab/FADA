import json
import os
import random
import threading
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly
import plotly.graph_objects as go
import torch
from flask import Flask, render_template, send_file
from plotly.subplots import make_subplots
from torch import Tensor

from holosoma.agents.callbacks.base_callback import RLEvalCallback
from holosoma.agents.ppo.ppo import PPO
from holosoma.envs.locomotion.locomotion_manager import LeggedRobotLocomotionManager
from holosoma.managers.observation.terms.locomotion import (
    get_base_ang_vel,
    get_base_lin_vel,
)


def _metric_values_to_list(value) -> list[float]:
    if torch.is_tensor(value):
        return value.detach().cpu().reshape(-1).tolist()
    return np.asarray(value, dtype=np.float32).reshape(-1).tolist()


def _append_linear_tracking_values(out: list[float], metrics: dict) -> bool:
    for key in ("rew_track_lin_vel_xy", "rew_tracking_lin_vel"):
        if key in metrics:
            out.extend(_metric_values_to_list(metrics[key]))
            return True
    if "rew_tracking_lin_vel_x" in metrics and "rew_tracking_lin_vel_y" in metrics:
        x_val = np.asarray(_metric_values_to_list(metrics["rew_tracking_lin_vel_x"]), dtype=np.float32)
        y_val = np.asarray(_metric_values_to_list(metrics["rew_tracking_lin_vel_y"]), dtype=np.float32)
        out.extend((x_val + y_val).reshape(-1).tolist())
        return True
    return False


class AnalysisPlotLocomotion(RLEvalCallback):
    training_loop: PPO
    env: LeggedRobotLocomotionManager

    def __init__(self, config, training_loop: PPO):
        super().__init__(config, training_loop)
        env: LeggedRobotLocomotionManager = self.training_loop.env
        self.env = env
        self.policy = self.training_loop.get_inference_policy()
        self.num_envs = self.env.num_envs
        self.logger = WebLogger(self.config.sim_dt)
        self.reset_buffers()
        self.log_single_robot = getattr(self.config, "log_single_robot", False)
        log_dir = getattr(self.config, "log_dir", None) or self.training_loop.log_dir or "."
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self._log_path = Path(log_dir) / "state_log.npz"
        self.num_episodes = 0
        self.per_episode_step = 0
        self.command_resample_interval = getattr(self.config, "command_resample_interval", None)

        # Track done status for each environment (FDM logic)
        self.env_done_status = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Track episode steps for each environment (FDM logic)
        self.env_episode_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Store episode data for each episode (FDM logic)
        self.episode_data_list = []
        # Reward logging (same semantics as train/rollout)
        self._running_episode_return = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._completed_rew_lin: list[float] = []
        self._completed_rew_ang: list[float] = []
        self._completed_returns: list[float] = []
        self._recorded_env_ids: set[int] = set()

        # Set fixed seed for reproducible command generation
        self.fixed_seed = getattr(config, "command_seed", 42)
        random.seed(self.fixed_seed)
        np.random.seed(self.fixed_seed)
        torch.manual_seed(self.fixed_seed)

    def reset_buffers(self):
        self.obs_buf = [[] for _ in range(self.num_envs)]
        self.critic_obs_buf = [[] for _ in range(self.num_envs)]
        self.act_buf = [[] for _ in range(self.num_envs)]

    def on_pre_evaluate_policy(self):
        # Doing this in two lines because of type annotation issues.
        self.robot_num_dofs = self.env.num_dofs
        self.log_dof_pos_limits = self.env.dof_pos_limits.cpu().numpy()
        self.log_dof_vel_limits = self.env.dof_vel_limits.cpu().numpy()
        self.log_dof_torque_limits = self.env.torque_limits.cpu().numpy()
        self.logger.set_robot_limits(
            self.log_dof_pos_limits, self.log_dof_vel_limits, self.log_dof_torque_limits
        )
        self.logger.set_robot_num_dofs(self.robot_num_dofs)

    def on_post_evaluate_policy(self):
        if self.episode_data_list:
            self.get_all_episode_data()

    def _generate_commands(self, actor_state):
        step = actor_state["step"]
        commands = self.env.command_manager.commands
        dones = actor_state.get("dones")
        resample = step == 0 or (dones is not None and dones.all())
        if (
            self.command_resample_interval is not None
            and self.command_resample_interval > 0
            and step % self.command_resample_interval == 0
        ):
            resample = True

        if resample:
            # Generate different commands for each environment
            commands[:, 0] = (torch.rand(self.num_envs, device=commands.device) * 2 - 1) * 1.0
            commands[:, 1] = (torch.rand(self.num_envs, device=commands.device) * 2 - 1) * 1.0
            commands[:, 2] = (torch.rand(self.num_envs, device=commands.device) * 2 - 1) * 1.0
        # commands[:, 3] = 0.0
        return actor_state

    def on_pre_eval_env_step(self, actor_state):
        if "step" not in actor_state:
            return actor_state

        obs_dict = actor_state.get("obs") or {}
        obs: Tensor | None = obs_dict.get("actor_obs") if isinstance(obs_dict, dict) else None
        critic_obs: Tensor | None = obs_dict.get("critic_obs") if isinstance(obs_dict, dict) else None
        actions: Tensor = actor_state["actions"]

        if obs is not None and critic_obs is not None:
            for i in range(self.num_envs):
                self.obs_buf[i].append(obs[i])
                self.critic_obs_buf[i].append(critic_obs[i])
                self.act_buf[i].append(actions[i])
        self._generate_commands(actor_state)

        # Create masks for valid (not done) environments (FDM logic)
        valid_mask = ~self.env_done_status

        # Unified logging format (FDM style)
        masked_actions = actions.clone()
        masked_dof_pos = self.env.simulator.dof_pos.clone()
        masked_dof_vel = self.env.simulator.dof_vel.clone()
        masked_torques = self.env.action_manager.get_term("joint_control").torques.clone()
        masked_commands = self.env.command_manager.commands.clone()
        masked_base_vel = get_base_lin_vel(self.env).clone()
        masked_base_ang_vel = get_base_ang_vel(self.env).clone()
        masked_contact_forces = self.env.simulator.contact_forces.clone()

        # Set data to zero for done environments
        masked_actions[~valid_mask] = 0.0
        masked_dof_pos[~valid_mask] = 0.0
        masked_dof_vel[~valid_mask] = 0.0
        masked_torques[~valid_mask] = 0.0
        masked_commands[~valid_mask] = 0.0
        masked_base_vel[~valid_mask] = 0.0
        masked_base_ang_vel[~valid_mask] = 0.0
        masked_contact_forces[~valid_mask] = 0.0

        # Log all data in FDM format: [num_envs, features] or [num_envs]
        self.logger.log_states(
            {
                "dof_pos_target": masked_actions.clone(),
                "dof_pos": masked_dof_pos.clone(),
                "dof_vel": masked_dof_vel.clone(),
                "dof_torque": masked_torques.clone(),
                "command_x": masked_commands[:, 0].clone(),
                "command_y": masked_commands[:, 1].clone(),
                "command_yaw": masked_commands[:, 2].clone(),
                "base_vel_x": masked_base_vel[:, 0].clone(),
                "base_vel_y": masked_base_vel[:, 1].clone(),
                "base_vel_z": masked_base_vel[:, 2].clone(),
                "base_vel_yaw": masked_base_ang_vel[:, 2].clone(),
                "contact_forces_z": masked_contact_forces[:, self.env.feet_indices, 2].clone(),
            }
        )
        return actor_state

    def on_post_eval_env_step(self, actor_state):
        self.per_episode_step += 1
        step = actor_state["step"]
        dones = actor_state["dones"]
        rewards = actor_state.get("rewards")
        extras = actor_state.get("extras") or {}

        if rewards is not None:
            r = rewards.float().flatten()
            self._running_episode_return += r

        if (step + 1) % getattr(self.config, "plot_update_interval", 100) == 0:
            pass

        # Update episode steps for active environments
        active_mask = ~self.env_done_status
        self.env_episode_steps[active_mask] += 1

        # Log env_done_status for each step
        self.logger.log_states({"env_done_status": self.env_done_status})

        newly_done = dones & ~self.env_done_status
        if torch.any(newly_done):
            done_ids = torch.nonzero(newly_done, as_tuple=False).squeeze(-1)
            if done_ids.dim() == 0:
                done_ids = done_ids.unsqueeze(0)
            ep = extras.get("episode")
            if ep is not None:
                # extras["episode"] is for the envs that just reset, same order as done_ids
                _append_linear_tracking_values(self._completed_rew_lin, ep)
                for key in ("rew_track_ang_vel_z", "rew_tracking_ang_vel"):
                    if key in ep:
                        v = ep[key]
                        self._completed_rew_ang.extend(v.cpu().tolist() if torch.is_tensor(v) else [v])
                        break
            self._completed_returns.extend(
                self._running_episode_return[done_ids].cpu().tolist()
            )
            for i in done_ids.cpu().tolist():
                self._recorded_env_ids.add(int(i))
            self._running_episode_return[newly_done] = 0.0

        self.env_done_status = (self.env_done_status | dones).bool()

        # Stop episode when max_eval_steps (if set) or max_episode_length reached
        max_len = getattr(self.training_loop, "_eval_max_steps", None)
        if max_len is None:
            max_len = getattr(self.env, "max_episode_length", None)
        if max_len is not None and (step + 1) >= max_len:
            self.env_done_status[:] = True

        # Check if all environments are done
        if self.env_done_status.all():
            # Include ongoing (truncated) envs in reward means
            ongoing_env_ids = [
                i for i in range(self.num_envs)
                if self.env_episode_steps[i].item() > 0 and i not in self._recorded_env_ids
            ]
            if ongoing_env_ids and getattr(self.env, "reward_manager", None) is not None:
                ongoing_t = torch.tensor(ongoing_env_ids, device=self.device, dtype=torch.long)
                ongoing_rates = self.env.reward_manager.get_episode_rates(ongoing_t)
                _append_linear_tracking_values(self._completed_rew_lin, ongoing_rates)
                for key in ("rew_track_ang_vel_z", "rew_tracking_ang_vel"):
                    if key in ongoing_rates:
                        self._completed_rew_ang.extend(ongoing_rates[key].cpu().tolist())
                        break
                self._completed_returns.extend(
                    self._running_episode_return[ongoing_t].cpu().tolist()
                )

            # Log final episode steps for all environments
            self.logger.log_states({"per_episode_step": self.env_episode_steps.clone()})

            # Merge current episode data: num_envs * steps * features (FDM format)
            episode_data = {}
            for key, value in self.logger.state_log.items():
                if not torch.is_tensor(value[0]):
                    episode_data[key] = np.array(value)
                else:
                    stacked = torch.stack(value, dim=1)
                    episode_data[key] = stacked.cpu().numpy()

            # Per-episode reward means (same semantics as train)
            n_lin = len(self._completed_rew_lin)
            n_ang = len(self._completed_rew_ang)
            n_ret = len(self._completed_returns)
            episode_data["rew_tracking_lin_vel"] = np.array(
                float(np.mean(self._completed_rew_lin)) if n_lin else float("nan")
            )
            episode_data["rew_tracking_ang_vel"] = np.array(
                float(np.mean(self._completed_rew_ang)) if n_ang else float("nan")
            )
            episode_data["total_reward"] = np.array(
                float(np.mean(self._completed_returns)) if n_ret else float("nan")
            )
            # Reconstruction loss: from dynamics target/prediction if logged (e.g. FDM), else 0
            if "actual_target" in self.logger.state_log and "predicted_target" in self.logger.state_log:
                at = torch.stack(self.logger.state_log["actual_target"], dim=1)
                pt = torch.stack(self.logger.state_log["predicted_target"], dim=1)
                diff = (at - pt).float()
                recon = float(torch.sqrt(torch.mean(diff * diff)).item())
                episode_data["reconstruction_loss"] = np.array(recon)
            else:
                episode_data["reconstruction_loss"] = np.array(0.0)

            # Store episode data
            self.episode_data_list.append(episode_data)

            # Signal to PPO that episode is complete
            actor_state["episode_complete"] = True

            # Optional: randomize base mass for next episode if supported
            if hasattr(self.env.simulator, "_randomize_base_mass"):
                self.env.simulator._randomize_base_mass()

        return actor_state

    def reset_for_next_episode(self):
        # Clear logger for next episode
        self.logger.state_log.clear()
        # Reset all environment states for next episode
        self.env_done_status.zero_()
        self.env_episode_steps.zero_()
        # Reward logging
        self._running_episode_return.zero_()
        self._completed_rew_lin = []
        self._completed_rew_ang = []
        self._completed_returns = []
        self._recorded_env_ids = set()

    def get_all_episode_data(self):
        if not self.episode_data_list:
            return None

        # Handle variable episode lengths by padding to max length
        final_data = {}
        for key in self.episode_data_list[0].keys():
            episode_tensors = [ep_data[key] for ep_data in self.episode_data_list]
            shapes = [ep.shape for ep in episode_tensors]
            if len(set(shapes)) == 1:
                final_data[key] = np.stack(episode_tensors, axis=0)
            else:
                max_dims = [max(dim_sizes) for dim_sizes in zip(*shapes)]
                sample_ep = episode_tensors[0]
                dtype = sample_ep.dtype
                if np.issubdtype(dtype, np.integer):
                    fill_value = 0
                elif np.issubdtype(dtype, np.floating):
                    fill_value = 0.0
                elif np.issubdtype(dtype, np.bool_):
                    fill_value = False
                else:
                    fill_value = 0
                padded_episodes = []
                for ep in episode_tensors:
                    pad_width = [(0, max_dim - current_dim) for max_dim, current_dim in zip(max_dims, ep.shape)]
                    padded_ep = np.pad(ep, pad_width, mode="constant", constant_values=fill_value)
                    padded_episodes.append(padded_ep)
                final_data[key] = np.stack(padded_episodes, axis=0)

        np.savez_compressed(self._log_path, **final_data)
        return final_data


class WebLogger:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.app = Flask(__name__)
        self.thread = None
        self.tracking_errors = []
        self.current_epsiode_errors = []

    def set_robot_limits(self, dof_pos_limits, dof_vel_limits, dof_torque_limits):
        self.log_dof_pos_limits = dof_pos_limits
        self.log_dof_vel_limits = dof_vel_limits
        self.log_dof_torque_limits = dof_torque_limits

    def set_robot_num_dofs(self, num_dofs):
        self.robot_num_dofs = num_dofs

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for k, v in dict.items():
            self.state_log[k].append(v)

    def log_rew(self, dict):
        for k, v in dict.items():
            self.rew_log[k].append(v)

    def save_states(self, file_name="state_log.npz"):
        save_dict = {}
        for k, v in self.state_log.items():
            if torch.is_tensor(v[0]):
                save_dict[k] = torch.stack(v).cpu().numpy()
            else:
                save_dict[k] = np.array(v)
        np.savez(file_name, **save_dict)

    def plot_states(self):
        log = {}
        for k, v in self.state_log.items():
            if torch.is_tensor(v[0]):
                log[k] = torch.stack(v).cpu().numpy()
            else:
                log[k] = np.array(v)

        num_rows = 2 + self.robot_num_dofs
        num_dofs = self.robot_num_dofs
        BLUE = "#1f77b4"
        RED = "#ff7f0e"
        YELLOW = "#ffd700"

        fig = make_subplots(
            rows=num_rows,
            cols=4,
            subplot_titles=(
                "Base linear velocity x",
                "Base linear velocity y",
                "Base angular velocity yaw",
                "Base linear velocity z",
            ),
            shared_xaxes=False,
            vertical_spacing=0.03,
            horizontal_spacing=0.05,
        )

        time = [i * self.dt for i in range(len(log["base_vel_x"]))]

        def add_trace(t, y, color, row, col, name, width=1):
            fig.add_trace(
                go.Scatter(x=t, y=y, mode="lines", line=dict(color=color, width=width), name=name, showlegend=False),
                row=row,
                col=col,
            )

        add_trace(time, log["base_vel_x"], BLUE, 1, 1, "Base lin vel x")
        add_trace(time, log["base_vel_y"], BLUE, 1, 2, "Base lin vel y")
        add_trace(time, log["base_vel_yaw"], BLUE, 1, 3, "Base ang vel yaw")
        add_trace(time, log["base_vel_z"], BLUE, 1, 4, "Base lin vel z")

        forces = log["contact_forces_z"]
        for i in range(forces[0].shape[0]):
            add_trace(time, [force[i] for force in forces], BLUE, 2, 1, f"Force {i}")

        add_trace(time, log["command_x"], BLUE, 2, 2, "Command x")
        add_trace(time, log["command_y"], BLUE, 2, 3, "Command y")
        add_trace(time, log["command_yaw"], BLUE, 2, 4, "Command yaw")

        def add_limit_lines(row, col, lower, upper, color=YELLOW):
            fig.add_shape(
                type="rect",
                x0=time[0],
                x1=time[-1],
                y0=lower,
                y1=upper,
                fillcolor=color,
                line=dict(width=0),
                layer="below",
                row=row,
                col=col,
            )

        for i in range(num_dofs):
            row = i + 3
            add_trace(time, [pos[i] for pos in log["dof_pos"]], BLUE, row, 1, f"DOF {i} pos")
            add_trace(time, [pos[i] for pos in log["dof_pos_target"]], RED, row, 1, f"DOF {i} pos target")
            add_limit_lines(row, 1, self.log_dof_pos_limits[i, 0], self.log_dof_pos_limits[i, 1])
            add_trace(time, [vel[i] for vel in log["dof_vel"]], BLUE, row, 2, f"DOF {i} vel")
            add_limit_lines(row, 2, -self.log_dof_vel_limits[i], self.log_dof_vel_limits[i])
            add_trace(time, [torque[i] for torque in log["dof_torque"]], BLUE, row, 3, f"DOF {i} torque")
            add_limit_lines(row, 3, -self.log_dof_torque_limits[i], self.log_dof_torque_limits[i])

            fig.add_trace(
                go.Scatter(
                    x=[vel[i] for vel in log["dof_vel"]],
                    y=[torque[i] for torque in log["dof_torque"]],
                    mode="markers",
                    marker=dict(color=BLUE, size=2),
                    showlegend=False,
                    name=f"DOF {i} Torque/Velocity",
                ),
                row=row,
                col=4,
            )

            fig.add_shape(
                type="line",
                x0=-self.log_dof_vel_limits[i],
                y0=-self.log_dof_torque_limits[i],
                x1=-self.log_dof_vel_limits[i],
                y1=self.log_dof_torque_limits[i],
                line=dict(color=YELLOW, width=2),
                row=row,
                col=4,
            )
            fig.add_shape(
                type="line",
                x0=self.log_dof_vel_limits[i],
                y0=-self.log_dof_torque_limits[i],
                x1=self.log_dof_vel_limits[i],
                y1=self.log_dof_torque_limits[i],
                line=dict(color=YELLOW, width=2),
                row=row,
                col=4,
            )
            fig.add_shape(
                type="line",
                x0=-self.log_dof_vel_limits[i],
                y0=-self.log_dof_torque_limits[i],
                x1=self.log_dof_vel_limits[i],
                y1=-self.log_dof_torque_limits[i],
                line=dict(color=YELLOW, width=2),
                row=row,
                col=4,
            )
            fig.add_shape(
                type="line",
                x0=-self.log_dof_vel_limits[i],
                y0=self.log_dof_torque_limits[i],
                x1=self.log_dof_vel_limits[i],
                y1=self.log_dof_torque_limits[i],
                line=dict(color=YELLOW, width=2),
                row=row,
                col=4,
            )

        fig.update_layout(height=300 * num_rows, width=1500, title_text="Robot State Plots", showlegend=True)

        for i in range(num_rows):
            for j in range(3):
                fig.update_xaxes(title_text="time [s]", row=i + 1, col=j + 1)
            fig.update_xaxes(title_text="", row=i + 1, col=4)

        fig.update_yaxes(title_text="base lin vel x [m/s]", row=1, col=1)
        fig.update_yaxes(title_text="base lin vel y [m/s]", row=1, col=2)
        fig.update_yaxes(title_text="base ang vel yaw [rad/s]", row=1, col=3)
        fig.update_yaxes(title_text="base lin vel z [m/s]", row=1, col=4)
        fig.update_yaxes(title_text="Forces z [N]", row=2, col=1)
        fig.update_yaxes(title_text="Command vel x ", row=2, col=2)
        fig.update_yaxes(title_text="Command vel y ", row=2, col=3)
        fig.update_yaxes(title_text="Command vel z ", row=2, col=4)

        for i in range(3, num_rows + 1):
            fig.update_yaxes(title_text="Position [rad]", row=i, col=1)
            fig.update_yaxes(title_text="Velocity [rad/s]", row=i, col=2)
            fig.update_yaxes(title_text="Torque [Nm]", row=i, col=3)
            fig.update_yaxes(title_text="Torque/Velocity", row=i, col=4)

        plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        self.socketio.emit("update_plots", plot_json)

    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")

    def __del__(self):
        if self.thread:
            self.socketio.stop()
            self.thread.join()
