"""Unitree-style locomotion reward terms for G1.

These terms mirror the structure used in ``unitree/rewards.py`` and
``unitree/velocity_env_cfg.py`` but operate on Holosoma's locomotion env API.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from holosoma.utils.rotations import quat_rotate_inverse, yaw_quat
from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.envs.locomotion.locomotion_manager import LeggedRobotLocomotionManager


def _resolve_names_from_patterns(names: list[str], patterns: list[str]) -> list[str]:
    resolved: list[str] = []
    for pattern in patterns:
        for name in names:
            if re.match(pattern, name) and name not in resolved:
                resolved.append(name)
    return resolved


def _joint_indices(
    env: LeggedRobotLocomotionManager,
    joint_names: list[str] | None = None,
    joint_name_patterns: list[str] | None = None,
) -> list[int]:
    if not hasattr(env, "_unitree_joint_name_to_idx"):
        env._unitree_joint_name_to_idx = {name: i for i, name in enumerate(env.dof_names)}

    selected_names = list(joint_names or [])
    if joint_name_patterns:
        selected_names.extend(_resolve_names_from_patterns(list(env.dof_names), joint_name_patterns))

    deduped_names: list[str] = []
    for name in selected_names:
        if name not in deduped_names:
            deduped_names.append(name)

    name_to_idx: dict[str, int] = env._unitree_joint_name_to_idx
    return [name_to_idx[name] for name in deduped_names if name in name_to_idx]


def _body_indices(
    env: LeggedRobotLocomotionManager,
    body_names: list[str] | None = None,
    body_name_patterns: list[str] | None = None,
) -> torch.Tensor:
    body_list = list(getattr(env.simulator, "body_names", getattr(env, "body_names", [])))
    selected_names = list(body_names or [])
    if body_name_patterns:
        selected_names.extend(_resolve_names_from_patterns(body_list, body_name_patterns))

    deduped_names: list[str] = []
    for name in selected_names:
        if name not in deduped_names:
            deduped_names.append(name)

    indices = [body_list.index(name) for name in deduped_names if name in body_list]
    return torch.tensor(indices, dtype=torch.long, device=env.device)


def _history_contact_mask(
    env: LeggedRobotLocomotionManager, body_indices: torch.Tensor, threshold: float
) -> torch.Tensor:
    if body_indices.numel() == 0:
        return torch.zeros((env.num_envs, 0), dtype=torch.bool, device=env.device)

    contact_history = getattr(env.simulator, "contact_forces_history", None)
    if isinstance(contact_history, torch.Tensor):
        # (num_envs, history, bodies, 3) -> (num_envs, bodies)
        return torch.max(torch.norm(contact_history[:, :, body_indices, :], dim=-1), dim=1)[0] > threshold

    return torch.norm(env.simulator.contact_forces[:, body_indices, :], dim=-1) > threshold


def energy(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Penalize joint energy proxy |qvel| * |torque|."""
    joint_term = env.action_manager.get_term("joint_control")
    torques = getattr(joint_term, "torques", None)
    if torques is None:
        torques = getattr(env.simulator, "dof_forces", None)
    if torques is None:
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    return torch.sum(torch.abs(env.simulator.dof_vel) * torch.abs(torques), dim=1)


def track_lin_vel_xy_yaw_frame_exp(
    env: LeggedRobotLocomotionManager,
    std: float,
    command_name: str | None = None,
) -> torch.Tensor:
    """Reward xy linear velocity tracking in the yaw-only body frame.

    This matches the intent of Unitree's ``track_lin_vel_xy_yaw_frame_exp``:
    roll/pitch should not distort planar velocity tracking.
    """
    del command_name  # Kept for config compatibility with Unitree-style reward specs.

    yaw_only_quat = yaw_quat(env.base_quat, w_last=True)
    lin_vel_world = env.simulator.robot_root_states[:, 7:10]
    lin_vel_yaw_frame = quat_rotate_inverse(yaw_only_quat, lin_vel_world, w_last=True)
    lin_vel_error = torch.sum(torch.square(env.command_manager.commands[:, :2] - lin_vel_yaw_frame[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / (std**2))


def track_ang_vel_z_exp(
    env: LeggedRobotLocomotionManager,
    std: float,
    command_name: str | None = None,
) -> torch.Tensor:
    """Reward yaw-rate tracking with Unitree Lab's ``exp(-err / std^2)`` kernel."""
    del command_name  # Kept for config compatibility with Unitree-style reward specs.

    ang_vel_world = env.simulator.robot_root_states[:, 10:13]
    ang_vel_body = quat_rotate_inverse(env.base_quat, ang_vel_world, w_last=True)
    ang_vel_error = torch.square(env.command_manager.commands[:, 2] - ang_vel_body[:, 2])
    return torch.exp(-ang_vel_error / (std**2))


def joint_deviation_l1(env: LeggedRobotLocomotionManager, joint_names: list[str]) -> torch.Tensor:
    """Penalize L1 deviation from default pose for a selected joint subset."""
    indices = _joint_indices(env, joint_names=joint_names)
    if len(indices) == 0:
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    return torch.sum(torch.abs(env.simulator.dof_pos[:, indices] - env.default_dof_pos[:, indices]), dim=1)


def joint_deviation_l1_patterns(
    env: LeggedRobotLocomotionManager, joint_name_patterns: list[str]
) -> torch.Tensor:
    """Penalize L1 deviation from default pose for joints matched by regex patterns."""
    indices = _joint_indices(env, joint_name_patterns=joint_name_patterns)
    if len(indices) == 0:
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    return torch.sum(torch.abs(env.simulator.dof_pos[:, indices] - env.default_dof_pos[:, indices]), dim=1)


def feet_gait(
    env: LeggedRobotLocomotionManager,
    period: float,
    offset: list[float],
    threshold: float = 0.5,
    command_name: str | None = None,
    min_cmd_threshold: float = 0.1,
    body_names: list[str] | None = None,
    body_name_patterns: list[str] | None = None,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward matching left/right contact pattern to a fixed gait clock."""
    body_indices = _body_indices(env, body_names=body_names, body_name_patterns=body_name_patterns)
    if body_indices.numel() == 0:
        body_indices = env.feet_indices.to(device=env.device, dtype=torch.long)

    if int(body_indices.numel()) != len(offset):
        raise ValueError(f"Expected {int(body_indices.numel())} gait offsets, got {len(offset)}")

    is_contact = _history_contact_mask(env, body_indices, threshold=force_threshold)
    global_phase = ((env.episode_length_buf.float() * env.dt) % period / period).unsqueeze(1)
    leg_phase = torch.cat([((global_phase + offset_i) % 1.0) for offset_i in offset], dim=1)

    reward = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    for i in range(int(body_indices.numel())):
        is_stance = leg_phase[:, i] < threshold
        reward += (~(is_stance ^ is_contact[:, i])).float()

    # The locomotion env currently exposes a single velocity command buffer.
    # ``command_name`` is accepted for config compatibility with Unitree-style setups.
    cmd_norm = torch.norm(env.command_manager.commands[:, :3], dim=1)
    reward *= (cmd_norm > min_cmd_threshold).float()
    return reward


def feet_gait_command_phase(
    env: LeggedRobotLocomotionManager,
    period: float | None = None,
    offset: list[float] | None = None,
    threshold: float = 0.5,
    command_name: str | None = None,
    min_cmd_threshold: float = 0.1,
    body_names: list[str] | None = None,
    body_name_patterns: list[str] | None = None,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward matching contact pattern to the shared locomotion_gait phase.

    Unlike ``feet_gait``, this reads ``locomotion_gait.phase`` from the command
    manager so the policy observations and gait reward use the same phase source.
    ``period``/``offset`` are accepted for config compatibility but the phase
    tensor already encodes per-env frequency and left/right phase offset.
    """
    del period, offset, command_name

    body_indices = _body_indices(env, body_names=body_names, body_name_patterns=body_name_patterns)
    if body_indices.numel() == 0:
        body_indices = env.feet_indices.to(device=env.device, dtype=torch.long)

    gait_state = env.command_manager.get_state("locomotion_gait")
    if gait_state is None or getattr(gait_state, "phase", None) is None:
        raise RuntimeError("locomotion_gait phase is required for feet_gait_command_phase.")

    phase = gait_state.phase
    if phase.shape[1] < int(body_indices.numel()):
        raise ValueError(f"Expected at least {int(body_indices.numel())} gait phases, got {phase.shape[1]}")

    is_contact = _history_contact_mask(env, body_indices, threshold=force_threshold)
    # LocomotionGait stores per-leg phase in radians over [-pi, pi). Map it to
    # [0, 1) so the existing stance threshold logic matches the configured gait cycle.
    leg_phase = torch.remainder(phase[:, : int(body_indices.numel())] / (2 * torch.pi), 1.0)

    reward = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    for i in range(int(body_indices.numel())):
        is_stance = leg_phase[:, i] < threshold
        reward += (~(is_stance ^ is_contact[:, i])).float()

    cmd_norm = torch.norm(env.command_manager.commands[:, :3], dim=1)
    reward *= (cmd_norm > min_cmd_threshold).float()
    return reward


def foot_clearance_reward(
    env: LeggedRobotLocomotionManager,
    target_height: float,
    std: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward swinging feet for clearing a target height.

    Use terrain-relative foot height and gate the target by planar foot speed.
    """
    feet_heights = env.terrain_manager.get_state("locomotion_terrain").feet_heights
    feet_vel_world = env.simulator._rigid_body_vel[:, env.feet_height_indices, :]

    foot_z_target_error = torch.square(feet_heights - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(feet_vel_world[:, :, :2], dim=2))
    return torch.exp(-torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1) / std)


def feet_slide(
    env: LeggedRobotLocomotionManager,
    body_names: list[str] | None = None,
    body_name_patterns: list[str] | None = None,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize contacting feet that move laterally in the base frame.

    This mirrors Unitree-style logic more closely than the generic Holosoma
    slippage term by:
    - selecting explicit foot bodies,
    - using contact history for a more stable contact mask,
    - computing root-relative body-frame planar foot velocity.
    """
    body_indices = _body_indices(env, body_names=body_names, body_name_patterns=body_name_patterns)
    if body_indices.numel() == 0:
        body_indices = env.feet_indices.to(device=env.device, dtype=torch.long)

    contacts = _history_contact_mask(env, body_indices, threshold=force_threshold).float()

    feet_vel_world = env.simulator._rigid_body_vel[:, body_indices, :]
    root_vel_world = env.simulator.robot_root_states[:, 7:10].unsqueeze(1)
    feet_vel_rel_world = feet_vel_world - root_vel_world

    feet_vel_body = torch.zeros_like(feet_vel_rel_world)
    for i in range(body_indices.numel()):
        feet_vel_body[:, i, :] = quat_rotate_inverse(
            env.base_quat,
            feet_vel_rel_world[:, i, :],
            w_last=True,
        )

    foot_lateral_vel = torch.norm(feet_vel_body[:, :, :2], dim=2)
    return torch.sum(foot_lateral_vel * contacts, dim=1)


def undesired_contacts(
    env: LeggedRobotLocomotionManager,
    threshold: float = 1.0,
    body_names: list[str] | None = None,
    body_name_patterns: list[str] | None = None,
) -> torch.Tensor:
    """Count undesired contacts on bodies matched by explicit names or patterns."""
    body_indices = _body_indices(env, body_names=body_names, body_name_patterns=body_name_patterns)
    if body_indices.numel() == 0:
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    is_contact = _history_contact_mask(env, body_indices, threshold=threshold)
    return torch.sum(is_contact, dim=1)


def base_height_world(env: LeggedRobotLocomotionManager, target_height: float) -> torch.Tensor:
    """Penalize absolute root height deviation in world coordinates."""
    return torch.square(env.simulator.robot_root_states[:, 2] - target_height)


def base_height_l2(env: LeggedRobotLocomotionManager, target_height: float) -> torch.Tensor:
    """Unitree Lab compatibility alias for world-frame base-height penalty."""
    return base_height_world(env, target_height=target_height)


def contact_phase_match(
    env: LeggedRobotLocomotionManager,
    period: float = 0.8,
    offset: list[float] | None = None,
    threshold: float = 0.55,
    force_threshold: float = 1.0,
    body_names: list[str] | None = None,
    body_name_patterns: list[str] | None = None,
) -> torch.Tensor:
    """Reward matching contact schedule to a fixed left/right phase pattern.

    This mirrors ``unitree_gym.g1_env._reward_contact``.
    """
    if offset is None:
        offset = [0.0, 0.5]

    body_indices = _body_indices(env, body_names=body_names, body_name_patterns=body_name_patterns)
    if body_indices.numel() == 0:
        body_indices = env.feet_indices.to(device=env.device, dtype=torch.long)
    if int(body_indices.numel()) != len(offset):
        raise ValueError(f"Expected {int(body_indices.numel())} phase offsets, got {len(offset)}")

    phase = ((env.episode_length_buf.float() * env.dt) % period / period).unsqueeze(1)
    leg_phase = torch.cat([((phase + offset_i) % 1.0) for offset_i in offset], dim=1)
    contacts = torch.norm(env.simulator.contact_forces[:, body_indices, :3], dim=2) > force_threshold

    reward = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    for i in range(int(body_indices.numel())):
        is_stance = leg_phase[:, i] < threshold
        reward += (~(contacts[:, i] ^ is_stance)).float()
    return reward


def feet_swing_height_penalty(
    env: LeggedRobotLocomotionManager,
    target_height: float = 0.08,
    force_threshold: float = 1.0,
    body_names: list[str] | None = None,
    body_name_patterns: list[str] | None = None,
) -> torch.Tensor:
    """Penalize swing feet deviating from a target world-frame height."""
    body_indices = _body_indices(env, body_names=body_names, body_name_patterns=body_name_patterns)
    if body_indices.numel() == 0:
        body_indices = env.feet_indices.to(device=env.device, dtype=torch.long)

    contacts = torch.norm(env.simulator.contact_forces[:, body_indices, :3], dim=2) > force_threshold
    feet_pos = env.simulator._rigid_body_pos[:, body_indices, :]
    pos_error = torch.square(feet_pos[:, :, 2] - target_height) * (~contacts).float()
    return torch.sum(pos_error, dim=1)


def contact_no_vel_penalty(
    env: LeggedRobotLocomotionManager,
    force_threshold: float = 1.0,
    body_names: list[str] | None = None,
    body_name_patterns: list[str] | None = None,
) -> torch.Tensor:
    """Penalize contacting feet that still have residual velocity."""
    body_indices = _body_indices(env, body_names=body_names, body_name_patterns=body_name_patterns)
    if body_indices.numel() == 0:
        body_indices = env.feet_indices.to(device=env.device, dtype=torch.long)

    contacts = (torch.norm(env.simulator.contact_forces[:, body_indices, :3], dim=2) > force_threshold).unsqueeze(-1)
    feet_vel = env.simulator._rigid_body_vel[:, body_indices, :3]
    return torch.sum(torch.square(feet_vel * contacts.float()), dim=(1, 2))


def joint_position_l2(
    env: LeggedRobotLocomotionManager,
    joint_names: list[str] | None = None,
    joint_name_patterns: list[str] | None = None,
) -> torch.Tensor:
    """Penalize absolute joint position magnitude on a selected subset."""
    indices = _joint_indices(env, joint_names=joint_names, joint_name_patterns=joint_name_patterns)
    if len(indices) == 0:
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    return torch.sum(torch.square(env.simulator.dof_pos[:, indices]), dim=1)
