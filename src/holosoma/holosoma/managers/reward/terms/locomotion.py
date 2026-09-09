"""Reward terms for locomotion tasks.

These terms are migrated from LeggedRobotBase._reward_* methods to be
compatible with the reward manager system.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from holosoma.managers.observation.terms.locomotion import (
    base_forward_vector,
    get_base_ang_vel,
    get_base_lin_vel,
    get_projected_gravity,
    gravity_vector,
)
from holosoma.utils.rotations import (
    quat_apply,
    quat_rotate_batched,
    quat_rotate_inverse,
    wrap_to_pi,
)
from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.envs.locomotion.locomotion_manager import LeggedRobotLocomotionManager


def _expected_foot_height(phi: torch.Tensor, swing_height: float) -> torch.Tensor:
    """Expected foot height from gait phase using a cubic Bézier profile."""

    def cubic_bezier_interpolation(y_start: torch.Tensor, y_end: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        y_diff = y_end - y_start
        bezier = x**3 + 3 * (x**2 * (1 - x))
        return y_start + y_diff * bezier

    x = (phi + torch.pi) / (2 * torch.pi)
    stance = cubic_bezier_interpolation(torch.zeros_like(x), torch.full_like(x, swing_height), 2 * x)
    swing = cubic_bezier_interpolation(torch.full_like(x, swing_height), torch.zeros_like(x), 2 * x - 1)
    return torch.where(x <= 0.5, stance, swing)


# ================================================================================================
# Termination Rewards
# ================================================================================================


def termination(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Terminal reward/penalty for early termination (excluding timeouts).

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    return (env.reset_buf * ~env.time_out_buf).float()


# ================================================================================================
# Penalty Rewards
# ================================================================================================


def penalty_action_rate(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Penalize changes in actions between steps.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    actions = env.action_manager.action
    prev_actions = env.action_manager.prev_action
    return torch.sum(torch.square(prev_actions - actions), dim=1)


def penalty_orientation(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Penalize non-flat base orientation.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    projected = get_projected_gravity(env)
    return torch.sum(torch.square(projected[:, :2]), dim=1)


def penalty_feet_ori(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Penalize feet orientation deviation from flat.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    left_quat = env.simulator._rigid_body_rot[:, env.feet_indices[0]]
    gravity = gravity_vector(env)
    left_gravity = quat_rotate_inverse(left_quat, gravity, w_last=True)
    right_quat = env.simulator._rigid_body_rot[:, env.feet_indices[1]]
    right_gravity = quat_rotate_inverse(right_quat, gravity, w_last=True)
    return (
        torch.sum(torch.square(left_gravity[:, :2]), dim=1) ** 0.5
        + torch.sum(torch.square(right_gravity[:, :2]), dim=1) ** 0.5
    )

def feet_heading_alignment(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Penalize feet heading alignment.
    """
    left_quat = env.simulator._rigid_body_rot[:, env.feet_indices[0]]
    right_quat = env.simulator._rigid_body_rot[:, env.feet_indices[1]]
    forward_left_feet = quat_apply(left_quat, base_forward_vector(env), w_last=True)
    left_heading = torch.atan2(forward_left_feet[:, 1], forward_left_feet[:, 0])
    forward_right_feet = quat_apply(right_quat, base_forward_vector(env), w_last=True)
    right_heading = torch.atan2(forward_right_feet[:, 1], forward_right_feet[:, 0])
    root_forward = quat_apply(env.base_quat, base_forward_vector(env), w_last=True)
    heading_root = torch.atan2(root_forward[:, 1], root_forward[:, 0])
    

    heading_diff_left = torch.abs(wrap_to_pi(left_heading - heading_root))
    heading_diff_right = torch.abs(wrap_to_pi(right_heading - heading_root))
    return heading_diff_left + heading_diff_right


# ================================================================================================
# Limit Rewards
# ================================================================================================


def limits_dof_pos(env: LeggedRobotLocomotionManager, soft_dof_pos_limit: float = 0.95) -> torch.Tensor:
    """Penalize joint positions too close to limits.

    Args:
        env: The environment instance
        soft_dof_pos_limit: Soft limit as fraction of hard limit

    Returns:
        Reward tensor [num_envs]
    """
    # Use soft limits as fraction of hard limits
    m = (env.simulator.hard_dof_pos_limits[:, 0] + env.simulator.hard_dof_pos_limits[:, 1]) / 2  # type: ignore[attr-defined]
    r = env.simulator.hard_dof_pos_limits[:, 1] - env.simulator.hard_dof_pos_limits[:, 0]  # type: ignore[attr-defined]
    lower_soft_limit = m - 0.5 * r * soft_dof_pos_limit
    upper_soft_limit = m + 0.5 * r * soft_dof_pos_limit

    out_of_limits = -(env.simulator.dof_pos - lower_soft_limit).clip(max=0.0)  # lower limit
    out_of_limits += (env.simulator.dof_pos - upper_soft_limit).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


# ================================================================================================
# Tracking and Task Rewards
# ================================================================================================


def tracking_lin_vel(env, tracking_sigma: float = 0.25) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes).

    Uses exponential reward: exp(-error / sigma)

    Args:
        env: The environment instance
        tracking_sigma: Sigma for exponential reward scaling

    Returns:
        Reward tensor [num_envs]
    """
    commands = env.command_manager.commands
    lin_vel_error = torch.sum(torch.square(commands[:, :2] - get_base_lin_vel(env)[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / tracking_sigma)


def tracking_ang_vel(env, tracking_sigma: float = 0.25) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw).

    Uses exponential reward: exp(-error / sigma)

    Args:
        env: The environment instance
        tracking_sigma: Sigma for exponential reward scaling

    Returns:
        Reward tensor [num_envs]
    """
    commands = env.command_manager.commands
    ang_vel = get_base_ang_vel(env)
    ang_vel_error = torch.square(commands[:, 2] - ang_vel[:, 2])
    return torch.exp(-ang_vel_error / tracking_sigma)


def penalty_ang_vel_xy(env) -> torch.Tensor:
    """Penalize xy axes base angular velocity.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    ang_vel = get_base_ang_vel(env)
    return torch.sum(torch.square(ang_vel[:, :2]), dim=1)

def penalty_lin_vel_z(env) -> torch.Tensor:
    """Penalize z axis base linear velocity.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    lin_vel = get_base_lin_vel(env)
    return torch.square(lin_vel[:, 2])


def penalty_close_feet_xy(env, close_feet_threshold: float = 0.05) -> torch.Tensor:
    """Penalize when feet are too close together in xy plane.

    Args:
        env: The environment instance
        close_feet_threshold: Minimum distance threshold between feet

    Returns:
        Reward tensor [num_envs]
    """
    left_foot_xy = env.simulator._rigid_body_pos[:, env.feet_indices[0], :2]
    right_foot_xy = env.simulator._rigid_body_pos[:, env.feet_indices[1], :2]

    # Get base orientation
    base_forward = quat_apply(env.base_quat, base_forward_vector(env), w_last=True)
    base_yaw = torch.atan2(base_forward[:, 1], base_forward[:, 0])

    # Calculate perpendicular distance in base-local coordinates
    feet_distance = torch.abs(
        torch.cos(base_yaw) * (left_foot_xy[:, 1] - right_foot_xy[:, 1])
        - torch.sin(base_yaw) * (left_foot_xy[:, 0] - right_foot_xy[:, 0])
    )

    # Return penalty when feet are too close
    return (feet_distance < close_feet_threshold).float()


def base_height(
    env, desired_base_height: float = 0.89, zero_vel_penalty_scale: float = 1.0, stance_penalty_scale: float = 1.0
) -> torch.Tensor:
    """Penalize base height away from target.

    Args:
        env: The environment instance
        desired_base_height: Target base height
        zero_vel_penalty_scale: Multiplier for base height penalty when robot has zero velocity commands
        stance_penalty_scale: Multiplier for base height penalty when robot is in stance mode

    Returns:
        Reward tensor [num_envs]
    """
    base_height_penalty = torch.square(
        env.terrain_manager.get_state("locomotion_terrain").base_heights - desired_base_height
    )

    # Apply stronger penalty for zero velocity commands if configured
    if zero_vel_penalty_scale != 1.0:
        commands = env.command_manager.commands
        zero_vel_mask = torch.norm(commands[:, :2], dim=1) < 0.1
        base_height_penalty = torch.where(
            zero_vel_mask, base_height_penalty * zero_vel_penalty_scale, base_height_penalty
        )

    # Apply stronger penalty for stance mode if configured (used in decoupled locomotion)
    if stance_penalty_scale != 1.0 and hasattr(env, "stance_mask"):
        base_height_penalty = torch.where(
            env.stance_mask, base_height_penalty * stance_penalty_scale, base_height_penalty
        )

    return base_height_penalty


def feet_phase(env, swing_height: float = 0.08, tracking_sigma: float = 0.25) -> torch.Tensor:
    """Reward for tracking desired foot height based on gait phase.

    Based on MuJoCo Playground's implementation.

    Args:
        env: The environment instance
        swing_height: Maximum height during swing phase
        tracking_sigma: Sigma for exponential reward scaling

    Returns:
        Reward tensor [num_envs]
    """
    # Get foot heights (relative to terrain)
    foot_z_left = env.terrain_manager.get_state("locomotion_terrain").feet_heights[:, 0]
    foot_z_right = env.terrain_manager.get_state("locomotion_terrain").feet_heights[:, 1]

    # Calculate expected foot heights based on phase
    gait_state = env.command_manager.get_state("locomotion_gait")
    rz_left = _expected_foot_height(gait_state.phase[:, 0], swing_height)
    rz_right = _expected_foot_height(gait_state.phase[:, 1], swing_height)

    # Calculate height tracking errors
    error_left = torch.square(foot_z_left - rz_left)
    error_right = torch.square(foot_z_right - rz_right)

    # Combine errors and apply exponential reward
    total_error = error_left + error_right

    return torch.exp(-total_error / tracking_sigma)


def pose(
    env,
    pose_weights: list[float],
) -> torch.Tensor:
    """Reward for maintaining default pose.

    Penalizes deviation from default joint positions with weighted importance.

    Args:
        env: The environment instance
        pose_weights: List of weights for each DOF (must match num_dof)

    Returns:
        Reward tensor [num_envs]
    """
    # Get current joint positions
    qpos = env.simulator.dof_pos

    # Convert pose_weights to tensor
    weights = torch.tensor(pose_weights, device=env.device, dtype=torch.float32)

    # Calculate squared deviation from default pose
    # Use env.default_dof_pos which is already set up from robot config
    pose_error = torch.square(qpos - env.default_dof_pos)

    # Weight and sum the errors
    weighted_error = pose_error * weights.unsqueeze(0)

    return torch.sum(weighted_error, dim=1)


def penalty_stumble(env) -> torch.Tensor:
    """Penalize feet hitting vertical surfaces.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    return torch.any(
        torch.norm(env.simulator.contact_forces[:, env.feet_indices, :2], dim=2)
        > 4 * torch.abs(env.simulator.contact_forces[:, env.feet_indices, 2]),
        dim=1,
    )


def penalty_foothold(env, foothold_epsilon: float = 0.01) -> torch.Tensor:
    """Sampling-based foothold penalty.

    For each foot in contact, sample a grid of points on the sole, transform to world,
    read terrain height at those XY, compute depth d_ij = z_sample - terrain_z, and count
    samples with d_ij < epsilon. Sum over both feet.

    Args:
        env: The environment instance
        foothold_epsilon: Threshold for foothold depth penalty

    Returns:
        Reward tensor [num_envs]
    """
    # Contact mask per foot
    contact = env.simulator.contact_forces[:, env.feet_indices, 2] > 1.0  # [E,2]
    if not (contact.any()):
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    # Accumulator
    penalty = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    for foot_idx_local in range(2):
        # Skip if no env has contact on this foot to save work
        if not contact[:, foot_idx_local].any():
            continue
        rb_idx = env.feet_indices[foot_idx_local]
        foot_pos_w = env.simulator._rigid_body_pos[:, rb_idx, :]  # [E,3]
        foot_quat_w = env.simulator._rigid_body_rot[:, rb_idx, :]  # [E,4]

        # Use precomputed sample points in the foot frame
        pts_local = env.foot_samples_local[foot_idx_local].unsqueeze(0).repeat(env.num_envs, 1, 1)

        # Rotate to world and translate
        pts_world = quat_rotate_batched(foot_quat_w, pts_local) + foot_pos_w.unsqueeze(1)

        # Query terrain height at those XY positions
        terrain_h = env._get_terrain_heights_at_points_world(pts_world)

        # Depth: world z minus terrain height
        depth = pts_world[:, :, 2] - terrain_h  # [E,S]

        # Indicator for d_ij > epsilon, only for envs with this foot in contact
        bad = (depth > foothold_epsilon).float()
        bad *= contact[:, foot_idx_local].unsqueeze(1).float()

        penalty += torch.sum(bad, dim=1)

    return penalty / env.num_foot_samples


def alive(env) -> torch.Tensor:
    """Reward for staying alive.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    return torch.ones(env.num_envs, dtype=torch.float, device=env.device)


def feet_stance_width(
    env: LeggedRobotLocomotionManager,
    desired_feet_stance_width: float = 0.30,
    feet_stance_width_std: float = 0.05,
) -> torch.Tensor:
    """Reward for maintaining desired stance width between feet.

    Rewards the robot for keeping feet at target lateral separation in body frame.
    Only applies reward when both feet are in contact.

    Args:
        env: The environment instance
        desired_feet_stance_width: Target lateral distance between feet (meters)
        feet_stance_width_std: Standard deviation for exponential reward scaling

    Returns:
        Reward tensor [num_envs]
    """
    # Get feet positions in world frame
    feet_pos = env.simulator._rigid_body_pos[:, env.feet_indices, :]

    # Translate to robot frame (subtract base position)
    cur_footsteps_translated = feet_pos - env.simulator.robot_root_states[:, :3].unsqueeze(1)

    # Rotate to body frame using inverse quaternion rotation
    footsteps_in_body_frame = torch.zeros(env.num_envs, 2, 3, device=env.device)
    for i in range(2):
        footsteps_in_body_frame[:, i, :] = quat_rotate_inverse(
            env.base_quat, cur_footsteps_translated[:, i, :], w_last=True
        )

    # Desired y-positions: +stance_width/2 and -stance_width/2
    stance_width = desired_feet_stance_width * torch.ones([env.num_envs, 1], device=env.device)
    desired_ys = torch.cat([stance_width / 2, -stance_width / 2], dim=1)

    # Calculate squared error
    stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])

    # Exponential reward
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / feet_stance_width_std**2)

    # Only when both feet are in contact
    reward *= torch.all(env.simulator.contact_forces[:, env.feet_indices, 2] > 1.0, dim=1).float()

    return reward


def penalty_feet_slippage(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Penalize foot slippage when in contact with ground.

    Penalizes horizontal (xy) foot velocity when feet are in contact, indicating slippage.
    Only applies penalty when contact forces exceed threshold.
    Vertical (z) velocity is ignored to allow natural foot lifting.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    # Get foot velocities [num_envs, num_feet, 3]
    foot_vel = env.simulator._rigid_body_vel[:, env.feet_indices]

    # Only consider horizontal velocity (x, y), ignore vertical (z) to allow foot lifting
    foot_vel_xy = foot_vel[:, :, :2]  # [num_envs, num_feet, 2]
    foot_vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [num_envs, num_feet]

    # Contact mask: feet with contact forces > 1.0
    contact_forces_norm = torch.norm(env.simulator.contact_forces[:, env.feet_indices, :], dim=-1)
    in_contact = (contact_forces_norm > 1.0).float()  # [num_envs, num_feet]

    # Penalty is horizontal velocity magnitude when in contact, summed over both feet
    penalty = torch.sum(foot_vel_xy_norm * in_contact, dim=1)

    return penalty


def feet_air_time_single_stance(
    env: LeggedRobotLocomotionManager, max_air_time: float = 0.4, min_vel_threshold: float = 0.1
) -> torch.Tensor:
    """Reward for single stance phase with long air/contact times.

    Rewards the robot for maintaining single stance (exactly one foot in contact) with
    long air times. The reward is based on the minimum time across feet (air or contact),
    encouraging smooth transitions between feet during walking/running.

    Args:
        env: The environment instance
        max_air_time: Maximum reward clamp value (seconds)
        min_vel_threshold: Minimum velocity command to receive reward

    Returns:
        Reward tensor [num_envs]
    """
    # Initialize state buffers if they don't exist
    if not hasattr(env, "feet_air_time"):
        env.feet_air_time = torch.zeros(
            env.num_envs, len(env.feet_indices), dtype=torch.float, device=env.device, requires_grad=False
        )
    if not hasattr(env, "feet_contact_time"):
        env.feet_contact_time = torch.zeros(
            env.num_envs, len(env.feet_indices), dtype=torch.float, device=env.device, requires_grad=False
        )
    if not hasattr(env, "last_contacts"):
        env.last_contacts = torch.zeros(
            env.num_envs, len(env.feet_indices), dtype=torch.bool, device=env.device, requires_grad=False
        )

    # Detect contact (filter contacts for reliability on meshes)
    contact = env.simulator.contact_forces[:, env.feet_indices, 2] > 1.0  # [num_envs, 2]
    contact_filt = torch.logical_or(contact, env.last_contacts)
    env.last_contacts = contact

    in_contact = contact_filt
    in_air = ~in_contact

    # Accumulate time in each state
    env.feet_air_time += env.dt * in_air.float()
    env.feet_contact_time += env.dt * in_contact.float()

    # Determine which mode (air or contact) each foot is currently in
    # Use contact time when foot is in contact, air time when foot is in air
    in_mode_time = torch.where(in_contact, env.feet_contact_time, env.feet_air_time)

    # Check for single stance: exactly one foot should be in contact
    single_stance = torch.sum(in_contact.int(), dim=1) == 1  # [num_envs]

    # Get the minimum time across feet (this will be the time of the foot that's in the air during single stance)
    # Only consider when in single stance
    min_time = torch.min(
        torch.where(single_stance.unsqueeze(-1), in_mode_time, torch.zeros_like(in_mode_time)), dim=1
    )[0]

    # Clamp reward to maximum threshold
    reward = torch.clamp(min_time, max=max_air_time)

    # No reward for zero/low velocity commands
    commands = env.command_manager.commands
    reward *= (torch.norm(commands[:, :2], dim=1) > min_vel_threshold).float()

    # Reset counters when foot state changes
    # Reset air time when foot makes contact (first contact)
    first_contact = (env.feet_air_time > 0.0) * contact_filt
    env.feet_air_time *= ~contact_filt

    # Reset contact time when foot leaves contact (first air)
    first_air = (env.feet_contact_time > 0.0) * in_air
    env.feet_contact_time *= ~in_air

    return reward
