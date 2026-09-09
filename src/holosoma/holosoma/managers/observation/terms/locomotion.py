"""Basic locomotion observation terms.

These functions compute individual observation components for legged locomotion tasks.
Each function mirrors the manager-based observation pipeline that replaced the legacy direct `_get_obs_*()` helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from typing import Any, Sequence

import torch

from holosoma.utils.rotations import quat_rotate_inverse
from holosoma.utils.torch_utils import get_axis_params, to_torch

if TYPE_CHECKING:
    from holosoma.envs.locomotion.locomotion_manager import LeggedRobotLocomotionManager
    from holosoma.managers.command.terms.locomotion import LocomotionGait


def _base_quat(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    return env.base_quat


def _as_obs_matrix(
    env: LeggedRobotLocomotionManager,
    value: torch.Tensor | Any | None,
    *,
    feature_dim: int | None = None,
    fill: float = 0.0,
) -> torch.Tensor:
    """Convert arbitrary data to a [num_envs, obs_dim] float tensor."""
    num_envs = env.num_envs
    device = env.device
    if value is None:
        dim = 0 if feature_dim is None else feature_dim
        return torch.full((num_envs, dim), fill, device=device, dtype=torch.float32)

    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=device)
    tensor = tensor.to(device=device, dtype=torch.float32)

    if tensor.ndim == 0:
        return tensor.reshape(1, 1).expand(num_envs, 1)

    if tensor.ndim == 1:
        if tensor.shape[0] == num_envs and (feature_dim is None or feature_dim == 1):
            return tensor.unsqueeze(1)
        return tensor.reshape(1, -1).expand(num_envs, -1)

    if tensor.shape[0] != num_envs:
        tensor = tensor.reshape(1, -1).expand(num_envs, -1)
    else:
        tensor = tensor.reshape(num_envs, -1)

    if feature_dim is not None and tensor.shape[1] != feature_dim:
        if tensor.shape[1] > feature_dim:
            tensor = tensor[:, :feature_dim]
        else:
            pad = torch.full((num_envs, feature_dim - tensor.shape[1]), fill, device=device, dtype=tensor.dtype)
            tensor = torch.cat([tensor, pad], dim=1)
    return tensor


def _zeros(env: LeggedRobotLocomotionManager, dim: int) -> torch.Tensor:
    return torch.zeros((env.num_envs, dim), device=env.device, dtype=torch.float32)


def _get_manager_state(env: LeggedRobotLocomotionManager, manager_name: str, state_name: str) -> Any | None:
    manager = getattr(env, manager_name, None)
    if manager is None:
        return None
    get_state = getattr(manager, "get_state", None)
    if not callable(get_state):
        return None
    return get_state(state_name)


def _get_action_term(env: LeggedRobotLocomotionManager) -> Any | None:
    action_manager = getattr(env, "action_manager", None)
    if action_manager is None:
        return None

    get_term = getattr(action_manager, "get_term", None)
    if callable(get_term):
        try:
            return get_term("joint_control")
        except Exception:  # pragma: no cover - defensive
            pass

    iter_terms = getattr(action_manager, "iter_terms", None)
    if callable(iter_terms):
        for _, term in iter_terms():
            if hasattr(term, "torques"):
                return term
    return None


def _terrain_state(env: LeggedRobotLocomotionManager) -> Any | None:
    return _get_manager_state(env, "terrain_manager", "locomotion_terrain")


def _num_bodies(env: LeggedRobotLocomotionManager) -> int:
    num_bodies = getattr(env, "num_bodies", None)
    if isinstance(num_bodies, int):
        return num_bodies
    sim = getattr(env, "simulator", None)
    body_list = getattr(sim, "_body_list", None)
    if body_list is not None:
        return len(body_list)
    return 0


def _get_body_index_map(env: LeggedRobotLocomotionManager) -> dict[str, int]:
    cache_attr = "_oracle_body_index_map"
    cached = getattr(env, cache_attr, None)
    if isinstance(cached, dict):
        return cached

    sim = getattr(env, "simulator", None)
    body_list = getattr(sim, "_body_list", None)
    if body_list is None:
        body_index_map: dict[str, int] = {}
    else:
        body_index_map = {str(name): idx for idx, name in enumerate(body_list)}
    setattr(env, cache_attr, body_index_map)
    return body_index_map


def _gather_body_values_by_names(
    env: LeggedRobotLocomotionManager,
    body_values: torch.Tensor | Any | None,
    body_names: list[str],
    *,
    fill: float,
) -> torch.Tensor:
    """Gather per-body values by body names with deterministic output dimension."""
    if len(body_names) == 0:
        return _zeros(env, 0)

    values_obs = _as_obs_matrix(env, body_values, feature_dim=_num_bodies(env), fill=fill)
    gathered = torch.full((env.num_envs, len(body_names)), fill, device=env.device, dtype=torch.float32)
    body_index_map = _get_body_index_map(env)
    for col, body_name in enumerate(body_names):
        body_index = body_index_map.get(body_name)
        if body_index is not None and body_index < values_obs.shape[1]:
            gathered[:, col] = values_obs[:, body_index]
    return gathered


def gravity_vector(env: LeggedRobotLocomotionManager, up_axis_idx: int = 2) -> torch.Tensor:
    axis = to_torch(get_axis_params(-1.0, up_axis_idx), device=env.device)
    return axis.unsqueeze(0).expand(env.num_envs, -1)


def base_forward_vector(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    axis = to_torch([1.0, 0.0, 0.0], device=env.device)
    return axis.unsqueeze(0).expand(env.num_envs, -1)


def get_base_lin_vel(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    root_states = env.simulator.robot_root_states
    lin_vel_world = root_states[:, 7:10]
    return quat_rotate_inverse(_base_quat(env), lin_vel_world, w_last=True)


def get_base_ang_vel(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    ang_vel_world = env.simulator.robot_root_states[:, 10:13]
    return quat_rotate_inverse(_base_quat(env), ang_vel_world, w_last=True)


def get_projected_gravity(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    return quat_rotate_inverse(_base_quat(env), gravity_vector(env), w_last=True)


def base_lin_vel(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Base linear velocity in base frame.

    Returns:
        Tensor of shape [num_envs, 3]

    Equivalent to:
        env._get_obs_base_lin_vel()
    """
    return get_base_lin_vel(env)


def base_ang_vel(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Base angular velocity in base frame.

    Returns:
        Tensor of shape [num_envs, 3]

    Equivalent to:
        env._get_obs_base_ang_vel()
    """
    return get_base_ang_vel(env)


def projected_gravity(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Gravity vector projected into base frame.

    Returns:
        Tensor of shape [num_envs, 3]

    Equivalent to:
        env._get_obs_projected_gravity()
    """
    return get_projected_gravity(env)


def dof_pos(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Joint positions relative to default positions.

    Returns:
        Tensor of shape [num_envs, num_dof]

    Equivalent to:
        env._get_obs_dof_pos()
    """
    return env.simulator.dof_pos - env.default_dof_pos


def dof_vel(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Joint velocities.

    Returns:
        Tensor of shape [num_envs, num_dof]

    Equivalent to:
        env._get_obs_dof_vel()
    """
    return env.simulator.dof_vel


def actions(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Last actions taken by the policy.

    Returns:
        Tensor of shape [num_envs, num_actions]

    Equivalent to:
        env._get_obs_actions()
    """
    return env.action_manager.action


def prev_actions(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Actions from the previous policy step."""
    action_manager = getattr(env, "action_manager", None)
    if action_manager is None:
        return _zeros(env, env.num_dof)
    return action_manager.prev_action


def command_lin_vel(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Commanded linear velocity (x, y).

    Returns:
        Tensor of shape [num_envs, 2]

    Equivalent to:
        env.command_manager.commands[:, :2]
    """
    return env.command_manager.commands[:, :2]


def command_ang_vel(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Commanded angular velocity (yaw).

    Returns:
        Tensor of shape [num_envs, 1]

    Equivalent to:
        env.command_manager.commands[:, 2:3]
    """
    return env.command_manager.commands[:, 2:3]


def sin_phase(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Sine of the gait phase.

    Returns:
        Tensor of shape [num_envs, 1]

    Note: Requires env to have 'phase' attribute (e.g., LeggedRobotLocomotionManager)
    """
    gait_state = env.command_manager.get_state("locomotion_gait")
    if gait_state is None:
        raise AttributeError("locomotion_gait is not registered with the command manager.")
    gait_state = cast("LocomotionGait", gait_state)
    phase = gait_state.phase
    if phase is None:
        raise RuntimeError("Gait phase tensor has not been initialized.")
    return torch.sin(phase)


def cos_phase(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Cosine of the gait phase.

    Returns:
        Tensor of shape [num_envs, 1]

    Note: Requires env to have 'phase' attribute (e.g., LeggedRobotLocomotionManager)
    """
    gait_state = env.command_manager.get_state("locomotion_gait")
    if gait_state is None:
        raise AttributeError("locomotion_gait is not registered with the command manager.")
    gait_state = cast("LocomotionGait", gait_state)
    phase = gait_state.phase
    if phase is None:
        raise RuntimeError("Gait phase tensor has not been initialized.")
    return torch.cos(phase)


def oracle_root_state(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Oracle root/base state including world and base-frame motion."""
    simulator = env.simulator
    root_states = getattr(simulator, "robot_root_states", None)
    if root_states is None or root_states.shape[1] < 13:
        root_pos = _zeros(env, 3)
        base_quat_obs = _zeros(env, 4)
        root_lin_vel_world = _zeros(env, 3)
        root_ang_vel_world = _zeros(env, 3)
    else:
        root_pos = _as_obs_matrix(env, root_states[:, :3], feature_dim=3)
        base_quat_obs = _as_obs_matrix(env, getattr(env, "base_quat", root_states[:, 3:7]), feature_dim=4)
        root_lin_vel_world = _as_obs_matrix(env, root_states[:, 7:10], feature_dim=3)
        root_ang_vel_world = _as_obs_matrix(env, root_states[:, 10:13], feature_dim=3)

    base_lin_vel_obs = _as_obs_matrix(env, get_base_lin_vel(env), feature_dim=3)
    base_ang_vel_obs = _as_obs_matrix(env, get_base_ang_vel(env), feature_dim=3)
    projected_gravity_obs = _as_obs_matrix(env, get_projected_gravity(env), feature_dim=3)
    base_linear_acc_obs = _as_obs_matrix(env, getattr(simulator, "base_linear_acc", None), feature_dim=3, fill=0.0)

    return torch.cat(
        [
            root_pos,
            base_quat_obs,
            root_lin_vel_world,
            root_ang_vel_world,
            base_lin_vel_obs,
            base_ang_vel_obs,
            projected_gravity_obs,
            base_linear_acc_obs,
        ],
        dim=-1,
    )


def oracle_joint_state(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Oracle joint state including derivatives and default offsets."""
    num_dof = env.num_dof
    simulator = env.simulator

    dof_pos_obs = _as_obs_matrix(env, getattr(simulator, "dof_pos", None), feature_dim=num_dof, fill=0.0)
    dof_vel_obs = _as_obs_matrix(env, getattr(simulator, "dof_vel", None), feature_dim=num_dof, fill=0.0)
    dof_acc_obs = _as_obs_matrix(env, getattr(simulator, "dof_acc", None), feature_dim=num_dof, fill=0.0)
    default_dof_pos_obs = _as_obs_matrix(env, getattr(env, "default_dof_pos", None), feature_dim=num_dof, fill=0.0)

    default_dof_pos_base = getattr(env, "default_dof_pos_base", None)
    if default_dof_pos_base is None:
        default_dof_bias_obs = _zeros(env, num_dof)
    else:
        default_dof_base_obs = _as_obs_matrix(env, default_dof_pos_base, feature_dim=num_dof, fill=0.0)
        default_dof_bias_obs = default_dof_pos_obs - default_dof_base_obs

    return torch.cat(
        [dof_pos_obs, dof_vel_obs, dof_acc_obs, default_dof_pos_obs, default_dof_bias_obs],
        dim=-1,
    )


def oracle_actuator_state(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Oracle actuator/control internals, delays, gains, and limits."""
    num_dof = env.num_dof
    num_envs = env.num_envs

    action_manager = getattr(env, "action_manager", None)
    action_term = _get_action_term(env)

    actions_obs = _as_obs_matrix(
        env,
        getattr(action_manager, "action", None) if action_manager is not None else None,
        feature_dim=num_dof,
        fill=0.0,
    )
    prev_actions_obs = _as_obs_matrix(
        env,
        getattr(action_manager, "prev_action", None) if action_manager is not None else None,
        feature_dim=num_dof,
        fill=0.0,
    )
    actions_after_delay_obs = _as_obs_matrix(
        env,
        getattr(action_term, "_actions_after_delay", None) if action_term is not None else None,
        feature_dim=num_dof,
        fill=0.0,
    )

    action_queue = getattr(action_term, "action_queue", None) if action_term is not None else None
    if action_queue is None:
        delay_range = getattr(env, "_ctrl_delay_step_range", [0, 0])
        queue_len = int(delay_range[1]) + 1 if len(delay_range) > 1 else 1
        action_queue_obs = _zeros(env, max(queue_len, 1) * num_dof)
    else:
        action_queue_obs = _as_obs_matrix(env, action_queue, fill=0.0)

    action_delay_idx_obs = _as_obs_matrix(env, getattr(env, "action_delay_idx", None), feature_dim=1, fill=0.0)

    applied_torque_src = getattr(action_term, "torques", None) if action_term is not None else None
    if applied_torque_src is None:
        applied_torque_src = getattr(env.simulator, "dof_forces", None)
    applied_torques_obs = _as_obs_matrix(env, applied_torque_src, feature_dim=num_dof, fill=0.0)

    prev_dof_vel_obs = _as_obs_matrix(
        env,
        getattr(action_term, "get_prev_dof_vel", lambda: None)() if action_term is not None else None,
        feature_dim=num_dof,
        fill=0.0,
    )

    p_gains_obs = _as_obs_matrix(env, getattr(env, "p_gains", None), feature_dim=num_dof, fill=0.0)
    d_gains_obs = _as_obs_matrix(env, getattr(env, "d_gains", None), feature_dim=num_dof, fill=0.0)
    action_scales_obs = _as_obs_matrix(env, getattr(env, "action_scales", None), feature_dim=num_dof, fill=0.0)
    torque_limits_obs = _as_obs_matrix(env, getattr(env, "torque_limits", None), feature_dim=num_dof, fill=0.0)

    actuator_state = _get_manager_state(env, "randomization_manager", "actuator_randomizer_state")
    kp_scale_obs = _as_obs_matrix(
        env,
        getattr(actuator_state, "kp_scale", None) if actuator_state is not None else None,
        feature_dim=num_dof,
        fill=1.0,
    )
    kd_scale_obs = _as_obs_matrix(
        env,
        getattr(actuator_state, "kd_scale", None) if actuator_state is not None else None,
        feature_dim=num_dof,
        fill=1.0,
    )
    rfi_lim_scale_obs = _as_obs_matrix(
        env,
        getattr(actuator_state, "rfi_lim_scale", None) if actuator_state is not None else None,
        feature_dim=num_dof,
        fill=1.0,
    )

    rfi_enabled = bool(getattr(action_term, "_randomize_torque_rfi", False)) if action_term is not None else False
    rfi_lim = float(getattr(action_term, "_rfi_lim", 0.0)) if action_term is not None else 0.0
    if not rfi_enabled and hasattr(env, "_pending_torque_rfi"):
        rfi_enabled = bool(env._pending_torque_rfi[0])
        rfi_lim = float(env._pending_torque_rfi[1])
    torque_rfi_meta = torch.tensor(
        [float(rfi_enabled), float(rfi_lim)],
        device=env.device,
        dtype=torch.float32,
    ).view(1, 2).expand(num_envs, -1)

    return torch.cat(
        [
            actions_obs,
            prev_actions_obs,
            actions_after_delay_obs,
            action_queue_obs,
            action_delay_idx_obs,
            applied_torques_obs,
            prev_dof_vel_obs,
            p_gains_obs,
            d_gains_obs,
            action_scales_obs,
            torque_limits_obs,
            kp_scale_obs,
            kd_scale_obs,
            rfi_lim_scale_obs,
            torque_rfi_meta,
        ],
        dim=-1,
    )


def oracle_contact_state(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Oracle contact force tensors and binary foot-contact indicators."""
    simulator = env.simulator
    num_bodies = _num_bodies(env)

    contact_forces = getattr(simulator, "contact_forces", None)
    if contact_forces is None:
        contact_forces_flat = _zeros(env, num_bodies * 3)
        feet_contact_binary = _zeros(env, len(getattr(env, "feet_indices", [])))
    else:
        contact_forces_flat = _as_obs_matrix(env, contact_forces, fill=0.0)
        feet_indices = getattr(env, "feet_indices", None)
        if feet_indices is None:
            feet_contact_binary = _zeros(env, 0)
        else:
            if not isinstance(feet_indices, torch.Tensor):
                feet_indices = torch.as_tensor(feet_indices, device=env.device, dtype=torch.long)
            else:
                feet_indices = feet_indices.to(device=env.device, dtype=torch.long)
            if feet_indices.numel() == 0:
                feet_contact_binary = _zeros(env, 0)
            else:
                feet_forces = contact_forces[:, feet_indices, :]
                feet_contact_binary = (torch.linalg.norm(feet_forces, dim=-1) > 1.0).float()

    contact_history = getattr(simulator, "contact_forces_history", None)
    contact_history_flat = _as_obs_matrix(env, contact_history, fill=0.0)

    return torch.cat([contact_forces_flat, contact_history_flat, feet_contact_binary], dim=-1)


def oracle_rigidbody_state(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Oracle full rigid-body position/orientation/velocity tensors."""
    simulator = env.simulator
    num_bodies = _num_bodies(env)

    rb_pos = _as_obs_matrix(env, getattr(simulator, "_rigid_body_pos", None), feature_dim=num_bodies * 3, fill=0.0)
    rb_rot = _as_obs_matrix(env, getattr(simulator, "_rigid_body_rot", None), feature_dim=num_bodies * 4, fill=0.0)
    rb_vel = _as_obs_matrix(env, getattr(simulator, "_rigid_body_vel", None), feature_dim=num_bodies * 3, fill=0.0)
    rb_ang_vel = _as_obs_matrix(
        env,
        getattr(simulator, "_rigid_body_ang_vel", None),
        feature_dim=num_bodies * 3,
        fill=0.0,
    )

    return torch.cat([rb_pos, rb_rot, rb_vel, rb_ang_vel], dim=-1)


def oracle_terrain_state(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Oracle terrain context, sampled heights, and root-origin relations."""
    terrain_state = _terrain_state(env)
    simulator = env.simulator

    terrain_base_heights = _as_obs_matrix(
        env,
        getattr(terrain_state, "base_heights", None) if terrain_state is not None else None,
        feature_dim=1,
        fill=0.0,
    )
    terrain_feet_heights = _as_obs_matrix(
        env,
        getattr(terrain_state, "feet_heights", None) if terrain_state is not None else None,
        fill=0.0,
    )
    terrain_ray_hits = _as_obs_matrix(
        env,
        getattr(terrain_state, "_ray_hits_world_base", None) if terrain_state is not None else None,
        fill=0.0,
    )
    env_origins = _as_obs_matrix(
        env,
        getattr(terrain_state, "env_origins", None) if terrain_state is not None else None,
        feature_dim=3,
        fill=0.0,
    )

    root_states = getattr(simulator, "robot_root_states", None)
    root_pos = _as_obs_matrix(env, root_states[:, :3] if root_states is not None else None, feature_dim=3, fill=0.0)
    root_xy_rel_origin = root_pos[:, :2] - env_origins[:, :2]
    root_z_minus_base_height = root_pos[:, 2:3] - terrain_base_heights

    return torch.cat(
        [
            terrain_base_heights,
            terrain_feet_heights,
            terrain_ray_hits,
            env_origins,
            root_xy_rel_origin,
            root_z_minus_base_height,
        ],
        dim=-1,
    )


def oracle_command_gait_state(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Oracle command tensor, gait latent state, and resample metadata."""
    num_envs = env.num_envs
    command_manager = getattr(env, "command_manager", None)
    commands_obs = _as_obs_matrix(
        env,
        getattr(command_manager, "commands", None) if command_manager is not None else None,
        fill=0.0,
    )

    gait_state = _get_manager_state(env, "command_manager", "locomotion_gait")
    phase_obs = _as_obs_matrix(env, getattr(gait_state, "phase", None) if gait_state is not None else None, fill=0.0)
    sin_phase_obs = torch.sin(phase_obs)
    cos_phase_obs = torch.cos(phase_obs)
    gait_freq_obs = _as_obs_matrix(
        env,
        getattr(gait_state, "gait_freq", None) if gait_state is not None else None,
        feature_dim=1,
        fill=0.0,
    )
    phase_dt_obs = _as_obs_matrix(
        env,
        getattr(gait_state, "phase_dt", None) if gait_state is not None else None,
        feature_dim=1,
        fill=0.0,
    )
    phase_offset_obs = _as_obs_matrix(
        env,
        getattr(gait_state, "phase_offset", None) if gait_state is not None else None,
        fill=0.0,
    )

    if commands_obs.shape[1] >= 3:
        stand_mask = torch.logical_and(
            torch.linalg.norm(commands_obs[:, :2], dim=1) < 0.01,
            torch.abs(commands_obs[:, 2]) < 0.01,
        ).float().unsqueeze(1)
    else:
        stand_mask = _zeros(env, 1)

    resample_time = 0.0
    if command_manager is not None and getattr(command_manager, "command_cfg", None) is not None:
        resample_time = float(getattr(command_manager.command_cfg, "locomotion_command_resampling_time", 0.0) or 0.0)
    resample_steps = float(int(resample_time / env.dt)) if env.dt > 0 and resample_time > 0 else 0.0
    resample_steps_obs = torch.full((num_envs, 1), resample_steps, device=env.device, dtype=torch.float32)

    return torch.cat(
        [
            commands_obs,
            phase_obs,
            sin_phase_obs,
            cos_phase_obs,
            gait_freq_obs,
            phase_dt_obs,
            phase_offset_obs,
            stand_mask,
            resample_steps_obs,
        ],
        dim=-1,
    )


def oracle_randomization_state(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Oracle domain-randomization truth values and scheduling state."""
    num_envs = env.num_envs
    num_bodies = _num_bodies(env)

    body_mass_obs = _as_obs_matrix(env, getattr(env, "oracle_body_mass", None), feature_dim=num_bodies, fill=0.0)
    link_mass_scale_obs = _as_obs_matrix(
        env,
        getattr(env, "oracle_link_mass_scale", None),
        feature_dim=num_bodies,
        fill=1.0,
    )
    base_mass_delta_obs = _as_obs_matrix(env, getattr(env, "oracle_base_mass_delta", None), feature_dim=1, fill=0.0)
    friction_coeff_obs = _as_obs_matrix(env, getattr(env, "oracle_friction_coeff", None), feature_dim=1, fill=1.0)
    base_com_bias_obs = _as_obs_matrix(env, getattr(env, "oracle_base_com_bias", None), feature_dim=3, fill=0.0)

    push_state = _get_manager_state(env, "randomization_manager", "push_randomizer_state")
    push_interval_obs = _as_obs_matrix(
        env,
        getattr(push_state, "push_interval_s", None) if push_state is not None else None,
        feature_dim=1,
        fill=0.0,
    )
    push_counter_obs = _as_obs_matrix(
        env,
        getattr(push_state, "push_robot_counter", None) if push_state is not None else None,
        feature_dim=1,
        fill=0.0,
    )

    if env.dt > 0:
        interval_steps = torch.clamp((push_interval_obs / env.dt).to(torch.int64), min=1)
        push_due_obs = (push_counter_obs.to(torch.int64) == interval_steps).float()
    else:
        push_due_obs = _zeros(env, 1)

    if push_state is not None:
        max_push_vel_obs = _as_obs_matrix(env, getattr(push_state, "max_push_vel", None), fill=0.0)
    else:
        max_push_vel_obs = _as_obs_matrix(env, getattr(env, "_max_push_vel", None), fill=0.0)
    if max_push_vel_obs.shape[1] == 0:
        max_push_vel_obs = _zeros(env, 2)
    last_push_vel_obs = _as_obs_matrix(
        env,
        getattr(env, "record_push_robot_vel_buf", None),
        feature_dim=max_push_vel_obs.shape[1],
        fill=0.0,
    )

    action_term = _get_action_term(env)
    torque_rfi_enabled = bool(getattr(action_term, "_randomize_torque_rfi", False)) if action_term is not None else False
    if not torque_rfi_enabled and hasattr(env, "_pending_torque_rfi"):
        torque_rfi_enabled = bool(env._pending_torque_rfi[0])
    actuator_state = _get_manager_state(env, "randomization_manager", "actuator_randomizer_state")
    dr_flags = torch.tensor(
        [
            float(bool(getattr(env, "_randomize_ctrl_delay", False))),
            float(bool(getattr(env, "_randomize_friction", False))),
            float(bool(getattr(env, "_randomize_link_mass", False))),
            float(bool(getattr(env, "_randomize_base_mass", False))),
            float(bool(getattr(env, "_randomize_push_robots", False) or getattr(env, "_push_robots_enabled", False))),
            float(bool(getattr(env, "_randomize_base_com", False))),
            float(torque_rfi_enabled),
            float(bool(getattr(env, "_randomize_dof_pos_bias", False))),
            float(bool(getattr(actuator_state, "enable_pd_gain", False) if actuator_state is not None else False)),
            float(bool(getattr(actuator_state, "enable_rfi_lim", False) if actuator_state is not None else False)),
        ],
        device=env.device,
        dtype=torch.float32,
    ).view(1, -1).expand(num_envs, -1)

    return torch.cat(
        [
            body_mass_obs,
            link_mass_scale_obs,
            base_mass_delta_obs,
            friction_coeff_obs,
            base_com_bias_obs,
            push_interval_obs,
            push_counter_obs,
            push_due_obs,
            max_push_vel_obs,
            last_push_vel_obs,
            dr_flags,
        ],
        dim=-1,
    )


def oracle_actuator_compact_state(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Compact privileged actuator/control state focused on dynamics-relevant internals."""
    num_dof = env.num_dof
    num_envs = env.num_envs
    action_manager = getattr(env, "action_manager", None)
    action_term = _get_action_term(env)

    prev_actions_obs = _as_obs_matrix(
        env,
        getattr(action_manager, "prev_action", None) if action_manager is not None else None,
        feature_dim=num_dof,
        fill=0.0,
    )
    actions_after_delay_obs = _as_obs_matrix(
        env,
        getattr(action_term, "_actions_after_delay", None) if action_term is not None else None,
        feature_dim=num_dof,
        fill=0.0,
    )
    action_delay_idx_obs = _as_obs_matrix(env, getattr(env, "action_delay_idx", None), feature_dim=1, fill=0.0)

    applied_torque_src = getattr(action_term, "torques", None) if action_term is not None else None
    if applied_torque_src is None:
        applied_torque_src = getattr(env.simulator, "dof_forces", None)
    applied_torques_obs = _as_obs_matrix(env, applied_torque_src, feature_dim=num_dof, fill=0.0)
    torque_limits_obs = _as_obs_matrix(env, getattr(env, "torque_limits", None), feature_dim=num_dof, fill=1.0)
    normalized_torques = applied_torques_obs / torch.clamp(torch.abs(torque_limits_obs), min=1e-6)
    normalized_torques = torch.clamp(normalized_torques, -5.0, 5.0)

    actuator_state = _get_manager_state(env, "randomization_manager", "actuator_randomizer_state")
    kp_scale_obs = _as_obs_matrix(
        env,
        getattr(actuator_state, "kp_scale", None) if actuator_state is not None else None,
        feature_dim=num_dof,
        fill=1.0,
    )
    kd_scale_obs = _as_obs_matrix(
        env,
        getattr(actuator_state, "kd_scale", None) if actuator_state is not None else None,
        feature_dim=num_dof,
        fill=1.0,
    )
    rfi_lim_scale_obs = _as_obs_matrix(
        env,
        getattr(actuator_state, "rfi_lim_scale", None) if actuator_state is not None else None,
        feature_dim=num_dof,
        fill=1.0,
    )

    rfi_enabled = bool(getattr(action_term, "_randomize_torque_rfi", False)) if action_term is not None else False
    rfi_lim = float(getattr(action_term, "_rfi_lim", 0.0)) if action_term is not None else 0.0
    if not rfi_enabled and hasattr(env, "_pending_torque_rfi"):
        rfi_enabled = bool(env._pending_torque_rfi[0])
        rfi_lim = float(env._pending_torque_rfi[1])
    torque_rfi_meta = torch.tensor(
        [float(rfi_enabled), float(rfi_lim)],
        device=env.device,
        dtype=torch.float32,
    ).view(1, 2).expand(num_envs, -1)

    return torch.cat(
        [
            prev_actions_obs,
            actions_after_delay_obs,
            action_delay_idx_obs,
            normalized_torques,
            kp_scale_obs,
            kd_scale_obs,
            rfi_lim_scale_obs,
            torque_rfi_meta,
        ],
        dim=-1,
    )


def oracle_contact_compact_state(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Compact contact state using only foot contact forces and binary contacts."""
    num_feet = int(getattr(env, "num_feet", 0))
    if num_feet <= 0:
        feet_indices_attr = getattr(env, "feet_indices", None)
        num_feet = int(len(feet_indices_attr)) if feet_indices_attr is not None else 2

    feet_force_obs = _zeros(env, num_feet * 3)
    feet_contact_binary = _zeros(env, num_feet)

    contact_forces = getattr(env.simulator, "contact_forces", None)
    feet_indices = getattr(env, "feet_indices", None)
    if contact_forces is None or feet_indices is None:
        return torch.cat([feet_force_obs, feet_contact_binary], dim=-1)

    if not isinstance(feet_indices, torch.Tensor):
        feet_indices = torch.as_tensor(feet_indices, device=env.device, dtype=torch.long)
    else:
        feet_indices = feet_indices.to(device=env.device, dtype=torch.long)
    if feet_indices.numel() == 0:
        return torch.cat([feet_force_obs, feet_contact_binary], dim=-1)

    feet_forces = contact_forces[:, feet_indices, :].to(device=env.device, dtype=torch.float32)
    feet_force_obs = torch.clamp(feet_forces / 100.0, -5.0, 5.0).reshape(env.num_envs, -1)
    feet_force_obs = _as_obs_matrix(env, feet_force_obs, feature_dim=num_feet * 3, fill=0.0)
    feet_contact_binary = (torch.linalg.norm(feet_forces, dim=-1) > 1.0).float()
    feet_contact_binary = _as_obs_matrix(env, feet_contact_binary, feature_dim=num_feet, fill=0.0)
    return torch.cat([feet_force_obs, feet_contact_binary], dim=-1)


def oracle_terrain_compact_state(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Compact terrain state with local terrain profile and root clearance."""
    terrain_state = _terrain_state(env)

    terrain_base_heights = _as_obs_matrix(
        env,
        getattr(terrain_state, "base_heights", None) if terrain_state is not None else None,
        feature_dim=1,
        fill=0.0,
    )
    terrain_feet_heights = _as_obs_matrix(
        env,
        getattr(terrain_state, "feet_heights", None) if terrain_state is not None else None,
        fill=0.0,
    )

    root_states = getattr(env.simulator, "robot_root_states", None)
    root_z = _as_obs_matrix(
        env,
        root_states[:, 2:3] if root_states is not None else None,
        feature_dim=1,
        fill=0.0,
    )
    root_z_minus_base_height = root_z - terrain_base_heights

    num_base_height_points = int(getattr(terrain_state, "_num_base_height_points", 0) if terrain_state is not None else 0)
    ray_hits_world = getattr(terrain_state, "_ray_hits_world_base", None) if terrain_state is not None else None
    if isinstance(ray_hits_world, torch.Tensor) and ray_hits_world.ndim == 3:
        local_terrain_profile = root_z - ray_hits_world[..., 2].to(device=env.device, dtype=torch.float32)
        local_terrain_profile = local_terrain_profile - terrain_base_heights
    else:
        local_terrain_profile = _zeros(env, num_base_height_points)
    local_terrain_profile = torch.clamp(local_terrain_profile, -1.0, 1.0)

    return torch.cat(
        [
            terrain_base_heights,
            terrain_feet_heights,
            local_terrain_profile,
            root_z_minus_base_height,
        ],
        dim=-1,
    )


def oracle_local_height_map(env: LeggedRobotLocomotionManager, clip_value: float = 1.0) -> torch.Tensor:
    """Local terrain elevation profile around the base in meters (no normalization)."""
    terrain_state = _terrain_state(env)
    num_base_height_points = int(getattr(terrain_state, "_num_base_height_points", 0) if terrain_state is not None else 0)
    ray_hits_world = getattr(terrain_state, "_ray_hits_world_base", None) if terrain_state is not None else None
    if num_base_height_points <= 0:
        return _zeros(env, 0)

    root_states = getattr(env.simulator, "robot_root_states", None)
    root_z = _as_obs_matrix(
        env,
        root_states[:, 2:3] if root_states is not None else None,
        feature_dim=1,
        fill=0.0,
    )
    terrain_base_heights = _as_obs_matrix(
        env,
        getattr(terrain_state, "base_heights", None) if terrain_state is not None else None,
        feature_dim=1,
        fill=0.0,
    )
    if isinstance(ray_hits_world, torch.Tensor) and ray_hits_world.ndim == 3:
        local_terrain_profile = root_z - ray_hits_world[..., 2].to(device=env.device, dtype=torch.float32)
        local_terrain_profile = local_terrain_profile - terrain_base_heights
    else:
        local_terrain_profile = _zeros(env, num_base_height_points)

    if clip_value > 0:
        local_terrain_profile = torch.clamp(local_terrain_profile, -clip_value, clip_value)
    return _as_obs_matrix(
        env,
        local_terrain_profile,
        feature_dim=num_base_height_points,
        fill=0.0,
    )


def oracle_external_push_vector(env: LeggedRobotLocomotionManager, clip_value: float = 5.0) -> torch.Tensor:
    """Per-env external push vector applied by the randomizer."""
    push_vec = _as_obs_matrix(env, getattr(env, "record_push_robot_vel_buf", None), feature_dim=2, fill=0.0)
    if clip_value > 0:
        push_vec = torch.clamp(push_vec, -clip_value, clip_value)
    return push_vec


def oracle_command_gait_compact_state(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Compact command and gait latent state."""
    command_manager = getattr(env, "command_manager", None)
    commands_obs = _as_obs_matrix(
        env,
        getattr(command_manager, "commands", None) if command_manager is not None else None,
        fill=0.0,
    )
    if commands_obs.shape[1] >= 3:
        commands_obs = commands_obs[:, :3]
    else:
        pad = _zeros(env, 3 - commands_obs.shape[1])
        commands_obs = torch.cat([commands_obs, pad], dim=-1)

    gait_state = _get_manager_state(env, "command_manager", "locomotion_gait")
    phase_obs = _as_obs_matrix(env, getattr(gait_state, "phase", None) if gait_state is not None else None, fill=0.0)
    if phase_obs.shape[1] >= 2:
        phase_obs = phase_obs[:, :2]
    else:
        phase_obs = torch.cat([phase_obs, _zeros(env, 2 - phase_obs.shape[1])], dim=-1)
    sin_phase_obs = torch.sin(phase_obs)
    cos_phase_obs = torch.cos(phase_obs)
    gait_freq_obs = _as_obs_matrix(
        env,
        getattr(gait_state, "gait_freq", None) if gait_state is not None else None,
        feature_dim=1,
        fill=0.0,
    )
    phase_dt_obs = _as_obs_matrix(
        env,
        getattr(gait_state, "phase_dt", None) if gait_state is not None else None,
        feature_dim=1,
        fill=0.0,
    )

    stand_mask = torch.logical_and(
        torch.linalg.norm(commands_obs[:, :2], dim=1) < 0.01,
        torch.abs(commands_obs[:, 2]) < 0.01,
    ).float().unsqueeze(1)

    return torch.cat(
        [
            commands_obs,
            sin_phase_obs,
            cos_phase_obs,
            gait_freq_obs,
            phase_dt_obs,
            stand_mask,
        ],
        dim=-1,
    )


def _resolve_terrain_static_friction(env: LeggedRobotLocomotionManager) -> float:
    terrain_manager = getattr(env, "terrain_manager", None)
    terrain_cfg = getattr(terrain_manager, "cfg", None)
    terrain_term = getattr(terrain_cfg, "terrain_term", None)
    friction = float(getattr(terrain_term, "static_friction", 1.0) if terrain_term is not None else 1.0)
    return friction if abs(friction) > 1e-6 else 1.0


def _ratio_minus_one_with_fallback(
    values: torch.Tensor,
    nominal: torch.Tensor,
    *,
    clamp_abs: float,
) -> torch.Tensor:
    """Compute values / nominal - 1 with stable fallbacks for missing caches."""
    nominal_valid = torch.abs(nominal) > 1e-6
    value_valid = torch.abs(values) > 1e-6
    safe_nominal = torch.where(nominal_valid, nominal, torch.where(value_valid, values, torch.ones_like(values)))
    normalized = values / safe_nominal - 1.0
    unknown = torch.logical_not(nominal_valid | value_valid)
    normalized = torch.where(unknown, torch.zeros_like(normalized), normalized)
    return torch.clamp(normalized, -clamp_abs, clamp_abs)


def oracle_dr_compact_state(env: LeggedRobotLocomotionManager, normalize: bool = False) -> torch.Tensor:
    """Compact DR truth values for oracle learning with optional nominal normalization."""
    randomize_link_names = list(getattr(env.robot_config, "randomize_link_body_names", []) or [])
    torso_name = str(getattr(env.robot_config, "torso_name", "") or "")
    selected_body_names: list[str] = []
    for body_name in randomize_link_names + ([torso_name] if torso_name else []):
        if body_name and body_name not in selected_body_names:
            selected_body_names.append(body_name)

    selected_body_mass_obs = _gather_body_values_by_names(
        env,
        getattr(env, "oracle_body_mass", None),
        selected_body_names,
        fill=0.0,
    )
    selected_nominal_mass_obs = _gather_body_values_by_names(
        env,
        getattr(env, "oracle_nominal_body_mass", None),
        selected_body_names,
        fill=0.0,
    )

    friction_coeff_obs = _as_obs_matrix(env, getattr(env, "oracle_friction_coeff", None), feature_dim=1, fill=1.0)
    base_com_bias_obs = _as_obs_matrix(env, getattr(env, "oracle_base_com_bias", None), feature_dim=3, fill=0.0)

    push_state = _get_manager_state(env, "randomization_manager", "push_randomizer_state")
    push_interval_obs = _as_obs_matrix(
        env,
        getattr(push_state, "push_interval_s", None) if push_state is not None else None,
        feature_dim=1,
        fill=0.0,
    )
    push_counter_obs = _as_obs_matrix(
        env,
        getattr(push_state, "push_robot_counter", None) if push_state is not None else None,
        feature_dim=1,
        fill=0.0,
    )
    if env.dt > 0:
        interval_steps = torch.clamp(push_interval_obs / env.dt, min=1.0)
        push_progress_obs = torch.clamp(push_counter_obs / interval_steps, 0.0, 1.0)
    else:
        push_progress_obs = _zeros(env, 1)

    if push_state is not None:
        max_push_vel_obs = _as_obs_matrix(env, getattr(push_state, "max_push_vel", None), feature_dim=2, fill=0.0)
        nominal_max_push_vel = getattr(push_state, "nominal_max_push_vel", None)
        if nominal_max_push_vel is None:
            nominal_max_push_vel = getattr(push_state, "max_push_vel", None)
        nominal_max_push_vel_obs = _as_obs_matrix(
            env,
            nominal_max_push_vel,
            feature_dim=2,
            fill=0.0,
        )
        push_interval_range = getattr(push_state, "push_interval_range", None)
        if isinstance(push_interval_range, Sequence) and len(push_interval_range) >= 2:
            push_interval_nominal = 0.5 * (float(push_interval_range[0]) + float(push_interval_range[1]))
        else:
            push_interval_nominal = float(push_interval_obs.mean().item()) if push_interval_obs.numel() > 0 else 0.0
    else:
        max_push_vel_obs = _as_obs_matrix(env, getattr(env, "_max_push_vel", None), feature_dim=2, fill=0.0)
        nominal_max_push_vel_obs = _as_obs_matrix(env, getattr(env, "_max_push_vel", None), feature_dim=2, fill=0.0)
        push_interval_nominal = float(push_interval_obs.mean().item()) if push_interval_obs.numel() > 0 else 0.0

    if normalize:
        selected_body_mass_obs = _ratio_minus_one_with_fallback(
            selected_body_mass_obs,
            selected_nominal_mass_obs,
            clamp_abs=5.0,
        )

        friction_nominal = _resolve_terrain_static_friction(env)
        friction_coeff_obs = torch.clamp(friction_coeff_obs / friction_nominal - 1.0, -5.0, 5.0)

        if push_interval_nominal > 1e-6:
            push_interval_obs = torch.clamp(push_interval_obs / push_interval_nominal - 1.0, -5.0, 5.0)
        else:
            push_interval_obs = _zeros(env, 1)

        max_push_vel_obs = _ratio_minus_one_with_fallback(
            max_push_vel_obs,
            nominal_max_push_vel_obs,
            clamp_abs=5.0,
        )

    return torch.cat(
        [
            selected_body_mass_obs,
            friction_coeff_obs,
            base_com_bias_obs,
            push_interval_obs,
            push_progress_obs,
            max_push_vel_obs,
        ],
        dim=-1,
    )


def oracle_randomization_compact_state(env: LeggedRobotLocomotionManager, normalize: bool = False) -> torch.Tensor:
    """Backward-compatible alias for compact DR truth values."""
    return oracle_dr_compact_state(env, normalize=normalize)


def oracle_episode_state(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Oracle rollout metadata including progress/reset/eval flags."""
    num_envs = env.num_envs
    max_episode_length = float(getattr(env, "max_episode_length", 1.0))
    if max_episode_length <= 0:
        max_episode_length = 1.0

    episode_progress = _as_obs_matrix(env, env.episode_length_buf.to(torch.float32) / max_episode_length, feature_dim=1)
    episode_progress = torch.clamp(episode_progress, 0.0, 1.0)
    episode_remaining = 1.0 - episode_progress

    reset_obs = _as_obs_matrix(env, env.reset_buf, feature_dim=1, fill=0.0)
    timeout_obs = _as_obs_matrix(env, env.time_out_buf, feature_dim=1, fill=0.0)
    eval_flag_obs = torch.full((num_envs, 1), float(bool(getattr(env, "is_evaluating", False))), device=env.device)

    return torch.cat([episode_progress, episode_remaining, reset_obs, timeout_obs, eval_flag_obs], dim=-1)
