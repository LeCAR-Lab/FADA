from types import SimpleNamespace

import torch

from holosoma.managers.reward.terms import locomotion_unitree


def _dummy_env(
    *,
    commands: torch.Tensor,
    root_states: torch.Tensor,
    base_quat: torch.Tensor,
    gait_phase: torch.Tensor | None = None,
    contact_forces: torch.Tensor | None = None,
    body_names: list[str] | None = None,
    rigid_body_vel: torch.Tensor | None = None,
    feet_height_indices: torch.Tensor | None = None,
    feet_heights: torch.Tensor | None = None,
) -> SimpleNamespace:
    gait_state = SimpleNamespace(phase=gait_phase) if gait_phase is not None else None
    terrain_state = SimpleNamespace(feet_heights=feet_heights) if feet_heights is not None else None
    return SimpleNamespace(
        num_envs=commands.shape[0],
        device=commands.device,
        base_quat=base_quat,
        command_manager=SimpleNamespace(
            commands=commands,
            get_state=lambda name: gait_state if name == "locomotion_gait" else None,
        ),
        terrain_manager=SimpleNamespace(
            get_state=lambda name: terrain_state if name == "locomotion_terrain" else None,
        ),
        feet_height_indices=feet_height_indices,
        simulator=SimpleNamespace(
            robot_root_states=root_states,
            contact_forces=contact_forces,
            body_names=body_names or [],
            _rigid_body_vel=rigid_body_vel,
        ),
    )


def test_track_ang_vel_z_exp_matches_unitree_kernel():
    commands = torch.tensor([[0.2, 0.0, 0.3], [0.0, 0.0, -0.5]], dtype=torch.float32)
    root_states = torch.zeros((2, 13), dtype=torch.float32)
    root_states[:, 10:13] = torch.tensor([[0.0, 0.0, 0.3], [0.0, 0.0, -0.1]], dtype=torch.float32)
    base_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)

    env = _dummy_env(commands=commands, root_states=root_states, base_quat=base_quat)
    reward = locomotion_unitree.track_ang_vel_z_exp(env, std=0.5, command_name="base_velocity")

    expected_error = torch.tensor([0.0, 0.16], dtype=torch.float32)
    expected_reward = torch.exp(-expected_error / (0.5**2))
    assert torch.allclose(reward, expected_reward)


def test_base_height_l2_is_world_frame_squared_error():
    commands = torch.zeros((2, 3), dtype=torch.float32)
    root_states = torch.zeros((2, 13), dtype=torch.float32)
    root_states[:, 2] = torch.tensor([0.78, 0.70], dtype=torch.float32)
    base_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)

    env = _dummy_env(commands=commands, root_states=root_states, base_quat=base_quat)
    reward = locomotion_unitree.base_height_l2(env, target_height=0.78)

    expected = torch.tensor([0.0, (0.70 - 0.78) ** 2], dtype=torch.float32)
    assert torch.allclose(reward, expected)


def test_foot_clearance_reward_uses_terrain_relative_heights():
    commands = torch.zeros((2, 3), dtype=torch.float32)
    root_states = torch.zeros((2, 13), dtype=torch.float32)
    base_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
    feet_heights = torch.tensor([[0.05, 0.05], [0.05, 0.08]], dtype=torch.float32)
    feet_height_indices = torch.tensor([0, 1], dtype=torch.long)
    rigid_body_vel = torch.zeros((2, 2, 3), dtype=torch.float32)
    rigid_body_vel[0, :, :2] = torch.tensor([[0.4, 0.0], [0.4, 0.0]], dtype=torch.float32)
    rigid_body_vel[1, :, :2] = torch.tensor([[0.4, 0.0], [0.4, 0.0]], dtype=torch.float32)

    env = _dummy_env(
        commands=commands,
        root_states=root_states,
        base_quat=base_quat,
        feet_heights=feet_heights,
        feet_height_indices=feet_height_indices,
        rigid_body_vel=rigid_body_vel,
    )
    reward = locomotion_unitree.foot_clearance_reward(env, target_height=0.05, std=0.05, tanh_mult=2.0)

    foot_velocity_tanh = torch.tanh(2.0 * torch.norm(rigid_body_vel[:, :, :2], dim=2))
    expected_error = torch.square(feet_heights - 0.05) * foot_velocity_tanh
    expected = torch.exp(-torch.sum(expected_error, dim=1) / 0.05)
    assert torch.allclose(reward, expected)


def test_feet_gait_command_phase_uses_shared_gait_state():
    commands = torch.tensor([[0.3, 0.0, 0.0], [0.3, 0.0, 0.0]], dtype=torch.float32)
    root_states = torch.zeros((2, 13), dtype=torch.float32)
    base_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
    gait_phase = torch.tensor(
        [
            [0.25 * torch.pi, -0.75 * torch.pi],
            [-0.75 * torch.pi, 0.25 * torch.pi],
        ],
        dtype=torch.float32,
    )
    contact_forces = torch.zeros((2, 2, 3), dtype=torch.float32)
    contact_forces[0, 0, 2] = 5.0
    contact_forces[1, 1, 2] = 5.0

    env = _dummy_env(
        commands=commands,
        root_states=root_states,
        base_quat=base_quat,
        gait_phase=gait_phase,
        contact_forces=contact_forces,
        body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
    )
    reward = locomotion_unitree.feet_gait_command_phase(
        env,
        period=0.8,
        offset=[0.0, 0.5],
        threshold=0.55,
        command_name="base_velocity",
        body_name_patterns=[".*ankle_roll.*"],
        force_threshold=1.0,
    )

    assert torch.allclose(reward, torch.tensor([2.0, 2.0], dtype=torch.float32))
