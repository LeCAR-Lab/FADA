from __future__ import annotations

import numpy as np
import pytest

from holosoma_inference.utils.chunk_twin_publisher import (
    build_q_target_chunk_abs,
    recover_obs_pred_dof_pos_abs,
)


def test_build_q_target_chunk_abs_with_front_padding() -> None:
    actions_chunk = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    default_angles = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)

    out = build_q_target_chunk_abs(
        actions_chunk_raw=actions_chunk,
        policy_action_scale=0.5,
        default_dof_angles=default_angles,
        num_dofs=4,
    )

    expected = np.array(
        [
            [10.0, 20.0, 30.5, 41.0],
            [10.0, 20.0, 31.5, 42.0],
        ],
        dtype=np.float32,
    )
    assert out.shape == (2, 4)
    assert np.allclose(out, expected)


def test_recover_obs_pred_dof_pos_abs_with_inverse_scale_and_padding() -> None:
    # term layout: [base_ang_vel(3), dof_pos(2), dof_vel(2), projected_gravity(3)]
    pred_obs = np.array(
        [
            [0, 0, 0, 2, 4, 9, 9, 1, 1, 1],
            [0, 0, 0, 6, 8, 9, 9, 1, 1, 1],
        ],
        dtype=np.float32,
    )
    default_angles = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)

    out = recover_obs_pred_dof_pos_abs(
        pred_next_obs=pred_obs,
        term_order=("base_ang_vel", "dof_pos", "dof_vel", "projected_gravity"),
        term_scale={"base_ang_vel": 1.0, "dof_pos": 2.0, "dof_vel": 1.0, "projected_gravity": 1.0},
        term_dims={"base_ang_vel": 3, "dof_pos": 2, "dof_vel": 2, "projected_gravity": 3},
        default_dof_angles=default_angles,
        num_dofs=4,
        max_horizon_frames=10,
    )

    expected = np.array(
        [
            [10.0, 20.0, 31.0, 42.0],
            [10.0, 20.0, 33.0, 44.0],
        ],
        dtype=np.float32,
    )
    assert out is not None
    assert out.shape == (2, 4)
    assert np.allclose(out, expected)


def test_recover_obs_pred_dof_pos_abs_none_input() -> None:
    out = recover_obs_pred_dof_pos_abs(
        pred_next_obs=None,
        term_order=("base_ang_vel", "dof_pos", "dof_vel", "projected_gravity"),
        term_scale={"base_ang_vel": 1.0, "dof_pos": 1.0, "dof_vel": 1.0, "projected_gravity": 1.0},
        term_dims={"base_ang_vel": 3, "dof_pos": 2, "dof_vel": 2, "projected_gravity": 3},
        default_dof_angles=np.zeros((4,), dtype=np.float32),
        num_dofs=4,
        max_horizon_frames=10,
    )
    assert out is None


def test_recover_obs_pred_dof_pos_abs_raises_on_zero_scale() -> None:
    pred_obs = np.zeros((1, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="scale for 'dof_pos' is zero"):
        recover_obs_pred_dof_pos_abs(
            pred_next_obs=pred_obs,
            term_order=("base_ang_vel", "dof_pos", "dof_vel", "projected_gravity"),
            term_scale={"base_ang_vel": 1.0, "dof_pos": 0.0, "dof_vel": 1.0, "projected_gravity": 1.0},
            term_dims={"base_ang_vel": 3, "dof_pos": 2, "dof_vel": 2, "projected_gravity": 3},
            default_dof_angles=np.zeros((4,), dtype=np.float32),
            num_dofs=4,
            max_horizon_frames=10,
        )
