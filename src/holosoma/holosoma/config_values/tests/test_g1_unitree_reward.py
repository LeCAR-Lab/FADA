"""Structural snapshot of the G1 Unitree loco reward preset.

WHAT THIS ESTABLISHES. That `reward.g1_29dof_loco_unitree` still has the term set, the
term ORDER, the per-term weights and the tracking-term `func`/`params` wiring recorded
below, and that `g1_29dof_loco_unitree_gym` is still a separate object rather than an
alias. It catches a one-sided edit: a term renamed, reordered, dropped, or reweighted in
`config_values/loco/g1/reward.py` without anyone noticing, on a preset that both
`g1_29dof_oracle` and `g1_29dof_deploy` train against.

WHAT THIS DOES NOT ESTABLISH. That any of these values is *correct*. The expectations are
a copy of the configuration's own numbers, not an independent derivation from anything --
no reference implementation, no paper table, no training outcome. There is no source here
that could disagree with the config, so nothing here can tell you a weight is wrong, only
that it moved. Editing the config and these literals together is green by construction.
"""

import sys

from holosoma.config_values.loco.g1 import experiment, reward

EXPECTED_UNITREE_TERM_ORDER = [
    "track_lin_vel_xy",
    "track_ang_vel_z",
    "alive",
    "base_linear_velocity",
    "base_angular_velocity",
    "joint_vel",
    "joint_acc",
    "action_rate",
    "dof_pos_limits",
    "energy",
    "pose",
    "flat_orientation_l2",
    "base_height",
    "gait",
    "feet_slide",
    "feet_clearance",
    "penalty_feet_ori",
    "penalty_close_feet_xy",
    "undesired_contacts",
]


def test_g1_unitree_reward_presets_share_single_source():
    """The gym preset must stay a distinct object, not an alias that edits both at once."""
    assert reward.g1_29dof_loco_unitree_gym is not reward.g1_29dof_loco_unitree


def test_g1_unitree_reward_config_matches_hybrid_pose_spec():
    """The preset's term set, term order, weights and tracking-term wiring are unchanged."""
    cfg = reward.g1_29dof_loco_unitree

    assert cfg.only_positive_rewards is False
    assert list(cfg.terms.keys()) == EXPECTED_UNITREE_TERM_ORDER

    expected_weights = {
        # The configuration is the source of truth for these two weights: both
        # `g1_29dof_oracle` and `g1_29dof_deploy` (config_values/loco/g1/experiment.py)
        # train against `g1_29dof_loco_unitree`. If the weights move, this expectation is
        # what moves -- not the config.
        "track_lin_vel_xy": 4.5,
        "track_ang_vel_z": 2.5,
        "alive": 0.15,
        "base_linear_velocity": -2.0,
        "base_angular_velocity": -0.05,
        "joint_vel": -0.001,
        "joint_acc": -2.5e-7,
        "action_rate": -0.2,
        "dof_pos_limits": -5.0,
        "energy": -2e-5,
        "pose": -0.5,
        "flat_orientation_l2": -5.0,
        "base_height": -10.0,
        "gait": 0.5,
        "feet_slide": -0.2,
        "feet_clearance": 1.0,
        "penalty_feet_ori": -2.0,
        "penalty_close_feet_xy": -5.0,
        "undesired_contacts": -1.0,
    }

    for term_name, expected_weight in expected_weights.items():
        assert cfg.terms[term_name].weight == expected_weight

    assert cfg.terms["track_lin_vel_xy"].func.endswith(":track_lin_vel_xy_yaw_frame_exp")
    assert cfg.terms["track_lin_vel_xy"].params == {"command_name": "base_velocity", "std": 0.5}
    assert cfg.terms["track_ang_vel_z"].func.endswith(":track_ang_vel_z_exp")
    assert cfg.terms["track_ang_vel_z"].params == {"command_name": "base_velocity", "std": 0.5}
    assert cfg.terms["pose"].func.endswith(":pose")
    assert cfg.terms["pose"].params == {"pose_weights": reward.g1_29dof_pose_weights}
    assert cfg.terms["base_height"].func.endswith(":base_height_l2")
    assert cfg.terms["base_height"].params == {"target_height": 0.78}
    assert cfg.terms["gait"].func.endswith(":feet_gait_command_phase")
    assert cfg.terms["gait"].params == {
        "threshold": 0.55,
        "command_name": "base_velocity",
        "body_name_patterns": reward.g1_unitree_ankle_roll_patterns,
        "force_threshold": 1.0,
    }
    assert cfg.terms["feet_slide"].params == {
        "body_name_patterns": reward.g1_unitree_ankle_roll_patterns,
        "force_threshold": 1.0,
    }
    assert cfg.terms["feet_clearance"].params == {"std": 0.05, "tanh_mult": 2.0, "target_height": 0.05}
    assert cfg.terms["penalty_feet_ori"].func.endswith(":penalty_feet_ori")
    assert cfg.terms["penalty_feet_ori"].params == {}
    assert cfg.terms["penalty_close_feet_xy"].func.endswith(":penalty_close_feet_xy")
    assert cfg.terms["penalty_close_feet_xy"].params == {"close_feet_threshold": 0.2}
    assert cfg.terms["undesired_contacts"].params == {
        "threshold": 1.0,
        "body_name_patterns": reward.g1_unitree_undesired_contact_patterns,
    }


def test_g1_unitree_gym_reward_config_is_preserved():
    cfg = reward.g1_29dof_loco_unitree_gym

    assert list(cfg.terms.keys()) == [
        "tracking_lin_vel",
        "tracking_ang_vel",
        "lin_vel_z",
        "ang_vel_xy",
        "orientation",
        "base_height",
        "dof_acc",
        "dof_vel",
        "action_rate",
        "dof_pos_limits",
        "alive",
        "hip_pos",
        "contact_no_vel",
        "feet_swing_height",
        "contact",
        "pose",
    ]
    assert cfg.terms["tracking_lin_vel"].weight == 1.5
    assert cfg.terms["tracking_ang_vel"].weight == 1.0
    assert cfg.terms["base_height"].func.endswith(":base_height_world")
    assert cfg.terms["dof_pos_limits"].params == {"soft_dof_pos_limit": 0.9}
    assert cfg.terms["hip_pos"].params == {"joint_name_patterns": reward.g1_unitree_gym_hip_joint_patterns}


def test_g1_reward_driven_experiments_use_unitree_entrypoint():
    assert experiment.g1_29dof_deploy.reward is reward.g1_29dof_loco_unitree
    assert experiment.g1_29dof_oracle.reward is reward.g1_29dof_loco_unitree
    assert experiment.g1_29dof_deploy.nightly.metrics == {
        "Episode/rew_tracking_ang_vel": [0.7, "inf"],
        "Episode/rew_tracking_lin_vel": [0.55, "inf"],
    }
    assert experiment.g1_29dof_oracle.nightly.metrics == {
        "Episode/rew_tracking_ang_vel": [0.7, "inf"],
        "Episode/rew_tracking_lin_vel": [0.55, "inf"],
    }


def test_this_file_says_what_it_cannot_establish():
    """The module docstring states the limit of what this file establishes, and that
    statement is pinned here.
    """
    docstring = sys.modules[__name__].__doc__ or ""
    assert "WHAT THIS DOES NOT ESTABLISH" in docstring, (
        "the module docstring no longer states the limit -- a snapshot presented without it "
        "reads as validation of the reward values, which this file cannot provide"
    )
    assert "copy of the configuration's own numbers" in docstring, docstring
