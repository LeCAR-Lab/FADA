"""Locomotion reward presets for the G1 robot."""

from dataclasses import replace

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg

g1_29dof_pose_weights = [
    0.01,  # left_hip_yaw_joint
    1.0,  # left_hip_roll_joint
    5.0,  # left_hip_pitch_joint
    0.01,  # left_knee_joint
    5.0,  # left_ankle_pitch_joint
    5.0,  # left_ankle_roll_joint
    0.01,  # right_hip_yaw_joint
    1.0,  # right_hip_roll_joint
    5.0,  # right_hip_pitch_joint
    0.01,  # right_knee_joint
    5.0,  # right_ankle_pitch_joint
    5.0,  # right_ankle_roll_joint
    50.0,  # waist_yaw_joint
    50.0,  # waist_roll_joint
    50.0,  # waist_pitch_joint
    50.0,  # left_shoulder_pitch_joint
    50.0,  # left_shoulder_roll_joint
    50.0,  # left_shoulder_yaw_joint
    50.0,  # left_elbow_joint
    50.0,  # left_wrist_roll_joint
    50.0,  # left_wrist_pitch_joint
    50.0,  # left_wrist_yaw_joint
    50.0,  # right_shoulder_pitch_joint
    50.0,  # right_shoulder_roll_joint
    50.0,  # right_shoulder_yaw_joint
    50.0,  # right_elbow_joint
    50.0,  # right_wrist_roll_joint
    50.0,  # right_wrist_pitch_joint
    50.0,  # right_wrist_yaw_joint
]

g1_29dof_loco = RewardManagerCfg(
    only_positive_rewards=False,
    terms={
        "tracking_lin_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_lin_vel",
            weight=2.0,
            params={"tracking_sigma": 0.25},
        ),
        "tracking_ang_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_ang_vel",
            weight=1.5,
            params={"tracking_sigma": 0.25},
        ),
        "penalty_ang_vel_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_ang_vel_xy",
            weight=-1.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_orientation": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_orientation",
            weight=-10.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_action_rate": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_action_rate",
            weight=-2.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "feet_phase": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:feet_phase",
            weight=5.0,
            params={"swing_height": 0.09, "tracking_sigma": 0.008},
        ),
        "pose": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:pose",
            weight=-0.5,
            params={
                "pose_weights": [
                    0.01,
                    1.0,
                    5.0,
                    0.01,
                    5.0,
                    5.0,
                    0.01,
                    1.0,
                    5.0,
                    0.01,
                    5.0,
                    5.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                ],
            },
            tags=["penalty_curriculum"],
        ),
        "penalty_close_feet_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_close_feet_xy",
            weight=-10.0,
            params={"close_feet_threshold": 0.15},
            tags=["penalty_curriculum"],
        ),
        "penalty_feet_ori": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_feet_ori",
            weight=-5.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "alive": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:alive",
            weight=1.0,
            params={},
        ),
    },
)

# FADA: match BeyondMimic-style reward magnitudes (action-rate/feet_phase weight
# retune + shared pose_weights list).
g1_29dof_loco = replace(
    g1_29dof_loco,
    terms={
        **g1_29dof_loco.terms,
        "penalty_action_rate": replace(g1_29dof_loco.terms["penalty_action_rate"], weight=-0.2),
        "feet_phase": replace(g1_29dof_loco.terms["feet_phase"], weight=1.0),
        "pose": replace(g1_29dof_loco.terms["pose"], params={"pose_weights": g1_29dof_pose_weights}),
    },
)

g1_29dof_loco_fast_sac = RewardManagerCfg(
    only_positive_rewards=False,
    terms={
        "tracking_lin_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_lin_vel",
            weight=2.0,
            params={"tracking_sigma": 0.25},
        ),
        "tracking_ang_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_ang_vel",
            weight=1.5,
            params={"tracking_sigma": 0.25},
        ),
        "penalty_ang_vel_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_ang_vel_xy",
            weight=-1.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_orientation": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_orientation",
            weight=-10.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_action_rate": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_action_rate",
            weight=-2.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "feet_phase": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:feet_phase",
            weight=5.0,
            params={"swing_height": 0.09, "tracking_sigma": 0.008},
        ),
        "pose": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:pose",
            weight=-0.5,
            params={
                "pose_weights": [
                    0.01,
                    1.0,
                    5.0,
                    0.01,
                    5.0,
                    5.0,
                    0.01,
                    1.0,
                    5.0,
                    0.01,
                    5.0,
                    5.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                ],
            },
            tags=["penalty_curriculum"],
        ),
        "penalty_close_feet_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_close_feet_xy",
            weight=-10.0,
            params={"close_feet_threshold": 0.15},
            tags=["penalty_curriculum"],
        ),
        "penalty_feet_ori": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_feet_ori",
            weight=-5.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "alive": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:alive",
            weight=10.0,
            params={},
        ),
    },
)

# FADA: reuse the shared pose_weights list.
g1_29dof_loco_fast_sac = replace(
    g1_29dof_loco_fast_sac,
    terms={
        **g1_29dof_loco_fast_sac.terms,
        "pose": replace(g1_29dof_loco_fast_sac.terms["pose"], params={"pose_weights": g1_29dof_pose_weights}),
    },
)

g1_unitree_ankle_roll_patterns = [".*ankle_roll.*"]
g1_unitree_undesired_contact_patterns = ["(?!.*ankle.*)(?!.*foot_contact_point.*).*"]
g1_unitree_arm_joint_patterns = [".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*"]
g1_unitree_waist_joint_patterns = ["waist.*"]
g1_unitree_leg_joint_patterns = [".*_hip_roll_joint", ".*_hip_yaw_joint"]
g1_unitree_gym_hip_joint_patterns = [".*_hip_roll_joint", ".*_hip_pitch_joint"]

g1_29dof_loco_unitree = RewardManagerCfg(
    only_positive_rewards=False,
    terms={
        "track_lin_vel_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion_unitree:track_lin_vel_xy_yaw_frame_exp",
            weight=4.5,
            params={"command_name": "base_velocity", "std": 0.5},
        ),
        "track_ang_vel_z": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion_unitree:track_ang_vel_z_exp",
            weight=2.5,
            params={"command_name": "base_velocity", "std": 0.5},
        ),
        "alive": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:alive",
            weight=0.15,
            params={},
        ),
        "base_linear_velocity": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_lin_vel_z",
            weight=-2.0,
            params={},
        ),
        "base_angular_velocity": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_ang_vel_xy",
            weight=-0.05,
            params={},
        ),
        "joint_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion_boostergym:penalty_dof_vel",
            weight=-0.001,
            params={},
        ),
        "joint_acc": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion_boostergym:penalty_dof_acc",
            weight=-2.5e-7,
            params={},
        ),
        "action_rate": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_action_rate",
            weight=-0.2,
            params={},
        ),
        "dof_pos_limits": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:limits_dof_pos",
            weight=-5.0,
            params={},
        ),
        "energy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion_unitree:energy",
            weight=-2e-5,
            params={},
        ),
        "pose": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:pose",
            weight=-0.5,
            params={"pose_weights": g1_29dof_pose_weights},
        ),
        "flat_orientation_l2": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_orientation",
            weight=-5.0,
            params={},
        ),
        "base_height": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion_unitree:base_height_l2",
            weight=-10.0,
            params={"target_height": 0.78},
        ),
        "gait": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion_unitree:feet_gait_command_phase",
            weight=0.5,
            params={
                "threshold": 0.55,
                "command_name": "base_velocity",
                "body_name_patterns": g1_unitree_ankle_roll_patterns,
                "force_threshold": 1.0,
            },
        ),
        "feet_slide": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion_unitree:feet_slide",
            weight=-0.2,
            params={"body_name_patterns": g1_unitree_ankle_roll_patterns, "force_threshold": 1.0},
        ),
        "feet_clearance": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion_unitree:foot_clearance_reward",
            weight=1.0,
            params={"std": 0.05, "tanh_mult": 2.0, "target_height": 0.05},
        ),
        "penalty_feet_ori": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_feet_ori",
            weight=-2.0,
            params={},
        ),
        "penalty_close_feet_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_close_feet_xy",
            weight=-5.0,
            params={"close_feet_threshold": 0.2},
        ),
        "undesired_contacts": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion_unitree:undesired_contacts",
            weight=-1.0,
            params={"threshold": 1.0, "body_name_patterns": g1_unitree_undesired_contact_patterns},
        ),
    },
)

g1_29dof_loco_unitree_slope = RewardManagerCfg(
    only_positive_rewards=False,
    terms={
        "track_lin_vel_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion_unitree:track_lin_vel_xy_yaw_frame_exp",
            weight=4.5,
            params={"command_name": "base_velocity", "std": 0.5},
        ),
        "track_ang_vel_z": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion_unitree:track_ang_vel_z_exp",
            weight=2.5,
            params={"command_name": "base_velocity", "std": 0.5},
        ),
        "alive": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:alive",
            weight=0.15,
            params={},
        ),
        "base_linear_velocity": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_lin_vel_z",
            weight=-2.0,
            params={},
        ),
        "base_angular_velocity": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_ang_vel_xy",
            weight=-0.05,
            params={},
        ),
        "joint_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion_boostergym:penalty_dof_vel",
            weight=-0.001,
            params={},
        ),
        "joint_acc": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion_boostergym:penalty_dof_acc",
            weight=-2.5e-7,
            params={},
        ),
        "action_rate": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_action_rate",
            weight=-0.2,
            params={},
        ),
        "dof_pos_limits": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:limits_dof_pos",
            weight=-5.0,
            params={},
        ),
        "energy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion_unitree:energy",
            weight=-2e-5,
            params={},
        ),
        "pose": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:pose",
            weight=-0.5,
            params={"pose_weights": g1_29dof_pose_weights},
        ),
        "flat_orientation_l2": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_orientation",
            weight=-5.0,
            params={},
        ),
        "base_height": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:base_height",
            weight=-10.0,
            params={"desired_base_height": 0.78},
        ),
        "gait": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion_unitree:feet_gait_command_phase",
            weight=0.5,
            params={
                "threshold": 0.55,
                "command_name": "base_velocity",
                "body_name_patterns": g1_unitree_ankle_roll_patterns,
                "force_threshold": 1.0,
            },
        ),
        "feet_slide": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion_unitree:feet_slide",
            weight=-0.2,
            params={"body_name_patterns": g1_unitree_ankle_roll_patterns, "force_threshold": 1.0},
        ),
        "feet_clearance": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion_unitree:foot_clearance_reward",
            weight=1.0,
            params={"std": 0.05, "tanh_mult": 2.0, "target_height": 0.05},
        ),
        "penalty_feet_ori": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_feet_ori",
            weight=-2.0,
            params={},
        ),
        "penalty_close_feet_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_close_feet_xy",
            weight=-5.0,
            params={"close_feet_threshold": 0.2},
        ),
        "undesired_contacts": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion_unitree:undesired_contacts",
            weight=-1.0,
            params={"threshold": 1.0, "body_name_patterns": g1_unitree_undesired_contact_patterns},
        ),
    },
)

g1_29dof_loco_unitree_gym = RewardManagerCfg(
    only_positive_rewards=False,
    terms={
        "tracking_lin_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_lin_vel",
            weight=1.5,
            params={"tracking_sigma": 0.25},
        ),
        "tracking_ang_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_ang_vel",
            weight=1.0,
            params={"tracking_sigma": 0.25},
        ),
        "lin_vel_z": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_lin_vel_z",
            weight=-2.0,
            params={},
        ),
        "ang_vel_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_ang_vel_xy",
            weight=-0.05,
            params={},
        ),
        "orientation": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_orientation",
            weight=-1.0,
            params={},
        ),
        "base_height": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion_unitree:base_height_world",
            weight=-10.0,
            params={"target_height": 0.78},
        ),
        "dof_acc": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion_boostergym:penalty_dof_acc",
            weight=-2.5e-7,
            params={},
        ),
        "dof_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion_boostergym:penalty_dof_vel",
            weight=-1e-3,
            params={},
        ),
        "action_rate": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_action_rate",
            weight=-0.2,
            params={},
        ),
        "dof_pos_limits": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:limits_dof_pos",
            weight=-5.0,
            params={"soft_dof_pos_limit": 0.9},
        ),
        "alive": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:alive",
            weight=0.15,
            params={},
        ),
        "hip_pos": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion_unitree:joint_position_l2",
            weight=-0.0,
            params={"joint_name_patterns": g1_unitree_gym_hip_joint_patterns},
        ),
        "contact_no_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion_unitree:contact_no_vel_penalty",
            weight=-0.2,
            params={"body_name_patterns": g1_unitree_ankle_roll_patterns, "force_threshold": 1.0},
        ),
        "feet_swing_height": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion_unitree:feet_swing_height_penalty",
            weight=-20.0,
            params={
                "body_name_patterns": g1_unitree_ankle_roll_patterns,
                "target_height": 0.08,
                "force_threshold": 1.0,
            },
        ),
        "contact": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion_unitree:contact_phase_match",
            weight=0.18,
            params={
                "period": 0.8,
                "offset": [0.0, 0.5],
                "threshold": 0.55,
                "body_name_patterns": g1_unitree_ankle_roll_patterns,
                "force_threshold": 1.0,
            },
        ),
        "pose": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:pose",
            weight=-0.5,
            params={"pose_weights": g1_29dof_pose_weights},
        ),
    },
)

g1_29dof_loco_unitree_slope_ar05 = replace(
    g1_29dof_loco_unitree_slope,
    terms={
        **g1_29dof_loco_unitree_slope.terms,
        "action_rate": replace(g1_29dof_loco_unitree_slope.terms["action_rate"], weight=-0.5),
    },
)

# ----------------------------------------------------------------------------
# Slope-friendly reward: relax pose / feet_ori / lin_vel_z to allow slope
# adaptation. flat_orientation_l2 kept at -5.0.
g1_29dof_loco_unitree_slope_sf = replace(
    g1_29dof_loco_unitree_slope,
    terms={
        **g1_29dof_loco_unitree_slope.terms,
        "pose": replace(g1_29dof_loco_unitree_slope.terms["pose"], weight=-0.1),
        "penalty_feet_ori": replace(g1_29dof_loco_unitree_slope.terms["penalty_feet_ori"], weight=-0.5),
        "base_linear_velocity": replace(
            g1_29dof_loco_unitree_slope.terms["base_linear_velocity"], weight=-0.5
        ),
    },
)

g1_29dof_loco_unitree_slope_sf_ar05 = replace(
    g1_29dof_loco_unitree_slope_sf,
    terms={
        **g1_29dof_loco_unitree_slope_sf.terms,
        "action_rate": replace(g1_29dof_loco_unitree_slope_sf.terms["action_rate"], weight=-0.5),
    },
)

__all__ = ["g1_29dof_loco", "g1_29dof_loco_fast_sac"]
__all__ += [
    "g1_29dof_loco_unitree",
    "g1_29dof_loco_unitree_gym",
    "g1_29dof_loco_unitree_slope",
    "g1_29dof_loco_unitree_slope_ar05",
    "g1_29dof_loco_unitree_slope_sf",
    "g1_29dof_loco_unitree_slope_sf_ar05",
]
