"""Locomotion reward presets for the T1 robot."""

from dataclasses import replace

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg

t1_29dof_loco = RewardManagerCfg(
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
        "feet_phase": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:feet_phase",
            weight=5.0,
            params={"swing_height": 0.09, "tracking_sigma": 0.008},
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
        "pose": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:pose",
            weight=-0.5,
            params={
                "pose_weights": [
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
                ],
            },
            tags=["penalty_curriculum"],
        ),
    },
)

t1_29dof_loco_fast_sac = RewardManagerCfg(
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

t1_23dof_loco = RewardManagerCfg(
    only_positive_rewards=False,
    terms={
        "tracking_lin_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_lin_vel",
            weight=4.5,
            params={"tracking_sigma": 0.25},
        ),
        "tracking_ang_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_ang_vel",
            weight=2.5,
            params={"tracking_sigma": 0.25},
        ),
        # "base_height": RewardTermCfg(
        #     func="holosoma.managers.reward.terms.locomotion:base_height",
        #     weight=-10.0,
        #     params={"desired_base_height": 0.68},
        # ),
        "feet_phase": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:feet_phase",
            weight=1.0,
            params={"swing_height": 0.12, "tracking_sigma": 0.008},
        ),
        "penalty_ang_vel_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_ang_vel_xy",
            weight=-1.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_lin_vel_z": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_lin_vel_z",
            weight=-0.05,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_orientation": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_orientation",
            weight=-0.5,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_action_rate": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_action_rate",
            weight=-2.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_close_feet_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_close_feet_xy",
            weight=-10.0,
            params={"close_feet_threshold": 0.25},
            tags=["penalty_curriculum"],
        ),
        "penalty_feet_ori": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_feet_ori",
            weight=-5.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "feet_heading_alignment": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:feet_heading_alignment",
            weight=-0.5,
            params={},
        ),
        "penalty_feet_slippage": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_feet_slippage",
            weight=-0.4,
            params={},
            tags=["penalty_curriculum"],
        ),
        # "feet_stance_width": RewardTermCfg(
        #     func="holosoma.managers.reward.terms.locomotion:feet_stance_width",
        #     weight=10.0,
        #     params={"desired_feet_stance_width": 0.30, "feet_stance_width_std": 0.05},
        # ),
        "alive": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:alive",
            weight=0.25,
            params={},
        ),
         "feet_air_time_single_stance": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:feet_air_time_single_stance",
            weight=6.0,
            params={"max_air_time": 0.4, "min_vel_threshold": 0.1},
        ),
        "pose": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:pose",
            weight=-0.5,
            params={
                "pose_weights": [
                    50.0,  # AAHead_yaw
                    50.0,  # Head_pitch
                    50.0,  # Left_Shoulder_Pitch
                    50.0,  # Left_Shoulder_Roll
                    50.0,  # Left_Elbow_Pitch
                    50.0,  # Left_Elbow_Yaw
                    50.0,  # Right_Shoulder_Pitch
                    50.0,  # Right_Shoulder_Roll
                    50.0,  # Right_Elbow_Pitch
                    50.0,  # Right_Elbow_Yaw
                    0.01,  # Waist
                    1.0,   # Left_Hip_Pitch
                    5.0,   # Left_Hip_Roll
                    0.01,  # Left_Hip_Yaw
                    5.0,   # Left_Knee_Pitch
                    5.0,   # Left_Ankle_Pitch
                    0.01,  # Left_Ankle_Roll
                    1.0,   # Right_Hip_Pitch
                    5.0,   # Right_Hip_Roll
                    0.01,  # Right_Hip_Yaw
                    5.0,   # Right_Knee_Pitch
                    5.0,   # Right_Ankle_Pitch
                    0.01,  # Right_Ankle_Roll
                ],
            },
            tags=["penalty_curriculum"],
        ),
    },
)

# plane with waist aligned to upper-body joints (50.0). Same as t1_23dof_loco but
# fixes the waist yaw DOF being nearly unconstrained (0.01 vs 50.0 for arms/head).
_plane_waist50_pose_weights = list(t1_23dof_loco.terms["pose"].params["pose_weights"])
_plane_waist50_pose_weights[10] = 50.0  # Waist: 0.01 → 50.0

t1_23dof_loco_waist50 = replace(
    t1_23dof_loco,
    terms={
        **t1_23dof_loco.terms,
        "pose": replace(
            t1_23dof_loco.terms["pose"],
            params={**t1_23dof_loco.terms["pose"].params, "pose_weights": _plane_waist50_pose_weights},
        ),
    },
)

__all__ = ["t1_29dof_loco", "t1_29dof_loco_fast_sac"]
__all__ += ["t1_23dof_loco", "t1_23dof_loco_waist50"]
