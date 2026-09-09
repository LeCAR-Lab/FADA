"""Default observation configurations for holosoma_inference.

This module provides pre-configured observation spaces for different
robot types and tasks, converted from the original YAML configurations.
"""

from __future__ import annotations

from importlib.metadata import entry_points

from holosoma_inference.config.config_types.observation import ObservationConfig

# =============================================================================
# Locomotion Observation Configurations
# =============================================================================

loco_g1_29dof = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "base_ang_vel",
            "projected_gravity",
            "command_lin_vel",
            "command_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
            "sin_phase",
            "cos_phase",
        ]
    },
    obs_dims={
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "command_lin_vel": 2,
        "command_ang_vel": 1,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
        "sin_phase": 2,
        "cos_phase": 2,
    },
    obs_scales={
        "base_lin_vel": 2.0,
        "base_ang_vel": 0.25,
        "projected_gravity": 1.0,
        "command_lin_vel": 1.0,
        "command_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 0.05,
        "actions": 1.0,
        "sin_phase": 1.0,
        "cos_phase": 1.0,
    },
    history_length_dict={
        "actor_obs": 1,
    },
)

loco_t1_23dof = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "base_ang_vel",
            "projected_gravity",
            "command_lin_vel",
            "command_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
            "sin_phase",
            "cos_phase",
        ]
    },
    obs_dims={
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "command_lin_vel": 2,
        "command_ang_vel": 1,
        "dof_pos": 23,
        "dof_vel": 23,
        "actions": 23,
        "sin_phase": 2,
        "cos_phase": 2,
    },
    obs_scales={
        "base_lin_vel": 1.0,  # T1 uses 1.0 (vs G1's 2.0)
        "base_ang_vel": 1.0,  # T1 uses 1.0 (vs G1's 0.25)
        "projected_gravity": 1.0,
        "command_lin_vel": 1.0,
        "command_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 0.1,  # T1 uses 0.1 (vs G1's 0.05)
        "actions": 1.0,
        "sin_phase": 1.0,
        "cos_phase": 1.0,
    },
    history_length_dict={
        "actor_obs": 1,
    },
)

loco_t1_29dof = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "base_ang_vel",
            "projected_gravity",
            "command_lin_vel",
            "command_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
            "sin_phase",
            "cos_phase",
        ]
    },
    obs_dims={
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "command_lin_vel": 2,
        "command_ang_vel": 1,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
        "sin_phase": 2,
        "cos_phase": 2,
    },
    obs_scales={
        "base_lin_vel": 1.0,  # T1 uses 1.0 (vs G1's 2.0)
        "base_ang_vel": 1.0,  # T1 uses 1.0 (vs G1's 0.25)
        "projected_gravity": 1.0,
        "command_lin_vel": 1.0,
        "command_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 0.1,  # T1 uses 0.1 (vs G1's 0.05)
        "actions": 1.0,
        "sin_phase": 1.0,
        "cos_phase": 1.0,
    },
    history_length_dict={
        "actor_obs": 1,
    },
)

loco_t1_23dof_deploy = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "base_ang_vel",
            "projected_gravity",
            "command_lin_vel",
            "command_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
            "sin_phase",
            "cos_phase",
        ],
        "obs_history": [
            "base_ang_vel",
            "projected_gravity",
            "dof_pos",
            "dof_vel",
        ],
        "actions_history": [
            "actions",
        ],
        "dynamics_obs": [
            "base_ang_vel",
            "projected_gravity",
            "dof_pos",
            "dof_vel",
        ],
    },
    obs_dims={
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "command_lin_vel": 2,
        "command_ang_vel": 1,
        "dof_pos": 23,
        "dof_vel": 23,
        "actions": 23,
        "sin_phase": 2,
        "cos_phase": 2,
    },
    obs_scales={
        "base_lin_vel": 1.0,
        "base_ang_vel": 0.25,
        "projected_gravity": 1.0,
        "command_lin_vel": 1.0,
        "command_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 0.05,
        "actions": 1.0,
        "sin_phase": 1.0,
        "cos_phase": 1.0,
    },
    history_length_dict={
        "actor_obs": 1,
        "obs_history": 5,
        "actions_history": 5,
        "dynamics_obs": 1,
    },
    past_history_groups={
        "obs_history": True,
        "actions_history": True,
    },
)

# Transformer dedicated observation profile: compact state terms (identity pre-scales).
loco_t1_23dof_transformer = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "base_ang_vel",
            "projected_gravity",
            "dof_pos",
            "dof_vel",
        ],
    },
    obs_dims={
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "command_lin_vel": 2,
        "command_ang_vel": 1,
        "dof_pos": 23,
        "dof_vel": 23,
        "actions": 23,
        "sin_phase": 2,
        "cos_phase": 2,
    },
    obs_scales={
        "base_lin_vel": 1.0,
        # Keep transformer pre-scales as identity in inference.
        # Transformer compact-observation scaling must come from ONNX metadata
        # (transformer_preprocess.term_scale) to avoid double scaling.
        "base_ang_vel": 1.0,
        "projected_gravity": 1.0,
        "command_lin_vel": 1.0,
        "command_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 1.0,
        "actions": 1.0,
        "sin_phase": 1.0,
        "cos_phase": 1.0,
    },
    history_length_dict={
        "actor_obs": 1,
    },
)

# =============================================================================
# G1 29-DOF — Deploy / Transformer
# Same observation *structure* as T1-23dof variants; dof_pos/dof_vel/actions use 29.
# Match obs_scales to your training export (these mirror the T1 deploy-family presets).
# loco_g1_29dof_deploy is the observation profile consumed by the deploy preset
# (policy_mode="deploy").
# =============================================================================

loco_g1_29dof_deploy = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "base_ang_vel",
            "projected_gravity",
            "command_lin_vel",
            "command_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
            "sin_phase",
            "cos_phase",
        ],
        "obs_history": [
            "base_ang_vel",
            "projected_gravity",
            "dof_pos",
            "dof_vel",
        ],
        "actions_history": [
            "actions",
        ],
        "dynamics_obs": [
            "base_ang_vel",
            "projected_gravity",
            "dof_pos",
            "dof_vel",
        ],
    },
    obs_dims={
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "command_lin_vel": 2,
        "command_ang_vel": 1,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
        "sin_phase": 2,
        "cos_phase": 2,
    },
    obs_scales={
        "base_lin_vel": 1.0,
        "base_ang_vel": 0.25,
        "projected_gravity": 1.0,
        "command_lin_vel": 1.0,
        "command_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 0.05,
        "actions": 1.0,
        "sin_phase": 1.0,
        "cos_phase": 1.0,
    },
    history_length_dict={
        "actor_obs": 1,
        "obs_history": 5,
        "actions_history": 5,
        "dynamics_obs": 1,
    },
    past_history_groups={
        "obs_history": True,
        "actions_history": True,
    },
)

loco_g1_29dof_transformer = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "base_ang_vel",
            "projected_gravity",
            "dof_pos",
            "dof_vel",
        ],
    },
    obs_dims={
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "command_lin_vel": 2,
        "command_ang_vel": 1,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
        "sin_phase": 2,
        "cos_phase": 2,
    },
    obs_scales={
        "base_lin_vel": 1.0,
        "base_ang_vel": 1.0,
        "projected_gravity": 1.0,
        "command_lin_vel": 1.0,
        "command_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 1.0,
        "actions": 1.0,
        "sin_phase": 1.0,
        "cos_phase": 1.0,
    },
    history_length_dict={
        "actor_obs": 1,
    },
)


# =============================================================================
# WBT (Whole Body Tracking) Observation Configurations
# =============================================================================

wbt = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "motion_command",
            "motion_ref_ori_b",
            "base_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
        ]
    },
    obs_dims={
        "motion_command": 58,
        "motion_ref_pos_b": 3,
        "motion_ref_ori_b": 6,
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
    },
    obs_scales={
        "actions": 1.0,
        "motion_command": 1.0,
        "motion_ref_pos_b": 1.0,
        "motion_ref_ori_b": 1.0,
        "base_lin_vel": 1.0,
        "base_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 1.0,
        "robot_body_pos_b": 1.0,
        "robot_body_ori_b": 1.0,
    },
    history_length_dict={
        "actor_obs": 1,
    },
)

# =============================================================================
# Default Configurations Dictionary
# =============================================================================

DEFAULTS = {
    "loco-g1-29dof": loco_g1_29dof,
    "loco-t1-23dof": loco_t1_23dof,
    "loco-t1-29dof": loco_t1_29dof,
    "loco-t1-23dof-deploy": loco_t1_23dof_deploy,
    "loco-t1-23dof-transformer": loco_t1_23dof_transformer,
    "loco-g1-29dof-deploy": loco_g1_29dof_deploy,
    "loco-g1-29dof-transformer": loco_g1_29dof_transformer,
    "wbt": wbt,
}
"""Dictionary of all available observation configurations.

Keys use hyphen-case naming convention for CLI compatibility.
"""

# Auto-discover observation configs from installed extensions
for ep in entry_points(group="holosoma.config.observation"):
    DEFAULTS[ep.name] = ep.load()
