"""Locomotion randomization presets for the T1 robot."""

from __future__ import annotations

from dataclasses import replace

from holosoma.config_types.randomization import RandomizationManagerCfg, RandomizationTermCfg

t1_29dof_randomization = RandomizationManagerCfg(
    setup_terms={
        "push_randomizer_state": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:PushRandomizerState",
            params={
                "push_interval_s": [5, 10],
                "max_push_vel": [1.0, 1.0],
                "enabled": True,
            },
        ),
        "setup_action_delay_buffers": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:setup_action_delay_buffers",
            params={
                "ctrl_delay_step_range": [0, 1],
                "enabled": True,
            },
        ),
        "setup_torque_rfi": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:setup_torque_rfi",
            params={
                "enabled": False,
                "rfi_lim": 0.1,
            },
        ),
        "setup_dof_pos_bias": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:setup_dof_pos_bias",
            params={
                "dof_pos_bias_range": [-0.01, 0.01],
                "enabled": False,
            },
        ),
        "actuator_randomizer_state": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:ActuatorRandomizerState",
            params={
                "kp_range": [0.9, 1.1],
                "kd_range": [0.9, 1.1],
                "rfi_lim_range": [0.5, 1.5],
                "enable_pd_gain": True,
                "enable_rfi_lim": False,
            },
        ),
        "mass_randomizer": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:randomize_mass_startup",
            params={
                "enable_link_mass": True,
                "link_mass_range": [0.9, 1.2],
                "enable_base_mass": True,
                "added_mass_range": [-1.0, 3.0],
            },
        ),
        "randomize_friction_startup": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:randomize_friction_startup",
            params={
                "friction_range": [0.1, 1.0],
                "enabled": True,
            },
        ),
        "randomize_base_com_startup": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:randomize_base_com_startup",
            params={
                "base_com_range": {"x": [-0.01, 0.01], "y": [-0.01, 0.01], "z": [-0.01, 0.01]},
                "enabled": False,
            },
        ),
    },
    reset_terms={
        "push_randomizer_state": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:PushRandomizerState"
        ),
        "actuator_randomizer_state": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:ActuatorRandomizerState"
        ),
        "randomize_push_schedule": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:randomize_push_schedule",
        ),
        "randomize_action_delay": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:randomize_action_delay",
        ),
        "randomize_dof_state": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:randomize_dof_state",
            params={
                "joint_pos_scale_range": [0.5, 1.5],
                "joint_pos_bias_range": [0.0, 0.0],
                "joint_vel_range": [0.0, 0.0],
                "randomize_dof_pos_bias": False,
            },
        ),
        "configure_torque_rfi": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:configure_torque_rfi",
        ),
    },
    step_terms={
        "push_randomizer_state": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:PushRandomizerState"
        ),
        "apply_pushes": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:apply_pushes",
        ),
    },
)

# ---------------------------------------------------------------------------
# FADA: widen the domain-randomization ranges and add fixed-payload support to
# mass_randomizer.
# These ranges define the training distribution; changing any of them changes what the
# policy is trained against.
# ---------------------------------------------------------------------------
t1_29dof_randomization = replace(
    t1_29dof_randomization,
    setup_terms={
        **t1_29dof_randomization.setup_terms,
        "push_randomizer_state": replace(
            t1_29dof_randomization.setup_terms["push_randomizer_state"],
            params={**t1_29dof_randomization.setup_terms["push_randomizer_state"].params, "max_push_vel": [0.1, 1.5]},
        ),
        "setup_torque_rfi": replace(
            t1_29dof_randomization.setup_terms["setup_torque_rfi"],
            params={**t1_29dof_randomization.setup_terms["setup_torque_rfi"].params, "enabled": True},
        ),
        "setup_dof_pos_bias": replace(
            t1_29dof_randomization.setup_terms["setup_dof_pos_bias"],
            params={
                "dof_pos_bias_range": [-0.05, 0.05],
                "enabled": True,
            },
        ),
        "actuator_randomizer_state": replace(
            t1_29dof_randomization.setup_terms["actuator_randomizer_state"],
            params={
                **t1_29dof_randomization.setup_terms["actuator_randomizer_state"].params,
                "kp_range": [0.8, 1.2],
                "kd_range": [0.8, 1.2],
            },
        ),
        "mass_randomizer": replace(
            t1_29dof_randomization.setup_terms["mass_randomizer"],
            params={
                **t1_29dof_randomization.setup_terms["mass_randomizer"].params,
                "link_mass_range": [0.8, 1.3],
                "added_mass_range": [-3.0, 6.0],
                "fixed_payload_body_names": [],
                "fixed_payload_added_masses": [],
                "replace_fixed_payload_dr": True,
            },
        ),
        "randomize_friction_startup": replace(
            t1_29dof_randomization.setup_terms["randomize_friction_startup"],
            params={
                **t1_29dof_randomization.setup_terms["randomize_friction_startup"].params,
                "friction_range": [0.1, 2.0],
            },
        ),
        "randomize_base_com_startup": replace(
            t1_29dof_randomization.setup_terms["randomize_base_com_startup"],
            params={
                "base_com_range": {"x": [-0.15, 0.15], "y": [-0.15, 0.15], "z": [-0.15, 0.15]},
                "enabled": True,
            },
        ),
    },
)

# T1-23dof shares the identical randomization schedule with T1-29dof.
t1_23dof_randomization = t1_29dof_randomization

__all__ = ["t1_29dof_randomization"]
__all__ += ["t1_23dof_randomization"]
