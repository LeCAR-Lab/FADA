"""Locomotion termination presets for the G1 robot."""

from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg

g1_29dof_termination = TerminationManagerCfg(
    terms={
        "contact": TerminationTermCfg(
            func="holosoma.managers.termination.terms.locomotion:contact_forces_exceeded",
            params={
                "force_threshold": 1.0,
                "contact_indices_attr": "termination_contact_indices",
            },
        ),
        "fail_safe_tilt": TerminationTermCfg(
            func="holosoma.managers.termination.terms.locomotion:gravity_tilt_exceeded",
            params={
                "threshold_x": 0.75,
                "threshold_y": 0.75,
                "enabled": True,
            },
        ),
        "fail_safe_height": TerminationTermCfg(
            func="holosoma.managers.termination.terms.locomotion:base_height_below_threshold",
            params={
                "min_height": 0.32,
                "enabled": True,
                "use_terrain_relative": True,
            },
        ),
        "timeout": TerminationTermCfg(
            func="holosoma.managers.termination.terms.common:timeout_exceeded",
            is_timeout=True,
        ),
    }
)

__all__ = ["g1_29dof_termination"]
