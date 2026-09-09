"""Locomotion termination presets for the T1 robot."""

from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg

t1_29dof_termination = TerminationManagerCfg(
    terms={
        "contact": TerminationTermCfg(
            func="holosoma.managers.termination.terms.locomotion:contact_forces_exceeded",
            params={
                "force_threshold": 1.0,
                "contact_indices_attr": "termination_contact_indices",
            },
        ),
        "timeout": TerminationTermCfg(
            func="holosoma.managers.termination.terms.common:timeout_exceeded",
            is_timeout=True,
        ),
    }
)

t1_23dof_termination = TerminationManagerCfg(
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
                "min_height": 0.28,
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

__all__ = ["t1_29dof_termination"]
__all__ += ["t1_23dof_termination"]
