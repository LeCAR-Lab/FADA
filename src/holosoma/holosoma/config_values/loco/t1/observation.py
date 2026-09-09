"""Locomotion observation presets for the T1 robot."""

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg

t1_29dof_loco_single_wolinvel = ObservationManagerCfg(
    groups={
        "actor_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=True,
            history_length=1,
            terms={
                "base_ang_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:base_ang_vel",
                    scale=1.0,
                    noise=0.0,
                ),
                "projected_gravity": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:projected_gravity",
                    scale=1.0,
                    noise=0.0,
                ),
                "command_lin_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:command_lin_vel",
                    scale=1.0,
                    noise=0.0,
                ),
                "command_ang_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:command_ang_vel",
                    scale=1.0,
                    noise=0.0,
                ),
                "dof_pos": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:dof_pos",
                    scale=1.0,
                    noise=0.01,
                ),
                "dof_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:dof_vel",
                    scale=0.1,
                    noise=0.1,
                ),
                "actions": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:actions",
                    scale=1.0,
                    noise=0.0,
                ),
                "sin_phase": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:sin_phase",
                    scale=1.0,
                    noise=0.0,
                ),
                "cos_phase": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:cos_phase",
                    scale=1.0,
                    noise=0.0,
                ),
            },
        ),
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms={
                "base_lin_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:base_lin_vel",
                    scale=1.0,
                    noise=0.0,
                ),
                "base_ang_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:base_ang_vel",
                    scale=1.0,
                    noise=0.0,
                ),
                "projected_gravity": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:projected_gravity",
                    scale=1.0,
                    noise=0.0,
                ),
                "command_lin_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:command_lin_vel",
                    scale=1.0,
                    noise=0.0,
                ),
                "command_ang_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:command_ang_vel",
                    scale=1.0,
                    noise=0.0,
                ),
                "dof_pos": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:dof_pos",
                    scale=1.0,
                    noise=0.0,
                ),
                "dof_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:dof_vel",
                    scale=0.1,
                    noise=0.0,
                ),
                "actions": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:actions",
                    scale=1.0,
                    noise=0.0,
                ),
                "sin_phase": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:sin_phase",
                    scale=1.0,
                    noise=0.0,
                ),
                "cos_phase": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:cos_phase",
                    scale=1.0,
                    noise=0.0,
                ),
            },
        ),
    }
)

t1_23dof_loco_single_wolinvel = ObservationManagerCfg(
    groups={
        "actor_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=True,
            history_length=1,
            terms={
                "base_ang_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:base_ang_vel",
                    scale=1.0,
                    noise=0.0,
                ),
                "projected_gravity": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:projected_gravity",
                    scale=1.0,
                    noise=0.0,
                ),
                "command_lin_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:command_lin_vel",
                    scale=1.0,
                    noise=0.0,
                ),
                "command_ang_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:command_ang_vel",
                    scale=1.0,
                    noise=0.0,
                ),
                "dof_pos": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:dof_pos",
                    scale=1.0,
                    noise=0.01,
                ),
                "dof_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:dof_vel",
                    scale=0.1,
                    noise=0.1,
                ),
                "actions": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:actions",
                    scale=1.0,
                    noise=0.0,
                ),
                "sin_phase": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:sin_phase",
                    scale=1.0,
                    noise=0.0,
                ),
                "cos_phase": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:cos_phase",
                    scale=1.0,
                    noise=0.0,
                ),
            },
        ),
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms={
                "base_lin_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:base_lin_vel",
                    scale=1.0,
                    noise=0.0,
                ),
                "base_ang_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:base_ang_vel",
                    scale=1.0,
                    noise=0.0,
                ),
                "projected_gravity": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:projected_gravity",
                    scale=1.0,
                    noise=0.0,
                ),
                "command_lin_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:command_lin_vel",
                    scale=1.0,
                    noise=0.0,
                ),
                "command_ang_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:command_ang_vel",
                    scale=1.0,
                    noise=0.0,
                ),
                "dof_pos": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:dof_pos",
                    scale=1.0,
                    noise=0.0,
                ),
                "dof_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:dof_vel",
                    scale=0.1,
                    noise=0.0,
                ),
                "actions": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:actions",
                    scale=1.0,
                    noise=0.0,
                ),
                "sin_phase": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:sin_phase",
                    scale=1.0,
                    noise=0.0,
                ),
                "cos_phase": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:cos_phase",
                    scale=1.0,
                    noise=0.0,
                ),
            },
        ),
    }
)

# t1_23dof_loco_single_wolinvel_deploy = ObservationManagerCfg(
#     groups={
#         "actor_obs": ObsGroupCfg(
#             concatenate=True,
#             enable_noise=True,
#             history_length=1,
#             terms={
#                 "base_ang_vel": ObsTermCfg(
#                     func="holosoma.managers.observation.terms.locomotion:base_ang_vel",
#                     scale=1.0,
#                     noise=0.0,
#                 ),
#                 "projected_gravity": ObsTermCfg(
#                     func="holosoma.managers.observation.terms.locomotion:projected_gravity",
#                     scale=1.0,
#                     noise=0.0,
#                 ),
#                 "command_lin_vel": ObsTermCfg(
#                     func="holosoma.managers.observation.terms.locomotion:command_lin_vel",
#                     scale=1.0,
#                     noise=0.0,
#                 ),
#                 "command_ang_vel": ObsTermCfg(
#                     func="holosoma.managers.observation.terms.locomotion:command_ang_vel",
#                     scale=1.0,
#                     noise=0.0,
#                 ),
#                 "dof_pos": ObsTermCfg(
#                     func="holosoma.managers.observation.terms.locomotion:dof_pos",
#                     scale=1.0,
#                     noise=0.01,
#                 ),
#                 "dof_vel": ObsTermCfg(
#                     func="holosoma.managers.observation.terms.locomotion:dof_vel",
#                     scale=0.1,
#                     noise=0.1,
#                 ),
#                 "actions": ObsTermCfg(
#                     func="holosoma.managers.observation.terms.locomotion:actions",
#                     scale=1.0,
#                     noise=0.0,
#                 ),
#                 "sin_phase": ObsTermCfg(
#                     func="holosoma.managers.observation.terms.locomotion:sin_phase",
#                     scale=1.0,
#                     noise=0.0,
#                 ),
#                 "cos_phase": ObsTermCfg(
#                     func="holosoma.managers.observation.terms.locomotion:cos_phase",
#                     scale=1.0,
#                     noise=0.0,
#                 ),
#             },
#         ),
#         "critic_obs": ObsGroupCfg(
#             concatenate=True,
#             enable_noise=False,
#             history_length=1,
#             terms={
#                 "base_lin_vel": ObsTermCfg(
#                     func="holosoma.managers.observation.terms.locomotion:base_lin_vel",
#                     scale=1.0,
#                     noise=0.0,
#                 ),
#                 "base_ang_vel": ObsTermCfg(
#                     func="holosoma.managers.observation.terms.locomotion:base_ang_vel",
#                     scale=1.0,
#                     noise=0.0,
#                 ),
#                 "projected_gravity": ObsTermCfg(
#                     func="holosoma.managers.observation.terms.locomotion:projected_gravity",
#                     scale=1.0,
#                     noise=0.0,
#                 ),
#                 "command_lin_vel": ObsTermCfg(
#                     func="holosoma.managers.observation.terms.locomotion:command_lin_vel",
#                     scale=1.0,
#                     noise=0.0,
#                 ),
#                 "command_ang_vel": ObsTermCfg(
#                     func="holosoma.managers.observation.terms.locomotion:command_ang_vel",
#                     scale=1.0,
#                     noise=0.0,
#                 ),
#                 "dof_pos": ObsTermCfg(
#                     func="holosoma.managers.observation.terms.locomotion:dof_pos",
#                     scale=1.0,
#                     noise=0.0,
#                 ),
#                 "dof_vel": ObsTermCfg(
#                     func="holosoma.managers.observation.terms.locomotion:dof_vel",
#                     scale=0.1,
#                     noise=0.0,
#                 ),
#                 "actions": ObsTermCfg(
#                     func="holosoma.managers.observation.terms.locomotion:actions",
#                     scale=1.0,
#                     noise=0.0,
#                 ),
#                 "sin_phase": ObsTermCfg(
#                     func="holosoma.managers.observation.terms.locomotion:sin_phase",
#                     scale=1.0,
#                     noise=0.0,
#                 ),
#                 "cos_phase": ObsTermCfg(
#                     func="holosoma.managers.observation.terms.locomotion:cos_phase",
#                     scale=1.0,
#                     noise=0.0,
#                 ),
#             },
#         ),

#         "obs_history": ObsGroupCfg(
#             concatenate=True,
#             enable_noise=True,
#             history_length=5,
#             past_history=True,
#             terms={
#                 "base_ang_vel": ObsTermCfg(
#                     func="holosoma.managers.observation.terms.locomotion:base_ang_vel",
#                     scale=1.0,
#                     noise=0.0,
#                 ),
#                 "projected_gravity": ObsTermCfg(
#                     func="holosoma.managers.observation.terms.locomotion:projected_gravity",
#                     scale=1.0,
#                     noise=0.0,
#                 ),
#                 "dof_pos": ObsTermCfg(
#                     func="holosoma.managers.observation.terms.locomotion:dof_pos",
#                     scale=1.0,
#                     noise=0.01,
#                 ),
#                 "dof_vel": ObsTermCfg(
#                     func="holosoma.managers.observation.terms.locomotion:dof_vel",
#                     scale=0.1,
#                     noise=0.1,
#                 ),
#             },
#         ),
#         "actions_history": ObsGroupCfg(
#             concatenate=True,
#             enable_noise=True,
#             history_length=5,
#             past_history=True,
#             terms={
#                 "actions": ObsTermCfg(
#                     func="holosoma.managers.observation.terms.locomotion:actions",
#                     scale=1.0,
#                     noise=0.0,
#                 ),
#             },
#         ),

#         "dynamics_obs": ObsGroupCfg(
#             concatenate=True,
#             enable_noise=True,
#             history_length=1,
#             past_history=False,
#             terms={
#                 "base_ang_vel": ObsTermCfg(
#                     func="holosoma.managers.observation.terms.locomotion:base_ang_vel",
#                     scale=1.0,
#                     noise=0.0,
#                 ),
#                 "projected_gravity": ObsTermCfg(
#                     func="holosoma.managers.observation.terms.locomotion:projected_gravity",
#                     scale=1.0,
#                     noise=0.0,
#                 ),
#                 "dof_pos": ObsTermCfg(
#                     func="holosoma.managers.observation.terms.locomotion:dof_pos",
#                     scale=1.0,
#                     noise=0.01,
#                 ),
#                 "dof_vel": ObsTermCfg(
#                     func="holosoma.managers.observation.terms.locomotion:dof_vel",
#                     scale=0.1,
#                     noise=0.1,
#                 ),
#             },
#         ),


        
#     }
# )

t1_23dof_loco_single_wolinvel_deploy = ObservationManagerCfg(
    groups={
        "actor_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms={
                "base_ang_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:base_ang_vel",
                    scale=0.25,
                    noise=0.3,
                ),
                "projected_gravity": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:projected_gravity",
                    scale=1.0,
                    noise=0.2,
                ),
                "command_lin_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:command_lin_vel",
                    scale=1.0,
                    noise=0.0,
                ),
                "command_ang_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:command_ang_vel",
                    scale=1.0,
                    noise=0.0,
                ),
                "dof_pos": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:dof_pos",
                    scale=1.0,
                    noise=0.01,
                ),
                "dof_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:dof_vel",
                    scale=0.05,
                    noise=1.0,
                ),
                "actions": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:actions",
                    scale=1.0,
                    noise=0.0,
                ),
                "sin_phase": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:sin_phase",
                    scale=1.0,
                    noise=0.0,
                ),
                "cos_phase": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:cos_phase",
                    scale=1.0,
                    noise=0.0,
                ),
            },
        ),
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms={
                "base_lin_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:base_lin_vel",
                    scale=2.0,
                    noise=0.0,
                ),
                "base_ang_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:base_ang_vel",
                    scale=0.25,
                    noise=0.3,
                ),
                "projected_gravity": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:projected_gravity",
                    scale=1.0,
                    noise=0.2,
                ),
                "command_lin_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:command_lin_vel",
                    scale=1.0,
                    noise=0.0,
                ),
                "command_ang_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:command_ang_vel",
                    scale=1.0,
                    noise=0.0,
                ),
                "dof_pos": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:dof_pos",
                    scale=1.0,
                    noise=0.01,
                ),
                "dof_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:dof_vel",
                    scale=0.05,
                    noise=1.0,
                ),
                "actions": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:actions",
                    scale=1.0,
                    noise=0.0,
                ),
                "sin_phase": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:sin_phase",
                    scale=1.0,
                    noise=0.0,
                ),
                "cos_phase": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:cos_phase",
                    scale=1.0,
                    noise=0.0,
                ),
            },
        ),
        "obs_history": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=5,
            past_history=True,
            terms={
                "base_ang_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:base_ang_vel",
                    scale=0.25,
                    noise=0.3,
                ),
                "projected_gravity": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:projected_gravity",
                    scale=1.0,
                    noise=0.2,
                ),
                "dof_pos": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:dof_pos",
                    scale=1.0,
                    noise=0.01,
                ),
                "dof_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:dof_vel",
                    scale=0.05,
                    noise=1.0,
                ),
            },
        ),
        "actions_history": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=5,
            past_history=True,
            terms={
                "actions": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:actions",
                    scale=1.0,
                    noise=0.0,
                ),
            },
        ),
        "dynamics_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            past_history=False,
            terms={
                "base_ang_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:base_ang_vel",
                    scale=0.25,
                    noise=0.3,
                ),
                "projected_gravity": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:projected_gravity",
                    scale=1.0,
                    noise=0.2,
                ),
                "dof_pos": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:dof_pos",
                    scale=1.0,
                    noise=0.01,
                ),
                "dof_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:dof_vel",
                    scale=0.05,
                    noise=1.0,
                ),
            },
        ),
    }
)


t1_23dof_loco_oracle = ObservationManagerCfg(
    groups={
        "actor_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms={
                "base_ang_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:base_ang_vel",
                    scale=0.25,
                    noise=0.0,
                ),
                "base_lin_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:base_lin_vel",
                    scale=2.0,
                    noise=0.0,
                ),
                "dof_pos": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:dof_pos",
                    scale=1.0,
                    noise=0.0,
                ),
                "dof_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:dof_vel",
                    scale=0.05,
                    noise=0.0,
                ),
                "actions": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:actions",
                    scale=1.0,
                    noise=0.0,
                ),
                "projected_gravity": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:projected_gravity",
                    scale=1.0,
                    noise=0.0,
                ),
                "oracle_actuator_compact_state": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:oracle_actuator_compact_state",
                    scale=1.0,
                    noise=0.0,
                ),
                "oracle_contact_compact_state": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:oracle_contact_compact_state",
                    scale=1.0,
                    noise=0.0,
                ),
                "oracle_terrain_compact_state": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:oracle_terrain_compact_state",
                    scale=1.0,
                    noise=0.0,
                ),
                "oracle_command_gait_compact_state": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:oracle_command_gait_compact_state",
                    scale=1.0,
                    noise=0.0,
                ),
                "oracle_randomization_compact_state": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:oracle_randomization_compact_state",
                    params={"normalize": False},
                    scale=1.0,
                    noise=0.0,
                ),
            },
        ),
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms={
                "base_ang_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:base_ang_vel",
                    scale=0.25,
                    noise=0.0,
                ),
                "base_lin_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:base_lin_vel",
                    scale=2.0,
                    noise=0.0,
                ),
                "dof_pos": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:dof_pos",
                    scale=1.0,
                    noise=0.0,
                ),
                "dof_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:dof_vel",
                    scale=0.05,
                    noise=0.0,
                ),
                "actions": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:actions",
                    scale=1.0,
                    noise=0.0,
                ),
                "projected_gravity": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:projected_gravity",
                    scale=1.0,
                    noise=0.0,
                ),
                "oracle_actuator_compact_state": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:oracle_actuator_compact_state",
                    scale=1.0,
                    noise=0.0,
                ),
                "oracle_contact_compact_state": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:oracle_contact_compact_state",
                    scale=1.0,
                    noise=0.0,
                ),
                "oracle_terrain_compact_state": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:oracle_terrain_compact_state",
                    scale=1.0,
                    noise=0.0,
                ),
                "oracle_command_gait_compact_state": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:oracle_command_gait_compact_state",
                    scale=1.0,
                    noise=0.0,
                ),
                "oracle_randomization_compact_state": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:oracle_randomization_compact_state",
                    params={"normalize": False},
                    scale=1.0,
                    noise=0.0,
                ),
            },
        ),
    }
)




__all__ = ["t1_29dof_loco_single_wolinvel"]
__all__ += ["t1_23dof_loco_single_wolinvel", "t1_23dof_loco_single_wolinvel_deploy", "t1_23dof_loco_oracle"]
