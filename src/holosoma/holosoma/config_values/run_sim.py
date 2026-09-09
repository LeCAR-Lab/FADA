"""
Configuration values for run_sim.py - sim2sim-optimized simulator defaults.

These configurations extend the base simulator configs with sim2sim-specific settings:
- Bridge enabled by default
- Virtual gantry enabled
- High FPS for real-time robot control
- DOF force sensors enabled (for IsaacGym)
"""

import dataclasses

import holosoma.config_values.simulator
from holosoma.config_types.simulator import BridgeConfig, VirtualGantryCfg

# IsaacGym with sim2sim optimizations
isaacgym = dataclasses.replace(
    holosoma.config_values.simulator.isaacgym,
    config=dataclasses.replace(
        holosoma.config_values.simulator.isaacgym.config,
        bridge=BridgeConfig(enabled=True),
        virtual_gantry=VirtualGantryCfg(enabled=True),
        sim=dataclasses.replace(
            holosoma.config_values.simulator.isaacgym.config.sim,
            fps=1000,  # Gym on CPU can reach ~900 on 4090 (for sim2sim i.e, num_envs=1)
            physx=dataclasses.replace(
                holosoma.config_values.simulator.isaacgym.config.sim.physx,
                enable_dof_force_sensors=True,  # required for force controls
            ),
        ),
    ),
)

# IsaacSim with sim2sim optimizations
isaacsim = dataclasses.replace(
    holosoma.config_values.simulator.isaacsim,
    config=dataclasses.replace(
        holosoma.config_values.simulator.isaacsim.config,
        bridge=BridgeConfig(enabled=True),
        virtual_gantry=VirtualGantryCfg(enabled=True),
        sim=dataclasses.replace(
            holosoma.config_values.simulator.isaacsim.config.sim,
            fps=200,
        ),
    ),
)

# MuJoCo with sim2sim optimizations
mujoco = dataclasses.replace(
    holosoma.config_values.simulator.mujoco,
    config=dataclasses.replace(
        holosoma.config_values.simulator.mujoco.config,
        bridge=BridgeConfig(enabled=True),
        virtual_gantry=VirtualGantryCfg(enabled=True),
        sim=dataclasses.replace(
            holosoma.config_values.simulator.mujoco.config.sim,
            fps=2000,  # mujoco can run faster
        ),
    ),
)

# MuJoCo Warp with sim2sim optimizations
mjwarp = dataclasses.replace(
    holosoma.config_values.simulator.mjwarp,
    config=dataclasses.replace(
        holosoma.config_values.simulator.mjwarp.config,
        bridge=BridgeConfig(enabled=True),
        virtual_gantry=VirtualGantryCfg(enabled=True),
        sim=dataclasses.replace(
            holosoma.config_values.simulator.mjwarp.config.sim,
            fps=400,
        ),
    ),
)

# FADA: publish ground-truth pose over ZMQ for external velocity estimators.
isaacgym = dataclasses.replace(
    isaacgym, config=dataclasses.replace(isaacgym.config, bridge=dataclasses.replace(isaacgym.config.bridge, publish_truth_pose=True))
)
isaacsim = dataclasses.replace(
    isaacsim, config=dataclasses.replace(isaacsim.config, bridge=dataclasses.replace(isaacsim.config.bridge, publish_truth_pose=True))
)
mujoco = dataclasses.replace(
    mujoco, config=dataclasses.replace(mujoco.config, bridge=dataclasses.replace(mujoco.config.bridge, publish_truth_pose=True))
)
mjwarp = dataclasses.replace(
    mjwarp, config=dataclasses.replace(mjwarp.config, bridge=dataclasses.replace(mjwarp.config.bridge, publish_truth_pose=True))
)

DEFAULTS = {
    "isaacgym": isaacgym,
    "isaacsim": isaacsim,
    "mujoco": mujoco,
    "mjwarp": mjwarp,
}
