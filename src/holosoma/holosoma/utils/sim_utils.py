"""Shared simulation utilities for holosoma.

This module provides common functionality for setting up and running simulations,
shared between eval_agent.py and run_sim.py.
"""

from __future__ import annotations

import dataclasses
import argparse
import os
import pickle
import sys
import threading
import time
import traceback
from typing import Any

import numpy as np
import zmq
from loguru import logger
from typing_extensions import Self

from holosoma.config_types.env import get_tyro_env_config
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.full_sim import FullSimConfig
from holosoma.config_types.run_sim import RunSimConfig
from holosoma.managers.terrain.manager import TerrainManager
from holosoma.utils.common import seeding
from holosoma.utils.helpers import get_class
from holosoma.utils.rate import RateLimiter
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.simulator_config import SimulatorType, get_simulator_type, set_simulator_type
from holosoma.utils.sync_rendezvous import SYNC_LOOPBACK_HOST, sync_peer_uid
from holosoma.utils.torch_utils import to_torch


def setup_simulator_imports(config: ExperimentConfig | RunSimConfig) -> None:
    """Setup simulator-specific imports without side effects.

    Parameters
    ----------
    config : ExperimentConfig | RunSimConfig
        Configuration containing simulator settings.
    """
    print("\n\n\nsimulator type: ", config.simulator)
    set_simulator_type(config.simulator)
    simulator_type = get_simulator_type()

    if simulator_type == SimulatorType.MUJOCO:
        import mujoco

        assert mujoco is not None
    elif simulator_type == SimulatorType.ISAACGYM:
        import isaacgym

        assert isaacgym is not None

    # IsaacSim imports handled in setup_isaaclab_launcher


def setup_isaaclab_launcher(config: ExperimentConfig | RunSimConfig, device: str | None = None) -> Any | None:
    """Handle IsaacSim-specific launcher setup.

    Parameters
    ----------
    config : ExperimentConfig | RunSimConfig
        Configuration containing simulator and training settings.
    device : str
        Resolved device string (e.g., 'cuda:0', 'cpu').

    Returns
    -------
    Any | None
        IsaacSim simulation app instance, or None for other simulators.
    """
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Run simulation with IsaacSim.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
    parser.add_argument("--env_spacing", type=int, default=20, help="Distance between environments in simulator.")
    parser.add_argument("--output_dir", type=str, default="logs", help="Directory to store the output.")
    AppLauncher.add_app_launcher_args(parser)

    # Parse known arguments to get argparse params
    args_cli, unknown_args = parser.parse_known_args()

    # Set values from config
    args_cli.num_envs = config.training.num_envs
    args_cli.seed = config.training.seed
    args_cli.env_spacing = config.simulator.config.scene.env_spacing
    args_cli.output_dir = config.logger.base_dir
    args_cli.headless = config.training.headless
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        # Distribute simulator across GPUs when using multi-gpu training
        args_cli.device = f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}"
    elif device is not None:
        # Use the resolved device
        args_cli.device = device
    else:  # AppLauncher auto-detects
        pass

    # Check if video recording is enabled and add --enable_cameras flag
    video_enabled = config.logger.video.enabled or config.logger.headless_recording
    if video_enabled:
        args_cli.enable_cameras = True

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    logger.info(f"IsaacSim args_cli: {args_cli}")
    logger.info(f"IsaacSim unknown_args: {unknown_args}")
    sys.argv = [sys.argv[0]] + unknown_args

    return simulation_app


def setup_keyboard_listener(env) -> threading.Thread:
    """Setup keyboard listener thread for simulation control.

    Parameters
    ----------
    env
        Environment instance to control.

    Returns
    -------
    threading.Thread
        Keyboard listener thread (already started).
    """

    def on_press(key, env):
        """Handle keyboard input for simulation control."""
        try:
            if hasattr(key, "char") and key.char:
                if key.char == "n":
                    if hasattr(env, "next_task"):
                        env.next_task()
                        logger.info("Moved to the next task.")
                # Force Control
                elif key.char == "1":
                    if hasattr(env, "apply_force_scale"):
                        env.apply_force_scale /= 2.0
                        logger.info(f"apply_force_scale: {env.apply_force_scale}")
                elif key.char == "2":
                    if hasattr(env, "apply_force_scale"):
                        env.apply_force_scale *= 2.0
                        logger.info(f"apply_force_scale: {env.apply_force_scale}")
        except AttributeError:
            pass

    def listen_for_keypress(env):
        """Listen for keyboard input in a separate thread."""
        try:
            # Delay import so that one can run the rest of this script in headless mode.
            # Trying to import pynput in headless mode gives the following error:
            # ImportError: this platform is not supported:
            # ('failed to acquire X connection: Bad display name ""', DisplayNameError(''))
            from pynput import keyboard as pynput_keyboard

            logger.info("Keyboard controls:")
            logger.info("  n - Next task (if supported)")
            logger.info("  1/2 - Decrease/Increase force scale (if supported)")

            with pynput_keyboard.Listener(on_press=lambda key: on_press(key, env)) as listener:
                listener.join()
        except ImportError:
            logger.warning("pynput not available - keyboard controls disabled")
        except Exception as e:
            logger.warning(f"Keyboard listener failed: {e}")

    key_listener_thread = threading.Thread(target=listen_for_keypress, args=(env,))
    key_listener_thread.daemon = True
    key_listener_thread.start()
    return key_listener_thread


def _resolve_direct_sim_terrain_config(config: RunSimConfig):
    """Apply deterministic spawn defaults for direct run_sim unless opted out."""
    terrain_cfg = config.terrain
    if not getattr(config, "deterministic_spawn", True):
        return terrain_cfg

    spawn_cfg = terrain_cfg.terrain_term.spawn
    if not spawn_cfg.randomize_tiles and spawn_cfg.xy_offset_range == 0.0:
        return terrain_cfg

    logger.info("Direct sim deterministic spawn enabled - using center-tile evaluation defaults")
    return dataclasses.replace(
        terrain_cfg,
        terrain_term=dataclasses.replace(
            terrain_cfg.terrain_term,
            spawn=dataclasses.replace(spawn_cfg, randomize_tiles=False, xy_offset_range=0.0),
        ),
    )


def setup_simulation_environment(
    config: ExperimentConfig | RunSimConfig, device: str | None = None
) -> tuple[Any, str, Any]:
    """Setup simulation environment with shared infrastructure.

    This function handles common setup for training, evaluation and direct simulation:
    - Simulator imports and initialization
    - Device selection and seeding
    - Environment creation
    - Keyboard listener setup (if not headless)

    Parameters
    ----------
    config : ExperimentConfig | RunSimConfig
        Configuration containing all simulation settings.
    device : str | None, optional
        Device to use for simulation. If None, auto-detects CUDA availability.

    Returns
    -------
    tuple[Any, str, Any]
        Tuple of (environment, device_string, simulation_app).
        simulation_app is None for simulators that don't need it (MuJoCo, IsaacGym).
    """
    logger.info("🚀 Setting up simulation environment...")

    # Setup simulator imports
    setup_simulator_imports(config)

    # Device selection - must happen before IsaacSim launcher setup
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Handle IsaacSim launcher if needed (for both ExperimentConfig and RunSimConfig)
    simulation_app = None
    if get_simulator_type() == SimulatorType.ISAACSIM:
        simulation_app = setup_isaaclab_launcher(config, device)

    # Set random seed if specified for experiment/eval style configs
    if isinstance(config, ExperimentConfig) and config.training.seed is not None:
        seeding(config.training.seed, torch_deterministic=config.training.torch_deterministic)
        logger.info(f"Seed: {config.training.seed}")

    # For RunSimConfig, we need a different approach since it doesn't have env_class or training configs
    if isinstance(config, RunSimConfig):
        if config.training.seed is not None:
            seeding(config.training.seed, torch_deterministic=config.training.torch_deterministic)
            logger.info(f"Direct sim seed: {config.training.seed}")

        # For run_sim.py, we'll create the simulator directly instead of using environment wrapper
        logger.info("Direct simulation mode - creating simulator directly, without experiment config")

        # Create FullSimConfig from RunSimConfig
        # Extract SimulatorInitConfig from SimulatorConfig
        full_config = FullSimConfig(
            simulator=config.simulator.config,  # Extract .config from SimulatorConfig
            robot=config.robot,
            training=config.training,
            logger=config.logger,
            experiment_dir=None,
        )

        # For compatibility, minimal proxy for TerrainManager since it depends on env
        class EnvProxy:
            def __init__(self, device):
                self.num_envs = 1
                self.device = device

        # For compatibility, wrap in a minimal object that has .sim attribute
        class DirectSimWrapper:
            def __init__(self, simulator):
                self.sim = simulator

            def reset(self):
                # Basic reset - just initialize the simulator if needed
                if hasattr(self.sim, "reset"):
                    self.sim.reset()

            def close(self):
                if hasattr(self.sim, "close"):
                    self.sim.close()

        # Use terrain configuration from RunSimConfig
        terrain_cfg = _resolve_direct_sim_terrain_config(config)
        terrain_manager = TerrainManager(terrain_cfg, env=EnvProxy(device), device=device)

        # Create simulator using get_class() to avoid circular imports
        simulator_class = get_class(config.simulator._target_)
        simulator = simulator_class(full_config, terrain_manager, device)

        # Now we have an "env" to return which is actually the direct simulator
        env = DirectSimWrapper(simulator)
        logger.debug("Direct simulator created successfully!")

    else:
        # Original ExperimentConfig path
        env_target = config.env_class
        tyro_env_config = get_tyro_env_config(config)

        logger.info(f"Creating environment: {env_target}")
        env_class = get_class(env_target)
        env = env_class(tyro_env_config, device=device)

        logger.debug("Environment created successfully!")

        # Setup keyboard listener if not headless
        if not config.training.headless:
            setup_keyboard_listener(env)

    return env, device, simulation_app


def close_simulation_app(simulation_app):
    """Close simulation app with workarounds for known issues.

    Parameters
    ----------
    simulation_app : Any
        The simulation app instance returned by init_sim_imports().
        Can be None for simulators that don't have an app (e.g., IsaacGym).
    """
    if simulation_app is not None and get_simulator_type() == SimulatorType.ISAACSIM:
        logger.info("Shutting down simulation app...")
        try:
            # Work-around for IsaacLab hanging headless.
            # Patch the close_stage method to avoid hanging
            import omni.usd

            context = omni.usd.get_context()
            context_class = context.__class__

            # Replace with a no-op version
            def noop_close_stage(self, *args, **kwargs):
                logger.info("Skipping close_stage() to avoid hanging")
                return True

            # Apply the patch
            context_class.close_stage = noop_close_stage
            logger.info("Successfully patched close_stage method")
        except Exception as e:
            logger.warning(f"Could not patch close_stage method: {e}")

        try:
            # Work-around for IsaacLab SimulationContext._app_control_on_stop_handle_fn
            # hanging in an infinite render() loop on shutdown. When simulation_app.close()
            # triggers a timeline STOP event, the callback spins waiting for the timeline to
            # start playing again — which never happens. Disabling the callback prevents this.
            from isaaclab.sim import SimulationContext

            sim_context = SimulationContext.instance()
            if sim_context is not None:
                sim_context._disable_app_control_on_stop_handle = True
                logger.info("Disabled SimulationContext app_control_on_stop_handle to prevent shutdown hang")
        except Exception as e:
            logger.warning(f"Could not disable app_control_on_stop_handle: {e}")

        # Now close the app
        simulation_app.close(wait_for_replicator=False)
        logger.info("Simulation app closed.")
    else:
        logger.info("Simulation app closed.")


class DirectSimulation:
    """Encapsulates direct simulation logic for run_sim.py.

    This class provides a clean interface for running direct simulations without
    training or evaluation environments, handling all initialization,
    loop management, and cleanup logic.

    Can be used as a context manager for resource management.

    Examples
    --------
    >>> with DirectSimulation(config, env, device, simulation_app) as sim:
    ...     sim.run()
    """

    def __init__(self, config: RunSimConfig, env: Any, device: str, simulation_app: Any):
        """Initialize DirectSimulation instance.

        Parameters
        ----------
        config : RunSimConfig
            Configuration containing all simulation settings.
        env : Any
            Environment wrapper containing the simulator.
        device : str
            Device for tensor operations.
        simulation_app : Any
            Simulation app instance (if any).
        """
        self.config = config
        self.env = env
        self.device = device
        self.simulation_app = simulation_app
        self.simulator = env.sim
        self._defer_startup_video_recording = False

    def __enter__(self) -> Self:
        """Context manager entry - initialize the simulation.

        Returns
        -------
        Self
            Self for use in the with statement.
        """
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - cleanup the simulation.

        Parameters
        ----------
        exc_type : type or None
            Exception type if an exception occurred.
        exc_val : Exception or None
            Exception instance if an exception occurred.
        exc_tb : traceback or None
            Traceback if an exception occurred.
        """
        self.cleanup()

    def initialize(self) -> None:
        """Handle the complete simulator initialization sequence.

        Performs the initialization process required for proper simulator
        lifecycle management. Ideally this is moved into the simulator interface and
        to simplify training, evaluation and direct usage.
        """
        logger.debug("Initializing simulator...")

        # Need to manually set headless since it's in training config currently
        self.simulator.set_headless(False)

        # Step 1: Basic setup
        self.simulator.setup()
        logger.debug("simulator.setup() completed")

        # Step 2: Setup terrain
        self.simulator.setup_terrain()
        logger.debug("simulator.setup_terrain() completed")

        # Step 3: Load assets (this initializes the bridge!)
        self.simulator.load_assets()
        logger.debug("simulator.load_assets() completed - bridge should now be initialized")

        # Step 4: Create environments (need to provide required parameters)
        env_origins = self._resolve_direct_env_origins()

        # Create base_init_state from robot config
        base_init_state = self._create_base_init_state()

        self.simulator.create_envs(1, env_origins, base_init_state)
        logger.debug("simulator.create_envs() completed")

        # Step 5: Prepare simulation
        self.simulator.prepare_sim()
        logger.debug("simulator.prepare_sim() completed")
        self._apply_direct_spawn_state()
        self._sync_direct_reset_pose_to_mujoco_defaults()
        logger.debug("Applied terrain-aware direct sim spawn state")

        self._defer_startup_video_recording = self._should_defer_startup_video_recording()
        if self._defer_startup_video_recording:
            self._set_auto_recording_enabled(False)
            logger.info("Deferring auto video recording until the first pipeline reset.")

        # Step 5.5: Initialize episode (positions virtual gantry, etc.)
        self.simulator.on_episode_start(env_id=0)
        self._log_direct_spawn_alignment("After init")
        logger.debug("simulator.on_episode_start() completed")

        # Step 6: Setup viewer if not headless
        if not self.config.training.headless:
            self.simulator.setup_viewer()
            logger.debug("simulator.setup_viewer() completed")

        logger.info("Simulator initialized")

        # Step 8: ZMQ control channel for external orchestration
        self._init_control_zmq()

        # Step 9: ZMQ sync channel for deterministic policy-sim stepping
        self._init_sync_zmq()

    def _resolve_direct_env_origins(self) -> torch.Tensor:
        """Resolve direct-sim environment origins from the terrain manager when available."""
        terrain_manager = getattr(self.simulator, "terrain_manager", None)
        if terrain_manager is not None:
            terrain_state = terrain_manager.get_state("locomotion_terrain")
            env_origins = getattr(terrain_state, "env_origins", None)
            if env_origins is not None:
                env_origins_tensor = torch.as_tensor(env_origins, device=self.device, dtype=torch.float32)
                if env_origins_tensor.dim() == 2 and env_origins_tensor.shape == (1, 3):
                    return env_origins_tensor
                logger.warning(
                    f"Direct sim terrain env_origins has unexpected shape {tuple(env_origins_tensor.shape)}; "
                    "falling back to origin."
                )

        logger.warning("Direct sim terrain env_origins unavailable; falling back to [0, 0, 0]")
        return torch.zeros(1, 3, device=self.device)

    def _apply_direct_spawn_state(self) -> None:
        """Align direct-sim robot root pose with the configured terrain origin."""
        sim = self.simulator
        robot_cfg = getattr(sim, "robot_config", None)
        root_states = getattr(sim, "robot_root_states", None)
        if robot_cfg is None or root_states is None:
            return

        sim_device = getattr(sim, "sim_device", self.device)
        state_dtype = torch.float32
        env_origins = getattr(getattr(sim, "scene", None), "env_origins", None)
        if env_origins is None:
            env_origins = self._resolve_direct_env_origins()
        env_origins = torch.as_tensor(env_origins, device=sim_device, dtype=state_dtype)
        if env_origins.dim() != 2 or env_origins.shape[1] != 3:
            raise ValueError(f"Direct sim env_origins must have shape [num_envs, 3], got {tuple(env_origins.shape)}")

        init_state = robot_cfg.init_state
        base_pos = torch.tensor(init_state.pos, device=sim_device, dtype=state_dtype)
        base_rot = torch.tensor(init_state.rot, device=sim_device, dtype=state_dtype)
        base_lin_vel = torch.tensor(init_state.lin_vel, device=sim_device, dtype=state_dtype)
        base_ang_vel = torch.tensor(init_state.ang_vel, device=sim_device, dtype=state_dtype)

        root_states[:, :3] = env_origins + base_pos.unsqueeze(0)
        root_states[:, 3:7] = base_rot.unsqueeze(0)
        root_states[:, 7:10] = base_lin_vel.unsqueeze(0)
        root_states[:, 10:13] = base_ang_vel.unsqueeze(0)

        env_ids = torch.arange(root_states.shape[0], device=sim_device, dtype=torch.long)
        sim.set_actor_root_state_tensor_robots(env_ids, root_states)
        if hasattr(sim, "refresh_sim_tensors"):
            sim.refresh_sim_tensors()
        gantry = getattr(sim, "virtual_gantry", None)
        if gantry is not None:
            gantry.set_position_to_robot()

    def _sync_direct_reset_pose_to_mujoco_defaults(self) -> None:
        """Make MuJoCo native/viewer reset use the terrain-aware spawn pose."""
        sim = self.simulator
        root_model = getattr(sim, "root_model", None)
        root_data = getattr(sim, "root_data", None)
        qpos_addr = getattr(sim, "robot_qpos_addr", None)
        if root_model is None or root_data is None or qpos_addr is None:
            return

        # MuJoCo's built-in reset path restores qpos from model.qpos0. Direct
        # sim moves the robot after model compilation, so keep qpos0's freejoint
        # root pose in sync or viewer/native resets will jump back to XML origin.
        root_model.qpos0[qpos_addr : qpos_addr + 7] = root_data.qpos[qpos_addr : qpos_addr + 7]

    def _should_defer_startup_video_recording(self) -> bool:
        """Record only actual pipeline runs, not simulator startup warmup."""
        recorder = getattr(self.simulator, "video_recorder", None)
        control_port = int(getattr(self.config.simulator.config, "control_zmq_port", -1))
        return bool(recorder is not None and recorder.enabled and control_port > 0)

    def _set_auto_recording_enabled(self, enabled: bool) -> None:
        """Toggle episode-triggered auto recording if the recorder supports it."""
        recorder = getattr(self.simulator, "video_recorder", None)
        if recorder is None:
            return
        setter = getattr(recorder, "set_auto_recording_enabled", None)
        if callable(setter):
            setter(bool(enabled))

    def _arm_deferred_video_recording(self) -> None:
        """Enable video recording on the first external reset and renumber from episode 1."""
        if not self._defer_startup_video_recording:
            return
        recorder = getattr(self.simulator, "video_recorder", None)
        if recorder is not None:
            reset_counter = getattr(recorder, "reset_episode_counter", None)
            if callable(reset_counter):
                reset_counter()
        self._set_auto_recording_enabled(True)
        self._defer_startup_video_recording = False
        logger.info("Pipeline video recording armed on first reset.")

    # ------------------------------------------------------------------
    # ZMQ control channel for pipeline automation
    # ------------------------------------------------------------------

    def _init_control_zmq(self) -> None:
        """Bind a ZMQ REP socket if ``control_zmq_port`` is configured (> 0)."""
        self._zmq_ctx: zmq.Context | None = None
        self._zmq_sock: zmq.Socket | None = None

        port = getattr(self.config.simulator.config, "control_zmq_port", -1)
        if port <= 0:
            return

        self._zmq_ctx = zmq.Context()
        self._zmq_sock = self._zmq_ctx.socket(zmq.REP)
        self._zmq_sock.bind(f"tcp://*:{port}")
        logger.info(f"Control ZMQ REP socket bound on tcp://*:{port}")

    def _poll_control_zmq(self) -> None:
        """Non-blocking poll for one incoming control command and reply."""
        if self._zmq_sock is None:
            return
        if self._zmq_sock.poll(timeout=0, flags=zmq.POLLIN):
            raw = self._zmq_sock.recv_json()
            reply = self._handle_control_command(raw)
            self._zmq_sock.send_json(reply)

    def _handle_control_command(self, msg: dict) -> dict:
        """Dispatch a JSON control message.  Returns a JSON-serialisable reply."""
        cmd = msg.get("cmd", "")
        gantry = self.simulator.virtual_gantry

        if cmd == "status":
            return {
                "ok": True,
                "gantry_enabled": gantry.enabled if gantry else None,
                "gantry_length": gantry.length if gantry else None,
            }

        if cmd == "gantry_set_length":
            if gantry is None:
                return {"ok": False, "error": "no gantry"}
            gantry.length = float(msg.get("length", gantry.length))
            logger.info(f"[ctrl] gantry length set to {gantry.length:.2f}")
            return {"ok": True, "length": gantry.length}

        if cmd == "gantry_enable":
            if gantry is None:
                return {"ok": False, "error": "no gantry"}
            gantry.set_enable(True)
            logger.info("[ctrl] gantry enabled")
            return {"ok": True, "enabled": True}

        if cmd == "gantry_disable":
            if gantry is None:
                return {"ok": False, "error": "no gantry"}
            gantry.set_enable(False)
            logger.info("[ctrl] gantry disabled (released)")
            return {"ok": True, "enabled": False}

        if cmd == "get_robot_state":
            return self._get_robot_state_reply()

        if cmd == "reset":
            self._reset_episode()
            # Only signal a Phase-1 restart when we are already in Phase 2.
            # A reset arriving during Phase 1 (before the policy sent SYNC) does not
            # restart Phase 1: the existing _saved_qpos is already correct and Phase 1b
            # restores it.
            if self._sync_in_phase2:
                self._sync_restart_requested = True
                logger.info("[ctrl] episode reset — signalling sync loop to restart Phase 1")
            else:
                logger.info("[ctrl] episode reset (Phase 1: continuing without restart)")
            return {"ok": True}

        if cmd == "shutdown":
            logger.info("[ctrl] shutdown requested")
            self._shutdown_requested = True
            return {"ok": True}

        return {"ok": False, "error": f"unknown command: {cmd}"}

    def _clear_direct_external_forces(self) -> None:
        """Clear MuJoCo external forces that can persist across teleports/resets."""
        applied_forces = getattr(self.simulator, "applied_forces", None)
        if applied_forces is None:
            return
        if hasattr(applied_forces, "zero_"):
            applied_forces.zero_()
            return
        applied_forces[...] = 0.0

    def _log_direct_spawn_alignment(self, context: str) -> None:
        """Log robot/gantry alignment after direct sim spawn/reset."""
        gantry = getattr(self.simulator, "virtual_gantry", None)
        root_states = getattr(self.simulator, "robot_root_states", None)
        if root_states is None:
            return

        robot_pos = root_states[0, :3].detach().cpu().numpy()
        if gantry is None:
            logger.info(f"{context}: robot_pos={robot_pos.tolist()} gantry=None")
            return

        gantry_point = gantry.point
        xy_error = float(((gantry_point[:2] - robot_pos[:2]) ** 2).sum() ** 0.5)
        logger.info(
            f"{context}: robot_pos={robot_pos.tolist()} "
            f"gantry_point={gantry_point.tolist()} gantry_xy_error={xy_error:.6f}"
        )

    def _reset_episode(self) -> None:
        """Full deterministic reset: gantry → physics → joints → buffers → episode.

        Guarantees every experiment starts from the exact same initial state.
        """
        import mujoco as mj
        import torch

        sim = self.simulator

        # Close out the previous episode's recording before we wipe simulator state.
        recorder = getattr(sim, "video_recorder", None)
        if recorder is not None and getattr(recorder, "is_recording", False):
            frame_count_fn = getattr(recorder, "_get_frame_count", None)
            frame_count = int(frame_count_fn()) if callable(frame_count_fn) else 0
            if frame_count > 0:
                sim.on_episode_end(env_id=0)

        # 1. Release gantry forces before teleporting the robot. We re-enable
        # it after the final spawn state is applied and the anchor is aligned.
        if sim.virtual_gantry is not None:
            sim.virtual_gantry.set_enable(False)
        self._clear_direct_external_forces()

        # 2. Multiple reset cycles to fully clear MuJoCo state
        for _ in range(3):
            mj.mj_resetData(sim.root_model, sim.root_data)
            sim._set_robot_initial_state()
            self._apply_direct_spawn_state()
            self._clear_direct_external_forces()
            mj.mj_forward(sim.root_model, sim.root_data)

        # 3. Set default joint angles + zero all velocities
        self._apply_default_joint_angles()

        # 4. Clear simulator-level torch buffers for full determinism
        if hasattr(sim, "commands") and sim.commands is not None:
            sim.commands.fill_(0.0)
        if hasattr(sim, "contact_forces_history"):
            sim.contact_forces_history.zero_()
        if hasattr(sim, "contact_forces"):
            sim.contact_forces.zero_()
        env_ids = torch.arange(sim.num_envs, device=sim.sim_device)
        if hasattr(sim, "clear_contact_forces_history"):
            sim.clear_contact_forces_history(env_ids)

        # 5. Step physics a few times with zero ctrl to settle transients
        sim.root_data.ctrl[:] = 0.0
        self._clear_direct_external_forces()
        for _ in range(10):
            mj.mj_step(sim.root_model, sim.root_data)
        # Re-apply pose after settling (physics steps may have shifted it)
        sim._set_robot_initial_state()
        self._apply_direct_spawn_state()
        self._apply_default_joint_angles()
        self._sync_direct_reset_pose_to_mujoco_defaults()
        if sim.virtual_gantry is not None:
            sim.virtual_gantry.set_enable(True)

        # 6. Re-initialise episode (repositions gantry anchor over robot, etc.)
        self._arm_deferred_video_recording()
        sim.on_episode_start(env_id=0)
        self._log_direct_spawn_alignment("After reset")

    def _get_robot_state_reply(self) -> dict:
        """Return the robot's base position, orientation and velocities for fall detection."""
        import numpy as np

        sim = self.simulator
        if sim.root_data is None or sim.root_model is None:
            return {"ok": False, "error": "sim not initialised"}

        robot_body_id = 1
        if robot_body_id >= sim.root_model.nbody:
            return {"ok": False, "error": "robot body not found"}

        pos = sim.root_data.xpos[robot_body_id].tolist()
        quat = sim.root_data.xquat[robot_body_id].tolist()  # [w,x,y,z]

        # Projected gravity in body frame (used for tilt detection)
        # gravity_world = [0, 0, -1], rotate into body frame
        rot_mat = np.zeros(9)
        import mujoco as mj
        mj.mju_quat2Mat(rot_mat, sim.root_data.xquat[robot_body_id])
        rot_mat = rot_mat.reshape(3, 3)
        proj_gravity = (rot_mat.T @ np.array([0.0, 0.0, -1.0])).tolist()
        uprightness_r22 = float(rot_mat[2, 2])

        gantry = getattr(sim, "virtual_gantry", None)
        gantry_info: dict[str, Any] = {}
        if gantry is not None:
            point = None if gantry.point is None else np.asarray(gantry.point, dtype=np.float64)
            gantry_info.update(
                {
                    "gantry_enabled": bool(gantry.enabled),
                    "gantry_length": float(gantry.length),
                    "gantry_point": None if point is None else point.tolist(),
                }
            )
            if point is not None:
                pos_np = np.asarray(pos, dtype=np.float64)
                dx = point - pos_np
                distance = float(np.linalg.norm(dx))
                gantry_info["gantry_distance"] = distance
                if gantry.enabled and distance > 1e-9:
                    root_states = getattr(sim, "robot_root_states", None)
                    if root_states is not None:
                        vel_np = root_states[0, 7:10].detach().cpu().numpy().astype(np.float64)
                    else:
                        vel_np = np.zeros(3, dtype=np.float64)
                    gantry_info["gantry_force_estimate"] = gantry._advance(pos_np, vel_np).tolist()

        return {
            "ok": True,
            "base_pos": pos,
            "base_quat_wxyz": quat,
            "base_height": float(pos[2]),
            "uprightness_r22": uprightness_r22,
            "projected_gravity": proj_gravity,
            **gantry_info,
        }

    def _apply_default_joint_angles(self) -> None:
        """Quietly set all DOF joint angles to robot_config defaults and zero DOF velocities."""
        import mujoco as mj
        import numpy as np

        sim = self.simulator
        if sim.robot_config is None or sim.root_model is None or sim.root_data is None:
            return

        default_angles = sim.robot_config.init_state.default_joint_angles
        for joint_name, angle in default_angles.items():
            mujoco_name = sim._get_prefixed_name(joint_name)
            for i in range(sim.root_model.njnt):
                if sim.root_model.joint(i).name == mujoco_name:
                    sim.root_data.qpos[sim.root_model.jnt_qposadr[i]] = angle
                    break

        # Zero all DOF velocities for a clean start
        sim.root_data.qvel[:] = np.zeros_like(sim.root_data.qvel)

        mj.mj_forward(sim.root_model, sim.root_data)

    def _cleanup_control_zmq(self) -> None:
        if self._zmq_sock is not None:
            self._zmq_sock.setsockopt(zmq.LINGER, 0)
            self._zmq_sock.close()
        if self._zmq_ctx is not None:
            self._zmq_ctx.destroy(linger=0)
        self._zmq_sock = None
        self._zmq_ctx = None

    # ------------------------------------------------------------------
    # ZMQ sync channel for deterministic policy-sim stepping
    # ------------------------------------------------------------------

    def _init_sync_zmq(self) -> None:
        """Bind a ZMQ PAIR socket if ``policy_sync_zmq_port`` is configured (> 0)."""
        self._sync_zmq_ctx: zmq.Context | None = None
        self._sync_zmq_sock: zmq.Socket | None = None
        self._sync_restart_requested: bool = False
        self._sync_in_phase2: bool = False

        port = getattr(self.config.simulator.config, "policy_sync_zmq_port", -1)
        if port <= 0:
            return

        self._sync_zmq_ctx = zmq.Context()
        self._sync_zmq_sock = self._sync_zmq_ctx.socket(zmq.PAIR)
        # Loopback, not "*": lock-step stepping is a same-host rendezvous between this
        # process and the policy process, so the channel is not bound on other interfaces.
        endpoint = f"tcp://{SYNC_LOOPBACK_HOST}:{port}"
        try:
            self._sync_zmq_sock.bind(endpoint)
        except zmq.ZMQError as exc:
            # Raise rather than fall back to wall-clock stepping: a busy port means
            # another simulator already owns this rendezvous.
            self._cleanup_sync_zmq()
            raise RuntimeError(
                f"Could not bind the policy sync socket on {endpoint}: {exc}. "
                "Another simulator is most likely already using this port. "
                "To run a second rollout at the same time, give it its own port on both sides: "
                "--simulator.config.policy-sync-zmq-port=<port> for the simulator and "
                "--task.sim-step-sync-url=tcp://127.0.0.1:<port> for the policy. "
                "To run without lock-step stepping, pass "
                "--simulator.config.policy-sync-zmq-port=-1."
            ) from exc
        logger.info(f"Policy sync ZMQ PAIR socket bound on {endpoint}")

    def _cleanup_sync_zmq(self) -> None:
        if self._sync_zmq_sock is not None:
            self._sync_zmq_sock.setsockopt(zmq.LINGER, 0)
            self._sync_zmq_sock.close()
        if self._sync_zmq_ctx is not None:
            self._sync_zmq_ctx.destroy(linger=0)
        self._sync_zmq_sock = None
        self._sync_zmq_ctx = None

    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the direct simulation loop with viewer sync and FPS logging.

        Supports two modes:

        - **Wall-clock mode** (default): ``rate_limiter.sleep()`` after each
          physics step.
        - **Sync mode** (``policy_sync_zmq_port > 0``): batch
          ``fps / rl_rate`` physics steps per policy step, then exchange
          DONE / STEP messages with the policy.  Guarantees deterministic
          physics-step counts per policy step.
        """
        self._shutdown_requested = False
        sim_frequency = self.config.simulator.config.sim.fps
        viewer_steps = self._calculate_viewer_steps()

        logger.info(f"Simulation rate: {sim_frequency} Hz ({1.0 / sim_frequency * 1000:.2f} ms)")
        logger.info(f"Viewer rate: {1 / self.config.viewer_dt:.1f} Hz (sync every {viewer_steps} steps)")

        simulator_type = get_simulator_type()
        if simulator_type in [SimulatorType.ISAACGYM, SimulatorType.ISAACSIM]:
            pre_step_refresh = self.simulator.refresh_sim_tensors
        else:
            pre_step_refresh = lambda: None  # noqa: E731

        skip_headless_mu_render = (
            self.config.training.headless and simulator_type == SimulatorType.MUJOCO
        )

        if self._sync_zmq_sock is not None:
            self._run_sync_loop(sim_frequency, viewer_steps, pre_step_refresh, skip_headless_mu_render)
        else:
            logger.info("Starting direct simulation loop (wall-clock rate limiting)...")
            logger.info("Press Ctrl+C to stop simulation")
            self._run_wallclock_loop(sim_frequency, viewer_steps, pre_step_refresh, skip_headless_mu_render)

    # ------------------------------------------------------------------
    # Wall-clock loop
    # ------------------------------------------------------------------

    def _run_wallclock_loop(
        self, sim_frequency: int, viewer_steps: int, pre_step_refresh, skip_headless_mu_render: bool,
    ) -> None:
        rate_limiter = RateLimiter(sim_frequency)
        step_count = 0
        start_time = time.time()
        fps_start_time = start_time

        while True:
            try:
                self._poll_control_zmq()
                if self._shutdown_requested:
                    logger.info("Shutdown requested via control channel")
                    break
                pre_step_refresh()
                self.simulator.simulate_at_each_physics_step()
                if step_count % viewer_steps == 0 and not skip_headless_mu_render:
                    self.simulator.render()
                if step_count > 0 and step_count % 1000 == 0:
                    fps_start_time = self._log_fps(step_count, fps_start_time)
                step_count += 1
                rate_limiter.sleep()
            except KeyboardInterrupt:
                logger.info("Simulation interrupted by user (Ctrl+C)")
                break
            except Exception as e:
                logger.error(f"Error during simulation step {step_count}: {e}")
                traceback.print_exc()
                break

        total_elapsed = time.time() - start_time
        avg_fps = step_count / total_elapsed if total_elapsed > 0 else 0
        logger.info(f"Simulation completed after {step_count} steps")
        logger.info(f"Average FPS: {avg_fps:.1f} (target: {sim_frequency})")

    # ------------------------------------------------------------------
    # Sync loop (deterministic lock-step with the policy process)
    # ------------------------------------------------------------------

    def _run_sync_loop(
        self, sim_frequency: int, viewer_steps: int, pre_step_refresh, skip_headless_mu_render: bool,
    ) -> None:
        sock = self._sync_zmq_sock
        step_count = 0
        batch_count = 0
        start_time = time.time()
        fps_start_time = start_time

        _has_mj_data = hasattr(self.simulator, "root_data") and self.simulator.root_data is not None
        _mj = None
        if _has_mj_data:
            import mujoco as _mj

        def _low_cmd_sequence() -> int | None:
            bridge = getattr(self.simulator, "bridge", None)
            robot_bridge = getattr(bridge, "robot_bridge", None)
            seq = getattr(robot_bridge, "_sdk2py_low_cmd_seq", None)
            if seq is None:
                return None
            try:
                return int(seq)
            except (TypeError, ValueError):
                return None

        def _reset_sync_low_cmd_for_phase2() -> None:
            bridge = getattr(self.simulator, "bridge", None)
            robot_bridge = getattr(bridge, "robot_bridge", None)
            resetter = getattr(robot_bridge, "reset_sync_low_cmd_array", None)
            if resetter is not None:
                resetter()

        def _capture_gantry_state() -> dict[str, Any] | None:
            gantry = getattr(self.simulator, "virtual_gantry", None)
            if gantry is None:
                return None
            return {
                "enabled": bool(gantry.enabled),
                "length": float(gantry.length),
                "point": np.array(gantry.point, dtype=np.float64, copy=True)
                if gantry.point is not None
                else None,
            }

        def _restore_gantry_state(state: dict[str, Any] | None) -> None:
            gantry = getattr(self.simulator, "virtual_gantry", None)
            if gantry is None or state is None:
                return
            point = state.get("point")
            if point is not None:
                gantry.point = np.array(point, dtype=np.float64, copy=True)
            gantry.length = float(state.get("length", gantry.length))
            gantry.set_enable(bool(state.get("enabled", gantry.enabled)))

        def _reset_mujoco_state_for_sync_phase2(
            saved_qpos: np.ndarray,
            gantry_state: dict[str, Any] | None,
        ) -> None:
            """Clear MuJoCo internals, then restore the deterministic sync start state."""
            if not _has_mj_data:
                return
            sim = self.simulator
            gantry = getattr(sim, "virtual_gantry", None)

            if gantry is not None:
                gantry.set_enable(False)
            self._clear_direct_external_forces()

            # mj_step carries state beyond qpos/qvel: solver warm-start, contacts,
            # external forces, actuator state. Phase 1 runs for a variable number of
            # wall-clock steps while the policy initializes, so those buffers are not
            # reproducible. Clear MuJoCo internals with a full reset, then put back the
            # deterministic qpos captured before Phase 1.
            for _ in range(3):
                _mj.mj_resetData(sim.root_model, sim.root_data)
                self._clear_direct_external_forces()
                _mj.mj_forward(sim.root_model, sim.root_data)

            sim.root_data.qpos[:] = saved_qpos
            sim.root_data.qvel[:] = 0.0
            sim.root_data.ctrl[:] = 0.0
            sim.root_data.time = 0.0
            self._clear_direct_external_forces()
            _mj.mj_forward(sim.root_model, sim.root_data)

            if hasattr(sim, "commands") and sim.commands is not None:
                sim.commands.fill_(0.0)
            if hasattr(sim, "contact_forces_history"):
                sim.contact_forces_history.zero_()
            if hasattr(sim, "contact_forces"):
                sim.contact_forces.zero_()
            try:
                import torch

                env_ids = torch.arange(sim.num_envs, device=sim.sim_device)
                if hasattr(sim, "clear_contact_forces_history"):
                    sim.clear_contact_forces_history(env_ids)
            except Exception:
                pass

            self._clear_direct_external_forces()
            self._sync_direct_reset_pose_to_mujoco_defaults()
            _restore_gantry_state(gantry_state)
            sim.root_data.time = 0.0
            _mj.mj_forward(sim.root_model, sim.root_data)

        def _apply_sync_low_cmd_payload(payload: dict[str, Any] | None) -> int | None:
            if not payload or payload.get("low_cmd_kind") != "booster":
                return None
            bridge = getattr(self.simulator, "bridge", None)
            robot_bridge = getattr(bridge, "robot_bridge", None)
            setter = getattr(robot_bridge, "set_sync_low_cmd_array", None)
            if setter is None:
                return None
            low_cmd = payload.get("low_cmd")
            seq = payload.get("low_cmd_seq")
            setter(low_cmd, seq=seq)
            try:
                return int(seq)
            except (TypeError, ValueError):
                return None

        def _sync_low_state_payload() -> dict[str, Any] | None:
            bridge = getattr(self.simulator, "bridge", None)
            robot_bridge = getattr(bridge, "robot_bridge", None)
            if robot_bridge is None:
                return None
            try:
                positions = self.simulator.dof_pos[0].detach().cpu().numpy().astype(np.float64, copy=True)
                velocities = self.simulator.dof_vel[0].detach().cpu().numpy().astype(np.float64, copy=True)
                quaternion, gyro, _acc = robot_bridge._get_base_imu_data()
                quat = quaternion.detach().cpu().numpy().astype(np.float64, copy=True)
                gyro_np = gyro.detach().cpu().numpy().astype(np.float64, copy=True)
                low_state = np.concatenate(
                    [
                        np.zeros(3, dtype=np.float64),
                        quat,
                        positions,
                        np.zeros(3, dtype=np.float64),
                        gyro_np,
                        velocities,
                    ]
                ).reshape(1, -1)
                return {
                    "type": "DONE",
                    "tick": int(float(self.simulator.time()) * 1e3),
                    "low_state": low_state.tolist(),
                }
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"Failed to build sync low-state payload: {exc}")
                return None

        command_ack_timeout_s = max(
            0.0,
            float(os.getenv("HOLOSOMA_SYNC_COMMAND_ACK_TIMEOUT_S", "0.2") or 0.0),
        )
        require_command_ack = str(
            os.getenv("HOLOSOMA_SYNC_REQUIRE_LOWCMD_ACK", "1")
        ).strip().lower() not in {"0", "false", "no", "off"}

        # Outer restart loop: re-enters Phase 1 after ctrl.reset() received
        # during Phase 2.  Each iteration = one full policy episode (Phase 1
        # handshake → Phase 1b settle → Phase 2 lock-step).  On ctrl.reset(),
        # _sync_restart_requested is set by _handle_control_command, Phase 2
        # breaks out, and this loop iterates back to Phase 1 so the sim runs
        # physics again (publishing DDS state) while the next policy initialises.
        while True:
            self._sync_restart_requested = False
            self._sync_in_phase2 = False  # entering Phase 1; reset is safe to ignore until Phase 2
            batch_count = 0  # reset per-episode so ack check skips the first 2 batches correctly

            # --- Phase 1: Run physics (publishing DDS state) while waiting for handshake ---
            # The policy needs DDS low-state to initialise before it can send the
            # SYNC handshake, and the bridge only publishes it while physics steps, so
            # physics runs at the normal sim rate during the wait, identical to the
            # wall-clock loop.
            #
            # Phase 1 runs for a variable number of steps (depends on policy init
            # time).  For bit-identical starting conditions in Phase 2, the initial
            # qpos/qvel are saved BEFORE Phase 1, then restored followed by a FIXED
            # settle after the handshake.  The gantry config (length, enable,
            # stiffness — set via control ZMQ during Phase 1) lives in the Python
            # virtual-gantry object, not in qpos/qvel, so it survives the restore.
            logger.info("Sync mode: stepping physics while waiting for policy handshake (SYNC:<rl_rate>) ...")
            if _has_mj_data:
                _saved_qpos = self.simulator.root_data.qpos.copy()
                _saved_qvel = self.simulator.root_data.qvel.copy()
                _robot_body_id = 1
                _saved_z = float(self.simulator.root_data.xpos[_robot_body_id][2])
                _saved_xy = self.simulator.root_data.xpos[_robot_body_id][:2].tolist()
                _g = getattr(self.simulator, "virtual_gantry", None)
                _g_info = ""
                if _g is not None:
                    _g_info = (f" gantry_point={_g.point.tolist()}"
                               f" gantry_enabled={_g.enabled} gantry_len={_g.length}")
                logger.info(
                    f"Sync Phase 1: saved robot_xy={_saved_xy} robot_z={_saved_z:.4f}{_g_info}"
                )
            rate_limiter = RateLimiter(sim_frequency)
            steps_per_batch: int | None = None
            phase1_steps = 0
            while steps_per_batch is None:
                # --- check for handshake (non-blocking) ---
                if sock.poll(timeout=0, flags=zmq.POLLIN):
                    msg = sock.recv()
                    try:
                        # "SYNC:<rl_rate>" or "SYNC:<rl_rate>:<uid>".  The uid guards
                        # against pairing with a different user's simulator: the default
                        # port is per-user, but two uids congruent modulo the port range
                        # map to the same port.
                        fields = msg.decode().split(":")
                        prefix, rate_str = fields[0], fields[1]
                        peer_uid = int(fields[2]) if len(fields) > 2 and fields[2] else None
                        if prefix == "SYNC":
                            own_uid = sync_peer_uid()
                            if peer_uid is not None and peer_uid != own_uid:
                                reason = (
                                    f"this simulator belongs to uid {own_uid}, the connecting policy to "
                                    f"uid {peer_uid}; refusing to lock-step across users. Give each user's "
                                    "simulator and policy their own port "
                                    "(--simulator.config.policy-sync-zmq-port / --task.sim-step-sync-url)."
                                )
                                logger.error(f"Sync handshake rejected: {reason}")
                                sock.send(
                                    pickle.dumps(
                                        {"type": "REJECT", "reason": reason},
                                        protocol=pickle.HIGHEST_PROTOCOL,
                                    )
                                )
                                continue
                            rl_rate = float(rate_str)
                            steps_per_batch = int(round(sim_frequency / rl_rate))
                            logger.info(
                                f"Sync handshake: rl_rate={rl_rate} Hz, fps={sim_frequency}, "
                                f"steps_per_batch={steps_per_batch} (after {phase1_steps} warm-up steps)"
                            )
                            break
                    except Exception as exc:
                        logger.warning(f"Sync handshake: bad message {msg!r}: {exc}")

                # --- service control channel ---
                self._poll_control_zmq()
                if self._shutdown_requested:
                    logger.info("Shutdown requested before sync handshake completed")
                    return
                if self._sync_restart_requested:
                    # reset received while waiting for handshake — restart Phase 1
                    logger.info("Sync: reset received during Phase 1 — restarting Phase 1")
                    break

                # --- step physics so bridge publishes DDS state ---
                # A simulator running on its own never receives the handshake, so Ctrl+C
                # here exits the same way the wall-clock loop does rather than unwinding
                # as an unhandled KeyboardInterrupt.
                try:
                    pre_step_refresh()
                    self.simulator.simulate_at_each_physics_step()
                    if phase1_steps % viewer_steps == 0 and not skip_headless_mu_render:
                        self.simulator.render()
                    phase1_steps += 1
                    rate_limiter.sleep()
                except KeyboardInterrupt:
                    logger.info("Simulation interrupted by user (Ctrl+C)")
                    return
                except Exception as exc:
                    logger.error(f"Error during sync warm-up step {phase1_steps}: {exc}")
                    traceback.print_exc()
                    return

            if self._shutdown_requested:
                return
            if self._sync_restart_requested:
                continue  # restart outer loop (re-enter Phase 1)

            # --- Phase 1b: Deterministic reset + fixed settle ---
            # Phase 1 intentionally runs wall-clock physics while the policy
            # initializes.  Do a full MuJoCo reset after handshake so solver
            # warm-start/contact/external-force state from that variable warm-up
            # cannot leak into Phase 2, then run a fixed settle under the gantry.
            _settle_seconds = 3.0
            _settle_steps = int(_settle_seconds * sim_frequency)
            if _has_mj_data:
                _phase1_gantry_state = _capture_gantry_state()
                _reset_mujoco_state_for_sync_phase2(_saved_qpos, _phase1_gantry_state)
                logger.info(
                    f"Sync: reset MuJoCo internals and restored deterministic qpos → running {_settle_steps} fixed settle steps "
                    f"({_settle_seconds}s at {sim_frequency} Hz) for deterministic Phase 2 ..."
                )
            else:
                logger.info(
                    f"Sync: running {_settle_steps} fixed settle steps "
                    f"({_settle_seconds}s at {sim_frequency} Hz) ..."
                )
            # Log gantry and robot state before settle for diagnostics.
            _gantry = getattr(self.simulator, "virtual_gantry", None)
            # Force the gantry on during the 3s settle regardless of the captured state:
            # the orchestrator's gantry_disable command can arrive on the control channel
            # before the policy's SYNC handshake, leaving the captured state False and the
            # settle running gantry-less.  The captured state is restored after the settle
            # so Phase 2 starts with whatever the orchestrator set.
            _settle_forced_gantry_on = False
            if _gantry is not None and not _gantry.enabled:
                _gantry.set_enable(True)
                _settle_forced_gantry_on = True
            if _has_mj_data and _gantry is not None:
                _robot_body_id = 1
                _pre_z = float(self.simulator.root_data.xpos[_robot_body_id][2])
                _g_pt = _gantry.point.tolist() if _gantry.point is not None else None
                _g_en = _gantry.enabled
                _g_len = _gantry.length
                logger.info(
                    f"Sync settle start: robot_z={_pre_z:.4f} "
                    f"gantry_enabled={_g_en} gantry_len={_g_len} "
                    f"gantry_point={_g_pt} "
                    f"forced_on={_settle_forced_gantry_on}"
                )

            # Step physics + gantry but skip bridge during settle so no
            # stale truth-pose messages pile up in the policy's ZMQ buffer.
            _backend = getattr(self.simulator, "backend", None)
            for _s in range(_settle_steps):
                if _gantry:
                    _gantry.step()
                if _backend is not None:
                    _backend.step()
                else:
                    pre_step_refresh()
                    self.simulator.simulate_at_each_physics_step()
                if _s % viewer_steps == 0 and not skip_headless_mu_render:
                    self.simulator.render()

            # Restore the gantry state captured before the settle.
            if _settle_forced_gantry_on and _gantry is not None:
                _gantry.set_enable(False)

            # Log robot height after settle — if already below threshold,
            # the settle failed to hold the robot.
            if _has_mj_data:
                _robot_body_id = 1
                _post_z = float(self.simulator.root_data.xpos[_robot_body_id][2])
                _post_en = _gantry.enabled if _gantry is not None else None
                logger.info(
                    f"Sync settle done: robot_z={_post_z:.4f} gantry_enabled={_post_en}"
                )
                if _post_z < 0.28:
                    logger.warning(
                        f"Sync settle: robot height {_post_z:.4f}m is below fall threshold "
                        f"BEFORE Phase 2 starts — gantry may have failed to hold the robot"
                    )

            # Reset sim time to 0 so Phase 2 starts at a consistent time origin.
            if _has_mj_data:
                self.simulator.root_data.time = 0.0
                logger.info("Sync: reset sim time to 0.0s — Phase 2 starting")

            # --- Phase 2: Main sync loop ---
            self._sync_in_phase2 = True
            logger.info(f"Starting direct simulation loop (SYNC mode: {steps_per_batch} physics steps/batch)...")
            if command_ack_timeout_s > 0.0:
                logger.info(
                    f"Sync lowcmd ack wait: up to {command_ack_timeout_s * 1000.0:.1f}ms "
                    "before each policy-driven physics batch "
                    f"({'required' if require_command_ack else 'best-effort'})"
                )

            _reset_sync_low_cmd_for_phase2()
            last_low_cmd_sequence = _low_cmd_sequence()
            pending_step_payload: dict[str, Any] | None = None
            _phase2_error: Exception | None = None
            try:
                while True:
                    direct_low_cmd_seq = _apply_sync_low_cmd_payload(pending_step_payload)
                    pending_step_payload = None
                    if direct_low_cmd_seq is not None:
                        last_low_cmd_sequence = direct_low_cmd_seq

                    if batch_count >= 2 and command_ack_timeout_s > 0.0 and direct_low_cmd_seq is None:
                        # The policy sends lowcmd over DDS and then immediately
                        # sends STEP over ZMQ.  Wait for the DDS subscriber callback
                        # to observe the command sequence for this policy step
                        # before the first physics substep consumes bridge torques.
                        deadline = time.time() + command_ack_timeout_s
                        observed_sequence = _low_cmd_sequence()
                        if require_command_ack and (
                            observed_sequence is None or last_low_cmd_sequence is None
                        ):
                            raise RuntimeError(
                                "Sync lowcmd ack is required, but the simulator cannot observe "
                                "the sdk2py lowcmd sequence. This would make command delivery "
                                "best-effort instead of deterministic."
                            )
                        while (
                            observed_sequence is not None
                            and last_low_cmd_sequence is not None
                            and observed_sequence == last_low_cmd_sequence
                            and time.time() < deadline
                        ):
                            time.sleep(0.0005)
                            observed_sequence = _low_cmd_sequence()
                        if observed_sequence is not None and observed_sequence != last_low_cmd_sequence:
                            last_low_cmd_sequence = observed_sequence
                        elif (
                            require_command_ack
                            and observed_sequence is not None
                            and last_low_cmd_sequence is not None
                        ):
                            raise RuntimeError(
                                "Sync lowcmd ack timeout: simulator did not observe a new "
                                f"policy lowcmd sequence within {command_ack_timeout_s * 1000.0:.1f}ms "
                                f"(last={last_low_cmd_sequence}, observed={observed_sequence}). "
                                "Aborting instead of advancing physics with a stale command."
                            )

                    # Run one batch of physics steps
                    _restart_from_batch = False
                    for _sub in range(steps_per_batch):
                        if _sub == 0:
                            # Check flags set by STEP-wait's _poll_control_zmq() — do NOT
                            # call _poll_control_zmq() here: control messages are applied
                            # after a batch, not mid-batch before the physics steps.
                            if self._shutdown_requested:
                                logger.info("Shutdown requested via control channel")
                                return
                            if self._sync_restart_requested:
                                logger.info("Sync: reset during Phase 2 physics — restarting Phase 1")
                                _restart_from_batch = True
                                break
                        pre_step_refresh()
                        self.simulator.simulate_at_each_physics_step()
                        if step_count % viewer_steps == 0 and not skip_headless_mu_render:
                            self.simulator.render()
                        if step_count > 0 and step_count % 1000 == 0:
                            fps_start_time = self._log_fps(step_count, fps_start_time)
                        step_count += 1

                    if _restart_from_batch:
                        break  # exit Phase 2 while True

                    # Tell policy this batch is finished
                    payload = _sync_low_state_payload()
                    if payload is None:
                        sock.send(b"DONE")
                    else:
                        sock.send(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
                    batch_count += 1

                    # Wait for next STEP (with control ZMQ polling for shutdown/restart)
                    while True:
                        # Service the control channel FIRST.  Both STEP paths below leave
                        # this loop via `break`, so a poll placed after them would only run
                        # on an iteration where no STEP arrived — i.e. only when the policy
                        # is late — and `gantry_set_length` / `gantry_disable` would time
                        # out with zmq.error.Again while the policy keeps up.  The poll is
                        # non-blocking (timeout=0) and returns immediately when nothing is
                        # pending.
                        self._poll_control_zmq()
                        if sock.poll(timeout=100, flags=zmq.POLLIN):
                            msg = sock.recv()
                            if msg == b"STEP":
                                pending_step_payload = None
                                break
                            try:
                                payload = pickle.loads(msg)
                            except Exception:
                                payload = None
                            if isinstance(payload, dict) and payload.get("type") == "STEP":
                                pending_step_payload = payload
                                break
                            logger.warning(f"Sync channel: unexpected message {msg!r}, ignoring")
                        if self._shutdown_requested:
                            logger.info("Shutdown requested during sync wait")
                            return
                        if self._sync_restart_requested:
                            logger.info("Sync: reset during Phase 2 wait — restarting Phase 1")
                            break  # break inner STEP-wait loop

                    if self._sync_restart_requested:
                        break  # break Phase 2 while True

            except KeyboardInterrupt:
                logger.info("Simulation interrupted by user (Ctrl+C)")
                break  # exit outer restart loop
            except Exception as e:
                _phase2_error = e
                logger.error(f"Error during sync simulation step {step_count}: {e}")
                traceback.print_exc()
                break  # exit outer restart loop

            if self._shutdown_requested or _phase2_error is not None:
                break  # exit outer restart loop (normal shutdown or unrecoverable error)
            if self._sync_restart_requested:
                continue  # restart outer loop (Phase 1 for next policy episode)
            break  # Phase 2 exited without restart — normal end

        total_elapsed = time.time() - start_time
        avg_fps = step_count / total_elapsed if total_elapsed > 0 else 0
        logger.info(
            f"Sync simulation completed: {step_count} physics steps, "
            f"{batch_count} batches, avg FPS {avg_fps:.1f}"
        )

    def cleanup(self) -> None:
        """Handle simulation cleanup."""
        self._cleanup_control_zmq()
        self._cleanup_sync_zmq()

        # Cleanup environment
        if hasattr(self.env, "close"):
            self.env.close()

        if self.simulator.video_recorder:
            self.simulator.video_recorder.cleanup()

        # Cleanup simulation app
        if self.simulation_app:
            close_simulation_app(self.simulation_app)

    def _create_base_init_state(self) -> torch.Tensor:
        """Create base initialization state tensor from robot configuration.

        Returns
        -------
        torch.Tensor
            Base initialization state tensor.
        """
        base_init_state_list = (
            self.config.robot.init_state.pos
            + self.config.robot.init_state.rot
            + self.config.robot.init_state.lin_vel
            + self.config.robot.init_state.ang_vel
        )
        return to_torch(base_init_state_list, device=self.device, requires_grad=False)

    def _calculate_viewer_steps(self) -> int:
        """Calculate viewer synchronization frequency.

        Returns
        -------
        int
            Number of simulation steps between viewer updates.
        """
        viewer_dt = self.config.viewer_dt
        sim_dt = 1.0 / self.config.simulator.config.sim.fps
        return max(1, int(viewer_dt / sim_dt))

    def _log_fps(self, step_count: int, fps_start_time: float) -> float:
        """Log FPS statistics for simulation performance monitoring.

        Parameters
        ----------
        step_count : int
            Current step count.
        fps_start_time : float
            Start time for FPS measurement.

        Returns
        -------
        float
            New start time for next FPS measurement.
        """
        elapsed = time.time() - fps_start_time
        fps = 1000 / elapsed
        logger.info(f"Simulation FPS: {fps:.1f}")
        return time.time()
