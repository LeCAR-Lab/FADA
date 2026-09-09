"""GPU-accelerated batched MuJoCo backend using mujoco_warp.

This backend provides GPU-accelerated parallel simulation of multiple environments
using mujoco_warp, which leverages Warp's kernel compilation and batched execution.

Key features:
- GPU-accelerated simulation via Warp kernels
- Batched parallel environments (1 to thousands)
- Zero-copy PyTorch tensor access via wp.to_torch()
- Automatic contact force computation (cfrc_ext)
- Efficient GPU->CPU sync for rendering only when needed

Optional Dependencies
---------------------
This module requires optional dependencies that are NOT installed by default:
  - warp-lang: GPU kernel compilation framework
  - mujoco-warp: MuJoCo integration with Warp

To enable GPU acceleration, reinstall with warp support:
  bash scripts/setup_mujoco.sh --with-warp

Or install dependencies manually:
  pip install warp-lang mujoco-warp

System Requirements:
  - CUDA-capable GPU required
  - CUDA toolkit installed

If these dependencies are not available, the system will gracefully fall back
to ClassicBackend (CPU-only simulation).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import torch
from loguru import logger

from .base import IMujocoBackend
from .warp_bridge import WarpBridge

if TYPE_CHECKING:
    from holosoma.config_types.full_sim import FullSimConfig
    from holosoma.simulator.mujoco.tensor_views import BaseMujocoView


class WarpBackend(IMujocoBackend):
    """GPU-accelerated batched MuJoCo backend using mujoco_warp.

    This backend wraps mujoco_warp for GPU-accelerated parallel simulation
    of multiple environments. It provides zero-copy access to simulation
    state via PyTorch tensors that share memory with Warp arrays.

    Key characteristics:
    - Multi-environment support (1 to thousands)
    - GPU-based computation via Warp kernels
    - Automatic contact force computation (cfrc_ext tensor)
    - Zero-copy PyTorch tensor access
    - Efficient GPU->CPU sync only for rendering

    Requirements:
    - warp-lang package
    - mujoco_warp package
    - CUDA-capable GPU
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, config: FullSimConfig, device: str):
        """Initialize WarpBackend with GPU context and batched data.

        Parameters
        ----------
        model : mujoco.MjModel
            Compiled MuJoCo model
        data : mujoco.MjData
            MuJoCo data structure (used for CPU rendering only)
        config : FullSimConfig
            Full simulation configuration
        device : str
            Device string (e.g., 'cuda:0')

        Raises
        ------
        ImportError
            If warp or mujoco_warp packages are not installed
        RuntimeError
            If GPU initialization fails
        """
        super().__init__(model, data, config, device)

        # Import warp packages (fail fast if not available)
        try:
            import mujoco_warp as mjw
            import warp as wp
        except ImportError as e:
            raise ImportError(
                "WarpBackend requires 'warp-lang' and 'mujoco_warp' packages. "
                "Install with: pip install warp-lang mujoco-warp"
            ) from e

        # Initialize Warp runtime
        wp.init()
        self.mjw_device = wp.get_device(device)

        logger.info(f"Initializing WarpBackend: {self.num_envs} envs on {device}")

        # Get memory allocation config
        warp_config = config.simulator.mujoco_warp
        nconmax_per_env = warp_config.nconmax_per_env
        njmax_per_env = warp_config.njmax_per_env

        # Auto-calculate njmax if not specified
        if njmax_per_env is None:
            njmax_per_env = max(nconmax_per_env * 6, model.nv * 4)

        logger.info(f"GPU memory allocation: nconmax={nconmax_per_env} per env, njmax={njmax_per_env} per env")

        # Create Warp model and batched data within GPU context
        with wp.ScopedDevice(self.mjw_device):
            # Upload model to GPU
            self.mjw_model = mjw.put_model(model)

            # Create bridge for tensor-like access to model fields (for randomization)
            self.warp_model_bridge = WarpBridge(self.mjw_model, nworld=self.num_envs)

            # Allocate batched data for parallel environments
            # Memory allocation strategy (following mujoco_warp API):
            # - nconmax: contacts per environment (not total across all environments)
            # - njmax: constraints per environment
            # - naconmax: total contacts across ALL environments (auto-calculated internally)
            self.mjw_data = mjw.make_data(
                model,
                nworld=self.num_envs,
                nconmax=nconmax_per_env,
                njmax=njmax_per_env,
            )

            # Create zero-copy PyTorch tensors tethered to Warp arrays
            # These tensors share memory with the Warp arrays - no copying!
            self.qpos_t = wp.to_torch(self.mjw_data.qpos)  # [num_envs, nq]
            self.qvel_t = wp.to_torch(self.mjw_data.qvel)  # [num_envs, nv]
            self.qacc_t = wp.to_torch(self.mjw_data.qacc)  # [num_envs, nv]
            self.ctrl_t = wp.to_torch(self.mjw_data.ctrl)  # [num_envs, nu]
            self.cfrc_t = wp.to_torch(self.mjw_data.cfrc_ext)  # [num_envs, nbody, 6]
            self.xfrc_applied_t = wp.to_torch(self.mjw_data.xfrc_applied)  # [num_envs, nbody, 6]

            # Rigid body state tensors (for zero-copy access during refresh_sim_tensors)
            self.xpos_t = wp.to_torch(self.mjw_data.xpos)  # [num_envs, nbody, 3] - positions
            self.xquat_t = wp.to_torch(self.mjw_data.xquat)  # [num_envs, nbody, 4] - orientations [w,x,y,z]
            self.cvel_t = wp.to_torch(self.mjw_data.cvel)  # [num_envs, nbody, 6] - velocities [ang(3), lin(3)]

        # Keep reference to CPU data for rendering (synced on demand)
        self.render_data = data

        # Capture simulation step as CUDA graph for optimal performance
        # This eliminates per-kernel launch overhead (~20-30 kernels per step)
        # and enables GPU pipelining, providing 5-10x speedup
        logger.info("Capturing CUDA graph for simulation step...")
        with wp.ScopedDevice(self.mjw_device):
            with wp.ScopedCapture() as capture:
                mjw.step(self.mjw_model, self.mjw_data)
            self.step_graph = capture.graph
        logger.info("CUDA graph captured successfully")

        logger.info(
            f"WarpBackend initialized: {model.nbody} bodies, {model.nq} qpos, {model.nv} qvel, {model.nu} actuators"
        )

    def initialize_state(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Initialize GPU state from CPU data after construction.

        This syncs the initial state set by _set_robot_initial_state()
        and _set_initial_joint_angles() to GPU tensors, overriding the
        default qpos0 values from the MJCF model.

        Parameters
        ----------
        model : mujoco.MjModel
            MuJoCo model (for context)
        data : mujoco.MjData
            CPU MjData with initial state to copy to GPU
        """
        import mujoco_warp as mjw
        import numpy as np
        import warp as wp

        logger.info("Syncing initial state from CPU to GPU...")

        with wp.ScopedDevice(self.mjw_device):
            # Copy qpos and qvel from CPU to GPU for all environments
            # Tile the state across all environments
            qpos_cpu = np.tile(data.qpos, (self.num_envs, 1))
            qvel_cpu = np.tile(data.qvel, (self.num_envs, 1))

            wp.copy(self.mjw_data.qpos, wp.array(qpos_cpu, dtype=float))
            wp.copy(self.mjw_data.qvel, wp.array(qvel_cpu, dtype=float))

            # Compute forward kinematics to update derived quantities
            # (body positions, orientations, etc.)
            mjw.forward(self.mjw_model, self.mjw_data)

        logger.info("Initial state synced to GPU successfully")

    def step(self) -> None:
        """Advance batched simulation by one timestep using CUDA graph.

        Launches the pre-captured CUDA graph containing all simulation kernels.
        This eliminates per-kernel launch overhead (~20-30 kernels/step) and
        provides 5-10x speedup compared to individual kernel launches.

        Note: No explicit synchronization here - allows CPU-GPU pipelining.
        GPU work completes asynchronously while CPU prepares next frame.
        Synchronization happens only when needed (e.g., in get_render_data()).
        """
        import warp as wp

        with wp.ScopedDevice(self.mjw_device):
            wp.capture_launch(self.step_graph)
            # No wp.synchronize() - let GPU work in parallel with CPU

    def get_render_data(self, world_id: int = 0) -> mujoco.MjData:
        """Sync GPU data to CPU for rendering.

        Copies state from the specified environment from GPU to CPU for
        visualization. This is the only operation that requires GPU->CPU
        synchronization.

        CRITICAL: We must synchronize BEFORE copying GPU→CPU to ensure the
        GPU has completed all pending work. Without this, we would copy
        stale/incomplete data from a previous frame.

        Parameters
        ----------
        world_id : int, default=0
            Which environment to sync for visualization (0 to num_envs-1)

        Returns
        -------
        mujoco.MjData
            CPU MjData with state from the specified environment
        """
        import mujoco_warp as mjw
        import warp as wp

        # Validate world_id
        if world_id < 0 or world_id >= self.num_envs:
            logger.warning(f"Invalid world_id {world_id}, clamping to [0, {self.num_envs - 1}]")
            world_id = max(0, min(world_id, self.num_envs - 1))

        with wp.ScopedDevice(self.mjw_device):
            # CRITICAL: Synchronize GPU before copying to CPU
            # This ensures all GPU kernels have completed and data is ready
            wp.synchronize()

            # Now safe to copy GPU→CPU with guaranteed fresh data
            mjw.get_data_into(self.render_data, self.model, self.mjw_data, world_id=world_id)

        return self.render_data

    def get_ctrl_tensor(self) -> torch.Tensor:
        """Return control tensor for direct zero-copy writing.

        Returns
        -------
        torch.Tensor
            Control tensor [num_envs, nu] sharing memory with Warp array
        """
        return self.ctrl_t

    def refresh_sim_tensors(self, contact_history_tensor: torch.Tensor) -> None:
        """Update contact force history.

        Unlike ClassicBackend, WarpBackend automatically computes contact
        forces in the cfrc_ext tensor during simulation. We just need to
        update the rolling history buffer.

        Parameters
        ----------
        contact_history_tensor : torch.Tensor
            Contact force history buffer [num_envs, history_len, num_bodies, 3]
        """
        # cfrc_ext is already computed by Warp: [num_envs, num_bodies, 6]
        # Take first 3 components (forces, ignore torques)
        forces = self.cfrc_t[..., :3]  # [num_envs, num_bodies, 3]

        # Update history: shift old values right, add current at position 0
        contact_history_tensor[:] = torch.cat([forces.unsqueeze(1), contact_history_tensor[:, :-1]], dim=1)

    def create_root_view(self, addrs: dict) -> BaseMujocoView:
        """Create root state view using zero-copy tensors.

        Parameters
        ----------
        addrs : dict
            Address dictionary with slices for pos, quat, vel, ang_vel

        Returns
        -------
        MjwRootStateView
            Root state view with quaternion conversion and zero-copy access
        """
        from holosoma.simulator.mujoco.mjw_views import MjwRootStateView

        return MjwRootStateView(
            qpos=self.qpos_t,
            qvel=self.qvel_t,
            pos_slice=addrs["pos_indices"],
            quat_slice=addrs["quat_indices"],
            vel_slice=addrs["vel_indices"],
            ang_vel_slice=addrs["ang_vel_indices"],
            num_envs=self.num_envs,
        )

    def create_dof_pos_view(self, indices: slice, num_dof: int) -> torch.Tensor:
        """Return DOF position tensor directly (no wrapper needed).

        wp.to_torch() already returns a native PyTorch tensor that:
        - Shares memory with Warp array (zero-copy)
        - Supports all PyTorch operations natively
        - Works seamlessly on GPU

        Parameters
        ----------
        indices : slice
            Slice into qpos array
        num_dof : int
            Number of degrees of freedom (unused, for interface compatibility)

        Returns
        -------
        torch.Tensor
            DOF positions [num_envs, num_dof] - native PyTorch tensor
        """
        return self.qpos_t[:, indices]

    def create_dof_vel_view(self, indices: slice, num_dof: int) -> torch.Tensor:
        """Return DOF velocity tensor directly (no wrapper needed).

        wp.to_torch() already returns a native PyTorch tensor that:
        - Shares memory with Warp array (zero-copy)
        - Supports all PyTorch operations natively
        - Works seamlessly on GPU

        Parameters
        ----------
        indices : slice
            Slice into qvel array
        num_dof : int
            Number of degrees of freedom (unused, for interface compatibility)

        Returns
        -------
        torch.Tensor
            DOF velocities [num_envs, num_dof] - native PyTorch tensor
        """
        return self.qvel_t[:, indices]

    def create_dof_acc_view(self, indices: slice, num_dof: int) -> torch.Tensor:
        """Return DOF acceleration tensor directly (no wrapper needed).

        wp.to_torch() already returns a native PyTorch tensor that:
        - Shares memory with Warp array (zero-copy)
        - Supports all PyTorch operations natively
        - Works seamlessly on GPU

        Parameters
        ----------
        indices : slice
            Slice into qacc array
        num_dof : int
            Number of degrees of freedom (unused, for interface compatibility)

        Returns
        -------
        torch.Tensor
            DOF accelerations [num_envs, num_dof] - native PyTorch tensor
        """
        return self.qacc_t[:, indices]

    def create_force_view(self, num_bodies: int) -> torch.Tensor:
        """Return contact force tensor directly (no wrapper needed).

        wp.to_torch() already returns a native PyTorch tensor that:
        - Shares memory with Warp array (zero-copy)
        - Supports all PyTorch operations natively
        - Works seamlessly on GPU

        Parameters
        ----------
        num_bodies : int
            Number of bodies (unused, for interface compatibility)

        Returns
        -------
        torch.Tensor
            Contact forces [num_envs, num_bodies, 3] - native PyTorch tensor
        """
        # cfrc_ext is [num_envs, num_bodies, 6], take first 3 components (forces only)
        return self.cfrc_t[..., :3]

    def create_dof_state_view(self, dof_addrs: dict, num_dof: int) -> BaseMujocoView:
        """Create DOF state view using zero-copy GPU tensors.

        Parameters
        ----------
        dof_addrs : dict
            Dictionary with 'dof_pos_indices' and 'dof_vel_indices' slices
        num_dof : int
            Number of degrees of freedom

        Returns
        -------
        MjwDofStateView
            DOF state view with IsaacGym flattened format [num_envs * num_dof, 2]
        """
        from holosoma.simulator.mujoco.mjw_views import MjwDofStateView

        return MjwDofStateView(
            qpos=self.qpos_t,
            qvel=self.qvel_t,
            dof_pos_indices=dof_addrs["dof_pos_indices"],
            dof_vel_indices=dof_addrs["dof_vel_indices"],
            num_envs=self.num_envs,
            num_dof=num_dof,
        )

    def get_applied_forces_view(self) -> torch.Tensor:
        """Get writable view for external applied forces (GPU tensor).

        Returns zero-copy PyTorch tensor for applying external forces and
        torques to bodies directly on the GPU.

        Returns
        -------
        torch.Tensor
            Writable GPU tensor [num_envs, num_bodies, 6]
            - [:, :, 0:3] = forces [fx, fy, fz]
            - [:, :, 3:6] = torques [tx, ty, tz]
        """
        return self.xfrc_applied_t

    def create_quaternion_view(self, quat_slice: slice):
        """Create quaternion view with format conversion.

        Parameters
        ----------
        quat_slice : slice
            Slice for extracting quaternion from qpos

        Returns
        -------
        MjwQuaternionView
            Quaternion view with [w,x,y,z] -> [x,y,z,w] conversion
        """
        from holosoma.simulator.mujoco.mjw_views import MjwQuaternionView

        return MjwQuaternionView(qpos=self.qpos_t, quat_slice=quat_slice, num_envs=self.num_envs)

    def create_angular_velocity_view(self, ang_vel_slice: slice):
        """Create angular velocity view with proper reshaping.

        Parameters
        ----------
        ang_vel_slice : slice
            Slice for extracting angular velocity from qvel

        Returns
        -------
        MjwAngularVelocityView
            Angular velocity view with proper multi-env access
        """
        from holosoma.simulator.mujoco.mjw_views import MjwAngularVelocityView

        return MjwAngularVelocityView(qvel=self.qvel_t, ang_vel_slice=ang_vel_slice, num_envs=self.num_envs)

    def get_rigid_body_state_views(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get zero-copy views of rigid body states from GPU.

        Returns native PyTorch tensors that share memory with Warp arrays,
        eliminating the need for CPU↔GPU synchronization or tensor allocation
        during refresh_sim_tensors().

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            (positions, orientations, linear_vel, angular_vel):
            - positions: [num_envs, num_bodies, 3] - body positions
            - orientations: [num_envs, num_bodies, 4] - quaternions in [x,y,z,w] format
            - linear_vel: [num_envs, num_bodies, 3] - linear velocities
            - angular_vel: [num_envs, num_bodies, 3] - angular velocities
        """
        # Position: already in correct format
        positions = self.xpos_t  # [N, nbody, 3]

        # Orientation: convert MuJoCo [w,x,y,z] → holosoma [x,y,z,w]
        quat_mj = self.xquat_t  # [N, nbody, 4] - [w,x,y,z]
        orientations = quat_mj[..., [1, 2, 3, 0]]  # [x,y,z,w]

        # Velocities: split cvel [angular(3), linear(3)]
        angular_vel = self.cvel_t[..., 0:3]  # [N, nbody, 3]
        linear_vel = self.cvel_t[..., 3:6]  # [N, nbody, 3]

        return positions, orientations, linear_vel, angular_vel

    def set_root_state(self, env_ids: torch.Tensor, root_states: torch.Tensor, root_addrs: dict) -> None:
        """Set robot root states via direct GPU tensor writes.

        Writes root states directly to GPU tensors without CPU roundtrip.
        Supports batched updates for multiple environments efficiently.

        Parameters
        ----------
        env_ids : torch.Tensor
            Environment IDs to update [num_selected_envs]
        root_states : torch.Tensor
            Root states [num_selected_envs, 13] in holosoma format:
            [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
        root_addrs : dict
            Address dictionary with 'robot_qpos_addr' and 'robot_qvel_addr'
        """
        # Extract state components
        pos = root_states[:, :3]  # [N, 3]
        quat_holo = root_states[:, 3:7]  # [N, 4] [qx, qy, qz, qw]
        lin_vel = root_states[:, 7:10]  # [N, 3]
        ang_vel = root_states[:, 10:13]  # [N, 3]

        # Convert quaternion: holosoma [qx,qy,qz,qw] -> MuJoCo [qw,qx,qy,qz]
        quat_mj = quat_holo[:, [3, 0, 1, 2]]

        # Get addresses
        qpos_addr = root_addrs["robot_qpos_addr"]
        qvel_addr = root_addrs["robot_qvel_addr"]

        # Vectorized GPU tensor writes using explicit expand for shape matching
        N = len(env_ids)

        # Position (3 elements) - explicit expand to [N, 3]
        col_idx_pos = torch.arange(3, device=env_ids.device) + qpos_addr
        env_idx = env_ids.unsqueeze(1).expand(N, 3)  # [N, 3]
        col_idx = col_idx_pos.unsqueeze(0).expand(N, 3)  # [N, 3]
        self.qpos_t[env_idx, col_idx] = pos

        # Quaternion (4 elements) - explicit expand to [N, 4]
        col_idx_quat = torch.arange(4, device=env_ids.device) + qpos_addr + 3
        env_idx = env_ids.unsqueeze(1).expand(N, 4)  # [N, 4]
        col_idx = col_idx_quat.unsqueeze(0).expand(N, 4)  # [N, 4]
        self.qpos_t[env_idx, col_idx] = quat_mj

        # Linear velocity (3 elements) - explicit expand to [N, 3]
        col_idx_lin = torch.arange(3, device=env_ids.device) + qvel_addr
        env_idx = env_ids.unsqueeze(1).expand(N, 3)  # [N, 3]
        col_idx = col_idx_lin.unsqueeze(0).expand(N, 3)  # [N, 3]
        self.qvel_t[env_idx, col_idx] = lin_vel

        # Angular velocity (3 elements) - explicit expand to [N, 3]
        col_idx_ang = torch.arange(3, device=env_ids.device) + qvel_addr + 3
        env_idx = env_ids.unsqueeze(1).expand(N, 3)  # [N, 3]
        col_idx = col_idx_ang.unsqueeze(0).expand(N, 3)  # [N, 3]
        self.qvel_t[env_idx, col_idx] = ang_vel

        # No mj_forward call - next step() will handle forward kinematics

    def set_dof_state(self, env_ids: torch.Tensor, dof_states: torch.Tensor, dof_addrs: dict) -> None:
        """Set DOF states via direct GPU tensor writes.

        Writes DOF positions and velocities directly to GPU tensors
        without CPU roundtrip. Supports batched updates efficiently.

        Parameters
        ----------
        env_ids : torch.Tensor
            Environment IDs to update [num_selected_envs]
        dof_states : torch.Tensor
            DOF states [num_all_envs * num_dofs, 2] in IsaacGym format
            where [:, 0] = positions, [:, 1] = velocities
            NOTE: Contains states for ALL environments, we select based on env_ids
        dof_addrs : dict
            Address dictionary with 'dof_qpos_addrs' and 'dof_qvel_addrs' lists
        """
        # Parse addresses
        qpos_addrs = dof_addrs["dof_qpos_addrs"]
        qvel_addrs = dof_addrs["dof_qvel_addrs"]
        num_dof = len(qpos_addrs)

        # Vectorized selection: extract only rows for specified env_ids
        # dof_states is [num_all_envs * num_dof, 2], need [len(env_ids) * num_dof, 2]
        offsets = env_ids.unsqueeze(1) * num_dof  # [len(env_ids), 1]
        dof_offsets = torch.arange(num_dof, device=env_ids.device).unsqueeze(0)  # [1, num_dof]
        indices = (offsets + dof_offsets).flatten()  # [len(env_ids) * num_dof]

        selected_dof_states = dof_states[indices]  # [len(env_ids) * num_dof, 2]

        # Reshape from flattened IsaacGym format
        dof_pos = selected_dof_states[:, 0].view(len(env_ids), num_dof)  # [num_selected_envs, num_dof]
        dof_vel = selected_dof_states[:, 1].view(len(env_ids), num_dof)  # [num_selected_envs, num_dof]

        # Vectorized GPU tensor writes using explicit expand for shape matching
        N = len(env_ids)

        # Convert address lists to tensors
        qpos_indices = torch.tensor(qpos_addrs, device=env_ids.device)  # [num_dof]
        qvel_indices = torch.tensor(qvel_addrs, device=env_ids.device)  # [num_dof]

        # Explicit expand to [N, num_dof] for both indices
        env_idx = env_ids.unsqueeze(1).expand(N, num_dof)  # [N, num_dof]
        qpos_idx = qpos_indices.unsqueeze(0).expand(N, num_dof)  # [N, num_dof]
        qvel_idx = qvel_indices.unsqueeze(0).expand(N, num_dof)  # [N, num_dof]

        self.qpos_t[env_idx, qpos_idx] = dof_pos
        self.qvel_t[env_idx, qvel_idx] = dof_vel

        # No mj_forward call - next step() will handle forward kinematics
