"""CPU-based single-environment MuJoCo backend implementation.

This backend wraps the standard MuJoCo CPU simulation with manual contact
force extraction. It maintains backward compatibility with existing single-
environment simulation code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from loguru import logger

import mujoco

from .base import IMujocoBackend

if TYPE_CHECKING:
    from holosoma.config_types.full_sim import FullSimConfig
    from holosoma.simulator.mujoco.tensor_views import BaseMujocoView


class ClassicBackend(IMujocoBackend):
    """CPU-based single-environment MuJoCo backend.

    This backend wraps the standard MuJoCo CPU simulation with manual contact
    force extraction. It maintains backward compatibility with existing single-
    environment simulation code.

    Key characteristics:
    - Single environment only (num_envs must be 1)
    - CPU-based computation
    - Manual contact force extraction via mj_contactForce
    - Numpy arrays with PyTorch tensor conversion
    - Compatible with existing tensor_views.py proxy system
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, config: FullSimConfig, device: str):
        """Initialize ClassicBackend with single-environment validation.

        Parameters
        ----------
        model : mujoco.MjModel
            Compiled MuJoCo model
        data : mujoco.MjData
            MuJoCo data structure (shared with frontend)
        config : FullSimConfig
            Full simulation configuration
        device : str
            Device string (typically 'cpu')

        Raises
        ------
        ValueError
            If num_envs > 1 (only single environment supported)
        """
        super().__init__(model, data, config, device)

        if self.num_envs > 1:
            raise ValueError(
                f"ClassicBackend only supports single environment, got {self.num_envs}. "
                f"Use WarpBackend (use_warp=True) for multi-environment simulation."
            )

        # Pre-allocate contact force tensor
        self._force_tensor = torch.zeros(1, model.nbody, 3, device=device)

        logger.info(f"ClassicBackend initialized: {model.nbody} bodies, device={device}")

    def step(self) -> None:
        """Advance simulation by one timestep using mj_step."""
        mujoco.mj_step(self.model, self.data)

    def get_render_data(self, world_id: int = 0) -> mujoco.MjData:
        """Return data for rendering (already on CPU).

        Parameters
        ----------
        world_id : int, default=0
            Ignored for ClassicBackend (single environment only)

        Returns
        -------
        mujoco.MjData
            The backend's data structure (no copy needed)
        """
        return self.data

    def get_ctrl_tensor(self) -> None:
        """Classic backend doesn't support direct tensor writes.

        Returns
        -------
        None
            Indicates that torque application must use the loop-based method
        """
        return

    def refresh_sim_tensors(self, contact_history_tensor: torch.Tensor) -> None:
        """Update contact forces using manual extraction.

        Extracts contact forces from MuJoCo's contact system using mj_contactForce,
        accumulates them per body, and updates the contact history.

        Parameters
        ----------
        contact_history_tensor : torch.Tensor
            Contact force history buffer [num_envs, history_len, num_bodies, 3]
        """
        # Reset force accumulator
        self._force_tensor.fill_(0.0)

        # Pre-allocate force/torque buffer for mj_contactForce
        forcetorque = np.zeros(6, dtype=np.float64)

        # Extract and accumulate contact forces
        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            # Get 6D force/torque vector
            mujoco.mj_contactForce(self.model, self.data, i, forcetorque)

            # Convert to torch tensor (forces only, ignore torques)
            force = torch.from_numpy(forcetorque[:3]).float().to(self.device)

            # Map geoms to bodies
            b1 = self.model.geom_bodyid[contact.geom1]
            b2 = self.model.geom_bodyid[contact.geom2]

            # Apply Newton's 3rd law: body1 gets -force, body2 gets +force
            if b1 < self.model.nbody:
                self._force_tensor[0, b1] -= force
            if b2 < self.model.nbody:
                self._force_tensor[0, b2] += force

        # Update history: shift old values right, add current at position 0
        contact_history_tensor[:] = torch.cat(
            [self._force_tensor.clone().unsqueeze(1), contact_history_tensor[:, :-1]], dim=1
        )

    def create_root_view(self, addrs: dict) -> BaseMujocoView:
        """Create root state view using existing tensor_views.

        Parameters
        ----------
        addrs : dict
            Address dictionary with slices for pos, quat, vel, ang_vel

        Returns
        -------
        BaseMujocoView
            MujocoRootStateView with quaternion conversion
        """
        from holosoma.simulator.mujoco.tensor_views import MujocoRootStateView

        return MujocoRootStateView(
            qpos_array=self.data.qpos,
            qvel_array=self.data.qvel,
            pos_indices=addrs["pos_indices"],
            quat_indices=addrs["quat_indices"],
            vel_indices=addrs["vel_indices"],
            ang_vel_indices=addrs["ang_vel_indices"],
            num_envs=1,
            device=self.device,
        )

    def create_dof_pos_view(self, indices: slice, num_dof: int) -> BaseMujocoView:
        """Create DOF position view.

        Parameters
        ----------
        indices : slice
            Slice into qpos array
        num_dof : int
            Number of degrees of freedom

        Returns
        -------
        BaseMujocoView
            View for DOF positions [1, num_dof]
        """
        from holosoma.simulator.mujoco.tensor_views import create_dof_position_view

        return create_dof_position_view(self.data.qpos, indices, 1, num_dof, self.device)

    def create_dof_vel_view(self, indices: slice, num_dof: int) -> BaseMujocoView:
        """Create DOF velocity view.

        Parameters
        ----------
        indices : slice
            Slice into qvel array
        num_dof : int
            Number of degrees of freedom

        Returns
        -------
        BaseMujocoView
            View for DOF velocities [1, num_dof]
        """
        from holosoma.simulator.mujoco.tensor_views import create_dof_velocity_view

        return create_dof_velocity_view(self.data.qvel, indices, 1, num_dof, self.device)

    def create_dof_acc_view(self, indices: slice, num_dof: int) -> BaseMujocoView:
        """Create DOF acceleration view.

        Parameters
        ----------
        indices : slice
            Slice into qacc array
        num_dof : int
            Number of degrees of freedom

        Returns
        -------
        BaseMujocoView
            View for DOF accelerations [1, num_dof]
        """
        from holosoma.simulator.mujoco.tensor_views import create_dof_acceleration_view

        return create_dof_acceleration_view(self.data.qacc, indices, 1, num_dof, self.device)

    def create_force_view(self, num_bodies: int) -> torch.Tensor:
        """Return pre-allocated contact force tensor.

        Parameters
        ----------
        num_bodies : int
            Number of bodies (for validation)

        Returns
        -------
        torch.Tensor
            Contact force tensor [1, num_bodies, 3]
        """
        return self._force_tensor

    def create_dof_state_view(self, dof_addrs: dict, num_dof: int) -> BaseMujocoView:
        """Create DOF state view using CPU numpy arrays.

        Parameters
        ----------
        dof_addrs : dict
            Dictionary with 'dof_pos_indices' and 'dof_vel_indices' slices
        num_dof : int
            Number of degrees of freedom

        Returns
        -------
        BaseMujocoView
            MujocoDofStateView with IsaacGym flattened format [1 * num_dof, 2]
        """
        from holosoma.simulator.mujoco.tensor_views import MujocoDofStateView

        return MujocoDofStateView(
            qpos_array=self.data.qpos,
            qvel_array=self.data.qvel,
            dof_pos_indices=dof_addrs["dof_pos_indices"],
            dof_vel_indices=dof_addrs["dof_vel_indices"],
            num_envs=1,
            num_dof=num_dof,
            device=self.device,
        )

    def get_applied_forces_view(self) -> np.ndarray:
        """Get writable view for external applied forces.

        Returns direct view of MuJoCo's xfrc_applied array for applying
        external forces and torques to bodies.

        Returns
        -------
        np.ndarray
            Writable numpy array view [num_bodies, 6]
        """
        return self.data.xfrc_applied

    def create_quaternion_view(self, quat_slice: slice) -> BaseMujocoView:
        """Create quaternion view with format conversion.

        Delegates to tensor_views factory function for CPU numpy array views.

        Parameters
        ----------
        quat_slice : slice
            Slice for extracting quaternion from qpos

        Returns
        -------
        BaseMujocoView
            View for quaternion [1, 4] with [w,x,y,z] -> [x,y,z,w] conversion
        """
        from holosoma.simulator.mujoco.tensor_views import create_quaternion_view

        return create_quaternion_view(qpos_array=self.data.qpos, indices=quat_slice, num_envs=1, device=self.device)

    def create_angular_velocity_view(self, ang_vel_slice: slice) -> BaseMujocoView:
        """Create angular velocity view with proper reshaping.

        Delegates to tensor_views factory function for CPU numpy array views.

        Parameters
        ----------
        ang_vel_slice : slice
            Slice for extracting angular velocity from qvel

        Returns
        -------
        BaseMujocoView
            View for angular velocity [1, 3]
        """
        from holosoma.simulator.mujoco.tensor_views import (
            create_base_angular_velocity_view,
        )

        return create_base_angular_velocity_view(
            qvel_array=self.data.qvel, indices=ang_vel_slice, num_envs=1, device=self.device
        )

    def set_root_state(self, env_ids: torch.Tensor, root_states: torch.Tensor, root_addrs: dict) -> None:
        """Set robot root states using CPU numpy arrays.

        Converts tensors to numpy, writes to MuJoCo data arrays,
        and calls mj_forward to update derived quantities.

        Parameters
        ----------
        env_ids : torch.Tensor
            Environment IDs to update (must be single environment)
        root_states : torch.Tensor
            Root states [num_selected_envs, 13] in holosoma format:
            [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
        root_addrs : dict
            Address dictionary with 'robot_qpos_addr' and 'robot_qvel_addr'

        Raises
        ------
        AssertionError
            If multiple environments specified
        """
        # ClassicBackend only supports single environment
        assert len(env_ids) <= 1, f"ClassicBackend only supports single environment, got {len(env_ids)}"

        if len(env_ids) == 0:
            return

        # Extract state components (convert to numpy)
        state = root_states[0]
        pos = state[:3].detach().cpu().numpy()
        quat_holo = state[3:7].detach().cpu().numpy()  # [qx, qy, qz, qw]
        lin_vel = state[7:10].detach().cpu().numpy()
        ang_vel = state[10:13].detach().cpu().numpy()

        # Convert quaternion: holosoma [qx,qy,qz,qw] -> MuJoCo [qw,qx,qy,qz]
        quat_mj = np.array([quat_holo[3], quat_holo[0], quat_holo[1], quat_holo[2]])

        # Get addresses
        qpos_addr = root_addrs["robot_qpos_addr"]
        qvel_addr = root_addrs["robot_qvel_addr"]

        # Write to MuJoCo data arrays
        self.data.qpos[qpos_addr : qpos_addr + 3] = pos
        self.data.qpos[qpos_addr + 3 : qpos_addr + 7] = quat_mj
        self.data.qvel[qvel_addr : qvel_addr + 3] = lin_vel
        self.data.qvel[qvel_addr + 3 : qvel_addr + 6] = ang_vel

        # Update derived quantities
        mujoco.mj_forward(self.model, self.data)

    def set_dof_state(self, env_ids: torch.Tensor, dof_states: torch.Tensor, dof_addrs: dict) -> None:
        """Set DOF states using CPU numpy arrays.

        Converts tensors to numpy, writes to MuJoCo data arrays,
        and calls mj_forward to update derived quantities.

        Parameters
        ----------
        env_ids : torch.Tensor
            Environment IDs to update (must be single environment)
        dof_states : torch.Tensor
            DOF states [num_selected_envs * num_dofs, 2] in IsaacGym format
            where [:, 0] = positions, [:, 1] = velocities
        dof_addrs : dict
            Address dictionary with 'dof_qpos_addrs' and 'dof_qvel_addrs' lists

        Raises
        ------
        AssertionError
            If multiple environments specified
        """
        # ClassicBackend only supports single environment
        assert len(env_ids) <= 1, f"ClassicBackend only supports single environment, got {len(env_ids)}"

        if len(env_ids) == 0:
            return

        # Parse addresses
        qpos_addrs = dof_addrs["dof_qpos_addrs"]
        qvel_addrs = dof_addrs["dof_qvel_addrs"]
        num_dof = len(qpos_addrs)

        # Reshape and convert to numpy
        dof_pos = dof_states[:, 0].view(len(env_ids), num_dof)[0].detach().cpu().numpy()
        dof_vel = dof_states[:, 1].view(len(env_ids), num_dof)[0].detach().cpu().numpy()

        # Write to MuJoCo data arrays
        for i, (qpos_idx, qvel_idx) in enumerate(zip(qpos_addrs, qvel_addrs)):
            self.data.qpos[qpos_idx] = dof_pos[i]
            self.data.qvel[qvel_idx] = dof_vel[i]

        # Update derived quantities
        mujoco.mj_forward(self.model, self.data)
