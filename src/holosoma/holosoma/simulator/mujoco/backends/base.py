"""Abstract base interface for MuJoCo simulation backends.

This module defines the contract that all MuJoCo backends must implement,
providing a consistent interface for simulation control, state access, and
data synchronization.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import numpy as np
import torch

import mujoco

if TYPE_CHECKING:
    from holosoma.config_types.full_sim import FullSimConfig
    from holosoma.simulator.mujoco.tensor_views import BaseMujocoView


class IMujocoBackend(abc.ABC):
    """Abstract interface for MuJoCo simulation backends.

    Defines the contract that all MuJoCo backends must implement, providing
    a consistent interface for simulation control, state access, and data
    synchronization.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, config: FullSimConfig, device: str):
        """Initialize backend with model, data, and configuration.

        Parameters
        ----------
        model : mujoco.MjModel
            Compiled MuJoCo model
        data : mujoco.MjData
            MuJoCo data structure (shared with frontend)
        config : FullSimConfig
            Full simulation configuration
        device : str
            Device string ('cpu' or 'cuda:0', etc.)
        """
        self.model = model
        self.data = data
        self.config = config
        self.device = device
        self.num_envs = config.training.num_envs

    @abc.abstractmethod
    def step(self) -> None:
        """Advance simulation by one timestep.

        Implementations should call the appropriate MuJoCo step function
        (mj_step for CPU, mjw.step for GPU).
        """
        ...

    @abc.abstractmethod
    def refresh_sim_tensors(self, contact_history_tensor: torch.Tensor) -> None:
        """Update simulation tensors (contacts, rigid bodies, etc).

        Parameters
        ----------
        contact_history_tensor : torch.Tensor
            Contact force history buffer to update [num_envs, history_len, num_bodies, 3]
        """
        ...

    @abc.abstractmethod
    def get_render_data(self, world_id: int = 0) -> mujoco.MjData:
        """Get MjData for rendering (may require GPU->CPU sync).

        Parameters
        ----------
        world_id : int, default=0
            Which environment to sync for rendering (0 to num_envs-1).
            ClassicBackend ignores this (single environment only).
            WarpBackend syncs the specified environment from GPU to CPU.

        Returns
        -------
        mujoco.MjData
            MuJoCo data structure for rendering
        """
        ...

    @abc.abstractmethod
    def get_ctrl_tensor(self) -> torch.Tensor | None:
        """Get control tensor for direct writing (None if not supported).

        Returns
        -------
        torch.Tensor | None
            Control tensor for zero-copy writes, or None if backend doesn't support it
        """
        ...

    # View factory methods
    @abc.abstractmethod
    def create_root_view(self, addrs: dict) -> BaseMujocoView:
        """Create view for robot root states.

        Parameters
        ----------
        addrs : dict
            Dictionary with keys: pos_indices, quat_indices, vel_indices, ang_vel_indices

        Returns
        -------
        BaseMujocoView
            View for 13-element root state [pos, quat, lin_vel, ang_vel]
        """
        ...

    @abc.abstractmethod
    def create_dof_pos_view(self, indices: slice, num_dof: int) -> BaseMujocoView:
        """Create view for DOF positions.

        Parameters
        ----------
        indices : slice
            Slice into qpos array
        num_dof : int
            Number of degrees of freedom

        Returns
        -------
        BaseMujocoView
            View for DOF positions [num_envs, num_dof]
        """
        ...

    @abc.abstractmethod
    def create_dof_vel_view(self, indices: slice, num_dof: int) -> BaseMujocoView:
        """Create view for DOF velocities.

        Parameters
        ----------
        indices : slice
            Slice into qvel array
        num_dof : int
            Number of degrees of freedom

        Returns
        -------
        BaseMujocoView
            View for DOF velocities [num_envs, num_dof]
        """
        ...

    @abc.abstractmethod
    def create_dof_acc_view(self, indices: slice, num_dof: int) -> BaseMujocoView:
        """Create view for DOF accelerations.

        Parameters
        ----------
        indices : slice
            Slice into qacc array
        num_dof : int
            Number of degrees of freedom

        Returns
        -------
        BaseMujocoView
            View for DOF accelerations [num_envs, num_dof]
        """
        ...

    @abc.abstractmethod
    def create_dof_state_view(self, dof_addrs: dict, num_dof: int) -> BaseMujocoView:
        """Create view for DOF states in IsaacGym flattened format.

        Returns view with shape [num_envs * num_dof, 2] where:
        - [:, 0] = positions
        - [:, 1] = velocities

        Parameters
        ----------
        dof_addrs : dict
            Dictionary with 'dof_pos_indices' and 'dof_vel_indices' slices
        num_dof : int
            Number of degrees of freedom

        Returns
        -------
        BaseMujocoView
            View for DOF states [num_envs * num_dof, 2]
        """
        ...

    @abc.abstractmethod
    def create_force_view(self, num_bodies: int) -> torch.Tensor:
        """Create tensor for contact forces [num_envs, num_bodies, 3].

        Parameters
        ----------
        num_bodies : int
            Number of bodies in the model

        Returns
        -------
        torch.Tensor
            Contact force tensor
        """
        ...

    @abc.abstractmethod
    def get_applied_forces_view(self) -> np.ndarray | torch.Tensor:
        """Get writable view for external applied forces.

        Returns a writable view to xfrc_applied array where forces and torques
        can be applied to bodies. Shape is [num_bodies, 6] where:
        - [:, 0:3] = forces [fx, fy, fz]
        - [:, 3:6] = torques [tx, ty, tz]

        Returns
        -------
        np.ndarray | torch.Tensor
            Writable view to applied forces array
        """
        ...

    @abc.abstractmethod
    def set_root_state(self, env_ids: torch.Tensor, root_states: torch.Tensor, root_addrs: dict) -> None:
        """Set robot root states for specified environments.

        Parameters
        ----------
        env_ids : torch.Tensor
            Environment IDs to update
        root_states : torch.Tensor
            Root states [num_selected_envs, 13] in holosoma format:
            [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
        root_addrs : dict
            Address dictionary with 'robot_qpos_addr' and 'robot_qvel_addr'
        """
        ...

    @abc.abstractmethod
    def set_dof_state(self, env_ids: torch.Tensor, dof_states: torch.Tensor, dof_addrs: dict) -> None:
        """Set DOF states for specified environments.

        Parameters
        ----------
        env_ids : torch.Tensor
            Environment IDs to update
        dof_states : torch.Tensor
            DOF states [num_selected_envs * num_dofs, 2] in IsaacGym format
        dof_addrs : dict
            Address dictionary with 'dof_qpos_addrs' and 'dof_qvel_addrs'
        """
        ...

    def get_rigid_body_state_views(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Get zero-copy views of rigid body states (optional optimization).

        This method provides efficient access to rigid body states without
        CPU↔GPU synchronization or tensor allocation overhead. Backends should
        implement this for optimal performance during refresh_sim_tensors().

        Returns
        -------
        tuple[torch.Tensor, ...] | None
            If supported, returns (positions, orientations, linear_vel, angular_vel):
            - positions: [num_envs, num_bodies, 3]
            - orientations: [num_envs, num_bodies, 4] (xyzw quaternion)
            - linear_vel: [num_envs, num_bodies, 3]
            - angular_vel: [num_envs, num_bodies, 3]

            If not supported (e.g., ClassicBackend), returns None.
        """
        return None  # Default implementation - backends can override

    @abc.abstractmethod
    def create_quaternion_view(self, quat_slice: slice):
        """Create quaternion view with format conversion.

        Converts between MuJoCo [w,x,y,z] and holosoma [x,y,z,w] quaternion formats.

        Parameters
        ----------
        quat_slice : slice
            Slice for extracting quaternion from qpos

        Returns
        -------
        BaseMujocoView
            View for quaternion [num_envs, 4] with format conversion
        """
        ...

    @abc.abstractmethod
    def create_angular_velocity_view(self, ang_vel_slice: slice):
        """Create angular velocity view with proper reshaping.

        Parameters
        ----------
        ang_vel_slice : slice
            Slice for extracting angular velocity from qvel

        Returns
        -------
        BaseMujocoView
            View for angular velocity [num_envs, 3]
        """
        ...
