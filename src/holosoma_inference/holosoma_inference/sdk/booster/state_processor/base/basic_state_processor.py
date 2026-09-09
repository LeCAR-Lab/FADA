from abc import ABC, abstractmethod

import numpy as np

from holosoma_inference.config.config_types.robot import RobotConfig


class BasicStateProcessor(ABC):
    """Abstract base class for state processor implementations."""

    def __init__(self, config: RobotConfig, lcm=None):
        self.lcm = lcm
        self.config = config
        self.num_motor = self.config.num_motors
        self.sdk_type = self.config.sdk_type
        self.motor_type = self.config.motor_type

        # Initialize state arrays
        self.num_dof = self.config.num_joints
        # 3 + 4 + num_dof (base_pos + base_quat + joint_pos)
        self._init_q = np.zeros(3 + 4 + self.num_dof)
        self.q = self._init_q
        self.dq = np.zeros(3 + 3 + self.num_dof)  # base_lin_vel + base_ang_vel + joint_vel
        self.ddq = np.zeros(3 + 3 + self.num_dof)  # base_lin_acc + base_ang_acc + joint_acc
        self.tau_est = np.zeros(3 + 3 + self.num_dof)  # base_lin_force + base_ang_torque + joint_torque
        self.temp_first = np.zeros(self.num_dof)
        self.temp_second = np.zeros(self.num_dof)
        self.robot_state_data = None

        # Initialize SDK-specific components
        self._init_sdk_components()

    @abstractmethod
    def _init_sdk_components(self):
        """Initialize SDK-specific components. Must be implemented by subclasses."""

    @abstractmethod
    def prepare_low_state(self, msg):
        """Prepare low-level state data from message. Must be implemented by subclasses."""

    def get_robot_state_data(self):
        """Get the current robot state data."""
        return self.robot_state_data

    @abstractmethod
    def _extract_imu_data(self, imu_state):
        """Extract IMU data from state message. Must be implemented by subclasses."""

    @abstractmethod
    def _extract_joint_data(self, robot_joint_state):
        """Extract joint data from state message. Must be implemented by subclasses."""

    def _create_robot_state_data(self):
        """Create the final robot state data array."""
        return np.array(
            self.q.tolist() + self.dq.tolist() + self.tau_est.tolist() + self.ddq.tolist(), dtype=np.float64
        ).reshape(1, -1)
