from booster_robotics_sdk import B1LowStateSubscriber

from holosoma_inference.utils.math.quat import rpy_to_quat

from ..base import BasicStateProcessor  # noqa: TID252


class BoosterStateProcessor(BasicStateProcessor):
    """Booster state processor implementation."""

    def _init_sdk_components(self):
        """Initialize Booster SDK-specific components."""

        robot_type = self.config.robot_type

        if robot_type in {"t1_23dof", "t1_29dof"}:
            self.robot_lowstate_subscriber = B1LowStateSubscriber(self.low_state_handler_b1)
            self.robot_lowstate_subscriber.InitChannel()
        else:
            raise NotImplementedError(f"Robot type {robot_type} is not supported")

    def prepare_low_state(self, msg):
        """Prepare Booster low-level state data from message."""
        if not msg:
            print("robot_low_state is None")
            return None

        imu_state = msg.imu_state
        self._extract_imu_data(imu_state)

        # Choose the appropriate motor state based on motor type
        if self.motor_type == "serial":
            robot_joint_state = msg.motor_state_serial
        elif self.motor_type == "parallel":
            robot_joint_state = msg.motor_state_parallel
        else:
            raise ValueError(f"Invalid motor type '{self.motor_type}'. Expected 'serial' or 'parallel'.")

        self._extract_joint_data(robot_joint_state)

        self.robot_state_data = self._create_robot_state_data()
        return self.robot_state_data

    def _extract_imu_data(self, imu_state):
        """Extract IMU data from Booster state message."""

        # base quaternion
        self.q[0:3] = 0.0  # base position (assumed to be at origin)
        rpy = imu_state.rpy
        self.q[3:7] = rpy_to_quat(rpy)
        self.dq[3:6] = imu_state.gyro
        self.ddq[0:3] = imu_state.acc

    def _extract_joint_data(self, robot_joint_state):
        """Extract joint data from Booster state message."""
        for i in range(self.num_dof):
            motor_idx = self.config.joint2motor[i]
            self.q[7 + i] = robot_joint_state[motor_idx].q
            self.dq[6 + i] = robot_joint_state[motor_idx].dq
            self.tau_est[6 + i] = robot_joint_state[motor_idx].tau_est

    def low_state_handler_b1(self, msg):
        """Handle Booster B1 low-level state messages."""
        self.robot_state_data = self.prepare_low_state(msg)
