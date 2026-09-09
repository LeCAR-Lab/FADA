from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowState_go
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowState_hg

from ..base import BasicStateProcessor  # noqa: TID252


class UnitreeStateProcessor(BasicStateProcessor):
    """Unitree state processor implementation."""

    def _init_sdk_components(self):
        """Initialize Unitree SDK-specific components."""

        robot_type = self.config.robot_type

        if "g1" in robot_type or "h1-2" in robot_type:
            self.robot_lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_hg)
            self.robot_lowstate_subscriber.Init(self.low_state_handler_hg, 1)
        elif "h1" in robot_type or "go2" in robot_type:
            self.robot_lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_go)
            self.robot_lowstate_subscriber.Init(self.low_state_handler_go, 1)
        else:
            raise NotImplementedError(f"Robot type {robot_type} is not supported")

    def prepare_low_state(self, msg):
        """Prepare Unitree low-level state data from message."""
        if not msg:
            print("robot_low_state is None")
            return None

        imu_state = msg.imu_state
        self._extract_imu_data(imu_state)

        robot_joint_state = msg.motor_state
        self._extract_joint_data(robot_joint_state)

        self.robot_state_data = self._create_robot_state_data()
        return self.robot_state_data

    def _extract_imu_data(self, imu_state):
        """Extract IMU data from Unitree state message."""
        # base quaternion
        self.q[0:3] = 0.0  # base position (assumed to be at origin)
        self.q[3:7] = imu_state.quaternion  # w, x, y, z
        self.dq[3:6] = imu_state.gyroscope
        self.ddq[0:3] = imu_state.accelerometer

    def _extract_joint_data(self, robot_joint_state):
        """Extract joint data from Unitree state message."""
        for i in range(self.num_dof):
            motor_idx = self.robot.JOINT2MOTOR[i]
            self.q[7 + i] = robot_joint_state[motor_idx].q
            self.dq[6 + i] = robot_joint_state[motor_idx].dq
            self.tau_est[6 + i] = robot_joint_state[motor_idx].tau_est

    def low_state_handler_go(self, msg):
        """Handle Unitree GO low-level state messages."""
        self.robot_state_data = self.prepare_low_state(msg)

    def low_state_handler_hg(self, msg):
        """Handle Unitree HG low-level state messages."""
        self.robot_state_data = self.prepare_low_state(msg)
