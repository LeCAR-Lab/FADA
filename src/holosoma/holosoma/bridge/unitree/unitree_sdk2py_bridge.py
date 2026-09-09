from __future__ import annotations

import os

from loguru import logger
from unitree_interface import (
    LowState,
    MessageType,
    MotorCommand,
    RobotType,
    UnitreeInterface,
    WirelessController,
)

from holosoma.bridge.base.basic_sdk2py_bridge import BasicSdk2Bridge


class UnitreeSdk2Bridge(BasicSdk2Bridge):
    """Unitree SDK bridge implementation using unitree_interface C++ bindings."""

    SUPPORTED_ROBOT_TYPES = {"g1_29dof", "h1", "h1-2", "go2_12dof"}

    @staticmethod
    def _should_use_sdk2py_backend(interface_name: str | None, _domain_id: int) -> bool:
        """Choose the DDS backend.  ``_domain_id`` does not select it, but stays in
        the signature to mirror the policy-side predicate in
        ``holosoma_inference/sdk/unitree/unitree_interface.py``; the two must agree
        and are asserted equivalent by
        ``src/holosoma/tests/bridge/test_unitree_lockstep_ack.py``.  It is still
        honoured for DDS isolation, by ``_init_sdk2py_components`` below.
        """
        forced = str(os.getenv("HOLOSOMA_UNITREE_BRIDGE_BACKEND", os.getenv("HOLOSOMA_UNITREE_BACKEND", ""))).strip().lower()
        if forced == "sdk2py":
            return True
        if forced == "pybind":
            return False
        iface = str(interface_name or "").strip().lower()
        # Loopback means this bridge is the *simulated* robot talking to a policy
        # process on the same host: a real Unitree robot is never reachable over
        # ``lo``.  On that path always take the sdk2py backend, because it is the
        # only Unitree backend that can carry the lock-step lowcmd ack -- the
        # pybind ``MotorCommand`` from ``create_robot()`` exposes only
        # q_target/dq_target/tau_ff/kp/kd and has no reserve/user-data field for
        # the policy's sequence number, so ``_sdk2py_low_cmd_seq`` can never be
        # observed there and lock-step aborts on its first ack check.  The choice is
        # made on the interface alone: any real NIC takes the pybind backend.
        return iface in {"lo", "lo0"}

    def _init_sdk_components(self):
        """Initialize Unitree SDK-specific components."""

        robot_type = self.robot.asset.robot_type

        # Validate robot type first
        if robot_type not in self.SUPPORTED_ROBOT_TYPES:
            raise ValueError(f"Invalid robot type '{robot_type}'. Unitree SDK supports: {self.SUPPORTED_ROBOT_TYPES}")

        # Map robot type to SDK enum
        robot_type_map = {
            "g1_29dof": RobotType.G1,
            "h1": RobotType.H1,
            "h1-2": RobotType.H1_2,
            "go2_12dof": RobotType.GO2,
        }

        # Map to message type (HG for humanoid robots with 35 motors, GO2 for others)
        message_type_map = {
            "g1_29dof": MessageType.HG,
            "h1": MessageType.GO2,
            "h1-2": MessageType.HG,
            "go2_12dof": MessageType.GO2,
        }

        sdk_robot_type = robot_type_map[robot_type]
        sdk_message_type = message_type_map[robot_type]

        # Get network interface from config
        interface_name = self.bridge_config.interface or "eth0"
        domain_id = int(getattr(self.bridge_config, "domain_id", 0) or 0)

        if self._should_use_sdk2py_backend(interface_name, domain_id):
            self._init_sdk2py_components(robot_type, interface_name, domain_id)
            return

        if domain_id != 0:
            logger.warning(
                "Unitree SDK bridge requested domain_id={} on interface '{}', but the current "
                "unitree_interface binding does not expose DDS domain configuration. "
                "Do not rely on domain_id alone to isolate concurrent G1/H1/GO2 sessions.",
                domain_id,
                interface_name,
            )
        logger.info(
            "Unitree SDK bridge using interface='{}' (requested domain_id={})",
            interface_name,
            domain_id,
        )

        # Create interface (handles DDS initialization internally)
        self.interface = UnitreeInterface(interface_name, sdk_robot_type, sdk_message_type)

        # Initialize data structures
        self.low_state = LowState(self.num_motor)
        self.low_cmd = MotorCommand(self.num_motor)
        self.wireless_controller = WirelessController()
        self._backend = "pybind"

    def _init_sdk2py_components(self, robot_type: str, interface_name: str, domain_id: int) -> None:
        try:
            from unitree_sdk2py.core.channel import (
                ChannelFactoryInitialize,
                ChannelPublisher,
                ChannelSubscriber,
            )
            from unitree_sdk2py.idl.default import (
                unitree_go_msg_dds__LowCmd_,
                unitree_go_msg_dds__LowState_,
                unitree_hg_msg_dds__LowCmd_,
                unitree_hg_msg_dds__LowState_,
            )
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as GoLowCmd
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as GoLowState
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as HgLowCmd
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as HgLowState
            from unitree_sdk2py.utils.crc import CRC
        except ImportError as e:
            raise RuntimeError(
                "Requested DDS-safe unitree sdk2py bridge backend for loopback/domain-isolated Unitree session, "
                "but unitree_sdk2py is unavailable."
            ) from e

        if robot_type in {"g1_29dof", "h1-2"}:
            self._sdk2py_lowcmd_default = unitree_hg_msg_dds__LowCmd_
            self._sdk2py_lowstate_default = unitree_hg_msg_dds__LowState_
            low_cmd_type = HgLowCmd
            low_state_type = HgLowState
        elif robot_type in {"h1", "go2_12dof"}:
            self._sdk2py_lowcmd_default = unitree_go_msg_dds__LowCmd_
            self._sdk2py_lowstate_default = unitree_go_msg_dds__LowState_
            low_cmd_type = GoLowCmd
            low_state_type = GoLowState
        else:
            raise ValueError(f"Unsupported Unitree robot type for sdk2py backend: {robot_type}")

        # ChannelFactoryInitialize can fail under concurrent init from multiple
        # processes (DDS library limitation). Retry with jittered backoff.
        import random as _rnd
        import time as _time

        _max_retries = 10
        for _attempt in range(_max_retries):
            try:
                ChannelFactoryInitialize(domain_id, interface_name)
                break
            except Exception:
                if _attempt + 1 >= _max_retries:
                    raise
                _delay = min(2.0 ** _attempt * 0.5, 10.0) + _rnd.uniform(0, 1.0)
                logger.warning(
                    "ChannelFactoryInitialize failed (attempt {}/{}), retrying in {:.1f}s ...",
                    _attempt + 1,
                    _max_retries,
                    _delay,
                )
                _time.sleep(_delay)
        self._sdk2py_crc = CRC()
        self.low_state = self._sdk2py_lowstate_default()
        self.low_cmd = self._sdk2py_lowcmd_default()
        self._sdk2py_low_cmd_seq = 0
        self._sdk2py_low_state_publisher = ChannelPublisher("rt/lowstate", low_state_type)
        self._sdk2py_low_state_publisher.Init()
        self._sdk2py_low_cmd_subscriber = ChannelSubscriber("rt/lowcmd", low_cmd_type)
        self._sdk2py_low_cmd_subscriber.Init(self._sdk2py_low_cmd_handler, 10)
        self._backend = "sdk2py"
        logger.info(
            "Unitree SDK bridge using sdk2py backend for DDS-safe isolation "
            "(interface='{}', domain_id={})",
            interface_name,
            domain_id,
        )

    def _sdk2py_low_cmd_handler(self, msg=None):
        self.low_cmd = msg
        try:
            reserve = getattr(msg, "reserve", None)
            if reserve is not None and len(reserve) > 0:
                self._sdk2py_low_cmd_seq = int(reserve[0])
        except Exception:
            pass

    def low_cmd_handler(self, msg=None):
        """Handle Unitree low-level command messages."""
        if getattr(self, "_backend", "pybind") == "sdk2py":
            return
        # Poll for incoming commands from DDS
        self.low_cmd = self.interface.read_incoming_command()

    def publish_low_state(self):
        """Publish Unitree low-level state using simulator-agnostic interface."""

        # Get simulator data
        positions, velocities, accelerations = self._get_dof_states()
        actuator_forces = self._get_actuator_forces()
        quaternion, gyro, acceleration = self._get_base_imu_data()

        # Publish truth pose if configured
        if self._truth_pose_puber is not None:
            pos, quat_wxyz = self._get_base_pose_data()
            self.publish_truth_pose(self.sim_time, pos, quat_wxyz)

        quat_array = quaternion.detach().cpu().numpy()

        if getattr(self, "_backend", "pybind") == "sdk2py":
            joint2motor = list(getattr(self.robot, "joint2motor", range(len(positions))))
            for j_id, m_id in enumerate(joint2motor):
                self.low_state.motor_state[m_id].q = float(positions[j_id])
                self.low_state.motor_state[m_id].dq = float(velocities[j_id])
                self.low_state.motor_state[m_id].ddq = float(accelerations[j_id])
                self.low_state.motor_state[m_id].tau_est = float(actuator_forces[j_id])

            self.low_state.imu_state.quaternion = [
                float(quat_array[0]),
                float(quat_array[1]),
                float(quat_array[2]),
                float(quat_array[3]),
            ]
            self.low_state.imu_state.gyroscope = gyro.detach().cpu().numpy().tolist()
            self.low_state.imu_state.accelerometer = acceleration.detach().cpu().numpy().tolist()
            self.low_state.tick = int(self.sim_time * 1e3)
            if hasattr(self.low_state, "crc"):
                self.low_state.crc = self._sdk2py_crc.Crc(self.low_state)
            self._sdk2py_low_state_publisher.Write(self.low_state)
            return

        # Populate motor state
        self.low_state.motor.q = positions.tolist()
        self.low_state.motor.dq = velocities.tolist()
        self.low_state.motor.ddq = accelerations.tolist()
        self.low_state.motor.tau_est = actuator_forces.tolist()

        # Populate IMU state
        self.low_state.imu.quat = [
            float(quat_array[0]),
            float(quat_array[1]),
            float(quat_array[2]),
            float(quat_array[3]),
        ]
        self.low_state.imu.omega = gyro.detach().cpu().numpy().tolist()
        self.low_state.imu.accel = acceleration.detach().cpu().numpy().tolist()
        self.low_state.tick = int(self.sim_time * 1e3)
        self.interface.publish_low_state(self.low_state)

    def publish_wireless_controller(self):
        """Publish wireless controller data using unitree_interface."""
        if getattr(self, "_backend", "pybind") == "sdk2py":
            return
        # Call base class to populate wireless_controller from joystick
        super().publish_wireless_controller()

        # Publish using C++ interface
        if self.joystick is not None:
            self.interface.publish_wireless_controller(self.wireless_controller)

    def compute_torques(self):
        """Compute torques using Unitree's unified command structure."""
        if not (hasattr(self, "low_cmd") and self.low_cmd):
            return self.torques

        try:
            if getattr(self, "_backend", "pybind") == "sdk2py":
                joint2motor = list(getattr(self.robot, "joint2motor", range(self.num_motor)))
                tau_ff = [float(self.low_cmd.motor_cmd[m_id].tau) for m_id in joint2motor]
                kp = [float(self.low_cmd.motor_cmd[m_id].kp) for m_id in joint2motor]
                kd = [float(self.low_cmd.motor_cmd[m_id].kd) for m_id in joint2motor]
                q_target = [float(self.low_cmd.motor_cmd[m_id].q) for m_id in joint2motor]
                dq_target = [float(self.low_cmd.motor_cmd[m_id].dq) for m_id in joint2motor]
                return self._compute_pd_torques(
                    tau_ff=tau_ff,
                    kp=kp,
                    kd=kd,
                    q_target=q_target,
                    dq_target=dq_target,
                )

            # Extract from Unitree's unified structure
            return self._compute_pd_torques(
                tau_ff=self.low_cmd.tau_ff,
                kp=self.low_cmd.kp,
                kd=self.low_cmd.kd,
                q_target=self.low_cmd.q_target,
                dq_target=self.low_cmd.dq_target,
            )
        except Exception as e:
            logger.error(f"Error computing torques: {e}")
            raise
