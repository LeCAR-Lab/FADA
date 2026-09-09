"""Unitree robot interface with pybind and sdk2py backends."""

from __future__ import annotations

import os
import struct
import time
from types import SimpleNamespace

import numpy as np
from loguru import logger

from holosoma_inference.config.config_types import RobotConfig
from holosoma_inference.config.config_types.task import TaskConfig
from holosoma_inference.sdk.base.base_interface import BaseInterface


def _should_use_unitree_sdk2py_backend(interface_str: str | None, _domain_id: int) -> bool:
    """Policy-side mirror of ``UnitreeSdk2Bridge._should_use_sdk2py_backend``.

    ``_domain_id`` does not select the backend (see the comment below) but stays in
    the signature so the two predicates keep identical shapes; it is still honoured
    for DDS isolation by ``_init_sdk2py_backend``.
    """
    # Both spellings are honoured, in the same precedence order as the simulator
    # side (``holosoma/bridge/unitree/unitree_sdk2py_bridge.py``).  They must
    # agree: the backend is a property of the *pair* of processes, and forcing
    # only one half leaves the policy publishing on a channel the simulator is
    # not subscribed to.
    forced = str(
        os.getenv("HOLOSOMA_UNITREE_BRIDGE_BACKEND", os.getenv("HOLOSOMA_UNITREE_BACKEND", ""))
    ).strip().lower()
    if forced == "sdk2py":
        return True
    if forced == "pybind":
        return False
    iface = str(interface_str or "").strip().lower()
    # Loopback means the peer is a simulator on this host, never a real robot
    # (no Unitree robot is reachable over ``lo``), and the simulator side selects
    # its backend by exactly the same test.  ``domain_id`` is not part of the
    # condition.
    return iface in {"lo", "lo0"}


class UnitreeInterface(BaseInterface):
    """Interface for Unitree robots using pybind or sdk2py."""

    def __init__(
        self,
        robot_config: RobotConfig,
        domain_id=0,
        interface_str=None,
        use_joystick=True,
        task_config: TaskConfig | None = None,
    ):
        super().__init__(robot_config, domain_id, interface_str, use_joystick, task_config)
        self._unitree_motor_order = None
        self._kp_level = 1.0
        self._kd_level = 1.0
        self._backend = "pybind"
        self._sdk2py_low_state_msg = None
        self._sdk2py_low_state_tick = None
        self._sdk2py_low_cmd_sequence = 0
        self._sync_low_state_array = None
        self._sync_low_state_tick = None
        self._init_interface_backend()

    def _init_interface_backend(self):
        if _should_use_unitree_sdk2py_backend(self.interface_str, int(self.domain_id)):
            self._init_sdk2py_backend()
            return
        self._init_pybind_backend()

    def _init_pybind_backend(self):
        """Initialize the original C++/pybind11 backend."""
        try:
            import unitree_interface
        except ImportError as e:
            raise ImportError("unitree_interface python binding not found.") from e

        if int(self.domain_id) != 0:
            logger.warning(
                "Unitree interface requested domain_id={} on interface '{}', but the current "
                "unitree_interface binding does not expose DDS domain configuration. "
                "Isolation for G1/H1/GO2 must come from distinct NICs/subnets, separate hosts, "
                "or network namespaces rather than domain_id alone.",
                self.domain_id,
                self.interface_str,
            )

        robot_type_map = {
            "G1": unitree_interface.RobotType.G1,
            "H1": unitree_interface.RobotType.H1,
            "H1_2": unitree_interface.RobotType.H1_2,
            "GO2": unitree_interface.RobotType.GO2,
        }
        message_type_map = {"HG": unitree_interface.MessageType.HG, "GO2": unitree_interface.MessageType.GO2}

        self.unitree_interface = unitree_interface.create_robot(
            self.interface_str,
            robot_type_map[self.robot_config.robot.upper()],
            message_type_map[self.robot_config.message_type.upper()],
        )
        self.unitree_interface.set_control_mode(unitree_interface.ControlMode.PR)
        self._backend = "pybind"

        if self.robot_config.robot.lower() == "go2":
            self._unitree_motor_order = (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)

    def _init_sdk2py_backend(self):
        """Initialize the sdk2py backend that honors domain_id per process."""
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
                "Requested DDS-safe unitree sdk2py backend for loopback/domain-isolated Unitree session, "
                "but unitree_sdk2py is unavailable."
            ) from e

        message_type = str(self.robot_config.message_type).upper()
        if message_type == "GO2":
            self._sdk2py_lowcmd_default = unitree_go_msg_dds__LowCmd_
            self._sdk2py_lowstate_default = unitree_go_msg_dds__LowState_
            low_cmd_type = GoLowCmd
            low_state_type = GoLowState
            self._unitree_motor_order = (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)
        else:
            self._sdk2py_lowcmd_default = unitree_hg_msg_dds__LowCmd_
            self._sdk2py_lowstate_default = unitree_hg_msg_dds__LowState_
            low_cmd_type = HgLowCmd
            low_state_type = HgLowState

        # ChannelFactoryInitialize can fail under concurrent init from multiple
        # processes (DDS library limitation). Retry with jittered backoff.
        import random as _rnd
        import time as _time

        _max_retries = 10
        for _attempt in range(_max_retries):
            try:
                ChannelFactoryInitialize(int(self.domain_id), self.interface_str)
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
        self._sdk2py_lowcmd_publisher = ChannelPublisher("rt/lowcmd", low_cmd_type)
        self._sdk2py_lowcmd_publisher.Init()
        self._sdk2py_lowstate_subscriber = ChannelSubscriber("rt/lowstate", low_state_type)
        self._sdk2py_lowstate_subscriber.Init(self._sdk2py_low_state_handler, 10)
        self._backend = "sdk2py"
        logger.info(
            "Unitree interface using sdk2py backend for DDS-safe isolation "
            "(interface='{}', domain_id={})",
            self.interface_str,
            self.domain_id,
        )

    def _sdk2py_low_state_handler(self, msg):
        self._sdk2py_low_state_msg = msg
        self._sdk2py_low_state_tick = getattr(msg, "tick", None)

    def reset_low_state_cache(self) -> None:
        """Drop cached DDS low-state so the next read must come from a fresh publish."""
        if self._backend != "sdk2py":
            return
        self._sdk2py_low_state_msg = None
        self._sdk2py_low_state_tick = None
        self._sync_low_state_array = None
        self._sync_low_state_tick = None

    def set_sync_low_state_array(self, low_state_array, tick: int | None = None) -> None:
        """Install a lock-step sim low-state snapshot received on the sync ZMQ channel."""
        self._sync_low_state_array = np.asarray(low_state_array, dtype=np.float64).reshape(1, -1).copy()
        self._sync_low_state_tick = int(tick) if tick is not None else None

    def wait_for_low_state_tick_at_most(self, max_tick_ms: int, timeout: float = 2.0) -> bool:
        """Wait for a post-reset low-state publish whose sim-time tick is near zero."""
        if self._backend != "sdk2py":
            return True
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            tick = self._sdk2py_low_state_tick
            if self._sdk2py_low_state_msg is not None and tick is not None:
                try:
                    tick_i = int(tick)
                except (TypeError, ValueError):
                    tick_i = None
                if tick_i is not None and 0 <= tick_i <= int(max_tick_ms):
                    logger.debug("Accepted post-reset low-state tick={}ms", tick_i)
                    return True
            time.sleep(0.005)
        logger.warning(
            "Timed out waiting for post-reset low-state tick <= {}ms; last tick={}",
            max_tick_ms,
            self._sdk2py_low_state_tick,
        )
        return False

    def _state_to_array(self, quat, motor_pos, base_ang_vel, motor_vel) -> np.ndarray:
        base_pos = np.zeros(3)
        base_lin_vel = np.zeros(3)
        joint_pos = np.zeros(self.robot_config.num_joints)
        joint_vel = np.zeros(self.robot_config.num_joints)
        motor_order = self._unitree_motor_order or self.robot_config.joint2motor

        for j_id in range(self.robot_config.num_joints):
            m_id = motor_order[j_id]
            joint_pos[j_id] = float(motor_pos[m_id])
            joint_vel[j_id] = float(motor_vel[m_id])

        return np.concatenate([base_pos, quat, joint_pos, base_lin_vel, base_ang_vel, joint_vel]).reshape(1, -1)

    def get_low_state(self) -> np.ndarray:
        """Get robot state as numpy array."""
        if self._sync_low_state_array is not None:
            return self._sync_low_state_array.copy()
        if self._backend == "sdk2py":
            state = self._sdk2py_low_state_msg
            if state is None:
                return None
            quat = np.asarray(getattr(state.imu_state, "quaternion", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64)
            motor_pos = np.asarray([motor.q for motor in state.motor_state], dtype=np.float64)
            base_ang_vel = np.asarray(getattr(state.imu_state, "gyroscope", [0.0, 0.0, 0.0]), dtype=np.float64)
            motor_vel = np.asarray([motor.dq for motor in state.motor_state], dtype=np.float64)
            return self._state_to_array(quat, motor_pos, base_ang_vel, motor_vel)

        state = self.unitree_interface.read_low_state()
        quat = np.array(state.imu.quat)
        motor_pos = np.array(state.motor.q)
        base_ang_vel = np.array(state.imu.omega)
        motor_vel = np.array(state.motor.dq)
        return self._state_to_array(quat, motor_pos, base_ang_vel, motor_vel)

    def send_low_command(
        self,
        cmd_q: np.ndarray,
        cmd_dq: np.ndarray,
        cmd_tau: np.ndarray,
        dof_pos_latest: np.ndarray = None,
        kp_override: np.ndarray = None,
        kd_override: np.ndarray = None,
    ):
        """Send low-level command to robot."""
        if self._backend == "sdk2py":
            self._send_low_command_sdk2py(cmd_q, cmd_dq, cmd_tau, kp_override=kp_override, kd_override=kd_override)
            return

        cmd_q_target = np.zeros(self.robot_config.num_motors)
        cmd_dq_target = np.zeros(self.robot_config.num_motors)
        cmd_tau_target = np.zeros(self.robot_config.num_motors)
        cmd_kp = np.zeros(self.robot_config.num_motors) if kp_override is not None else None
        cmd_kd = np.zeros(self.robot_config.num_motors) if kd_override is not None else None

        motor_order = self._unitree_motor_order or self.robot_config.joint2motor
        for j_id in range(self.robot_config.num_joints):
            m_id = motor_order[j_id]
            cmd_q_target[m_id] = float(cmd_q[j_id])
            cmd_dq_target[m_id] = float(cmd_dq[j_id])
            cmd_tau_target[m_id] = float(cmd_tau[j_id])
            if cmd_kp is not None:
                cmd_kp[m_id] = float(kp_override[j_id])
            if cmd_kd is not None:
                cmd_kd[m_id] = float(kd_override[j_id])

        cmd = self.unitree_interface.create_zero_command()
        cmd.q_target = list(cmd_q_target)
        cmd.dq_target = list(cmd_dq_target)
        cmd.tau_ff = list(cmd_tau_target)

        motor_kp = np.array(cmd_kp if cmd_kp is not None else self.robot_config.motor_kp)
        motor_kd = np.array(cmd_kd if cmd_kd is not None else self.robot_config.motor_kd)
        cmd.kp = list(motor_kp * self._kp_level)
        cmd.kd = list(motor_kd * self._kd_level)

        self.unitree_interface.write_low_command(cmd)

    def _send_low_command_sdk2py(self, cmd_q, cmd_dq, cmd_tau, kp_override=None, kd_override=None):
        cmd = self._sdk2py_lowcmd_default()
        motor2joint = list(self.robot_config.motor2joint)
        default_q = np.asarray(self.robot_config.default_motor_angles, dtype=np.float64)
        motor_kp = np.asarray(self.robot_config.motor_kp, dtype=np.float64)
        motor_kd = np.asarray(self.robot_config.motor_kd, dtype=np.float64)

        if hasattr(cmd, "mode_pr"):
            cmd.mode_pr = 0
        if hasattr(cmd, "mode_machine"):
            cmd.mode_machine = 0

        for m_id in range(len(cmd.motor_cmd)):
            motor_cmd = cmd.motor_cmd[m_id]
            j_id = motor2joint[m_id] if m_id < len(motor2joint) else -1
            if j_id == -1:
                motor_cmd.mode = 0
                motor_cmd.q = float(default_q[m_id]) if m_id < len(default_q) else 0.0
                motor_cmd.dq = 0.0
                motor_cmd.tau = 0.0
                motor_cmd.kp = 0.0
                motor_cmd.kd = 0.0
                continue

            motor_cmd.mode = 1
            motor_cmd.q = float(cmd_q[j_id])
            motor_cmd.dq = float(cmd_dq[j_id])
            motor_cmd.tau = float(cmd_tau[j_id])
            kp_value = float(kp_override[j_id]) if kp_override is not None else float(motor_kp[m_id])
            kd_value = float(kd_override[j_id]) if kd_override is not None else float(motor_kd[m_id])
            motor_cmd.kp = kp_value * self._kp_level
            motor_cmd.kd = kd_value * self._kd_level

        if hasattr(cmd, "wireless_remote"):
            cmd.wireless_remote = [0 for _ in range(len(cmd.wireless_remote))]
        if hasattr(cmd, "reserve") and len(cmd.reserve) > 0:
            self._sdk2py_low_cmd_sequence = (int(self._sdk2py_low_cmd_sequence) % 255) + 1
            cmd.reserve[0] = int(self._sdk2py_low_cmd_sequence)
        if hasattr(cmd, "crc"):
            cmd.crc = self._sdk2py_crc.Crc(cmd)
        self._sdk2py_lowcmd_publisher.Write(cmd)

    def get_joystick_msg(self):
        """Get wireless controller message."""
        if self._backend == "sdk2py":
            if not self.use_joystick or self._sdk2py_low_state_msg is None:
                return None
            remote = list(getattr(self._sdk2py_low_state_msg, "wireless_remote", []))
            if len(remote) < 24:
                return None
            keys = int(remote[2]) | (int(remote[3]) << 8)
            lx = struct.unpack("f", bytes(remote[4:8]))[0]
            rx = struct.unpack("f", bytes(remote[8:12]))[0]
            ly = struct.unpack("f", bytes(remote[20:24]))[0]
            return SimpleNamespace(keys=keys, lx=lx, ly=ly, rx=rx)
        return self.unitree_interface.read_wireless_controller()

    def get_joystick_key(self, wc_msg=None):
        """Get current key from joystick message."""
        if wc_msg is None:
            wc_msg = self.get_joystick_msg()
        if wc_msg is None:
            return None
        return self._wc_key_map.get(getattr(wc_msg, "keys", 0), None)

    @property
    def kp_level(self):
        """Get proportional gain level."""
        return self._kp_level

    @kp_level.setter
    def kp_level(self, value):
        """Set proportional gain level."""
        self._kp_level = value

    @property
    def kd_level(self):
        """Get derivative gain level."""
        return self._kd_level

    @kd_level.setter
    def kd_level(self, value):
        """Set derivative gain level."""
        self._kd_level = value
