"""Base interface for robot control."""

from abc import ABC, abstractmethod

import numpy as np

from holosoma_inference.config.config_types import RobotConfig


class BaseInterface(ABC):
    """
    Abstract base class for robot control interfaces.
    """

    def __init__(self, robot_config: RobotConfig, domain_id=0, interface_str=None, use_joystick=True):
        self.robot_config = robot_config
        self.domain_id = domain_id
        self.interface_str = interface_str
        self.use_joystick = use_joystick
        # Initialize key state tracking for joystick
        self._key_states: dict[str, bool] = {}
        self._last_key_states: dict[str, bool] = {}
        self._wc_key_map = self._default_wc_key_map()

    @abstractmethod
    def get_low_state(self) -> np.ndarray:
        """
        Get robot state as numpy array.

        Returns:
            np.ndarray with shape (1, 3+4+N+3+3+N) containing:
            [base_pos(3), quat(4), joint_pos(N), lin_vel(3), ang_vel(3), joint_vel(N)]
        """
        raise NotImplementedError

    @abstractmethod
    def send_low_command(
        self,
        cmd_q: np.ndarray,
        cmd_dq: np.ndarray,
        cmd_tau: np.ndarray,
        dof_pos_latest: np.ndarray = None,
        kp_override: np.ndarray = None,
        kd_override: np.ndarray = None,
    ):
        """
        Send low-level command to robot.

        Args:
            cmd_q: target joint positions (N,)
            cmd_dq: target joint velocities (N,)
            cmd_tau: feedforward torques (N,)
            dof_pos_latest: latest joint positions (N,)
            kp_override: optional KP gains override (N,)
            kd_override: optional KD gains override (N,)
        """
        raise NotImplementedError

    def update_config(self, robot_config: RobotConfig):
        """
        Update the robot configuration and propagate to internal components.

        Override in subclasses that need to update internal SDK components
        when the config changes (e.g., after loading KP/KD from ONNX metadata).

        Args:
            robot_config: The new robot configuration.
        """
        self.robot_config = robot_config

    @abstractmethod
    def get_joystick_msg(self):
        raise NotImplementedError

    @abstractmethod
    def get_joystick_key(self, wc_msg=None):
        raise NotImplementedError

    def process_joystick_input(self, lin_vel_command, ang_vel_command, stand_command, upper_body_motion_active):
        """
        Process joystick input and update commands in a unified way.

        Args:
            lin_vel_command: np.ndarray, shape (1, 2)
            ang_vel_command: np.ndarray, shape (1, 1)
            stand_command: np.ndarray, shape (1, 1)
            upper_body_motion_active: bool

        Returns:
            (lin_vel_command, ang_vel_command, key_states): updated values
        """
        wc_msg = self.get_joystick_msg()
        if wc_msg is None:
            return lin_vel_command, ang_vel_command, self._key_states
        # Process stick input
        if getattr(wc_msg, "keys", 0) == 0 and not upper_body_motion_active:
            lx = getattr(wc_msg, "lx", 0.0)
            ly = getattr(wc_msg, "ly", 0.0)
            rx = getattr(wc_msg, "rx", 0.0)
            lin_vel_command[0, 1] = -(lx if abs(lx) > 0.1 else 0.0) * stand_command[0, 0]
            lin_vel_command[0, 0] = (ly if abs(ly) > 0.1 else 0.0) * stand_command[0, 0]
            ang_vel_command[0, 0] = -(rx if abs(rx) > 0.1 else 0.0) * stand_command[0, 0]
        # Process button input
        cur_key = self.get_joystick_key(wc_msg)
        self._last_key_states = self._key_states.copy()
        if cur_key:
            self._key_states[cur_key] = True
        else:
            self._key_states = dict.fromkeys(self._wc_key_map.values(), False)

        return lin_vel_command, ang_vel_command, self._key_states

    def _default_wc_key_map(self):
        """Default wireless controller key mapping."""
        return {
            1: "R1",
            2: "L1",
            3: "L1+R1",
            4: "start",
            8: "select",
            10: "L1+select",
            16: "R2",
            32: "L2",
            64: "F1",
            128: "F2",
            256: "A",
            264: "select+A",
            512: "B",
            520: "select+B",
            768: "A+B",
            1024: "X",
            1032: "select+X",
            1280: "A+X",
            1536: "B+X",
            2048: "Y",
            2304: "A+Y",
            2560: "B+Y",
            2056: "select+Y",
            3072: "X+Y",
            4096: "up",
            4097: "R1+up",
            4352: "A+up",
            4608: "B+up",
            4104: "select+up",
            5120: "X+up",
            6144: "Y+up",
            8192: "right",
            8193: "R1+right",
            8448: "A+right",
            9216: "X+right",
            10240: "Y+right",
            8200: "select+right",
            16384: "down",
            16392: "select+down",
            16385: "R1+down",
            16640: "A+down",
            16896: "B+down",
            17408: "X+down",
            18432: "Y+down",
            32768: "left",
            32769: "R1+left",
            32776: "select+left",
            33024: "A+left",
            33792: "X+left",
            34816: "Y+left",
        }

    @property
    @abstractmethod
    def kp_level(self):
        raise NotImplementedError

    @kp_level.setter
    @abstractmethod
    def kp_level(self, value):
        raise NotImplementedError

    @property
    @abstractmethod
    def kd_level(self):
        raise NotImplementedError

    @kd_level.setter
    @abstractmethod
    def kd_level(self, value):
        raise NotImplementedError
