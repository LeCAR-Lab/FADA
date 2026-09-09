"""Booster robot interface using sdk2py."""

import numpy as np
from termcolor import colored

from holosoma_inference.config.config_types import RobotConfig
from holosoma_inference.config.config_types.task import TaskConfig
from holosoma_inference.sdk.base.base_interface import BaseInterface


class BoosterInterface(BaseInterface):
    """Interface for Booster robots using sdk2py."""

    def __init__(
        self,
        robot_config: RobotConfig,
        domain_id=0,
        interface_str=None,
        use_joystick=True,
        task_config: TaskConfig | None = None,
    ):
        super().__init__(robot_config, domain_id, interface_str, use_joystick, task_config)
        self._sync_low_state_array: np.ndarray | None = None
        self._sync_low_state_tick: int | None = None
        self._init_sdk2py()
        if use_joystick:
            self._init_joystick()

    def reset_low_state_cache(self) -> None:
        """Drop cached low-state so the next read must come from a fresh publish."""
        self._sync_low_state_array = None
        self._sync_low_state_tick = None
        if hasattr(self.state_processor, "robot_state_data"):
            self.state_processor.robot_state_data = None

    def set_sync_low_state_array(self, low_state_array, tick: int | None = None) -> None:
        """Install a lock-step sim low-state snapshot received on the sync ZMQ channel.

        Sim publishes a 13 + 2*ndof array — base_pos(3) + quat(4) + joint_pos(ndof) +
        base_lin_vel(3) + base_ang_vel(3) + joint_vel(ndof). We mirror UnitreeInterface
        exactly: store as-is, no padding. Downstream policy code reads only this
        prefix; the ``tau_est`` / ``ddq`` tails Booster's DDS state_processor produces
        are absent here, but those entries are not consumed by inference (logging
        slices that touch them get empty arrays the same way as the G1 path does).
        Bypassing the async DDS callback is what makes Booster bit-exact under
        ``sync_sim_policy``.
        """
        self._sync_low_state_array = (
            np.asarray(low_state_array, dtype=np.float64).reshape(1, -1).copy()
        )
        self._sync_low_state_tick = int(tick) if tick is not None else None

    def _init_sdk2py(self):
        """Initialize sdk2py components."""
        from holosoma_inference.sdk.booster.command_sender import create_command_sender
        from holosoma_inference.sdk.booster.state_processor import create_state_processor

        self.command_sender = create_command_sender(self.robot_config)
        self.state_processor = create_state_processor(self.robot_config)

    def _init_joystick(self):
        """Initialize booster joystick/remote control."""
        from holosoma_inference.sdk.booster.command_sender.booster.joystick_message import BoosterJoystickMessage
        from holosoma_inference.sdk.booster.command_sender.booster.remote_control_service import (
            BoosterRemoteControlService,
        )

        try:
            self.booster_remote_control = BoosterRemoteControlService()
            self.booster_joystick_msg = BoosterJoystickMessage(self.booster_remote_control)
            print(colored("Booster Remote Control Service Initialized", "green"))
        except ImportError as e:
            print(colored(f"Warning: Failed to initialize booster remote control: {e}", "yellow"))
            self.booster_remote_control = None
            self.booster_joystick_msg = None

    def update_config(self, robot_config: RobotConfig):
        """Update config and propagate to sdk2py components."""
        super().update_config(robot_config)
        self.command_sender.config = robot_config
        self.state_processor.config = robot_config

    def get_low_state(self) -> np.ndarray:
        """Get robot state as numpy array."""
        if self._sync_low_state_array is not None:
            return self._sync_low_state_array.copy()
        return self.state_processor.get_robot_state_data()

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
        self.command_sender.send_command(
            cmd_q,
            cmd_dq,
            cmd_tau,
            dof_pos_latest,
            kp_override=kp_override,
            kd_override=kd_override,
        )

    def get_sync_low_cmd_payload(self):
        """Return the latest command snapshot for lock-step sim, if available."""
        getter = getattr(self.command_sender, "get_sync_low_cmd_payload", None)
        if getter is None:
            return None
        return getter()

    def get_joystick_msg(self):
        """Get wireless controller message."""
        return self.booster_joystick_msg if hasattr(self, "booster_joystick_msg") else None

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
        return self.command_sender.kp_level

    @kp_level.setter
    def kp_level(self, value):
        """Set proportional gain level."""
        self.command_sender.kp_level = value

    @property
    def kd_level(self):
        """Get derivative gain level."""
        return getattr(self.command_sender, "kd_level", 1.0)

    @kd_level.setter
    def kd_level(self, value):
        """Set derivative gain level."""
        self.command_sender.kd_level = value
