"""Booster robot interface using sdk2py."""

import numpy as np
from termcolor import colored

from holosoma_inference.config.config_types import RobotConfig
from holosoma_inference.sdk.base.base_interface import BaseInterface


class BoosterInterface(BaseInterface):
    """Interface for Booster robots using sdk2py."""

    def __init__(self, robot_config: RobotConfig, domain_id=0, interface_str=None, use_joystick=True):
        super().__init__(robot_config, domain_id, interface_str, use_joystick)
        self._init_sdk2py()
        if use_joystick:
            self._init_joystick()

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
