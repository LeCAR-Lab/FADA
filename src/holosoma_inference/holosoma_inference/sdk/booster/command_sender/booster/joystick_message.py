class BoosterJoystickMessage:
    """Simple message class to provide unified interface for booster joystick data."""

    def __init__(self, remote_control_service):
        self.remote_control_service = remote_control_service

    @property
    def lx(self):
        """Left stick X axis (lateral movement)."""
        return self.remote_control_service.lx

    @property
    def ly(self):
        """Left stick Y axis (forward/backward movement)."""
        return self.remote_control_service.ly

    @property
    def rx(self):
        """Right stick X axis (yaw rotation)."""
        return self.remote_control_service.rx

    @property
    def keys(self):
        """Button states as integer bitmask."""
        return self.remote_control_service.keys
