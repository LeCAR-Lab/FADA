from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import evdev


@dataclass
class JoystickConfig:
    max_vx: float = 0.8
    max_vy: float = 0.5
    max_vyaw: float = 0.5
    control_threshold: float = 0.1
    # logitech - left and right analog sticks
    left_x_axis: evdev.ecodes = evdev.ecodes.ABS_X  # Left stick X (left/right)
    left_y_axis: evdev.ecodes = evdev.ecodes.ABS_Y  # Left stick Y (forward/back)
    right_x_axis: evdev.ecodes = evdev.ecodes.ABS_RX  # Right stick X (yaw rotation)
    right_y_axis: evdev.ecodes = evdev.ecodes.ABS_RY  # Right stick Y (not used currently)

    # beitong
    # left_x_axis: evdev.ecodes = evdev.ecodes.ABS_X
    # left_y_axis: evdev.ecodes = evdev.ecodes.ABS_Y
    # right_x_axis: evdev.ecodes = evdev.ecodes.ABS_RX
    # right_y_axis: evdev.ecodes = evdev.ecodes.ABS_RY


class BoosterRemoteControlService:
    """Service for handling joystick remote control input for Booster robots."""

    def __init__(self, config: JoystickConfig | None = None):
        """Initialize remote control service with optional configuration."""
        self.config = config or JoystickConfig()
        self._lock = threading.Lock()
        self._running = True

        self.vx = 0.0
        self.vy = 0.0
        self.vyaw = 0.0
        self.lx = 0.0
        self.ly = 0.0
        self.rx = 0.0
        self.keys = 0

        try:
            self._init_joystick()
            self._start_joystick_thread()
        except Exception as e:
            print(f"Failed to initialize joystick: {e}")
            self.joystick = None
            self.joystick_runner = None

    def _init_joystick(self) -> None:
        """Initialize and validate joystick connection using evdev."""
        devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
        joystick = None

        for device in devices:
            caps = device.capabilities()
            # Check for both absolute axes and keys
            if evdev.ecodes.EV_ABS in caps and evdev.ecodes.EV_KEY in caps:
                abs_info = caps.get(evdev.ecodes.EV_ABS, [])
                # Look for typical gamepad axes
                axes = [code for (code, info) in abs_info]
                if all(
                    code in axes
                    for code in [self.config.left_x_axis, self.config.left_y_axis, self.config.right_x_axis]
                ):
                    absinfo = {code: info for code, info in abs_info}
                    self.axis_ranges = {
                        self.config.left_x_axis: absinfo[self.config.left_x_axis],
                        self.config.left_y_axis: absinfo[self.config.left_y_axis],
                        self.config.right_x_axis: absinfo[self.config.right_x_axis],
                        self.config.right_y_axis: absinfo[self.config.right_y_axis],
                    }
                    print(f"Found suitable joystick: {device.name}")
                    joystick = device
                    break

        if not joystick:
            raise RuntimeError("No suitable joystick found")

        self.joystick = joystick
        print(f"Selected joystick: {joystick.name}")

    def _start_joystick_thread(self):
        """Start joystick polling thread."""
        self.joystick_runner = threading.Thread(target=self._run_joystick)
        self.joystick_runner.daemon = True
        self.joystick_runner.start()

    def _run_joystick(self):
        """Poll joystick events."""
        while self._running:
            if not self._process_events():
                break

    def _process_events(self):
        """Process joystick events."""
        try:
            for event in self.joystick.read_loop():
                if event.type == evdev.ecodes.EV_ABS:
                    # Handle axis events
                    self._handle_axis(event.code, event.value)
                elif event.type == evdev.ecodes.EV_KEY:
                    # Handle button events
                    self._handle_button(event.code, event.value)
            return True
        except BlockingIOError:
            # No events available
            time.sleep(0.01)
            return True
        except Exception:
            return False

    def _handle_axis(self, code: int, value: int):
        """Handle axis events."""
        try:
            # Left stick - movement
            if code == self.config.left_y_axis:  # Left stick Y (forward/back)
                self.vx = self._scale(value, self.config.max_vx, self.config.control_threshold, code)
                self.ly = self.vx  # Map for compatibility with unitree interface
            elif code == self.config.left_x_axis:  # Left stick X (left/right)
                self.vy = self._scale(value, self.config.max_vy, self.config.control_threshold, code)
                self.lx = -self.vy  # Map for compatibility with unitree interface (inverted)
            # Right stick - rotation
            elif code == self.config.right_x_axis:  # Right stick X (yaw rotation)
                self.vyaw = self._scale(value, self.config.max_vyaw, self.config.control_threshold, code)
                self.rx = -self.vyaw  # Map for compatibility with unitree interface (inverted)
            elif code == self.config.right_y_axis:  # Right stick Y (not used currently)
                pass  # Could be used for pitch control in the future
            # D-pad handling via HAT axes (most common for controllers)
            elif code == evdev.ecodes.ABS_HAT0X:  # D-pad horizontal
                self._handle_dpad_axis("horizontal", value)
            elif code == evdev.ecodes.ABS_HAT0Y:  # D-pad vertical
                self._handle_dpad_axis("vertical", value)
            # Analog triggers
            elif code == evdev.ecodes.ABS_RZ:  # Right trigger (R2)
                self._handle_analog_trigger("R2", value)
            elif code == evdev.ecodes.ABS_Z:  # Left trigger (L2) - some controllers
                self._handle_analog_trigger("L2", value)
            else:
                pass
        except Exception:
            raise

    def _handle_button(self, code: int, value: int):
        """Handle button events."""
        # Track individual button states
        if not hasattr(self, "_button_states"):
            self._button_states = {}

        if value == 1:  # Button pressed
            self._button_states[code] = True
        elif value == 0:  # Button released
            self._button_states[code] = False

        # Calculate combined keys value based on button combinations
        self.keys = self._calculate_keys_value()

    def _handle_dpad_axis(self, direction: str, value: int):
        """Handle D-pad input via HAT axes."""
        # Track D-pad states
        if not hasattr(self, "_dpad_states"):
            self._dpad_states = {}

        if direction == "horizontal":
            # HAT0X: -1 = left, 0 = center, 1 = right
            self._dpad_states["left"] = value == -1
            self._dpad_states["right"] = value == 1
            if value == 0:  # Released
                self._dpad_states["left"] = False
                self._dpad_states["right"] = False
        elif direction == "vertical":
            # HAT0Y: -1 = up, 0 = center, 1 = down
            self._dpad_states["up"] = value == -1
            self._dpad_states["down"] = value == 1
            if value == 0:  # Released
                self._dpad_states["up"] = False
                self._dpad_states["down"] = False

        # Update keys value
        self.keys = self._calculate_keys_value()

    def _handle_analog_trigger(self, trigger_name: str, value: int):
        """Handle analog trigger as button press."""
        # Track trigger states
        if not hasattr(self, "_trigger_states"):
            self._trigger_states = {}

        # Treat trigger as pressed if value is above threshold (usually > 128 for 8-bit triggers)
        threshold = 128
        is_pressed = value > threshold

        self._trigger_states[trigger_name] = is_pressed

        # Update keys value
        self.keys = self._calculate_keys_value()

    def _calculate_keys_value(self):
        """Calculate keys value based on current button states to match unitree mapping."""
        keys_value = 0

        # Check button states
        if hasattr(self, "_button_states"):
            button_states = self._button_states

            # Map common buttons to unitree equivalents
            # BTN_A -> A (256), BTN_B -> B (512), BTN_X -> X (1024), BTN_Y -> Y (2048)
            # BTN_START -> start (4), BTN_SELECT -> select (8)
            # BTN_TR -> R1 (1), BTN_TL -> L1 (2), BTN_TR2 -> R2 (16), BTN_TL2 -> L2 (32)

            if button_states.get(evdev.ecodes.BTN_A, False):
                keys_value |= 256  # A
                print("A pressed")
            if button_states.get(evdev.ecodes.BTN_B, False):
                keys_value |= 512  # B
                print("B pressed")
            if button_states.get(evdev.ecodes.BTN_X, False):
                keys_value |= 1024  # X
                print("X pressed")
            if button_states.get(evdev.ecodes.BTN_Y, False):
                keys_value |= 2048  # Y
                print("Y pressed")
            if button_states.get(evdev.ecodes.BTN_START, False):
                keys_value |= 4  # start
                print("Start pressed")
            if button_states.get(evdev.ecodes.BTN_SELECT, False):
                keys_value |= 8  # select
                print("Select pressed")
            if button_states.get(evdev.ecodes.BTN_TR, False):
                keys_value |= 1  # R1
                print("R1 pressed")
            if button_states.get(evdev.ecodes.BTN_TL, False):
                keys_value |= 2  # L1
                print("L1 pressed")
            if button_states.get(evdev.ecodes.BTN_TR2, False):
                keys_value |= 16  # R2
                print("R2 pressed")
            if button_states.get(evdev.ecodes.BTN_TL2, False):
                keys_value |= 32  # L2
                print("L2 pressed")

            # Add D-pad support
            if button_states.get(evdev.ecodes.BTN_DPAD_UP, False):
                keys_value |= 4096  # up
                print("Up pressed")
            if button_states.get(evdev.ecodes.BTN_DPAD_DOWN, False):
                keys_value |= 16384  # down
                print("Down pressed")
            if button_states.get(evdev.ecodes.BTN_DPAD_LEFT, False):
                keys_value |= 32768  # left
                print("Left pressed")
            if button_states.get(evdev.ecodes.BTN_DPAD_RIGHT, False):
                keys_value |= 8192  # right
                print("Right pressed")

        # Check D-pad states from HAT axes
        if hasattr(self, "_dpad_states"):
            dpad_states = self._dpad_states
            if dpad_states.get("up", False):
                keys_value |= 4096  # up
                print("D-pad Up pressed")
            if dpad_states.get("down", False):
                keys_value |= 16384  # down
                print("D-pad Down pressed")
            if dpad_states.get("left", False):
                keys_value |= 32768  # left
                print("D-pad Left pressed")
            if dpad_states.get("right", False):
                keys_value |= 8192  # right
                print("D-pad Right pressed")

        # Check analog trigger states
        if hasattr(self, "_trigger_states"):
            trigger_states = self._trigger_states
            if trigger_states.get("R2", False):
                keys_value |= 16  # R2
                print("R2 pressed")
            if trigger_states.get("L2", False):
                keys_value |= 32  # L2
                print("L2 pressed")

        return keys_value

    def _scale(self, value: float, max_val: float, threshold: float, axis_code: int) -> float:
        """Scale joystick input to velocity command using actual axis ranges."""
        absinfo = self.axis_ranges[axis_code]
        min_in = absinfo.min
        max_in = absinfo.max

        mapped_value = ((value - min_in) / (max_in - min_in) * 2 - 1) * max_val

        if abs(mapped_value) < threshold:
            return 0.0
        return -mapped_value

    def close(self):
        """Clean up resources."""
        self._running = False
        if hasattr(self, "joystick") and self.joystick is not None:
            self.joystick.close()
        if hasattr(self, "joystick_runner") and self.joystick_runner is not None:
            self.joystick_runner.join(timeout=1.0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
