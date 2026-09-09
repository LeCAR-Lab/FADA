"""Dual-mode policy with runtime switching between two policy instances."""

from __future__ import annotations

import itertools
import time
from pathlib import Path

from loguru import logger
from termcolor import colored

from holosoma_inference.config.config_types.inference import InferenceConfig


def _select_policy_class(config: InferenceConfig):
    """Determine policy class from explicit task.policy_mode or legacy config fallback."""
    from importlib.metadata import entry_points

    from holosoma_inference.policies.locomotion import (
        LocomotionPolicy,
        LocomotionPolicy_Deploy,
    )
    from holosoma_inference.policies.locomotion_fada import (
        LocomotionPolicy_FADA,
    )
    from holosoma_inference.policies.wbt import WholeBodyTrackingPolicy

    mode = getattr(getattr(config, "task", None), "policy_mode", "auto")
    explicit_map = {
        # "transformer" selects the plain compact-obs transformer backbone with no planner/IDM
        # teacher heads. LocomotionPolicy_FADA gracefully degrades to that behavior when the
        # loaded ONNX has no IDM/FDM teacher I/O, so it dispatches to the same class as "fada".
        "transformer": LocomotionPolicy_FADA,
        "fada": LocomotionPolicy_FADA,
        "deploy": LocomotionPolicy_Deploy,
        "wbt": WholeBodyTrackingPolicy,
    }
    if mode in explicit_map:
        return explicit_map[mode]

    robot_type = config.robot.robot_type
    actor_obs = config.observation.obs_dict.get("actor_obs", [])

    if "motion_command" in actor_obs:
        for ep in entry_points(group="holosoma.policies.wbt"):
            if ep.name == robot_type:
                return ep.load()
        return WholeBodyTrackingPolicy

    for ep in entry_points(group="holosoma.policies.locomotion"):
        if ep.name == robot_type:
            return ep.load()
    return LocomotionPolicy


class DualModePolicy:
    """Wrap two policies, optionally with robust/test session semantics.

    Typical wiring: ``primary`` is the policy under test/deployment -- usually
    LocomotionPolicy_FADA (policy_mode="fada" or "transformer") -- and
    ``secondary`` is a robust safety fallback, dispatched via whatever
    policy_mode its own config carries. The secondary accepts any registered
    policy_mode (e.g. "deploy" -> LocomotionPolicy_Deploy, "auto" -> a
    FastSAC/robust checkpoint).
    """

    def __init__(self, primary_config: InferenceConfig, secondary_config: InferenceConfig):
        primary_cls = _select_policy_class(primary_config)
        secondary_cls = _select_policy_class(secondary_config)

        logger.info(
            colored(f"Dual-mode: primary={primary_cls.__name__}, secondary={secondary_cls.__name__}", "magenta")
        )

        self.primary = primary_cls(config=primary_config)

        # Secondary is the robust/safety fallback policy for this dual-mode session, selected
        # the same way as primary (via task.policy_mode on secondary_config). It shares the
        # primary's hardware interface via `_shared_hardware_source`.
        logger.info(colored("Initializing secondary policy (shared hardware)...", "magenta"))
        secondary = object.__new__(secondary_cls)
        secondary._shared_hardware_source = self.primary
        secondary.__init__(config=secondary_config)
        self.secondary = secondary

        self.active = self.primary
        self.active_label = "primary"

        self._session_mode = bool(getattr(primary_config.task, "dual_mode_session_enabled", False))
        self._start_in_secondary = bool(getattr(primary_config.task, "dual_mode_start_in_secondary", True))
        self._test_start_delay_s = float(getattr(primary_config.task, "dual_mode_test_start_delay_s", -1.0))
        self._post_secondary_hold_s = float(getattr(primary_config.task, "dual_mode_post_secondary_hold_s", 0.0))
        self._exit_after_post_hold = bool(getattr(primary_config.task, "dual_mode_exit_after_post_hold", False))
        self._projected_gravity_z_min = float(
            getattr(primary_config.task, "dual_mode_projected_gravity_z_min", -0.7)
        )

        self._primary_iteration = 0
        self._secondary_iteration = 0
        self._auto_test_deadline: float | None = None
        self._post_hold_deadline: float | None = None

        if self._session_mode:
            self._init_session_mode()

        self._patch_button_handlers()

        if self._session_mode:
            logger.info(
                colored(
                    "Dual-mode session ready. Start in robust secondary, press X/x to enter or exit test session.",
                    "magenta",
                )
            )
        else:
            logger.info(colored("Dual-mode ready. Press X (joystick) or x (keyboard) to switch policies.", "magenta"))

    def _patch_button_handlers(self):
        """Intercept X/x for mode/session switching; delegate all other keys."""
        self._orig_joy = {
            id(self.primary): self.primary.handle_joystick_button,
            id(self.secondary): self.secondary.handle_joystick_button,
        }
        self._orig_kb = {
            id(self.primary): self.primary.handle_keyboard_button,
            id(self.secondary): self.secondary.handle_keyboard_button,
        }

        def patched_joy(cur_key):
            if cur_key == "X":
                if self._session_mode:
                    self._handle_session_toggle()
                else:
                    self._handle_legacy_mode_switch()
            else:
                self._orig_joy[id(self.active)](cur_key)

        def patched_kb(keycode):
            if keycode == "x":
                if self._session_mode:
                    self._handle_session_toggle()
                else:
                    self._handle_legacy_mode_switch()
            else:
                self._orig_kb[id(self.active)](keycode)

        self.primary.handle_joystick_button = patched_joy
        self.primary.handle_keyboard_button = patched_kb
        self.secondary.handle_joystick_button = patched_joy
        self.secondary.handle_keyboard_button = patched_kb

    def _copy_key_states(self, source, target):
        if hasattr(source, "key_states"):
            target.key_states = source.key_states.copy()
            target.last_key_states = source.key_states.copy()

    def _activate_target(self, target, label: str, previous=None):
        if previous is not None:
            self._copy_key_states(previous, target)
        target._resolve_control_gains()
        self.active = target
        self.active_label = label

    def _write_signal_file(self, path: str | None, label: str):
        if not path:
            return
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        path_obj.write_text(str(time.time()))
        logger.info(f"{label} signal written to {path}")

    def _init_session_mode(self):
        previous = self.primary
        if self._start_in_secondary:
            self._enter_secondary_robust(reason="startup", schedule_post_hold=False, announce=False, previous=previous)
        else:
            self._enter_primary_test_session(reason="startup", announce=False, previous=previous)
        self._write_signal_file(getattr(self.primary.config.task, "ready_signal_file", None), "Ready")
        if self._test_start_delay_s >= 0 and self.active is self.secondary:
            self._auto_test_deadline = time.perf_counter() + self._test_start_delay_s

    def _enter_secondary_robust(
        self,
        *,
        reason: str,
        schedule_post_hold: bool,
        announce: bool = True,
        previous=None,
    ):
        old_active = previous if previous is not None else self.active
        if old_active is not None:
            old_active._skip_current_iteration = True
            old_active._handle_stop_policy()

        self.secondary.reset_runtime_state_for_new_session()
        self.secondary.prepare_unrecorded_session()
        if hasattr(self.secondary, "randomize_commands"):
            self.secondary.randomize_commands = False
        if hasattr(self.secondary, "command_resample_steps"):
            self.secondary.command_resample_steps = None
        if hasattr(self.secondary, "last_command_change_iteration"):
            self.secondary.last_command_change_iteration = -1
        if hasattr(self.secondary, "command_transition_start_iteration"):
            self.secondary.command_transition_start_iteration = -1
        self.secondary.lin_vel_command[0, :] = 0.0
        self.secondary.ang_vel_command[0, 0] = 0.0
        self.secondary.stand_command[0, 0] = 0.0
        self.secondary.base_height_command[0, 0] = self.secondary.desired_base_height
        self.secondary._handle_start_policy()

        self._secondary_iteration = 0
        self._activate_target(self.secondary, "secondary", previous=old_active)

        if schedule_post_hold and self._exit_after_post_hold:
            self._post_hold_deadline = time.perf_counter() + max(0.0, self._post_secondary_hold_s)
        else:
            self._post_hold_deadline = None

        if announce:
            logger.info(
                colored(
                    f"Entered robust secondary mode ({type(self.secondary).__name__}); reason={reason}",
                    "magenta",
                    attrs=["bold"],
                )
            )

    def _enter_primary_test_session(self, *, reason: str, announce: bool = True, previous=None):
        old_active = previous if previous is not None else self.active
        if old_active is not None:
            old_active._skip_current_iteration = True
            old_active._handle_stop_policy()

        self.primary.reset_runtime_state_for_new_session()
        self.primary.prepare_recorded_session(reset_outputs=True)
        self.primary._handle_start_policy()

        self._primary_iteration = 0
        self._auto_test_deadline = None
        self._post_hold_deadline = None
        self._activate_target(self.primary, "primary", previous=old_active)

        if announce:
            logger.info(
                colored(
                    f"Entered test primary session ({type(self.primary).__name__}); reason={reason}",
                    "magenta",
                    attrs=["bold"],
                )
            )

    def _exit_primary_session(self, reason: str):
        self.primary.prepare_unrecorded_session()
        self._enter_secondary_robust(reason=reason, schedule_post_hold=True)

    def _handle_session_toggle(self):
        if self.active is self.secondary:
            self._enter_primary_test_session(reason="manual_toggle")
        else:
            self._exit_primary_session("manual_toggle")

    def _handle_legacy_mode_switch(self):
        """Switch from active to inactive policy."""
        previous = self.active
        previous._skip_current_iteration = True
        previous._handle_stop_policy()

        target = self.secondary if previous is self.primary else self.primary
        target_label = "secondary" if target is self.secondary else "primary"

        target._resolve_control_gains()
        self._copy_key_states(previous, target)

        self.active = target
        self.active_label = target_label
        self.active._init_phase_components()
        self.active._handle_start_policy()

        logger.info(
            colored(
                f"Switched to {self.active_label} policy ({type(self.active).__name__})",
                "magenta",
                attrs=["bold"],
            )
        )

    def _maybe_auto_start_test_session(self):
        if not self._session_mode or self.active is not self.secondary:
            return
        if self._auto_test_deadline is None:
            return
        if time.perf_counter() < self._auto_test_deadline:
            return
        self._enter_primary_test_session(reason="auto_delay_elapsed")

    def _check_primary_safety(self) -> bool:
        if not self._session_mode or self.active is not self.primary:
            return False
        robot_state = self.primary.interface.get_low_state()
        if robot_state is None:
            return False
        projected_gravity_z = self.primary.get_projected_gravity_z(robot_state)
        if projected_gravity_z <= self._projected_gravity_z_min:
            return False
        logger.warning(
            colored(
                f"Unsafe projected_gravity_z={projected_gravity_z:.3f} > {self._projected_gravity_z_min:.3f}; "
                "switching to robust secondary.",
                "red",
            )
        )
        self._exit_primary_session("projected_gravity_guard")
        return True

    def _finalize(self):
        if self._session_mode:
            self.primary._after_run_loop()
        else:
            self.active._after_run_loop()

    def _run_session_mode(self):
        for _ in itertools.count():
            self._maybe_auto_start_test_session()

            if self._check_primary_safety():
                if self._post_hold_deadline is not None and time.perf_counter() >= self._post_hold_deadline:
                    return
                self.active.rate.sleep()
                continue

            policy = self.active
            iteration = self._primary_iteration if policy is self.primary else self._secondary_iteration
            should_stop = policy._run_iteration(iteration)

            if policy is self.primary:
                self._primary_iteration += 1
                if policy is self.active and should_stop:
                    self._exit_primary_session("test_session_completed")
            else:
                self._secondary_iteration += 1
                if policy is self.active and should_stop:
                    return

            if self._post_hold_deadline is not None and time.perf_counter() >= self._post_hold_deadline:
                return

            self.active.rate.sleep()

    def _run_legacy_mode(self):
        for it in itertools.count():
            policy = self.active
            should_stop = policy._run_iteration(it)
            if policy is self.active and should_stop:
                return
            self.active.rate.sleep()

    def run(self):
        """Main run loop."""
        try:
            if self._session_mode:
                self._run_session_mode()
            else:
                self._run_legacy_mode()
        except KeyboardInterrupt:
            pass
        finally:
            self._finalize()
