import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from termcolor import colored

from holosoma_inference.config.config_types.inference import InferenceConfig

from .base import BasePolicy


def _step6_fall_projected_gravity_z_max() -> float | None:
    """Step 6's fall threshold, expressed in the collector's sign convention.

    `projected_gravity_z == -R[2,2]` exactly, so `plot_mocap_raw.STEP6_FALL_R22_MAX` (0.5)
    is `projected_gravity_z > -0.5` here. Imported lazily; returns `None` when the import
    fails, meaning the cross-check is unavailable.
    """
    try:
        from holosoma_inference.utils.plot_mocap_raw import STEP6_FALL_R22_MAX

        return -float(STEP6_FALL_R22_MAX)
    except Exception:  # pragma: no cover - reporting must never break a run
        return None


def _step6_fall_warmup_fraction() -> float:
    """Step 6's warmup fraction; falls back to 0.05 when the lazy import fails."""
    try:
        from holosoma_inference.utils.plot_mocap_raw import STEP6_FALL_WARMUP_FRACTION

        return float(STEP6_FALL_WARMUP_FRACTION)
    except Exception:  # pragma: no cover - reporting must never break a run
        return 0.05


_STEP6_FALL_WARMUP_FRACTION = _step6_fall_warmup_fraction()


def _tilt_deg_from_projected_gravity_z(threshold: float) -> float:
    """Body tilt in degrees at which a `projected_gravity_z > threshold` test starts firing."""
    return math.degrees(math.acos(max(-1.0, min(1.0, -float(threshold)))))


class LocomotionPolicy(BasePolicy):
    def __init__(self, config):
        super().__init__(config)
        self.is_standing = False

    def get_current_obs_buffer_dict(self, robot_state_data):
        current_obs_buffer_dict = super().get_current_obs_buffer_dict(robot_state_data)
        current_obs_buffer_dict["actions"] = self.last_policy_action
        current_obs_buffer_dict["command_lin_vel"] = self.lin_vel_command
        current_obs_buffer_dict["command_ang_vel"] = self.ang_vel_command
        current_obs_buffer_dict["command_stand"] = self.stand_command

        # Add phase observations only if they are configured
        if "sin_phase" in self.obs_dict.get("actor_obs", []):
            current_obs_buffer_dict["sin_phase"] = self._get_obs_sin_phase()
        if "cos_phase" in self.obs_dict.get("actor_obs", []):
            current_obs_buffer_dict["cos_phase"] = self._get_obs_cos_phase()

        return current_obs_buffer_dict

    def _get_obs_sin_phase(self):
        """Calculate sin phase for gait."""
        return np.array([np.sin(self.phase[0, :])])

    def _get_obs_cos_phase(self):
        """Calculate cos phase for gait."""
        return np.array([np.cos(self.phase[0, :])])

    def update_phase_time(self):
        """Update phase time."""
        phase_tp1 = self.phase + self.phase_dt
        self.phase = np.fmod(phase_tp1 + np.pi, 2 * np.pi) - np.pi
        if np.linalg.norm(self.lin_vel_command[0]) < 0.01 and np.linalg.norm(self.ang_vel_command[0]) < 0.01:
            # Robot should stand still - set both feet to same phase
            self.phase[0, :] = np.pi * np.ones(2)
            self.is_standing = True
        elif self.is_standing:
            # When the robot starts to move, reset the phase to initial state
            self.phase = np.array([[0.0, np.pi]])
            self.is_standing = False

    def handle_keyboard_button(self, keycode):
        """Handle keyboard button presses for locomotion."""
        # Call parent handler for common commands
        super().handle_keyboard_button(keycode)

        # Locomotion-specific commands
        if keycode in ["w", "s", "a", "d"]:
            self._handle_velocity_control(keycode)
        elif keycode in ["q", "e"]:
            self._handle_angular_velocity_control(keycode)
        elif keycode == "=":
            self._handle_stand_command()
        elif keycode == "z":
            self._handle_zero_velocity()

        self._print_control_status()

    def handle_joystick_button(self, cur_key):
        """Handle joystick button presses for locomotion."""
        # Call parent handler for common commands
        super().handle_joystick_button(cur_key)

        # Locomotion-specific commands
        if cur_key == "start":
            self._handle_stand_command()
        elif cur_key == "L2":
            self._handle_zero_velocity()

    def _handle_velocity_control(self, keycode):
        """Handle linear velocity control."""
        if not self.stand_command[0, 0]:
            return

        if keycode == "w":
            self.lin_vel_command[0, 0] += 0.1
        elif keycode == "s":
            self.lin_vel_command[0, 0] -= 0.1
        elif keycode == "a":
            self.lin_vel_command[0, 1] += 0.1
        elif keycode == "d":
            self.lin_vel_command[0, 1] -= 0.1

    def _handle_angular_velocity_control(self, keycode):
        """Handle angular velocity control."""
        if keycode == "q":
            self.ang_vel_command[0, 0] -= 0.1
        elif keycode == "e":
            self.ang_vel_command[0, 0] += 0.1

    def _handle_stand_command(self):
        """Handle stand command toggle."""
        self.stand_command[0, 0] = 1 - self.stand_command[0, 0]
        if self.stand_command[0, 0] == 0:
            self.ang_vel_command[0, 0] = 0.0
            self.lin_vel_command[0, 0] = 0.0
            self.lin_vel_command[0, 1] = 0.0
            self.logger.info(colored("Stance command", "blue"))
        else:
            self.base_height_command[0, 0] = self.desired_base_height
            self.logger.info(colored("Walk command", "blue"))

    def _handle_zero_velocity(self):
        """Handle zero velocity command."""
        self.ang_vel_command[0, 0] = 0.0
        self.lin_vel_command[0, 0] = 0.0
        self.lin_vel_command[0, 1] = 0.0
        self.logger.info(colored("Velocities set to zero", "blue"))

    def _print_control_status(self):
        """Print current control status."""
        super()._print_control_status()

        # Extract values for better formatting
        lin_vel_x = self.lin_vel_command[0, 0]
        lin_vel_y = self.lin_vel_command[0, 1]
        ang_vel_z = self.ang_vel_command[0, 0]
        is_walking = self.stand_command[0, 0] == 1

        # Print with clear labels and units
        mode = "Walking" if is_walking else "Standing"
        status = "✓ applied" if is_walking else "✗ not applied"
        print(f"Linear velocity: x={lin_vel_x:+.2f} m/s, y={lin_vel_y:+.2f} m/s")
        print(f"Angular velocity: {ang_vel_z:+.2f} rad/s")
        print(f"Mode: {mode} ({status})")
        print("💡 Terminal keys: W/A/S/D (lin) | Q/E (ang) | = (toggle mode)")
        print("🎬 MuJoCo keys (in simulator only): 7/8 (band) | 9 (toggle) | BACKSPACE (reset)")

class LocomotionPolicy_Deploy(BasePolicy):
    """
    LocomotionPolicy_Deploy class for custom modifications with data collection support.

    Combines the generic deploy infrastructure -- command orchestration, session
    control, data collection -- with locomotion-specific observation/command
    handling. Must stay directly instantiable: it also serves as the dual-mode
    deployment secondary/safety fallback (policy_mode="deploy").
    """

    def __init__(self, config: InferenceConfig):
        """Initialize LocomotionPolicy_Deploy with same parameters as base BasePolicy."""
        super().__init__(config)

        # Command randomization config (deploy-only, matches train/eval). The getattr
        # fallback mirrors TaskConfig.randomize_commands' own default (True).
        self.randomize_commands = getattr(self.config.task, "randomize_commands", True)
        self.command_resampling_time = float(getattr(self.config.task, "command_resampling_time", 0.0))
        self.command_resample_steps = (
            max(1, int(self.command_resampling_time * self.rl_rate))
            if self.randomize_commands and self.command_resampling_time > 0
            else None
        )
        self.command_lin_vel_x_range = getattr(self.config.task, "command_lin_vel_x_range", (-1.0, 1.0))
        self.command_lin_vel_y_range = getattr(self.config.task, "command_lin_vel_y_range", (-1.0, 1.0))
        self.command_ang_vel_range = getattr(self.config.task, "command_ang_vel_range", (-1.0, 1.0))
        self.command_stand_prob = float(getattr(self.config.task, "command_stand_prob", 0.0))
        self.initial_zero_command_steps = max(
            0, int(getattr(self.config.task, "initial_zero_command_steps", 0) or 0)
        )
        self.last_random_resample_iteration = -1
        self._initial_zero_command_hold_logged = False
        self._initial_zero_command_release_logged = False

        # Maximum evaluation time limit
        self.max_eval_time = float(getattr(self.config.task, "max_eval_time", -1.0))
        self.eval_start_time = None  # Will be set when run() starts
        self.max_time_exceeded_logged = False  # Track if we've already logged the warning

        # Initialize data collector if enabled
        self.data_collector = None
        # Fall accounting for collected data. A MuJoCo/real rollout has no environment to
        # terminate it: the loop only stops on --task.max-steps / --task.max-eval-time, so a
        # robot that falls keeps being recorded, and every post-fall step lands in the H5
        # with dones=False and becomes an ordinary IDM LoRA training window.
        #
        # The default writes dones=False for those steps and only reports the post-fall
        # count; --task.collect-mark-fall-terminal marks them terminal instead, which changes
        # which windows the finetune sees.
        self.collect_mark_fall_terminal = bool(getattr(self.config.task, "collect_mark_fall_terminal", False))
        # Same criterion and same default as the dual-mode safety guard
        # (`dual_mode_projected_gravity_z_min`): upright is projected_gravity_z ~= -1, and
        # -0.7 is ~46 degrees of body tilt.
        self.collect_fall_projected_gravity_z_max = float(
            getattr(self.config.task, "collect_fall_projected_gravity_z_max", -0.7)
        )
        # The OTHER criterion, watched alongside this one for REPORTING ONLY. Step 6's
        # pre/post-finetune comparison calls a rollout fallen at `R[2,2] < 0.5` (60.000 deg)
        # while this collector fires at `projected_gravity_z > -0.7` (45.573 deg). Nothing
        # below feeds `dones`, the latch, or the post-fall count; it only lets the collection
        # report name both verdicts.
        self._step6_fall_projected_gravity_z_max = _step6_fall_projected_gravity_z_max()
        self._collection_step6_fall_step = None
        self._collection_fall_latched = False
        self._collection_fall_detectable = True
        self._collection_steps = 0
        self._collection_post_fall_steps = 0
        self._collection_fall_step = None
        self._collection_fall_signal = None
        if self.config.task.collect_data:
            self._init_data_collector()

        self.is_standing = False

    def _collection_done_flag(self, robot_state_data) -> np.ndarray:
        """Account for a fall on this collected step and return its `dones` row.

        The fall is always detected and counted (`_collection_post_fall_steps`, reported by
        `_log_collection_fall_report`). The returned flag is all-False unless
        `--task.collect-mark-fall-terminal true` is set, which terminates the episode at the
        fall.

        The fall state is latched: "post-fall steps" counts every step after the first fall,
        not only the tilted ones, and `DataCollector` ORs each row into a sticky per-env mask
        and zeroes every later step's payload, so the flag must stay True once raised.

        Detection reports itself unavailable once and stops when the tilt signal cannot be
        read or when `--task.debug.force-upright-imu` has pinned projected gravity to
        [0, 0, -1].
        """
        self._collection_steps += 1
        # Keep sampling after the collector has latched only while step 6's stricter
        # criterion has not yet fired: the 60-degree crossing comes at or after the
        # 45.573-degree one, so stopping at the latch would leave the second verdict
        # permanently "no fall". `get_projected_gravity_z` reads state and mutates nothing,
        # and neither branch below changes the latch, the count, or the `dones` row.
        if self._collection_fall_detectable and not (
            self._collection_fall_latched and getattr(self, "_collection_step6_fall_step", None) is not None
        ):
            self._detect_collection_fall(robot_state_data)
        if self._collection_fall_latched:
            self._collection_post_fall_steps += 1
            if self.collect_mark_fall_terminal:
                return np.ones(1, dtype=np.bool_)
        return np.zeros(1, dtype=np.bool_)

    def _detect_collection_fall(self, robot_state_data) -> None:
        """Latch `_collection_fall_latched` the first time the robot is past the tilt limit."""
        if getattr(self.config.task.debug, "force_upright_imu", False):
            # projected_gravity is hardcoded to [0, 0, -1], so the check can never fire.
            self._collection_fall_detectable = False
            self.logger.warning(
                "Fall detection is unavailable because --task.debug.force-upright-imu pins "
                "projected_gravity to [0, 0, -1]; the collected dataset's post-fall step count "
                "cannot be reported."
            )
            return
        try:
            projected_gravity_z = self.get_projected_gravity_z(robot_state_data)
        except Exception as exc:  # pragma: no cover - defensive, mirrors the collect block below
            self._collection_fall_detectable = False
            self.logger.warning(f"Fall detection unavailable ({exc}); post-fall steps cannot be reported.")
            return
        # Report-only: the first step at which step 6's criterion would have fired. Recorded
        # before the latch below so a single step can satisfy both.
        step6_max = getattr(self, "_step6_fall_projected_gravity_z_max", None)
        if (
            getattr(self, "_collection_step6_fall_step", None) is None
            and step6_max is not None
            and projected_gravity_z > step6_max
        ):
            self._collection_step6_fall_step = self._collection_steps - 1
        if self._collection_fall_latched:
            return
        if projected_gravity_z > self.collect_fall_projected_gravity_z_max:
            self._latch_collection_fall(
                "tilt",
                f"projected_gravity_z={projected_gravity_z:.3f} > "
                f"{self.collect_fall_projected_gravity_z_max:.3f}",
            )

    def _latch_collection_fall(self, signal: str, detail: str) -> None:
        self._collection_fall_latched = True
        self._collection_fall_step = self._collection_steps - 1
        self._collection_fall_signal = signal
        self.logger.warning(
            f"Fall detected during data collection at recorded step {self._collection_fall_step} "
            f"[{signal}]: {detail}."
        )

    def collection_fall_report(self) -> "dict[str, object]":
        """Machine-readable summary of what the just-collected dataset contains.

        Carries both fall verdicts. `fall_step` / `post_fall_steps` are this collector's and
        are the only ones that affect the dataset; the `step6_*` keys record what step 6's
        stricter criterion would have said about the same rollout.
        """
        collected_steps = int(getattr(self, "_collection_steps", 0))
        detectable = bool(getattr(self, "_collection_fall_detectable", True))
        collector_max = float(getattr(self, "collect_fall_projected_gravity_z_max", -0.7))
        step6_max = getattr(self, "_step6_fall_projected_gravity_z_max", None)
        step6_cross = getattr(self, "_collection_step6_fall_step", None)
        # Step 6 ignores the first 5% of the run before it starts looking; this collector
        # watches from step 0. A crossing inside that window yields the "indeterminate"
        # verdict below.
        step6_warmup_steps = max(1, int(collected_steps * _STEP6_FALL_WARMUP_FRACTION))
        if not detectable or step6_max is None:
            step6_verdict = "unavailable"
        elif step6_cross is None:
            step6_verdict = "no_fall"
        elif int(step6_cross) >= step6_warmup_steps:
            step6_verdict = "fell"
        else:
            step6_verdict = "indeterminate"
        collector_verdict = (
            "fell" if getattr(self, "_collection_fall_step", None) is not None else "no_fall"
        ) if detectable else "unavailable"
        disagree: bool | None = None
        if collector_verdict in ("fell", "no_fall") and step6_verdict in ("fell", "no_fall"):
            disagree = collector_verdict != step6_verdict
        return {
            "fall_detection_available": detectable,
            "collected_steps": collected_steps,
            "fall_step": getattr(self, "_collection_fall_step", None),
            "fall_signal": getattr(self, "_collection_fall_signal", None),
            "post_fall_steps": int(getattr(self, "_collection_post_fall_steps", 0)),
            "marked_terminal": bool(getattr(self, "collect_mark_fall_terminal", False)),
            # -- the two criteria, side by side (reporting only) ------------------------
            "collector_verdict": collector_verdict,
            "collector_fall_threshold_projected_gravity_z": collector_max,
            "collector_fall_tilt_deg": _tilt_deg_from_projected_gravity_z(collector_max),
            "step6_verdict": step6_verdict,
            "step6_fall_threshold_projected_gravity_z": None if step6_max is None else float(step6_max),
            "step6_fall_tilt_deg": None if step6_max is None else _tilt_deg_from_projected_gravity_z(float(step6_max)),
            "step6_warmup_steps": step6_warmup_steps,
            "step6_first_crossing_step": None if step6_cross is None else int(step6_cross),
            "fall_criteria_disagree": disagree,
        }

    def _log_both_fall_criteria(self, report: "dict[str, object]") -> None:
        """Print both fall verdicts for this rollout, with an extra line when they differ.

        The two criteria disagree between 45.573 and 60.000 degrees of tilt. Reporting only.
        """
        collector_deg = float(report["collector_fall_tilt_deg"])
        collector_thr = float(report["collector_fall_threshold_projected_gravity_z"])
        fall_step = report.get("fall_step")
        collector_line = (
            f"fell at recorded step {int(fall_step)}" if fall_step is not None else "no fall"
        )
        self.logger.warning(
            f"[collection] FALL CRITERIA: this collector (projected_gravity_z > {collector_thr:.3f}, "
            f"tilt > {collector_deg:.3f} deg, watched from step 0) says {collector_line}. "
            "This is the verdict the dataset and its post-fall count are based on."
        )
        step6_verdict = str(report["step6_verdict"])
        if step6_verdict == "unavailable":
            self.logger.warning(
                "[collection] FALL CRITERIA: step 6's criterion could not be evaluated for this "
                "run, so whether the pre/post-finetune comparison will call this rollout fallen "
                "is unknown here. Read the FALL CRITERIA lines printed by step 6 itself."
            )
            return
        step6_deg = float(report["step6_fall_tilt_deg"])
        step6_thr = float(report["step6_fall_threshold_projected_gravity_z"])
        crossing = report.get("step6_first_crossing_step")
        if step6_verdict == "fell":
            step6_line = f"would also call this fallen, from recorded step {int(crossing)}"
        elif step6_verdict == "indeterminate":
            step6_line = (
                f"first crossed at recorded step {int(crossing)}, which is inside the "
                f"{int(report['step6_warmup_steps'])}-step window step 6 ignores, so its verdict "
                "cannot be determined from here"
            )
        else:
            step6_line = "would NOT call this rollout fallen"
        self.logger.warning(
            f"[collection] FALL CRITERIA: step 6's comparison (projected_gravity_z > {step6_thr:.3f}, "
            f"tilt > {step6_deg:.3f} deg, first {int(report['step6_warmup_steps'])} steps ignored) "
            f"{step6_line}. The two criteria are deliberately not unified -- see "
            "plot_mocap_raw.STEP6_FALL_R22_MAX."
        )
        if not report.get("fall_criteria_disagree"):
            return
        if fall_step is not None:
            self.logger.warning(
                "[collection] FALL CRITERIA DISAGREE: this dataset is marked as containing a fall, "
                "but step 6 will draw no fall marker and will report tracking error and tracking "
                "return over the WHOLE run for the same rollout. Do not read that as a clean "
                "tracking measurement."
            )
        else:
            self.logger.warning(
                "[collection] FALL CRITERIA DISAGREE: step 6 will call this rollout fallen and "
                "truncate its exported tracking metrics, while this dataset carries no fall and "
                "the FADA IDM finetune will train on every step of it."
            )

    def _log_collection_fall_report(self) -> None:
        """State how many steps of the collected dataset were recorded after a fall.

        Printed on every `--task.collect-data` run.
        """
        # getattr-guarded: LocomotionPolicy_Deploy can also be constructed without __init__
        # in teardown-only paths.
        if self.data_collector is None or int(getattr(self, "_collection_steps", 0)) == 0:
            return
        report = self.collection_fall_report()
        # Also written next to the dataset as collection_fall_report.json.
        try:
            report_path = Path(self.data_collector.output_dir) / "collection_fall_report.json"
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
            self.logger.info(f"Collection fall report written to {report_path}")
        except Exception as exc:  # reporting must never break a finished run
            self.logger.warning(f"Could not write collection fall report: {exc}")
        self._log_both_fall_criteria(report)
        if not report["fall_detection_available"]:
            self.logger.warning(
                "[collection] fall detection was unavailable for this run; the collected dataset's "
                "post-fall step count is unknown. Inspect the rollout before finetuning on it."
            )
            return
        collected_steps = int(report["collected_steps"])
        post_fall_steps = int(report["post_fall_steps"])
        if post_fall_steps == 0:
            self.logger.info(
                f"[collection] no fall detected in {collected_steps} recorded steps "
                f"(projected_gravity_z stayed <= {self.collect_fall_projected_gravity_z_max:.2f})."
            )
            return
        fraction = 100.0 * post_fall_steps / max(1, collected_steps)
        signal = report.get("fall_signal") or "unknown"
        if report["marked_terminal"]:
            self.logger.warning(
                f"[collection] robot fell at recorded step {report['fall_step']} [{signal}]; "
                f"{post_fall_steps}/{collected_steps} steps ({fraction:.1f}%) are "
                "post-fall and were marked terminal, so the FADA finetune loader truncates the episode there."
            )
            return
        self.logger.warning(
            f"[collection] robot fell at recorded step {report['fall_step']} [{signal}]; "
            f"{post_fall_steps}/{collected_steps} steps ({fraction:.1f}%) of this "
            "dataset were recorded AFTER the fall and are written non-terminal (dones=False), so the FADA "
            "IDM finetune will train on them as ordinary windows. Discard this dataset, trim it with "
            "--trim-tail-steps, or re-collect with --task.collect-mark-fall-terminal true."
        )

    def _maybe_auto_start_policy(self):
        """Auto-start policy actions before entering the main loop if configured."""
        if getattr(self.config.task, "auto_start_policy", False) and not self.use_policy_action:
            self._handle_start_policy()

    def reset_runtime_state_for_new_session(self):
        """Reset base runtime state and per-session random-command scheduling."""
        super().reset_runtime_state_for_new_session()
        self.last_random_resample_iteration = -1
        self._initial_zero_command_hold_logged = False
        self._initial_zero_command_release_logged = False

    def _zero_out_commands(self):
        """Force the locomotion command triple to the standing zero-command state."""
        self.stand_command[0, 0] = 0.0
        self.lin_vel_command[0, :] = 0.0
        self.ang_vel_command[0, 0] = 0.0

    def _hold_initial_zero_random_commands(self, iteration: int) -> bool:
        """Keep random eval commands at zero during the configured warm-up window."""
        if not self.randomize_commands:
            return False
        if self.initial_zero_command_steps <= 0:
            return False
        if iteration < self.initial_zero_command_steps:
            self._zero_out_commands()
            if not self._initial_zero_command_hold_logged:
                self.logger.info(
                    "Holding random eval commands at zero for the first "
                    f"{self.initial_zero_command_steps} control steps."
                )
                self._initial_zero_command_hold_logged = True
            return True
        if not self._initial_zero_command_release_logged:
            self.logger.info(
                "Initial zero-command hold finished at iteration "
                f"{iteration}; random command sampling is now active."
            )
            self._initial_zero_command_release_logged = True
        return False

    def prepare_recorded_session(self, reset_outputs: bool):
        """Enable recording for the active session and reset per-session artifacts if requested."""
        self._set_session_recording_enabled(True)
        if reset_outputs:
            self._reset_session_output_buffers()
            if self.data_collector is not None and hasattr(self.data_collector, "reset_session"):
                self.data_collector.reset_session()
        if self.data_collector is not None:
            self.data_collector.start_episode()
            # Clear the sticky fall flag and step counters so they do not carry into the new
            # episode.
            self._collection_fall_latched = False
            self._collection_fall_detectable = True
            self._collection_steps = 0
            self._collection_post_fall_steps = 0
            self._collection_fall_step = None
            self._last_reset_time = time.perf_counter()
        self.eval_start_time = None
        self.max_time_exceeded_logged = False
        if self.max_eval_time > 0:
            self.eval_start_time = time.perf_counter()
            self.logger.info(f"Maximum evaluation time set to {self.max_eval_time:.2f} seconds")

    def prepare_unrecorded_session(self):
        """Disable per-session outputs while keeping the policy active."""
        self._set_session_recording_enabled(False)

    def _write_ready_signal(self):
        """Write optional ready signal file."""
        ready_file = getattr(self.config.task, "ready_signal_file", None)
        if not ready_file:
            return
        from pathlib import Path as _Path
        _Path(ready_file).parent.mkdir(parents=True, exist_ok=True)
        _Path(ready_file).write_text(str(time.time()))
        self.logger.info(f"Ready signal written to {ready_file}")

    def _wait_for_release_done_signal(self, timeout_s: float = 60.0):
        """Block the next sync STEP until the orchestrator has released the gantry."""
        release_file = getattr(self.config.task, "release_done_signal_file", None)
        if not release_file:
            return
        from pathlib import Path as _Path

        release_path = _Path(release_file)
        deadline = time.time() + float(timeout_s)
        while time.time() < deadline:
            if release_path.exists():
                self.logger.info(f"Release-done signal observed: {release_file}")
                return
            time.sleep(0.005)
        raise TimeoutError(
            f"Timed out waiting for release-done signal after {timeout_s:.1f}s: {release_file}"
        )

    def _write_done_signal(self):
        """Write optional done signal file."""
        done_file = getattr(self.config.task, "done_signal_file", None)
        if not done_file:
            return
        from pathlib import Path as _Path
        try:
            _Path(done_file).parent.mkdir(parents=True, exist_ok=True)
            _Path(done_file).write_text(str(time.time()))
            self.logger.info(f"Done signal written to {done_file}")
        except Exception:
            pass

    def _get_eval_elapsed(self, iteration: int) -> float:
        """Return elapsed evaluation time in seconds.

        Under sync stepping the wall clock is meaningless (slow inference
        inflates it).  Use *sim time* instead: ``iteration / rl_rate``.
        For real-time or ROS modes fall back to wall-clock.
        """
        from holosoma_inference.utils.rate import SimStepSyncRate

        if isinstance(getattr(self, "rate", None), SimStepSyncRate):
            return iteration / self.rl_rate
        if self.eval_start_time is not None:
            return time.perf_counter() - self.eval_start_time
        return 0.0

    def _should_stop(self, iteration: int) -> bool:
        """Check whether the main loop should terminate.

        Termination conditions (any one triggers exit):
        1. ``max_steps`` reached.
        2. ``exit_after_max_eval_time`` is True **and** ``max_eval_time`` exceeded.
        """
        # Condition 1: max_steps
        max_steps = getattr(self.config.task, "max_steps", -1)
        if max_steps is not None and max_steps >= 0 and (iteration + 1) >= max_steps:
            return True

        # Condition 2: max_eval_time exit
        if (
            getattr(self.config.task, "exit_after_max_eval_time", True)
            and self.max_eval_time > 0
            and self.eval_start_time is not None
        ):
            if self._get_eval_elapsed(iteration) >= self.max_eval_time:
                return True

        return False

    def _resolve_data_collection_output_dir(self) -> str:
        """Resolve where the collected H5 dataset is written.

        ``--task.data-collection-output-dir`` takes priority when explicitly set; otherwise
        collected data is colocated with everything else under the resolved log output dir
        (preserves pre-existing behavior for callers that never set the flag).
        """
        override = getattr(self.config.task, "data_collection_output_dir", None)
        if override:
            return override
        return self._resolve_log_output_dir()

    def _resolve_planned_collection_steps(self) -> "tuple[int | None, str]":
        """How many control steps this run intends to record, and where that number came from.

        Stamped into the collected H5 so an interrupted collection is distinguishable from a
        complete one. Consults the same two `_should_stop` conditions that end the loop, in
        the same priority order.

        Returns (None, reason) when neither bound is set (open-ended run).
        """
        max_steps = getattr(self.config.task, "max_steps", -1)
        if max_steps is not None and int(max_steps) > 0:
            return int(max_steps), "--task.max-steps"
        max_eval_time = float(getattr(self, "max_eval_time", -1.0) or -1.0)
        rl_rate = float(getattr(self, "rl_rate", 0.0) or 0.0)
        if getattr(self.config.task, "exit_after_max_eval_time", True) and max_eval_time > 0 and rl_rate > 0:
            return int(round(max_eval_time * rl_rate)), f"--task.max-eval-time {max_eval_time} x rl_rate {rl_rate}"
        return None, "no --task.max-steps and no exiting --task.max-eval-time: open-ended run"

    def _init_data_collector(self):
        """Initialize data collector if enabled."""
        try:
            # Import here to avoid circular dependencies
            import sys
            from pathlib import Path

            # Add holosoma to path if needed
            holosoma_path = Path(__file__).parent.parent.parent.parent / "holosoma" / "holosoma"
            if str(holosoma_path) not in sys.path:
                sys.path.insert(0, str(holosoma_path.parent.parent))

            from holosoma.utils.data_collector import DataCollector
            robot_type = getattr(self.robot_config, "robot_type", "unknown")
            # Detect simulator type based on network interface
            # "lo" (loopback) indicates simulation (sim2sim), otherwise real robot
            if self.config.task.interface == "lo":
                simulator_type = "mujoco"  # MuJoCo sim2sim
            else:
                simulator_type = "real"  # Real robot deployment

            obs_dict = self._build_data_collection_obs_dict()

            # Keep collected trajectories in the same folder as inference logs, unless the user
            # explicitly overrode this with --task.data-collection-output-dir.
            output_dir = Path(self._resolve_data_collection_output_dir())

            planned_steps, planned_steps_source = self._resolve_planned_collection_steps()

            self.data_collector = DataCollector(
                output_dir=str(output_dir),
                dataset_name=self.config.task.data_collection_dataset_name,
                compress=self.config.task.data_collection_compress,
                batch_size=10,
                robot_type=robot_type,
                simulator=simulator_type,
                policy_checkpoint=str(self.config.task.model_path) if isinstance(self.config.task.model_path, str) else None,
                num_envs=1,  # Deploy mode is always single environment
                obs_dict=obs_dict,
                skip_obs_keys=getattr(self.config.task, "data_collection_skip_obs_keys", ("actor_obs", "critic_obs")),
                planned_steps=planned_steps,
                planned_steps_source=planned_steps_source,
            )
            self.logger.info(f"Data collection enabled: {output_dir}/{self.config.task.data_collection_dataset_name}.h5")
        except Exception as e:
            # This method only runs when --task.collect-data was passed, so a failure here
            # means the H5 dataset cannot be produced. Raise rather than leaving
            # data_collector=None, so __init__ (and run_policy.py's top-level try/except)
            # exits non-zero.
            self.data_collector = None
            raise RuntimeError(
                f"--task.collect-data was requested but data collector initialization failed: {e}"
            ) from e

    def _validate_data_collection_output(self) -> None:
        """Verify the H5 dataset requested via --task.collect-data was actually produced.

        Must be called only after `self.data_collector.close()`. Raises if the collector's
        HDF5 file was not cleanly flushed/closed, was never written to disk, or if *this run*
        did not itself add at least one complete episode with at least one collected step.

        Checks `new_episode_count`/`new_step_count` (episodes/steps saved by this
        DataCollector instance) rather than `saved_episode_count` (the total episode count in
        the H5 file, which includes anything already there from a previous run), because
        appending to an existing dataset.h5 across multiple collection runs is supported.
        """
        from pathlib import Path

        collector = self.data_collector
        if collector is None:
            return
        h5_path = Path(collector.output_dir) / f"{collector.dataset_name}.h5"
        if getattr(collector, "h5_file", None) is not None:
            raise RuntimeError(
                f"Data collection HDF5 file was not cleanly flushed/closed: {h5_path}"
            )
        if not h5_path.exists():
            raise RuntimeError(f"Data collection produced no HDF5 file at: {h5_path}")
        saved_episode_count = getattr(collector, "saved_episode_count", 0)
        new_episode_count = getattr(collector, "new_episode_count", 0)
        new_step_count = getattr(collector, "new_step_count", 0)
        if new_episode_count < 1 or new_step_count < 1:
            raise RuntimeError(
                f"Data collection produced zero new episodes/steps in this run at {h5_path} "
                f"(new episodes: {new_episode_count}, new steps: {new_step_count}; "
                f"total episodes already in file: {saved_episode_count}). The run likely "
                "ended before any episode finished, or collect_step() was never called."
            )
        self.logger.info(
            f"Data collection verified: {h5_path} "
            f"({new_episode_count} new episode(s), {new_step_count} new step(s); "
            f"{saved_episode_count} total episode(s) in file)"
        )

    def rl_inference(self, robot_state_data, obs=None):
        """Perform RL inference, returning both raw and scaled actions."""
        if obs is None:
            obs = self.prepare_obs_for_rl(robot_state_data)
        if getattr(self.config.task, "print_observations", False):
            self._print_observations(obs)
        policy_action = self.policy(obs)
        policy_action = np.clip(policy_action, -100, 100)

        # raw (unscaled) action matches training/eval logging
        self.last_policy_action = policy_action.copy()
        # scaled action used for control
        self.scaled_policy_action = policy_action * self.policy_action_scale

        return self.scaled_policy_action, policy_action

    def _reset_recon_k_step_log(self):
        """Reset K-step recon running stats (call at start of run)."""
        self._recon_k_step_buffer = []
        self._recon_running_sum = 0.0
        self._recon_running_count = 0
        self._current_step_recon = None
        self._current_step_planner_first_step = None
        self._current_step_idm_inverse = None
        self._current_step_fdm_forward = None

    def _log_pending_dynamics_targets(self, obs_for_rl):
        """Log dynamics recon: use K-step MSE (all horizons) so log matches exit plot."""
        self._current_step_recon = None
        if self._pending_dynamics_prediction is None:
            return
        if obs_for_rl is None or "dynamics_obs" not in obs_for_rl:
            return

        actual = np.asarray(obs_for_rl["dynamics_obs"]).reshape(-1)
        predicted = np.asarray(self._pending_dynamics_prediction)
        if predicted.ndim == 1:
            predicted = predicted.reshape(1, -1)
        K = predicted.shape[0]

        # Append this prediction (from previous step) to buffer; keep last K only
        self._recon_k_step_buffer.append(predicted.copy())
        if len(self._recon_k_step_buffer) > K:
            self._recon_k_step_buffer.pop(0)
        # At step t we have actual_t; add all valid (pred[t-j, j] vs actual_t) for j=1..min(K, len(buffer))
        buf = self._recon_k_step_buffer
        for j in range(1, min(K, len(buf)) + 1):
            pred_j = buf[-j][j - 1]  # horizon j: prediction made j steps ago
            self._recon_running_sum += float(np.mean((pred_j - actual) ** 2))
            self._recon_running_count += 1

        recon_mse = (
            self._recon_running_sum / self._recon_running_count
            if self._recon_running_count > 0
            else 0.0
        )
        recon_rmse = float(np.sqrt(recon_mse))
        pred_store = predicted.reshape(1, -1).copy() if predicted.ndim == 1 else predicted.copy()
        self._current_step_recon = (actual.copy(), pred_store)
        if self.state_logger is not None and self._session_recording_enabled:
            self.state_logger.log_states(
                {
                    "actual_target": actual.copy(),
                    "predicted_target": predicted.copy(),
                    "reconstruction_loss": np.array([recon_mse], dtype=np.float32),
                    "obs_reconstruction_loss": np.array([recon_mse], dtype=np.float32),
                    "obs_reconstruction_rmse": np.array([recon_rmse], dtype=np.float32),
                }
            )
        self._pending_dynamics_prediction = None
        self._last_dynamics_obs_for_log = actual.copy()

    def _flush_pending_dynamics_log(self):
        """Ensure pending dynamics prediction is logged once (end-of-run alignment). Uses same K-step running MSE."""
        if self._pending_dynamics_prediction is None:
            return

        predicted = np.asarray(self._pending_dynamics_prediction)
        if predicted.ndim == 1:
            predicted = predicted.reshape(1, -1)
        actual = (
            np.asarray(self._last_dynamics_obs_for_log).reshape(-1)
            if self._last_dynamics_obs_for_log is not None
            else predicted[0:1].reshape(-1)
        )
        # Optionally add horizon-1 term for this last prediction (no future actual); keep running MSE as-is for log
        recon_mse = (
            self._recon_running_sum / self._recon_running_count
            if self._recon_running_count > 0
            else float(np.mean((predicted[0] - actual) ** 2))
        )
        recon_rmse = float(np.sqrt(recon_mse))
        if self.state_logger is not None and self._session_recording_enabled:
            self.state_logger.log_states(
                {
                    "actual_target": actual.copy(),
                    "predicted_target": predicted.copy(),
                    "reconstruction_loss": np.array([recon_mse], dtype=np.float32),
                    "obs_reconstruction_loss": np.array([recon_mse], dtype=np.float32),
                    "obs_reconstruction_rmse": np.array([recon_rmse], dtype=np.float32),
                }
            )
        self._pending_dynamics_prediction = None

    def policy_action(self):
        """Execute policy action and send commands to robot with data collection support."""
        # Snapshot flags to prevent races when dual-mode switching happens mid-cycle.
        use_policy = self.use_policy_action
        get_ready = self.get_ready_state

        kp_override = None
        kd_override = None
        scaled_policy_action = None
        raw_policy_action = None

        # Stage 1: Read State
        with self.latency_tracker.measure("read_state"):
            robot_state_data = self.interface.get_low_state()

        # Stage 2: Pre-processing
        with self.latency_tracker.measure("preprocessing"):
            # Determine target joint positions
            if get_ready:
                q_target = self.get_init_target(robot_state_data)
                self.init_count = min(self.init_count, 500)
            elif not use_policy:
                manual_cmd = self._get_manual_command(robot_state_data)
                if manual_cmd is not None:
                    q_target = manual_cmd["q"]
                    kp_override = manual_cmd.get("kp")
                    kd_override = manual_cmd.get("kd")
                else:
                    q_target = robot_state_data[:, 7 : 7 + self.num_dofs]
            else:
                # Prepare for inference - any preprocessing before RL inference
                pass

        obs_for_rl = None

        # Stage 3: Inference
        if use_policy and not get_ready:
            with self.latency_tracker.measure("inference"):
                obs_for_rl = self.prepare_obs_for_rl(robot_state_data)
                self._log_pending_dynamics_targets(obs_for_rl)
                if obs_for_rl is not None and "dynamics_obs" in obs_for_rl:
                    self._last_dynamics_obs_for_log = np.asarray(obs_for_rl["dynamics_obs"]).copy()
                inference_result = self.rl_inference(robot_state_data, obs_for_rl)
                if isinstance(inference_result, (list, tuple)) and len(inference_result) == 3:
                    scaled_policy_action, raw_policy_action, dynamics_prediction = inference_result
                else:
                    scaled_policy_action, raw_policy_action = inference_result
                    dynamics_prediction = None
                self._pending_dynamics_prediction = dynamics_prediction

        # Stage 4: Post-processing
        with self.latency_tracker.measure("postprocessing"):
            if use_policy and not get_ready:
                if scaled_policy_action.shape[1] != self.num_dofs:
                    if not self.upper_body_controller:
                        scaled_policy_action = np.concatenate(
                            [np.zeros((1, self.num_dofs - scaled_policy_action.shape[1])), scaled_policy_action], axis=1
                        )
                    else:
                        raise NotImplementedError("Upper body controller not implemented")
                q_target = scaled_policy_action + self.default_dof_angles
                q_target = self._apply_residual_upper_body(q_target, scaled_policy_action)

            # Prepare command (reuse pre-allocated arrays)
            self.cmd_q[:] = q_target[0]

        # Stage 5: Data Collection (if enabled) - BEFORE sending command to ensure (s_t, a_t) pairing
        # Collect current state s_t and action a_t before executing the action
        # This ensures we collect (s_t, a_t) not (s_{t+1}, a_t)
        if (
            self.data_collector is not None
            and use_policy
            and scaled_policy_action is not None
            and self._session_recording_enabled
        ):
            try:
                # Only forward the observation groups that the data collector will actually save,
                # and keep them as numpy arrays to avoid a redundant numpy->torch->numpy round-trip.
                collector_obs_keys = tuple(
                    getattr(self.data_collector, "obs_dict", {}).keys()
                ) or tuple(self.obs_buf_dict.keys())
                obs_dict_for_collection = {
                    k: v for k, v in self.obs_buf_dict.items() if k in collector_obs_keys
                }

                actions_for_collection = raw_policy_action
                if not isinstance(actions_for_collection, (np.ndarray, torch.Tensor)):
                    actions_for_collection = np.asarray(actions_for_collection)

                self.data_collector.collect_step(
                    obs_dict=obs_dict_for_collection,
                    actions=actions_for_collection,
                    # Inference has no environment termination signal, so the fall is
                    # detected here from projected gravity; see _collection_done_flag().
                    dones=self._collection_done_flag(robot_state_data),
                    rewards=None,  # Not available in inference
                )
            except Exception as e:
                self.logger.warning(f"Error during data collection: {e}")

        # Stage 5.5: Unified log (mocap raw + command + recon) or state log. Command and recon use the same timestamp (same policy step).
        proc = getattr(self.interface, "vel_state_processor", None)
        if proc is not None and use_policy and not get_ready:
            # Pull latest mocap so command can use same time base (MuJoCo: sim_time → zero error; real: wall clock).
            if hasattr(self.interface, "get_pose_state"):
                self.interface.get_pose_state()
            ts = time.time()
            if getattr(proc, "get_last_mocap_timestamp", None) is not None:
                t_mocap = proc.get_last_mocap_timestamp()
                if t_mocap is not None:
                    ts = t_mocap
            if getattr(proc, "record_command", None):
                proc.record_command(ts, float(self.lin_vel_command[0, 0]), float(self.lin_vel_command[0, 1]), float(self.ang_vel_command[0, 0]))
            if getattr(proc, "record_reconstruction", None) and self._current_step_recon is not None:
                proc.record_reconstruction(ts, self._current_step_recon[0], self._current_step_recon[1])
            if (
                getattr(proc, "record_planner_first_step_obs", None)
                and self._current_step_planner_first_step is not None
            ):
                proc.record_planner_first_step_obs(
                    ts,
                    self._current_step_planner_first_step[0],
                    self._current_step_planner_first_step[1],
                )
            if getattr(proc, "record_idm_inverse", None) and self._current_step_idm_inverse is not None:
                proc.record_idm_inverse(ts, self._current_step_idm_inverse[0], self._current_step_idm_inverse[1])
            if getattr(proc, "record_fdm_forward", None) and self._current_step_fdm_forward is not None:
                proc.record_fdm_forward(ts, self._current_step_fdm_forward[0], self._current_step_fdm_forward[1])
            self._current_step_recon = None
            self._current_step_planner_first_step = None
            self._current_step_idm_inverse = None
            self._current_step_fdm_forward = None
        self._maybe_log_inference_state(robot_state_data, q_target, obs_for_rl)

        # Stage 6: Action Pub
        with self.latency_tracker.measure("action_pub"):
            self.interface.send_low_command(
                self.cmd_q,
                self.cmd_dq,
                self.cmd_tau,
                robot_state_data[0, 7 : 7 + self.num_dofs],
                kp_override=kp_override,
                kd_override=kd_override,
            )

    def _before_run_loop(self):
        """Prepare logging, timers, and command sequence for standalone execution."""
        self._reset_recon_k_step_log()
        self.prepare_unrecorded_session()
        self._prime_sync_rate_before_recorded_session()
        self._maybe_auto_start_policy()
        self._ready_signal_written = False

    def _prime_sync_rate_before_recorded_session(self):
        """Move sync sim past handshake/settle before the first recorded policy action."""
        from holosoma_inference.utils.rate import SimStepSyncRate

        if not isinstance(getattr(self, "rate", None), SimStepSyncRate):
            return
        if getattr(self, "_sync_rate_primed_before_session", False):
            return

        self.logger.info(
            "Priming SimStepSyncRate before recorded session with two sync batches "
            "so first policy observation is post-sync-settle, not Phase-1 warm-up state."
        )
        if hasattr(self.interface, "reset_low_state_cache"):
            self.interface.reset_low_state_cache()
        self.rate.sleep()
        if hasattr(self.interface, "reset_low_state_cache"):
            self.interface.reset_low_state_cache()
        self.rate.sleep()
        if hasattr(self.interface, "wait_for_low_state_tick_at_most"):
            self.interface.wait_for_low_state_tick_at_most(max_tick_ms=100, timeout=2.0)
        self._sync_rate_primed_before_session = True

        # The sync prime advances the simulator to a deterministic post-settle
        # state. Drop any policy-side warm-up/history state gathered before
        # that point so iteration 0 starts from the same logical state each run.
        self.reset_runtime_state_for_new_session()
        self.prepare_unrecorded_session()

    def _run_iteration(self, it: int) -> bool:
        """Execute one deploy loop iteration and return True when the standalone run should stop."""
        self.latency_tracker.start_cycle()

        max_time_exceeded = False
        if self.max_eval_time > 0 and self.eval_start_time is not None:
            elapsed_time = self._get_eval_elapsed(it)
            if elapsed_time >= self.max_eval_time:
                max_time_exceeded = True
                self.lin_vel_command[0, :] = 0.0
                self.ang_vel_command[0, 0] = 0.0
                self.stand_command[0, 0] = 0.0
                if not self.max_time_exceeded_logged:
                    self.logger.warning(
                        f"Maximum evaluation time ({self.max_eval_time:.2f}s) exceeded. "
                        f"Commands set to zero. Elapsed: {elapsed_time:.2f}s"
                    )
                    self.max_time_exceeded_logged = True

        if not max_time_exceeded:
            if self.use_joystick and self.interface.get_joystick_msg() is not None:
                self.process_joystick_input()
            if getattr(self, "_skip_current_iteration", False):
                self._skip_current_iteration = False
                self.latency_tracker.end_cycle()
                return False
            self._maybe_resample_commands(it)
        if self.use_phase:
            self.update_phase_time()
        if getattr(self, "_skip_current_iteration", False):
            self._skip_current_iteration = False
            self.latency_tracker.end_cycle()
            return False

        self.policy_action()

        if (
            self.use_policy_action
            and not self.get_ready_state
            and not getattr(self, "_ready_signal_written", False)
        ):
            # Start session recording and signal "ready" only after one completed
            # policy-action cycle, so the parent does not release the gantry before commands
            # are live.
            self.prepare_recorded_session(reset_outputs=True)
            self._write_ready_signal()
            self._wait_for_release_done_signal()
            self._ready_signal_written = True

        self.latency_tracker.end_cycle()

        if it % 50 == 0 and self.use_policy_action:
            debug_str = f"RL FPS: {self.latency_tracker.get_fps():.2f} | {self.latency_tracker.get_stats_str()}"
            self.logger.info(debug_str, flush=True)

        return self._should_stop(it)

    def _after_run_loop(self):
        """Finalize standalone deploy execution."""
        # Flush HDF5 before the pipeline sees the done signal (it polls done, then waits on the process).
        collection_error: Exception | None = None
        if self.data_collector is not None:
            try:
                self.data_collector.close()
                self.logger.info("Data collection completed")
                self._log_collection_fall_report()
                self._validate_data_collection_output()
            except Exception as exc:  # noqa: BLE001 - re-raised below, after teardown/signals still run
                self.logger.error(f"Data collection validation failed: {exc}")
                collection_error = exc
        # Delegate rate.close() and state_logger flush/save to the parent.
        super()._after_run_loop()
        self._write_done_signal()
        if collection_error is not None:
            # --task.collect-data was explicitly requested; a run that "completes" without a
            # valid H5 dataset must not exit 0 (see _init_data_collector for the same contract
            # on the init-failure path).
            raise RuntimeError(
                f"--task.collect-data requested but data collection validation failed: {collection_error}"
            ) from collection_error


    # ============================================================================
    # Command Randomization Helpers
    # ============================================================================

    def process_joystick_input(self):
        """Process joystick input."""
        super().process_joystick_input()

    def _maybe_resample_commands(self, iteration: int):
        """Resample commands on a fixed interval if randomization is enabled."""
        if not self.randomize_commands or self.command_resample_steps is None:
            return
        if self._hold_initial_zero_random_commands(iteration):
            return
        if self.last_random_resample_iteration < 0:
            self._resample_commands()
            self.last_random_resample_iteration = iteration
            return
        if iteration - self.last_random_resample_iteration < self.command_resample_steps:
            return
        self._resample_commands()
        self.last_random_resample_iteration = iteration

    def _resample_commands(self):
        """Sample new commands similar to train/eval command manager."""
        if np.random.rand() < self.command_stand_prob:
            self.stand_command[0, 0] = 0
            self.lin_vel_command[0, :] = 0.0
            self.ang_vel_command[0, 0] = 0.0
            return

        self.stand_command[0, 0] = 1
        self.lin_vel_command[0, 0] = np.random.uniform(*self.command_lin_vel_x_range)
        self.lin_vel_command[0, 1] = np.random.uniform(*self.command_lin_vel_y_range)
        self.ang_vel_command[0, 0] = np.random.uniform(*self.command_ang_vel_range)

    def get_current_obs_buffer_dict(self, robot_state_data):
        current_obs_buffer_dict = super().get_current_obs_buffer_dict(robot_state_data)
        current_obs_buffer_dict["actions"] = self.last_policy_action
        current_obs_buffer_dict["command_lin_vel"] = self.lin_vel_command
        current_obs_buffer_dict["command_ang_vel"] = self.ang_vel_command
        current_obs_buffer_dict["command_stand"] = self.stand_command

        # Add phase observations only if they are configured
        if "sin_phase" in self.obs_dict.get("actor_obs", []):
            current_obs_buffer_dict["sin_phase"] = self._get_obs_sin_phase()
        if "cos_phase" in self.obs_dict.get("actor_obs", []):
            current_obs_buffer_dict["cos_phase"] = self._get_obs_cos_phase()

        return current_obs_buffer_dict

    def _get_obs_sin_phase(self):
        """Calculate sin phase for gait."""
        return np.array([np.sin(self.phase[0, :])])

    def _get_obs_cos_phase(self):
        """Calculate cos phase for gait."""
        return np.array([np.cos(self.phase[0, :])])

    def update_phase_time(self):
        """Update phase time."""
        phase_tp1 = self.phase + self.phase_dt
        self.phase = np.fmod(phase_tp1 + np.pi, 2 * np.pi) - np.pi
        if np.linalg.norm(self.lin_vel_command[0]) < 0.01 and np.linalg.norm(self.ang_vel_command[0]) < 0.01:
            # Robot should stand still - set both feet to same phase
            self.phase[0, :] = np.pi * np.ones(2)
            self.is_standing = True
        elif self.is_standing:
            # When the robot starts to move, reset the phase to initial state
            self.phase = np.array([[0.0, np.pi]])
            self.is_standing = False

    def handle_keyboard_button(self, keycode):
        """Handle keyboard button presses for locomotion."""
        # Call parent handler for common commands
        super().handle_keyboard_button(keycode)

        # Locomotion-specific commands
        if keycode in ["w", "s", "a", "d"]:
            self._handle_velocity_control(keycode)
        elif keycode in ["q", "e"]:
            self._handle_angular_velocity_control(keycode)
        elif keycode == "=":
            self._handle_stand_command()
        elif keycode == "z":
            self._handle_zero_velocity()

        self._print_control_status()

    def handle_joystick_button(self, cur_key):
        """Handle joystick button presses for locomotion."""
        # Call parent handler for common commands
        super().handle_joystick_button(cur_key)

        # Locomotion-specific commands
        if cur_key == "start":
            self._handle_stand_command()
        elif cur_key == "L2":
            self._handle_zero_velocity()

    def _handle_velocity_control(self, keycode):
        """Handle linear velocity control."""
        if getattr(self, "randomize_commands", False):
            return
        if not self.stand_command[0, 0]:
            return

        if keycode == "w":
            self.lin_vel_command[0, 0] += 0.1
        elif keycode == "s":
            self.lin_vel_command[0, 0] -= 0.1
        elif keycode == "a":
            self.lin_vel_command[0, 1] += 0.1
        elif keycode == "d":
            self.lin_vel_command[0, 1] -= 0.1

    def _handle_angular_velocity_control(self, keycode):
        """Handle angular velocity control."""
        if getattr(self, "randomize_commands", False):
            return
        if keycode == "q":
            self.ang_vel_command[0, 0] -= 0.1
        elif keycode == "e":
            self.ang_vel_command[0, 0] += 0.1

    def _handle_stand_command(self):
        """Handle stand command toggle."""
        if getattr(self, "randomize_commands", False):
            return
        self.stand_command[0, 0] = 1 - self.stand_command[0, 0]
        if self.stand_command[0, 0] == 0:
            self.ang_vel_command[0, 0] = 0.0
            self.lin_vel_command[0, 0] = 0.0
            self.lin_vel_command[0, 1] = 0.0
            self.logger.info(colored("Stance command", "blue"))
        else:
            self.base_height_command[0, 0] = self.desired_base_height
            self.logger.info(colored("Walk command", "blue"))

    def _handle_zero_velocity(self):
        """Handle zero velocity command."""
        if getattr(self, "randomize_commands", False):
            return
        self.ang_vel_command[0, 0] = 0.0
        self.lin_vel_command[0, 0] = 0.0
        self.lin_vel_command[0, 1] = 0.0
        self.logger.info(colored("Velocities set to zero", "blue"))

    def _print_control_status(self):
        """Print current control status."""
        super()._print_control_status()

        # Extract values for better formatting
        lin_vel_x = self.lin_vel_command[0, 0]
        lin_vel_y = self.lin_vel_command[0, 1]
        ang_vel_z = self.ang_vel_command[0, 0]
        is_walking = self.stand_command[0, 0] == 1

        # Print with clear labels and units
        mode = "Walking" if is_walking else "Standing"
        status = "✓ applied" if is_walking else "✗ not applied"
        print(f"Linear velocity: x={lin_vel_x:+.2f} m/s, y={lin_vel_y:+.2f} m/s")
        print(f"Angular velocity: {ang_vel_z:+.2f} rad/s")
        print(f"Mode: {mode} ({status})")
        print("💡 Terminal keys: W/A/S/D (lin) | Q/E (ang) | = (toggle mode)")
        print("🎬 MuJoCo keys (in simulator only): 7/8 (band) | 9 (toggle) | BACKSPACE (reset)")

