from __future__ import annotations

import hashlib
import itertools
import json
import os
import sys
import threading
import time
from collections import deque
from dataclasses import replace
from pathlib import Path

import netifaces as ni
import numpy as np
import onnx
import onnxruntime
import torch
from loguru import logger
from sshkeyboard import listen_keyboard
from termcolor import colored

from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.config.config_types.robot import RobotConfig
from holosoma_inference.sdk import create_interface
from holosoma_inference.utils.inference_logger import InferenceLogger
from holosoma_inference.utils.latency import LatencyTracker
from holosoma_inference.utils.math.quat import quat_rotate_inverse
from holosoma_inference.utils.rate import RateLimiter
from holosoma_inference.utils.sync_rendezvous import DEFAULT_SIM_STEP_SYNC_URL, sync_endpoint_has_listener
from holosoma_inference.utils.wandb import load_checkpoint

TRANSFORMER_FINETUNE_DYNAMICS_TERMS = (
    "base_ang_vel",
    "dof_pos",
    "dof_vel",
    "projected_gravity",
)
TRANSFORMER_FINETUNE_COMMAND_TERMS = (
    "command_lin_vel",
    "command_ang_vel",
    "sin_phase",
    "cos_phase",
)

_TRT_PREWARM_MANIFEST_NAME = "_trt_prewarm_manifest.json"


def _file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _first_cuda_visible_device() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return "default"
    first = visible.split(",", 1)[0].strip()
    return first or "default"


class BasePolicy:
    """
    Base policy class for Holosoma deployment on humanoid robots.

    Supports both simulation and real robot deployment with keyboard/joystick controls.
    """

    def __init__(self, config: InferenceConfig):
        """Initialize the base policy with configuration and model."""
        self.config = config
        self._configure_runtime_threads()
        # Initialize robot config
        self._init_robot_config(self.config.robot)
        # Initialize SDK components (ChannelFactory, etc. — no ChangeMode yet)
        self._init_sdk_components()
        # Initialize observation config (needed by setup_policy for obs_scales)
        self._init_obs_config()
        # Load policy ONNX / build TRT engines BEFORE communication init,
        # so that ChangeMode(kCustom) happens only after policy is ready.
        self._init_policy_components(
            self.config.task.model_path, self.config.task.policy_action_scale, self.config.task.rl_rate
        )
        # Initialize communication components (creates interface, triggers ChangeMode).
        # Must be after _init_policy_components so robot enters custom mode
        # only when the policy is fully loaded and warmed up.
        self._init_communication_components()
        # Wait for the first valid robot state message before proceeding.
        # Because policy loading happens before ChangeMode, the DDS subscriber
        # has had very little time to receive data; poll until it arrives.
        self._wait_for_first_robot_state()
        # Propagate ONNX KP/KD to interface (needs both policy and interface ready)
        self._resolve_control_gains()
        # Initialize command components
        self._init_command_components()
        # Initialize input handlers
        self._init_input_handlers()
        # Initialize phase components
        self._init_phase_components()
        # Initialize latency tracking
        self._init_latency_tracking()
        # Initialize optional inference logging
        self._init_logging_components()

    # ============================================================================
    # Initialization Methods
    # ============================================================================

    def _init_robot_config(self, robot_config: RobotConfig):
        """Initialize robot configuration and parameters."""
        self.robot_config = robot_config
        self.num_dofs = self.robot_config.num_joints
        self.default_dof_angles = np.array(self.robot_config.default_dof_angles)
        # Setup dof names and indices
        self._setup_dof_mappings()

    def _setup_dof_mappings(self):
        """Setup DOF names and their corresponding indices."""
        self.dof_names = self.robot_config.dof_names
        self.lower_dof_names = self.robot_config.dof_names_lower_body

        # These are used by derived classes, so keep them
        if self.lower_dof_names:
            self.lower_dof_indices = [self.dof_names.index(dof) for dof in self.lower_dof_names]
        else:
            self.lower_dof_indices = []

    def _init_sdk_components(self):
        """Additional SDK components initialization based on robot type."""
        if hasattr(self, "_shared_hardware_source"):
            self.sdk_type = self._shared_hardware_source.sdk_type
            return
        self.sdk_type = self.robot_config.sdk_type
        if self.sdk_type == "booster":
            from booster_robotics_sdk import ChannelFactory

            ip = ni.ifaddresses(self.config.task.interface)[ni.AF_INET][0]["addr"]
            ChannelFactory.Instance().Init(self.config.task.domain_id, ip)
        else:
            pass  # No channel initialization needed for Unitree binding / other robots

    def _init_obs_config(self):
        """Initialize observation metadata and history buffers."""
        self.obs_config = self.config.observation
        self.obs_scales = self.obs_config.obs_scales
        self.obs_dims = self.obs_config.obs_dims
        self.obs_dict = self.obs_config.obs_dict
        self.obs_dim_dict = self._calculate_obs_dim_dict()
        self.history_length_dict = self.obs_config.history_length_dict

        # Initialize per-term history buffers using deques
        self._initialize_history_state()

    def _initialize_history_state(self):
        """Create per-term history deques and zero-initialized flattened buffers."""
        self.obs_history_buffers: dict[str, dict[str, deque[np.ndarray]]] = {}
        self.obs_terms_sorted: dict[str, list[str]] = {}
        self.obs_buf_dict: dict[str, np.ndarray] = {}

        for group, term_names in self.obs_dict.items():
            self.obs_terms_sorted[group] = sorted(term_names)
            history_len = self.history_length_dict.get(group, 1)
            self.obs_history_buffers[group] = {}
            flattened_terms: list[np.ndarray] = []

            for term in self.obs_terms_sorted[group]:
                term_dim = self.obs_dims[term]
                self.obs_history_buffers[group][term] = deque(maxlen=history_len)
                flattened_terms.append(np.zeros((1, term_dim * history_len), dtype=np.float32))

            self.obs_buf_dict[group] = np.concatenate(flattened_terms, axis=1) if flattened_terms else np.zeros((1, 0))

    def _init_communication_components(self):
        """Initialize appropriate robot interface."""
        if hasattr(self, "_shared_hardware_source"):
            self.interface = self._shared_hardware_source.interface
            return
        logger.info("Initializing robot interface (DDS/SDK)...")
        self.interface = create_interface(
            self.robot_config,
            self.config.task.domain_id,
            self.config.task.interface,
            self.config.task.use_joystick,
            task_config=self.config.task,
        )
        logger.info("Robot interface initialized.")

    def _wait_for_first_robot_state(self, timeout: float = 5.0, poll_interval: float = 0.01):
        """Block until the first valid robot state arrives from the interface.

        Policy loading happens before ChangeMode, so the DDS subscriber may not
        have received any state yet.  This polls ``get_low_state()`` until it
        returns a non-None array, or raises after *timeout* seconds.
        """
        import time as _time

        deadline = _time.monotonic() + timeout
        while _time.monotonic() < deadline:
            state = self.interface.get_low_state()
            if state is not None:
                logger.info("First robot state received — interface ready.")
                return
            _time.sleep(poll_interval)
        raise RuntimeError(
            f"No robot state received within {timeout}s after ChangeMode. "
            "Check that the robot is powered on and the DDS network is reachable."
        )

    def _init_policy_components(self, model_path, policy_action_scale, rl_rate):
        """Initialize policy-related components."""
        self.policy_action_scale = policy_action_scale
        self.rl_rate = rl_rate
        self.model_paths = self._collect_model_paths(model_path)
        self._policy_states: list[dict] = []
        self.last_policy_action = np.zeros((1, self.num_dofs))
        self.scaled_policy_action = np.zeros((1, self.num_dofs))
        resolved_paths: list[str] = []

        for path in self.model_paths:
            local_path = self._resolve_model_path(str(path))
            resolved_paths.append(local_path)
            self.setup_policy(local_path)
            self._policy_states.append(self._capture_policy_state())

        self.model_paths = resolved_paths
        self.active_policy_index = 0
        self.active_model_path = None
        self._activate_policy(0, announce=False)

    def _collect_model_paths(self, model_path):
        """Normalize model_path into a list of up to nine entries."""
        if isinstance(model_path, (list, tuple)):
            paths = list(model_path)
        elif model_path is not None:
            paths = [model_path]
        else:
            paths = []

        paths = [str(path) for path in paths if path]
        if not paths:
            raise ValueError("At least one model_path must be provided for policy initialization.")
        if len(paths) > 9:
            # Error out instead of warning
            raise ValueError("Received more than nine model paths. Only up to nine model paths are supported.")
        return paths

    def _resolve_model_path(self, model_path: str) -> str:
        """Resolve model path, downloading from W&B if required."""
        if model_path.startswith(("wandb://", "https://")):
            download_dir = self.config.task.wandb_download_dir
            logger.info(f"Downloading checkpoint from W&B: {model_path}")
            checkpoint_path = load_checkpoint(None, model_path, download_dir)
            resolved_path = str(checkpoint_path)
            logger.info("Checkpoint downloaded to: %s", resolved_path)
            return resolved_path
        return model_path

    def _capture_policy_state(self) -> dict:
        """Capture the current policy state for later reuse."""
        return {
            "onnx_policy_session": self.onnx_policy_session,
            "onnx_input_names": self.onnx_input_names,
            "onnx_output_names": self.onnx_output_names,
            "policy_callable": self.policy,
            "onnx_kp": self.onnx_kp,
            "onnx_kd": self.onnx_kd,
        }

    def _restore_policy_state(self, state: dict):
        """Restore a previously captured policy state."""
        self.onnx_policy_session = state["onnx_policy_session"]
        self.onnx_input_names = state["onnx_input_names"]
        self.onnx_output_names = state["onnx_output_names"]
        self.policy = state["policy_callable"]
        self.onnx_kp = state["onnx_kp"]
        self.onnx_kd = state["onnx_kd"]

    def _activate_policy(self, index: int, announce: bool = True):
        """Activate a preloaded policy."""
        if not (0 <= index < len(self.model_paths)):
            return

        self._restore_policy_state(self._policy_states[index])
        self.last_policy_action.fill(0.0)
        self.scaled_policy_action.fill(0.0)
        self.active_policy_index = index
        self.active_model_path = self.model_paths[index]
        self._on_policy_switched(self.active_model_path)

        if announce and len(self.model_paths) > 1 and hasattr(self, "logger"):
            name = Path(self.active_model_path).name
            self.logger.info(colored(f"Switched to policy [{index + 1}]: {name}", "blue"))

    def _try_switch_policy_key(self, keycode: str) -> bool:
        """Switch policy slot if a numeric key is pressed."""
        if len(self.model_paths) <= 1:
            return False
        if not keycode.isdigit():
            return False
        slot = int(keycode)
        if slot == 0:
            return False
        index = slot - 1
        if index == self.active_policy_index:
            return True
        if 0 <= index < len(self.model_paths):
            self._activate_policy(index)
            return True
        return False

    def _on_policy_switched(self, model_path: str):
        """Hook for derived classes to reset state after loading a new policy."""
        _ = model_path

    def _init_command_components(self):
        """Initialize control-related components and commands."""
        self.use_policy_action = False
        self.init_count = 0
        self.get_ready_state = False
        self.desired_base_height = self.config.task.desired_base_height
        self.gait_period = self.config.task.gait_period

        # Initialize command arrays
        self.lin_vel_command = np.array([[0.0, 0.0]])
        self.ang_vel_command = np.array([[0.0]])
        self.stand_command = np.array([[0]])
        self.base_height_command = np.array([[self.desired_base_height]])

        # These are used by derived classes, so keep them
        self.waist_dofs_command = np.zeros((1, 3))
        self.phase_time = np.zeros((1, 1))

        # Upper body controller
        self.upper_body_controller = None

        # Pre-allocate command arrays for postprocessing
        self.cmd_q = np.zeros(self.num_dofs)
        self.cmd_dq = np.zeros(self.num_dofs)
        self.cmd_tau = np.zeros(self.num_dofs)

    def _init_phase_components(self):
        """Initialize phase components."""
        self.use_phase = self.config.task.use_phase
        if self.use_phase:
            self.phase = np.zeros((1, 2))
            self.phase[:, 0] = 0.0  # left foot starts at 0
            self.phase[:, 1] = np.pi  # right foot starts at pi
            self.phase_dt = 2 * np.pi / (self.rl_rate * self.gait_period)

    def _init_latency_tracking(self):
        """Initialize latency tracking components."""
        self.latency_tracker = LatencyTracker(window_size=int(self.rl_rate))

    def _init_logging_components(self):
        """Initialize optional inference logger. When plot_mocap_raw_on_exit is True, use unified log only (no state_log)."""
        self.state_logger: InferenceLogger | None = None
        self._session_recording_enabled = True
        self._pending_dynamics_prediction = None
        self._missing_truth_vel_warned = False
        self._last_dynamics_obs_for_log = None
        self._current_step_recon = None  # (actual, predicted) aligned pair; record in Stage 5.5 with command timestamp
        self._current_step_planner_first_step = None  # (actual_obs, predicted_obs) aligned pair for planner first-step MSE
        self._current_step_idm_inverse = None  # (actual_action_chunk, predicted_action_chunk) aligned pair for IDM MSE
        self._current_step_fdm_forward = None  # (actual_obs_chunk, predicted_obs_chunk) aligned pair for FDM MSE
        # K-step recon: running MSE over all horizons (same metric as exit plot)
        self._recon_k_step_buffer: list = []  # last K predictions, each (K, dim)
        self._recon_running_sum = 0.0
        self._recon_running_count = 0
        use_unified_only = getattr(self.config.task, "plot_mocap_raw_on_exit", False)
        if use_unified_only:
            return
        if getattr(self.config.task, "log_states", False):
            output_dir = self._resolve_log_output_dir()
            filename = getattr(self.config.task, "log_filename", "state_log")
            self.state_logger = InferenceLogger(output_dir, filename)
        self._set_session_recording_enabled(True)

    def _resolve_log_base_output_dir(self) -> str:
        """Resolve base log directory; 'auto' places logs next to checkpoint."""
        cfg_dir = getattr(self.config.task, "log_output_dir", "logs/inference")
        if cfg_dir != "auto":
            return cfg_dir

        exp_name = getattr(self.config.task, "log_exp_name", None)
        # Use active model path if available
        model_path = getattr(self, "active_model_path", None)
        if model_path:
            base_dir = Path(model_path).parent
            if exp_name:
                return str(base_dir / "inference" / exp_name)
            return str(base_dir / "inference")

        # Fallback
        return "logs/inference"

    def _resolve_log_output_dir(self) -> str:
        """Resolve run log directory with a stable per-run timestamp subdir."""
        cached = getattr(self, "_resolved_log_output_dir", None)
        if isinstance(cached, str) and cached:
            return cached

        base_dir = Path(self._resolve_log_base_output_dir())
        run_timestamp = getattr(self, "_log_run_timestamp", None)
        if not isinstance(run_timestamp, str) or run_timestamp == "":
            run_timestamp = time.strftime("%Y%m%d_%H%M%S")
            self._log_run_timestamp = run_timestamp

        resolved = str(base_dir / run_timestamp)
        self._resolved_log_output_dir = resolved
        return resolved

    def _set_session_recording_enabled(self, enabled: bool):
        """Enable or disable session-scoped logging / mocap recording."""
        self._session_recording_enabled = bool(enabled)
        proc = getattr(getattr(self, "interface", None), "vel_state_processor", None)
        if proc is not None and hasattr(proc, "set_recording_enabled"):
            proc.set_recording_enabled(self._session_recording_enabled)

    def _reset_session_output_buffers(self):
        """Clear in-memory outputs so the next session starts from a clean slate."""
        if self.state_logger is not None and hasattr(self.state_logger, "reset_session"):
            self.state_logger.reset_session()
        proc = getattr(getattr(self, "interface", None), "vel_state_processor", None)
        if proc is not None and hasattr(proc, "reset_session_history"):
            proc.reset_session_history()
        # Flush any stale ZMQ truth messages (e.g. from Phase 1 warm-up in sync
        # stepping) so they don't contaminate the upcoming recording session.
        if proc is not None and hasattr(proc, "flush"):
            proc.flush()

    def reset_runtime_state_for_new_session(self):
        """Reset observation, command, phase, and model runtime buffers for a fresh session."""
        self._initialize_history_state()
        self._init_command_components()
        self._init_phase_components()
        self._init_latency_tracking()
        self.last_policy_action = np.zeros((1, self.num_dofs))
        self.scaled_policy_action = np.zeros((1, self.num_dofs))
        self._pending_dynamics_prediction = None
        self._missing_truth_vel_warned = False
        self._last_dynamics_obs_for_log = None
        self._current_step_recon = None
        self._current_step_planner_first_step = None
        self._current_step_idm_inverse = None
        self._current_step_fdm_forward = None
        self._reset_recon_k_step_log()
        if hasattr(self, "_initial_base_pos"):
            delattr(self, "_initial_base_pos")
        if hasattr(self, "_initial_base_quat_xyzw"):
            delattr(self, "_initial_base_quat_xyzw")
        if getattr(self, "active_model_path", None):
            self._on_policy_switched(self.active_model_path)

    def prepare_recorded_session(self, reset_outputs: bool):
        """Enable recording for a fresh session."""
        self._set_session_recording_enabled(True)
        if reset_outputs:
            self._reset_session_output_buffers()

    def prepare_unrecorded_session(self):
        """Disable recording for the current session."""
        self._set_session_recording_enabled(False)

    def _init_input_handlers(self):
        """Initialize input handlers (ROS, joystick, keyboard)."""
        if hasattr(self, "_shared_hardware_source"):
            self.logger = self._shared_hardware_source.logger
            self.rate = self._shared_hardware_source.rate
            self.rl_rate = self._shared_hardware_source.rl_rate
            self.use_joystick = self._shared_hardware_source.use_joystick
            return
        self._init_rate_handler()
        self._init_input_device()

    def _init_rate_handler(self):
        """Initialize rate handler (ROS, wall-clock, or sim-sync)."""
        self.rl_rate = self.config.task.rl_rate
        if self.config.task.use_ros:
            import rclpy

            rclpy.init(args=None)
            self.node = rclpy.create_node("policy_node")
            self.logger = self.node.get_logger()
            self.rate = self.node.create_rate(self.rl_rate)
            thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
            thread.start()
        else:
            self.logger = logger
            sync_url = getattr(self.config.task, "sim_step_sync_url", None)
            if sync_url and not self._sync_endpoint_available(sync_url):
                sync_url = None
            if sync_url:
                from holosoma_inference.utils.rate import SimStepSyncRate

                def _set_sync_low_state(low_state_array, tick=None):
                    setter = getattr(getattr(self, "interface", None), "set_sync_low_state_array", None)
                    if setter is not None:
                        setter(low_state_array, tick=tick)

                def _get_sync_low_cmd_payload():
                    getter = getattr(getattr(self, "interface", None), "get_sync_low_cmd_payload", None)
                    if getter is None:
                        return None
                    return getter()

                self.rate = SimStepSyncRate(
                    sync_url,
                    rl_rate=self.rl_rate,
                    low_state_callback=_set_sync_low_state,
                    command_payload_callback=_get_sync_low_cmd_payload,
                )
                self.logger.info(f"Using SimStepSyncRate (sync_url={sync_url})")
            else:
                self.rate = RateLimiter(self.rl_rate)

    def _sync_endpoint_available(self, sync_url: str) -> bool:
        """Return True if lock-step stepping should be used for ``sync_url``.

        Lock-step stepping is on by default, and the default endpoint is one the
        MuJoCo simulator binds.  A ZMQ ``connect()`` succeeds unconditionally, so
        the endpoint is probed for a listener first:

        * the *default* endpoint with nothing listening falls back to wall-clock
          rate limiting and logs that it did;
        * an endpoint passed explicitly with nothing listening raises.
        """
        if sync_endpoint_has_listener(sync_url):
            return True
        if sync_url != DEFAULT_SIM_STEP_SYNC_URL:
            raise RuntimeError(
                f"No simulator is listening at --task.sim-step-sync-url={sync_url}. "
                "Start the simulator with a matching "
                "--simulator.config.policy-sync-zmq-port first, or pass "
                "--task.sim-step-sync-url=None to run without lock-step stepping."
            )
        message = (
            "No simulator listening at the default lock-step endpoint "
            + str(sync_url)
            + "; using wall-clock rate limiting. This is expected for real-robot inference and for "
            "any run without a synchronized simulator. To lock-step, start the simulator first "
            "(it binds this endpoint by default)."
        )
        self.logger.info(message)
        return False

    def _init_input_device(self):
        """Initialize input device (joystick or keyboard)."""
        if self.config.task.use_joystick:
            self._init_joystick_handler()
        else:
            self._init_keyboard_handler()

    def _init_joystick_handler(self):
        """Initialize joystick handler."""
        if sys.platform == "darwin":
            self.logger.warning("Joystick is not supported on Windows or Mac.")
            self.logger.warning("Using keyboard instead")
            self.use_joystick = False
            self._init_keyboard_handler()
        else:
            self.logger.info("Using joystick")
            self.use_joystick = True

    def _init_keyboard_handler(self):
        """Initialize keyboard handler."""
        self.logger.info("Using keyboard")
        self.use_joystick = False
        # Check if running in a TTY environment
        if not sys.stdin.isatty():
            self.logger.warning("Not running in a TTY environment - keyboard input disabled")
            self.logger.warning("This is normal for automated tests or non-interactive environments")
            self.logger.info("Auto-starting policy in non-interactive mode")
            self.use_policy_action = True
            return
        # Start keyboard listener in a daemon thread
        threading.Thread(target=self.start_key_listener, daemon=True).start()
        self.logger.info("Keyboard Listener Initialized")

    def _configure_runtime_threads(self) -> None:
        """Apply per-process CPU thread limits before loading ONNX or entering the control loop."""
        task_cfg = getattr(self.config, "task", None)
        if task_cfg is None:
            return

        cpu_thread_cap = int(getattr(task_cfg, "cpu_thread_cap", 0) or 0)
        torch_threads = int(getattr(task_cfg, "torch_num_threads", 0) or 0)
        if torch_threads <= 0 and cpu_thread_cap > 0:
            torch_threads = cpu_thread_cap
        if torch_threads > 0:
            torch.set_num_threads(torch_threads)
            logger.info(f"Torch CPU threads capped at {torch_threads}")

        torch_interop_threads = int(getattr(task_cfg, "torch_num_interop_threads", 0) or 0)
        if torch_interop_threads <= 0 and cpu_thread_cap > 0:
            torch_interop_threads = 1
        if torch_interop_threads > 0:
            try:
                torch.set_num_interop_threads(torch_interop_threads)
                logger.info(f"Torch inter-op CPU threads capped at {torch_interop_threads}")
            except RuntimeError as exc:
                logger.warning(
                    "Could not set torch inter-op threads to "
                    f"{torch_interop_threads}: {exc}"
                )

    def _build_onnx_session_options(self) -> onnxruntime.SessionOptions:
        """Build ONNX Runtime session options from task config."""
        task_cfg = getattr(self.config, "task", None)
        opts = onnxruntime.SessionOptions()
        opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

        cpu_thread_cap = int(getattr(task_cfg, "cpu_thread_cap", 0) or 0)
        intra_threads = int(getattr(task_cfg, "onnx_intra_op_threads", 0) or 0)
        inter_threads = int(getattr(task_cfg, "onnx_inter_op_threads", 0) or 0)
        if intra_threads <= 0 and cpu_thread_cap > 0:
            intra_threads = cpu_thread_cap
        if inter_threads <= 0 and cpu_thread_cap > 0:
            inter_threads = 1

        if intra_threads > 0:
            opts.intra_op_num_threads = intra_threads
        if inter_threads > 0:
            opts.inter_op_num_threads = inter_threads

        exec_mode = str(getattr(task_cfg, "onnx_execution_mode", "sequential")).lower()
        if exec_mode == "parallel":
            opts.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
        else:
            opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL

        trt_deterministic = bool(getattr(task_cfg, "tensorrt_deterministic", False))
        onnx_provider = str(getattr(task_cfg, "onnx_provider", "cpu")).lower()
        if trt_deterministic and onnx_provider == "tensorrt":
            # Force deterministic CUDA kernels for any ops that fall back from TRT EP to CUDA EP.
            opts.use_deterministic_compute = True

        return opts

    def _preload_onnx_gpu_runtime(self, requested_provider: str) -> None:
        """Best-effort preload of CUDA/cuDNN/TensorRT shared libraries for ORT GPU providers.

        On Linux, pip-installed nvidia-* packages place shared libraries under
        ``site-packages/nvidia/<pkg>/lib/`` which is typically NOT on
        ``LD_LIBRARY_PATH`` unless the user sourced ``source_inference_setup.sh``.
        We discover these paths at runtime, update ``LD_LIBRARY_PATH`` for child
        processes, and pre-load critical shared libraries via ``ctypes.CDLL`` with
        ``RTLD_GLOBAL`` so that ONNX Runtime can resolve them.
        """
        if requested_provider not in {"cuda", "tensorrt"}:
            return

        # --- Linux: ensure pip-installed nvidia libs are discoverable ----------
        if sys.platform == "linux":
            self._ensure_nvidia_libs_on_ld_path(requested_provider)

        # --- Windows: use ORT's built-in DLL preloader -------------------------
        preload = getattr(onnxruntime, "preload_dlls", None)
        if preload is not None:
            try:
                preload()
                logger.info(f"Preloaded ONNX Runtime GPU shared libraries for provider='{requested_provider}'")
            except Exception as exc:
                logger.warning(
                    "Failed to preload ONNX Runtime GPU shared libraries for provider='{}': {}",
                    requested_provider,
                    exc,
                )

    @staticmethod
    def _ensure_nvidia_libs_on_ld_path(requested_provider: str) -> None:
        """Add pip-installed nvidia/tensorrt lib dirs to LD_LIBRARY_PATH and pre-load .so files.

        Setting ``LD_LIBRARY_PATH`` alone is insufficient because the dynamic
        linker only reads it at process start.  We therefore also load every
        ``.so`` file from the pip-installed ``nvidia`` and ``tensorrt_libs``
        package trees with ``RTLD_GLOBAL`` so that subsequent ``dlopen()``
        calls by ONNX Runtime can resolve them.
        """
        import ctypes
        import importlib.util
        import pathlib
        import site

        # Collect candidate lib directories from:
        #   1. nvidia/*/lib  (cuDNN, cuBLAS, cuFFT, cuRAND, CUDA runtime, etc.)
        #   2. tensorrt_libs/ (libnvinfer, libnvinfer_plugin, etc.)
        _lib_dirs: list[pathlib.Path] = []

        _nvidia_spec = importlib.util.find_spec("nvidia")
        if _nvidia_spec is not None and _nvidia_spec.submodule_search_locations:
            _nvidia_root = pathlib.Path(list(_nvidia_spec.submodule_search_locations)[0])
            _lib_dirs.extend(sorted(_nvidia_root.glob("*/lib")))

        # tensorrt_libs is a flat directory under site-packages (not under nvidia/)
        for _sp in site.getsitepackages() + [site.getusersitepackages()]:
            _trt_libs = pathlib.Path(_sp) / "tensorrt_libs"
            if _trt_libs.is_dir():
                _lib_dirs.append(_trt_libs)
                break

        if not _lib_dirs:
            return

        _ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        _ld_entries = set(_ld_path.split(":")) if _ld_path else set()
        _added: list[str] = []
        for _lib_dir in _lib_dirs:
            _lib_str = str(_lib_dir)
            if _lib_str not in _ld_entries:
                _added.append(_lib_str)

        if _added:
            _new_ld = ":".join(_added)
            os.environ["LD_LIBRARY_PATH"] = f"{_new_ld}:{_ld_path}" if _ld_path else _new_ld
            logger.info(
                "Added {} lib dirs to LD_LIBRARY_PATH for provider='{}'",
                len(_added),
                requested_provider,
            )

        # Pre-load .so files with RTLD_GLOBAL so ONNX Runtime can resolve them.
        _all_lib_dirs = _added + [d for d in _ld_path.split(":") if d and ("nvidia" in d or "tensorrt" in d)]
        _loaded = 0
        for _lib_dir_str in _all_lib_dirs:
            _lib_dir_path = pathlib.Path(_lib_dir_str)
            if not _lib_dir_path.is_dir():
                continue
            # Load both versioned (.so.X) and unversioned (.so) — TensorRT
            # ships libnvinfer.so.10 without a further minor-version suffix.
            for _so in sorted(_lib_dir_path.glob("*.so*")):
                if _so.is_symlink() or _so.suffix == ".py":
                    continue
                try:
                    ctypes.CDLL(str(_so), mode=ctypes.RTLD_GLOBAL)
                    _loaded += 1
                except OSError as _exc:
                    logger.debug("Could not preload {}: {}", _so.name, _exc)
        if _loaded:
            logger.info("Pre-loaded {} GPU shared libraries for provider='{}'", _loaded, requested_provider)

    def _resolve_onnx_providers(self, model_path: str) -> list[tuple[str, dict[str, object]]]:
        """Resolve ONNX Runtime providers from task config with graceful fallback."""
        task_cfg = getattr(self.config, "task", None)
        provider = str(getattr(task_cfg, "onnx_provider", "cpu")).lower()
        self._preload_onnx_gpu_runtime(provider)
        available = set(onnxruntime.get_available_providers())

        if provider == "tensorrt":
            if "TensorrtExecutionProvider" not in available:
                logger.warning(
                    "Requested onnx_provider='tensorrt' but TensorrtExecutionProvider is unavailable. "
                    f"Available providers: {sorted(available)}. Falling back."
                )
                provider = "cuda" if "CUDAExecutionProvider" in available else "cpu"
            else:
                trt_fp16 = bool(getattr(task_cfg, "tensorrt_fp16", True))
                trt_deterministic = bool(getattr(task_cfg, "tensorrt_deterministic", False))
                trt_cache_dir = str(getattr(task_cfg, "tensorrt_cache_dir", ""))
                if not trt_cache_dir:
                    # Use a separate cache dir for deterministic FP32 engines to avoid
                    # contamination from non-deterministic (FP16 / TF32-on) cached engines.
                    cache_name = (
                        "trt_engine_cache_fp32_det"
                        if trt_deterministic
                        else "trt_engine_cache"
                    )
                    trt_cache_dir = str(Path(model_path).parent / cache_name)
                Path(trt_cache_dir).mkdir(parents=True, exist_ok=True)
                if bool(getattr(task_cfg, "tensorrt_require_engine_cache", False)):
                    self._validate_trt_prewarm_manifest(
                        model_path=model_path,
                        cache_dir=trt_cache_dir,
                        deterministic=trt_deterministic,
                    )
                if trt_deterministic:
                    # Disable TF32 truncation (Ampere+): FP32 GEMMs use full mantissa.
                    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
                    # cuDNN/cuBLAS deterministic algorithms — for ORT fallback ops.
                    os.environ["CUDNN_DETERMINISTIC"] = "1"
                    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                    # PYTHONHASHSEED must be set before Python starts; setting it here has no
                    # effect on the current process.  It is set in _build_policy_env so that
                    # the subprocess inherits the value before interpreter startup.
                    # PyTorch determinism — no-op for ORT/TRT inference but guards any future
                    # torch ops that might be added to the preprocessing/postprocessing path.
                    import torch as _torch
                    _torch.backends.cudnn.deterministic = True
                    _torch.backends.cudnn.benchmark = False
                    _torch.use_deterministic_algorithms(True, warn_only=True)
                    # Fix Python-level random seeds to match the configured per-run seed.
                    import random as _random

                    import numpy as _np
                    _run_seed = getattr(task_cfg, "seed", None)
                    if _run_seed is not None:
                        _np.random.seed(int(_run_seed))
                        _random.seed(int(_run_seed))
                    # Force FP32: eliminates FP16 within-kernel rounding non-determinism.
                    trt_fp16 = False
                timing_cache_name = (
                    "trt_timing_cache_fp32_det"
                    if trt_deterministic
                    else "trt_timing_cache"
                )
                trt_timing_cache_dir = str(Path(trt_cache_dir).parent / timing_cache_name)
                Path(trt_timing_cache_dir).mkdir(parents=True, exist_ok=True)
                trt_provider_options = {
                    "trt_fp16_enable": trt_fp16,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": trt_cache_dir,
                    "trt_max_workspace_size": str(2 << 30),
                    **(
                        {
                            "trt_force_sequential_engine_build": "True",
                            "trt_timing_cache_enable": "True",
                            "trt_timing_cache_path": trt_timing_cache_dir,
                        }
                        if trt_deterministic
                        else {}
                    ),
                }
                providers: list[tuple[str, dict[str, object]]] = [
                    ("TensorrtExecutionProvider", trt_provider_options),
                ]
                if "CUDAExecutionProvider" in available:
                    providers.append(("CUDAExecutionProvider", {}))
                if "CPUExecutionProvider" in available:
                    providers.append(("CPUExecutionProvider", {}))
                logger.info(
                    f"ONNX provider: TensorRT (fp16={trt_fp16}, cache={trt_cache_dir})"
                )
                return providers

        if provider == "cuda":
            if "CUDAExecutionProvider" not in available:
                logger.warning(
                    "Requested onnx_provider='cuda' but CUDAExecutionProvider is unavailable. "
                    f"Available providers: {sorted(available)}. Falling back to CPU."
                )
                provider = "cpu"
            else:
                providers = [("CUDAExecutionProvider", {})]
                if "CPUExecutionProvider" in available:
                    providers.append(("CPUExecutionProvider", {}))
                logger.info("ONNX provider: CUDA with CPU fallback")
                return providers

        if provider != "cpu":
            logger.warning(f"Unknown onnx_provider '{provider}', falling back to CPU")
        logger.info("ONNX provider: CPU")
        return [("CPUExecutionProvider", {})]

    def _validate_trt_prewarm_manifest(
        self,
        *,
        model_path: str,
        cache_dir: str,
        deterministic: bool,
    ) -> None:
        """Fail fast if the pipeline did not prewarm this ONNX TRT cache."""
        manifest_path = Path(cache_dir) / _TRT_PREWARM_MANIFEST_NAME
        if not manifest_path.is_file():
            raise RuntimeError(
                "TensorRT deterministic eval requires a prewarmed engine cache, but "
                f"{manifest_path} is missing. Run through the MuJoCo pipeline prewarm "
                "or clear/rebuild the cache before launching policy workers."
            )
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Could not read TensorRT prewarm manifest {manifest_path}: {exc}") from exc

        expected_path = str(Path(model_path).resolve())
        expected_hash = _file_sha256(model_path)
        mismatches: list[str] = []
        if manifest.get("onnx_path") != expected_path:
            mismatches.append(f"onnx_path={manifest.get('onnx_path')!r} != {expected_path!r}")
        if manifest.get("onnx_sha256") != expected_hash:
            mismatches.append("onnx_sha256 mismatch")
        if bool(manifest.get("deterministic")) != bool(deterministic):
            mismatches.append(
                f"deterministic={manifest.get('deterministic')!r} != {bool(deterministic)!r}"
            )
        providers = manifest.get("providers")
        if not isinstance(providers, list) or "TensorrtExecutionProvider" not in providers:
            mismatches.append(f"providers={providers!r} does not include TensorrtExecutionProvider")
        if mismatches:
            raise RuntimeError(
                "TensorRT prewarm manifest does not match this policy process; refusing to "
                "let ONNX Runtime rebuild an engine inside the eval worker. "
                f"manifest={manifest_path}; " + "; ".join(mismatches)
            )
        logger.info(
            "Verified shared TensorRT prewarm manifest for cache={} gpu={} model={}",
            cache_dir,
            _first_cuda_visible_device(),
            expected_path,
        )

    def _validate_active_onnx_providers(
        self,
        *,
        requested_provider: str,
        active_providers: list[str],
    ) -> None:
        """Fail fast when a requested GPU provider silently falls back to CPU."""
        requested = str(requested_provider or "cpu").lower()
        active = list(active_providers)

        if requested == "cuda" and "CUDAExecutionProvider" not in active:
            raise RuntimeError(
                "Requested onnx_provider='cuda', but the active ONNX Runtime providers are "
                f"{active}. This usually means the CUDA/cuDNN runtime is not fully available "
                "to the process, so inference would silently fall back to CPU."
            )
        if requested == "tensorrt" and "TensorrtExecutionProvider" not in active:
            raise RuntimeError(
                "Requested onnx_provider='tensorrt', but TensorrtExecutionProvider is not in the "
                f"active providers {active}. Running with CUDA-only fallback would produce "
                "non-TRT outputs that are incomparable to TRT-run baselines."
            )

    def _create_onnx_session(self, model_path: str) -> onnxruntime.InferenceSession:
        """Create an ONNX Runtime session with configured providers and thread limits."""
        requested_provider = str(getattr(self.config.task, "onnx_provider", "cpu")).lower()
        session_options = self._build_onnx_session_options()
        providers = self._resolve_onnx_providers(model_path)
        session = onnxruntime.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=providers,
        )
        self._validate_active_onnx_providers(
            requested_provider=requested_provider,
            active_providers=session.get_providers(),
        )
        logger.info(
            "ONNX Runtime active providers: "
            f"{session.get_providers()} | "
            f"intra_op={session_options.intra_op_num_threads or 'auto'} "
            f"inter_op={session_options.inter_op_num_threads or 'auto'} "
            f"mode={getattr(self.config.task, 'onnx_execution_mode', 'sequential')}"
        )
        return session

    # ============================================================================
    # Policy Methods
    # ============================================================================

    def _write_model_loaded_signal(self) -> None:
        """Write the model-loaded signal file (TRT compilation done, warmup complete)."""
        sig = getattr(getattr(self, "config", None), "task", None)
        sig_file = getattr(sig, "model_loaded_signal_file", None) if sig else None
        if not sig_file:
            return
        Path(sig_file).parent.mkdir(parents=True, exist_ok=True)
        Path(sig_file).write_text(str(time.time()))
        _log = getattr(self, "logger", logger)
        _log.info(f"Model-loaded signal written to {sig_file}")

    def setup_policy(self, model_path):
        """Setup ONNX policy model and extract metadata."""
        self.onnx_policy_session = self._create_onnx_session(model_path)
        self._write_model_loaded_signal()
        input_names = [inp.name for inp in self.onnx_policy_session.get_inputs()]
        output_names = [out.name for out in self.onnx_policy_session.get_outputs()]

        self.onnx_input_names = input_names
        self.onnx_output_names = output_names

        # Extract metadata from ONNX model (hard fault if fails)
        onnx_model = onnx.load(model_path)
        metadata = {}
        for prop in onnx_model.metadata_props:
            metadata[prop.key] = json.loads(prop.value)

        # Extract KP/KD from metadata (will be None if not present)
        self.onnx_kp = np.array(metadata["kp"]) if "kp" in metadata else None
        self.onnx_kd = np.array(metadata["kd"]) if "kd" in metadata else None

        if self.onnx_kp is not None:
            logger.info(f"Loaded KP/KD from ONNX metadata: {Path(model_path).name}")

        def policy_act(obs_dict):
            # For example,obs_dict contains:
            # {
            #     'actor_obs_lower_body': np.array([...]),
            #     'actor_obs_upper_body': np.array([...]),
            #     'estimator_obs': np.array([...])
            # }
            input_feed = {name: obs_dict[name] for name in self.onnx_input_names}
            outputs = self.onnx_policy_session.run(self.onnx_output_names, input_feed)
            return outputs[0]  # just return outputs[0] as only "action" is needed

        self.policy = policy_act

    def _resolve_control_gains(self):
        """Resolve KP/KD values with priority: config override > ONNX metadata > error.

        Creates a new config instance with resolved values if needed.
        """
        # Check if config has explicit KP/KD values
        config_has_kp = hasattr(self.robot_config, "motor_kp") and self.robot_config.motor_kp is not None
        config_has_kd = hasattr(self.robot_config, "motor_kd") and self.robot_config.motor_kd is not None

        if config_has_kp and config_has_kd:
            # Config already has values (override) - nothing to do
            logger.info(colored("Using KP/KD from config (override)", "yellow"))
            kp_values = np.array(self.robot_config.motor_kp)
            kd_values = np.array(self.robot_config.motor_kd)
        elif self.onnx_kp is not None and self.onnx_kd is not None:
            # Use ONNX metadata (default) - create new config with values
            logger.info(colored("Using KP/KD from ONNX metadata", "green"))
            kp_values = self.onnx_kp
            kd_values = self.onnx_kd
            # Create new config instance with ONNX values
            self.robot_config = replace(
                self.robot_config, motor_kp=tuple(kp_values.tolist()), motor_kd=tuple(kd_values.tolist())
            )
            # Update interface's robot_config and propagate to internal SDK components
            self.interface.update_config(self.robot_config)
        else:
            # No values available - error
            raise ValueError(
                "No KP/KD values found. Either provide them in robot config "
                "or ensure ONNX model has metadata attached during training."
            )

        # Validate dimensions
        if len(kp_values) != self.robot_config.num_motors:
            raise ValueError(
                f"KP array length ({len(kp_values)}) does not match num_motors ({self.robot_config.num_motors})"
            )
        if len(kd_values) != self.robot_config.num_motors:
            raise ValueError(
                f"KD array length ({len(kd_values)}) does not match num_motors ({self.robot_config.num_motors})"
            )

    def _calculate_obs_dim_dict(self):
        """Calculate observation dimensions for each observation type."""
        obs_dim_dict = {}
        for key in self.obs_dict:
            obs_dim_dict[key] = 0
            for obs_name in self.obs_dict[key]:
                obs_dim_dict[key] += self.obs_dims[obs_name]
        return obs_dim_dict

    def _print_observations(self, obs: dict[str, np.ndarray]) -> None:
        """Print observation vector with term naming for debugging.

        Args:
            obs: Dictionary mapping observation group names to their flattened arrays.
        """
        np.set_printoptions(suppress=True, precision=3)
        print("\n========== Observation Vector ==========")
        for group_name, group_obs in obs.items():
            print(f"\n{group_name}:")
            if group_name in self.obs_dict:
                start_idx = 0
                for term_name in self.obs_terms_sorted.get(group_name, []):
                    term_dim = self.obs_dims[term_name]
                    history_len = self.history_length_dict.get(group_name, 1)
                    total_dim = term_dim * history_len
                    term_values = group_obs[0, start_idx : start_idx + total_dim]
                    print(f"  {term_name:20s} (dim={term_dim:2d}, hist={history_len}): {term_values}")
                    start_idx += total_dim

        # Joint table: dof_name | q (deg) | dq | action
        self._print_joint_table(obs)
        print("========================================\n")

    def _print_joint_table(self, obs: dict[str, np.ndarray]) -> None:
        """Print a compact per-joint table: name | q(°) | dq(°/s) | act(°)."""
        # Walk obs_terms_sorted + obs_dims to locate dof_pos / dof_vel slices
        q = dq = None
        for grp, buf in obs.items():
            col = 0
            for term in self.obs_terms_sorted.get(grp, []):
                dim = self.obs_dims[term] * self.history_length_dict.get(grp, 1)
                if q is None and term == "dof_pos":
                    q = buf[0, col : col + dim] / self.obs_scales.get("dof_pos", 1.0)
                if dq is None and term == "dof_vel":
                    dq = buf[0, col : col + dim] / self.obs_scales.get("dof_vel", 1.0)
                col += dim
        act = self.scaled_policy_action[0] if self.scaled_policy_action is not None else None
        d = np.degrees
        w = max(len(n) for n in self.dof_names)
        print(f"\n  {'joint':<{w}}  {'q(°)':>7}  {'dq(°/s)':>8}  {'act(°)':>7}")
        print(f"  {'─' * (w + 29)}")
        for i, name in enumerate(self.dof_names):
            qi = f"{d(q[i]):7.1f}" if q is not None and i < len(q) else "    n/a"
            di = f"{d(dq[i]):8.1f}" if dq is not None and i < len(dq) else "     n/a"
            ai = f"{d(act[i]):7.1f}" if act is not None and i < len(act) else "    n/a"
            print(f"  {name:<{w}}  {qi}  {di}  {ai}")

    def rl_inference(self, robot_state_data, obs=None):
        """Perform RL inference to get policy action."""
        if obs is None:
            obs = self.prepare_obs_for_rl(robot_state_data)
        if getattr(self.config.task, "print_observations", False):
            self._print_observations(obs)
        policy_action = self.policy(obs)
        policy_action = np.clip(policy_action, -100, 100)

        self.last_policy_action = policy_action.copy()
        self.scaled_policy_action = policy_action * self.policy_action_scale
        if self.config.task.debug.force_zero_action:
            self.scaled_policy_action = np.zeros_like(self.scaled_policy_action)

        return self.scaled_policy_action

    # ============================================================================
    # Observation Processing Methods
    # ============================================================================

    def get_current_obs_buffer_dict(self, robot_state_data):
        """Extract current observation data from robot state."""
        current_obs_buffer_dict = {}

        # Extract base and joint data
        current_obs_buffer_dict["base_quat"] = robot_state_data[:, 3:7]
        if self.config.task.debug.force_zero_angular_velocity:
            current_obs_buffer_dict["base_ang_vel"] = np.zeros((1, 3))
        else:
            current_obs_buffer_dict["base_ang_vel"] = robot_state_data[:, 7 + self.num_dofs + 3 : 7 + self.num_dofs + 6]
        current_obs_buffer_dict["dof_pos"] = robot_state_data[:, 7 : 7 + self.num_dofs] - self.default_dof_angles
        current_obs_buffer_dict["dof_vel"] = robot_state_data[
            :, 7 + self.num_dofs + 6 : 7 + self.num_dofs + 6 + self.num_dofs
        ]

        # Use pre-computed corrected gravity if available from interface, else compute
        # This logic seems very brittle. TODO: Return a dataclass instead of just a numpy array.
        expected_len = (
            7 + self.num_dofs + 6 + self.num_dofs
        )  # base_pos(3) + quat(4) + dof_pos + lin_vel(3) + ang_vel(3) + dof_vel
        if self.config.task.debug.force_upright_imu:
            current_obs_buffer_dict["projected_gravity"] = np.array([[0.0, 0.0, -1.0]])
        elif robot_state_data.shape[1] == expected_len + 3:
            current_obs_buffer_dict["projected_gravity"] = robot_state_data[:, expected_len : expected_len + 3]
        else:
            v = np.array([[0, 0, -1]])
            current_obs_buffer_dict["projected_gravity"] = quat_rotate_inverse(current_obs_buffer_dict["base_quat"], v)

        return current_obs_buffer_dict

    def get_projected_gravity_z(self, robot_state_data) -> float:
        """Return the current projected_gravity z component from raw robot state."""
        current_obs_buffer_dict = self.get_current_obs_buffer_dict(robot_state_data)
        projected_gravity = np.asarray(current_obs_buffer_dict["projected_gravity"], dtype=np.float64).reshape(1, -1)
        return float(projected_gravity[0, 2])

    def parse_current_obs_dict(self, current_obs_buffer_dict):
        """Parse observation buffer into observation dictionary with per-term scaling."""
        current_obs_dict: dict[str, dict[str, np.ndarray]] = {}
        for group, term_names in self.obs_terms_sorted.items():
            grouped_terms: dict[str, np.ndarray] = {}
            for term in term_names:
                if term not in current_obs_buffer_dict:
                    raise KeyError(f"Observation term '{term}' missing from current observation buffer.")
                term_obs = current_obs_buffer_dict[term]
                if term_obs.ndim == 1:
                    term_obs = term_obs.reshape(1, -1)
                scale = self.obs_scales[term]
                grouped_terms[term] = (term_obs * scale).astype(np.float32, copy=False)
            current_obs_dict[group] = grouped_terms
        return current_obs_dict

    def _prepare_group_observations(self, robot_state_data):
        """Return flattened observations per group with history applied per term."""
        current_obs_buffer_dict = self.get_current_obs_buffer_dict(robot_state_data)
        current_obs_dict = self.parse_current_obs_dict(current_obs_buffer_dict)

        group_outputs = self._update_obs_history(current_obs_dict)
        self._add_transformer_collection_payload(current_obs_buffer_dict)
        return group_outputs

    def _update_obs_history(self, current_obs_dict: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        """Update observation history buffers and return flattened observations per group."""
        group_outputs: dict[str, np.ndarray] = {}

        for group, term_dict in current_obs_dict.items():
            history_len = self.history_length_dict.get(group, 1)
            flattened_terms: list[np.ndarray] = []

            for term in self.obs_terms_sorted[group]:
                obs = np.asarray(term_dict[term], dtype=np.float32, order="C")
                if obs.ndim == 1:
                    obs = obs.reshape(1, -1)

                buffer = self.obs_history_buffers[group][term]
                buffer.append(obs.copy())

                history = list(buffer)
                if len(history) < history_len:
                    missing = history_len - len(history)
                    history = [np.zeros_like(obs)] * missing + history

                # Match training order: time dimension first, then flatten into [history_len * term_dim].
                stacked = np.stack(history[-history_len:], axis=1)
                flattened_terms.append(stacked.reshape(obs.shape[0], -1))

            group_outputs[group] = (
                np.concatenate(flattened_terms, axis=1).astype(np.float32, copy=False)
                if flattened_terms
                else np.zeros((1, 0), dtype=np.float32)
            )

        self.obs_buf_dict = {group: value.copy() for group, value in group_outputs.items()}
        return group_outputs

    def _should_emit_transformer_collection_payload(self) -> bool:
        return bool(getattr(self.config.task, "collect_data", False))

    def _transformer_collection_obs_dict(
        self,
        *,
        dynamics_terms: tuple[str, ...] = TRANSFORMER_FINETUNE_DYNAMICS_TERMS,
    ) -> dict[str, list[str]]:
        return {
            "raw_dynamics_obs": list(dynamics_terms),
            "current_command": list(TRANSFORMER_FINETUNE_COMMAND_TERMS),
        }

    def _build_data_collection_obs_dict(self) -> dict[str, list[str]] | None:
        if not hasattr(self.config, "observation") or not hasattr(self.config.observation, "obs_dict"):
            return None
        obs_dict = {str(k): list(v) for k, v in dict(self.config.observation.obs_dict).items()}
        if self._should_emit_transformer_collection_payload():
            obs_dict.update(self._transformer_collection_obs_dict())
        return obs_dict

    def _ensure_transformer_collection_command_terms(
        self, current_obs_buffer_dict: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        obs = dict(current_obs_buffer_dict)
        if "command_lin_vel" not in obs and hasattr(self, "lin_vel_command"):
            obs["command_lin_vel"] = self.lin_vel_command
        if "command_ang_vel" not in obs and hasattr(self, "ang_vel_command"):
            obs["command_ang_vel"] = self.ang_vel_command
        if "sin_phase" not in obs and hasattr(self, "_get_obs_sin_phase"):
            obs["sin_phase"] = self._get_obs_sin_phase()
        if "cos_phase" not in obs and hasattr(self, "_get_obs_cos_phase"):
            obs["cos_phase"] = self._get_obs_cos_phase()
        return obs

    def _add_transformer_collection_payload(
        self, current_obs_buffer_dict: dict[str, np.ndarray]
    ) -> None:
        """Emit transformer-compatible raw trajectory fields for any locomotion policy."""
        if not self._should_emit_transformer_collection_payload():
            return

        obs = self._ensure_transformer_collection_command_terms(current_obs_buffer_dict)
        missing_dynamics = [term for term in TRANSFORMER_FINETUNE_DYNAMICS_TERMS if term not in obs]
        missing_command = [term for term in TRANSFORMER_FINETUNE_COMMAND_TERMS if term not in obs]
        if missing_dynamics or missing_command:
            return

        raw_chunks: list[np.ndarray] = []
        for term in TRANSFORMER_FINETUNE_DYNAMICS_TERMS:
            value = np.asarray(obs[term], dtype=np.float32).reshape(1, -1)
            raw_chunks.append(value)
        raw_dynamics_obs = np.concatenate(raw_chunks, axis=1).astype(np.float32, copy=False)

        current_command = np.concatenate(
            [
                np.asarray(obs[term], dtype=np.float32).reshape(1, -1)
                for term in TRANSFORMER_FINETUNE_COMMAND_TERMS
            ],
            axis=1,
        ).astype(np.float32, copy=False)

        self.obs_buf_dict["raw_dynamics_obs"] = raw_dynamics_obs
        self.obs_buf_dict["current_command"] = current_command

    def _resolve_base_velocities(self, robot_state_data):
        """Get base linear/angular velocity, preferring ground-truth estimates if available."""
        truth_vel = None
        if hasattr(self.interface, "get_vel_state"):
            truth_vel = self.interface.get_vel_state()

        if truth_vel is not None:
            base_lin_vel = truth_vel[:, 0:3]
            base_ang_vel = truth_vel[:, 3:6]
        else:
            if not self._missing_truth_vel_warned:
                self.logger.warning(
                    "Truth velocity unavailable (no ZMQ/mocap). Logging zeros for base linear/angular velocity."
                )
                self._missing_truth_vel_warned = True
            base_lin_vel = np.zeros((1, 3), dtype=np.float64)
            base_ang_vel = np.zeros((1, 3), dtype=np.float64)
        return base_lin_vel, base_ang_vel

    def _extract_joint_states(self, robot_state_data):
        """Extract joint position/velocity/torque slices from robot_state_data."""
        dof_pos = robot_state_data[:, 7 : 7 + self.num_dofs]
        dof_vel = robot_state_data[:, 7 + self.num_dofs + 6 : 7 + self.num_dofs + 6 + self.num_dofs]
        tau_start = 7 + self.num_dofs + 6 + self.num_dofs + 6  # skip base force/torque entries
        dof_tau = robot_state_data[:, tau_start : tau_start + self.num_dofs]
        return dof_pos, dof_vel, dof_tau

    def _apply_residual_upper_body(self, q_target: np.ndarray, scaled_action: np.ndarray) -> np.ndarray:
        """Hook for subclasses that train with ``residual_upper_body_action=True``.

        Default = identity (target = action + default). Subclasses that opt in
        replace the upper-body slice with ``action + ref_upper_dof_pos`` so the
        deploy-side target formula matches the training-time controller.
        """
        return q_target

    def _maybe_log_inference_state(self, robot_state_data, q_target, obs_for_rl=None):
        """Log inference step in eval-compatible format (minus contact forces)."""
        if self.state_logger is None or not self.use_policy_action or not self._session_recording_enabled:
            # Still drive vel_state_processor so ZMQ mocap is polled and recorded every step (unified log on exit)
            if getattr(self.interface, "vel_state_processor", None) is not None and hasattr(self.interface, "get_pose_state"):
                self.interface.get_pose_state()
            return

        dof_pos, dof_vel, dof_tau = self._extract_joint_states(robot_state_data)
        base_lin_vel, base_ang_vel = self._resolve_base_velocities(robot_state_data)

        # Get base pose (position and orientation) - prefer ground truth from ZMQ/mocap
        truth_pose = None
        if hasattr(self.interface, "get_pose_state"):
            truth_pose = self.interface.get_pose_state()
        
        if truth_pose is not None:
            # Use ground truth pose from ZMQ/mocap
            base_pos = truth_pose[:, 0:3]  # [x, y, z] in world frame
            base_quat_xyzw = truth_pose[:, 3:7]  # [x, y, z, w] in world frame
        else:
            # Fallback to robot_state_data (may be less accurate)
            base_pos = robot_state_data[:, 0:3]  # [x, y, z] in world frame
            base_quat_xyzw = robot_state_data[:, 3:7]  # [x, y, z, w] in world frame
        
        # Store initial pose for coordinate transformation (only on first call)
        if not hasattr(self, "_initial_base_pos"):
            self._initial_base_pos = base_pos.copy()
            self._initial_base_quat_xyzw = base_quat_xyzw.copy()
        
        # Extract yaw angle from quaternion (xyzw format)
        # yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
        qx, qy, qz, qw = base_quat_xyzw[0, 0], base_quat_xyzw[0, 1], base_quat_xyzw[0, 2], base_quat_xyzw[0, 3]
        base_yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

        # Match eval logging: use raw (unscaled) policy action as target delta
        if self.use_policy_action and getattr(self, "last_policy_action", None) is not None:
            dof_pos_target = np.asarray(self.last_policy_action)
        elif q_target is not None:
            dof_pos_target = np.asarray(q_target) - self.default_dof_angles
        else:
            dof_pos_target = np.zeros_like(dof_pos)

        log_payload = {
            "dof_pos_target": dof_pos_target.copy(),
            "dof_pos": dof_pos.copy(),
            "dof_vel": dof_vel.copy(),
            "dof_torque": dof_tau.copy(),
            "command_x": np.array([self.lin_vel_command[0, 0]]),
            "command_y": np.array([self.lin_vel_command[0, 1]]),
            "command_yaw": np.array([self.ang_vel_command[0, 0]]),
            "base_vel_x": np.array([base_lin_vel[0, 0]]),
            "base_vel_y": np.array([base_lin_vel[0, 1]]),
            "base_vel_z": np.array([base_lin_vel[0, 2]]),
            "base_vel_yaw": np.array([base_ang_vel[0, 2]]),
            "base_pos_x": np.array([base_pos[0, 0]]),
            "base_pos_y": np.array([base_pos[0, 1]]),
            "base_pos_z": np.array([base_pos[0, 2]]),
            "base_yaw": np.array([base_yaw]),
            "base_quat_x": np.array([base_quat_xyzw[0, 0]]),
            "base_quat_y": np.array([base_quat_xyzw[0, 1]]),
            "base_quat_z": np.array([base_quat_xyzw[0, 2]]),
            "base_quat_w": np.array([base_quat_xyzw[0, 3]]),
            "initial_base_pos_x": np.array([self._initial_base_pos[0, 0]]),
            "initial_base_pos_y": np.array([self._initial_base_pos[0, 1]]),
            "initial_base_pos_z": np.array([self._initial_base_pos[0, 2]]),
            "initial_base_quat_x": np.array([self._initial_base_quat_xyzw[0, 0]]),
            "initial_base_quat_y": np.array([self._initial_base_quat_xyzw[0, 1]]),
            "initial_base_quat_z": np.array([self._initial_base_quat_xyzw[0, 2]]),
            "initial_base_quat_w": np.array([self._initial_base_quat_xyzw[0, 3]]),
            "env_done_status": np.array([False]),
        }

        self.state_logger.log_states(log_payload)

    def prepare_obs_for_rl(self, robot_state_data):
        """Prepare observations for RL inference."""
        group_outputs = self._prepare_group_observations(robot_state_data)
        if "actor_obs" not in group_outputs:
            raise KeyError("Observation group 'actor_obs' is not configured for this policy.")
        return {"actor_obs": group_outputs["actor_obs"].astype(np.float32, copy=False)}

    # ============================================================================
    # Control/Command Methods
    # ============================================================================

    def get_init_target(self, robot_state_data):
        """Get initialization target joint positions."""
        dof_pos = robot_state_data[:, 7 : 7 + self.num_dofs]
        if self.get_ready_state:
            # Interpolate from current dof_pos to default angles
            q_target = dof_pos + (self.default_dof_angles - dof_pos) * (self.init_count / 500)
            self.init_count += 1
            return q_target
        return dof_pos

    def policy_action(self):
        """Execute policy action and send commands to robot."""

        # Snapshot flags to prevent race with mode-switch handler thread
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

        # Stage 3: Inference
        if use_policy and not get_ready:
            with self.latency_tracker.measure("inference"):
                scaled_policy_action = self.rl_inference(robot_state_data)

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
            self.cmd_q[:] = q_target

        # Stage 5: Optional inference logging (before sending command)
        self._maybe_log_inference_state(robot_state_data, q_target)

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

    def _get_manual_command(self, robot_state_data):
        """Optional manual command when policy control is disabled."""
        return

    def _get_obs_phase_time(self):
        """Calculate phase time for gait."""
        cur_time = time.perf_counter() * self.stand_command[0, 0]
        phase_time = cur_time % self.gait_period / self.gait_period
        self.phase_time[:, 0] = phase_time
        return self.phase_time

    def update_phase_time(self):
        """Update phase time."""
        phase_tp1 = self.phase + self.phase_dt
        self.phase = np.fmod(phase_tp1 + np.pi, 2 * np.pi) - np.pi

    # ============================================================================
    # Input Handler Methods
    # ============================================================================

    def start_key_listener(self):
        """Start keyboard listener thread."""

        def on_press(keycode):
            try:
                self.handle_keyboard_button(keycode)
            except AttributeError:
                pass  # Handle special keys if needed

        try:
            listener = listen_keyboard(on_press=on_press)
            listener.start()
            listener.join()
        except OSError as e:
            # Handle termios errors in non-TTY environments
            self.logger.warning("Could not start keyboard listener: %s", e)
            self.logger.warning("Keyboard input will not be available")

    def process_joystick_input(self):
        """Process joystick input and update commands using interface."""
        # Store previous key states for edge detection
        self.last_key_states = self.key_states.copy() if hasattr(self, "key_states") else {}

        # Process joystick input - returns (lin_vel, ang_vel, key_states)
        self.lin_vel_command, self.ang_vel_command, self.key_states = self.interface.process_joystick_input(
            self.lin_vel_command, self.ang_vel_command, self.stand_command, False
        )

        # Handle button presses (edge detection: only trigger on press, not hold)
        for key, is_pressed in self.key_states.items():
            if is_pressed and not self.last_key_states.get(key, False):
                self.handle_joystick_button(key)
                self._print_control_status()

    # ============================================================================
    # Button Handler Methods
    # ============================================================================

    def handle_keyboard_button(self, keycode):
        """Handle keyboard button presses."""
        if self._try_switch_policy_key(keycode):
            pass
        elif keycode == "]":
            self._handle_start_policy()
        elif keycode == "o":
            self._handle_stop_policy()
        elif keycode == "i":
            self._handle_init_state()
        elif keycode in ["v", "b", "f", "g", "r"]:
            self._handle_kp_control(keycode)

        self._print_control_status()

    def handle_joystick_button(self, cur_key):
        """Handle joystick button presses."""
        if cur_key == "A":
            self._handle_start_policy()
        elif cur_key == "B":
            self._handle_stop_policy()
        elif cur_key == "Y":
            self._handle_init_state()
        elif cur_key in ["up", "down", "left", "right", "F1"]:
            # TODO: Make this more intuitive
            self._handle_joystick_kp_control(cur_key)
        elif cur_key == "select":
            # Cycle to next policy
            next_index = (self.active_policy_index + 1) % len(self.model_paths)
            self._activate_policy(next_index)
        elif cur_key == "L1+R1":
            # Kill program, works on G1 joystick only.
            self.logger.info(colored("Killing program via joystick command", "red"))
            sys.exit(0)

    # ============================================================================
    # Control Action Methods
    # ============================================================================

    def _handle_start_policy(self):
        """Handle start policy action."""
        self.use_policy_action = True
        self.get_ready_state = False
        self.logger.info(colored("Using policy actions", "blue"))
        self.phase = np.array([[0.0, np.pi]])
        if hasattr(self.interface, "no_action"):
            self.interface.no_action = 0

    def _handle_stop_policy(self):
        """Handle stop policy action."""
        self.use_policy_action = False
        self.get_ready_state = False
        self.logger.info("Actions set to zero")
        if hasattr(self.interface, "no_action"):
            self.interface.no_action = 1

    def _handle_init_state(self):
        """Handle initialization state."""
        self.get_ready_state = True
        self.init_count = 0
        self.logger.info("Setting to init state")
        if hasattr(self.interface, "no_action"):
            self.interface.no_action = 0

    def _handle_kp_control(self, keycode):
        """Handle keyboard KP control."""
        if keycode == "v":
            self.interface.kp_level -= 0.01
        elif keycode == "b":
            self.interface.kp_level += 0.01
        elif keycode == "f":
            self.interface.kp_level -= 0.1
        elif keycode == "g":
            self.interface.kp_level += 0.1
        elif keycode == "r":
            self.interface.kp_level = 1.0

    def _handle_joystick_kp_control(self, keycode):
        """Handle joystick KP control."""
        if keycode == "down":
            self.interface.kp_level -= 0.1
        elif keycode == "up":
            self.interface.kp_level += 0.1
        elif keycode == "left":
            self.interface.kp_level -= 0.01
        elif keycode == "right":
            self.interface.kp_level += 0.01
        elif keycode == "F1":
            self.interface.kp_level = 1.0

    def _print_control_status(self):
        """Print current control status."""
        self.logger.info("------------ Control Status ------------")
        if self.active_model_path:
            total = len(self.model_paths)
            name = Path(self.active_model_path).name
            debug_str = (
                f"Active policy [{self.active_policy_index + 1}/{total}]: {name} Kp level {self.interface.kp_level:.2f}"
            )
            self.logger.info(debug_str)

    # Dynamics recon logging (K-step MSE) — implemented on LocomotionPolicy_Deploy; stubs for
    # policies that inherit BasePolicy only (e.g. WholeBodyTrackingPolicy).
    def _reset_recon_k_step_log(self):
        """Reset K-step recon running stats at start of run. No-op unless overridden."""
        return

    def _flush_pending_dynamics_log(self):
        """Flush pending dynamics prediction to state log at end of run. No-op unless overridden."""
        return

    # ============================================================================
    # Main Run Method
    # ============================================================================

    def _before_run_loop(self):
        """Hook called once before entering the standalone run loop."""
        return

    def _run_iteration(self, it: int) -> bool:
        """Execute one loop iteration. Return True to stop the loop."""
        self.latency_tracker.start_cycle()

        if self.use_joystick and self.interface.get_joystick_msg() is not None:
            self.process_joystick_input()
        if getattr(self, "_skip_current_iteration", False):
            self._skip_current_iteration = False
            self.latency_tracker.end_cycle()
            return False
        if self.use_phase:
            self.update_phase_time()
        if getattr(self, "_skip_current_iteration", False):
            self._skip_current_iteration = False
            self.latency_tracker.end_cycle()
            return False

        self.policy_action()

        self.latency_tracker.end_cycle()

        if it % 50 == 0 and self.use_policy_action:
            debug_str = f"RL FPS: {self.latency_tracker.get_fps():.2f} | {self.latency_tracker.get_stats_str()}"
            self.logger.info(debug_str, flush=True)

        return False

    def _after_run_loop(self):
        """Hook called once after leaving the standalone run loop."""
        if hasattr(self, "rate") and hasattr(self.rate, "close"):
            self.rate.close()
        if self.state_logger is not None:
            self._flush_pending_dynamics_log()
            self.state_logger.save()

    def run(self):
        """Main run loop for the policy."""
        self._reset_recon_k_step_log()
        try:
            self._before_run_loop()
            for it in itertools.count():
                should_stop = self._run_iteration(it)
                if should_stop:
                    break

                self.rate.sleep()

        except KeyboardInterrupt:
            pass
        finally:
            self._after_run_loop()

