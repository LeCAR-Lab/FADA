"""Task configuration types for holosoma_inference."""

from __future__ import annotations

import typing

from pydantic.dataclasses import dataclass

from holosoma_inference.utils.sync_rendezvous import DEFAULT_SIM_STEP_SYNC_URL


@dataclass(frozen=True)
class DebugConfig:
    """Debug overrides for quick testing."""

    force_upright_imu: bool = False
    """Override projected_gravity with [0, 0, -1] (perfectly upright)."""

    force_zero_angular_velocity: bool = False
    """Override base_ang_vel with [0, 0, 0]."""

    force_zero_action: bool = False
    """Zero out the scaled policy action (robot holds default pose)."""


@dataclass(frozen=True)
class TaskConfig:
    """Task execution configuration for policy inference."""

    model_path: str | list[str]
    """Path to ONNX model(s). Supports local paths and wandb:// URIs. Required field."""

    policy_mode: typing.Literal[
        "auto",
        "transformer",
        "fada",
        "deploy",
        "wbt",
    ] = "auto"
    """Policy class selection mode. Use explicit mode for deterministic dispatch; auto keeps compatibility fallback."""

    rl_rate: float = 50
    """Policy inference rate in Hz."""

    seed: int | None = None
    """Global random seed for reproducible inference. None disables seeding."""

    policy_action_scale: float = 0.25
    """Scaling factor applied to policy actions."""

    action_scales_by_effort_limit_over_p_gain: bool = False
    """Use per-joint scaling: ``policy_action_scale * effort_limit / p_gain``."""

    use_phase: bool = True
    """Whether to use gait phase observations."""

    gait_period: float = 1.0
    """Gait cycle period in seconds."""

    domain_id: int = 0
    """DDS domain ID for communication."""

    interface: str = "lo"
    """Network interface name."""

    use_joystick: bool = False
    """Enable joystick control input."""

    joystick_type: str = "xbox"
    """Joystick type."""

    joystick_device: int = 0
    """Joystick device index."""

    use_sim_time: bool = False
    """Use synchronized simulation time for WBT policies."""

    sim_step_sync_url: str | None = DEFAULT_SIM_STEP_SYNC_URL
    """ZMQ PAIR endpoint for synchronous policy-simulator stepping (on by default).

    When set, the policy uses lock-step synchronization with the MuJoCo
    simulator instead of wall-clock rate limiting.  Each policy step sends a
    STEP message and waits for DONE from the sim (which has completed exactly
    fps/rl_rate physics steps), so the number of physics steps per policy
    action is fixed by configuration rather than by the simulator's achieved
    throughput.

    The default matches the simulator's own default port -- see
    ``holosoma_inference/utils/sync_rendezvous.py`` for how it is derived.  The
    endpoint is probed before the control loop starts: with no simulator
    listening (real-robot inference, or any run without a synchronized
    simulator) the policy logs that fact and uses wall-clock rate limiting.  An
    endpoint passed explicitly on the command line is never downgraded that
    way; if nothing is listening there, start-up fails.

    Set to None to opt out of lock-step stepping entirely.
    """

    wandb_download_dir: str = "/tmp"
    """Directory for downloading W&B checkpoints."""

    # Deprecation candidates:
    desired_base_height: float = 0.75
    """Target base height in meters."""

    residual_upper_body_action: bool = False
    """Whether to use residual control for upper body."""

    use_ros: bool = False
    """Use ROS2 for rate limiting."""

    # Inference logging / ground-truth velocity options
    log_states: bool = True
    """Enable eval-format logging during inference."""

    log_output_dir: str = "logs/inference"
    """Directory to store inference logs."""

    log_filename: str = "state_log"
    """Base filename for inference logs ('.npz' added automatically if missing)."""

    log_exp_name: str | None = None
    """Optional subfolder name under auto log dir (e.g., inference/<exp_name>)."""

    vel_state_source: typing.Literal["none", "zmq"] = "zmq"
    """Source for ground-truth base velocity (none|zmq). Default zmq so unified log and exit plots are enabled."""

    vel_state_zmq_url: str = "tcp://127.0.0.1:6000"
    """ZMQ endpoint for ground-truth pose (mocap/bridge)."""

    vel_state_target_name: str = "base"
    """Object name to filter incoming pose messages."""

    vel_state_orientation_order: typing.Literal["xyzw", "wxyz"] = "xyzw"
    """Incoming quaternion order for pose messages."""

    plot_mocap_raw_on_exit: bool = True
    """If True, record all mocap raw data (pos + quat) and plot vs time when the program exits (Ctrl+C or normal exit)."""

    mocap_plot_hz: float = 50.0
    """When > 0, resample mocap to this rate (e.g. 50) for plotting and velocity computation only. Reduces velocity spikes from high-rate noise. 0 = use raw rate."""

    plot_trim_head: int = 0
    """Trim this many steps from the start when plotting (and inline RMSE)."""

    plot_trim_tail: int = 0
    """Trim this many steps from the end when plotting (and inline RMSE)."""

    plot_smooth_window: int = 0
    """Moving average window for velocity smoothing in plots (0 = no smoothing)."""

    plot_max_linear_vel: float = 1.0
    """Outlier clipping threshold for linear velocity |v_x|, |v_y| in exit plots (m/s)."""

    plot_max_angular_vel: float = 2.0
    """Outlier clipping threshold for angular velocity in plots (rad/s)."""

    tracking_reward_checkpoint_path: str | None = None
    """Optional checkpoint path used to annotate exit velocity plots with tracking return."""

    # Data collection settings
    collect_data: bool = False
    """Enable data collection during inference."""

    data_collection_output_dir: str | None = None
    """Output directory for the collected HDF5 dataset. When set, this takes priority over
    ``log_output_dir`` for data collection only (state logs / plots / mocap output still use
    ``log_output_dir``); useful for routing large collected datasets to separate storage. When
    None (default), collected data is written under the same resolved ``log_output_dir`` as
    everything else."""

    data_collection_dataset_name: str = "dataset"
    """Name of the collected dataset."""

    data_collection_compress: bool = True
    """Whether to compress HDF5 data."""

    collect_mark_fall_terminal: bool = False
    """Opt in to marking collected steps terminal from the moment the robot has fallen.

    Default False: every collected step is written with ``dones=False``, so
    ``load_trajectories_from_h5`` (which truncates a trajectory at its first ``done``) keeps
    the whole episode including the post-fall tail.

    A fall is detected and reported either way: the end-of-collection summary states how many
    recorded steps are post-fall. Set True to have the finetune loader cut the episode at the
    fall, which changes which windows the IDM LoRA finetune sees."""

    collect_fall_projected_gravity_z_max: float = -0.7
    """Fall threshold on projected-gravity z used by the collection-time fall report.

    Upright is ``projected_gravity_z ~= -1``; -0.7 is 45.57 degrees of body tilt. Same
    criterion and same default as ``dual_mode_projected_gravity_z_min``, so the collector and the
    dual-mode safety guard agree on what "fallen" means. Only affects the reported count unless
    ``collect_mark_fall_terminal`` is also True.

    KNOWN DISAGREEMENT: step 6's own fall detector
    (``plot_mocap_raw.STEP6_FALL_R22_MAX``) uses ``R[2,2] < 0.5`` -- 60 degrees, and blind to
    the first 5% of the run. ``projected_gravity_z`` is exactly ``-R[2,2]``, so a rollout
    tilted between 45.57 and 60 degrees is fallen here and upright there. Step 6's
    ``fall_step`` truncates its exported tracking error and zeroes its tracking return. See
    the comment on ``STEP6_FALL_R22_MAX`` and
    ``utils/tests/test_fall_threshold_discrepancy.py``."""

    data_collection_skip_obs_keys: tuple[str, ...] = ("actor_obs", "critic_obs")
    """Observation groups to skip when collecting data."""

    # Command randomization for inference (matches train/eval behavior). Random commands are
    # the only supported command source for deploy-mode eval/collection, hence the True
    # default.
    randomize_commands: bool = True
    """Enable automatic random command generation during inference."""

    command_resampling_time: float = 10.0
    """Seconds between command resamples; <=0 disables resampling."""

    command_lin_vel_x_range: tuple[float, float] = (-1.0, 1.0)
    """Range for x linear velocity commands."""

    command_lin_vel_y_range: tuple[float, float] = (-1.0, 1.0)
    """Range for y linear velocity commands."""

    command_ang_vel_range: tuple[float, float] = (-1.0, 1.0)
    """Range for yaw angular velocity commands."""

    command_stand_prob: float = 0.2
    """Probability to sample a standing command (all zeros)."""

    initial_zero_command_steps: int = 0
    """Hold random eval commands at zero for the first N control steps."""

    auto_start_policy: bool = False
    """Automatically start policy actions before the main loop."""

    dual_mode_session_enabled: bool = False
    """Enable robust/test session semantics when secondary policy is configured."""

    dual_mode_start_in_secondary: bool = True
    """If True, dual-mode session starts in the secondary/robust policy."""

    dual_mode_test_start_delay_s: float = -1.0
    """Delay before auto-starting test session in dual-mode. <0 means manual key trigger."""

    dual_mode_post_secondary_hold_s: float = 0.0
    """Hold time in secondary/robust mode after test exits before optional auto-exit."""

    dual_mode_exit_after_post_hold: bool = False
    """If True, dual-mode session exits automatically after post-test secondary hold."""

    dual_mode_projected_gravity_z_min: float = -0.7
    """Minimum safe projected_gravity z for primary/test policy. Values above trigger fallback."""

    max_steps: int = -1
    """Maximum number of policy steps before exiting. -1 means no limit."""

    # Transformer chunk twin visualization bridge (inference -> simulator)
    chunk_twin_publish: bool = False
    """Enable publishing transformer chunk predictions for simulator-side twin visualization."""

    chunk_twin_pub_url: str = "tcp://127.0.0.1:6001"
    """ZMQ PUB endpoint used by inference to publish chunk twin packets."""

    chunk_twin_publish_obs: bool = False
    """Include observation-prediction twin data in published packets when available."""

    chunk_twin_max_horizon_frames: int = 10
    """Maximum number of horizon frames to publish per packet for twin visualization."""

    chunk_twin_publish_every_n_steps: int = 1
    """Publish one chunk twin packet every N policy steps (>=1)."""


    max_eval_time: float = -1.0
    """Maximum evaluation time in seconds. If exceeded, commands will be set to zero (stop). -1.0 means no limit."""

    exit_after_max_eval_time: bool = True
    """If True, terminate the policy run loop when max_eval_time is exceeded.
    If False, only zero out commands but keep running until max_steps."""

    model_loaded_signal_file: str | None = None
    """If set, write this file as soon as the ONNX/TRT model is loaded and the
    first inference warmup is done.  Written BEFORE the first policy_action(),
    so the orchestrator can start its per-policy-cycle timeout only from this
    point (i.e. TRT compilation time is excluded from the ready timeout)."""

    ready_signal_file: str | None = None
    """If set, write this file once the policy is fully initialised and about
    to enter the main loop.  An external orchestrator can poll for this file
    to know the policy is ready (e.g. for gantry release timing)."""

    release_done_signal_file: str | None = None
    """If set, wait for this file immediately after writing ready_signal_file.
    The MuJoCo pipeline writes it after the gantry release command is acked,
    ensuring the next sync-step batch starts only after release."""

    done_signal_file: str | None = None
    """If set, write this file when the policy main loop exits (before atexit
    handlers like plot saving).  An external orchestrator can poll for this
    file to know the eval phase has finished and proceed immediately."""

    print_observations: bool = False
    """Print observation vectors for debugging."""

    motion_start_timestep: int = 0
    """Starting timestep for motion clip playback."""

    motion_end_timestep: int | None = None
    """Ending timestep for motion clip playback. If None, plays until the end."""

    # ONNX Runtime execution provider settings
    onnx_provider: typing.Literal["cpu", "cuda", "tensorrt"] = "cpu"
    """ONNX Runtime execution provider. 'cpu' uses CPUExecutionProvider (default),
    'cuda' uses CUDAExecutionProvider, 'tensorrt' uses TensorrtExecutionProvider
    with CUDA fallback. Requires onnxruntime-gpu for cuda/tensorrt."""

    cpu_thread_cap: int = 0
    """Optional global cap for per-process CPU compute threads.

    When >0, automatic inference uses this as the default cap for ONNX Runtime
    intra-op threads and torch CPU threads unless those are overridden below.
    `0` keeps library defaults.
    """

    onnx_intra_op_threads: int = 0
    """Explicit ONNX Runtime intra-op thread count. `0` means auto/default."""

    onnx_inter_op_threads: int = 0
    """Explicit ONNX Runtime inter-op thread count. `0` means auto/default."""

    onnx_execution_mode: typing.Literal["sequential", "parallel"] = "sequential"
    """ONNX Runtime execution mode. `sequential` is usually more stable for many
    concurrent policy workers; `parallel` may help single-process throughput."""

    torch_num_threads: int = 0
    """Optional torch CPU thread count. `0` means auto/default."""

    torch_num_interop_threads: int = 0
    """Optional torch CPU inter-op thread count. `0` means auto/default."""

    tensorrt_fp16: bool = True
    """Enable FP16 precision for TensorRT. Reduces memory and increases throughput
    with minimal accuracy loss on most models. Only used when onnx_provider='tensorrt'."""

    tensorrt_cache_dir: str = ""
    """Directory for caching TensorRT engine files. Empty string means auto-derive
    from model path (recommended). The first run builds engines (slow), subsequent
    runs load from cache (fast). Only used when onnx_provider='tensorrt'."""

    tensorrt_deterministic: bool = False
    """Enable deterministic TensorRT inference. Disables TF32, forces FP32, enables
    ORT deterministic compute, and uses a persistent TRT timing cache to make kernel
    selection reproducible across builds. Residual non-determinism from TRT-internal
    CUDA atomics (e.g. attention reduction) remains; CLB would eliminate it at the
    cost of ~5-20x inference slowdown. Only used when onnx_provider='tensorrt'."""

    tensorrt_require_engine_cache: bool = False
    """Fail before ONNX Runtime session creation if deterministic TRT prewarm has
    not produced a matching per-GPU engine-cache manifest. Used by the MuJoCo
    pipeline to prevent policy subprocesses from silently building TRT engines."""

    debug: DebugConfig = DebugConfig()
    """Debug overrides for quick testing."""
