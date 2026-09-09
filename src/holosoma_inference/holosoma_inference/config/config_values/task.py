"""Default task configurations for holosoma_inference."""

from __future__ import annotations

from pathlib import Path

from holosoma_inference.config.config_types.task import TaskConfig

_MODELS_DIR = Path(__file__).parent.parent.parent / "models"

# Locomotion task
locomotion = TaskConfig(
    model_path="",  # Must be provided by user
    policy_mode="auto",
    rl_rate=50,
    policy_action_scale=0.25,
    use_phase=True,
    gait_period=1.0,
    desired_base_height=0.75,
    residual_upper_body_action=False,
    domain_id=0,
    interface="lo",
    use_joystick=False,
    joystick_type="xbox",
    joystick_device=0,
    use_ros=False,
    wandb_download_dir="/tmp",
    randomize_commands=False,
    command_resampling_time=4.0,
    command_lin_vel_x_range=(-1.0, 1.0),
    command_lin_vel_y_range=(-1.0, 1.0),
    command_ang_vel_range=(-1.0, 1.0),
    command_stand_prob=0.2,
    initial_zero_command_steps=0,
    seed=None,
    auto_start_policy=True,
    dual_mode_session_enabled=False,
    dual_mode_start_in_secondary=True,
    dual_mode_test_start_delay_s=-1.0,
    dual_mode_post_secondary_hold_s=0.0,
    dual_mode_exit_after_post_hold=False,
    dual_mode_projected_gravity_z_min=-0.7,
    max_steps=-1,
    log_states=True,
    log_output_dir="auto",
    log_filename="state_log",
    log_exp_name=None,
    vel_state_source="zmq",
    vel_state_zmq_url="tcp://127.0.0.1:6000",
    vel_state_target_name="base",
    vel_state_orientation_order="xyzw",
    plot_mocap_raw_on_exit=True,
    mocap_plot_hz=50.0,
)

# Whole-body tracking task
wbt = TaskConfig(
    model_path="",  # Must be provided by user
    policy_mode="wbt",
    rl_rate=50,
    policy_action_scale=1.0,
    action_scales_by_effort_limit_over_p_gain=True,
    use_phase=False,
    gait_period=1.0,
    desired_base_height=0.75,
    residual_upper_body_action=False,
    domain_id=0,
    interface="lo",
    use_joystick=False,
    joystick_type="xbox",
    joystick_device=0,
    use_ros=False,
    wandb_download_dir="/tmp",
    seed=None,
    auto_start_policy=False,
    dual_mode_session_enabled=False,
    dual_mode_start_in_secondary=True,
    dual_mode_test_start_delay_s=-1.0,
    dual_mode_post_secondary_hold_s=0.0,
    dual_mode_exit_after_post_hold=False,
    dual_mode_projected_gravity_z_min=-0.7,
    max_steps=-1,
    vel_state_source="none",
)

# Safety locomotion (FastSAC) — used as default secondary for dual-mode
safety_locomotion_g1 = TaskConfig(
    model_path=str(_MODELS_DIR / "loco" / "g1_29dof" / "fastsac_g1_29dof.onnx"),
    rl_rate=50,
    policy_action_scale=0.25,
    use_phase=True,
    gait_period=1.0,
    desired_base_height=0.75,
    residual_upper_body_action=False,
    domain_id=0,
    interface="lo",
    use_joystick=False,
    joystick_type="xbox",
    joystick_device=0,
    use_ros=False,
    wandb_download_dir="/tmp",
    dual_mode_session_enabled=False,
    dual_mode_start_in_secondary=True,
    dual_mode_test_start_delay_s=-1.0,
    dual_mode_post_secondary_hold_s=0.0,
    dual_mode_exit_after_post_hold=False,
    dual_mode_projected_gravity_z_min=-0.7,
)

DEFAULTS = {
    "locomotion": locomotion,
    "wbt": wbt,
    "safety_locomotion_g1": safety_locomotion_g1,
}
