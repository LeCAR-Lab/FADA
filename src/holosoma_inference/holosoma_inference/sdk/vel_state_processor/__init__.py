from holosoma_inference.config.config_types.task import TaskConfig

from .basic_vel_state_processor import BasicVelStateProcessor
from .zmq_vel_state_processor import ZMQVelStateProcessor


def create_vel_state_processor(task_config: TaskConfig) -> BasicVelStateProcessor | None:
    """Factory for velocity state processors based on task configuration."""
    source = getattr(task_config, "vel_state_source", "none")
    if source in (None, "none"):
        return None

    if source == "zmq":
        return ZMQVelStateProcessor(
            target_name=getattr(task_config, "vel_state_target_name", "base"),
            zmq_address=getattr(task_config, "vel_state_zmq_url", "tcp://127.0.0.1:6000"),
            orientation_order=getattr(task_config, "vel_state_orientation_order", "xyzw"),
            record_mocap_history=getattr(task_config, "plot_mocap_raw_on_exit", True),
            mocap_plot_hz=float(getattr(task_config, "mocap_plot_hz", 50.0)),
            plot_trim_head=int(getattr(task_config, "plot_trim_head", 0)),
            plot_trim_tail=int(getattr(task_config, "plot_trim_tail", 0)),
            smooth_window=int(getattr(task_config, "plot_smooth_window", 0)),
            max_linear_vel=float(getattr(task_config, "plot_max_linear_vel", 1.0)),
            max_angular_vel=float(getattr(task_config, "plot_max_angular_vel", 2.0)),
        )

    raise ValueError(f"Unsupported velocity state source '{source}'")


__all__ = [
    "BasicVelStateProcessor",
    "ZMQVelStateProcessor",
    "create_vel_state_processor",
]
