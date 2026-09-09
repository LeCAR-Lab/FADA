"""Robot communication package."""

from __future__ import annotations

from importlib.metadata import entry_points

from loguru import logger

# Auto-discover SDK interfaces from installed packages using lazy loading.
# Lazy loading is to avoid errors from SDK dependencies from extensions (e.g. ROS2) when working with other SDKs.
_entry_points = {ep.name: ep for ep in entry_points(group="holosoma.sdk")}
_registry = {}  # Cache for loaded interfaces


def create_interface(robot_config, domain_id=0, interface_str=None, use_joystick=True, task_config=None):
    """Create interface from registry.

    If *interface_str* is ``"auto"``, the network interface is resolved
    automatically via :func:`holosoma_inference.utils.network.detect_robot_interface`.

    *task_config* is forwarded when the SDK class supports it (e.g. ZMQ mocap / vel_state).
    """
    robot_family = str(getattr(robot_config, "robot", "") or "").strip().lower().replace("-", "_")

    # Resolve "auto" interface before passing to the SDK backend
    if interface_str == "auto":
        from holosoma_inference.utils.network import detect_robot_interface

        interface_str = detect_robot_interface(robot_family)
    elif interface_str not in {None, "", "lo", "lo0"}:
        from holosoma_inference.utils.network import validate_robot_interface

        ok, detail = validate_robot_interface(interface_str, robot_family)
        if ok:
            logger.info(f"[network] {detail}")
        else:
            logger.warning(f"[network] {detail}")

    sdk_type = robot_config.sdk_type
    if sdk_type not in _entry_points:
        raise ValueError(f"Unknown sdk_type: {sdk_type}. Available: {sorted(_entry_points.keys())}")

    # Lazy load: only load the entry point when actually needed
    if sdk_type not in _registry:
        _registry[sdk_type] = _entry_points[sdk_type].load()

    iface_cls = _registry[sdk_type]
    return iface_cls(robot_config, domain_id, interface_str, use_joystick, task_config)


__all__ = [
    "create_interface",
]
