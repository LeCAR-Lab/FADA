import sys
from importlib.metadata import entry_points

from .base import BasicSdk2Bridge

# Auto-discover bridge implementations from installed extensions
# Handle Python 3.8/3.9 vs 3.10+ API difference for entry_points
if sys.version_info >= (3, 10):
    _bridge_registry = {ep.name: ep.load for ep in entry_points(group="holosoma.bridge")}
else:
    _eps = entry_points()
    _bridge_eps = _eps.get("holosoma.bridge", [])
    _bridge_registry = {ep.name: ep.load for ep in _bridge_eps}


def create_sdk2py_bridge(simulator, robot_config, bridge_config, lcm=None):
    """
    Factory function to create the appropriate SDK2Py bridge based on configuration.

    Uses entry points for SDK selection, allowing extensions to register their own
    bridge implementations without modifying the main codebase.

    Args:
        simulator: BaseSimulator instance (simulator-agnostic)
        robot_config: Robot configuration dataclass (with .bridge containing RobotBridgeConfig)
        bridge_config: Bridge configuration dataclass (simulator-level settings)
        lcm: LCM instance (optional, for LCM-based bridges)

    Returns:
        An instance of the appropriate bridge class
    """
    sdk_type = robot_config.bridge.sdk_type

    if sdk_type not in _bridge_registry:
        raise ValueError(f"Unsupported SDK type: {sdk_type}. Available: {list(_bridge_registry.keys())}")

    # Lazy load the bridge class
    bridge_cls = _bridge_registry[sdk_type]()
    return bridge_cls(simulator, robot_config, bridge_config, lcm)


__all__ = [
    "BasicSdk2Bridge",
    "create_sdk2py_bridge",
]
