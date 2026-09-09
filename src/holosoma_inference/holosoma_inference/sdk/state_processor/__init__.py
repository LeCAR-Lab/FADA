from holosoma_inference.config.config_types.robot import RobotConfig

from .base import BasicStateProcessor


def create_state_processor(config: RobotConfig, lcm=None):
    """
    Factory function to create the appropriate state processor based on configuration.

    Args:
        config: Robot configuration dictionary
        lcm: LCM instance (optional, for LCM-based processors)

    Returns:
        An instance of the appropriate state processor class
    """
    sdk_type = config.sdk_type

    if sdk_type == "unitree":
        from .unitree import UnitreeStateProcessor

        return UnitreeStateProcessor(config, lcm)
    if sdk_type == "booster":
        from .booster import BoosterStateProcessor

        return BoosterStateProcessor(config, lcm)
    if sdk_type in ["lcm_unitree", "lcm_booster"]:
        raise ValueError(
            f"LCM SDK types are no longer supported. Please use 'unitree' or 'booster' instead of '{sdk_type}'"
        )
    raise ValueError(f"Unsupported SDK type: {sdk_type}")


# For backward compatibility
StateProcessor = create_state_processor

__all__ = [
    "BasicStateProcessor",
    "BoosterStateProcessor",
    "StateProcessor",  # Backward compatibility
    "UnitreeStateProcessor",
    "create_state_processor",
]
