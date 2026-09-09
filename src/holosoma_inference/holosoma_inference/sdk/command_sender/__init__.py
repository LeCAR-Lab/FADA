from holosoma_inference.config.config_types.robot import RobotConfig

from .base import BasicCommandSender


def create_command_sender(config: RobotConfig, lcm=None):
    """
    Factory function to create the appropriate command sender based on configuration.

    Args:
        config: Robot configuration dictionary
        lcm: LCM instance (optional, for LCM-based senders)

    Returns:
        An instance of the appropriate command sender class
    """
    sdk_type = config.sdk_type

    if sdk_type == "unitree":
        from .unitree import UnitreeCommandSender

        return UnitreeCommandSender(config, lcm)
    if sdk_type == "booster":
        from .booster import BoosterCommandSender

        return BoosterCommandSender(config, lcm)
    raise ValueError(f"Unsupported SDK type: {sdk_type}")


# For backward compatibility
CommandSender = create_command_sender

__all__ = [
    "BasicCommandSender",
    "BoosterCommandSender",
    "CommandSender",  # Backward compatibility
    "UnitreeCommandSender",
    "create_command_sender",
]
