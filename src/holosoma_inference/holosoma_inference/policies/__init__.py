from .base import BasePolicy
from .locomotion import LocomotionPolicy
from .wbt import WholeBodyTrackingPolicy

__all__ = ["BasePolicy", "LocomotionPolicy", "WholeBodyTrackingPolicy"]

from .dual_mode import DualModePolicy
from .locomotion import LocomotionPolicy_Deploy
from .locomotion_fada import LocomotionPolicy_FADA

__all__ += [
    "DualModePolicy",
    "LocomotionPolicy_Deploy",
    "LocomotionPolicy_FADA",
]
