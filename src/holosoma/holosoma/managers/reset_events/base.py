"""Base protocol for reset events.

This module defines the protocol interface that reset events must implement
to be managed by the ResetEventManager.
"""

from typing import Protocol, runtime_checkable

from holosoma.simulator.types import EnvIds


@runtime_checkable
class ResetEvent(Protocol):
    """Protocol for reset events that can be managed by ResetEventManager.

    Defines the interface that all reset events must implement to be compatible
    with the reset event management system.
    """

    def reset(self, env_ids: EnvIds) -> None:
        """Execute the reset event for specified environment IDs."""
        ...

    @property
    def name(self) -> str:
        """Unique identifier for this reset event.

        Returns
        -------
        str
            The unique name identifier for this reset event.
        """
        ...
