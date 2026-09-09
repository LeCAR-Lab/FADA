"""Base classes for action terms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.config_types.action import ActionTermCfg


class ActionTermBase(ABC):
    """Base class for action terms.

    Action terms process a portion of the raw action vector and apply it to
    the environment. This could involve:
    - Joint position/velocity/torque control
    - End-effector control
    - Task-space control
    - Any custom action processing

    Each action term manages its own action dimension and processing logic.
    """

    _raw_actions: torch.Tensor | None = None
    _processed_actions: torch.Tensor | None = None
    """
    Buffers for raw and processed actions.

    Subclasses should allocate these in their ``__init__`` implementations, e.g.::

        self._raw_actions = torch.zeros(env.num_envs, action_dim, device=env.device)
        self._processed_actions = torch.zeros(env.num_envs, action_dim, device=env.device)

    Then update them within ``process_actions``::

        self._raw_actions[:] = actions
        self._processed_actions[:] = <processed_version>
    """

    def __init__(self, cfg: ActionTermCfg, env: Any):
        """Initialize action term.

        Parameters
        ----------
        cfg : ActionTermCfg
            Configuration for this action term
        env : Any
            Environment instance (typically BaseTask or subclass)
        """
        self.cfg = cfg
        self.env = env

    def setup(self) -> None:
        """Setup action term after all managers are initialized.

        This method is called after all managers (including randomization manager)
        have been created and their setup() methods called. Use this for any
        initialization that depends on other managers being fully set up.

        Default implementation does nothing. Override in subclasses if needed.
        """
        # Default implementation - subclasses override for deferred initialization
        return

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Dimension of the action term.

        Returns
        -------
        int
            Number of action values this term expects
        """

    @property
    def raw_actions(self) -> torch.Tensor:
        """The input/raw actions sent to the term.

        Returns
        -------
        torch.Tensor
            Raw action tensor [num_envs, action_dim]
        """
        if self._raw_actions is None:
            return torch.zeros(self.env.num_envs, self.action_dim, device=self.env.device)
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """The actions computed by the term after processing.

        Returns
        -------
        torch.Tensor
            Processed action tensor [num_envs, action_dim]
        """
        if self._processed_actions is None:
            return torch.zeros(self.env.num_envs, self.action_dim, device=self.env.device)
        return self._processed_actions

    @abstractmethod
    def process_actions(self, actions: torch.Tensor) -> None:
        """Process raw actions.

        This method is called once per environment step to pre-process the raw
        actions. The processed actions are stored internally and applied during
        the simulation step.

        Parameters
        ----------
        actions : torch.Tensor
            Raw action tensor [num_envs, action_dim]
        """

    @abstractmethod
    def apply_actions(self) -> None:
        """Apply processed actions to the environment.

        This method is called at every simulation step to apply the processed
        actions to the environment (e.g., set joint targets, forces, etc.).
        """

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset action term state for specified environments.

        Parameters
        ----------
        env_ids : torch.Tensor or None, optional
            Environment IDs to reset. If None, reset all environments.
        """
        # Default implementation: reset action buffers
        if env_ids is None:
            if self._raw_actions is not None:
                self._raw_actions.zero_()
            if self._processed_actions is not None:
                self._processed_actions.zero_()
        else:
            if self._raw_actions is not None:
                self._raw_actions[env_ids] = 0.0
            if self._processed_actions is not None:
                self._processed_actions[env_ids] = 0.0
