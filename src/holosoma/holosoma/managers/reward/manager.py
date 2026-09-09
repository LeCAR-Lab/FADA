"""Reward manager for computing reward signals."""

from __future__ import annotations

import importlib
from typing import Any

import torch

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg
from holosoma.config_types.reward import MultiAgentRewardManagerCfg, MultiAgentRewardTermCfg

from .base import RewardTermBase


class RewardManager:
    """Manages reward computation as a weighted sum of individual terms.

    The reward manager computes the total reward by evaluating each configured
    reward term, multiplying by its weight and the environment's time step (dt),
    and summing the results. It tracks episodic sums for logging and supports
    both stateless (function) and stateful (class) reward terms.

    Parameters
    ----------
    cfg : RewardManagerCfg
        Configuration specifying reward terms and settings.
    env : Any
        Environment instance (typically a ``BaseTask`` subclass).
    device : str
        Device where tensors should be allocated.
    """

    def __new__(cls, cfg: RewardManagerCfg, env: Any, device: str):
        # FADA: dispatch to MultiAgentRewardManager for MultiAgentRewardManagerCfg
        # configs without requiring callers (e.g. BaseTask) to branch on cfg type.
        # Returning a bare object() of the subtype (not a fully-constructed
        # instance) avoids double-running __init__: Python's normal protocol
        # then calls MultiAgentRewardManager.__init__ exactly once on it.
        if cls is RewardManager and isinstance(cfg, MultiAgentRewardManagerCfg):
            return object.__new__(MultiAgentRewardManager)
        return super().__new__(cls)

    def __init__(self, cfg: RewardManagerCfg, env: Any, device: str):
        self.cfg = cfg
        self.env = env
        self.device = device
        self.logger = getattr(env, "logger", None)

        # Storage for resolved functions and stateful terms
        self._term_funcs: dict[str, Any] = {}
        self._term_instances: dict[str, RewardTermBase] = {}
        self._term_names: list[str] = []
        self._term_cfgs: list[RewardTermCfg] = []

        # Initialize terms
        self._initialize_terms()

        # Buffers for reward tracking
        self._reward_buf = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Episode sums for each term (for logging)
        self._episode_sums: dict[str, torch.Tensor] = {}
        self._episode_sums_raw: dict[str, torch.Tensor] = {}
        self.last_scaled_term_rewards: dict[str, torch.Tensor] = {}
        self.last_raw_term_rewards: dict[str, torch.Tensor] = {}
        for term_name in self._term_names:
            self._episode_sums[term_name] = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            self._episode_sums_raw[term_name] = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

    _TERM_ALIASES = {
        "track_lin_vel_xy": "tracking_lin_vel",
        "track_ang_vel_z": "tracking_ang_vel",
    }

    def _add_episode_aliases(self, metrics: dict[str, torch.Tensor], prefix: str) -> None:
        for source_name, alias_name in self._TERM_ALIASES.items():
            source_key = f"{prefix}{source_name}"
            alias_key = f"{prefix}{alias_name}"
            if source_key in metrics:
                metrics.setdefault(alias_key, metrics[source_key])

    def _initialize_terms(self) -> None:
        """Initialize reward terms and resolve their functions/classes."""
        for term_name, term_cfg in self.cfg.terms.items():
            # Skip terms with zero weight
            if term_cfg.weight == 0.0:
                continue

            # Resolve function or class
            func = self._resolve_function(term_cfg.func)

            # Check if it's a class (stateful) or function (stateless)
            if isinstance(func, type) and issubclass(func, RewardTermBase):
                # Stateful term - instantiate
                instance = func(term_cfg, self.env)
                self._term_instances[term_name] = instance
            else:
                # Stateless function
                self._term_funcs[term_name] = func

            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)

    def _resolve_function(self, func: Any | str) -> Any:
        """Resolve a reward callable or class from a string specification.

        Parameters
        ----------
        func : Any or str
            Function or class reference, or a string like ``"module:object_name"``.

        Returns
        -------
        Any
            Resolved callable or class.

        Raises
        ------
        ValueError
            If the string path is malformed or the target cannot be imported.
        """
        if isinstance(func, str):
            # Parse string like "module.path:function_name"
            if ":" not in func:
                raise ValueError(f"Function string must be in format 'module:function', got: {func}")

            module_path, func_name = func.split(":", 1)
            try:
                module = importlib.import_module(module_path)
                return getattr(module, func_name)
            except (ImportError, AttributeError) as e:
                raise ValueError(f"Failed to import function '{func}': {e}") from e
        return func

    @property
    def active_terms(self) -> list[str]:
        """Names of active reward terms."""
        return self._term_names

    @property
    def episode_sums(self) -> dict[str, torch.Tensor]:
        """Episodic sums for each reward term (scaled)."""
        return self._episode_sums

    @property
    def episode_sums_raw(self) -> dict[str, torch.Tensor]:
        """Episodic sums for each reward term (raw, unscaled)."""
        return self._episode_sums_raw

    def compute(self, dt: float) -> torch.Tensor:
        """Compute the total reward as a weighted sum of individual terms.

        Each reward term is evaluated, scaled by its configured weight and the
        environment time step, and accumulated into the total reward. Episodic
        sums are updated for logging purposes.

        Notes
        -----
        Curriculum scaling is handled by directly modifying term weights via
        :meth:`set_term_cfg`, rather than through extra scaling parameters.

        Parameters
        ----------
        dt : float
            Environment time-step interval.

        Returns
        -------
        torch.Tensor
            Net reward tensor with shape ``[num_envs]``.
        """
        # Reset computation
        self._reward_buf[:] = 0.0
        self.last_scaled_term_rewards = {}
        self.last_raw_term_rewards = {}

        # Iterate over all reward terms
        for term_name, term_cfg in zip(self._term_names, self._term_cfgs):
            # Compute raw reward value
            if term_name in self._term_instances:
                # Stateful term
                instance = self._term_instances[term_name]
                rew_raw = instance(self.env, **term_cfg.params)
            else:
                # Stateless function
                func = self._term_funcs[term_name]
                rew_raw = func(self.env, **term_cfg.params)

            # Validate shape
            if rew_raw.shape[0] != self.env.num_envs:
                raise ValueError(
                    f"Reward term '{term_name}' returned wrong shape. "
                    f"Expected [{self.env.num_envs}], got {rew_raw.shape}"
                )

            # Scale by weight and dt
            rew_scaled = rew_raw * term_cfg.weight * dt
            self.last_scaled_term_rewards[term_name] = rew_scaled.detach().clone()
            self.last_raw_term_rewards[term_name] = rew_raw.detach().clone()

            # Accumulate
            self._reward_buf += rew_scaled

            # Track episodic sums
            self._episode_sums[term_name] += rew_scaled
            self._episode_sums_raw[term_name] += rew_raw

        # Optionally clip to positive
        if self.cfg.only_positive_rewards:
            self._reward_buf[:] = torch.clip(self._reward_buf, min=0.0)

        return self._reward_buf

    def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, dict[str, torch.Tensor]]:
        """Reset reward tracking and return episodic sums for logging.

        Parameters
        ----------
        env_ids : torch.Tensor or None, optional
            Environment IDs to reset. If ``None``, reset all environments.

        Returns
        -------
        dict[str, dict[str, torch.Tensor]]
            Dictionary mirroring the direct reward path structure::

                {
                    "episode": {term_name: tensor_per_reset_env},
                    "episode_all": {term_name: tensor_per_all_envs},
                    "raw_episode": {...},
                    "raw_episode_all": {...},
                }
        """
        extras: dict[str, dict[str, torch.Tensor]] = {
            "episode": {},
            "episode_all": {},
            "raw_episode": {},
            "raw_episode_all": {},
        }

        # Resolve environment ids to operate on
        if env_ids is None:
            env_ids_tensor: torch.Tensor | None = None
            env_ids_slice: slice | torch.Tensor = slice(None)
        else:
            if isinstance(env_ids, torch.Tensor):
                env_ids_tensor = env_ids.to(device=self.device, dtype=torch.long)
            else:
                env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

            env_ids_slice = env_ids_tensor

        # Helper to detach values before zeroing internal buffers
        def _clone(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.detach().clone()

        # Normalize by max episode length (matches FAR-Holosoma reference). A constant
        # divisor avoids dividing by a per-env length that is still 0 (before an env's first
        # reset, or after init_at_random_ep_len).
        # Interpretation: "fraction of the max-possible reward over a full max-length episode."
        max_ep_len_s = max(float(self.env.max_episode_length_s), float(self.env.dt))

        # Populate scaled reward statistics
        for term_name in self._term_names:
            rew_all = self._episode_sums[term_name] / max_ep_len_s
            extras["episode_all"][f"rew_{term_name}"] = _clone(rew_all)
            if env_ids_tensor is None:
                extras["episode"][f"rew_{term_name}"] = _clone(rew_all)
            else:
                extras["episode"][f"rew_{term_name}"] = _clone(rew_all[env_ids_slice])

            # Reset episodic sums for the completed environments
            self._episode_sums[term_name][env_ids_slice] = 0.0

        # Populate raw (unscaled) reward statistics
        for term_name in self._term_names:
            rew_raw_all = self._episode_sums_raw[term_name] / max_ep_len_s
            extras["raw_episode_all"][f"raw_rew_{term_name}"] = _clone(rew_raw_all)
            if env_ids_tensor is None:
                extras["raw_episode"][f"raw_rew_{term_name}"] = _clone(rew_raw_all)
            else:
                extras["raw_episode"][f"raw_rew_{term_name}"] = _clone(rew_raw_all[env_ids_slice])

            self._episode_sums_raw[term_name][env_ids_slice] = 0.0

        self._add_episode_aliases(extras["episode"], prefix="rew_")
        self._add_episode_aliases(extras["episode_all"], prefix="rew_")
        self._add_episode_aliases(extras["raw_episode"], prefix="raw_rew_")
        self._add_episode_aliases(extras["raw_episode_all"], prefix="raw_rew_")

        # Reset stateful reward terms
        for instance in self._term_instances.values():
            instance.reset(env_ids=env_ids_tensor)

        return extras

    def get_episode_rates(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return reward rate for the given envs, normalized by max episode length.

        Use this to include envs that have not yet reset in episode logging, so the
        mean is over all envs (completed + ongoing), not just failed/short episodes.

        Normalization matches :meth:`reset` (and FAR-Holosoma reference) — divides by
        ``max_episode_length_s`` so per-term values are comparable across short / long
        episodes and across the same vs. different runs.

        Parameters
        ----------
        env_ids : torch.Tensor
            Environment indices, shape (n,) or (n, 1).

        Returns
        -------
        dict[str, torch.Tensor]
            Keys like ``rew_{term_name}``, values shape (n,) with normalized rate per env.
        """
        if env_ids.numel() == 0:
            return {}
        env_ids = env_ids.flatten().to(device=self.device)
        max_ep_len_s = max(float(self.env.max_episode_length_s), float(self.env.dt))
        out: dict[str, torch.Tensor] = {}
        for term_name in self._term_names:
            out[f"rew_{term_name}"] = (self._episode_sums[term_name][env_ids] / max_ep_len_s).detach()
        self._add_episode_aliases(out, prefix="rew_")
        return out

    def get_raw_episode_rates(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return raw (unscaled) reward rate, normalized by ``max_episode_length_s``."""
        if env_ids.numel() == 0:
            return {}
        env_ids = env_ids.flatten().to(device=self.device)
        max_ep_len_s = max(float(self.env.max_episode_length_s), float(self.env.dt))
        out: dict[str, torch.Tensor] = {}
        for term_name in self._term_names:
            out[f"raw_rew_{term_name}"] = (self._episode_sums_raw[term_name][env_ids] / max_ep_len_s).detach()
        self._add_episode_aliases(out, prefix="raw_rew_")
        return out

    def get_term(self, name: str) -> Any:
        """Get reward term function or instance by name.

        Parameters
        ----------
        name : str
            Name of the reward term.

        Returns
        -------
        Any
            Reward term function or instance.

        Raises
        ------
        KeyError
            If the term name is not found.
        """
        if name in self._term_instances:
            return self._term_instances[name]
        if name in self._term_funcs:
            return self._term_funcs[name]
        raise KeyError(f"Reward term '{name}' not found")

    def get_term_cfg(self, name: str) -> RewardTermCfg:
        """Get reward term configuration by name.

        Parameters
        ----------
        name : str
            Name of the reward term.

        Returns
        -------
        RewardTermCfg
            Configuration for the specified reward term.

        Raises
        ------
        KeyError
            If the term name is not found.
        """
        try:
            idx = self._term_names.index(name)
            return self._term_cfgs[idx]
        except ValueError:
            raise KeyError(f"Reward term '{name}' not found")

    def set_term_cfg(self, name: str, cfg: RewardTermCfg) -> None:
        """Set reward term configuration by name.

        Parameters
        ----------
        name : str
            Name of the reward term.
        cfg : RewardTermCfg
            New configuration for the term.

        Raises
        ------
        KeyError
            If the term name is not found.
        """
        try:
            idx = self._term_names.index(name)
            self._term_cfgs[idx] = cfg
        except ValueError:
            raise KeyError(f"Reward term '{name}' not found")

    def __str__(self) -> str:
        """String representation of reward manager."""
        msg = f"<RewardManager> contains {len(self._term_names)} active terms.\n"
        msg += "Terms:\n"
        for name, cfg in zip(self._term_names, self._term_cfgs):
            msg += f"  - {name}: weight={cfg.weight}\n"
        return msg


class MultiAgentRewardManager(RewardManager):
    """Reward manager that also accumulates per-body scaled rewards for decoupled PPO.

    Subclasses :class:`RewardManager` and overrides :meth:`compute` to additionally
    fill ``self.last_multi_agent_rewards``: a dict keyed by body group
    (``lower_body``, ``upper_body``, ...) that PPO-MA reads each step via
    ``extras["rewards_ma"]`` to drive per-body advantage / value learning.

    Each :class:`MultiAgentRewardTermCfg` term carries an ``ma_reward_group``
    in {``lower_body``, ``upper_body``, ``shared``, ``None``}. ``None`` and
    ``shared`` add the scaled reward to every group buffer; a specific group
    adds only to that buffer. Plain :class:`RewardTermCfg` terms (no group
    field) are treated as ``shared``.
    """

    def __init__(self, cfg: MultiAgentRewardManagerCfg, env: Any, device: str):
        super().__init__(cfg, env, device)
        self._ma_cfg = cfg
        self._ma_bufs: dict[str, torch.Tensor] | None = None
        self.last_multi_agent_rewards: dict[str, torch.Tensor] = {}
        keys = cfg.multi_agent_body_keys
        if keys:
            self._ma_bufs = {
                k: torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device) for k in keys
            }

    def compute(self, dt: float) -> torch.Tensor:
        self._reward_buf[:] = 0.0
        self.last_scaled_term_rewards = {}
        self.last_raw_term_rewards = {}
        if self._ma_bufs is not None:
            for b in self._ma_bufs.values():
                b.zero_()

        for term_name, term_cfg in zip(self._term_names, self._term_cfgs):
            if term_name in self._term_instances:
                instance = self._term_instances[term_name]
                rew_raw = instance(self.env, **term_cfg.params)
            else:
                func = self._term_funcs[term_name]
                rew_raw = func(self.env, **term_cfg.params)

            if rew_raw.shape[0] != self.env.num_envs:
                raise ValueError(
                    f"Reward term '{term_name}' returned wrong shape. "
                    f"Expected [{self.env.num_envs}], got {rew_raw.shape}"
                )

            rew_scaled = rew_raw * term_cfg.weight * dt
            self.last_scaled_term_rewards[term_name] = rew_scaled.detach().clone()
            self.last_raw_term_rewards[term_name] = rew_raw.detach().clone()
            self._reward_buf += rew_scaled

            if self._ma_bufs is not None:
                grp = (
                    term_cfg.ma_reward_group
                    if isinstance(term_cfg, MultiAgentRewardTermCfg)
                    else None
                )
                grp = grp or "shared"
                if grp == "shared":
                    for b in self._ma_bufs.values():
                        b += rew_scaled
                elif grp in self._ma_bufs:
                    self._ma_bufs[grp] += rew_scaled
                else:
                    raise ValueError(
                        f"Reward term '{term_name}' has ma_reward_group={grp!r} but "
                        f"multi_agent_body_keys={self._ma_cfg.multi_agent_body_keys!r}"
                    )

            self._episode_sums[term_name] += rew_scaled
            self._episode_sums_raw[term_name] += rew_raw

        if self.cfg.only_positive_rewards:
            self._reward_buf[:] = torch.clip(self._reward_buf, min=0.0)

        if self._ma_bufs is not None:
            self.last_multi_agent_rewards = {k: v.clone() for k, v in self._ma_bufs.items()}
        else:
            self.last_multi_agent_rewards = {}

        return self._reward_buf
