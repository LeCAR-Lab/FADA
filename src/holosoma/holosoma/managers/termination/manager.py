"""Termination manager for aggregating termination conditions."""

from __future__ import annotations

import importlib
from typing import Any

from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg
from holosoma.utils.safe_torch_import import torch

from .base import TerminationTermBase


class TerminationManager:
    """Evaluate termination terms and aggregate reset/timeout flags.

    The termination manager runs the configured termination terms each
    step, combining their boolean outputs into ``reset`` and ``timeout``
    masks. Terms can be functions or subclasses of
    :class:`TerminationTermBase`, allowing environments to avoid
    duplicating termination logic.

    Parameters
    ----------
    cfg : TerminationManagerCfg
        Termination manager configuration describing available terms.
    env : Any
        Environment instance whose state is evaluated for termination.
    device : str
        Device identifier for tensors created in the manager.
    """

    def __init__(self, cfg: TerminationManagerCfg, env: Any, device: str):
        self.cfg = cfg
        self.env = env
        self.device = device
        self.logger = getattr(env, "logger", None)

        self._term_funcs: dict[str, Any] = {}
        self._term_instances: dict[str, TerminationTermBase] = {}
        self._term_names: list[str] = []
        self._term_cfgs: list[TerminationTermCfg] = []

        # Expose non-timeout / timeout termination masks so that downstream
        # consumers (e.g. adaptive motion sampling in MotionCommand) can
        # distinguish failure-triggered resets from timeouts.  This mirrors
        # IsaacLab's TerminationManager API which provides the same fields.
        self.terminated = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
        self.time_outs = torch.zeros_like(self.terminated)
        self._last_term_results: dict[str, torch.Tensor] = {}
        self._last_term_is_timeout: dict[str, bool] = {}

        self._initialize_terms()

    def _initialize_terms(self) -> None:
        for term_name, term_cfg in self.cfg.terms.items():
            func = self._resolve_function(term_cfg.func)

            if isinstance(func, type) and issubclass(func, TerminationTermBase):
                instance = func(term_cfg, self.env)
                self._term_instances[term_name] = instance
            else:
                self._term_funcs[term_name] = func

            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)

    def _resolve_function(self, func: Any | str) -> Any:
        if isinstance(func, str):
            if ":" not in func:
                raise ValueError(f"Function string must be in format 'module:function', got: {func}")

            module_path, func_name = func.split(":", 1)
            try:
                module = importlib.import_module(module_path)
                return getattr(module, func_name)
            except (ImportError, AttributeError) as exc:
                raise ValueError(f"Failed to import termination function '{func}': {exc}") from exc
        return func

    def check(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate termination terms.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            reset_flags, timeout_flags
        """
        reset_flags = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
        timeout_flags = torch.zeros_like(reset_flags)

        for term_name, term_cfg in zip(self._term_names, self._term_cfgs):
            if term_name in self._term_instances:
                result = self._term_instances[term_name](self.env, **term_cfg.params)
            else:
                result = self._term_funcs[term_name](self.env, **term_cfg.params)

            if result.dtype != torch.bool:
                raise TypeError(
                    f"Termination term '{term_name}' returned dtype {result.dtype}, expected torch.bool tensor."
                )

            if term_cfg.is_timeout:
                timeout_flags |= result
            else:
                reset_flags |= result
            self._last_term_results[term_name] = result.detach().clone()
            self._last_term_is_timeout[term_name] = bool(term_cfg.is_timeout)

        self.terminated = reset_flags.clone()
        self.time_outs = timeout_flags.clone()
        return reset_flags, timeout_flags

    def summarize_newly_done(self, newly_done: torch.Tensor) -> dict[str, Any]:
        """Summarize which termination terms fired for the just-finished envs."""
        if newly_done.dtype != torch.bool:
            newly_done = newly_done.bool()
        newly_done = newly_done.to(device=self.device)
        term_done_counts: dict[str, int] = {}
        primary_done_counts: dict[str, int] = {}
        overlap_events = 0

        fired_masks: list[tuple[str, torch.Tensor]] = []
        for term_name in self._term_names:
            result = self._last_term_results.get(term_name)
            if result is None:
                continue
            mask = result.to(device=self.device, dtype=torch.bool) & newly_done
            count = int(mask.sum().item())
            term_done_counts[term_name] = count
            if count > 0:
                fired_masks.append((term_name, mask))

        if fired_masks:
            stacked = torch.stack([mask for _, mask in fired_masks], dim=0)
            overlap_events = int((stacked.sum(dim=0) > 1).sum().item())
            unassigned = newly_done.clone()
            for term_name, mask in fired_masks:
                primary_mask = mask & unassigned
                primary_done_counts[term_name] = int(primary_mask.sum().item())
                unassigned = unassigned & ~primary_mask

        return {
            "term_done_counts": term_done_counts,
            "primary_done_counts": primary_done_counts,
            "overlap_events": overlap_events,
        }

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset stateful terms.

        Parameters
        ----------
        env_ids : torch.Tensor or None, optional
            Environment IDs to reset. If ``None``, reset all environments.
        """
        for instance in self._term_instances.values():
            instance.reset(env_ids=env_ids)

        if env_ids is None:
            self.terminated.zero_()
            self.time_outs.zero_()
        else:
            self.terminated[env_ids] = False
            self.time_outs[env_ids] = False
