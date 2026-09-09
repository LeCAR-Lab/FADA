"""Curriculum hooks for locomotion tasks."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import torch

from holosoma.managers.curriculum.base import CurriculumTermBase


class AverageEpisodeLengthTracker(CurriculumTermBase):
    """Track moving average of episode length for locomotion tasks."""

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        params = cfg.params or {}
        base_num_compute_average_epl = float(params.get("num_compute_average_epl", 1000))
        base_denominator = getattr(env, "BASE_NUM_ENVS", env.num_envs)
        self.num_compute_average_epl = max(1, int(base_num_compute_average_epl * env.num_envs / base_denominator))
        self.average_episode_length = torch.tensor(0.0, device=env.device, dtype=torch.float)
        self._suppress_next_update = False

    def setup(self) -> None:
        self.average_episode_length = torch.as_tensor(
            float(self.average_episode_length), device=self.env.device, dtype=torch.float
        )

    def reset(self, env_ids) -> None:
        if env_ids is None:
            return

        if not torch.is_tensor(env_ids):
            env_ids_tensor = torch.as_tensor(env_ids, device=self.env.device, dtype=torch.long)
        else:
            env_ids_tensor = env_ids.to(device=self.env.device, dtype=torch.long)

        if env_ids_tensor.numel() == 0:
            return

        pending = self.env._pending_episode_lengths
        mask_tensor = self.env._pending_episode_update_mask

        update_mask = mask_tensor.index_select(0, env_ids_tensor)
        if not torch.any(update_mask):
            return
        active_ids = env_ids_tensor[update_mask]

        if active_ids.numel() == 0:
            return

        episode_lengths = pending.index_select(0, active_ids).to(dtype=torch.float)
        if episode_lengths.numel() == 0:
            return

        self.update(active_ids, episode_lengths)

        zeros_long = torch.zeros(active_ids.shape, device=pending.device, dtype=pending.dtype)
        pending.index_copy_(0, active_ids, zeros_long)

        zeros_bool = torch.zeros(active_ids.shape, device=mask_tensor.device, dtype=mask_tensor.dtype)
        mask_tensor.index_copy_(0, active_ids, zeros_bool)

    def step(self) -> None:
        return

    def update(self, env_ids: torch.Tensor, episode_lengths: torch.Tensor) -> None:
        if self._suppress_next_update:
            self._suppress_next_update = False
            return

        num = env_ids.numel()
        if num == 0:
            return
        current_average = torch.mean(episode_lengths.to(dtype=torch.float), dtype=torch.float)
        weight = min(num / self.num_compute_average_epl, 1.0)
        self.average_episode_length = self.average_episode_length * (1 - weight) + current_average * weight

    def suppress_next_update(self) -> None:
        self._suppress_next_update = True

    def get_average(self) -> torch.Tensor:
        return self.average_episode_length

    def set_average(self, value: float | torch.Tensor, *, suppress_update: bool = True) -> None:
        self.average_episode_length = torch.as_tensor(float(value), device=self.env.device, dtype=torch.float)
        if suppress_update:
            self._suppress_next_update = True

    def state_dict(self) -> dict[str, Any]:
        return {
            "average_episode_length": self.average_episode_length.detach().to("cpu"),
            "suppress_next_update": self._suppress_next_update,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        avg = state.get("average_episode_length")
        if avg is not None:
            self.average_episode_length = torch.as_tensor(float(avg), device=self.env.device, dtype=torch.float)
        self._suppress_next_update = bool(state.get("suppress_next_update", False))


class TerrainLevelCurriculum(CurriculumTermBase):
    """Per-env terrain difficulty curriculum (IsaacLab `terrain_levels_vel`-style).

    Each env starts at a low difficulty row of the terrain grid. After every
    episode termination, we look at how long the just-completed episode lasted
    (via ``env._pending_episode_lengths``) and either promote, demote, or hold
    the env's level. The new level is written back into the per-env
    ``env_origins`` so the *next* spawn lands on the new tile.

    Notes
    -----
    * `_reset_root_states` is called BEFORE `curriculum_manager.reset`, so the
      level update kicks in for the spawn AFTER next, not the immediate one
      (a 1-episode lag).
    * The col index per env is fixed (set deterministically by env_id at init)
      so we only ever vary the row (= difficulty).
    * Level state is checkpointed via state_dict/load_state_dict.
    """

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        params = cfg.params or {}
        self.enabled: bool = bool(params.get("enabled", True))
        self.level_up_threshold: float = float(params.get("level_up_threshold", 800.0))
        self.level_down_threshold: float = float(params.get("level_down_threshold", 300.0))
        self.initial_level: int = int(params.get("initial_level", 0))
        # When True, init each env's level uniformly at random over [0, num_rows). With the
        # short rollouts used for DAgger / data collection, ep_len never crosses up_threshold,
        # so level-up never fires and envs stay at L0.
        self.randomize_init_levels: bool = bool(params.get("randomize_init_levels", False))
        self.log_interval: int = int(params.get("log_interval", 100))
        # Cache for the per-env levels, computed lazily after terrain init.
        self.terrain_levels: torch.Tensor | None = None
        self._env_cols: torch.Tensor | None = None
        self._origin_grid: torch.Tensor | None = None
        self._num_rows: int = 0
        self._num_cols: int = 0
        self._setup_done = False
        self._reset_calls: int = 0

    # ------------------------------------------------------------------
    def _ensure_setup(self) -> None:
        if self._setup_done:
            return
        terrain_state = self.env.terrain_manager.get_state("locomotion_terrain")
        terrain = terrain_state.terrain
        self._num_rows = int(terrain.num_rows)
        self._num_cols = int(terrain.num_cols)
        # Bring the [num_rows, num_cols, 3] grid onto device.
        grid_np = np.asarray(terrain._env_origins, dtype=np.float32)
        self._origin_grid = torch.from_numpy(grid_np).to(self.env.device)
        # Fixed col per env: matches the original sample_env_origins layout.
        num_envs = self.env.num_envs
        cols = np.floor_divide(np.arange(num_envs), max(num_envs / self._num_cols, 1)).astype(np.int64)
        cols = np.clip(cols, 0, self._num_cols - 1)
        self._env_cols = torch.from_numpy(cols).to(self.env.device)
        # Levels start at initial_level for everyone, OR uniformly at random over all rows
        # when randomize_init_levels=True (e.g., DAgger / data collection).
        if self.randomize_init_levels:
            levels = torch.randint(
                low=0,
                high=self._num_rows,
                size=(num_envs,),
                device=self.env.device,
                dtype=torch.long,
            )
        else:
            levels = torch.full(
                (num_envs,),
                int(np.clip(self.initial_level, 0, self._num_rows - 1)),
                device=self.env.device,
                dtype=torch.long,
            )
        self.terrain_levels = levels
        # Write initial origins so envs spawn at the configured starting level.
        self._write_origins(torch.arange(num_envs, device=self.env.device, dtype=torch.long))
        self._setup_done = True

    def _write_origins(self, env_ids: torch.Tensor) -> None:
        """Update env_origins[env_ids] from current terrain_levels."""
        if not torch.is_tensor(env_ids) or env_ids.numel() == 0:
            return
        terrain_state = self.env.terrain_manager.get_state("locomotion_terrain")
        levels = self.terrain_levels.index_select(0, env_ids)
        cols = self._env_cols.index_select(0, env_ids)
        new_origins = self._origin_grid[levels, cols]
        terrain_state.env_origins[env_ids] = new_origins.to(terrain_state.env_origins.dtype)

    # ------------------------------------------------------------------
    def setup(self) -> None:
        # Terrain manager may not be ready yet; defer to first reset call.
        try:
            self._ensure_setup()
        except Exception:
            # Will retry on first reset.
            self._setup_done = False

    def reset(self, env_ids) -> None:
        if not self.enabled:
            return
        if env_ids is None:
            return
        if not torch.is_tensor(env_ids):
            env_ids_tensor = torch.as_tensor(env_ids, device=self.env.device, dtype=torch.long)
        else:
            env_ids_tensor = env_ids.to(device=self.env.device, dtype=torch.long)
        if env_ids_tensor.numel() == 0:
            return

        self._ensure_setup()

        # Pure-random mode: skip all level promotions/demotions and reassign EACH resetting
        # env to a fresh uniformly random terrain row. Used when randomize_init_levels=True
        # (DAgger / data collection on full terrain): each env's next episode lands on an
        # independently sampled difficulty.
        if self.randomize_init_levels:
            new_levels = torch.randint(
                low=0,
                high=self._num_rows,
                size=(env_ids_tensor.numel(),),
                device=self.env.device,
                dtype=torch.long,
            )
            self.terrain_levels.index_copy_(0, env_ids_tensor, new_levels)
            self._write_origins(env_ids_tensor)
            self._reset_calls += 1
            if self.log_interval > 0 and self._reset_calls % self.log_interval == 0:
                counts = torch.bincount(self.terrain_levels, minlength=self._num_rows)
                mean_lvl = float(self.terrain_levels.float().mean().item())
                counts_list = counts.cpu().tolist()
                print(
                    f"[TerrainLevelCurriculum] (pure-random) reset_call={self._reset_calls} "
                    f"mean_level={mean_lvl:.2f} dist={counts_list}"
                )
            return

        # Read the just-completed episode length for these envs.
        ep_lens = self.env._pending_episode_lengths.index_select(0, env_ids_tensor).to(torch.float)
        # Skip the very-first reset (ep_len=0) so we don't demote env at startup.
        non_zero_mask = ep_lens > 0
        if not torch.any(non_zero_mask):
            # Even on the startup reset, we still want to write the initial origins.
            self._write_origins(env_ids_tensor)
            return
        active_ids = env_ids_tensor[non_zero_mask]
        active_lens = ep_lens[non_zero_mask]

        # Decide level deltas.
        up_mask = active_lens >= self.level_up_threshold
        down_mask = active_lens < self.level_down_threshold
        deltas = torch.zeros_like(active_ids, dtype=torch.long)
        deltas[up_mask] = 1
        deltas[down_mask] = -1

        new_levels = self.terrain_levels.index_select(0, active_ids) + deltas
        new_levels = torch.clamp(new_levels, min=0, max=self._num_rows - 1)
        self.terrain_levels.index_copy_(0, active_ids, new_levels)

        # Write the new origins for ALL the envs being reset.
        self._write_origins(env_ids_tensor)

        # Periodic logging of level distribution.
        self._reset_calls += 1
        if self.log_interval > 0 and self._reset_calls % self.log_interval == 0:
            counts = torch.bincount(self.terrain_levels, minlength=self._num_rows)
            mean_lvl = float(self.terrain_levels.float().mean().item())
            counts_list = counts.cpu().tolist()
            print(
                f"[TerrainLevelCurriculum] reset_call={self._reset_calls} "
                f"mean_level={mean_lvl:.2f} dist={counts_list}"
            )

    def step(self) -> None:
        return

    # ------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        if self.terrain_levels is None:
            return {}
        return {
            "terrain_levels": self.terrain_levels.detach().cpu().tolist(),
            "num_rows": self._num_rows,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        levels_list = state.get("terrain_levels")
        if levels_list is None:
            return
        self._ensure_setup()
        if self.terrain_levels is None:
            return
        levels = torch.as_tensor(levels_list, device=self.env.device, dtype=torch.long)
        # Defensive size mismatch handling (e.g. resuming with different num_envs).
        n = min(levels.numel(), self.terrain_levels.numel())
        if n > 0:
            self.terrain_levels[:n] = levels[:n]
        # Re-apply origins.
        self._write_origins(torch.arange(self.terrain_levels.numel(), device=self.env.device, dtype=torch.long))


class PenaltyCurriculum(CurriculumTermBase):
    """Stateful penalty curriculum that scales reward term weights based on episode length.

    This curriculum term adaptively scales penalty reward weights during training.
    When episodes are short (robot falls quickly), penalties are reduced to make
    learning easier. As episodes get longer (robot stays up), penalties gradually
    increase to refine behavior.
    """

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)

        # Get parameters from config
        params = cfg.params
        self.enabled = params.get("enabled", True)
        self.tag = params.get("tag", "penalty_curriculum")
        self.min_scale = float(params.get("min_scale", 0.0))
        self.max_scale = float(params.get("max_scale", 1.0))
        self.level_down_threshold = float(params.get("level_down_threshold", 150.0))
        self.level_up_threshold = float(params.get("level_up_threshold", 850.0))
        self.degree = float(params.get("degree", 0.0))

        # State variables (previously stored on env)
        self.current_scale = float(params.get("initial_scale", 1.0))
        self.penalty_reward_names: list[str] = []
        self.original_weights: dict[str, float] = {}

    def setup(self) -> None:
        """Setup penalty curriculum - identify rewards and apply initial scaling."""
        if not self.enabled or not hasattr(self.env, "reward_manager"):
            return

        # Identify penalty rewards by tag using public reward manager APIs
        for term_name in self.env.reward_manager.active_terms:
            term_cfg = self.env.reward_manager.get_term_cfg(term_name)
            if self.tag in term_cfg.tags:
                self.penalty_reward_names.append(term_name)

        # Store original weights and apply initial scaling
        for name in self.penalty_reward_names:
            if name not in self.env.reward_manager.active_terms:
                continue

            term_cfg = self.env.reward_manager.get_term_cfg(name)
            # Store original weight
            self.original_weights[name] = float(term_cfg.weight)
            # Apply initial scale
            scaled_cfg = replace(term_cfg, weight=term_cfg.weight * self.current_scale)
            self.env.reward_manager.set_term_cfg(name, scaled_cfg)

        # Set flag for logging compatibility
        self.env.use_reward_penalty_curriculum = True
        self.env.reward_penalty_scale = self.current_scale

    def reset(self, env_ids) -> None:
        """Update penalty scale based on average episode length."""
        if not self.enabled or not hasattr(self.env, "reward_manager"):
            return

        if not self.penalty_reward_names or not self.original_weights:
            return

        average_length = float(self.env.average_episode_length)

        # Update current scale based on episode length
        if average_length < self.level_down_threshold:
            self.current_scale *= 1.0 - self.degree
        elif average_length > self.level_up_threshold:
            self.current_scale *= 1.0 + self.degree

        # Clamp scale
        self.current_scale = float(np.clip(self.current_scale, self.min_scale, self.max_scale))

        # Apply scale to each penalty reward's weight
        for name in self.penalty_reward_names:
            if name not in self.original_weights or name not in self.env.reward_manager.active_terms:
                continue
            term_cfg = self.env.reward_manager.get_term_cfg(name)
            # Set weight = original_weight * current_scale
            scaled_cfg = replace(term_cfg, weight=self.original_weights[name] * self.current_scale)
            self.env.reward_manager.set_term_cfg(name, scaled_cfg)

        # Update for logging
        self.env.reward_penalty_scale = self.current_scale

        # Update log_dict for WandB logging
        if hasattr(self.env, "log_dict"):
            self.env.log_dict["penalty_scale"] = torch.tensor(self.current_scale, dtype=torch.float)

    def step(self) -> None:
        """Clamp penalty scale within bounds each step."""
        if not self.enabled or not hasattr(self.env, "reward_manager"):
            return

        if not self.penalty_reward_names or not self.original_weights:
            return

        # Clamp current_scale
        self.current_scale = float(np.clip(self.current_scale, self.min_scale, self.max_scale))

        # Re-apply clamped scale to weights
        for name in self.penalty_reward_names:
            if name not in self.original_weights or name not in self.env.reward_manager.active_terms:
                continue
            term_cfg = self.env.reward_manager.get_term_cfg(name)
            scaled_cfg = replace(term_cfg, weight=self.original_weights[name] * self.current_scale)
            self.env.reward_manager.set_term_cfg(name, scaled_cfg)

    def state_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "current_scale": float(self.current_scale),
            "penalty_reward_names": list(self.penalty_reward_names),
            "original_weights": {k: float(v) for k, v in self.original_weights.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.enabled = bool(state.get("enabled", self.enabled))
        self.current_scale = float(state.get("current_scale", self.current_scale))
        names = state.get("penalty_reward_names")
        if isinstance(names, list):
            self.penalty_reward_names = [str(name) for name in names]
        original_weights = state.get("original_weights")
        if isinstance(original_weights, dict):
            self.original_weights = {str(k): float(v) for k, v in original_weights.items()}

        reward_manager = getattr(self.env, "reward_manager", None)
        if reward_manager is None:
            return

        for name in self.penalty_reward_names:
            if name not in self.original_weights or name not in reward_manager.active_terms:
                continue
            term_cfg = reward_manager.get_term_cfg(name)
            scaled_cfg = replace(term_cfg, weight=self.original_weights[name] * self.current_scale)
            reward_manager.set_term_cfg(name, scaled_cfg)

        self.env.reward_penalty_scale = self.current_scale
        if hasattr(self.env, "log_dict"):
            self.env.log_dict["penalty_scale"] = torch.tensor(self.current_scale, dtype=torch.float)


# ================================================================================================
# Legacy stateless functions (backward compatibility)
# ================================================================================================


def configure_reward_penalty(
    env,
    *,
    enabled: bool = True,
    tag: str = "penalty_curriculum",
    initial_scale: float = 1.0,
    min_scale: float = 0.0,
    max_scale: float = 1.0,
    level_down_threshold: float = 150.0,
    level_up_threshold: float = 750.0,
    degree: float = 0.0,
) -> None:
    """Configure reward-penalty curriculum parameters.

    This modifies the reward term weights directly in the reward manager,
    scaling them by initial_scale and storing the original weights for reference.

    Args:
        enabled: Whether to enable penalty curriculum
        tag: Tag to filter reward terms (e.g., "penalty_curriculum").
        initial_scale: Initial scaling factor
        min_scale: Minimum scaling factor
        max_scale: Maximum scaling factor
        level_down_threshold: Episode length threshold for decreasing penalty scale
        level_up_threshold: Episode length threshold for increasing penalty scale
        degree: Adjustment rate when updating penalty scale
    """
    env.use_reward_penalty_curriculum = bool(enabled)

    # Determine which rewards to apply curriculum to
    # Use tag-based selection
    penalty_names = []
    for term_name in env.reward_manager.active_terms:
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        if tag in term_cfg.tags:
            penalty_names.append(term_name)

    env._curriculum_penalty_reward_names = penalty_names

    # Store original weights and apply initial scaling
    env._curriculum_penalty_original_weights = {}
    if env.use_reward_penalty_curriculum and hasattr(env, "reward_manager"):
        for name in penalty_names:
            if name not in env.reward_manager.active_terms:
                continue
            term_cfg = env.reward_manager.get_term_cfg(name)
            # Store original weight
            env._curriculum_penalty_original_weights[name] = float(term_cfg.weight)
            # Apply initial scale
            scaled_cfg = replace(term_cfg, weight=term_cfg.weight * initial_scale)
            env.reward_manager.set_term_cfg(name, scaled_cfg)

    env._curriculum_penalty_cfg = {
        "min_scale": float(min_scale),
        "max_scale": float(max_scale),
        "level_down_threshold": float(level_down_threshold),
        "level_up_threshold": float(level_up_threshold),
        "degree": float(degree),
        "current_scale": float(initial_scale),
    }

    # Set reward_penalty_scale for logging compatibility
    env.reward_penalty_scale = float(initial_scale)


def update_reward_penalty(env, env_ids, **_) -> None:
    """Update penalty scale based on average episode length.

    Modifies reward term weights directly in the reward manager.
    """
    if not getattr(env, "use_reward_penalty_curriculum", False):
        return

    cfg = getattr(env, "_curriculum_penalty_cfg", None)
    penalty_names = getattr(env, "_curriculum_penalty_reward_names", [])
    original_weights = getattr(env, "_curriculum_penalty_original_weights", {})
    if not cfg or not penalty_names or not hasattr(env, "reward_manager"):
        return

    average_length = float(env.average_episode_length)
    degree = cfg["degree"]

    # Update current scale based on episode length
    current_scale = cfg["current_scale"]
    if average_length < cfg["level_down_threshold"]:
        current_scale *= 1.0 - degree
    elif average_length > cfg["level_up_threshold"]:
        current_scale *= 1.0 + degree

    # Clamp scale
    current_scale = float(np.clip(current_scale, cfg["min_scale"], cfg["max_scale"]))
    cfg["current_scale"] = current_scale

    # Apply scale to each penalty reward's weight
    for name in penalty_names:
        if name not in original_weights or name not in env.reward_manager.active_terms:
            continue
        term_cfg = env.reward_manager.get_term_cfg(name)
        # Set weight = original_weight * current_scale
        scaled_cfg = replace(term_cfg, weight=original_weights[name] * current_scale)
        env.reward_manager.set_term_cfg(name, scaled_cfg)

    # Update reward_penalty_scale for logging
    env.reward_penalty_scale = current_scale

    # Update log_dict for WandB logging
    if hasattr(env, "log_dict"):
        import torch

        env.log_dict["penalty_scale"] = torch.tensor(env.reward_penalty_scale, dtype=torch.float)


def clamp_reward_penalty(env, **_) -> None:
    """Ensure penalty scale stays within configured bounds each step.

    Re-applies clamping to reward weights in case of any drift.
    """
    if not getattr(env, "use_reward_penalty_curriculum", False):
        return

    cfg = getattr(env, "_curriculum_penalty_cfg", None)
    penalty_names = getattr(env, "_curriculum_penalty_reward_names", [])
    original_weights = getattr(env, "_curriculum_penalty_original_weights", {})
    if not cfg or not penalty_names or not hasattr(env, "reward_manager"):
        return

    # Clamp current_scale
    current_scale = cfg.get("current_scale", 1.0)
    current_scale = float(np.clip(current_scale, cfg["min_scale"], cfg["max_scale"]))
    cfg["current_scale"] = current_scale

    # Re-apply clamped scale to weights
    for name in penalty_names:
        if name not in original_weights or name not in env.reward_manager.active_terms:
            continue
        term_cfg = env.reward_manager.get_term_cfg(name)
        scaled_cfg = replace(term_cfg, weight=original_weights[name] * current_scale)
        env.reward_manager.set_term_cfg(name, scaled_cfg)
