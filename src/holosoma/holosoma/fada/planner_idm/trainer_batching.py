from __future__ import annotations

import copy
import random
from typing import Any

import numpy as np

from holosoma.fada.common.compact_obs import (
    COMPACT_TERM_ORDER,
    canonicalize_compact_term_noise,
    canonicalize_compact_term_scale,
)
from holosoma.fada.common.dataset import ReplayBuffer
from holosoma.fada.planner_idm.loss_utils import weighted_horizon_mse
from holosoma.utils.safe_torch_import import F, optim, torch


class _BatchingMixin:
    @staticmethod
    def _empty_source_counts() -> dict[str, int]:
        return {
            "optimal": 0,
            "suboptimal": 0,
            "suboptimal_expert": 0,
            "online": 0,
            "online_trajectory": 0,
        }

    @classmethod
    def _merge_source_counts(cls, *counts: dict[str, int]) -> dict[str, int]:
        merged = cls._empty_source_counts()
        for count in counts:
            for key in merged:
                merged[key] += int(count.get(key, 0))
        return merged

    @staticmethod
    def _prefix_source_counts(prefix: str, counts: dict[str, int]) -> dict[str, int]:
        return {
            f"{prefix}_optimal_batch_size": int(counts.get("optimal", 0)),
            f"{prefix}_suboptimal_batch_size": int(counts.get("suboptimal", 0)),
            f"{prefix}_suboptimal_expert_batch_size": int(counts.get("suboptimal_expert", 0)),
            f"{prefix}_online_batch_size": int(counts.get("online", 0)),
            f"{prefix}_online_trajectory_batch_size": int(counts.get("online_trajectory", 0)),
        }

    def _obs_term_slices(self) -> list[tuple[str, int, int]]:
        """Return (term_name, start_idx, end_idx) for each compact obs term."""
        ndof = (self.obs_dim - 6) // 2  # base_ang_vel=3, projected_gravity=3
        slices = []
        idx = 0
        for term_name in COMPACT_TERM_ORDER:
            if term_name == "base_ang_vel":
                dim = 3
            elif term_name == "dof_pos":
                dim = ndof
            elif term_name == "dof_vel":
                dim = ndof
            elif term_name == "projected_gravity":
                dim = 3
            else:
                raise ValueError(f"Unknown compact obs term: {term_name}")
            slices.append((term_name, idx, idx + dim))
            idx += dim
        return slices

    def _build_augment_noise_scales(self) -> torch.Tensor | None:
        """Build per-dim noise scale vector in scaled compact obs space."""
        if not bool(self.cfg.augment_obs_noise):
            return None
        term_noise = canonicalize_compact_term_noise(self.cfg.compact_obs_term_noise)
        # Apply per-term noise overrides (kept separate from compact_obs_term_noise
        # to avoid breaking offline cache metadata matching).
        if self.cfg.augment_obs_noise_overrides:
            for term, val in self.cfg.augment_obs_noise_overrides.items():
                term_noise[term] = float(val)
        term_scale = canonicalize_compact_term_scale(self.cfg.compact_obs_term_scale)
        noise_vec = torch.zeros(self.obs_dim, dtype=torch.float32, device=self.device)
        for term_name, start, end in self._obs_term_slices():
            noise_vec[start:end] = float(term_noise[term_name]) * float(term_scale[term_name])
        if not hasattr(self, "_noise_scales_logged"):
            effective = {t: f"{term_noise[t]}*{term_scale[t]}={term_noise[t]*term_scale[t]:.6f}" for t in term_noise}
            print(f"[Noise] Effective augment noise (noise*scale): {effective}")
            self._noise_scales_logged = True
        return noise_vec

    def _apply_obs_noise(self, obs: torch.Tensor) -> torch.Tensor:
        """Add per-term uniform noise to obs tensor (B, T, D) or (B, D)."""
        noise_scales = self._build_augment_noise_scales()
        if noise_scales is None:
            return obs
        shape = obs.shape
        if obs.ndim == 3:
            noise_scales = noise_scales.view(1, 1, -1)
        elif obs.ndim == 2:
            noise_scales = noise_scales.view(1, -1)
        noise = (torch.rand(shape, device=obs.device, dtype=obs.dtype) * 2.0 - 1.0) * noise_scales
        return obs + noise

    def _apply_history_mask(
        self,
        history_obs: torch.Tensor,
        history_act: torch.Tensor,
        history_valid_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Randomly zero out history time steps and update valid mask."""
        ratio = float(self.cfg.augment_history_mask_ratio)
        if ratio <= 0.0:
            return history_obs, history_act, history_valid_mask
        B, T, _ = history_obs.shape
        # Never mask the last time step (current obs)
        mask_candidates = T - 1
        if mask_candidates <= 0:
            return history_obs, history_act, history_valid_mask
        # Generate random mask for first T-1 steps
        drop_mask = torch.rand((B, mask_candidates), device=history_obs.device) < ratio
        # Pad with False for the last time step (never drop)
        full_drop = torch.cat([drop_mask, torch.zeros((B, 1), device=history_obs.device, dtype=torch.bool)], dim=1)
        history_obs = history_obs.clone()
        history_act = history_act.clone()
        history_obs[full_drop] = 0.0
        history_act[full_drop] = 0.0
        if history_valid_mask is not None:
            history_valid_mask = history_valid_mask.clone()
            history_valid_mask[full_drop] = False
        return history_obs, history_act, history_valid_mask

    def _apply_future_mask(self, future: torch.Tensor) -> torch.Tensor:
        """Randomly zero out future time steps (works for both obs and actions)."""
        ratio = float(self.cfg.augment_future_mask_ratio)
        if ratio <= 0.0:
            return future
        B, T, _ = future.shape
        drop_mask = torch.rand((B, T), device=future.device) < ratio
        future = future.clone()
        future[drop_mask] = 0.0
        return future

    def _apply_action_noise(self, actions: torch.Tensor) -> torch.Tensor:
        """Add uniform noise to action tensor (B, T, D) or (B, D)."""
        level = float(self.cfg.augment_action_noise_level)
        if level <= 0.0:
            return actions
        noise = (torch.rand_like(actions) * 2.0 - 1.0) * level
        return actions + noise

    def _to_delta_obs(self, target_obs: torch.Tensor, ref_obs: torch.Tensor) -> torch.Tensor:
        """Convert absolute future obs target to delta relative to ref_obs.

        target_obs: (B, K, D) — absolute future observations
        ref_obs: (B, D) — reference (current) observation
        Returns: (B, K, D) — delta predictions
        """
        return target_obs - ref_obs.unsqueeze(1)

    def _from_delta_obs(self, delta_obs: torch.Tensor, ref_obs: torch.Tensor) -> torch.Tensor:
        """Reconstruct absolute future obs from delta prediction.

        delta_obs: (B, K, D) — delta predictions
        ref_obs: (B, D) — reference (current) observation
        Returns: (B, K, D) — absolute future observations
        """
        return delta_obs + ref_obs.unsqueeze(1)

    def _compute_per_term_mse(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, float]:
        """Compute MSE decomposed by compact obs term."""
        results = {}
        for term_name, start, end in self._obs_term_slices():
            term_loss = F.mse_loss(pred[..., start:end], target[..., start:end])
            results[term_name] = float(term_loss.detach().cpu().item())
        return results

    def _build_sample_stats(
        self,
        *,
        planner_counts: dict[str, int],
        idm_counts: dict[str, int],
    ) -> dict[str, int]:
        total_counts = self._merge_source_counts(planner_counts, idm_counts)
        stats = {
            "optimal_batch_size": int(total_counts["optimal"]),
            "suboptimal_batch_size": int(total_counts["suboptimal"]),
            "suboptimal_expert_batch_size": int(total_counts["suboptimal_expert"]),
            "online_batch_size": int(total_counts["online"]),
            "online_trajectory_batch_size": int(total_counts["online_trajectory"]),
        }
        stats.update(self._prefix_source_counts("planner", planner_counts))
        stats.update(self._prefix_source_counts("idm", idm_counts))
        return stats

    def _sample_chunk_from_buffer(self, buffer: ReplayBuffer, batch_size: int) -> dict[str, np.ndarray]:
        return buffer.sample_chunk(batch_size)

    def _resolve_buffers(self, *, use_val: bool = False) -> tuple[ReplayBuffer, ReplayBuffer, ReplayBuffer]:
        """Return (optimal, suboptimal, online) buffer triple for train or val."""
        if use_val:
            opt = self.offline_val_buffer if self.offline_val_buffer is not None else self.offline_buffer
            sub = self.suboptimal_offline_val_buffer if self.suboptimal_offline_val_buffer is not None else self.suboptimal_offline_buffer
            onl = self.online_val_buffer if self.online_val_buffer is not None else self.online_buffer
            return opt, sub, onl
        return self.offline_buffer, self.suboptimal_offline_buffer, self.online_buffer

    def _sample_from_sources(
        self,
        *,
        optimal_bs: int,
        suboptimal_bs: int,
        online_bs: int,
        use_val: bool = False,
    ) -> tuple[dict[str, np.ndarray], dict[str, int]]:
        opt_buf, sub_buf, onl_buf = self._resolve_buffers(use_val=use_val)
        counts = self._empty_source_counts()
        batches: list[dict[str, np.ndarray]] = []
        for source_name, batch_size, buffer in (
            ("optimal", int(optimal_bs), opt_buf),
            ("suboptimal", int(suboptimal_bs), sub_buf),
            ("online", int(online_bs), onl_buf),
        ):
            if batch_size <= 0:
                continue
            if buffer.num_valid_chunks() <= 0:
                raise RuntimeError(
                    f"{source_name} buffer has no valid chunks for sampling "
                    f"(requested_batch_size={batch_size})"
                )
            batches.append(self._sample_chunk_from_buffer(buffer, batch_size))
            counts[source_name] = batch_size
        if not batches:
            raise RuntimeError("No replay sources were selected for sampling")
        batch = batches[0]
        for extra_batch in batches[1:]:
            batch = self._combine_batches(batch, extra_batch)
        return batch, counts

    def _planner_batch_counts(self, *, mode: str) -> dict[str, int]:
        # Planner always uses oracle-relabeled targets for all sources, so no
        # online-trajectory / suboptimal-expert splits are applied; those keys
        # are kept at 0 for schema consistency with IDM counts.
        #
        # Warmup (mode="offline") uses a separate ratio so warmup can target a
        # balanced optimal:suboptimal split (e.g., 50/50) without affecting the
        # DAgger-phase offline:online 50/50 balance.
        if mode == "offline":
            base_suboptimal_ratio = float(self.cfg.warmup_planner_suboptimal_batch_ratio)
        else:
            base_suboptimal_ratio = float(self.cfg.planner_suboptimal_batch_ratio)
        effective_suboptimal_ratio = base_suboptimal_ratio if self.planner_suboptimal_enabled else 0.0
        suboptimal_bs = int(round(int(self.cfg.batch_size) * effective_suboptimal_ratio))
        suboptimal_bs = max(0, min(suboptimal_bs, int(self.cfg.batch_size)))
        remaining = int(self.cfg.batch_size) - suboptimal_bs
        if mode == "offline":
            return {
                "optimal": int(remaining),
                "suboptimal": int(suboptimal_bs),
                "suboptimal_expert": 0,
                "online": 0,
                "online_trajectory": 0,
            }
        if mode != "mixed":
            raise ValueError(f"Unknown planner train mode: {mode}")
        if remaining <= 0:
            return {
                "optimal": 0,
                "suboptimal": int(suboptimal_bs),
                "suboptimal_expert": 0,
                "online": 0,
                "online_trajectory": 0,
            }
        optimal_bs = int(round(remaining * float(self.cfg.mixed_batch_offline_ratio)))
        optimal_bs = max(0, min(optimal_bs, remaining))
        online_bs = remaining - optimal_bs
        return {
            "optimal": int(optimal_bs),
            "suboptimal": int(suboptimal_bs),
            "suboptimal_expert": 0,
            "online": int(online_bs),
            "online_trajectory": 0,
        }

    def _idm_batch_counts(self, *, mode: str) -> dict[str, int]:
        # Warmup (mode="offline") uses a separate ratio so warmup can target a
        # balanced optimal:(suboptimal_real + suboptimal_expert) split (e.g.,
        # 50/50) without affecting the DAgger-phase offline:online 50/50 balance.
        if mode == "offline":
            base_suboptimal_ratio = float(self.cfg.warmup_idm_suboptimal_batch_ratio)
        else:
            base_suboptimal_ratio = float(self.cfg.idm_suboptimal_batch_ratio)
        effective_suboptimal_ratio = base_suboptimal_ratio if self.suboptimal_enabled else 0.0
        total_suboptimal_bs = int(round(int(self.cfg.batch_size) * effective_suboptimal_ratio))
        total_suboptimal_bs = max(0, min(total_suboptimal_bs, int(self.cfg.batch_size)))
        # Split total suboptimal budget between real-trajectory and expert-shadow targets.
        expert_ratio = float(self.cfg.suboptimal_expert_batch_ratio)
        expert_ratio = max(0.0, min(1.0, expert_ratio))
        suboptimal_expert_bs = int(round(total_suboptimal_bs * expert_ratio))
        suboptimal_expert_bs = max(0, min(suboptimal_expert_bs, total_suboptimal_bs))
        suboptimal_bs = total_suboptimal_bs - suboptimal_expert_bs
        remaining = int(self.cfg.batch_size) - total_suboptimal_bs
        if mode == "offline":
            # No online data in offline/warmup mode.
            return {
                "optimal": int(remaining),
                "suboptimal": int(suboptimal_bs),
                "suboptimal_expert": int(suboptimal_expert_bs),
                "online": 0,
                "online_trajectory": 0,
            }
        if mode != "mixed":
            raise ValueError(f"Unknown IDM train mode: {mode}")
        if remaining <= 0:
            return {
                "optimal": 0,
                "suboptimal": int(suboptimal_bs),
                "suboptimal_expert": int(suboptimal_expert_bs),
                "online": 0,
                "online_trajectory": 0,
            }
        optimal_bs = int(round(remaining * float(self.cfg.mixed_batch_offline_ratio)))
        optimal_bs = max(0, min(optimal_bs, remaining))
        total_online_bs = remaining - optimal_bs
        # Split total online budget between oracle-shadow and real-trajectory targets.
        # When use_generalized_idm=False, skip oracle-shadow online data so IDM only
        # sees real dynamics (executed actions, not oracle-relabeled actions).
        if not bool(self.cfg.use_generalized_idm):
            online_trajectory_bs = total_online_bs
            online_bs = 0
        else:
            trajectory_ratio = float(self.cfg.online_trajectory_batch_ratio)
            trajectory_ratio = max(0.0, min(1.0, trajectory_ratio))
            online_trajectory_bs = int(round(total_online_bs * trajectory_ratio))
            online_trajectory_bs = max(0, min(online_trajectory_bs, total_online_bs))
            online_bs = total_online_bs - online_trajectory_bs
        return {
            "optimal": int(optimal_bs),
            "suboptimal": int(suboptimal_bs),
            "suboptimal_expert": int(suboptimal_expert_bs),
            "online": int(online_bs),
            "online_trajectory": int(online_trajectory_bs),
        }

    def _sample_planner_batch(self, *, mode: str, use_val: bool = False) -> tuple[dict[str, np.ndarray], dict[str, int]]:
        counts = self._planner_batch_counts(mode=mode)
        # When planner_suboptimal is enabled, suboptimal_offline_buffer contains
        # oracle-relabeled data — planner reads target_next_observations (oracle
        # future obs) while IDM reads trajectory_target_* (real dynamics).
        batch, _ = self._sample_from_sources(
            optimal_bs=counts["optimal"],
            suboptimal_bs=counts["suboptimal"],
            online_bs=counts["online"],
            use_val=use_val,
        )
        return batch, counts

    def _sample_idm_batch(self, *, mode: str, use_val: bool = False) -> tuple[dict[str, np.ndarray], dict[str, int]]:
        counts = self._idm_batch_counts(mode=mode)
        batch, _ = self._sample_from_sources(
            optimal_bs=counts["optimal"],
            suboptimal_bs=counts["suboptimal"],
            online_bs=counts["online"],
            use_val=use_val,
        )
        return batch, counts

    def _sample_idm_batches_by_source(self, *, mode: str, use_val: bool = False) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, int]]:
        opt_buf, sub_buf, onl_buf = self._resolve_buffers(use_val=use_val)
        counts = self._idm_batch_counts(mode=mode)
        batches: dict[str, dict[str, np.ndarray]] = {}
        # online_trajectory / suboptimal_expert draw from the same buffers as
        # online / suboptimal respectively, but consumers route them to the
        # alternate target fields (trajectory_target_* for *_trajectory sources,
        # expert_* oracle-shadow for *_expert sources).
        for source_name, batch_size, buffer in (
            ("optimal", int(counts["optimal"]), opt_buf),
            ("suboptimal", int(counts["suboptimal"]), sub_buf),
            ("suboptimal_expert", int(counts["suboptimal_expert"]), sub_buf),
            ("online", int(counts["online"]), onl_buf),
            ("online_trajectory", int(counts["online_trajectory"]), onl_buf),
        ):
            if batch_size <= 0:
                continue
            if buffer.num_valid_chunks() <= 0:
                raise RuntimeError(
                    f"{source_name} buffer has no valid chunks for sampling "
                    f"(requested_batch_size={batch_size})"
                )
            batches[source_name] = self._sample_chunk_from_buffer(buffer, batch_size)
        if not batches:
            raise RuntimeError("No replay sources were selected for IDM sampling")
        return batches, counts

    def _normalize_batch_inputs(
        self,
        *,
        history_obs: torch.Tensor,
        history_act: torch.Tensor,
        current_command: torch.Tensor,
        target_actions: torch.Tensor,
        target_next_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        model_history_obs = history_obs
        model_history_act = history_act
        model_current_command = current_command
        target_actions_loss_input = target_actions
        target_obs_loss_input = target_next_obs
        if self._use_io_normalization():
            obs_mean, obs_std = self._ensure_obs_norm_stats_for_training()
            action_mean, action_std = self._ensure_action_norm_stats_for_training()
            command_mean, command_std = self._ensure_command_norm_stats_for_training()
            obs_mean_seq = obs_mean.view(1, 1, -1)
            obs_std_seq = obs_std.view(1, 1, -1)
            action_mean_seq = action_mean.view(1, 1, -1)
            action_std_seq = action_std.view(1, 1, -1)
            command_mean_seq = command_mean.view(1, -1)
            command_std_seq = command_std.view(1, -1)
            model_history_obs = (history_obs - obs_mean_seq) / obs_std_seq
            model_history_act = (history_act - action_mean_seq) / action_std_seq
            model_current_command = (current_command - command_mean_seq) / command_std_seq
            target_actions_loss_input = (target_actions - action_mean_seq) / action_std_seq
            target_obs_loss_input = (target_next_obs - obs_mean_seq) / obs_std_seq
        return (
            model_history_obs,
            model_history_act,
            model_current_command,
            target_actions_loss_input,
            target_obs_loss_input,
        )

    def _compute_planner_batch_loss_from_tensors(
        self,
        *,
        history_obs: torch.Tensor,
        history_act: torch.Tensor,
        current_command: torch.Tensor,
        target_next_obs: torch.Tensor,
        history_valid_mask: torch.Tensor | None,
        augment: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float] | None]:
        # Save clean reference obs for delta conversion (before noise)
        ref_obs = history_obs[:, -1, :]  # (B, D)

        # Apply augmentation (noise + masking) to inputs only
        if augment:
            history_obs = self._apply_obs_noise(history_obs)
            history_act = self._apply_action_noise(history_act)
            history_obs, history_act, history_valid_mask = self._apply_history_mask(
                history_obs, history_act, history_valid_mask,
            )

        (
            model_history_obs,
            model_history_act,
            model_current_command,
            _target_actions_loss_input,
            target_obs_loss_input,
        ) = self._normalize_batch_inputs(
            history_obs=history_obs,
            history_act=history_act,
            current_command=current_command,
            target_actions=torch.zeros(
                (history_obs.shape[0], self.cfg.pred_horizon, self.act_dim),
                device=history_obs.device,
                dtype=history_obs.dtype,
            ),
            target_next_obs=target_next_obs,
        )

        # Delta target: predict change relative to current obs
        if bool(self.cfg.planner_predict_delta):
            # Normalize ref_obs the same way as target
            if self._use_io_normalization():
                obs_mean, obs_std = self._ensure_obs_norm_stats_for_training()
                ref_obs_norm = (ref_obs - obs_mean) / obs_std
            else:
                ref_obs_norm = ref_obs
            target_obs_loss_input = self._to_delta_obs(target_obs_loss_input, ref_obs_norm)

        pred_future_obs = self.student_policy.planner(
            model_history_obs,
            model_current_command,
            history_action=(model_history_act if self.student_policy.planner_use_action_history else None),
            history_valid_mask=history_valid_mask,
        )
        planner_loss = F.mse_loss(pred_future_obs, target_obs_loss_input)
        per_term = self._compute_per_term_mse(pred_future_obs, target_obs_loss_input)
        weighted_total = float(self.cfg.planner_obs_loss_coef) * planner_loss
        return weighted_total, planner_loss, per_term

    def _resolve_idm_future_obs(
        self,
        *,
        planner_future_obs: torch.Tensor,
        teacher_future_obs: torch.Tensor,
        ref_obs_norm: torch.Tensor | None = None,
        use_teacher_forcing: bool | None = None,
        teacher_forcing_ratio: float | None = None,
        detach_override: bool | None = None,
    ) -> torch.Tensor:
        use_tf = bool(self.cfg.idm_use_teacher_forcing) if use_teacher_forcing is None else bool(use_teacher_forcing)
        teacher_ratio = float(self.cfg.idm_teacher_forcing_ratio) if teacher_forcing_ratio is None else float(
            teacher_forcing_ratio
        )
        if not use_tf:
            teacher_ratio = 0.0
        # Reconstruct absolute from delta if planner uses delta prediction
        should_detach = bool(self.cfg.idm_detach_planner_future_obs) if detach_override is None else bool(detach_override)
        if should_detach:
            planner_source = planner_future_obs.detach()
        else:
            planner_source = planner_future_obs
        if bool(self.cfg.planner_predict_delta) and ref_obs_norm is not None:
            planner_source = self._from_delta_obs(planner_source, ref_obs_norm)
        if teacher_ratio <= 0.0:
            return planner_source
        if teacher_ratio >= 1.0:
            return teacher_future_obs
        batch_size = int(planner_future_obs.shape[0])
        mix_mask = (
            torch.rand((batch_size, 1, 1), device=planner_future_obs.device, dtype=planner_future_obs.dtype)
            < teacher_ratio
        ).to(dtype=planner_future_obs.dtype)
        return mix_mask * teacher_future_obs + (1.0 - mix_mask) * planner_source

    def _compute_idm_batch_loss_from_tensors(
        self,
        *,
        history_obs: torch.Tensor,
        history_act: torch.Tensor,
        current_command: torch.Tensor,
        target_actions: torch.Tensor,
        target_next_obs: torch.Tensor,
        history_valid_mask: torch.Tensor | None,
        use_teacher_forcing: bool | None = None,
        teacher_forcing_ratio: float | None = None,
        detach_override: bool | None = None,
        augment: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Save clean reference obs for delta reconstruction (before noise)
        ref_obs = history_obs[:, -1, :]  # (B, D)

        # Apply augmentation to inputs
        if augment:
            history_obs = self._apply_obs_noise(history_obs)
            history_act = self._apply_action_noise(history_act)
            history_obs, history_act, history_valid_mask = self._apply_history_mask(
                history_obs, history_act, history_valid_mask,
            )

        (
            model_history_obs,
            model_history_act,
            model_current_command,
            target_actions_loss_input,
            target_obs_loss_input,
        ) = self._normalize_batch_inputs(
            history_obs=history_obs,
            history_act=history_act,
            current_command=current_command,
            target_actions=target_actions,
            target_next_obs=target_next_obs,
        )

        # Resolve effective TF ratio early to skip planner forward on pure-TF paths.
        _use_tf = bool(self.cfg.idm_use_teacher_forcing) if use_teacher_forcing is None else bool(use_teacher_forcing)
        _teacher_ratio = (
            float(self.cfg.idm_teacher_forcing_ratio) if teacher_forcing_ratio is None else float(teacher_forcing_ratio)
        )
        if not _use_tf:
            _teacher_ratio = 0.0

        if _teacher_ratio >= 1.0:
            # Pure TF: teacher obs is used directly — skip planner forward entirely.
            idm_future_obs = target_obs_loss_input
        else:
            # Compute normalized ref_obs for delta reconstruction
            ref_obs_norm: torch.Tensor | None = None
            if bool(self.cfg.planner_predict_delta):
                if self._use_io_normalization():
                    obs_mean, obs_std = self._ensure_obs_norm_stats_for_training()
                    ref_obs_norm = (ref_obs - obs_mean) / obs_std
                else:
                    ref_obs_norm = ref_obs

            planner_future_obs = self.student_policy.planner(
                model_history_obs,
                model_current_command,
                history_action=(model_history_act if self.student_policy.planner_use_action_history else None),
                history_valid_mask=history_valid_mask,
            )
            idm_future_obs = self._resolve_idm_future_obs(
                planner_future_obs=planner_future_obs,
                teacher_future_obs=target_obs_loss_input,
                ref_obs_norm=ref_obs_norm,
                use_teacher_forcing=use_teacher_forcing,
                teacher_forcing_ratio=teacher_forcing_ratio,
                detach_override=detach_override,
            )
        # Apply future obs noise + masking to IDM input (after teacher forcing resolution)
        if augment:
            idm_future_obs = self._apply_obs_noise(idm_future_obs)
            idm_future_obs = self._apply_future_mask(idm_future_obs)
        idm_current_command = self.student_policy._resolve_idm_current_command(
            model_current_command,
            idm_use_current_command_for_history=self.student_policy.idm_use_current_command_for_history,
        )
        pred_actions = self.student_policy.predict_idm_actions(
            model_history_obs,
            model_history_act,
            idm_current_command,
            idm_future_obs,
            history_valid_mask=history_valid_mask,
        )
        # Hybrid IDM outputs (B, 1, A); slice target to first step. Gamma is enforced
        # to 0.0 at config-validation time, so this is consistent and not a silent fallback.
        if pred_actions.shape[1] != target_actions_loss_input.shape[1]:
            if pred_actions.shape[1] != 1:
                raise RuntimeError(
                    f"IDM pred_actions shape {tuple(pred_actions.shape)} incompatible with "
                    f"target shape {tuple(target_actions_loss_input.shape)}"
                )
            target_actions_loss_input = target_actions_loss_input[:, :1, :]
        idm_loss_unweighted = F.mse_loss(pred_actions, target_actions_loss_input)
        gamma = float(self.cfg.idm_action_loss_horizon_gamma)
        idm_loss = weighted_horizon_mse(pred_actions, target_actions_loss_input, gamma)
        weighted_total = float(self.cfg.idm_action_loss_coef) * idm_loss
        return weighted_total, idm_loss, idm_loss_unweighted

    def _compute_planner_batch_loss_from_batch(
        self,
        batch: dict[str, np.ndarray],
        *,
        augment: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float] | None]:
        history_obs, history_act, current_command, _target_actions, target_next_obs, history_valid_mask = (
            self._batch_to_tensors(batch)
        )
        if target_next_obs is None:
            raise RuntimeError("planner batch requires target_next_observations")
        return self._compute_planner_batch_loss_from_tensors(
            history_obs=history_obs,
            history_act=history_act,
            current_command=current_command,
            target_next_obs=target_next_obs,
            history_valid_mask=history_valid_mask,
            augment=augment,
        )

    def _compute_idm_batch_loss_from_batch(
        self,
        batch: dict[str, np.ndarray],
        *,
        use_teacher_forcing: bool | None = None,
        teacher_forcing_ratio: float | None = None,
        use_trajectory_targets: bool = False,
        detach_override: bool | None = None,
        augment: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        history_obs, history_act, current_command, target_actions, target_next_obs, history_valid_mask = (
            self._batch_to_tensors(batch)
        )
        # For suboptimal data, use trajectory targets (real causal dynamics from
        # executed_act / obs) instead of chunk targets (oracle-relabeled).
        # For optimal/online data, keep expert chunk targets (expert causality).
        if use_trajectory_targets:
            if "trajectory_target_actions" in batch:
                target_actions = torch.from_numpy(batch["trajectory_target_actions"]).to(
                    device=self.device, dtype=torch.float32,
                )
            if "trajectory_target_next_observations" in batch:
                target_next_obs = torch.from_numpy(batch["trajectory_target_next_observations"]).to(
                    device=self.device, dtype=torch.float32,
                )
            # Drop boundary samples whose trajectory targets are zero-filled
            # (anchor too close to episode end for K consecutive steps).
            if "trajectory_target_valid" in batch:
                valid_mask = torch.from_numpy(batch["trajectory_target_valid"]).to(device=self.device)
                if not valid_mask.all():
                    valid_idx = valid_mask.nonzero(as_tuple=True)[0]
                    if valid_idx.numel() == 0:
                        zero = torch.zeros((), device=self.device, dtype=torch.float32)
                        return zero, zero, zero
                    history_obs = history_obs[valid_idx]
                    history_act = history_act[valid_idx]
                    current_command = current_command[valid_idx]
                    target_actions = target_actions[valid_idx]
                    if target_next_obs is not None:
                        target_next_obs = target_next_obs[valid_idx]
                    history_valid_mask = history_valid_mask[valid_idx]
        if target_next_obs is None:
            raise RuntimeError("IDM batch requires target_next_observations")
        return self._compute_idm_batch_loss_from_tensors(
            history_obs=history_obs,
            history_act=history_act,
            current_command=current_command,
            target_actions=target_actions,
            target_next_obs=target_next_obs,
            history_valid_mask=history_valid_mask,
            use_teacher_forcing=use_teacher_forcing,
            teacher_forcing_ratio=teacher_forcing_ratio,
            detach_override=detach_override,
            augment=augment,
        )

    def _idm_source_uses_teacher_forcing(self, source_name: str) -> bool:
        # Trajectory sources (suboptimal, online_trajectory) always use TF=1.0:
        # their labels are real executed actions that must be paired with real obs.
        # Oracle-shadow sources (optimal, suboptimal_expert, online) follow the
        # global cfg flag and idm_teacher_forcing_ratio uniformly.
        if source_name in {"suboptimal", "online_trajectory"}:
            return True
        return bool(self.cfg.idm_use_teacher_forcing)

    def _compute_idm_batch_loss_from_source_batches(
        self,
        batches_by_source: dict[str, dict[str, np.ndarray]],
        *,
        force_teacher_forcing_ratio: float | None = None,
        detach_override: bool | None = None,
        augment: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weighted_losses: list[torch.Tensor] = []
        idm_losses: list[torch.Tensor] = []
        idm_losses_unweighted: list[torch.Tensor] = []
        batch_sizes: list[int] = []
        # Real-causal sources (suboptimal, online_trajectory) route targets to the
        # executed-action / real-next-obs fields; oracle-causal sources (optimal,
        # online, suboptimal_expert) keep oracle shadow chunks.
        for source_name in ("optimal", "suboptimal", "suboptimal_expert", "online", "online_trajectory"):
            batch = batches_by_source.get(source_name)
            if batch is None:
                continue
            # Trajectory sources are always fully TF (ratio=1.0); oracle-shadow sources
            # (optimal, suboptimal_expert, online) share the same idm_teacher_forcing_ratio.
            # force_teacher_forcing_ratio overrides all per-source ratios when provided.
            if force_teacher_forcing_ratio is not None:
                tf_ratio = float(force_teacher_forcing_ratio)
                use_tf = tf_ratio > 0.0
            else:
                tf_ratio = 1.0 if source_name in {"suboptimal", "online_trajectory"} else None
                use_tf = self._idm_source_uses_teacher_forcing(source_name)
            loss_total, idm_loss, idm_loss_uw = self._compute_idm_batch_loss_from_batch(
                batch,
                use_teacher_forcing=use_tf,
                use_trajectory_targets=(source_name in {"suboptimal", "online_trajectory"}),
                teacher_forcing_ratio=tf_ratio,
                detach_override=detach_override,
                augment=augment,
            )
            weighted_losses.append(loss_total)
            idm_losses.append(idm_loss)
            idm_losses_unweighted.append(idm_loss_uw)
            batch_sizes.append(int(batch["history_obs"].shape[0]))
        if not weighted_losses or not batch_sizes:
            raise RuntimeError("No IDM source batches were provided for loss computation")
        total_batch = int(sum(batch_sizes))
        device = weighted_losses[0].device
        dtype = weighted_losses[0].dtype
        weights = [torch.as_tensor(bs / total_batch, device=device, dtype=dtype) for bs in batch_sizes]
        loss_total = sum(weight * loss for weight, loss in zip(weights, weighted_losses, strict=False))
        idm_loss = sum(weight * loss for weight, loss in zip(weights, idm_losses, strict=False))
        idm_loss_unweighted = sum(weight * loss for weight, loss in zip(weights, idm_losses_unweighted, strict=False))
        return loss_total, idm_loss, idm_loss_unweighted

    def _compute_fdm_batch_loss_from_source_batches(
        self,
        batches_by_source: dict[str, dict[str, np.ndarray]],
        *,
        augment: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # FDM is permanently disabled (see model.py / config.py); always zero.
        zero = torch.zeros((), device=self.device, dtype=torch.float32)
        return zero, zero

    def _combine_batches(
        self,
        batch_a: dict[str, np.ndarray],
        batch_b: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        merged: dict[str, np.ndarray] = {}
        for key in batch_a.keys():
            merged[key] = np.concatenate([batch_a[key], batch_b[key]], axis=0)
        return merged

    def _sample_batch(self, *, mode: str) -> tuple[dict[str, np.ndarray], dict[str, int]]:
        if mode == "offline":
            if self.offline_buffer.num_valid_chunks() <= 0:
                raise RuntimeError("Offline buffer has no valid chunks for training")
            return self._sample_chunk_from_buffer(self.offline_buffer, self.cfg.batch_size), {
                "offline_batch_size": int(self.cfg.batch_size),
                "online_batch_size": 0,
            }

        if mode == "online":
            if self.online_buffer.num_valid_chunks() <= 0:
                raise RuntimeError("Online buffer has no valid chunks for training")
            return self._sample_chunk_from_buffer(self.online_buffer, self.cfg.batch_size), {
                "offline_batch_size": 0,
                "online_batch_size": int(self.cfg.batch_size),
            }

        if mode != "mixed":
            raise ValueError(f"Unknown train mode: {mode}")

        ratio = float(self.cfg.mixed_batch_offline_ratio)
        offline_bs = int(round(self.cfg.batch_size * ratio))
        online_bs = int(self.cfg.batch_size - offline_bs)
        offline_bs = max(1, min(offline_bs, self.cfg.batch_size - 1))
        online_bs = self.cfg.batch_size - offline_bs

        offline_chunks = self.offline_buffer.num_valid_chunks()
        online_chunks = self.online_buffer.num_valid_chunks()
        if offline_chunks <= 0 or online_chunks <= 0:
            raise RuntimeError(
                "Mixed training requires both offline and online data. "
                f"offline_chunks={offline_chunks}, online_chunks={online_chunks}"
            )

        offline_batch = self._sample_chunk_from_buffer(self.offline_buffer, offline_bs)
        online_batch = self._sample_chunk_from_buffer(self.online_buffer, online_bs)
        return self._combine_batches(offline_batch, online_batch), {
            "offline_batch_size": int(offline_bs),
            "online_batch_size": int(online_bs),
        }

    def _batch_to_tensors(
        self,
        batch: dict[str, np.ndarray],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        history_obs = torch.from_numpy(batch["history_obs"]).to(device=self.device, dtype=torch.float32)
        history_act = torch.from_numpy(batch["history_act"]).to(device=self.device, dtype=torch.float32)
        current_command = torch.from_numpy(batch["current_command"]).to(device=self.device, dtype=torch.float32)
        history_valid_mask = (
            torch.from_numpy(batch["history_valid_mask"]).to(device=self.device, dtype=torch.bool)
            if "history_valid_mask" in batch
            else None
        )
        target_actions = torch.from_numpy(batch["target_actions"]).to(device=self.device, dtype=torch.float32)
        target_next_obs = (
            torch.from_numpy(batch["target_next_observations"]).to(device=self.device, dtype=torch.float32)
            if "target_next_observations" in batch
            else None
        )
        return history_obs, history_act, current_command, target_actions, target_next_obs, history_valid_mask

    def _compute_batch_losses(
        self,
        history_obs: torch.Tensor,
        history_act: torch.Tensor,
        current_command: torch.Tensor,
        target_actions: torch.Tensor,
        target_next_obs: torch.Tensor | None,
        history_valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.student_model.training:
            history_obs = self._apply_obs_noise(history_obs)
            history_obs, history_act, history_valid_mask = self._apply_history_mask(
                history_obs, history_act, history_valid_mask,
            )
            # Planner-IDM masks future observations when they are IDM inputs. Here
            # target_next_obs is the regression label, not an input, so
            # augment_future_mask_ratio has no effect.
        use_io_norm = self._use_io_normalization()
        model_history_obs = history_obs
        model_history_act = history_act
        model_current_command = current_command
        target_actions_loss_input = target_actions
        target_obs_loss_input = target_next_obs
        if use_io_norm:
            obs_mean, obs_std = self._ensure_obs_norm_stats_for_training()
            action_mean, action_std = self._ensure_action_norm_stats_for_training()
            command_mean, command_std = self._ensure_command_norm_stats_for_training()
            obs_mean_seq = obs_mean.view(1, 1, -1)
            obs_std_seq = obs_std.view(1, 1, -1)
            action_mean_seq = action_mean.view(1, 1, -1)
            action_std_seq = action_std.view(1, 1, -1)
            command_mean_seq = command_mean.view(1, -1)
            command_std_seq = command_std.view(1, -1)
            model_history_obs = (history_obs - obs_mean_seq) / obs_std_seq
            model_history_act = (history_act - action_mean_seq) / action_std_seq
            model_current_command = (current_command - command_mean_seq) / command_std_seq
            target_actions_loss_input = (target_actions - action_mean_seq) / action_std_seq
            if target_next_obs is not None:
                target_obs_loss_input = (target_next_obs - obs_mean_seq) / obs_std_seq
        if self.cfg.predict_future_obs:
            pred_actions, pred_next_obs = self.student_model(
                model_history_obs,
                model_history_act,
                model_current_command,
                history_valid_mask=history_valid_mask,
                return_obs=True,
            )
            if target_next_obs is None:
                raise RuntimeError(
                    "predict_future_obs=True but batch is missing target_next_observations. "
                    "Ensure ReplayBuffer is created with require_future_obs_targets=True."
                )
            if target_obs_loss_input is None:
                raise RuntimeError(
                    "predict_future_obs=True but normalized target_next_observations is missing."
                )
            loss_action = F.mse_loss(pred_actions, target_actions_loss_input)
            loss_obs = F.mse_loss(pred_next_obs, target_obs_loss_input)
            loss_total = loss_action + float(self.cfg.obs_loss_coef) * loss_obs
            return loss_total, loss_action, loss_obs

        pred_actions = self.student_model(
            model_history_obs,
            model_history_act,
            model_current_command,
            history_valid_mask=history_valid_mask,
        )
        loss_action = F.mse_loss(pred_actions, target_actions_loss_input)
        loss_obs = torch.zeros((), device=loss_action.device, dtype=loss_action.dtype)
        loss_total = loss_action
        return loss_total, loss_action, loss_obs

    def _val_buffers_ready(self, *, mode: str) -> bool:
        """Check whether trajectory-level val buffers have data for the given mode."""
        opt_buf, sub_buf, onl_buf = self._resolve_buffers(use_val=True)
        if mode == "offline":
            return opt_buf.num_valid_chunks() > 0
        if mode == "mixed":
            return opt_buf.num_valid_chunks() > 0 and onl_buf.num_valid_chunks() > 0
        return False

    @torch.no_grad()
    def _estimate_planner_validation_losses(
        self,
        *,
        mode: str,
        num_batches: int,
    ) -> dict[str, float] | None:
        if num_batches <= 0:
            return None
        opt_buf, _, onl_buf = self._resolve_buffers(use_val=True)
        if mode == "offline" and opt_buf.num_valid_chunks() <= 0:
            return None
        if mode == "mixed" and (opt_buf.num_valid_chunks() <= 0 or onl_buf.num_valid_chunks() <= 0):
            return None
        was_training = self.student_model.training
        self.student_model.eval()
        weighted_losses: list[float] = []
        planner_losses: list[float] = []
        per_term_accum: dict[str, list[float]] = {}
        naive_baseline_accum: list[float] = []
        for _ in range(num_batches):
            batch, _ = self._sample_planner_batch(mode=mode, use_val=True)
            planner_total, planner_loss, per_term = self._compute_planner_batch_loss_from_batch(batch, augment=False)
            weighted_losses.append(float(planner_total.detach().cpu().item()))
            planner_losses.append(float(planner_loss.detach().cpu().item()))
            if per_term is not None:
                for term_name, term_val in per_term.items():
                    per_term_accum.setdefault(term_name, []).append(float(term_val))
            # Naive baseline: predicting zero delta (or predicting current obs for absolute)
            history_obs, _, _, _, target_next_obs, _ = self._batch_to_tensors(batch)
            if target_next_obs is not None:
                ref_obs = history_obs[:, -1, :]
                if bool(self.cfg.planner_predict_delta):
                    # Delta mode: naive = predict zero (future = current)
                    if self._use_io_normalization():
                        obs_mean, obs_std = self._ensure_obs_norm_stats_for_training()
                        ref_obs_norm = (ref_obs - obs_mean) / obs_std
                        target_norm = (target_next_obs - obs_mean.view(1, 1, -1)) / obs_std.view(1, 1, -1)
                        delta_target = target_norm - ref_obs_norm.unsqueeze(1)
                    else:
                        delta_target = target_next_obs - ref_obs.unsqueeze(1)
                    naive_baseline_accum.append(float(F.mse_loss(
                        torch.zeros_like(delta_target), delta_target,
                    ).detach().cpu().item()))
                else:
                    # Absolute mode: naive = predict current obs for all future steps
                    naive_pred = ref_obs.unsqueeze(1).expand_as(target_next_obs)
                    target_for_naive = target_next_obs
                    if self._use_io_normalization():
                        obs_mean, obs_std = self._ensure_obs_norm_stats_for_training()
                        naive_pred = (naive_pred - obs_mean.view(1, 1, -1)) / obs_std.view(1, 1, -1)
                        target_for_naive = (target_next_obs - obs_mean.view(1, 1, -1)) / obs_std.view(1, 1, -1)
                    naive_baseline_accum.append(float(F.mse_loss(
                        naive_pred, target_for_naive,
                    ).detach().cpu().item()))
        if was_training:
            self.student_model.train()
        result: dict[str, float] = {
            "total": float(np.mean(weighted_losses)),
            "planner_future_obs_loss": float(np.mean(planner_losses)),
        }
        for term_name, vals in per_term_accum.items():
            result[f"term/{term_name}"] = float(np.mean(vals))
        if naive_baseline_accum:
            result["naive_baseline"] = float(np.mean(naive_baseline_accum))
        return result

    @torch.no_grad()
    def _estimate_idm_validation_losses(
        self,
        *,
        mode: str,
        num_batches: int,
    ) -> dict[str, float] | None:
        if num_batches <= 0:
            return None
        opt_buf, sub_buf, onl_buf = self._resolve_buffers(use_val=True)
        idm_counts_ready = self._idm_batch_counts(mode=mode)
        sub_need = int(idm_counts_ready["suboptimal"]) + int(idm_counts_ready["suboptimal_expert"])
        if sub_need > 0 and sub_buf.num_valid_chunks() <= 0:
            return None
        if mode == "offline" and opt_buf.num_valid_chunks() <= 0:
            return None
        if mode == "mixed":
            idm_counts = idm_counts_ready
            online_need = int(idm_counts["online"]) + int(idm_counts["online_trajectory"])
            if online_need > 0 and onl_buf.num_valid_chunks() <= 0:
                return None
            if idm_counts["optimal"] > 0 and opt_buf.num_valid_chunks() <= 0:
                return None
        was_training = self.student_model.training
        self.student_model.eval()
        weighted_losses: list[float] = []
        idm_losses: list[float] = []
        idm_losses_unweighted: list[float] = []
        for _ in range(num_batches):
            batches_by_source, _ = self._sample_idm_batches_by_source(mode=mode, use_val=True)
            idm_total, idm_loss, idm_loss_uw = self._compute_idm_batch_loss_from_source_batches(batches_by_source, augment=False)
            weighted_losses.append(float(idm_total.detach().cpu().item()))
            idm_losses.append(float(idm_loss.detach().cpu().item()))
            idm_losses_unweighted.append(float(idm_loss_uw.detach().cpu().item()))
        if was_training:
            self.student_model.train()
        return {
            "total": float(np.mean(weighted_losses)),
            "idm_action_loss": float(np.mean(idm_losses)),
            "idm_action_loss_unweighted": float(np.mean(idm_losses_unweighted)),
        }

    @torch.no_grad()
    def _estimate_fdm_validation_losses(
        self,
        *,
        mode: str,
        num_batches: int,
    ) -> dict[str, float] | None:
        # FDM is permanently disabled (see model.py / config.py); no validation losses.
        return None

    @torch.no_grad()
    def _estimate_validation_losses(self, buffer: ReplayBuffer, *, num_batches: int) -> dict[str, float] | None:
        if num_batches <= 0:
            return None
        if buffer.num_valid_chunks() <= 0:
            return None

        was_training = self.student_model.training
        self.student_model.eval()
        losses_total: list[float] = []
        losses_action: list[float] = []
        losses_obs: list[float] = []
        for _ in range(num_batches):
            batch = self._sample_chunk_from_buffer(buffer, self.cfg.batch_size)
            history_obs, history_act, current_command, target_actions, target_next_obs, history_valid_mask = (
                self._batch_to_tensors(batch)
            )
            loss_total, loss_action, loss_obs = self._compute_batch_losses(
                history_obs,
                history_act,
                current_command,
                target_actions,
                target_next_obs,
                history_valid_mask,
            )
            losses_total.append(float(loss_total.detach().cpu().item()))
            losses_action.append(float(loss_action.detach().cpu().item()))
            losses_obs.append(float(loss_obs.detach().cpu().item()))

        if was_training:
            self.student_model.train()
        if not losses_total:
            return None
        return {
            "total": float(np.mean(losses_total)),
            "action": float(np.mean(losses_action)),
            "obs_reconstruction": float(np.mean(losses_obs)),
        }

    def _compute_obs_norm_stats_from_offline(self) -> None:
        self._refresh_obs_norm_stats(include_online=len(self.online_buffer) > 0)

    def _compute_action_norm_stats_from_offline(self) -> None:
        self._refresh_action_norm_stats(include_online=len(self.online_buffer) > 0)

    def _compute_command_norm_stats_from_offline(self) -> None:
        self._refresh_command_norm_stats(include_online=len(self.online_buffer) > 0)

    @staticmethod
    def _counted_union_mean_std(
        stats: list[tuple[int, np.ndarray, np.ndarray]],
        *,
        eps: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not stats:
            raise RuntimeError("Cannot compute shared normalization statistics from empty buffers")
        total = int(sum(count for count, _, _ in stats))
        if total <= 0:
            raise RuntimeError("Cannot compute shared normalization statistics from empty buffers")
        mean_acc: np.ndarray | None = None
        second_moment_acc: np.ndarray | None = None
        for count, mean, std in stats:
            if count <= 0:
                continue
            count_f = float(count)
            mean64 = mean.astype(np.float64, copy=False)
            std64 = std.astype(np.float64, copy=False)
            second = np.square(std64) + np.square(mean64)
            if mean_acc is None:
                mean_acc = count_f * mean64
                second_moment_acc = count_f * second
            else:
                mean_acc += count_f * mean64
                second_moment_acc += count_f * second
        if mean_acc is None or second_moment_acc is None:
            raise RuntimeError("Failed to accumulate shared normalization statistics")
        mean_out = (mean_acc / float(total)).astype(np.float32, copy=False)
        var_out = np.maximum(second_moment_acc / float(total) - np.square(mean_out.astype(np.float64)), float(eps) ** 2)
        std_out = np.sqrt(var_out).astype(np.float32, copy=False)
        std_out = np.maximum(std_out, np.float32(eps))
        return mean_out, std_out

    def _buffers_for_shared_norm(self, *, include_online: bool) -> list[tuple[str, ReplayBuffer]]:
        buffers: list[tuple[str, ReplayBuffer]] = [("optimal_offline", self.offline_buffer)]
        if self.cfg.shared_norm_source == "optimal_offline":
            return buffers
        if self.cfg.shared_norm_source == "optimal_like":
            if include_online and len(self.online_buffer) > 0:
                buffers.append(("online_dagger", self.online_buffer))
            return buffers
        if self.cfg.shared_norm_source == "idm_visible_union":
            if len(self.suboptimal_offline_buffer) > 0:
                buffers.append(("suboptimal_offline", self.suboptimal_offline_buffer))
            if include_online and len(self.online_buffer) > 0:
                buffers.append(("online_dagger", self.online_buffer))
            return buffers
        raise RuntimeError(f"Unsupported shared_norm_source: {self.cfg.shared_norm_source}")

    def _refresh_obs_norm_stats(self, *, include_online: bool) -> None:
        buffers = self._buffers_for_shared_norm(include_online=include_online)
        stats = [
            (len(buffer), *buffer.get_obs_mean_std(eps=self.obs_norm_eps))
            for _, buffer in buffers
            if len(buffer) > 0
        ]
        mean_np, std_np = self._counted_union_mean_std(stats, eps=self.obs_norm_eps)
        self.obs_norm_mean = torch.from_numpy(mean_np).to(device=self.device, dtype=torch.float32)
        self.obs_norm_std = torch.from_numpy(std_np).to(device=self.device, dtype=torch.float32)
        self.obs_norm_source = "+".join(name for name, buffer in buffers if len(buffer) > 0)

    def _refresh_action_norm_stats(self, *, include_online: bool) -> None:
        buffers = self._buffers_for_shared_norm(include_online=include_online)
        stats = [
            (len(buffer), *buffer.get_expert_action_mean_std(eps=self.action_norm_eps))
            for _, buffer in buffers
            if len(buffer) > 0
        ]
        mean_np, std_np = self._counted_union_mean_std(stats, eps=self.action_norm_eps)
        self.action_norm_mean = torch.from_numpy(mean_np).to(device=self.device, dtype=torch.float32)
        self.action_norm_std = torch.from_numpy(std_np).to(device=self.device, dtype=torch.float32)
        self.action_norm_source = "+".join(name for name, buffer in buffers if len(buffer) > 0)

    def _refresh_command_norm_stats(self, *, include_online: bool) -> None:
        buffers = self._buffers_for_shared_norm(include_online=include_online)
        stats = [
            (len(buffer), *buffer.get_command_mean_std(eps=self.command_norm_eps))
            for _, buffer in buffers
            if len(buffer) > 0
        ]
        mean_np, std_np = self._counted_union_mean_std(stats, eps=self.command_norm_eps)
        self.command_norm_mean = torch.from_numpy(mean_np).to(device=self.device, dtype=torch.float32)
        self.command_norm_std = torch.from_numpy(std_np).to(device=self.device, dtype=torch.float32)
        self.command_norm_source = "+".join(name for name, buffer in buffers if len(buffer) > 0)

    def _refresh_shared_norm_stats(self, *, include_online: bool) -> None:
        self._refresh_obs_norm_stats(include_online=include_online)
        if self._use_io_normalization():
            self._refresh_action_norm_stats(include_online=include_online)
            self._refresh_command_norm_stats(include_online=include_online)
