from __future__ import annotations

from typing import Any

from holosoma.managers.observation.terms import locomotion as obs_terms
from holosoma.utils.safe_torch_import import torch

COMPACT_TERM_ORDER = ("base_ang_vel", "dof_pos", "dof_vel", "projected_gravity")
DEFAULT_COMPACT_TERM_SCALE = {
    "base_ang_vel": 0.25,
    "dof_pos": 1.0,
    "dof_vel": 0.05,
    "projected_gravity": 1.0,
}
DEFAULT_COMPACT_TERM_NOISE = {
    "base_ang_vel": 0.3,
    "dof_pos": 0.01,
    "dof_vel": 1.0,
    "projected_gravity": 0.2,
}


def canonicalize_compact_term_scale(term_scale: dict[str, float] | None) -> dict[str, float]:
    if term_scale is None:
        return {k: float(v) for k, v in DEFAULT_COMPACT_TERM_SCALE.items()}
    out: dict[str, float] = {}
    missing = [name for name in COMPACT_TERM_ORDER if name not in term_scale]
    if missing:
        raise ValueError(f"compact_obs term_scale missing terms: {missing}")
    for name in COMPACT_TERM_ORDER:
        out[name] = float(term_scale[name])
    return out


def canonicalize_compact_term_noise(term_noise: dict[str, float] | None) -> dict[str, float]:
    # compact_obs_add_noise (env-extraction noise) is constant-folded to False and does
    # not consume this. The remaining consumer is augment_obs_noise, the separate
    # always-on training-batch noise augmentation (see `_build_augment_noise_scales` in
    # the DAgger trainer).
    if term_noise is None:
        return {k: float(v) for k, v in DEFAULT_COMPACT_TERM_NOISE.items()}
    out: dict[str, float] = {}
    missing = [name for name in COMPACT_TERM_ORDER if name not in term_noise]
    if missing:
        raise ValueError(f"compact_obs term_noise missing terms: {missing}")
    for name in COMPACT_TERM_ORDER:
        out[name] = float(term_noise[name])
    return out


def _extract_raw_compact_terms(env: Any) -> dict[str, torch.Tensor]:
    return {
        "base_ang_vel": obs_terms.base_ang_vel(env),
        "dof_pos": obs_terms.dof_pos(env),
        "dof_vel": obs_terms.dof_vel(env),
        "projected_gravity": obs_terms.projected_gravity(env),
    }


def get_compact_obs_preprocess(
    *,
    term_scale: dict[str, float] | None,
    term_noise: dict[str, float] | None,
) -> dict[str, Any]:
    # Observation-extraction noise (compact_obs_add_noise) has been constant-folded away:
    # it was always False across every trained checkpoint, so noise_enabled/term_noise_effective
    # are now fixed constants. term_noise itself stays a real parameter (not hardcoded to the
    # DEFAULT_COMPACT_TERM_NOISE dict): it is checkpoint metadata reflecting whatever magnitude
    # was actually resolved for augment_obs_noise (the separate, always-on training-batch
    # augmentation kept unmodified per body.tex:124), which can differ per expert checkpoint.
    resolved_scale = canonicalize_compact_term_scale(term_scale)
    resolved_noise = canonicalize_compact_term_noise(term_noise)
    effective_noise = {name: 0.0 for name in COMPACT_TERM_ORDER}
    return {
        "term_order": list(COMPACT_TERM_ORDER),
        "term_scale": resolved_scale,
        "term_noise": resolved_noise,
        "term_noise_effective": effective_noise,
        "noise_enabled": False,
    }


def extract_compact_obs(
    env: Any,
    *,
    term_scale: dict[str, float] | None,
) -> torch.Tensor:
    resolved_scale = canonicalize_compact_term_scale(term_scale)
    raw_terms = _extract_raw_compact_terms(env)

    chunks: list[torch.Tensor] = []
    for term_name in COMPACT_TERM_ORDER:
        value = raw_terms[term_name] * float(resolved_scale[term_name])
        chunks.append(value)
    return torch.cat(chunks, dim=1)
