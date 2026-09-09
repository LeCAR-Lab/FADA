from __future__ import annotations

from typing import Any

import numpy as np

from holosoma.utils.safe_torch_import import torch

COMMAND_PROFILE_LOCOMOTION = "locomotion"
DEFAULT_COMMAND_PROFILE = COMMAND_PROFILE_LOCOMOTION

CURRENT_COMMAND_COMPONENTS: tuple[str, ...] = (
    "command_lin_vel",
    "command_ang_vel",
    "sin_phase",
    "cos_phase",
)
CURRENT_COMMAND_DIM = 7

COMMAND_PROFILE_COMPONENTS: dict[str, tuple[str, ...]] = {
    COMMAND_PROFILE_LOCOMOTION: CURRENT_COMMAND_COMPONENTS,
}
COMMAND_PROFILE_DIMS: dict[str, int] = {
    COMMAND_PROFILE_LOCOMOTION: CURRENT_COMMAND_DIM,
}


def normalize_command_profile(profile: str | None) -> str:
    key = (profile or DEFAULT_COMMAND_PROFILE).strip().lower().replace("-", "_")
    if key not in COMMAND_PROFILE_DIMS:
        supported = ", ".join(sorted(COMMAND_PROFILE_DIMS))
        raise ValueError(f"Unsupported command_profile={profile!r}. Supported: {supported}")
    return key


def command_components_for_profile(profile: str | None) -> tuple[str, ...]:
    return COMMAND_PROFILE_COMPONENTS[normalize_command_profile(profile)]


def command_dim_for_profile(profile: str | None) -> int:
    return int(COMMAND_PROFILE_DIMS[normalize_command_profile(profile)])


def _phase_features(env: Any) -> tuple[torch.Tensor, torch.Tensor]:
    command_manager = getattr(env, "command_manager", None)
    gait_state = command_manager.get_state("locomotion_gait") if command_manager is not None else None
    phase = getattr(gait_state, "phase", None)
    if phase is None:
        raise AttributeError("locomotion_gait is not registered with the command manager.")
    return torch.sin(phase), torch.cos(phase)


def _as_2d_float32_np(array: np.ndarray | list[float] | list[list[float]]) -> np.ndarray:
    out = np.asarray(array, dtype=np.float32)
    if out.ndim == 1:
        out = out.reshape(1, -1)
    return out


def build_current_command_torch(
    *,
    command_lin_vel: torch.Tensor,
    command_ang_vel: torch.Tensor,
    sin_phase: torch.Tensor | None = None,
    cos_phase: torch.Tensor | None = None,
    profile: str | None = None,
) -> torch.Tensor:
    profile = normalize_command_profile(profile)
    batch = int(command_lin_vel.shape[0])
    required: list[tuple[str, torch.Tensor, int]] = [
        ("command_lin_vel", command_lin_vel, 2),
        ("command_ang_vel", command_ang_vel, 1),
    ]
    pieces: list[torch.Tensor] = [command_lin_vel, command_ang_vel]
    if profile == COMMAND_PROFILE_LOCOMOTION:
        if sin_phase is None or cos_phase is None:
            raise ValueError("locomotion command profile requires sin_phase and cos_phase")
        required.extend(
            [
                ("sin_phase", sin_phase, 2),
                ("cos_phase", cos_phase, 2),
            ]
        )
        pieces.extend([sin_phase, cos_phase])
    for name, value, width in required:
        if value.ndim != 2 or value.shape[0] != batch or value.shape[1] != width:
            raise ValueError(
                f"{name} shape mismatch: got {tuple(value.shape)}, expected {(batch, width)}"
            )
    out = torch.cat(pieces, dim=1)
    expected_dim = command_dim_for_profile(profile)
    if out.shape[1] != expected_dim:
        raise ValueError(f"current_command dim mismatch for {profile}: got {out.shape[1]}, expected {expected_dim}")
    return out


def extract_current_command_torch(env: Any, *, profile: str | None = None) -> torch.Tensor:
    profile = normalize_command_profile(profile)
    commands = env.command_manager.commands
    if commands.ndim != 2 or commands.shape[1] < 3:
        raise ValueError(
            "env.command_manager.commands must be rank-2 with at least 3 dims, "
            f"got {tuple(commands.shape)}"
        )
    sin_phase = cos_phase = None
    if profile == COMMAND_PROFILE_LOCOMOTION:
        sin_phase, cos_phase = _phase_features(env)
    return build_current_command_torch(
        command_lin_vel=commands[:, :2],
        command_ang_vel=commands[:, 2:3],
        sin_phase=sin_phase,
        cos_phase=cos_phase,
        profile=profile,
    )


def extract_tracking_command_torch(env: Any) -> torch.Tensor:
    commands = env.command_manager.commands
    if commands.ndim != 2:
        raise ValueError(f"env.command_manager.commands must be rank-2, got {tuple(commands.shape)}")
    if commands.shape[1] >= 3:
        return commands[:, :3]
    pad = torch.zeros((commands.shape[0], 3 - commands.shape[1]), device=commands.device, dtype=commands.dtype)
    return torch.cat([commands, pad], dim=1)


def build_current_command_np(
    *,
    command_lin_vel: np.ndarray | list[float] | list[list[float]],
    command_ang_vel: np.ndarray | list[float] | list[list[float]],
    sin_phase: np.ndarray | list[float] | list[list[float]] | None = None,
    cos_phase: np.ndarray | list[float] | list[list[float]] | None = None,
    profile: str | None = None,
) -> np.ndarray:
    profile = normalize_command_profile(profile)
    lin = _as_2d_float32_np(command_lin_vel)
    ang = _as_2d_float32_np(command_ang_vel)
    batch = int(lin.shape[0])
    required: list[tuple[str, np.ndarray, int]] = [
        ("command_lin_vel", lin, 2),
        ("command_ang_vel", ang, 1),
    ]
    pieces: list[np.ndarray] = [lin, ang]
    if profile == COMMAND_PROFILE_LOCOMOTION:
        if sin_phase is None or cos_phase is None:
            raise ValueError("locomotion command profile requires sin_phase and cos_phase")
        sin_p = _as_2d_float32_np(sin_phase)
        cos_p = _as_2d_float32_np(cos_phase)
        required.extend(
            [
                ("sin_phase", sin_p, 2),
                ("cos_phase", cos_p, 2),
            ]
        )
        pieces.extend([sin_p, cos_p])
    for name, value, width in required:
        if value.ndim != 2 or value.shape[0] != batch or value.shape[1] != width:
            raise ValueError(
                f"{name} shape mismatch: got {value.shape}, expected {(batch, width)}"
            )
    out = np.concatenate(pieces, axis=1).astype(np.float32, copy=False)
    expected_dim = command_dim_for_profile(profile)
    if out.shape[1] != expected_dim:
        raise ValueError(f"current_command dim mismatch for {profile}: got {out.shape[1]}, expected {expected_dim}")
    return out
