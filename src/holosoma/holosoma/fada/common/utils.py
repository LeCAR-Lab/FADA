"""Helpers shared by the common/ and planner_idm/ modules."""

from __future__ import annotations

from typing import Any

from holosoma.utils.helpers import get_class

# An oracle checkpoint stores, in `experiment_config.algo._target_`, the dotted path of
# the algo class that produced it, and DAgger reconstructs the expert by resolving that
# string. This map translates dotted paths that name a class by a path it no longer has
# into the path it has now.
_LEGACY_ALGO_TARGETS = {
    "holosoma.agents.ppo.ppo.PPO_DeLA": "holosoma.agents.ppo.ppo.PPO_Deploy",
}


def resolve_algo_class(target: str) -> type:
    """Resolve a checkpoint's stored ``algo._target_`` to a class.

    Behaves exactly like ``get_class`` for any path that still exists; only
    paths in ``_LEGACY_ALGO_TARGETS`` are rewritten first. Unknown paths are
    passed through untouched so a genuinely missing class still raises rather
    than being masked.
    """
    return get_class(_LEGACY_ALGO_TARGETS.get(str(target), str(target)))


def _derive_motor_gains_from_robot_cfg(robot_cfg: dict[str, Any]) -> tuple[list[float], list[float]] | None:
    dof_names = robot_cfg.get("dof_names")
    control_cfg = robot_cfg.get("control")
    if not isinstance(dof_names, (list, tuple)) or not isinstance(control_cfg, dict):
        return None
    stiffness = control_cfg.get("stiffness")
    damping = control_cfg.get("damping")
    if not isinstance(stiffness, dict) or not isinstance(damping, dict):
        return None

    kp_list: list[float] = []
    kd_list: list[float] = []
    for dof_name_raw in dof_names:
        dof_name = str(dof_name_raw)
        matches = [pattern for pattern in stiffness.keys() if str(pattern) in dof_name]
        if len(matches) != 1:
            return None
        pattern = str(matches[0])
        kp_list.append(float(stiffness[pattern]))
        kd_list.append(float(damping[pattern]))
    if not kp_list or len(kp_list) != len(kd_list):
        return None
    return kp_list, kd_list


def _find_motor_gains(node: Any) -> tuple[list[float], list[float]] | None:
    if isinstance(node, dict):
        robot_cfg = node.get("robot")
        if isinstance(robot_cfg, dict):
            derived = _derive_motor_gains_from_robot_cfg(robot_cfg)
            if derived is not None:
                return derived
        kp = node.get("motor_kp")
        kd = node.get("motor_kd")
        if isinstance(kp, (list, tuple)) and isinstance(kd, (list, tuple)):
            return [float(v) for v in kp], [float(v) for v in kd]
        for value in node.values():
            found = _find_motor_gains(value)
            if found is not None:
                return found
        return None
    if isinstance(node, list):
        for value in node:
            found = _find_motor_gains(value)
            if found is not None:
                return found
    return None


def _parse_step_milestones(raw: str | None) -> frozenset[int] | None:
    """Parse --save-step-milestones '1000,2000,4000' → frozenset({1000, 2000, 4000})."""
    if raw is None:
        return None
    token = str(raw).strip()
    if not token:
        return None
    values: list[int] = []
    for part in token.split(","):
        piece = part.strip()
        if not piece:
            continue
        values.append(int(piece, 10))
    if not values:
        return None
    if any(v <= 0 for v in values):
        raise ValueError(f"save-step-milestones must be positive integers, got {raw!r}")
    return frozenset(values)
