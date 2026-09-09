from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from holosoma.utils.safe_torch_import import torch


class _LoggingMixin:
    @staticmethod
    def _load_reward_curve_from_tensorboard(checkpoint_dir: Path) -> dict[int, float]:
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        except Exception:
            return {}

        event_files = sorted(checkpoint_dir.glob("events.out.tfevents.*"))
        if not event_files:
            return {}
        try:
            accumulator = EventAccumulator(str(event_files[-1]))
            accumulator.Reload()
            tags = set(accumulator.Tags().get("scalars", []))
            if "Train/mean_reward" not in tags:
                return {}
            curve: dict[int, float] = {}
            for scalar in accumulator.Scalars("Train/mean_reward"):
                curve[int(scalar.step)] = float(scalar.value)
            return curve
        except Exception:
            return {}

    @staticmethod
    def _load_reward_curve_from_json(checkpoint_dir: Path) -> dict[int, float]:
        """Load a pre-computed reward curve shipped alongside a released checkpoint set.

        Released oracle checkpoint bundles ship a `reward_curve.json` (mapping
        stringified train step -> mean reward) next to the `model_*.pt` files, since
        the raw `events.out.tfevents.*` / wandb `output.log` sources this is derived
        from are not part of the release artifact. Checked first (before tfevents and
        wandb) so a release checkpoint directory never silently falls through to the
        other two sources.
        """
        json_path = checkpoint_dir / "reward_curve.json"
        if not json_path.is_file():
            return {}
        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(raw, dict):
            return {}
        try:
            return {int(step): float(reward) for step, reward in raw.items()}
        except (TypeError, ValueError):
            return {}

    @staticmethod
    def _load_reward_curve_from_output_log(checkpoint_dir: Path) -> dict[int, float]:
        output_logs = sorted(checkpoint_dir.glob(".wandb/wandb/run-*/files/output.log"))
        if not output_logs:
            return {}

        curve: dict[int, float] = {}
        iter_pattern = re.compile(r"Learning iteration\s+(\d+)/")
        reward_pattern = re.compile(r"Mean reward:\s*(-?\d+(?:\.\d+)?)")
        current_iteration: int | None = None
        try:
            text = output_logs[-1].read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return {}
        for line in text.splitlines():
            iter_match = iter_pattern.search(line)
            if iter_match:
                current_iteration = int(iter_match.group(1))
            reward_match = reward_pattern.search(line)
            if reward_match and current_iteration is not None:
                curve[current_iteration] = float(reward_match.group(1))
        return curve

    def _load_reward_curve(self, checkpoint_dir: Path) -> dict[int, float]:
        curve = self._load_reward_curve_from_json(checkpoint_dir)
        if curve:
            return curve
        curve = self._load_reward_curve_from_tensorboard(checkpoint_dir)
        if curve:
            return curve
        return self._load_reward_curve_from_output_log(checkpoint_dir)

    def dump_training_log(self, path: str | Path, *, logs: list[dict[str, Any]]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(logs, indent=2), encoding="utf-8")
        return path
