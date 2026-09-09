from __future__ import annotations

from pathlib import Path

from holosoma.fada.planner_idm.trainer_logging import _LoggingMixin


def _load_reward_curve(checkpoint_dir: Path) -> dict[int, float]:
    # _load_reward_curve is an instance method of _LoggingMixin (mixed into
    # FADATrainer); the mixin has no required constructor state, so a bare
    # instance is enough to exercise it directly without building a full trainer.
    return _LoggingMixin()._load_reward_curve(checkpoint_dir)


def test_reward_curve_json_takes_priority(tmp_path: Path) -> None:
    (tmp_path / "reward_curve.json").write_text('{"100": 1.0, "200": 2.0}')
    curve = _load_reward_curve(tmp_path)
    assert curve == {100: 1.0, 200: 2.0}


def test_missing_reward_curve_returns_empty(tmp_path: Path) -> None:
    assert _load_reward_curve(tmp_path) == {}
