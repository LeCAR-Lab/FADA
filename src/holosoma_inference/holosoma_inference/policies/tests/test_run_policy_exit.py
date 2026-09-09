from __future__ import annotations

import signal
from pathlib import Path
from types import SimpleNamespace

import pytest

from holosoma_inference import run_policy as run_policy_module


class _DummyVelStateProcessor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def plot_mocap_raw_on_exit(
        self,
        save_path: str | None = None,
        tracking_reward_checkpoint_path: str | None = None,
    ) -> None:
        self.calls.append((save_path or "", tracking_reward_checkpoint_path))


class _InterruptingPolicy:
    last_instance: _InterruptingPolicy | None = None

    def __init__(self, config) -> None:
        self.config = config
        self.interface = SimpleNamespace(vel_state_processor=_DummyVelStateProcessor())
        type(self).last_instance = self

    def _resolve_log_output_dir(self) -> str:
        return "/tmp/run-policy-exit-test"

    def run(self) -> None:
        raise KeyboardInterrupt


class _SignalInterruptingPolicy:
    last_instance: _SignalInterruptingPolicy | None = None
    signal_handlers: dict[int, object] = {}

    def __init__(self, config) -> None:
        self.config = config
        self.interface = SimpleNamespace(vel_state_processor=_DummyVelStateProcessor())
        type(self).last_instance = self

    def _resolve_log_output_dir(self) -> str:
        return "/tmp/run-policy-signal-test"

    def run(self) -> None:
        handler = type(self).signal_handlers[signal.SIGINT]
        handler(signal.SIGINT, None)


def test_run_policy_saves_exit_plots_on_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    registered_callbacks: list = []

    monkeypatch.setattr(run_policy_module, "restore_terminal_settings", lambda: None)
    monkeypatch.setattr(run_policy_module.atexit, "register", lambda fn: registered_callbacks.append(fn))
    monkeypatch.setattr(
        run_policy_module,
        "_resolve_policy_candidates",
        lambda config: ("test", [_InterruptingPolicy], False),
    )

    config = SimpleNamespace(
        robot=SimpleNamespace(robot_type="t1_23dof"),
        observation=SimpleNamespace(obs_dict={"actor_obs": ["foo"]}),
        task=SimpleNamespace(
            rl_rate=50,
            model_path="/tmp/model.onnx",
            seed=None,
            plot_mocap_raw_on_exit=True,
            tracking_reward_checkpoint_path="/tmp/model.pt",
            use_joystick=False,
        ),
        secondary=None,
    )

    with pytest.raises(KeyboardInterrupt):
        run_policy_module.run_policy(config)

    assert len(registered_callbacks) == 1
    callback = registered_callbacks[0]
    assert _InterruptingPolicy.last_instance is not None
    proc = _InterruptingPolicy.last_instance.interface.vel_state_processor

    assert proc.calls == [("/tmp/run-policy-exit-test/mocap_raw.png", "/tmp/model.pt")]

    callback()
    assert proc.calls == [("/tmp/run-policy-exit-test/mocap_raw.png", "/tmp/model.pt")]


def test_run_policy_saves_exit_plots_from_sigint_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    registered_callbacks: list = []
    signal_handlers: dict[int, object] = {}

    monkeypatch.setattr(run_policy_module, "restore_terminal_settings", lambda: None)
    monkeypatch.setattr(run_policy_module.atexit, "register", lambda fn: registered_callbacks.append(fn))
    monkeypatch.setattr(
        run_policy_module,
        "_resolve_policy_candidates",
        lambda config: ("test", [_SignalInterruptingPolicy], False),
    )

    def _capture_signal(sig, handler):
        previous = signal_handlers.get(sig)
        signal_handlers[sig] = handler
        _SignalInterruptingPolicy.signal_handlers = signal_handlers
        return previous

    monkeypatch.setattr(run_policy_module.signal, "signal", _capture_signal)

    config = SimpleNamespace(
        robot=SimpleNamespace(robot_type="t1_23dof"),
        observation=SimpleNamespace(obs_dict={"actor_obs": ["foo"]}),
        task=SimpleNamespace(
            rl_rate=50,
            model_path="/tmp/model.onnx",
            seed=None,
            plot_mocap_raw_on_exit=True,
            tracking_reward_checkpoint_path="/tmp/model.pt",
            use_joystick=False,
        ),
        secondary=None,
    )

    with pytest.raises(KeyboardInterrupt):
        run_policy_module.run_policy(config)

    assert _SignalInterruptingPolicy.last_instance is not None
    proc = _SignalInterruptingPolicy.last_instance.interface.vel_state_processor
    assert proc.calls == [("/tmp/run-policy-signal-test/mocap_raw.png", "/tmp/model.pt")]
