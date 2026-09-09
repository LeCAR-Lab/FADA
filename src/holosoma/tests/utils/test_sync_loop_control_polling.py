"""Control-channel servicing in the lock-step sync loop.

``DirectSimulation._run_sync_loop`` waits for the policy's ``STEP`` message
between physics batches.  Both paths that recognise ``STEP`` leave that wait
loop with ``break``, so a ``self._poll_control_zmq()`` call placed *after* them
only ever executes on an iteration where no ``STEP`` arrived — that is, only
when the policy is late.  When the policy keeps up (the normal case at 50 Hz)
the control socket is then never serviced, and ``gantry_set_length`` /
``gantry_disable`` sent by an orchestrator time out with ``zmq.error.Again``.

This exercises the loop with a fake sync socket that always has a ``STEP``
ready — the "policy keeps up" case — and asserts the control channel is polled
inside every STEP wait.  No simulator is required: the loop is driven
with a stand-in ``self`` carrying only the attributes it touches.
"""

from __future__ import annotations

import pytest

from holosoma.utils.sim_utils import DirectSimulation

SIM_FPS = 100
RL_RATE = 50.0
# Phase 1b runs a hard-coded 3.0 s settle at ``sim_frequency`` before Phase 2.
SETTLE_STEPS = 3 * SIM_FPS


class _FakeSyncSocket:
    """Sync PAIR socket stand-in: one handshake, then ``STEP`` always ready."""

    def __init__(self, events: list[str]) -> None:
        self._events = events
        self._handshake_sent = False

    def poll(self, timeout: int = 0, flags: int = 0) -> bool:
        return True

    def recv(self) -> bytes:
        if not self._handshake_sent:
            self._handshake_sent = True
            return f"SYNC:{RL_RATE}".encode()
        self._events.append("recv_step")
        return b"STEP"

    def send(self, payload: bytes) -> None:
        self._events.append("send_done")


class _FakeSimulator:
    """Minimal simulator: no ``root_data``, no bridge, no gantry, no backend."""

    root_data = None
    bridge = None
    virtual_gantry = None
    backend = None

    def __init__(self) -> None:
        self.physics_steps = 0

    def simulate_at_each_physics_step(self) -> None:
        self.physics_steps += 1

    def render(self) -> None:  # pragma: no cover - skipped by the test config
        raise AssertionError("render() must not be called with rendering skipped")


class _FakeRunner:
    """Stand-in ``self`` for ``DirectSimulation._run_sync_loop``.

    Shuts the loop down after ``stop_after_batches`` STEP waits so the test
    terminates: the shutdown flag is raised from ``_poll_control_zmq``, which
    is how a real control-channel ``shutdown`` command reaches the loop.
    """

    def __init__(self, events: list[str], stop_after_batches: int) -> None:
        self.events = events
        self.stop_after_batches = stop_after_batches
        self.simulator = _FakeSimulator()
        self._sync_zmq_sock = _FakeSyncSocket(events)
        self._shutdown_requested = False
        self._sync_restart_requested = False
        self._sync_in_phase2 = False
        self.control_polls = 0

    def _poll_control_zmq(self) -> None:
        self.control_polls += 1
        if self._sync_in_phase2:
            self.events.append("poll_control")
            if self.events.count("poll_control") >= self.stop_after_batches:
                self._shutdown_requested = True

    def _log_fps(self, step_count: int, fps_start_time: float) -> float:
        return fps_start_time


def _run_loop(stop_after_batches: int = 3) -> tuple[_FakeRunner, list[str]]:
    events: list[str] = []
    runner = _FakeRunner(events, stop_after_batches)
    DirectSimulation._run_sync_loop(
        runner,
        SIM_FPS,
        viewer_steps=10**9,
        pre_step_refresh=lambda: None,
        skip_headless_mu_render=True,
    )
    return runner, events


def test_control_channel_is_polled_inside_every_step_wait() -> None:
    """Every ``DONE`` -> ``STEP`` wait services the control channel."""
    _runner, events = _run_loop(stop_after_batches=3)

    waits = [i for i, event in enumerate(events) if event == "send_done"]
    assert waits, "loop never completed a physics batch"

    for start in waits:
        following = events[start + 1 :]
        if "recv_step" not in following:
            continue  # final wait, cut short by shutdown
        step_at = following.index("recv_step")
        assert "poll_control" in following[:step_at], (
            "control channel was not polled between DONE and the next STEP; "
            "gantry commands cannot be serviced while the policy keeps up"
        )


def test_sync_loop_runs_deterministic_batches_and_reaches_phase2() -> None:
    """Sanity check on the harness itself: the loop really got to Phase 2."""
    runner, events = _run_loop(stop_after_batches=3)

    steps_per_batch = round(SIM_FPS / RL_RATE)
    batches = events.count("send_done")
    assert batches >= 1
    # Phase 1 contributes one warm-up step before the handshake is read, plus a
    # fixed settle, plus exactly ``steps_per_batch`` steps per completed batch.
    assert runner.simulator.physics_steps == pytest.approx(
        SETTLE_STEPS + steps_per_batch * batches, abs=steps_per_batch
    )
    assert runner._sync_in_phase2 is True
