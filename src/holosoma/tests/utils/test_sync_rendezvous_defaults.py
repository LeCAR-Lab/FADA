"""Simulator-side contract for the default lock-step rendezvous.

Lock-step stepping is on by default and works only if the simulator binds exactly
the endpoint the policy connects to with no flags passed on either side.  Nothing
at runtime cross-checks that, and the two halves of the default live in *different
installable packages* that cannot import each other.  These tests pin the
agreement, plus the two collision behaviours the default port makes reachable: a
second simulator on a busy port, and a policy owned by another user.
"""

from __future__ import annotations

import dataclasses
import pickle
import socket
import types

import pytest
import zmq
from holosoma.config_types.simulator import SimulatorInitConfig
from holosoma.utils import sync_rendezvous
from holosoma.utils.sim_utils import DirectSimulation


def _field_default(name: str):
    return {f.name: f for f in dataclasses.fields(SimulatorInitConfig)}[name].default


def test_simulator_default_port_matches_the_rendezvous_helper() -> None:
    assert _field_default("policy_sync_zmq_port") == sync_rendezvous.DEFAULT_POLICY_SYNC_PORT


def test_lock_step_is_on_by_default() -> None:
    """A port > 0 is what switches the sim loop out of wall-clock rate limiting."""
    assert _field_default("policy_sync_zmq_port") > 0


def test_default_url_and_default_port_describe_the_same_endpoint() -> None:
    port = sync_rendezvous.default_policy_sync_port()
    assert sync_rendezvous.default_sim_step_sync_url() == f"tcp://{sync_rendezvous.SYNC_LOOPBACK_HOST}:{port}"


def test_default_port_is_per_user_and_deterministic() -> None:
    """Same uid -> same port (the two processes must agree without talking)."""
    assert sync_rendezvous.default_policy_sync_port(1234) == sync_rendezvous.default_policy_sync_port(1234)
    # Different uids map to different ports.
    assert sync_rendezvous.default_policy_sync_port(1234) != sync_rendezvous.default_policy_sync_port(1235)


def test_default_port_avoids_the_ephemeral_range() -> None:
    """A derived port must not land where the kernel hands out source ports."""
    lowest = sync_rendezvous.SYNC_PORT_RANGE_START
    highest = sync_rendezvous.SYNC_PORT_RANGE_START + sync_rendezvous.SYNC_PORT_RANGE_SIZE - 1
    assert 1024 < lowest <= highest < 32768
    for uid in (0, 1, 499, 500, 12345, 2**31):
        assert lowest <= sync_rendezvous.default_policy_sync_port(uid) <= highest


def test_inference_package_agrees_when_it_is_installed() -> None:
    """Cross-package drift guard.

    The authoritative copy of this assertion lives in the inference suite
    (``holosoma_inference/utils/tests/test_sync_rendezvous.py``), which runs only
    in an environment that has both packages.  Repeated here for runs where both
    are importable.
    """
    inference = pytest.importorskip("holosoma_inference.utils.sync_rendezvous")
    assert inference.default_policy_sync_port() == sync_rendezvous.default_policy_sync_port()
    assert inference.default_sim_step_sync_url() == sync_rendezvous.default_sim_step_sync_url()


# ---------------------------------------------------------------------------
# Collision behaviour
# ---------------------------------------------------------------------------


def _runner_for_port(port: int) -> types.SimpleNamespace:
    """Stand-in ``self`` carrying only what ``_init_sync_zmq`` reads."""
    runner = types.SimpleNamespace(
        config=types.SimpleNamespace(
            simulator=types.SimpleNamespace(config=types.SimpleNamespace(policy_sync_zmq_port=port))
        )
    )
    runner._cleanup_sync_zmq = types.MethodType(DirectSimulation._cleanup_sync_zmq, runner)
    return runner


def test_second_simulator_on_a_busy_port_fails_with_an_actionable_message() -> None:
    """Two rollouts started with no flags target the same port.

    The second simulator refuses to start rather than sharing the port or falling
    back to wall-clock stepping, and its message names the flags that separate the
    two runs.
    """
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    blocker.bind((sync_rendezvous.SYNC_LOOPBACK_HOST, 0))
    blocker.listen(1)
    busy_port = blocker.getsockname()[1]
    try:
        runner = _runner_for_port(busy_port)
        with pytest.raises(RuntimeError) as excinfo:
            DirectSimulation._init_sync_zmq(runner)
        message = str(excinfo.value)
        assert str(busy_port) in message
        assert "--simulator.config.policy-sync-zmq-port" in message
        assert "--task.sim-step-sync-url" in message
        # The failed attempt must not leave a half-open context behind.
        assert runner._sync_zmq_sock is None
        assert runner._sync_zmq_ctx is None
    finally:
        blocker.close()


def _bind_sync_zmq_on_a_free_port(attempts: int = 8) -> types.SimpleNamespace:
    """Bind the sync socket on a port nothing else on this machine holds.

    The port cannot be reserved in advance -- holding it is what the code under
    test does -- so a port borrowed from the OS may be taken between the probe and
    the bind.  Retry on the ``RuntimeError`` ``_init_sync_zmq`` raises for a failed
    bind, and re-raise on the last attempt.
    """
    for attempt in range(attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((sync_rendezvous.SYNC_LOOPBACK_HOST, 0))
            free_port = probe.getsockname()[1]
        runner = _runner_for_port(free_port)
        try:
            DirectSimulation._init_sync_zmq(runner)
        except RuntimeError:
            if attempt == attempts - 1:
                raise
            continue
        return runner
    raise AssertionError("unreachable")


def test_sync_socket_is_bound_loopback_only() -> None:
    """The sync socket binds on the loopback host, so it is not reachable off-host."""
    runner = _bind_sync_zmq_on_a_free_port()
    try:
        assert runner._sync_zmq_sock is not None
        endpoint = runner._sync_zmq_sock.getsockopt_string(zmq.LAST_ENDPOINT)
        assert endpoint.startswith(f"tcp://{sync_rendezvous.SYNC_LOOPBACK_HOST}:")
    finally:
        DirectSimulation._cleanup_sync_zmq(runner)


def test_disabled_port_binds_nothing() -> None:
    """``-1`` is the opt-out: no socket and no context are created."""
    runner = _runner_for_port(-1)
    DirectSimulation._init_sync_zmq(runner)
    assert runner._sync_zmq_sock is None
    assert runner._sync_zmq_ctx is None


# ---------------------------------------------------------------------------
# Handshake identity: the cross-user case the derived port does not cover
# ---------------------------------------------------------------------------

_SIM_FPS = 100
_RL_RATE = 50.0


class _HandshakeSocket:
    """Sync PAIR stand-in replaying a scripted sequence of handshake attempts."""

    def __init__(self, handshakes: list[bytes], sent: list[bytes]) -> None:
        self._handshakes = list(handshakes)
        self._sent = sent

    def poll(self, timeout: int = 0, flags: int = 0) -> bool:
        return True

    def recv(self) -> bytes:
        if self._handshakes:
            return self._handshakes.pop(0)
        return b"STEP"

    def send(self, payload: bytes) -> None:
        self._sent.append(payload)


class _NullSimulator:
    root_data = None
    bridge = None
    virtual_gantry = None
    backend = None

    def __init__(self) -> None:
        self.physics_steps = 0

    def simulate_at_each_physics_step(self) -> None:
        self.physics_steps += 1

    def render(self) -> None:  # pragma: no cover - rendering is skipped
        raise AssertionError("render() must not be called")


class _HandshakeRunner:
    """Stand-in ``self`` that shuts the loop down once Phase 2 is reached."""

    def __init__(self, handshakes: list[bytes]) -> None:
        self.sent: list[bytes] = []
        self.simulator = _NullSimulator()
        self._sync_zmq_sock = _HandshakeSocket(handshakes, self.sent)
        self._shutdown_requested = False
        self._sync_restart_requested = False
        self._sync_in_phase2 = False

    def _poll_control_zmq(self) -> None:
        if self._sync_in_phase2:
            self._shutdown_requested = True

    def _log_fps(self, step_count: int, fps_start_time: float) -> float:
        return fps_start_time


def _run_handshakes(handshakes: list[bytes]) -> _HandshakeRunner:
    runner = _HandshakeRunner(handshakes)
    DirectSimulation._run_sync_loop(
        runner,
        _SIM_FPS,
        viewer_steps=10**9,
        pre_step_refresh=lambda: None,
        skip_headless_mu_render=True,
    )
    return runner


def _rejections(runner: _HandshakeRunner) -> list[dict]:
    out = []
    for payload in runner.sent:
        # DONE frames are plain bytes, not pickles; those are not rejections.
        try:
            decoded = pickle.loads(payload)
        except Exception:  # noqa: S112
            continue
        if isinstance(decoded, dict) and decoded.get("type") == "REJECT":
            out.append(decoded)
    return out


def test_handshake_from_another_user_is_rejected_not_served() -> None:
    """Two uids can collide modulo the port range.

    The simulator answers a foreign uid with an explicit refusal and keeps waiting
    for its own user's policy.
    """
    own = sync_rendezvous.sync_peer_uid()
    runner = _run_handshakes([f"SYNC:{_RL_RATE}:{own + 1}".encode(), f"SYNC:{_RL_RATE}:{own}".encode()])

    rejects = _rejections(runner)
    assert len(rejects) == 1
    assert str(own + 1) in rejects[0]["reason"]
    # The simulator goes on to serve its own user's policy.
    assert runner._sync_in_phase2 is True


def test_handshake_from_the_same_user_is_accepted() -> None:
    own = sync_rendezvous.sync_peer_uid()
    runner = _run_handshakes([f"SYNC:{_RL_RATE}:{own}".encode()])

    assert _rejections(runner) == []
    assert runner._sync_in_phase2 is True


def test_handshake_without_a_uid_is_still_accepted() -> None:
    """The uid field of the ``SYNC:<rate>[:<uid>]`` frame is optional."""
    runner = _run_handshakes([f"SYNC:{_RL_RATE}".encode()])

    assert _rejections(runner) == []
    assert runner._sync_in_phase2 is True
