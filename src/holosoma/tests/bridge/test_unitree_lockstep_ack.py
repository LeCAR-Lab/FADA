"""Unitree lock-step lowcmd-ack contract.

Lock-step stepping is on by default, and before every policy-driven physics batch
the simulator requires an *observable* lowcmd sequence: proof that the command it
is about to integrate is the one the policy just sent.  Booster/T1 satisfies this
two ways over (its bridge stamps the sequence unconditionally, *and* its policy
ships the command in-band on the STEP message).  Unitree/G1 has only one way: the
``sdk2py`` DDS backend, whose ``LowCmd`` carries a ``reserve`` field the policy
stamps and the simulator's subscriber reads back.

The ``pybind`` backend has no such field -- ``create_robot()``'s ``MotorCommand``
exposes only q_target/dq_target/tau_ff/kp/kd -- so a G1 running on it never
produces an ack and lock-step aborts on its first check.

These tests pin the backend-selection predicate, the fact that the simulator's and
the policy's predicates agree, and that the ack requirement still aborts the run
when it cannot be met.
"""

from __future__ import annotations

import sys
import types

import pytest

# ---------------------------------------------------------------------------
# Importing the simulator-side bridge
# ---------------------------------------------------------------------------
# ``unitree_sdk2py_bridge`` imports the ``unitree_interface`` C++ binding at
# module scope.  The binding is installed in the MuJoCo and inference
# environments but not in the IsaacSim one, and the backend predicate under test
# does not touch it, so stub it when it is missing rather than skipping.
if "unitree_interface" not in sys.modules:
    try:  # pragma: no cover - depends on the environment the suite runs in
        import unitree_interface  # noqa: F401
    except ImportError:  # pragma: no cover
        _stub = types.ModuleType("unitree_interface")
        for _name in (
            "LowState",
            "MessageType",
            "MotorCommand",
            "RobotType",
            "UnitreeInterface",
            "WirelessController",
        ):
            setattr(_stub, _name, type(_name, (), {}))
        sys.modules["unitree_interface"] = _stub

from holosoma.bridge.unitree.unitree_sdk2py_bridge import UnitreeSdk2Bridge  # noqa: E402
from holosoma.utils.sim_utils import DirectSimulation  # noqa: E402

_SIM_BACKEND_ENV = "HOLOSOMA_UNITREE_BRIDGE_BACKEND"
_POLICY_BACKEND_ENV = "HOLOSOMA_UNITREE_BACKEND"

_sim_selects_sdk2py = UnitreeSdk2Bridge._should_use_sdk2py_backend

# Interfaces a real Unitree robot is reached on.  A robot is never reachable over
# loopback, so the interface discriminates a same-host simulator from hardware.
_REAL_ROBOT_INTERFACES = ("eth0", "enp3s0", "eno1", "wlan0")
_LOOPBACK_INTERFACES = ("lo", "lo0", "LO", " lo ")


@pytest.fixture(autouse=True)
def _clear_backend_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neither variable may be set: these tests pin the *unforced* decision."""
    monkeypatch.delenv(_SIM_BACKEND_ENV, raising=False)
    monkeypatch.delenv(_POLICY_BACKEND_ENV, raising=False)


def _policy_selects_sdk2py():
    """The policy-side twin, which lives in the other installable package."""
    module = pytest.importorskip("holosoma_inference.sdk.unitree.unitree_interface")
    return module._should_use_unitree_sdk2py_backend


# ---------------------------------------------------------------------------
# The defect itself
# ---------------------------------------------------------------------------


def test_release_default_g1_selects_the_backend_that_can_ack() -> None:
    """``interface='lo'`` with ``domain_id=0`` selects ``sdk2py`` on both sides.

    That pair is the simulator bridge's auto-detected interface on Linux and the
    policy's ``TaskConfig`` default -- the configuration README step 4 gives with
    no flags.  Under ``pybind`` G1 aborts at physics step 80 with "Sync lowcmd ack
    is required".
    """
    assert _sim_selects_sdk2py("lo", 0) is True
    assert _policy_selects_sdk2py()("lo", 0) is True


@pytest.mark.parametrize("interface", _LOOPBACK_INTERFACES)
@pytest.mark.parametrize("domain_id", [0, 1, 82])
def test_loopback_always_selects_sdk2py(interface: str, domain_id: int) -> None:
    """Loopback means a same-host simulator peer, whatever the DDS domain is.

    ``domain_id`` is a DDS isolation knob and does not gate the selection.
    """
    assert _sim_selects_sdk2py(interface, domain_id) is True
    assert _policy_selects_sdk2py()(interface, domain_id) is True


# ---------------------------------------------------------------------------
# Constraint: real-robot paths must not move
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("interface", _REAL_ROBOT_INTERFACES)
@pytest.mark.parametrize("domain_id", [0, 1, 82])
def test_real_robot_interfaces_stay_on_pybind(interface: str, domain_id: int) -> None:
    """A real NIC stays on ``pybind`` for every domain id.

    Selecting ``sdk2py`` for a real NIC would be a hardware behaviour change.
    """
    assert _sim_selects_sdk2py(interface, domain_id) is False
    assert _policy_selects_sdk2py()(interface, domain_id) is False


# ---------------------------------------------------------------------------
# The two halves must agree
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("interface", [*_LOOPBACK_INTERFACES, *_REAL_ROBOT_INTERFACES, None, ""])
@pytest.mark.parametrize("domain_id", [0, 1, 82])
def test_simulator_and_policy_predicates_agree(interface: str | None, domain_id: int) -> None:
    """The backend is a property of the *pair* of processes.

    They negotiate nothing at runtime and live in packages that cannot import
    each other; on a disagreement one side publishes on a channel the other is
    not subscribed to, with no error from either.
    """
    assert _sim_selects_sdk2py(interface, domain_id) == _policy_selects_sdk2py()(interface, domain_id)


@pytest.mark.parametrize("env_name", [_SIM_BACKEND_ENV, _POLICY_BACKEND_ENV])
@pytest.mark.parametrize(("value", "expected"), [("sdk2py", True), ("pybind", False)])
def test_either_env_var_name_moves_both_halves(
    monkeypatch: pytest.MonkeyPatch, env_name: str, value: str, expected: bool
) -> None:
    """Both env var spellings steer both processes.

    ``HOLOSOMA_UNITREE_BRIDGE_BACKEND`` and ``HOLOSOMA_UNITREE_BACKEND`` each
    override the predicate on the simulator side and on the policy side; the
    accepted values are ``sdk2py`` and ``pybind``.
    """
    monkeypatch.setenv(env_name, value)
    assert _sim_selects_sdk2py("eth0", 0) is expected
    assert _policy_selects_sdk2py()("eth0", 0) is expected


# ---------------------------------------------------------------------------
# The ack guarantee must still bite
# ---------------------------------------------------------------------------
#
# These drive the real Phase-2 loop with a bridge that cannot report a sequence
# and assert the run stops instead of integrating a stale command.

_SIM_FPS = 100
_RL_RATE = 50.0
_STEPS_PER_BATCH = int(_SIM_FPS / _RL_RATE)
# The ack is only checked from the third batch on (``batch_count >= 2``), so an
# aborting run gets exactly two batches' worth of physics in.
_STEPS_BEFORE_FIRST_ACK_CHECK = 2 * _STEPS_PER_BATCH


class _MuteBridge:
    """A bridge that never publishes a lowcmd sequence (the pybind G1 case)."""

    _sdk2py_low_cmd_seq = None


class _AckingBridge:
    """A bridge that reports a fresh sequence per read (the sdk2py G1 case)."""

    def __init__(self) -> None:
        self._seq = 0

    @property
    def _sdk2py_low_cmd_seq(self) -> int:
        self._seq += 1
        return self._seq


class _StepSocket:
    """Sync PAIR stand-in: one handshake, then an unbounded run of STEPs."""

    def __init__(self) -> None:
        self._handshake_pending = True

    def poll(self, timeout: int = 0, flags: int = 0) -> bool:
        return True

    def recv(self) -> bytes:
        if self._handshake_pending:
            self._handshake_pending = False
            return f"SYNC:{_RL_RATE}".encode()
        return b"STEP"

    def send(self, payload: bytes) -> None:
        return None


class _CountingSimulator:
    root_data = None
    virtual_gantry = None
    backend = None

    def __init__(self, robot_bridge: object) -> None:
        self.bridge = types.SimpleNamespace(robot_bridge=robot_bridge)
        self.physics_steps = 0

    def simulate_at_each_physics_step(self) -> None:
        self.physics_steps += 1

    def render(self) -> None:  # pragma: no cover - rendering is skipped
        raise AssertionError("render() must not be called")


class _AckRunner:
    """Stand-in ``self`` that lets Phase 2 run a bounded number of batches.

    Phase 1 also steps physics (the handshake wait and the fixed 3 s settle), so
    the counter is rebased where the loop sets ``_sync_in_phase2``, making
    ``phase2_steps`` independent of the settle length.
    """

    def __init__(self, robot_bridge: object, stop_after: int) -> None:
        self.simulator = _CountingSimulator(robot_bridge)
        self._sync_zmq_sock = _StepSocket()
        self._shutdown_requested = False
        self._sync_restart_requested = False
        self._phase2_base: int | None = None
        self._stop_after = stop_after
        self._in_phase2 = False

    @property
    def _sync_in_phase2(self) -> bool:
        return self._in_phase2

    @_sync_in_phase2.setter
    def _sync_in_phase2(self, value: bool) -> None:
        self._in_phase2 = value
        if value and self._phase2_base is None:
            self._phase2_base = self.simulator.physics_steps

    @property
    def phase2_steps(self) -> int:
        if self._phase2_base is None:
            return 0
        return self.simulator.physics_steps - self._phase2_base

    def _poll_control_zmq(self) -> None:
        if self.phase2_steps >= self._stop_after:
            self._shutdown_requested = True

    def _log_fps(self, step_count: int, fps_start_time: float) -> float:
        return fps_start_time


def _run_phase2(robot_bridge: object, stop_after: int) -> _AckRunner:
    runner = _AckRunner(robot_bridge, stop_after)
    DirectSimulation._run_sync_loop(
        runner,
        _SIM_FPS,
        viewer_steps=10**9,
        pre_step_refresh=lambda: None,
        skip_headless_mu_render=True,
    )
    return runner


def test_missing_lowcmd_sequence_still_aborts_the_run() -> None:
    """A bridge that cannot report a sequence stops the run at the first ack check.

    It does not degrade to best-effort command delivery.
    """
    # Ask for far more batches than the ack check allows, so "stopped early" can
    # only mean the ack aborted it.
    runner = _run_phase2(_MuteBridge(), stop_after=40 * _STEPS_PER_BATCH)
    assert runner.phase2_steps == _STEPS_BEFORE_FIRST_ACK_CHECK
    assert runner._shutdown_requested is False  # it aborted; it was not asked to stop


def test_a_bridge_that_acks_runs_past_the_check() -> None:
    """A bridge that acks runs past the check: the abort above is the ack, not the harness."""
    wanted = 10 * _STEPS_PER_BATCH
    runner = _run_phase2(_AckingBridge(), stop_after=wanted)
    assert runner.phase2_steps > _STEPS_BEFORE_FIRST_ACK_CHECK
    assert runner.phase2_steps >= wanted
