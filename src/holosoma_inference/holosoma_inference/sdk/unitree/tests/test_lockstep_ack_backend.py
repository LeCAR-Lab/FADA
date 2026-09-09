"""Policy-side half of the Unitree lock-step lowcmd-ack contract.

The simulator half is ``src/holosoma/tests/bridge/test_unitree_lockstep_ack.py``.
Lock-step stepping requires the simulator to observe a policy-stamped lowcmd
sequence before each physics batch, and on Unitree only the ``sdk2py`` backend
can carry one -- the ``pybind`` ``MotorCommand`` has no field to put it in.

The backend must be the *same* on both sides, and the two predicates live in
packages that cannot import each other.  Each suite therefore pins its own half
without an ``importorskip``, and the simulator suite additionally asserts the two
agree whenever both packages are importable.
"""

from __future__ import annotations

import pytest
from holosoma_inference.config.config_types.task import TaskConfig
from holosoma_inference.sdk.unitree.unitree_interface import _should_use_unitree_sdk2py_backend

_SIM_BACKEND_ENV = "HOLOSOMA_UNITREE_BRIDGE_BACKEND"
_POLICY_BACKEND_ENV = "HOLOSOMA_UNITREE_BACKEND"

_REAL_ROBOT_INTERFACES = ("eth0", "enp3s0", "eno1", "wlan0")


@pytest.fixture(autouse=True)
def _clear_backend_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_SIM_BACKEND_ENV, raising=False)
    monkeypatch.delenv(_POLICY_BACKEND_ENV, raising=False)


def test_task_config_defaults_are_the_case_that_must_work() -> None:
    """Pin the inputs, so the assertions below stay about the *default* user path."""
    assert TaskConfig.interface == "lo"
    assert TaskConfig.domain_id == 0


def test_release_default_selects_the_backend_that_can_ack() -> None:
    """The shipped defaults select the backend that can carry the lowcmd ack."""
    assert _should_use_unitree_sdk2py_backend(TaskConfig.interface, TaskConfig.domain_id) is True


@pytest.mark.parametrize("interface", ["lo", "lo0"])
@pytest.mark.parametrize("domain_id", [0, 1, 82])
def test_loopback_always_selects_sdk2py(interface: str, domain_id: int) -> None:
    """``domain_id`` is an isolation knob and must not gate the ack path."""
    assert _should_use_unitree_sdk2py_backend(interface, domain_id) is True


@pytest.mark.parametrize("interface", _REAL_ROBOT_INTERFACES)
@pytest.mark.parametrize("domain_id", [0, 1, 82])
def test_real_robot_interfaces_stay_on_pybind(interface: str, domain_id: int) -> None:
    """A real robot is reached over a real NIC and keeps the backend it had."""
    assert _should_use_unitree_sdk2py_backend(interface, domain_id) is False


@pytest.mark.parametrize("env_name", [_SIM_BACKEND_ENV, _POLICY_BACKEND_ENV])
@pytest.mark.parametrize(("value", "expected"), [("sdk2py", True), ("pybind", False)])
def test_both_env_var_spellings_are_honoured(
    monkeypatch: pytest.MonkeyPatch, env_name: str, value: str, expected: bool
) -> None:
    """Both ``HOLOSOMA_UNITREE_BRIDGE_BACKEND`` and ``HOLOSOMA_UNITREE_BACKEND`` force
    the backend on this half, matching the simulator side."""
    monkeypatch.setenv(env_name, value)
    assert _should_use_unitree_sdk2py_backend("eth0", 0) is expected


def test_bridge_spelling_wins_when_both_are_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Precedence must match the simulator's, or the pair still disagrees."""
    monkeypatch.setenv(_SIM_BACKEND_ENV, "pybind")
    monkeypatch.setenv(_POLICY_BACKEND_ENV, "sdk2py")
    assert _should_use_unitree_sdk2py_backend("eth0", 0) is False
