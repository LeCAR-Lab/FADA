"""Policy-side contract for the default lock-step rendezvous.

Lock-step stepping is on by default, so the policy points at a simulator
endpoint without being told to.  Three things have to hold:

1. the default endpoint must be the one the simulator binds by default,
2. a run with no simulator (real-robot inference, most notably) must not hang
   on a dead endpoint, and
3. a simulator that refuses this policy must surface as an error, never as a
   silent pairing with someone else's physics.
"""

from __future__ import annotations

import contextlib
import dataclasses
import ipaddress
import pickle
import socket
import types
from collections.abc import Iterator

import pytest
from holosoma_inference.config.config_types.task import TaskConfig
from holosoma_inference.policies import base as policy_base
from holosoma_inference.policies.base import BasePolicy
from holosoma_inference.utils import sync_rendezvous
from holosoma_inference.utils.rate import SimStepSyncRate


def _field_default(name: str):
    return {f.name: f for f in dataclasses.fields(TaskConfig)}[name].default


def test_task_default_url_matches_the_rendezvous_helper() -> None:
    assert _field_default("sim_step_sync_url") == sync_rendezvous.DEFAULT_SIM_STEP_SYNC_URL


def test_lock_step_is_on_by_default() -> None:
    assert _field_default("sim_step_sync_url") is not None


def test_default_url_and_default_port_describe_the_same_endpoint() -> None:
    parsed = sync_rendezvous.parse_tcp_endpoint(sync_rendezvous.DEFAULT_SIM_STEP_SYNC_URL)
    assert parsed == (sync_rendezvous.SYNC_LOOPBACK_HOST, sync_rendezvous.default_policy_sync_port())


def test_simulator_package_default_port_agrees() -> None:
    """The two halves of the default live in packages that cannot import each other.

    They are kept in step by duplication, so this asserts they agree wherever
    both are importable.  The inference environment has both.
    """
    simulator = pytest.importorskip("holosoma.utils.sync_rendezvous")
    assert simulator.default_policy_sync_port() == sync_rendezvous.default_policy_sync_port()
    assert simulator.default_sim_step_sync_url() == sync_rendezvous.DEFAULT_SIM_STEP_SYNC_URL

    simulator_config = pytest.importorskip("holosoma.config_types.simulator")
    port_field = {f.name: f for f in dataclasses.fields(simulator_config.SimulatorInitConfig)}
    port = port_field["policy_sync_zmq_port"].default
    assert sync_rendezvous.parse_tcp_endpoint(_field_default("sim_step_sync_url")) == (
        sync_rendezvous.SYNC_LOOPBACK_HOST,
        port,
    )


# ---------------------------------------------------------------------------
# Probing: a default endpoint with nothing behind it is a normal configuration
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _reserved_dead_endpoint() -> Iterator[str]:
    """Yield a loopback URL that is dead *and* stays dead for the whole test.

    The socket is ``bind``-ed but never ``listen``-ed.  It holds the port (a
    second ``bind`` gets ``EADDRINUSE``, with or without ``SO_REUSEADDR``, and
    ZMQ is no exception), so nothing else on the machine can take it; and a
    ``connect()`` to it is refused, because the kernel has no listening socket to
    hand the connection to.  The port is therefore simultaneously held and dead.
    """
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        holder.bind((sync_rendezvous.SYNC_LOOPBACK_HOST, 0))
        yield f"tcp://{sync_rendezvous.SYNC_LOOPBACK_HOST}:{holder.getsockname()[1]}"
    finally:
        holder.close()


def test_listener_probe_detects_a_live_endpoint() -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind((sync_rendezvous.SYNC_LOOPBACK_HOST, 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    try:
        url = f"tcp://{sync_rendezvous.SYNC_LOOPBACK_HOST}:{port}"
        assert sync_rendezvous.sync_endpoint_has_listener(url, wait_s=0.0) is True
    finally:
        listener.close()


def test_listener_probe_reports_a_dead_endpoint() -> None:
    with _reserved_dead_endpoint() as url:
        assert sync_rendezvous.sync_endpoint_has_listener(url, wait_s=0.0) is False


def test_non_tcp_transports_are_not_probed() -> None:
    """``ipc://``/``inproc://`` cannot be socket-probed; do not claim they are dead."""
    assert sync_rendezvous.sync_endpoint_has_listener("ipc:///tmp/does-not-exist", wait_s=0.0) is True


def _policy_stub(sync_url: str | None) -> types.SimpleNamespace:
    logged: list[str] = []
    stub = types.SimpleNamespace(
        logger=types.SimpleNamespace(info=logged.append),
        logged=logged,
    )
    stub._sync_endpoint_available = types.MethodType(BasePolicy._sync_endpoint_available, stub)
    stub.sync_url = sync_url
    return stub


def test_the_policy_treats_the_shipped_default_as_the_default() -> None:
    """Pins what the two tests below monkeypatch.

    They substitute a default endpoint they own, because the shipped one is a
    fixed port any simulator on this machine may hold.  This asserts the value
    they replace is the one the policy ships with.
    """
    assert policy_base.DEFAULT_SIM_STEP_SYNC_URL == sync_rendezvous.DEFAULT_SIM_STEP_SYNC_URL


def test_dead_default_endpoint_falls_back_to_wall_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dead *default* endpoint falls back to wall-clock rate limiting.

    A ZMQ ``connect()`` succeeds even with nothing listening, so the endpoint is
    probed first.  This covers the default branch only; an explicitly named
    endpoint raises instead (next test).  ``DEFAULT_SIM_STEP_SYNC_URL`` is
    monkeypatched to a port this test owns and has proven dead.
    """
    stub = _policy_stub(None)
    with _reserved_dead_endpoint() as dead_url:
        monkeypatch.setattr(policy_base, "DEFAULT_SIM_STEP_SYNC_URL", dead_url)

        assert stub._sync_endpoint_available(dead_url) is False
    assert any("wall-clock rate limiting" in line for line in stub.logged)


def test_dead_explicit_endpoint_is_an_error_not_a_downgrade(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicitly named endpoint with no listener raises instead of falling back."""
    stub = _policy_stub(None)
    # Two reservations: one stands in for the default (so "explicit" cannot
    # accidentally *be* the default, whatever ports this machine hands out), and
    # one is the named endpoint, held dead for the duration of the assertion.
    with _reserved_dead_endpoint() as default_url, _reserved_dead_endpoint() as explicit:
        monkeypatch.setattr(policy_base, "DEFAULT_SIM_STEP_SYNC_URL", default_url)

        with pytest.raises(RuntimeError) as excinfo:
            stub._sync_endpoint_available(explicit)
        message = str(excinfo.value)
        assert explicit in message
        assert "--simulator.config.policy-sync-zmq-port" in message


# ---------------------------------------------------------------------------
# Handshake identity
# ---------------------------------------------------------------------------


class _ScriptedSocket:
    def __init__(self, reply: bytes) -> None:
        self.sent: list[bytes] = []
        self._reply = reply

    def send(self, payload: bytes) -> None:
        self.sent.append(payload)

    def poll(self, timeout: int = 0, flags: int = 0) -> bool:
        return True

    def recv(self) -> bytes:
        return self._reply


def _rate_with_socket(reply: bytes) -> SimStepSyncRate:
    rate = SimStepSyncRate.__new__(SimStepSyncRate)
    rate._sock = _ScriptedSocket(reply)
    rate._rl_rate = 50.0
    rate._recv_timeout_ms = 1000
    rate._low_state_callback = None
    rate._command_payload_callback = None
    rate._first_step = True
    return rate


def test_handshake_carries_the_user_id() -> None:
    """The handshake frame is `SYNC:<rl_rate>:<peer_uid>`."""
    rate = _rate_with_socket(b"DONE")
    rate.sleep()

    assert rate._sock.sent == [f"SYNC:50.0:{sync_rendezvous.sync_peer_uid()}".encode()]


def test_rejected_handshake_raises_with_the_simulator_reason() -> None:
    reason = "this simulator belongs to uid 1000, the connecting policy to uid 1001"
    rate = _rate_with_socket(pickle.dumps({"type": "REJECT", "reason": reason}))

    with pytest.raises(RuntimeError) as excinfo:
        rate.sleep()
    assert reason in str(excinfo.value)


# ---------------------------------------------------------------------------
# The sync channel deserializes the simulator's replies with pickle.  It is a
# same-host rendezvous by construction -- the simulator binds 127.0.0.1 only
# (SYNC_LOOPBACK_HOST is a module constant, not a flag) -- so a non-loopback
# --task.sim-step-sync-url can never reach a Holosoma simulator and can only
# point that pickle reader at another host.  These tests pin the refusal.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "tcp://127.0.0.1:18500",
        "tcp://127.5.6.7:18500",
        "tcp://localhost:18500",
        "tcp://[::1]:18500",
        "tcp://::1:18500",  # ZMQ accepts the unbracketed form too
        "tcp://[::ffff:127.0.0.1]:18500",  # IPv4-mapped loopback
        "tcp://127.0.0.1:2222",  # an SSH-forwarded port terminates on this host
        "ipc:///tmp/holosoma-sync",
        "inproc://sync",
    ],
)
def test_same_host_sync_urls_are_accepted(url: str) -> None:
    assert sync_rendezvous.sync_url_is_loopback(url) is True


@pytest.mark.parametrize(
    "url",
    [
        "tcp://10.0.0.5:18500",
        "tcp://192.168.1.20:18500",
        "tcp://[2001:db8::1]:18500",
        "tcp://a-host-that-does-not-resolve.invalid:18500",
    ],
)
def test_off_host_sync_urls_are_rejected(url: str) -> None:
    assert sync_rendezvous.sync_url_is_loopback(url) is False


# Unparseable and off-host URLs alike must classify as non-loopback: `None` from
# `parse_tcp_endpoint()` means "not a well-formed TCP endpoint", not "same-host".
@pytest.mark.parametrize(
    "url",
    [
        "pgm://224.0.0.1:5555",  # multicast: the opposite of same-host
        "epgm://eth0;224.0.0.1:5555",
        "ws://example.com:80",  # a routable peer on someone else's machine
        "wss://example.com:443",
        "udp://224.0.0.1:5555",
        "vmci://1:5555",  # VM-to-host, i.e. across a boundary
        "tcp://127.0.0.1",  # well-known host, but no port -- unparseable
        "tcp://127.0.0.1:",
        "tcp://127.0.0.1:abc",
        "tcp://127.0.0.1:0",  # not a connectable port
        "tcp://127.0.0.1:99999",  # out of range
        "tcp://127.0.0.1:-1",
        "tcp://:18500",  # no host
        "tcp://",
        "nonsense",
        "",
        "TCP://127.0.0.1:18500",  # not a ZMQ transport spelling; classify -> reject
    ],
)
def test_unclassifiable_or_off_host_transports_are_rejected(url: str) -> None:
    """URLs that cannot be parsed as a same-host endpoint classify as non-loopback."""
    assert sync_rendezvous.sync_url_is_loopback(url) is False


def test_a_non_string_is_rejected_rather_than_crashing() -> None:
    assert sync_rendezvous.sync_url_is_loopback(None) is False  # type: ignore[arg-type]
    assert sync_rendezvous.sync_url_is_loopback(18500) is False  # type: ignore[arg-type]


def test_same_host_transports_are_a_whitelist_not_an_else_branch() -> None:
    """The accept path names the transports it trusts: `SAME_HOST_TRANSPORTS`."""
    assert sync_rendezvous.SAME_HOST_TRANSPORTS == ("ipc://", "inproc://")
    for transport in sync_rendezvous.SAME_HOST_TRANSPORTS:
        assert sync_rendezvous.sync_url_is_loopback(f"{transport}whatever") is True


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("tcp://127.0.0.1:18500", ("127.0.0.1", 18500)),
        ("tcp://[::1]:18500", ("[::1]", 18500)),
        ("tcp://127.0.0.1", None),
        ("tcp://127.0.0.1:", None),
        ("tcp://127.0.0.1:0", None),
        ("tcp://127.0.0.1:65536", None),
        ("tcp://127.0.0.1:-1", None),
        ("tcp://127.0.0.1: 80", None),
        ("tcp://:18500", None),
        ("ipc:///tmp/x", None),
        ("nonsense", None),
    ],
)
def test_tcp_endpoint_parsing_is_strict(url: str, expected) -> None:
    """`None` from the parser means only "not a well-formed TCP endpoint"."""
    assert sync_rendezvous.parse_tcp_endpoint(url) == expected


def test_the_default_sync_url_is_loopback() -> None:
    assert sync_rendezvous.sync_url_is_loopback(sync_rendezvous.DEFAULT_SIM_STEP_SYNC_URL) is True


def test_constructing_the_rate_with_an_off_host_url_is_refused() -> None:
    """Refused at construction, before any socket exists to receive a pickle."""
    with pytest.raises(ValueError, match="non-loopback sync endpoint") as excinfo:
        SimStepSyncRate("tcp://10.0.0.5:18500", rl_rate=50.0)
    message = str(excinfo.value)
    assert "tcp://10.0.0.5:18500" in message
    assert "pickle" in message
    assert "--task.sim-step-sync-url" in message


def test_loopback_url_still_constructs() -> None:
    """A loopback URL still constructs a `SimStepSyncRate`."""
    rate = SimStepSyncRate(sync_rendezvous.DEFAULT_SIM_STEP_SYNC_URL, rl_rate=50.0)
    try:
        assert rate._rl_rate == 50.0
    finally:
        rate.close()


# ---------------------------------------------------------------------------
# Address pinning.  `sync_url_is_loopback` returns a bool, so a caller using it
# would hand ZMQ the original string and ZMQ would resolve the name a second time;
# DNS answers are not required to agree between two lookups.
#
# `resolve_loopback_sync_url` instead returns the numeric address it validated, and
# the connect uses that, so no name reaches ZMQ.
# ---------------------------------------------------------------------------


class _RecordingSocket:
    def __init__(self) -> None:
        self.connected: list[str] = []
        self.closed = False

    def connect(self, url: str) -> None:
        self.connected.append(url)

    def setsockopt(self, *args, **kwargs) -> None:
        return None

    def close(self, *args, **kwargs) -> None:
        self.closed = True


class _RecordingContext:
    def __init__(self) -> None:
        self.sockets: list[_RecordingSocket] = []

    def socket(self, _kind) -> _RecordingSocket:
        sock = _RecordingSocket()
        self.sockets.append(sock)
        return sock

    def term(self, *args, **kwargs) -> None:
        return None

    def destroy(self, *args, **kwargs) -> None:
        return None


@pytest.fixture
def recording_zmq(monkeypatch: pytest.MonkeyPatch) -> _RecordingContext:
    """Replace the `zmq` module `SimStepSyncRate.__init__` imports, and record connects."""
    import sys

    context = _RecordingContext()
    fake = types.SimpleNamespace(PAIR=object(), RCVTIMEO=object(), LINGER=object(), Context=lambda: context)
    monkeypatch.setitem(sys.modules, "zmq", fake)
    return context


def _connected_url(context: _RecordingContext) -> str:
    assert len(context.sockets) == 1
    assert len(context.sockets[0].connected) == 1
    return context.sockets[0].connected[0]


def test_the_connect_uses_a_numeric_address_not_the_name_that_was_checked(
    recording_zmq: _RecordingContext,
) -> None:
    """`tcp://localhost:...` reaches ZMQ as `tcp://127.0.0.1:...` (or `[::1]`)."""
    SimStepSyncRate("tcp://localhost:18500", rl_rate=50.0)

    connected = _connected_url(recording_zmq)
    host, port = sync_rendezvous.parse_tcp_endpoint(connected)
    assert port == 18500
    literal = host[1:-1] if host.startswith("[") else host
    # Parses as an address == it is not a name. `localhost` raises ValueError here.
    assert ipaddress.ip_address(literal).is_loopback


def test_a_name_that_rebinds_after_the_check_cannot_redirect_the_connect(
    monkeypatch: pytest.MonkeyPatch, recording_zmq: _RecordingContext
) -> None:
    """First lookup says loopback, every later lookup says routable (DNS rebinding).

    The resolver is consulted exactly once and the connect carries what it returned.
    """
    lookups: list[str] = []

    def rebinding_getaddrinfo(host, *args, **kwargs):
        lookups.append(host)
        if len(lookups) == 1:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))]
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("203.0.113.7", 0))]

    monkeypatch.setattr(sync_rendezvous.socket, "getaddrinfo", rebinding_getaddrinfo)

    SimStepSyncRate("tcp://rebind.example:18500", rl_rate=50.0)

    assert lookups == ["rebind.example"], "the name must be resolved exactly once"
    assert _connected_url(recording_zmq) == "tcp://127.0.0.1:18500"


def test_the_pinned_url_is_recorded_on_the_instance(recording_zmq: _RecordingContext) -> None:
    """`sync_url` holds what was connected to, not what was requested."""
    rate = SimStepSyncRate("tcp://localhost:18500", rl_rate=50.0)
    assert rate.sync_url == _connected_url(recording_zmq)
    assert "localhost" not in rate.sync_url


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("tcp://127.0.0.1:18500", "tcp://127.0.0.1:18500"),
        ("tcp://127.5.6.7:18500", "tcp://127.5.6.7:18500"),
        ("tcp://[::1]:18500", "tcp://[::1]:18500"),
        ("tcp://::1:18500", "tcp://[::1]:18500"),  # unbracketed in, bracketed out
        ("ipc:///tmp/holosoma-sync", "ipc:///tmp/holosoma-sync"),
        ("inproc://sync", "inproc://sync"),
        ("tcp://10.0.0.5:18500", None),
        ("tcp://127.0.0.1", None),
        ("pgm://224.0.0.1:5555", None),
        ("nonsense", None),
        ("", None),
    ],
)
def test_resolution_pins_tcp_and_passes_host_free_transports_through(url: str, expected) -> None:
    """`ipc`/`inproc` name no host, so there is nothing to pin and nothing to re-resolve."""
    assert sync_rendezvous.resolve_loopback_sync_url(url) == expected


def test_the_boolean_facade_still_agrees_with_the_resolver() -> None:
    """`sync_url_is_loopback(url)` equals `resolve_loopback_sync_url(url) is not None`."""
    for url in (
        "tcp://127.0.0.1:18500",
        "tcp://localhost:18500",
        "ipc:///tmp/x",
        "tcp://10.0.0.5:18500",
        "nonsense",
        "",
    ):
        assert sync_rendezvous.sync_url_is_loopback(url) is (
            sync_rendezvous.resolve_loopback_sync_url(url) is not None
        )


def test_the_peer_is_not_authenticated_and_the_code_says_so() -> None:
    """The module and class docstrings state that the peer is not authenticated.

    Loopback-only bounds the pickle reader to this host; it does not identify the
    peer -- no key, no signature, no challenge -- so any local process that occupies
    the accepted endpoint first can answer with a crafted pickle.
    """
    for text in (sync_rendezvous.__doc__, SimStepSyncRate.__doc__):
        assert text is not None
        lowered = text.lower()
        assert "authenticate" in lowered
        assert "pickle" in lowered
