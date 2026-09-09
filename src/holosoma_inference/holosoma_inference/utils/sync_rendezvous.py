"""Default rendezvous for lock-step policy/simulator stepping (policy side).

The constants and the ``sync_peer_uid`` / ``default_policy_sync_port`` /
``default_sim_step_sync_url`` functions below are duplicated verbatim from
``holosoma/utils/sync_rendezvous.py``.  The duplication is required because the
simulator and the policy run in *separate* environments (see the four
``scripts/source_*_setup.sh`` variants), so neither package may import the
other.  ``src/holosoma/tests/utils/test_sync_rendezvous_defaults.py`` imports
both copies and fails if they ever disagree.

Read that module's docstring for why the default port is derived from the user
id instead of being a single fixed number.  This copy adds one policy-only
helper, :func:`sync_endpoint_has_listener`, described below, and one
policy-only guard, :func:`resolve_loopback_sync_url` (with its boolean façade
:func:`sync_url_is_loopback`).

**What the guard does and does not protect against.**  It establishes that the
endpoint is on this host.  It does *not* authenticate the peer: the sync channel
carries no key, no signature and no challenge, and the policy deserializes the
replies with :mod:`pickle`.  Any local process that can occupy the accepted
endpoint before the simulator does -- another user's process on a shared machine
whose uid happens to be congruent modulo :data:`SYNC_PORT_RANGE_SIZE`, or
anything running as this user -- can therefore return a crafted pickle and
execute code in the policy process.  Closing that needs an authenticated
transport (CURVE, or a pre-shared secret in the handshake), which this release
does not ship.  The loopback pin below removes the *remote* reach of that
reader; it does not make the reader safe against a local peer.
"""

from __future__ import annotations

import ipaddress
import os
import socket
import time

SYNC_PORT_RANGE_START = 18500
"""First port of the range the per-user default is drawn from."""

SYNC_PORT_RANGE_SIZE = 500
"""Width of that range; the uid is reduced modulo this value."""

SYNC_LOOPBACK_HOST = "127.0.0.1"
"""The sync channel is a same-host rendezvous and is bound loopback-only."""

SYNC_HANDSHAKE_PREFIX = "SYNC"
"""Handshake message is ``SYNC:<rl_rate>`` or ``SYNC:<rl_rate>:<uid>``."""


def sync_peer_uid() -> int:
    """Return the current user id, or ``0`` on platforms without ``os.getuid``."""
    getuid = getattr(os, "getuid", None)
    if getuid is None:  # pragma: no cover - Windows only
        return 0
    return int(getuid())


def default_policy_sync_port(uid: int | None = None) -> int:
    """Return this user's default lock-step port."""
    if uid is None:
        uid = sync_peer_uid()
    return SYNC_PORT_RANGE_START + (int(uid) % SYNC_PORT_RANGE_SIZE)


def default_sim_step_sync_url(uid: int | None = None) -> str:
    """Return the ZMQ endpoint matching :func:`default_policy_sync_port`."""
    return f"tcp://{SYNC_LOOPBACK_HOST}:{default_policy_sync_port(uid)}"


DEFAULT_SIM_STEP_SYNC_URL = default_sim_step_sync_url()
"""Resolved once at import so the dataclass default and ``--help`` agree."""


SAME_HOST_TRANSPORTS = ("ipc://", "inproc://")
"""ZMQ transports that cannot leave this machine, whitelisted by name.

``ipc`` is a Unix domain socket and ``inproc`` is a pointer handoff inside one
process; neither can name a peer on another host.  Everything else ZMQ supports
either can (``tcp``, ``ws``, ``wss``, ``udp``, ``vmci``) or is multicast and
therefore explicitly off-host (``pgm``, ``epgm``).  This is a whitelist rather
than a "not tcp" test -- see :func:`sync_url_is_loopback`.
"""


def parse_tcp_endpoint(url: str) -> tuple[str, int] | None:
    """Split ``tcp://host:port`` into ``(host, port)``; return ``None`` otherwise.

    ``None`` means "this is not a well-formed TCP endpoint", and covers two
    different situations that callers must not conflate: a non-TCP transport
    (``ipc://``, ``inproc://``, ...) and a malformed string.  Callers decide what
    each means for them; :func:`sync_url_is_loopback` distinguishes them by
    checking the transport itself first.

    The port is validated as a real TCP port (1-65535).  Without that,
    ``tcp://127.0.0.1:-1`` and ``tcp://h:0`` parsed successfully.
    """
    if not isinstance(url, str) or not url.startswith("tcp://"):
        return None
    authority = url[len("tcp://") :]
    host, sep, port = authority.rpartition(":")
    if not sep or not host:
        return None
    if not port.isdigit():  # rejects "", "-1", "80/path", " 80"
        return None
    number = int(port)
    if not 1 <= number <= 65535:
        return None
    return host, number


def resolve_loopback_sync_url(url: str) -> str | None:
    """Return a URL pinned to the address this guard validated, or ``None`` to refuse.

    This is :func:`sync_url_is_loopback` plus **the address that was checked**.  ZMQ
    resolves a *name* a second time, and a DNS answer is free to differ between the two
    lookups (DNS rebinding), so validating a name and then handing the same name to ZMQ
    would leave the connect unpinned.

    Callers must connect to what comes back from here, never to ``url``:

    * ``ipc://`` / ``inproc://`` -- returned unchanged.  They name no host, so there
      is nothing to resolve and nothing to re-resolve.
    * ``tcp://host:port`` -- returned as ``tcp://<numeric literal>:<port>``, where the
      literal is the first address ``getaddrinfo`` gave for ``host`` and every address
      it gave was loopback.  IPv6 comes back bracketed, which is the spelling ZMQ
      wants.  ``tcp://localhost:18500`` therefore becomes ``tcp://127.0.0.1:18500``
      (or ``tcp://[::1]:18500``), and no further name lookup happens anywhere.
    * anything else -- ``None``.

    Names remain spellable -- ``tcp://localhost:...`` is accepted -- and no second
    lookup happens.

    See :func:`sync_url_is_loopback` for exactly what is accepted and rejected, and
    this module's docstring for the peer-authentication limit that remains.
    """
    if not isinstance(url, str):
        return None
    if url.startswith(SAME_HOST_TRANSPORTS):
        return url
    if not url.startswith("tcp://"):
        # Unknown or off-host transport -- including multicast pgm/epgm, ws/wss,
        # udp, vmci -- and anything that is not a ZMQ endpoint at all.
        return None
    endpoint = parse_tcp_endpoint(url)
    if endpoint is None:
        return None  # tcp:// but malformed: no port, non-numeric port, out of range
    host, port = endpoint
    # ZMQ writes IPv6 literals bracketed (``tcp://[::1]:1234``); getaddrinfo does not
    # accept the brackets.  Stripped here rather than in ``parse_tcp_endpoint`` so the
    # existing listener probe's behaviour is untouched.
    host = host[1:-1] if host.startswith("[") and host.endswith("]") else host
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except OSError:
        return None
    if not infos:
        return None
    pinned: str | None = None
    for info in infos:
        address = info[4][0]
        try:
            parsed = ipaddress.ip_address(address.split("%", 1)[0])
        except ValueError:
            return None
        if not parsed.is_loopback:
            return None
        if pinned is None:
            pinned = f"[{parsed}]" if parsed.version == 6 else str(parsed)
    if pinned is None:
        return None
    return f"tcp://{pinned}:{port}"


def sync_url_is_loopback(url: str) -> bool:
    """Return True only if ``url`` provably names an endpoint on this host.

    The lock-step channel is a same-host rendezvous by construction: the
    simulator binds ``tcp://127.0.0.1:<port>`` (``SYNC_LOOPBACK_HOST``, not a
    configurable field -- only the *port* is a flag), so no non-loopback address
    can ever reach a Holosoma simulator.  Callers use this to refuse a
    non-loopback ``--task.sim-step-sync-url`` rather than open a connection that
    could not work but *would* feed attacker-chosen bytes to the sync channel's
    ``pickle.loads``.

    **Anything this function cannot classify is rejected**, including multicast
    ``pgm://``/``epgm://``, ``ws://``/``wss://``, ``udp://``, ``vmci://``, a ``tcp://``
    URL with a missing or out-of-range port, and any string that is not a ZMQ endpoint.

    Accepted:

    * the transports in :data:`SAME_HOST_TRANSPORTS` (``ipc``, ``inproc``),
      which cannot address another host at all;
    * a well-formed ``tcp://host:port`` whose host resolves *entirely* to
      loopback addresses -- ``127.0.0.0/8``, ``::1`` (bracketed or not),
      IPv4-mapped loopback, and names such as ``localhost``.  Any port is fine,
      including an SSH-forwarded one: the forward terminates on this machine.

    Rejected: every other transport, a malformed URL of any kind, a name that
    does not resolve (unresolvable is not evidence of safety), and a name that
    resolves to both a loopback and a routable address (that is not a
    loopback-*only* endpoint).

    **This is a façade over :func:`resolve_loopback_sync_url` and answers a
    narrower question than a connecting caller needs.**  A boolean says "the name
    resolved to loopback once"; it does not say *to what*, so a caller that then
    connects to ``url`` re-resolves it and may reach somewhere else.  Anything
    that opens a socket must use :func:`resolve_loopback_sync_url` and connect to
    its return value.  This function remains for callers that only classify.
    """
    return resolve_loopback_sync_url(url) is not None


def sync_endpoint_has_listener(url: str, wait_s: float = 1.0, poll_interval_s: float = 0.1) -> bool:
    """Return True if something is accepting TCP connections at ``url``.

    ZMQ ``connect()`` is asynchronous and never fails, so a policy pointed at a
    dead endpoint would block for the whole receive timeout on its first step
    and only then raise.  A plain TCP connect answers "is a simulator there?"
    immediately, which lets the caller decide *before* entering the control loop
    -- either falling back to wall-clock stepping (no simulator on this machine,
    e.g. real-robot inference) or failing with a message naming the endpoint.

    ``wait_s`` gives a short grace period against scheduling jitter, kept small
    because it is also dead time on every run that has no simulator -- the
    documented procedure starts the simulator first, so the endpoint is normally
    bound well before the policy reaches this point.  Non-TCP transports cannot
    be probed and are reported as live.
    """
    endpoint = parse_tcp_endpoint(url)
    if endpoint is None:
        return True
    host, port = endpoint
    deadline = time.monotonic() + max(0.0, float(wait_s))
    while True:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            pass
        if time.monotonic() >= deadline:
            return False
        time.sleep(max(0.0, float(poll_interval_s)))
