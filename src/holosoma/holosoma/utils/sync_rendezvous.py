"""Default rendezvous for lock-step policy/simulator stepping.

Lock-step stepping needs the two processes -- the simulator (this package) and
the policy (``holosoma_inference``) -- to agree on one TCP endpoint without the
user having to pass a flag to either.  This module defines that endpoint.

**The same three constants and the same two functions are duplicated verbatim in
``holosoma_inference/utils/sync_rendezvous.py``.** The simulator and the policy
run in *separate* environments (see the four ``scripts/source_*_setup.sh``
variants), so neither package may import the other.
``src/holosoma/tests/utils/test_sync_rendezvous_defaults.py`` imports both copies
and fails if they ever disagree.

The default port is derived from the user id, so two users get the same port only
if their uids are congruent modulo :data:`SYNC_PORT_RANGE_SIZE`.  That residual
case is caught by the uid carried in the handshake (see
:data:`SYNC_HANDSHAKE_PREFIX` users in ``sim_utils.py`` and
``holosoma_inference/utils/rate.py``), which makes the simulator reject a policy
belonging to a different user.

Two rollouts by the *same* user collide on the default port; the simulator's bind
then fails with a message naming the two override flags.

The range ``[18500, 19000)`` sits above the commonly registered service ports and
below the Linux default ephemeral range (``net.ipv4.ip_local_port_range``,
normally starting at 32768).
"""

from __future__ import annotations

import os

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


DEFAULT_POLICY_SYNC_PORT = default_policy_sync_port()
"""Resolved once at import so the dataclass default and ``--help`` agree."""
