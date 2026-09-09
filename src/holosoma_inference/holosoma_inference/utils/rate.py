import threading
import time
import pickle

from holosoma_inference.utils.sync_rendezvous import resolve_loopback_sync_url, sync_peer_uid


class PreciseRateLimiter:
    """
    High-precision rate limiter using time.perf_counter() for nanosecond accuracy

    Features:
    - Uses time.perf_counter() for highest precision time measurement
    - Automatic drift compensation
    - Thread-safe
    - Supports dynamic frequency adjustment
    """

    def __init__(self, frequency: float, max_sleep_time: float = 0.1):
        """
        Initialize rate limiter

        Args:
            frequency: Target frequency (Hz)
            max_sleep_time: Maximum single sleep time (seconds), used to avoid long blocking
        """
        self.frequency = frequency
        self.period = 1.0 / frequency
        self.max_sleep_time = max_sleep_time

        # Time control variables
        self._next_time = None
        self._last_sleep_time = 0.0
        self._drift_compensation = 0.0
        self._lock = threading.Lock()

        # Statistics
        self._total_sleeps = 0
        self._total_drift = 0.0
        self._max_drift = 0.0

    def sleep(self) -> float:
        """
        Sleep until next cycle

        Returns:
            Actual sleep time (seconds)
        """
        with self._lock:
            current_time = time.perf_counter()

            # Initialize next target time
            if self._next_time is None:
                self._next_time = current_time + self.period

            # Calculate required sleep time
            sleep_time = self._next_time - current_time

            # If already timed out, record drift and return immediately
            if sleep_time <= 0:
                drift = -sleep_time
                self._total_drift += drift
                self._max_drift = max(self._max_drift, drift)
                self._next_time = current_time + self.period
                return 0.0

            # Segmented sleep for higher precision
            actual_sleep_time = 0.0
            while sleep_time > 0:
                # Calculate this sleep time
                this_sleep = min(sleep_time, self.max_sleep_time)

                # Sleep
                time.sleep(this_sleep)

                # Update actual sleep time
                actual_sleep_time += this_sleep
                sleep_time -= this_sleep

                # Check if interrupted
                if sleep_time > 0:
                    current_time = time.perf_counter()
                    sleep_time = self._next_time - current_time

            # Update next target time
            self._next_time += self.period

            # Record statistics
            self._total_sleeps += 1
            self._last_sleep_time = actual_sleep_time

            return actual_sleep_time

    def set_frequency(self, frequency: float):
        """
        Dynamically set frequency

        Args:
            frequency: New target frequency (Hz)
        """
        with self._lock:
            self.frequency = frequency
            self.period = 1.0 / frequency

    def get_stats(self) -> dict:
        """
        Get statistics

        Returns:
            Dictionary containing statistics
        """
        with self._lock:
            avg_drift = self._total_drift / max(self._total_sleeps, 1)
            return {
                "frequency": self.frequency,
                "period": self.period,
                "total_sleeps": self._total_sleeps,
                "total_drift": self._total_drift,
                "max_drift": self._max_drift,
                "avg_drift": avg_drift,
                "last_sleep_time": self._last_sleep_time,
            }

    def reset_stats(self):
        """Reset statistics"""
        with self._lock:
            self._total_sleeps = 0
            self._total_drift = 0.0
            self._max_drift = 0.0


# For backward compatibility, provide RateLimiter alias
RateLimiter = PreciseRateLimiter


class SimStepSyncRate:
    """Lock-step synchronization rate handler for MuJoCo sim-to-sim.

    Replaces wall-clock ``PreciseRateLimiter`` when running with a
    synchronized MuJoCo simulator.  Instead of sleeping a fixed wall-clock
    interval, each call to :meth:`sleep`:

    1. Sends ``b"STEP"`` to the simulator (telling it to run the next batch).
    2. Waits for ``b"DONE"`` from the simulator (batch complete).

    This guarantees exactly ``fps / rl_rate`` physics steps per policy step
    regardless of wall-clock execution speed.

    On the very first call the policy sends a handshake message
    ``b"SYNC:<rl_rate>"`` so the simulator can compute the batch size.

    ``sync_url`` must name a loopback endpoint.  ``_wait_done`` deserializes the
    simulator's replies with ``pickle.loads``, and the simulator binds this
    channel loopback-only (``SYNC_LOOPBACK_HOST`` in ``sync_rendezvous``, which
    is not a configurable field), so a non-loopback URL can never reach a real
    Holosoma simulator -- it can only point the pickle reader at some other
    host.  Refusing it here removes that combination without touching the wire
    protocol; SSH-forwarded setups are unaffected because a forwarded port is
    loopback on this side.

    **The connect uses the address the guard resolved, not the string the user
    typed.**  ``resolve_loopback_sync_url`` returns ``tcp://<numeric>:<port>``
    and :attr:`sync_url` records it; ZMQ never sees the hostname, so there is no
    second name lookup that could answer differently from the guard's (DNS
    rebinding).

    **What this does not do: authenticate the peer.**  The channel has no key,
    no signature and no challenge, so *any local process* that binds the accepted
    endpoint before the simulator can answer with a crafted pickle and execute
    code here.  The guard bounds the reader to this host; it does not make it
    safe against something already on this host.  See
    ``holosoma_inference/utils/sync_rendezvous.py``'s module docstring.
    """

    def __init__(
        self,
        sync_url: str,
        rl_rate: float,
        recv_timeout_ms: int = 60_000,
        low_state_callback=None,
        command_payload_callback=None,
    ):
        import zmq

        pinned_url = resolve_loopback_sync_url(sync_url)
        if pinned_url is None:
            raise ValueError(
                f"SimStepSyncRate: refusing a non-loopback sync endpoint {sync_url!r}. "
                "The lock-step channel is a same-host rendezvous -- the simulator binds it on "
                "127.0.0.1 only -- and its replies are deserialized with pickle, so a remote "
                "endpoint could not work and would only expose that reader. Point "
                "--task.sim-step-sync-url at 127.0.0.1 (forward the port over SSH if the "
                "simulator runs elsewhere), or pass --task.sim-step-sync-url=None to run "
                "without lock-step stepping."
            )

        # Connect to the resolved address, never to `sync_url`: passing the hostname on
        # would let ZMQ perform a second, unchecked DNS lookup. See the class docstring.
        self.sync_url = pinned_url
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.PAIR)
        self._sock.connect(pinned_url)
        self._rl_rate = rl_rate
        self._recv_timeout_ms = recv_timeout_ms
        self._low_state_callback = low_state_callback
        self._command_payload_callback = command_payload_callback
        self._first_step = True

    def _handshake(self) -> None:
        """Complete the initial SYNC handshake on the first sleep.

        The uid is part of the handshake so a simulator owned by a different
        user rejects this policy instead of lock-stepping with it -- see
        ``holosoma_inference/utils/sync_rendezvous.py``.
        """
        if not self._first_step:
            return
        self._first_step = False
        self._sock.send(f"SYNC:{self._rl_rate}:{sync_peer_uid()}".encode())
        self._wait_done()

    def sleep(self) -> float:
        """Send STEP to sim and block until sim replies DONE."""
        if self._first_step:
            self._handshake()
            return 0.0

        self._sock.send(self._step_message())
        self._wait_done()
        return 0.0

    def _step_message(self) -> bytes:
        """Build a STEP message, optionally carrying a deterministic command payload."""
        if self._command_payload_callback is None:
            return b"STEP"
        payload = self._command_payload_callback()
        if not payload:
            return b"STEP"
        if not isinstance(payload, dict):
            raise RuntimeError(f"SimStepSyncRate: command payload must be a dict, got {type(payload)!r}")
        msg = {"type": "STEP"}
        msg.update(payload)
        return pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL)

    def _wait_done(self) -> None:
        import zmq

        if self._sock.poll(timeout=self._recv_timeout_ms, flags=zmq.POLLIN):
            msg = self._sock.recv()
            if msg == b"DONE":
                return
            try:
                payload = pickle.loads(msg)
            except Exception as exc:
                raise RuntimeError(f"SimStepSyncRate: unexpected message from sim: {msg!r}") from exc
            if isinstance(payload, dict) and payload.get("type") == "REJECT":
                raise RuntimeError(
                    f"SimStepSyncRate: the simulator refused this policy: {payload.get('reason')}"
                )
            if not isinstance(payload, dict) or payload.get("type") != "DONE":
                raise RuntimeError(f"SimStepSyncRate: unexpected payload from sim: {payload!r}")
            if self._low_state_callback is not None and "low_state" in payload:
                self._low_state_callback(payload["low_state"], payload.get("tick"))
        else:
            raise RuntimeError(
                f"SimStepSyncRate: timed out waiting for sim DONE "
                f"after {self._recv_timeout_ms}ms (sim may have crashed)"
            )

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
        if self._ctx is not None:
            self._ctx.term()
        self._sock = None
        self._ctx = None
