import time

import numpy as np
import zmq
from loguru import logger

from .basic_vel_state_processor import BasicVelStateProcessor


class ZMQVelStateProcessor(BasicVelStateProcessor):
    """Velocity processor that subscribes to ZMQ pose messages."""

    def __init__(
        self,
        target_name: str,
        zmq_address: str = "tcp://127.0.0.1:6000",
        orientation_order: str = "xyzw",
        recv_timeout_ms: int = 1,
        record_mocap_history: bool = False,
        mocap_plot_hz: float = 0.0,
        plot_trim_head: int = 0,
        plot_trim_tail: int = 0,
        smooth_window: int = 0,
        max_linear_vel: float = 1.0,
        max_angular_vel: float = 2.0,
    ):
        super().__init__(
            target_name,
            orientation_order=orientation_order,
            record_mocap_history=record_mocap_history,
            mocap_plot_hz=mocap_plot_hz,
            plot_trim_head=plot_trim_head,
            plot_trim_tail=plot_trim_tail,
            smooth_window=smooth_window,
            max_linear_vel=max_linear_vel,
            max_angular_vel=max_angular_vel,
        )

        self.zmq_address = zmq_address
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(self.zmq_address)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.RCVTIMEO, recv_timeout_ms)

    def _process_message(self, message):
        """Parse incoming message and update velocities if it matches target."""
        if not isinstance(message, dict):
            return

        pose = message.get("pose", None)
        name = message.get("name", self.target_name)
        if self.target_name and name != self.target_name:
            return

        if pose is None:
            pose = message

        position = pose.get("position")
        orientation = pose.get("orientation")
        # Timestamp in seconds for exit-plot alignment. MuJoCo: sim_time from bridge. Real: send wall clock seconds in pose["timestamp"] so mocap_ts and cmd_ts share the same clock; if omitted, time.time() at receive is used for both.
        timestamp = pose.get("timestamp", time.time())

        if position is None or orientation is None:
            return

        try:
            self.update(position, orientation, timestamp)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Failed to update velocity from pose message: {e}")

    def poll(self) -> None:
        """Consume all pending ZMQ messages."""
        while True:
            try:
                message = self.socket.recv_pyobj(flags=zmq.NOBLOCK)
            except zmq.Again:  # noqa: PERF203
                break
            self._process_message(message)

    def flush(self) -> None:
        """Drain all pending ZMQ messages without recording.

        Called when resetting session buffers to discard stale messages
        (e.g. Phase 1 truth poses) before Phase 2 recording begins.
        """
        count = 0
        while True:
            try:
                self.socket.recv_pyobj(flags=zmq.NOBLOCK)
                count += 1
            except zmq.Again:
                break
        if count:
            logger.debug(f"ZMQ flush: discarded {count} stale messages")

    def get_vel_state(self):
        """Return latest velocity after polling for new samples."""
        self.poll()
        return super().get_vel_state()

    def get_pose_state(self):
        """Return latest pose after polling for new samples."""
        self.poll()
        return super().get_pose_state()