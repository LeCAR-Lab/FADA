#!/usr/bin/env python3
"""
Publish Vicon mocap poses over ZMQ for inference logging/truth velocity.

Message format matches ZMQVelStateProcessor:
{"name": "<publish_name>", "pose": {"timestamp": t, "position": [x,y,z], "orientation": [x,y,z,w]}}

Implementation only -- no standalone CLI entry point is shipped for this module (see
tools/check_entrypoints.py). Instantiate `ViconZMQPublisher` directly.

`pyvicon_datastream` is an OPTIONAL dependency (`pip install holosoma-inference[mocap]`),
not a declared one: Vicon hardware is not part of the FADA pipeline and nothing else in
this package needs it. It is therefore imported inside `ViconZMQPublisher.__init__`
rather than at module scope, so this module imports in an install without the extra and
the failure happens on instantiation, with a message naming the extra.
"""

import time
from threading import Thread
from typing import List

import numpy as np
import zmq
from scipy.spatial.transform import Rotation as R


class ViconZMQPublisher:
    def __init__(
        self,
        vicon_object_names: List[str],
        publish_names: List[str],
        frequency: int = 200,
        vicon_tracker_ip: str = "127.0.0.1",
        zmq_pub_port: int = 6000,
    ):
        assert len(vicon_object_names) == len(publish_names), "object_names and publish_names length mismatch"

        self.vicon_tracker_ip = vicon_tracker_ip
        self.freq = frequency
        self.vicon_object_names = vicon_object_names
        self.publish_names = publish_names

        # Connect to Vicon DataStream. Imported here, not at module scope: see the module
        # docstring -- this is the optional `[mocap]` extra.
        try:
            from pyvicon_datastream import tools  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - needs the extra absent
            raise ImportError(
                "ViconZMQPublisher needs the optional Vicon DataStream client, which is not "
                "installed. Install it with `pip install holosoma-inference[mocap]` (or "
                "`pip install pyvicon-datastream`). It is optional because Vicon hardware is "
                "not part of the FADA pipeline."
            ) from exc

        self.tracker = tools.ObjectTracker(self.vicon_tracker_ip)
        if not self.tracker.is_connected:
            raise RuntimeError(f"Connection to {self.vicon_tracker_ip} failed")
        print(f"Connected to Vicon DataStream at {self.vicon_tracker_ip}")

        # Initialize ZMQ publisher
        self.ctx = zmq.Context.instance()
        self.socket = self.ctx.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{zmq_pub_port}")
        print(f"Publishing mocap data on tcp://*:{zmq_pub_port}")

        self.freq_counter = 0
        self.publish_rate = self.freq
        self.state_thread = Thread(target=self.state_publisher_thread, daemon=True)
        self.state_thread.start()

    def get_vicon_data(self, vicon_object_name):
        position = self.tracker.get_position(vicon_object_name)
        if not position:
            return None

        try:
            obj = position[2][0]
            _, _, x, y, z, roll, pitch, yaw = obj
            current_time = time.time()

            pos = np.array([x, y, z]) / 1000.0  # mm -> m
            quat_xyzw = R.from_euler("XYZ", [roll, pitch, yaw], degrees=False).as_quat()
            print(f"Vicon data for {vicon_object_name}: {pos}, {quat_xyzw}")
            return {
                "timestamp": current_time,
                "position": pos.tolist(),
                "orientation": quat_xyzw.tolist(),  # [x, y, z, w]
            }
        except Exception as e:  # noqa: BLE001
            print(f"Error retrieving Vicon data for {vicon_object_name}: {e}")
            return None

    def log_frequency(self):
        print(f"Vicon data publishing frequency: {self.freq_counter} Hz")
        self.freq_counter = 0

    def state_publisher_thread(self):
        print("Starting Vicon → ZMQ publisher thread")
        last_log_time = time.time()

        while True:
            try:
                for vicon_object_name, publish_name in zip(self.vicon_object_names, self.publish_names):
                    data = self.get_vicon_data(vicon_object_name)
                    print(data)
                    if data is None:
                        continue

                    message = {"name": publish_name, "pose": data}
                    self.socket.send_pyobj(message)

                self.freq_counter += 1

                now = time.time()
                if now - last_log_time >= 1.0:
                    self.log_frequency()
                    last_log_time = now

                time.sleep(1.0 / self.publish_rate)
            except Exception as e:  # noqa: BLE001
                print(f"Error in publisher loop: {str(e)}")
                time.sleep(0.1)

    def main_loop(self):
        print("Running main loop… Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Exiting ViconZMQPublisher…")
