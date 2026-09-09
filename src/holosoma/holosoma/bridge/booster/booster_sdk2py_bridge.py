import numpy as np
from booster_robotics_sdk import (  # type: ignore[import-not-found]
    B1LowCmdSubscriber,
    B1LowStatePublisher,
    LowCmd,
    LowCmdType,
    LowState,
    MotorCmd,
    MotorState,
)
from loguru import logger

from holosoma.bridge.base import BasicSdk2Bridge
from holosoma.utils.rotations import get_euler_xyz
from holosoma.utils.safe_torch_import import torch


class BoosterSdk2Bridge(BasicSdk2Bridge):
    """Booster SDK2Py bridge implementation."""

    SUPPORTED_ROBOT_TYPES = {"t1_23dof", "t1_29dof"}

    def _init_sdk_components(self):
        """Initialize Booster SDK-specific components."""

        from booster_robotics_sdk import ChannelFactory

        # Use bridge config for domain_id and interface
        domain_id = self.bridge_config.domain_id

        # Note: holosoma_inference/booster is not using interface
        ChannelFactory.Instance().Init(domain_id)

        logger.info(f"Booster SDK factory initialized with domain_id={domain_id}")

        robot_type = self.robot.asset.robot_type
        if robot_type in self.SUPPORTED_ROBOT_TYPES:
            self.LowCmd = LowCmd
            self.LowState = LowState
            self.LowCmdType = LowCmdType
            self.MotorCmd = MotorCmd
            self.low_cmd = self.LowCmd()
            if self.motor_type == "serial":
                self.low_cmd.cmd_type = self.LowCmdType.SERIAL
            elif self.motor_type == "parallel":
                self.low_cmd.cmd_type = self.LowCmdType.PARALLEL
            self.motor_cmds = [MotorCmd() for _ in range(self.num_motor)]
            self.low_cmd.motor_cmd = self.motor_cmds
            self._sdk2py_low_cmd_seq = 0
        else:
            # Raise an error if robot_type is not valid
            raise ValueError(f"Invalid robot type '{robot_type}'. Booster SDK supports: {self.SUPPORTED_ROBOT_TYPES}")

        # Booster sdk message
        self.low_state = LowState()
        self.low_state.motor_state_serial = [MotorState() for _ in range(self.num_motor)]
        self.low_state.motor_state_parallel = [MotorState() for _ in range(self.num_motor)]

        # Stable numpy-array snapshot of the most recently received command.
        # Booster's DDS layer may reuse the underlying ``msg.motor_cmd`` buffer for
        # subsequent messages — keeping a reference to those C++ objects in
        # ``self.low_cmd.motor_cmd`` and reading them later from ``compute_torques``
        # would race with the next handler invocation. We snapshot scalars
        # ``[tau, kp, kd, q, dq]`` per motor immediately inside the handler, then
        # atomically replace the array reference. ``compute_torques`` reads only
        # from this snapshot, so torques applied by the simulator are deterministic
        # in lock-step (a new ``_sdk2py_low_cmd_seq`` is published only after the
        # snapshot is in place, mirroring the seq-last ordering Unitree relies on).
        self._cmd_snapshot = np.zeros((self.num_motor, 5), dtype=np.float64)
        self._use_sync_direct_low_cmd = False

        # Initialize Booster SDK components (factory should be initialized by SimulatorBridge)
        self.low_state_puber = B1LowStatePublisher()
        self.low_cmd_suber = B1LowCmdSubscriber(self.low_cmd_handler)
        self.low_state_puber.InitChannel()
        self.low_cmd_suber.InitChannel()
        logger.info("Booster SDK components initialized successfully")
        # TODO: wireless controller for booster

    def low_cmd_handler(self, msg=None):
        """Handle Booster low-level command messages."""
        if not msg:
            return
        if getattr(self, "_use_sync_direct_low_cmd", False):
            return
        try:
            motor_cmd = getattr(msg, "motor_cmd", None)
            if motor_cmd is None or len(motor_cmd) == 0:
                return
            # Build snapshot first (read all DDS values out of the SDK's buffer);
            # only then publish the new sequence so ack-wait observers see fresh data.
            n = min(self.num_motor, len(motor_cmd))
            snap = np.zeros((self.num_motor, 5), dtype=np.float64)
            for i in range(n):
                m = motor_cmd[i]
                snap[i, 0] = float(m.tau)
                snap[i, 1] = float(m.kp)
                snap[i, 2] = float(m.kd)
                snap[i, 3] = float(m.q)
                snap[i, 4] = float(m.dq)
            seq = int(float(getattr(motor_cmd[0], "weight", 0.0)))
            # Atomic reference swap (single Python attribute set), then publish seq.
            self._cmd_snapshot = snap
            self._sdk2py_low_cmd_seq = seq
            # Keep ``self.low_cmd`` populated for any consumer that introspects it,
            # but compute_torques no longer relies on it.
            new_cmd = self.LowCmd()
            new_cmd.cmd_type = self.LowCmdType.SERIAL if self.motor_type == "serial" else self.LowCmdType.PARALLEL
            new_cmd.motor_cmd = self.motor_cmds  # pre-allocated stable list
            self.low_cmd = new_cmd
        except Exception as e:
            logger.error(f"Error in low_cmd_handler: {e}")

    def set_sync_low_cmd_array(self, low_cmd_array, seq: int | None = None) -> None:
        """Install a lowcmd snapshot delivered over the lock-step ZMQ channel.

        In sync sim mode this bypasses Booster DDS for the command path. The
        policy still publishes DDS lowcmd for compatibility, but physics consumes
        this in-band STEP payload so command delivery is tied to the exact sim
        batch that requested it.
        """
        arr = np.asarray(low_cmd_array, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 5:
            raise ValueError(f"Expected low_cmd array with shape (N, 5), got {arr.shape}")
        snap = np.zeros((self.num_motor, 5), dtype=np.float64)
        n = min(self.num_motor, arr.shape[0])
        snap[:n, :] = arr[:n, :]
        self._cmd_snapshot = snap
        if seq is not None:
            self._sdk2py_low_cmd_seq = int(seq)
        self._use_sync_direct_low_cmd = True

    def reset_sync_low_cmd_array(self) -> None:
        """Enter direct-sync command mode with a deterministic zero command."""
        self._cmd_snapshot = np.zeros((self.num_motor, 5), dtype=np.float64)
        self._sdk2py_low_cmd_seq = 0
        self._use_sync_direct_low_cmd = True

    def publish_low_state(self):
        """Publish Booster low-level state using simulator-agnostic interface."""
        if self.low_state_puber is None:
            return

        num_motors = self.num_motor
        imu = self.low_state.imu_state

        if self.motor_type == "serial":
            motor_state = self.low_state.motor_state_serial
        elif self.motor_type == "parallel":
            motor_state = self.low_state.motor_state_parallel
        else:
            raise ValueError(f"Invalid motor type '{self.motor_type}'. Expected 'serial' or 'parallel'.")

        positions, velocities, accelerations = self._get_dof_states()
        actuator_forces = self._get_actuator_forces()
        for i in range(num_motors):
            m = motor_state[i]
            m.q = positions[i]
            m.dq = velocities[i]
            m.ddq = accelerations[i]
            m.tau_est = actuator_forces[i]

        quaternion, gyro, acceleration = self._get_base_imu_data()
        roll, pitch, yaw = get_euler_xyz(quaternion.unsqueeze(0), w_last=False)  # w_last=False for [w,x,y,z]
        rpy = torch.stack([roll, pitch, yaw], dim=-1).squeeze(0).detach().cpu().numpy()
        imu.rpy = rpy
        imu.gyro = gyro.detach().cpu().numpy()
        imu.acc = acceleration.detach().cpu().numpy()

        if self._truth_pose_puber is not None:
            pos, quat_wxyz = self._get_base_pose_data()
            self.publish_truth_pose(self.sim_time, pos, quat_wxyz)

        self.low_state_puber.Write(self.low_state)

    def compute_torques(self):
        """Compute torques from the latest command snapshot.

        Reads ``self._cmd_snapshot`` (an immutable numpy array set atomically by
        ``low_cmd_handler``) instead of dereferencing C++ MotorCmd objects whose
        underlying DDS buffers may be overwritten by subsequent messages.
        """
        snap = getattr(self, "_cmd_snapshot", None)
        if snap is None:
            return self.torques
        try:
            # Local copy stabilises against concurrent reference replacement on
            # ``self._cmd_snapshot`` from the SDK callback thread.
            snap_local = snap
            tau_ff = snap_local[:, 0]
            kp = snap_local[:, 1]
            kd = snap_local[:, 2]
            q_target = snap_local[:, 3]
            dq_target = snap_local[:, 4]
            return self._compute_pd_torques(tau_ff, kp, kd, q_target, dq_target)
        except Exception as e:
            logger.error(f"Error computing torques: {e}")
            raise
