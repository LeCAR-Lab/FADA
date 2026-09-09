from booster_robotics_sdk import (
    B1LocoClient,
    B1LowCmdPublisher,
    LowCmd,
    LowCmdType,
    MotorCmd,
    RobotMode,
)

from holosoma_inference.config.config_types.robot import RobotConfig

from ..base import BasicCommandSender  # noqa: TID252


class BoosterCommandSender(BasicCommandSender):
    """Booster command sender implementation."""

    def _init_sdk_components(self):
        """Initialize Booster SDK-specific components."""

        robot_type = self.config.robot_type

        if robot_type in ["t1_23dof", "t1_29dof"]:
            self.LowCmd = LowCmd
            self.LowCmdType = LowCmdType
            self.MotorCmd = MotorCmd
            self.lowcmd_publisher_ = B1LowCmdPublisher()
            self.client = B1LocoClient()
            self.lowcmd_publisher_.InitChannel()
            self.client.Init()
            self._sdk2py_low_cmd_sequence = 0
            self._last_sync_low_cmd_payload = None
            self.init_booster_low_cmd()
            self.create_prepare_cmd(self.low_cmd, self.config)
            self._send_cmd(self.low_cmd)
            self.client.ChangeMode(RobotMode.kCustom)
            self.dof_names = self.config.dof_names
            self.dof_names_parallel_mech = self.config.dof_names_parallel_mech
            self.parallel_mech_indexes = [self.dof_names.index(name) for name in self.dof_names_parallel_mech]
        else:
            raise NotImplementedError(f"Robot type {robot_type} is not supported yet")

    def init_booster_low_cmd(self):
        """Initialize Booster low-level command."""
        self.low_cmd = self.LowCmd()
        if self.motor_type == "serial":
            self.low_cmd.cmd_type = self.LowCmdType.SERIAL
        elif self.motor_type == "parallel":
            self.low_cmd.cmd_type = self.LowCmdType.PARALLEL
        self.motor_cmds = [self.MotorCmd() for _ in range(self.config.num_motors)]
        self.low_cmd.motor_cmd = self.motor_cmds

    def _stamp_low_cmd_sequence(self, low_cmd):
        """Stamp a small sequence number into Booster lowcmd for sim-side sync ack.

        Booster LowCmd has no reserve/user-data field like Unitree. In custom
        mode, MotorCmd.weight is not used by the current controller or sim
        bridge torque path, so motor 0 carries the same 1..255 sequence that
        Unitree stores in reserve[0].
        """
        motor_cmd = getattr(low_cmd, "motor_cmd", None)
        if motor_cmd is None or len(motor_cmd) == 0:
            return
        self._sdk2py_low_cmd_sequence = (int(self._sdk2py_low_cmd_sequence) % 255) + 1
        motor_cmd[0].weight = float(self._sdk2py_low_cmd_sequence)

    def _snapshot_low_cmd_payload(self, low_cmd):
        """Capture the command scalars needed by lock-step sim before SDK handoff."""
        motor_cmd = getattr(low_cmd, "motor_cmd", None)
        if motor_cmd is None:
            self._last_sync_low_cmd_payload = None
            return
        self._last_sync_low_cmd_payload = {
            "low_cmd_kind": "booster",
            "low_cmd_seq": int(self._sdk2py_low_cmd_sequence),
            "low_cmd": [
                [
                    float(m.tau),
                    float(m.kp),
                    float(m.kd),
                    float(m.q),
                    float(m.dq),
                ]
                for m in motor_cmd
            ],
        }

    def get_sync_low_cmd_payload(self):
        """Return the most recent lowcmd snapshot for ZMQ sync stepping."""
        return self._last_sync_low_cmd_payload

    def send_command(self, cmd_q, cmd_dq, cmd_tau, dof_pos_latest=None, kp_override=None, kd_override=None):
        """Send command to Booster robot."""
        # In booster, we need to fill the motor_cmds first
        self.low_cmd = self.LowCmd()
        if self.motor_type == "serial":
            self.low_cmd.cmd_type = self.LowCmdType.SERIAL
        elif self.motor_type == "parallel":
            self.low_cmd.cmd_type = self.LowCmdType.PARALLEL
        else:
            raise NotImplementedError(f"Motor type {self.motor_type} is not supported yet")
        self.low_cmd.motor_cmd = self.motor_cmds

        motor_cmd = self.low_cmd.motor_cmd
        self._fill_motor_commands(
            motor_cmd,
            cmd_q,
            cmd_dq,
            cmd_tau,
            kp_override=kp_override,
            kd_override=kd_override,
        )

        # Send command
        self._stamp_low_cmd_sequence(self.low_cmd)
        self._snapshot_low_cmd_payload(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)

    def _send_cmd(self, cmd):
        """Send command to robot."""
        self._stamp_low_cmd_sequence(cmd)
        self._snapshot_low_cmd_payload(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def init_cmd_t1(self, low_cmd):
        """Initialize T1 command."""
        low_cmd.cmd_type = self.LowCmdType.SERIAL
        motorCmds = [self.MotorCmd() for _ in range(self.config.num_motors)]
        low_cmd.motor_cmd = motorCmds

        num_motors = min(len(motorCmds), self.config.num_motors)
        for i in range(num_motors):
            low_cmd.motor_cmd[i].q = 0.0
            low_cmd.motor_cmd[i].dq = 0.0
            low_cmd.motor_cmd[i].tau = 0.0
            low_cmd.motor_cmd[i].kp = 0.0
            low_cmd.motor_cmd[i].kd = 0.0
            # weight is not effective in custom mode
            low_cmd.motor_cmd[i].weight = 0.0

    def create_prepare_cmd(self, low_cmd, cfg: RobotConfig):
        """Create prepare command for T1."""
        self.init_cmd_t1(low_cmd)
        # Use motor_kp, motor_kd, and default_motor_angles from RobotConfig
        # Note: motor_kp and motor_kd may be None during initialization (loaded from ONNX later)
        num_motors = min(len(low_cmd.motor_cmd), cfg.num_motors)
        for i in range(num_motors):
            low_cmd.motor_cmd[i].kp = cfg.motor_kp[i] if cfg.motor_kp is not None else 0.0
            low_cmd.motor_cmd[i].kd = cfg.motor_kd[i] if cfg.motor_kd is not None else 0.0
            low_cmd.motor_cmd[i].q = cfg.default_motor_angles[i]
        return low_cmd
