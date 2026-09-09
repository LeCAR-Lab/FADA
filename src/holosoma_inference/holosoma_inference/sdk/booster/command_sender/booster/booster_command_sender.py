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
        self.lowcmd_publisher_.Write(self.low_cmd)

    def _send_cmd(self, cmd):
        """Send command to robot."""
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
