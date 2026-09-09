import pickle
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


# In a source checkout, put both packages on the path so this test runs without an
# install. In an installed tree there is no `.git` anywhere above this file, so `next(...)`
# returns None and nothing is added -- both packages are already importable there.
ROOT = next((parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists()), None)
if ROOT is not None:
    sys.path.insert(0, str(ROOT / "src" / "holosoma"))
    sys.path.insert(0, str(ROOT / "src" / "holosoma_inference"))


class FakeMotorCmd:
    def __init__(self):
        self.tau = 0.0
        self.kp = 0.0
        self.kd = 0.0
        self.q = 0.0
        self.dq = 0.0
        self.weight = 0.0


class FakeLowCmd:
    def __init__(self):
        self.cmd_type = None
        self.motor_cmd = []


class FakePublisher:
    def InitChannel(self):
        pass

    def Write(self, msg):
        self.last_msg = msg


class FakeSubscriber:
    def __init__(self, handler=None):
        self.handler = handler

    def InitChannel(self):
        pass


class FakeClient:
    def Init(self):
        pass

    def ChangeMode(self, mode):
        self.mode = mode


class FakeChannelFactory:
    @classmethod
    def Instance(cls):
        return cls()

    def Init(self, *args, **kwargs):
        pass


fake_booster_sdk = types.ModuleType("booster_robotics_sdk")
fake_booster_sdk.B1LocoClient = FakeClient
fake_booster_sdk.B1LowCmdPublisher = FakePublisher
fake_booster_sdk.B1LowCmdSubscriber = FakeSubscriber
fake_booster_sdk.B1LowStatePublisher = FakePublisher
fake_booster_sdk.ChannelFactory = FakeChannelFactory
fake_booster_sdk.LowCmd = FakeLowCmd
fake_booster_sdk.LowCmdType = SimpleNamespace(SERIAL=1, PARALLEL=2)
fake_booster_sdk.LowState = type("FakeLowState", (), {})
fake_booster_sdk.MotorCmd = FakeMotorCmd
fake_booster_sdk.MotorState = type("FakeMotorState", (), {})
fake_booster_sdk.RobotMode = SimpleNamespace(kCustom=1)
sys.modules["booster_robotics_sdk"] = fake_booster_sdk
sys.modules.setdefault("pygame", types.SimpleNamespace(event=types.SimpleNamespace(get=lambda: None)))

fake_logger = SimpleNamespace(
    debug=lambda *args, **kwargs: None,
    error=lambda *args, **kwargs: None,
    info=lambda *args, **kwargs: None,
    warning=lambda *args, **kwargs: None,
)
fake_loguru = types.ModuleType("loguru")
fake_loguru.logger = fake_logger
sys.modules.setdefault("loguru", fake_loguru)

fake_zmq = types.ModuleType("zmq")
fake_zmq.Again = RuntimeError
fake_zmq.NOBLOCK = 1
fake_zmq.POLLIN = 1
fake_zmq.Context = SimpleNamespace(instance=lambda: SimpleNamespace(socket=lambda *_args, **_kwargs: None))
sys.modules.setdefault("zmq", fake_zmq)

fake_torch = types.ModuleType("torch")
fake_torch.nn = types.ModuleType("torch.nn")
fake_torch.optim = types.ModuleType("torch.optim")
fake_torch.amp = types.ModuleType("torch.amp")
fake_torch.amp.GradScaler = object
fake_torch.amp.autocast = object
fake_torch.utils = types.ModuleType("torch.utils")
fake_torch.utils.tensorboard = types.ModuleType("torch.utils.tensorboard")
fake_torch.utils.tensorboard.SummaryWriter = object
sys.modules.setdefault("torch", fake_torch)
sys.modules.setdefault("torch.nn", fake_torch.nn)
sys.modules.setdefault("torch.nn.functional", types.ModuleType("torch.nn.functional"))
sys.modules.setdefault("torch.optim", fake_torch.optim)
sys.modules.setdefault("torch.amp", fake_torch.amp)
sys.modules.setdefault("torch.utils", fake_torch.utils)
sys.modules.setdefault("torch.utils.tensorboard", fake_torch.utils.tensorboard)
fake_tensordict = types.ModuleType("tensordict")
fake_tensordict.TensorDict = object
sys.modules.setdefault("tensordict", fake_tensordict)
fake_rotations = types.ModuleType("holosoma.utils.rotations")
fake_rotations.get_euler_xyz = lambda *args, **kwargs: (None, None, None)
sys.modules.setdefault("holosoma.utils.rotations", fake_rotations)


class BoosterSyncDirectTests(unittest.TestCase):
    def test_rate_step_message_carries_command_payload(self):
        from holosoma_inference.utils.rate import SimStepSyncRate

        rate = SimStepSyncRate.__new__(SimStepSyncRate)
        rate._command_payload_callback = lambda: {
            "low_cmd_kind": "booster",
            "low_cmd_seq": 7,
            "low_cmd": [[1.0, 2.0, 3.0, 4.0, 5.0]],
        }

        payload = pickle.loads(rate._step_message())

        self.assertEqual(payload["type"], "STEP")
        self.assertEqual(payload["low_cmd_kind"], "booster")
        self.assertEqual(payload["low_cmd_seq"], 7)
        self.assertEqual(payload["low_cmd"], [[1.0, 2.0, 3.0, 4.0, 5.0]])

    def test_booster_command_sender_snapshots_stamped_low_cmd(self):
        from holosoma_inference.sdk.booster.command_sender.booster.booster_command_sender import (
            BoosterCommandSender,
        )

        cfg = SimpleNamespace(
            robot_type="t1_23dof",
            sdk_type="booster",
            motor_type="serial",
            num_motors=2,
            num_joints=2,
            motor2joint=(0, 1),
            motor_kp=(10.0, 20.0),
            motor_kd=(1.0, 2.0),
            default_motor_angles=(0.0, 0.0),
            dof_names=("j0", "j1"),
            dof_names_parallel_mech=(),
            weak_motor_joint_index=(),
        )
        sender = BoosterCommandSender(cfg)

        sender.send_command(
            np.array([0.3, -0.4]),
            np.array([0.1, -0.2]),
            np.array([1.5, -2.5]),
        )
        payload = sender.get_sync_low_cmd_payload()

        self.assertEqual(payload["low_cmd_kind"], "booster")
        self.assertEqual(payload["low_cmd_seq"], 2)
        self.assertEqual(payload["low_cmd"][0], [1.5, 10.0, 1.0, 0.3, 0.1])
        self.assertEqual(payload["low_cmd"][1], [-2.5, 20.0, 2.0, -0.4, -0.2])

    def test_direct_bridge_command_ignores_later_async_dds_overwrite(self):
        from holosoma.bridge.booster.booster_sdk2py_bridge import BoosterSdk2Bridge

        bridge = BoosterSdk2Bridge.__new__(BoosterSdk2Bridge)
        bridge.num_motor = 2
        bridge._cmd_snapshot = np.zeros((2, 5), dtype=np.float64)
        bridge._sdk2py_low_cmd_seq = 0
        bridge._use_sync_direct_low_cmd = False

        direct = [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]]
        bridge.set_sync_low_cmd_array(direct, seq=7)

        dds_cmd = FakeMotorCmd()
        dds_cmd.tau = 100.0
        dds_cmd.weight = 8.0
        bridge.low_cmd_handler(SimpleNamespace(motor_cmd=[dds_cmd]))

        np.testing.assert_array_equal(bridge._cmd_snapshot, np.asarray(direct, dtype=np.float64))
        self.assertEqual(bridge._sdk2py_low_cmd_seq, 7)

        bridge.reset_sync_low_cmd_array()
        np.testing.assert_array_equal(bridge._cmd_snapshot, np.zeros((2, 5), dtype=np.float64))
        self.assertEqual(bridge._sdk2py_low_cmd_seq, 0)
        self.assertTrue(bridge._use_sync_direct_low_cmd)


if __name__ == "__main__":
    unittest.main()
