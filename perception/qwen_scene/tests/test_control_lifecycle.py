"""Controller lifecycle checks with simulated buses; no hardware connections."""
import contextlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'SO101/code/scripts'))
import teleop_table_lerobot as controller


class Bus:
    def __init__(self, calibrated=True):
        self.is_connected = False
        self.is_calibrated = calibrated
        self.writes = []
        self.values = {}
        self.torque = False

    def connect(self): self.is_connected = True
    def disconnect(self, disable_torque=True):
        if disable_torque: self.torque = False
        self.is_connected = False
    def sync_read(self, *args, **kwargs): return dict.fromkeys(controller.JOINTS, 0.)
    def sync_write(self, *args, **kwargs): self.writes.append(args)
    def read(self, register, joint, **kwargs): return self.values.get((register, joint), 0)
    def write(self, register, joint, value, **kwargs):
        self.writes.append((register, joint, value))
        self.values[(register, joint)] = value
    def disable_torque(self, *args): self.torque = False
    @contextlib.contextmanager
    def torque_disabled(self, *args):
        self.torque = False
        yield
        self.torque = True


class Device:
    def __init__(self, calibrated=True):
        self.bus = Bus(calibrated)
        self.id = 'simulated'
        self.actions = []
    @property
    def is_connected(self): return self.bus.is_connected
    def connect(self):
        self.bus.connect()
        self.bus.torque = True
    def disconnect(self): self.bus.disconnect()
    def get_action(self): return dict.fromkeys((f'{j}.pos' for j in controller.JOINTS), 0.)
    def get_observation(self): return self.get_action()
    def send_action(self, value):
        self.actions.append(value)
        raise KeyboardInterrupt


class Lifecycle(unittest.TestCase):
    def run_controller(self, leader, follower, root):
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(controller, 'PROJECT_ROOT', root))
            stack.enter_context(patch.object(controller, 'SO101Leader', return_value=leader))
            stack.enter_context(patch.object(controller, 'SO101Follower', return_value=follower))
            stack.enter_context(patch.object(controller, 'SO101LeaderConfig', side_effect=lambda **kw: SimpleNamespace(**kw)))
            stack.enter_context(patch.object(controller, 'SO101FollowerConfig', side_effect=lambda **kw: SimpleNamespace(**kw)))
            stack.enter_context(patch.object(sys, 'argv', ['teleop', '--leader-port', 'fake-leader', '--follower-port', 'fake-follower', '--require-calibration', '--quiet', '--status-file', str(root/'status.json')]))
            controller.main()

    def test_calibration_mismatch_never_writes_or_enables(self):
        leader, follower = Device(False), Device()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                self.run_controller(leader, follower, Path(directory))
        self.assertEqual(leader.bus.writes, [])
        self.assertEqual(follower.bus.writes, [])
        self.assertFalse(follower.bus.torque)

    def test_interrupt_restores_gains_and_disables_torque(self):
        leader, follower = Device(), Device()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.run_controller(leader, follower, root)
            self.assertIn('stopped', (root/'status.json').read_text())
        self.assertEqual(len(follower.actions), 1)
        self.assertFalse(follower.bus.torque)
        self.assertFalse(follower.bus.is_connected)
        for joint in controller.P_BY_JOINT:
            self.assertEqual(follower.bus.values[('P_Coefficient', joint)], controller.BASELINE_P)


if __name__ == '__main__': unittest.main()
