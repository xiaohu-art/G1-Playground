import dataclasses
import importlib
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "g1_playground.g1_env"


def wire_quaternion(yaw: float):
    return [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]


class FakeControlEndpoint:
    def __init__(self, cfg):
        self.cfg = cfg
        self.sport = SimpleNamespace(position=[1.5, -0.25, 0.2], velocity=[0.1, 0.0, -0.02], body_height=0.75033)
        self.failure = None
        self.robot_state = SimpleNamespace(
            motor_state=SimpleNamespace(q=[0.0] * 29, dq=[0.0] * 29),
            imu_state=SimpleNamespace(quaternion=wire_quaternion(0.0), gyroscope=[0.0] * 3),
            wireless_remote=bytes(40),
        )

    def get_robot_state(self):
        return self.robot_state

    def get_sport_state(self):
        if self.failure is not None:
            raise self.failure
        return self.sport

    def shutdown(self):
        pass


class TestOdometryBinding(unittest.TestCase):
    def test_native_sport_state_exposes_body_height(self):
        try:
            import unitree_cpp
        except ImportError:
            self.skipTest("unitree_cpp binding is not built")
        state = unitree_cpp.SportState()
        self.assertTrue(hasattr(state, "body_height"))
        self.assertIsInstance(state.body_height, float)
        self.assertEqual(state.body_height, 0.0)
        self.assertEqual(len(state.position), 3)
        self.assertEqual(len(state.velocity), 3)


class TestReadOdometry(unittest.TestCase):
    def setUp(self):
        self.missing = object()
        self.old_binding = sys.modules.get("unitree_cpp", self.missing)
        self.old_module = sys.modules.pop(MODULE_NAME, None)
        binding = types.ModuleType("unitree_cpp")
        binding.RobotState = SimpleNamespace
        binding.G1DdsControlEndpoint = FakeControlEndpoint
        sys.modules["unitree_cpp"] = binding
        self.module = importlib.import_module(MODULE_NAME)
        from g1_playground.utils.math import TransformAlignment

        self.env = self.module.G1Env.__new__(self.module.G1Env)
        self.endpoint = FakeControlEndpoint({})
        self.env.control_endpoint = self.endpoint
        self.env.remote_controller_handler = None
        self.env.born_place_align = False
        self.env.base_align = TransformAlignment(yaw_only=True, xy_only=True)

    def tearDown(self):
        sys.modules.pop(MODULE_NAME, None)
        if self.old_module is not None:
            sys.modules[MODULE_NAME] = self.old_module
        if self.old_binding is self.missing:
            sys.modules.pop("unitree_cpp", None)
        else:
            sys.modules["unitree_cpp"] = self.old_binding

    def test_returns_frozen_snapshot_with_scalar_height(self):
        odometry = self.env.read_odometry()
        self.assertIsInstance(odometry, self.module.G1Odometry)
        np.testing.assert_allclose(odometry.position, [1.5, -0.25, 0.75033], rtol=1e-6)
        np.testing.assert_allclose(odometry.raw_position, [1.5, -0.25, 0.2])
        np.testing.assert_allclose(odometry.velocity, [0.1, 0.0, -0.02])
        self.assertIsInstance(odometry.body_height, float)
        self.assertAlmostEqual(odometry.body_height, 0.75033, places=5)
        for array in (odometry.position, odometry.velocity):
            self.assertFalse(array.flags.writeable)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            odometry.position = np.zeros(3)

    def test_unavailable_and_stale_both_degrade_to_none(self):
        for message in ("Sport state data is not available", "Sport state data is stale"):
            with self.subTest(message=message):
                self.endpoint.failure = RuntimeError(message)
                self.assertIsNone(self.env.read_odometry())
        self.endpoint.failure = None
        self.assertIsNotNone(self.env.read_odometry())


class TestRecorder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recorder = importlib.import_module("g1_playground.utils.recorder")

    def state(self):
        return SimpleNamespace(
            dof_pos=np.arange(29, dtype=np.float32),
            dof_vel=np.zeros(29, dtype=np.float32),
            base_quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            base_ang_vel=np.zeros(3, dtype=np.float32),
        )

    def test_records_nan_when_odometry_is_absent(self):
        log = self.recorder.recorder(4)
        self.recorder.record(log, 0.0, self.state(), np.zeros(29), None)
        self.assertEqual(log.count, 1)
        self.assertTrue(np.isnan(log.base_pos[0]).all())
        self.assertTrue(np.isnan(log.base_lin_vel[0]).all())
        self.assertTrue(np.isnan(log.body_height[0]))

    def test_records_position_velocity_and_height(self):
        odometry = SimpleNamespace(
            position=np.array([1.0, 2.0, 0.3]),
            raw_position=np.array([9.0, 8.0, 0.5]),
            velocity=np.array([0.4, 0.0, 0.0]),
            body_height=0.75033,
        )
        log = self.recorder.recorder(4)
        self.recorder.record(log, 0.02, self.state(), np.ones(29), odometry)
        np.testing.assert_allclose(log.base_pos[0], [1.0, 2.0, 0.3])
        np.testing.assert_allclose(log.sport_position[0], [9.0, 8.0, 0.5])
        np.testing.assert_allclose(log.base_lin_vel[0], [0.4, 0.0, 0.0])
        self.assertAlmostEqual(float(log.body_height[0]), 0.75033, places=5)

    def test_capacity_is_never_exceeded(self):
        log = self.recorder.recorder(2)
        for index in range(5):
            self.recorder.record(log, 0.02 * index, self.state(), np.zeros(29), None)
        self.assertEqual(log.count, 2)


class TestRecordingIsHydraControlled(unittest.TestCase):
    def test_both_roots_default_to_disabled(self):
        from omegaconf import OmegaConf

        for name in ("run_pipeline", "run_loco_track"):
            with self.subTest(root=name):
                root = OmegaConf.load(REPO_ROOT / f"configs/{name}.yaml")
                self.assertIn("recording", root)
                self.assertEqual(set(root.recording), {"enabled", "seconds", "directory"})
                self.assertIs(root.recording.enabled, False)
                self.assertGreater(root.recording.seconds, 0)


class TestBornPlaceRebase(unittest.TestCase):
    def setUp(self):
        from g1_playground.utils.math import TransformAlignment

        self.missing = object()
        self.old_binding = sys.modules.get("unitree_cpp", self.missing)
        self.old_module = sys.modules.pop(MODULE_NAME, None)
        binding = types.ModuleType("unitree_cpp")
        binding.RobotState = SimpleNamespace
        binding.G1DdsControlEndpoint = FakeControlEndpoint
        sys.modules["unitree_cpp"] = binding
        self.module = importlib.import_module(MODULE_NAME)
        self.env = self.module.G1Env.__new__(self.module.G1Env)
        self.endpoint = FakeControlEndpoint({})
        self.env.control_endpoint = self.endpoint
        self.env.remote_controller_handler = None
        self.env.born_place_align = False
        self.env.base_align = TransformAlignment(yaw_only=True, xy_only=True)

    def tearDown(self):
        sys.modules.pop(MODULE_NAME, None)
        if self.old_module is not None:
            sys.modules[MODULE_NAME] = self.old_module
        if self.old_binding is self.missing:
            sys.modules.pop("unitree_cpp", None)
        else:
            sys.modules["unitree_cpp"] = self.old_binding

    def capture(self, yaw: float) -> None:
        self.endpoint.robot_state.imu_state.quaternion = wire_quaternion(yaw)
        state = self.env.read()
        odometry = self.env.read_odometry()
        self.assertTrue(self.env.set_born_place(state.base_quat, odometry.position))

    def test_capture_succeeds_once_and_is_refused_afterwards(self):
        self.capture(0.0)
        self.assertTrue(self.env.born_place_align)
        self.assertFalse(self.env.set_born_place(self.env.read().base_quat, self.env.read_odometry().position))

    def test_origin_maps_to_zero_and_keeps_body_height(self):
        self.capture(0.0)
        odometry = self.env.read_odometry()
        np.testing.assert_allclose(odometry.position[:2], [0.0, 0.0], atol=1e-6)
        self.assertAlmostEqual(float(odometry.position[2]), 0.75033, places=5)
        np.testing.assert_allclose(odometry.raw_position, [1.5, -0.25, 0.2])

    def test_displacement_is_expressed_in_the_captured_heading(self):
        self.capture(np.pi / 2.0)
        self.endpoint.sport.position = [2.5, -0.25, 0.2]
        odometry = self.env.read_odometry()
        np.testing.assert_allclose(odometry.position[:2], [0.0, -1.0], atol=1e-5)
        self.assertAlmostEqual(float(odometry.position[2]), 0.75033, places=5)

    def test_captured_heading_reads_back_as_zero_yaw(self):
        self.capture(np.pi / 4.0)
        quat = self.env.read().base_quat
        yaw = np.arctan2(2.0 * (quat[3] * quat[2] + quat[0] * quat[1]), 1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2))
        self.assertAlmostEqual(float(yaw), 0.0, places=5)

    def test_tilt_is_untouched_by_the_rebase(self):
        from g1_playground.utils.math import get_gravity_orientation

        pitch = 0.2
        tilted = [np.cos(pitch / 2.0), 0.0, np.sin(pitch / 2.0), 0.0]
        self.endpoint.robot_state.imu_state.quaternion = tilted
        before = get_gravity_orientation(self.env.read().base_quat)

        self.capture(np.pi / 3.0)
        self.endpoint.robot_state.imu_state.quaternion = tilted
        after = get_gravity_orientation(self.env.read().base_quat)

        np.testing.assert_allclose(after, before, atol=1e-6)
        self.assertAlmostEqual(float(np.arccos(np.clip(-after[2], -1.0, 1.0))), pitch, places=4)

    def test_recorder_stores_the_frozen_origin(self):
        recorder = importlib.import_module("g1_playground.utils.recorder")
        log = recorder.recorder(4)
        recorder.record(log, 0.0, self.env.read(), np.zeros(29), self.env.read_odometry(), self.env)
        self.assertFalse(bool(log.rebase_active[0]))
        self.assertTrue(np.isnan(log.rebase_origin_pos[0]).all())

        self.capture(np.pi / 2.0)
        recorder.record(log, 0.02, self.env.read(), np.zeros(29), self.env.read_odometry(), self.env)
        self.assertTrue(bool(log.rebase_active[1]))
        np.testing.assert_allclose(log.rebase_origin_pos[1], [1.5, -0.25, 0.0], atol=1e-6)
        expected = [0.0, 0.0, np.sin(np.pi / 4.0), np.cos(np.pi / 4.0)]
        np.testing.assert_allclose(log.rebase_origin_quat[1], expected, atol=1e-6)
