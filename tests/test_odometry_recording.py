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


def load_pipeline():
    spec = importlib.util.spec_from_file_location("g1_pipeline_recorder", REPO_ROOT / "scripts/pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeControlEndpoint:
    def __init__(self, cfg):
        self.cfg = cfg
        self.sport = SimpleNamespace(position=[1.5, -0.25, 0.2], velocity=[0.1, 0.0, -0.02], body_height=0.75033)
        self.failure = None

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
        self.env = self.module.G1Env.__new__(self.module.G1Env)
        self.endpoint = FakeControlEndpoint({})
        self.env.control_endpoint = self.endpoint

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
        np.testing.assert_allclose(odometry.position, [1.5, -0.25, 0.2])
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


class TestPipelineRecorder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = load_pipeline()

    def state(self):
        return SimpleNamespace(
            dof_pos=np.arange(29, dtype=np.float32),
            dof_vel=np.zeros(29, dtype=np.float32),
            base_quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            base_ang_vel=np.zeros(3, dtype=np.float32),
        )

    def test_records_nan_when_odometry_is_absent(self):
        log = self.pipeline.recorder(4)
        self.pipeline.record(log, 0.0, self.state(), np.zeros(29), None)
        self.assertEqual(log.count, 1)
        self.assertTrue(np.isnan(log.base_pos[0]).all())
        self.assertTrue(np.isnan(log.base_lin_vel[0]).all())
        self.assertTrue(np.isnan(log.body_height[0]))

    def test_records_position_velocity_and_height(self):
        odometry = SimpleNamespace(
            position=np.array([1.0, 2.0, 0.3]), velocity=np.array([0.4, 0.0, 0.0]), body_height=0.75033
        )
        log = self.pipeline.recorder(4)
        self.pipeline.record(log, 0.02, self.state(), np.ones(29), odometry)
        np.testing.assert_allclose(log.base_pos[0], [1.0, 2.0, 0.3])
        np.testing.assert_allclose(log.base_lin_vel[0], [0.4, 0.0, 0.0])
        self.assertAlmostEqual(float(log.body_height[0]), 0.75033, places=5)

    def test_capacity_is_never_exceeded(self):
        log = self.pipeline.recorder(2)
        for index in range(5):
            self.pipeline.record(log, 0.02 * index, self.state(), np.zeros(29), None)
        self.assertEqual(log.count, 2)


if __name__ == "__main__":
    unittest.main()
