import importlib.util
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts/body_hand_pipeline.py"


def load_launcher() -> ModuleType:
    spec = importlib.util.spec_from_file_location("g1_playground_test_body_hand", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeState:
    def __init__(self):
        self.dof_pos = np.zeros(29)
        self.dof_vel = np.zeros(29)
        self.base_quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.base_ang_vel = np.zeros(3)


class FakeHandState:
    def __init__(self, age=0.001):
        self.joint_pos = np.zeros(12)
        self.joint_vel = np.zeros(12)
        self.age = age

    @property
    def stale(self):
        return not np.isfinite(self.age) or self.age > 0.3


class FakeEnv:
    def __init__(self, odometry=True):
        self.state = FakeState()
        self.odometry = SimpleNamespace(position=np.zeros(3)) if odometry else None
        self.commands = []
        self.origins = []

    def read(self):
        return self.state

    def read_odometry(self):
        return self.odometry

    def step(self, target):
        self.commands.append(np.asarray(target))

    def set_born_place(self, quaternion, position):
        self.origins.append((np.asarray(quaternion), np.asarray(position)))


class FakeHandEnv:
    def __init__(self, age=0.001):
        self.state = FakeHandState(age)
        self.commands = []

    def read(self):
        return self.state

    def step(self, target):
        self.commands.append(np.asarray(target))


class TestInputBoundary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_launcher()

    def test_a_healthy_frame_passes(self):
        state, hand_state, odometry = self.module.read_frame(FakeEnv(), FakeHandEnv())
        self.assertEqual(state.dof_pos.shape, (29,))
        self.assertEqual(hand_state.joint_pos.shape, (12,))
        self.assertEqual(odometry.position.shape, (3,))

    def test_a_stale_hand_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "stale"):
            self.module.read_frame(FakeEnv(), FakeHandEnv(age=1.0))

    def test_missing_odometry_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Odometry"):
            self.module.read_frame(FakeEnv(odometry=False), FakeHandEnv())


class TestMotionPlayback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_launcher()

    def test_every_motion_frame_runs_once(self):
        env = FakeEnv()
        hand_env = FakeHandEnv()
        seen_frames = []

        class Policy:
            dt = 0.0
            motion = SimpleNamespace(num_frames=4)

            def get_observation(self, frame, *args):
                seen_frames.append(frame)
                return np.asarray([frame], dtype=np.float32)

            def act(self, observation):
                return np.full(29, observation[0]), np.full(12, observation[0])

        self.module.run_motion(env, hand_env, Policy(), log=None)
        self.assertEqual(seen_frames, [0, 1, 2, 3])
        self.assertEqual(len(env.commands), 4)
        self.assertEqual(len(hand_env.commands), 4)

    def test_capture_uses_a_valid_frame_and_aligns_once(self):
        env = FakeEnv()
        hand_env = FakeHandEnv()
        policy = SimpleNamespace(
            motion=SimpleNamespace(align_calls=0),
            reset_calls=0,
        )
        policy.motion.align = lambda: setattr(policy.motion, "align_calls", policy.motion.align_calls + 1)
        policy.reset = lambda: setattr(policy, "reset_calls", policy.reset_calls + 1)

        self.module.capture_origin(env, hand_env, policy)
        self.assertEqual(len(env.origins), 1)
        self.assertEqual(policy.motion.align_calls, 1)
        self.assertEqual(policy.reset_calls, 1)


if __name__ == "__main__":
    unittest.main()
