import importlib.util
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

LAUNCHER = Path(__file__).resolve().parents[1] / "scripts/loco_body_hand_pipeline.py"


def load_launcher():
    spec = importlib.util.spec_from_file_location("g1_playground_test_loco_hoi", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class State:
    def __init__(self):
        self.dof_pos = np.zeros(29)
        self.dof_vel = np.zeros(29)
        self.base_quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.base_ang_vel = np.zeros(3)


class HandState:
    joint_pos = np.zeros(12)
    joint_vel = np.zeros(12)
    age = 0.001
    stale = False


class Odometry:
    position = np.array([0.0, 0.0, 0.78], dtype=np.float32)
    raw_position = np.array([1.5, -0.25, 0.2], dtype=np.float32)
    velocity = np.zeros(3, dtype=np.float32)
    body_height = 0.78


class Environment:
    def __init__(self):
        self.commands = []
        self.rebases = 0
        self.born_place_align = False
        self.base_align = SimpleNamespace(
            base_pos=np.zeros(3, dtype=np.float32),
            base_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )

    def read(self):
        return State()

    def read_odometry(self):
        return Odometry()

    def set_born_place(self, quat, position):
        self.rebases += 1
        self.born_place_align = True
        self.base_align.base_pos = np.asarray(position).copy()
        return True

    def set_gains(self, stiffness, damping):
        pass

    def step(self, target):
        self.commands.append(np.asarray(target).copy())


class HandEnvironment:
    def __init__(self):
        self.commands = []

    def read(self):
        return HandState()

    def step(self, target):
        self.commands.append(np.asarray(target).copy())


class HoiPolicy:
    observation_dim = 9994
    action_dim = 41

    def __init__(self):
        self.motion = SimpleNamespace(
            num_frames=3,
            anchor_pos=np.tile([0.0, 0.0, 0.77], (3, 1)),
            anchor_quat=np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)),
            align=lambda: setattr(self, "aligned", True),
        )
        self.observation = SimpleNamespace(min_distance=0.25, max_distance=3.0)
        self.last_action = np.zeros(41)
        self.frames = []
        self.depth_inputs = []
        self.first_action_inputs = []
        self.aligned = False

    def reference_targets(self):
        return np.full(29, 0.4), np.full(12, 0.3)

    def get_observation(self, frame, anchor_pos, anchor_quat, state, hand_state, *, depth_m):
        self.frames.append(frame)
        self.depth_inputs.append(depth_m)
        return np.zeros(9994)

    def act(self, observation):
        self.first_action_inputs.append(self.last_action.copy())
        self.last_action[:] = 1.0
        return np.full(29, 0.4), np.full(12, 0.3)

    def reset(self):
        self.last_action.fill(0.0)


class TrackPolicy:
    reference_root_height = 0.76
    standing_target = np.full(29, 0.1)

    def __init__(self):
        self.observation = SimpleNamespace(
            anchor_pose=lambda root_pos, root_quat, joint_pos: (np.asarray(root_pos), np.asarray(root_quat))
        )
        self.last_action = np.zeros(29)
        self.action_inputs = []
        self.reset_calls = 0

    def set_reference(self, **reference):
        self.reference = reference

    def act(self, state, control):
        self.action_inputs.append(self.last_action.copy())
        self.last_action[:] = -0.7
        return np.full(29, -0.2)

    def reset(self):
        self.reset_calls += 1
        self.last_action.fill(0.0)


class LocomotionPolicy:
    def act(self, state, control):
        return np.full(29, -0.6)


class Camera:
    def __init__(self):
        self.sequence = 0

    def read(self):
        self.sequence += 1
        return SimpleNamespace(
            depth_m=np.ones((72, 128), dtype=np.float32),
            timestamp=time.monotonic(),
            sequence=self.sequence,
        )


class Controller:
    def __init__(self):
        self.reads = 0

    def read(self):
        self.reads += 1
        return {"axes": {}}, self.reads > 10


class TestLocoHoiPipeline(unittest.TestCase):
    def test_complete_locomotion_track_hoi_track_flow(self):
        module = load_launcher()
        run = SimpleNamespace(
            env=Environment(),
            hand_env=HandEnvironment(),
            controller=Controller(),
            loco=LocomotionPolicy(),
            hoi=HoiPolicy(),
            track=TrackPolicy(),
            camera=Camera(),
            depth_preview=None,
            depth_stale_seconds=0.15,
            dt=0.001,
            log=module.build_log(32, 12),
            started=0.0,
            origin=0.0,
            key_descriptor=-1,
            hoi_stiffness=np.full(29, 40.0),
            hoi_damping=np.full(29, 1.0),
        )
        polls = 0

        def keys(descriptor):
            nonlocal polls
            polls += 1
            if polls == 1:
                return {module.LOCO_TO_TRACK_KEY}, descriptor
            if polls == 4:
                return {module.TRACK_TO_HOI_KEY}, descriptor
            return set(), descriptor

        with patch.object(module, "poll_keys", side_effect=keys), patch.object(module.time, "sleep"):
            with self.assertRaises(module.Stop):
                module.run_pipeline(run, np.zeros(29), np.zeros(12), 2, 2)

        phases = [module.PHASES[index] for index in run.log.phase[: run.log.count]]
        self.assertEqual(
            phases,
            [
                "track",
                "track_to_hoi",
                "track_to_hoi",
                "track",
                "hoi",
                "hoi",
                "hoi",
                "track_to_default",
                "track_to_default",
                "track",
            ],
        )
        self.assertEqual(run.env.rebases, 1)
        self.assertTrue(run.hoi.aligned)
        self.assertEqual(run.hoi.frames, [0, 1, 2])
        self.assertEqual(run.camera.sequence, 3)
        self.assertTrue(all(depth is not None for depth in run.hoi.depth_inputs))
        np.testing.assert_array_equal(run.hoi.first_action_inputs[0], 0.0)
        self.assertEqual(run.track.reset_calls, 2)
        self.assertEqual(len(run.env.commands), len(run.hand_env.commands))


if __name__ == "__main__":
    unittest.main()
