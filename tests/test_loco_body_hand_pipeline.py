import importlib.util
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf

from tests.config_helpers import CONFIG_DIR, compose_config

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts/loco_body_hand_pipeline.py"


def load_launcher() -> ModuleType:
    spec = importlib.util.spec_from_file_location("g1_playground_test_loco_body_hand", LAUNCHER)
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


class FakeOdometry:
    def __init__(self):
        self.position = np.array([0.0, 0.0, 0.78], dtype=np.float32)
        self.raw_position = np.array([1.5, -0.25, 0.2], dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.body_height = 0.78


class FakeEnv:
    def __init__(self, events):
        self.events = events
        self.born_place_calls = 0
        self.commands = []
        self.born_place_align = False
        self.base_align = SimpleNamespace(
            base_pos=np.zeros(3, dtype=np.float32),
            base_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )

    def read(self):
        return FakeState()

    def read_odometry(self):
        return FakeOdometry()

    def set_born_place(self, quat, position):
        self.born_place_calls += 1
        self.born_place_align = True
        self.base_align.base_pos = np.asarray(position, dtype=np.float32).copy()
        self.events.append(("born_place", np.asarray(position).copy()))
        return True

    def set_gains(self, stiffness, damping):
        self.events.append(("gains", float(np.asarray(stiffness)[0])))

    def step(self, target):
        self.commands.append(np.asarray(target, dtype=np.float64).copy())

    def activate_commands(self):
        self.events.append(("activate", 0.0))


class FakeHandEnv:
    num_dofs = 12

    def __init__(self):
        self.targets = []

    def read(self):
        return FakeHandState()

    def step(self, target):
        self.targets.append(np.asarray(target, dtype=np.float64).copy())


class FakeMotion:
    num_frames = 414

    def __init__(self, events):
        self.events = events
        self.align_calls = 0
        self.anchor_pos = np.zeros((self.num_frames, 3), dtype=np.float32)
        self.anchor_pos[:, 2] = 0.77
        self.anchor_quat = np.zeros((self.num_frames, 4), dtype=np.float32)
        self.anchor_quat[:, 0] = 1.0

    def align(self):
        self.align_calls += 1
        self.events.append(("align", 0.0))


class FakeHoi:
    action_dim = 41
    observation_dim = 463
    freq = 50
    dt = 0.02

    def __init__(self, events):
        self.events = events
        self.motion = FakeMotion(events)
        self.last_action = np.zeros(41, dtype=np.float32)
        self.frames = []
        self.reset_calls = 0
        self.targets = []
        self.action_inputs = []

    def get_observation(self, frame, anchor_pos, anchor_quat, state, hand_state):
        self.frames.append(int(frame))
        return np.zeros(463, dtype=np.float32)

    def act(self, observation):
        self.action_inputs.append(self.last_action.copy())
        self.last_action = np.full(41, len(self.action_inputs), dtype=np.float32)
        body = np.full(29, 0.4)
        self.targets.append(body.copy())
        return body, np.full(12, 0.3)

    @staticmethod
    def reference_targets():
        return np.full(29, 0.4), np.full(12, 0.3)

    def reset(self):
        self.events.append(("hoi_reset", len(self.action_inputs)))
        self.reset_calls += 1
        self.last_action.fill(0.0)


class FakeObservation:
    def anchor_pose(self, root_pos, root_quat, joint_pos):
        return np.asarray(root_pos, dtype=np.float32).copy(), np.asarray(root_quat, dtype=np.float32).copy()


class FakeTrack:
    freq = 50
    dt = 0.02
    reference_root_height = 0.76

    def __init__(self, events):
        self.events = events
        self.observation = FakeObservation()
        self.references = []
        self.last_action = np.zeros(29)
        self.action_inputs = []
        self.reset_action_indices = []
        self.reset_reference_indices = []

    @property
    def standing_target(self):
        return np.full(29, 0.1)

    def set_reference(self, root_height, root_quat, joint_pos, joint_vel, anchor_lin_vel_w, anchor_ang_vel_w):
        self.references.append(
            SimpleNamespace(
                root_height=float(root_height),
                root_quat=np.asarray(root_quat, dtype=np.float64).copy(),
                joint_pos=np.asarray(joint_pos, dtype=np.float64).copy(),
                joint_vel=np.asarray(joint_vel, dtype=np.float64).copy(),
                lin_vel_w=np.asarray(anchor_lin_vel_w, dtype=np.float64).copy(),
                ang_vel_w=np.asarray(anchor_ang_vel_w, dtype=np.float64).copy(),
            )
        )

    def act(self, state, control):
        self.action_inputs.append(self.last_action.copy())
        self.last_action = np.full(29, -0.7)
        return np.full(29, -0.2)

    def reset(self):
        self.events.append(("track_reset", len(self.action_inputs)))
        self.reset_action_indices.append(len(self.action_inputs))
        self.reset_reference_indices.append(len(self.references))
        self.last_action.fill(0.0)


class FakeLoco:
    freq = 50
    dt = 0.02

    def __init__(self):
        self.calls = 0

    @property
    def standing_target(self):
        return np.zeros(29)

    def act(self, state, control):
        self.calls += 1
        return np.full(29, -0.6)

    def reset(self):
        pass


class FakeController:
    def __init__(self, shutdown_after):
        self.shutdown = False
        self.reads = 0
        self.shutdown_after = shutdown_after

    def read(self):
        self.reads += 1
        return {"axes": {}}, self.shutdown or self.reads > self.shutdown_after


def make_run(module, *, log=None, shutdown_after=780):
    events = []
    run = SimpleNamespace(
        env=FakeEnv(events),
        hand_env=FakeHandEnv(),
        controller=FakeController(shutdown_after),
        loco=FakeLoco(),
        hoi=FakeHoi(events),
        track=FakeTrack(events),
        dt=0.001,
        log=log,
        started=0.0,
        origin=0.0,
        key_descriptor=-1,
        hoi_stiffness=np.full(29, 40.0),
        hoi_damping=np.full(29, 1.0),
    )
    run.events = events
    return run


class TestStateMachine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_launcher()

    def drive(self):
        module = self.module
        log = module.build_log(1000, 12)
        run = make_run(module, log=log)
        polls = 0

        def keys(descriptor):
            nonlocal polls
            polls += 1
            if polls == 2:
                return {module.LOCO_TO_TRACK_KEY}, descriptor
            if polls == 8:
                return {module.TRACK_TO_HOI_KEY}, descriptor
            return set(), descriptor

        with patch.object(module, "poll_keys", side_effect=keys), patch.object(module.time, "sleep"):
            with self.assertRaises(module.Stop):
                module.run_pipeline(run, np.zeros(29), np.zeros(12), 100, 250)
        return run

    def test_only_policy_ownership_is_a_top_level_mode(self):
        self.assertEqual([mode.name for mode in self.module.Mode], ["LOCOMOTION", "TRACK", "HOI"])

    def test_the_two_keys_create_separate_handoffs(self):
        run = self.drive()
        phases = [self.module.PHASES[index] for index in run.log.phase[: run.log.count]]
        self.assertEqual(
            phases[:8], ["locomotion", "track", "track", "track", "track", "track", "track", "track_to_hoi"]
        )
        self.assertEqual(phases.count("track_to_hoi"), 100)
        self.assertEqual(phases.count("hoi"), 414)
        self.assertEqual(phases.count("track_to_default"), 250)

    def test_rebase_alignment_and_gain_switch_happen_only_on_left_bracket(self):
        run = self.drive()
        self.assertEqual(run.env.born_place_calls, 1)
        self.assertEqual(run.hoi.motion.align_calls, 1)
        self.assertEqual([value for name, value in run.events if name == "gains"], [40.0])
        captured = [value for name, value in run.events if name == "born_place"]
        np.testing.assert_allclose(captured[0], FakeOdometry().raw_position)

    def test_track_holds_one_live_reference_while_waiting_for_right_bracket(self):
        run = self.drive()
        first_transition_reference = run.track.references[1]
        np.testing.assert_allclose(run.track.references[0].joint_pos, FakeState().dof_pos)
        self.assertAlmostEqual(run.track.references[0].root_height, FakeOdometry().position[2])
        self.assertGreater(float(np.abs(first_transition_reference.joint_pos).max()), 0.0)
        self.assertEqual(run.track.reset_action_indices[0], 0)
        np.testing.assert_allclose(run.track.action_inputs[0], 0.0)

    def test_right_bracket_does_not_reset_track_history(self):
        run = self.drive()
        reset_names = [name for name, _ in run.events if name.endswith("_reset")]
        self.assertEqual(reset_names, ["track_reset", "hoi_reset", "track_reset"])
        self.assertEqual(len(run.track.reset_action_indices), 2)
        self.assertGreater(run.track.reset_action_indices[1], 100)

    def test_each_track_transition_finishes_with_a_static_reference(self):
        run = self.drive()
        default_start = run.track.reset_reference_indices[1]
        for reference in (run.track.references[default_start - 1], run.track.references[-1]):
            np.testing.assert_allclose(reference.joint_vel, 0.0)
            np.testing.assert_allclose(reference.lin_vel_w, 0.0)
            np.testing.assert_allclose(reference.ang_vel_w, 0.0)

    def test_both_brackets_in_one_read_do_not_skip_track_hold(self):
        module = self.module
        run = make_run(module, shutdown_after=20)
        polls = 0

        def keys(descriptor):
            nonlocal polls
            polls += 1
            both = {module.LOCO_TO_TRACK_KEY, module.TRACK_TO_HOI_KEY}
            return (both if polls == 2 else set()), descriptor

        with patch.object(module, "poll_keys", side_effect=keys), patch.object(module.time, "sleep"):
            with self.assertRaises(module.Stop):
                module.run_pipeline(run, np.zeros(29), np.zeros(12), 10, 10)

        self.assertEqual(run.machine.mode, module.Mode.TRACK)
        self.assertIsNone(run.machine.track_goal)
        self.assertEqual(run.hoi.frames, [])

    def test_hoi_runs_every_reference_frame_once_and_sends_its_targets_directly(self):
        run = self.drive()
        self.assertEqual(run.hoi.frames, list(range(414)))
        phases = np.asarray(run.log.phase[: run.log.count])
        hoi_rows = np.flatnonzero(phases == self.module.PHASES.index("hoi"))
        commands = np.asarray(run.env.commands)[hoi_rows]
        np.testing.assert_allclose(commands, np.asarray(run.hoi.targets))
        np.testing.assert_allclose(run.hoi.action_inputs[0], 0.0)

    def test_track_resets_after_hoi_and_returns_the_reference_to_default(self):
        run = self.drive()
        second_reset_action = run.track.reset_action_indices[1]
        second_reset_reference = run.track.reset_reference_indices[1]
        np.testing.assert_allclose(run.track.action_inputs[second_reset_action], 0.0)
        live = run.track.references[second_reset_reference]
        np.testing.assert_allclose(live.joint_pos, FakeState().dof_pos)
        np.testing.assert_allclose(run.track.references[-1].joint_pos, run.track.standing_target)
        self.assertAlmostEqual(run.track.references[-1].root_height, run.track.reference_root_height)
        self.assertEqual(run.machine.mode, self.module.Mode.TRACK)
        self.assertIsNone(run.machine.track_goal)

    def test_the_hand_is_commanded_on_every_control_frame(self):
        run = self.drive()
        self.assertEqual(len(run.hand_env.targets), len(run.env.commands))
        np.testing.assert_allclose(run.hand_env.targets[-1], np.full(12, 0.3))


class TestSafetyBoundary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_launcher()

    def test_a_shutdown_request_stops_the_run(self):
        run = make_run(self.module)
        run.controller.shutdown = True
        with self.assertRaises(self.module.Stop):
            self.module.read_frame(run)

    def test_a_fallen_robot_stops_the_run(self):
        run = make_run(self.module)
        fallen = FakeState()
        fallen.base_quat = np.array([0.7071, 0.0, 0.0, 0.7071])
        run.env.read = lambda: fallen
        with self.assertRaises(self.module.Stop):
            self.module.read_frame(run)

    def test_rebase_requires_odometry(self):
        run = make_run(self.module)
        machine = SimpleNamespace(hand_command=np.zeros(12))
        with self.assertRaisesRegex(self.module.Stop, "cannot capture the Track origin"):
            self.module.enter_track(run, machine, FakeState(), None)

    def test_stale_hand_state_stops_track_to_hoi_but_not_track_entry(self):
        module = self.module
        run = make_run(module)
        run.hand_env.read = lambda: FakeHandState(age=1.0)
        polls = 0

        def keys(descriptor):
            nonlocal polls
            polls += 1
            if polls == 1:
                return {module.LOCO_TO_TRACK_KEY}, descriptor
            if polls == 2:
                return {module.TRACK_TO_HOI_KEY}, descriptor
            return set(), descriptor

        with patch.object(module, "poll_keys", side_effect=keys), patch.object(module.time, "sleep"):
            with self.assertRaisesRegex(module.Stop, "Inspire hand state is stale"):
                module.run_pipeline(run, np.zeros(29), np.zeros(12), 10, 10)

        self.assertEqual(run.env.born_place_calls, 1)
        self.assertEqual(run.hoi.frames, [])

    def test_hoi_checks_each_odometry_frame_after_rebase(self):
        module = self.module
        run = make_run(module)
        odometry_reads = 0
        polls = 0

        def read_odometry():
            nonlocal odometry_reads
            odometry_reads += 1
            return None if odometry_reads >= 4 else FakeOdometry()

        def keys(descriptor):
            nonlocal polls
            polls += 1
            if polls == 1:
                return {module.LOCO_TO_TRACK_KEY}, descriptor
            if polls == 2:
                return {module.TRACK_TO_HOI_KEY}, descriptor
            return set(), descriptor

        run.env.read_odometry = read_odometry
        with patch.object(module, "poll_keys", side_effect=keys), patch.object(module.time, "sleep"):
            with self.assertRaisesRegex(module.Stop, "Odometry is unavailable during HOI"):
                module.run_pipeline(run, np.zeros(29), np.zeros(12), 1, 10)

        self.assertEqual(run.hoi.frames, [])


class TestConfiguration(unittest.TestCase):
    def test_the_run_root_composes_with_both_deployments(self):
        for deployment in ("sim", "real"):
            with self.subTest(deployment=deployment):
                cfg = compose_config(deployment, config_name="run_loco_hoi_track")
                self.assertEqual(cfg.env._target_, "g1_playground.g1_env.G1Env")
                self.assertIs(cfg.env.enable_odometry, True)
                self.assertEqual(len(cfg.inspire.dof.joint_names), 12)
                self.assertEqual(len(cfg.loco.dof.joint_names), 29)
                self.assertIn("observation", cfg.hoi)
                self.assertEqual(len(cfg.track.dof.joint_names), 29)

    def test_the_run_root_uses_policy_roles_not_task_names(self):
        root = OmegaConf.load(CONFIG_DIR / "run_loco_hoi_track.yaml")
        self.assertEqual(set(root), {"defaults", "device", "startup", "handover", "recording", "env", "hydra"})
        self.assertEqual(set(root.startup), {"ramp_seconds"})
        self.assertEqual(set(root.handover), {"to_hoi_seconds", "to_default_seconds"})
        defaults = [str(item) for item in root.defaults]
        self.assertTrue(any("policy@hoi" in item for item in defaults))
        self.assertTrue(any("policy@track" in item for item in defaults))

    def test_only_one_environment_of_each_kind_is_created(self):
        source = LAUNCHER.read_text()
        self.assertEqual(source.count("instantiate(cfg.env"), 1)
        self.assertEqual(source.count("InspireHandEnv("), 1)
        self.assertEqual(source.count("activate_commands()"), 1)


if __name__ == "__main__":
    unittest.main()
