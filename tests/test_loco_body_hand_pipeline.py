import importlib.util
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace

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
            base_pos=np.zeros(3, dtype=np.float32), base_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
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

    def align(self):
        self.align_calls += 1
        self.events.append(("align", 0.0))


class FakeLargebox:
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
        self.acts = 0
        self.targets = []

    def get_observation(self, frame, anchor_pos, anchor_quat, state, hand_state):
        self.frames.append(int(frame))
        return np.zeros(463, dtype=np.float32)

    def act(self, observation):
        self.acts += 1
        body = np.full(29, 3.0) if self.acts > self.motion.num_frames else np.full(29, 0.4)
        self.targets.append(body.copy())
        return body, np.full(12, 0.3)

    def reset(self):
        self.events.append(("largebox_reset", 0.0))
        self.reset_calls += 1


class FakeObservation:
    def anchor_pose(self, root_pos, root_quat, joint_pos):
        return np.asarray(root_pos, dtype=np.float32).copy(), np.asarray(root_quat, dtype=np.float32).copy()


class FakeStandTrack:
    freq = 50
    dt = 0.02
    reference_root_height = 0.76

    def __init__(self, events):
        self.events = events
        self.observation = FakeObservation()
        self.calls = 0
        self.reset_calls = 0
        self.applied = []
        self.references = []
        self.last_action = np.zeros(29)

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
        self.events.append(("set_reference", 0.0))

    def act(self, state, control):
        self.calls += 1
        self.events.append(("stand_act", 0.0))
        self.last_action = np.full(29, -0.7)
        return np.full(29, -0.2)

    def accept_applied_target(self, target):
        self.applied.append(np.asarray(target, dtype=np.float64).copy())

    def reset(self):
        self.events.append(("stand_reset", 0.0))
        self.reset_calls += 1


class FakeLoco:
    freq = 50
    dt = 0.02

    def __init__(self):
        self.reset_calls = 0

    @property
    def standing_target(self):
        return np.zeros(29)

    def act(self, state, control):
        return np.full(29, -0.6)

    def reset(self):
        self.reset_calls += 1


class FakeController:
    def __init__(self, shutdown_after=None):
        self.shutdown = False
        self.reads = 0
        self.shutdown_after = shutdown_after

    def read(self):
        self.reads += 1
        stop = self.shutdown or (self.shutdown_after is not None and self.reads > self.shutdown_after)
        return {"axes": {}}, stop


def make_run(module, log=None, shutdown_after=None):
    events = []
    run = SimpleNamespace(
        env=FakeEnv(events),
        hand_env=FakeHandEnv(),
        controller=FakeController(shutdown_after),
        loco=FakeLoco(),
        largebox=FakeLargebox(events),
        stand_track=FakeStandTrack(events),
        dt=0.001,
        log=log,
        started=0.0,
        origin=0.0,
        key_descriptor=-1,
        body_rate_limit=0.05,
        largebox_stiffness=np.full(29, 40.0),
        largebox_damping=np.full(29, 1.0),
    )
    run.events = events
    return run


class TestHandoverSequence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_launcher()

    def drive(self, hold_frames=20):
        module = self.module
        run = make_run(module, shutdown_after=None)
        state, hand_state, odometry = FakeState(), FakeHandState(), FakeOdometry()
        body, hand = np.zeros(29), np.zeros(12)
        state, odometry = module.capture_largebox_origin(run, state, hand_state, odometry)
        body, hand = module.blend_to_largebox(run, 100, hand)
        run.commands_after_crossfade = len(run.env.commands)
        body, hand = module.run_largebox_motion(run)
        run.largebox_frames_at_handover = len(run.largebox.frames)
        run.commands_at_largebox_end = len(run.env.commands)
        run.largebox_last_command = run.env.commands[-1].copy()
        state, hand_state, odometry, _ = module.read_frame(run)
        run.measured_body = np.asarray(state.dof_pos, dtype=np.float64).copy()
        start, body = module.capture_standing_reference(run, state, odometry)
        run.commands_at_standing = len(run.env.commands)
        body, hand = module.track_to_standing(run, 250, body, hand, start)
        run.controller.shutdown_after = run.controller.reads + hold_frames
        with self.assertRaises(module.Stop):
            module.hold_stand_track(run, body, hand)
        return run, body, hand

    def test_the_origin_is_captured_exactly_once(self):
        run, _, _ = self.drive()
        self.assertEqual(run.env.born_place_calls, 1)
        self.assertEqual(run.largebox.motion.align_calls, 1)

    def test_the_origin_uses_the_raw_odometry_position(self):
        run, _, _ = self.drive()
        captured = [value for name, value in run.events if name == "born_place"]
        np.testing.assert_allclose(captured[0], FakeOdometry().raw_position)

    def test_the_reference_runs_every_frame_once_in_order(self):
        run, _, _ = self.drive()
        played = run.largebox.frames
        self.assertEqual(set(played[:100]), {0})
        self.assertEqual(played[100:], list(range(414)))

    def test_the_last_frame_is_never_replayed_after_the_reference_ends(self):
        run, _, _ = self.drive()
        self.assertEqual(len(run.largebox.frames), run.largebox_frames_at_handover)
        self.assertEqual(run.largebox.frames[-1], 413)
        self.assertEqual(run.largebox.frames.count(413), 1)

    def test_the_standing_tracker_is_reset_once_when_the_reference_ends(self):
        run, _, _ = self.drive()
        order = [name for name, _ in run.events if name.endswith("_reset")]
        self.assertEqual(order, ["largebox_reset", "stand_reset"])
        self.assertEqual(run.stand_track.reset_calls, 1)

    def test_the_gains_switch_once_and_never_switch_back(self):
        run, _, _ = self.drive()
        gains = [value for name, value in run.events if name == "gains"]
        self.assertEqual(gains, [40.0])

    def test_the_crossfade_is_the_eased_policy_blend(self):
        run, _, _ = self.drive()
        crossfade = np.asarray(run.env.commands[: run.commands_after_crossfade])
        progress = np.arange(1, 101, dtype=np.float64) / 100.0
        alpha = progress**2 * (3.0 - 2.0 * progress)
        expected = np.repeat((-0.6 + alpha)[:, None], 29, axis=1)
        np.testing.assert_allclose(crossfade, expected)
        np.testing.assert_allclose(crossfade[-1], run.largebox.targets[99])

    def test_the_largebox_phase_executes_policy_targets_directly(self):
        run, _, _ = self.drive()
        commands = np.asarray(run.env.commands)
        largebox = commands[run.commands_after_crossfade : run.commands_at_largebox_end]
        targets = np.asarray(run.largebox.targets[100 : 100 + run.largebox.motion.num_frames])
        np.testing.assert_allclose(largebox, targets)

    def test_a_sudden_network_target_is_not_rate_limited_during_motion(self):
        run, _, _ = self.drive()
        commands = np.asarray(run.env.commands[run.commands_after_crossfade : run.commands_at_largebox_end])
        targets = np.asarray(run.largebox.targets[100 : 100 + run.largebox.motion.num_frames])
        jumps = np.flatnonzero(np.abs(np.diff(targets, axis=0)).max(axis=1) > 1.0)
        self.assertGreater(len(jumps), 0, "the fake policy never produced a sudden target")
        spike = int(jumps[0]) + 1
        target_step = targets[spike] - targets[spike - 1]
        command_step = commands[spike] - commands[spike - 1]
        self.assertGreater(float(np.abs(target_step).max()), 2.0)
        np.testing.assert_allclose(command_step, target_step)

    def test_the_first_standing_command_starts_from_the_measured_pose(self):
        run, _, _ = self.drive()
        first = run.env.commands[run.commands_at_standing]
        self.assertLessEqual(float(np.abs(first - run.measured_body).max()), run.body_rate_limit + 1e-9)

    def test_the_out_of_range_largebox_target_never_leaks_into_the_tracker(self):
        run, _, _ = self.drive()
        first = run.env.commands[run.commands_at_standing]
        self.assertGreater(float(np.abs(run.largebox_last_command - run.measured_body).max()), 1.0)
        self.assertFalse(np.array_equal(first, run.largebox_last_command))
        self.assertGreater(float(np.abs(first - run.largebox_last_command).max()), 1.0)

    def test_the_standing_phases_are_rate_limited_within_themselves(self):
        run, _, _ = self.drive(hold_frames=40)
        standing = np.asarray(run.env.commands[run.commands_at_standing :])
        steps = np.abs(np.diff(standing, axis=0)).max(axis=1)
        self.assertLessEqual(float(steps.max()), run.body_rate_limit + 1e-9)

    def test_the_hand_is_commanded_on_every_frame_and_holds_the_largebox_target(self):
        run, _, hand = self.drive()
        self.assertEqual(len(run.hand_env.targets), len(run.env.commands))
        np.testing.assert_allclose(hand, np.full(12, 0.3))
        np.testing.assert_allclose(run.hand_env.targets[-1], np.full(12, 0.3))

    def test_the_tracker_is_reset_once_after_the_reference_ends(self):
        run, _, _ = self.drive()
        names = [name for name, _ in run.events]
        self.assertEqual(names.count("stand_reset"), 1)
        self.assertLess(names.index("largebox_reset"), names.index("stand_reset"))
        self.assertLess(names.index("stand_reset"), names.index("stand_act"))

    def test_the_first_reference_is_the_live_pose_not_the_last_command(self):
        run, _, _ = self.drive()
        first = run.stand_track.references[0]
        np.testing.assert_allclose(first.joint_pos, FakeState().dof_pos)
        np.testing.assert_allclose(first.joint_vel, 0.0)
        np.testing.assert_allclose(first.lin_vel_w, 0.0)
        np.testing.assert_allclose(first.ang_vel_w, 0.0)
        self.assertAlmostEqual(first.root_height, float(FakeOdometry().position[2]))

    def test_the_reference_walks_to_the_default_pose_and_stops_there(self):
        run, _, _ = self.drive()
        references = run.stand_track.references
        np.testing.assert_allclose(references[-1].joint_pos, run.stand_track.standing_target)
        np.testing.assert_allclose(references[-1].joint_vel, 0.0)
        self.assertAlmostEqual(references[-1].root_height, run.stand_track.reference_root_height)
        travel = [float(np.abs(item.joint_pos - run.stand_track.standing_target).max()) for item in references]
        self.assertTrue(all(b <= a + 1e-9 for a, b in zip(travel[:-1], travel[1:], strict=True)))

    def test_the_reference_velocity_follows_the_eased_derivative(self):
        run, _, _ = self.drive()
        speed = [float(np.abs(item.joint_vel).max()) for item in run.stand_track.references[1:-1]]
        peak = max(speed)
        self.assertGreater(peak, 0.0)
        self.assertLess(speed[0] / peak, 0.05)
        self.assertLess(speed[-1] / peak, 0.05)
        self.assertAlmostEqual(speed[len(speed) // 2] / peak, 1.0, places=2)

    def test_each_phase_records_only_the_policy_that_ran(self):
        module = self.module
        log = module.build_log(4000, FakeLargebox([]), FakeStandTrack([]), 12)
        run = make_run(module, log=log)
        state, hand_state, odometry, _ = module.read_frame(run)
        body, hand = np.zeros(29), np.zeros(12)
        body, hand = module.run_largebox_motion(run)
        largebox_rows = log.count
        start, body = module.capture_standing_reference(run, state, odometry)
        module.track_to_standing(run, 30, body, hand, start)

        phases = [module.PHASES[index] for index in log.phase[: log.count]]
        self.assertEqual(set(phases[:largebox_rows]), {"largebox"})
        self.assertEqual(set(phases[largebox_rows:]), {"to_standing"})
        self.assertTrue(bool(np.isfinite(log.largebox_action[:largebox_rows]).all()))
        self.assertTrue(bool(np.isnan(log.largebox_action[largebox_rows : log.count]).all()))
        self.assertTrue(bool(np.isnan(log.stand_action[:largebox_rows]).all()))
        self.assertTrue(bool(np.isfinite(log.stand_action[largebox_rows : log.count]).all()))
        np.testing.assert_allclose(log.stand_action[log.count - 1], np.full(29, -0.7))

    def test_the_applied_target_is_fed_back_on_every_frame(self):
        run, _, _ = self.drive(hold_frames=20)
        blended = run.env.commands[run.commands_at_standing :]
        seeded, applied = run.stand_track.applied[0], run.stand_track.applied[1:]
        np.testing.assert_allclose(seeded, run.measured_body)
        self.assertFalse(np.array_equal(seeded, run.largebox_last_command))
        self.assertEqual(len(applied), len(blended))
        for value, command in zip(applied, blended, strict=True):
            np.testing.assert_allclose(value, command)

    def test_the_hold_phase_keeps_the_rate_limit(self):
        run, _, _ = self.drive(hold_frames=40)
        steps = np.abs(np.diff(np.asarray(run.env.commands[run.commands_at_standing :]), axis=0)).max(axis=1)
        self.assertLessEqual(float(steps.max()), run.body_rate_limit + 1e-9)

    def test_the_standing_phase_does_not_need_odometry(self):
        module = self.module
        run = make_run(module)
        state, _, odometry, _ = module.read_frame(run)
        start, body = module.capture_standing_reference(run, state, odometry)
        run.env.read_odometry = lambda: None
        body, hand = module.track_to_standing(run, 5, body, np.full(12, 0.3), start)
        run.controller.shutdown_after = run.controller.reads + 5
        with self.assertRaises(module.Stop):
            module.hold_stand_track(run, body, hand)
        self.assertGreaterEqual(run.stand_track.calls, 10)


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

    def test_missing_odometry_stops_only_the_largebox_phases(self):
        run = make_run(self.module)
        run.env.read_odometry = lambda: None
        self.module.read_frame(run)
        with self.assertRaises(self.module.Stop):
            self.module.require_largebox_inputs(FakeHandState(), None)

    def test_a_stale_hand_stops_only_the_largebox_phases(self):
        run = make_run(self.module)
        stale = FakeHandState(age=1.0)
        run.hand_env.read = lambda: stale
        self.module.read_frame(run)
        with self.assertRaises(self.module.Stop):
            self.module.require_largebox_inputs(stale, FakeOdometry())


class TestConfiguration(unittest.TestCase):
    def test_the_run_root_composes_with_both_deployments(self):
        for deployment in ("sim", "real"):
            with self.subTest(deployment=deployment):
                cfg = compose_config(deployment, config_name="run_loco_largebox_track")
                self.assertEqual(cfg.env._target_, "g1_playground.g1_env.G1Env")
                self.assertIs(cfg.env.enable_odometry, True)
                self.assertEqual(len(cfg.inspire.dof.joint_names), 12)
                self.assertEqual(len(cfg.loco.dof.joint_names), 29)
                self.assertIn("observation", cfg.largebox)
                self.assertEqual(len(cfg.stand_track.dof.joint_names), 29)

    def test_the_runner_owns_startup_handover_and_recording(self):
        root = OmegaConf.load(CONFIG_DIR / "run_loco_largebox_track.yaml")
        self.assertEqual(set(root), {"defaults", "device", "startup", "handover", "recording", "env", "hydra"})
        self.assertEqual(
            set(root.handover),
            {"to_largebox_seconds", "to_standing_seconds", "body_rate_limit"},
        )

    def test_the_existing_single_policy_launchers_are_untouched(self):
        for name in ("body_hand_pipeline.py", "loco_track_pipeline.py", "pipeline.py", "track_pipeline.py"):
            with self.subTest(name=name):
                source = (REPO_ROOT / "scripts" / name).read_text()
                self.assertNotIn("run_loco_largebox_track", source)

    def test_only_one_environment_of_each_kind_is_created(self):
        source = LAUNCHER.read_text()
        self.assertEqual(source.count("instantiate(cfg.env"), 1)
        self.assertEqual(source.count("InspireHandEnv("), 1)
        self.assertEqual(source.count("activate_commands()"), 1)


if __name__ == "__main__":
    unittest.main()
