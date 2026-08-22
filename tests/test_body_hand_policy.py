import unittest

import numpy as np
from omegaconf import OmegaConf

from g1_playground.policy.body_hand import BodyHandPolicy
from g1_playground.utils.dof import DoFAdapter
from tests.body_hand_helpers import (
    CONFIG_DIR,
    MODEL_PATH,
    REPO_ROOT,
    body_joint_names,
    hand_joint_names,
    inspire_cfg,
    motion_cfg,
    motion_data,
    policy_cfg,
    policy_data,
    session,
    training_bundle,
)
from tests.config_helpers import compose_config

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def build_policy(cfg=None, motion=None, body=None, hand=None, mimic=None, inspire=None):
    inspire = inspire if inspire is not None else inspire_cfg()
    return BodyHandPolicy(
        cfg if cfg is not None else policy_cfg(),
        motion if motion is not None else motion_cfg(),
        device="cpu",
        runtime_body_joint_names=body if body is not None else body_joint_names(),
        runtime_hand_joint_names=hand if hand is not None else inspire.dof.joint_names,
        hand_mimic=mimic if mimic is not None else inspire.mimic,
    )


class TestModelInterface(unittest.TestCase):
    def test_the_policy_config_carries_the_runtime_parameters(self):
        config = policy_data()
        self.assertEqual(config["frequency"], 50)
        self.assertEqual(len(config["observation"]["joint_names"]), 53)
        self.assertEqual(
            len(config["action"]["body"]["joint_names"]) + len(config["action"]["hand"]["joint_names"]), 41
        )

    def test_the_graph_signature_matches_the_policy_config(self):
        graph = session()
        policy = build_policy()
        self.assertEqual(list(graph.get_inputs()[0].shape), [1, policy.observation_dim])
        self.assertEqual(list(graph.get_outputs()[0].shape), [1, policy.action_dim])

    def test_the_graph_does_not_own_deployment_configuration(self):
        self.assertEqual(session().get_modelmeta().custom_metadata_map, {})


class TestBodyHandPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = build_policy()
        cls.config = policy_data()

    def test_the_complete_motion_is_loaded(self):
        self.assertEqual(self.policy.motion.num_frames, 414)

    def test_targets_are_split_by_joint_name(self):
        names = list(self.config["action"]["body"]["joint_names"]) + list(self.config["action"]["hand"]["joint_names"])
        marker = np.arange(41, dtype=np.float32)
        cfg = policy_cfg()
        cfg.action.body.control.lower = [-1e3] * 29
        cfg.action.body.control.upper = [1e3] * 29
        wide = build_policy(cfg=cfg)
        body, hand = wide.split(marker)
        for index, name in enumerate(body_joint_names()):
            self.assertEqual(float(body[index]), float(names.index(name)))
        for index, name in enumerate(hand_joint_names()):
            self.assertEqual(float(hand[index]), float(names.index(name)))

    def test_a_wrong_sized_action_is_rejected_like_the_dof_adapter(self):
        for size in (40, 41 + 4, 53):
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    self.policy.split(np.zeros(size, dtype=np.float32))
                with self.assertRaises(ValueError):
                    self.policy.process(np.zeros(size, dtype=np.float32))

    def test_a_wrong_sized_joint_reading_is_rejected(self):
        for body, hand in ((28, 12), (29, 11), (35, 12)):
            with self.subTest(body=body, hand=hand):
                with self.assertRaises(ValueError):
                    self.policy.assembler.positions(np.zeros(body), np.zeros(hand))

    def test_the_hand_action_mapping_matches_the_published_table(self):
        self.assertEqual(self.policy.hand_to_runtime.indices.tolist(), [7, 8, 6, 5, 11, 9, 2, 3, 1, 0, 10, 4])

    def test_a_naive_slice_would_send_the_wrong_targets(self):
        marker = np.arange(41, dtype=np.float32)
        cfg = policy_cfg()
        cfg.action.body.control.lower = [-1e3] * 29
        cfg.action.body.control.upper = [1e3] * 29
        wide = build_policy(cfg=cfg)
        body, hand = wide.split(marker)
        self.assertFalse(np.array_equal(body, marker[:29]))
        self.assertFalse(np.array_equal(hand, marker[29:]))

    def test_body_targets_are_clipped_by_the_policy_configuration(self):
        cfg = policy_cfg()
        adapter = DoFAdapter(cfg.action.body.joint_names, body_joint_names())
        control = cfg.action.body.control
        body, _ = self.policy.split(np.full(41, 1e3, dtype=np.float32))
        np.testing.assert_allclose(body, adapter.fit(control.upper), atol=1e-5)
        body, _ = self.policy.split(np.full(41, -1e3, dtype=np.float32))
        np.testing.assert_allclose(body, adapter.fit(control.lower), atol=1e-5)

    def test_hand_targets_are_left_to_the_inspire_boundary(self):
        _, hand = self.policy.split(np.full(41, 1e3, dtype=np.float32))
        self.assertTrue(bool(np.all(hand > 100.0)))

    def test_processing_is_scale_then_offset_in_action_order(self):
        raw = np.linspace(-1.0, 1.0, 41).astype(np.float32)
        expected = raw * np.asarray(self.config["action"]["scale"], dtype=np.float32) + np.asarray(
            self.config["action"]["offset"], dtype=np.float32
        )
        np.testing.assert_allclose(self.policy.process(raw), expected, atol=1e-6)

    def test_the_previous_raw_action_feeds_the_next_observation(self):
        policy = build_policy()
        body_state = SimpleState(np.zeros(29), np.zeros(29), np.zeros(3))
        hand_state = SimpleState(np.zeros(12), np.zeros(12))
        np.testing.assert_allclose(policy.last_action, 0.0)
        policy.act(policy.get_observation(0, np.zeros(3), IDENTITY, body_state, hand_state))
        action = policy.last_action
        self.assertGreater(float(np.abs(action).max()), 0.0)
        observation = policy.get_observation(0, np.zeros(3), IDENTITY, body_state, hand_state)
        np.testing.assert_allclose(observation[-policy.action_dim :], action, atol=1e-6)

    def test_the_policy_keeps_no_observation_state(self):
        policy = build_policy()
        self.assertFalse(hasattr(policy, "last_observation"))
        self.assertFalse(hasattr(policy, "_last_observation"))
        self.assertEqual(
            [name for name in vars(policy) if name.startswith("_last")],
            ["_last_action"],
        )

    def test_a_non_finite_action_stops_the_run(self):
        policy = build_policy()
        body_state = SimpleState(np.zeros(29), np.zeros(29), np.zeros(3))
        hand_state = SimpleState(np.zeros(12), np.zeros(12))
        observation = policy.get_observation(0, np.zeros(3), IDENTITY, body_state, hand_state)
        policy.act(observation)
        policy.infer = lambda values: np.full(policy.action_dim, np.nan, dtype=np.float32)
        with self.assertRaises(RuntimeError):
            policy.act(observation)

    def test_the_caller_holds_the_array_the_network_saw(self):
        policy = build_policy()
        body_state = SimpleState(np.zeros(29), np.zeros(29), np.zeros(3))
        hand_state = SimpleState(np.zeros(12), np.zeros(12))
        seen = []
        original = policy.infer
        policy.infer = lambda values: (seen.append(values.copy()), original(values))[1]
        for _ in range(3):
            observation = policy.get_observation(0, np.zeros(3), IDENTITY, body_state, hand_state)
            policy.act(observation)
            np.testing.assert_allclose(observation, seen[-1], atol=0.0)
            self.assertFalse(np.array_equal(observation[-policy.action_dim :], policy.last_action))

    def test_reference_targets_come_back_in_the_deployment_orders(self):
        body, hand = self.policy.reference_targets()
        names = list(self.config["observation"]["joint_names"])
        frame = self.policy.motion.joint_pos[0]
        for index, name in enumerate(body_joint_names()):
            self.assertAlmostEqual(float(body[index]), float(frame[names.index(name)]), places=6)
        for index, name in enumerate(hand_joint_names()):
            self.assertAlmostEqual(float(hand[index]), float(frame[names.index(name)]), places=6)


class SimpleState:
    def __init__(self, joint_pos, joint_vel, base_ang_vel=None):
        self.dof_pos = joint_pos
        self.dof_vel = joint_vel
        self.joint_pos = joint_pos
        self.joint_vel = joint_vel
        self.base_ang_vel = base_ang_vel


class TestPolicyRejections(unittest.TestCase):
    def test_a_renamed_body_joint_is_refused(self):
        renamed = body_joint_names()
        renamed[0] = "not_a_joint"
        with self.assertRaises(ValueError):
            build_policy(body=renamed)

    def test_a_missing_hand_joint_is_refused(self):
        with self.assertRaises(ValueError):
            build_policy(hand=hand_joint_names()[:-1])

    def test_a_renamed_hand_joint_is_refused(self):
        renamed = hand_joint_names()
        renamed[0] = "not_a_joint"
        with self.assertRaises(ValueError):
            build_policy(hand=renamed)

    def test_a_mimic_driver_outside_the_hand_is_refused(self):
        mimic = inspire_cfg().mimic
        mimic.L_index_intermediate_joint.driver = "left_knee_joint"
        with self.assertRaises(ValueError):
            build_policy(mimic=mimic)

    def test_the_policy_never_reaches_into_an_inspire_configuration(self):
        import inspect

        from g1_playground.policy.body_hand import body_hand_policy

        source = inspect.getsource(body_hand_policy)
        self.assertNotIn("inspire", source.lower())
        self.assertIn("runtime_body_joint_names", source)
        self.assertIn("runtime_hand_joint_names", source)
        self.assertIn("hand_mimic", source)


class TestConfigurationOwnership(unittest.TestCase):
    def test_the_policy_fragment_owns_the_network_semantics(self):
        cfg = policy_cfg()
        self.assertEqual(set(cfg), {"policy_file", "frequency", "observation", "action"})
        self.assertEqual(set(cfg.observation), {"joint_names", "default_joint_pos", "future_offsets", "anchor_body"})
        self.assertEqual(set(cfg.action), {"body", "hand", "scale", "offset"})
        self.assertEqual(set(cfg.action.body), {"joint_names", "control"})
        self.assertEqual(set(cfg.action.body.control), {"stiffness", "damping", "lower", "upper"})
        self.assertEqual(set(cfg.action.hand), {"joint_names"})
        for banned in ("contract_file", "golden_file", "motion_file", "start_frame", "stop_frame", "loop", "startup"):
            with self.subTest(key=banned):
                self.assertNotIn(banned, cfg)

    def test_body_control_parameters_use_policy_order_not_mjcf_order(self):
        cfg = policy_cfg()
        self.assertNotEqual(list(cfg.action.body.joint_names), body_joint_names())

    def test_the_motion_fragment_only_selects_the_file(self):
        cfg = motion_cfg()
        self.assertEqual(set(cfg), {"file"})

    def test_the_runner_owns_startup_and_recording(self):
        root = OmegaConf.load(CONFIG_DIR / "run_body_hand.yaml")
        self.assertEqual(set(root), {"defaults", "device", "startup", "recording", "env", "hydra"})
        self.assertEqual(set(root.startup), {"ramp_seconds", "blend_seconds"})
        self.assertEqual(set(root.recording), {"enabled", "directory"})

    def test_the_run_root_composes_with_both_deployments(self):
        for deployment in ("sim", "real"):
            with self.subTest(deployment=deployment):
                cfg = compose_config(deployment, config_name="run_body_hand")
                self.assertEqual(cfg.env._target_, "g1_playground.g1_env.G1Env")
                self.assertIs(cfg.env.enable_odometry, True)
                self.assertEqual(cfg.motion.file, "assets/motions/largebox/sub16_largebox_022_v00.npz")

    def test_the_composed_runtime_dof_carries_the_policy_gains(self):
        cfg = policy_cfg()
        adapter = DoFAdapter(cfg.action.body.joint_names, body_joint_names())
        self.assertEqual(adapter.fit(cfg.action.body.control.stiffness).shape, (29,))
        self.assertEqual(adapter.fit(cfg.action.body.control.damping).shape, (29,))

    def test_the_gains_match_the_training_configuration(self):
        bundle = training_bundle()
        if bundle is None:
            self.skipTest("training bundle is not available")
        import yaml

        with open(bundle / "deployment_contract.yaml") as handle:
            contract = yaml.safe_load(handle)
        names = contract["joint_names_53"]
        cfg = policy_cfg()
        control = cfg.action.body.control
        for index, name in enumerate(cfg.action.body.joint_names):
            slot = names.index(name)
            self.assertAlmostEqual(float(control.stiffness[index]), float(contract["joint_stiffness_53"][slot]), 3)
            self.assertAlmostEqual(float(control.damping[index]), float(contract["joint_damping_53"][slot]), 3)
            limits = contract["joint_pos_limits_53"][slot]
            self.assertAlmostEqual(float(control.lower[index]), float(limits[0]), 3)
            self.assertAlmostEqual(float(control.upper[index]), float(limits[1]), 3)

    def test_the_staged_motion_matches_the_policy_config(self):
        cfg = policy_cfg()
        motion = motion_data()
        self.assertEqual([str(name) for name in motion["joint_names"]], list(cfg.observation.joint_names))
        self.assertEqual(int(np.asarray(motion["fps"]).reshape(-1)[0]), cfg.frequency)

    def test_the_old_configuration_names_are_gone(self):
        for name in ("run_largebox.yaml", "policy/largebox_body_hand.yaml"):
            with self.subTest(name=name):
                self.assertFalse((CONFIG_DIR / name).exists())

    def test_every_referenced_asset_exists(self):
        self.assertTrue((REPO_ROOT / str(policy_cfg().policy_file)).is_file())
        self.assertTrue((REPO_ROOT / str(motion_cfg().file)).is_file())
        self.assertEqual(MODEL_PATH, REPO_ROOT / str(policy_cfg().policy_file))
        self.assertTrue(MODEL_PATH.with_suffix(".onnx.data").is_file())


if __name__ == "__main__":
    unittest.main()
