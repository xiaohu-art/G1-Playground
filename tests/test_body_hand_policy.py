import unittest

import numpy as np
from omegaconf import OmegaConf

from g1_playground.policy.body_hand import BodyHandPolicy
from tests.body_hand_helpers import (
    MODEL_PATH,
    REPO_ROOT,
    body_joint_names,
    hand_joint_names,
    inspire_cfg,
    motion_cfg,
    policy_cfg,
)
from tests.config_helpers import compose_config
from tests.runner_helpers import body_hand_runner

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def build_policy(cfg=None, motion=None, body=None, hand=None, mimic=None, runner=None):
    inspire = inspire_cfg()
    return BodyHandPolicy(
        cfg if cfg is not None else policy_cfg(),
        motion if motion is not None else motion_cfg(),
        runtime_body_joint_names=body if body is not None else body_joint_names(),
        runtime_hand_joint_names=hand if hand is not None else inspire.dof.joint_names,
        hand_mimic=mimic if mimic is not None else inspire.mimic,
        runner=runner if runner is not None else body_hand_runner(),
    )


class State:
    def __init__(self, joint_pos, joint_vel, base_ang_vel=None):
        self.dof_pos = joint_pos
        self.dof_vel = joint_vel
        self.joint_pos = joint_pos
        self.joint_vel = joint_vel
        self.base_ang_vel = base_ang_vel


class TestBodyHandPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = build_policy()
        cls.config = policy_cfg()

    def test_model_and_motion_interfaces(self):
        self.assertEqual(self.policy.runner.input_names, ("obs",))
        self.assertEqual(self.policy.runner.output_names, ("actions",))
        self.assertEqual(self.policy.runner.shape("obs"), (1, 787))
        self.assertEqual(self.policy.runner.shape("actions"), (1, 41))
        self.assertEqual(self.policy.motion.num_frames, 404)
        self.assertEqual(self.policy.motion.object_pos.shape, (404, 3))
        self.assertEqual(self.policy.motion.contact_label.shape, (404, 54))

    def test_action_transform_and_joint_name_mapping(self):
        raw = np.linspace(-100.0, 100.0, 41, dtype=np.float32)
        expected = raw * np.asarray(self.config.action.scale) + np.asarray(self.config.action.offset)
        processed = self.policy.process(raw)
        np.testing.assert_allclose(processed, expected, atol=1e-6)

        action_names = list(self.config.action.body.joint_names) + list(self.config.action.hand.joint_names)
        body, hand = self.policy.split(processed)
        for values, runtime_names in ((body, body_joint_names()), (hand, hand_joint_names())):
            for value, name in zip(values, runtime_names, strict=True):
                self.assertAlmostEqual(float(value), float(processed[action_names.index(name)]), places=6)

    def test_last_raw_action_is_the_only_recurrent_policy_state(self):
        policy = build_policy()
        body = State(np.zeros(29), np.zeros(29), np.zeros(3))
        hand = State(np.zeros(12), np.zeros(12))
        policy.act(policy.get_observation(0, np.zeros(3), IDENTITY, body, hand))
        observation = policy.get_observation(0, np.zeros(3), IDENTITY, body, hand)

        np.testing.assert_allclose(observation[-41:], policy.last_action, atol=1e-6)
        self.assertEqual([name for name in vars(policy) if name.startswith("_last")], ["_last_action"])

    def test_non_finite_action_is_rejected(self):
        policy = build_policy()
        policy.infer = lambda observation: np.full(41, np.nan, dtype=np.float32)
        with self.assertRaises(RuntimeError):
            policy.act(np.zeros(787, dtype=np.float32))

    def test_reference_targets_follow_runtime_joint_orders(self):
        body, hand = self.policy.reference_targets()
        names = list(self.config.observation.joint_names)
        frame = self.policy.motion.joint_pos[0]
        for values, runtime_names in ((body, body_joint_names()), (hand, hand_joint_names())):
            for value, name in zip(values, runtime_names, strict=True):
                self.assertAlmostEqual(float(value), float(frame[names.index(name)]), places=6)


class TestPolicyInputRejection(unittest.TestCase):
    def test_runtime_joint_sets_must_match_the_policy(self):
        renamed_body = body_joint_names()
        renamed_body[0] = "not_a_joint"
        renamed_hand = hand_joint_names()
        renamed_hand[0] = "not_a_joint"
        for overrides in (
            {"body": renamed_body},
            {"hand": hand_joint_names()[:-1]},
            {"hand": renamed_hand},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                build_policy(**overrides)

    def test_mimic_driver_must_come_from_the_hand(self):
        mimic = inspire_cfg().mimic
        mimic.L_index_intermediate_joint.driver = "left_knee_joint"
        with self.assertRaises(ValueError):
            build_policy(mimic=mimic)

    def test_motion_bundle_requires_object_data_and_a_known_clip(self):
        no_object = OmegaConf.create({"file": "assets/motions/woodchair_v02.npz", "name": "sub17_woodchair_000_v02"})
        with self.assertRaisesRegex(KeyError, "object_pos_w"):
            build_policy(motion=no_object)
        with self.assertRaisesRegex(ValueError, "is not in"):
            build_policy(motion=motion_cfg(name="sub16_largebox_missing_v02"))

    def test_model_width_is_checked_during_initialization(self):
        runner = body_hand_runner()
        runner.shapes["obs"] = (1, 786)
        with self.assertRaisesRegex(ValueError, "Expected a 786D observation"):
            build_policy(runner=runner)


class TestDeploymentConfiguration(unittest.TestCase):
    def test_policy_and_motion_assets_form_one_complete_contract(self):
        config = policy_cfg()
        self.assertEqual(config.frequency, 50)
        self.assertEqual(len(config.observation.joint_names), 53)
        self.assertEqual(len(config.action.body.joint_names) + len(config.action.hand.joint_names), 41)
        self.assertNotEqual(list(config.action.body.joint_names), body_joint_names())

        with np.load(REPO_ROOT / str(motion_cfg().file), allow_pickle=False) as motions:
            names = [str(name) for name in motions["motion_names"]]
            self.assertEqual(len(names), 46)
            self.assertTrue(all(name.endswith("_v02") for name in names))
            self.assertEqual([str(name) for name in motions["joint_names"]], list(config.observation.joint_names))
            self.assertEqual(int(np.asarray(motions["fps"]).reshape(-1)[0]), config.frequency)
            frames = int(motions["motion_lengths"].sum())
            for field, width in (("object_pos_w", 3), ("object_quat_w", 4), ("contact_label", 54)):
                self.assertEqual(motions[field].shape, (frames, width))

        self.assertTrue(MODEL_PATH.is_file())
        self.assertTrue(MODEL_PATH.with_suffix(".onnx.data").is_file())

    def test_runtime_roots_compose_the_canonical_policy(self):
        for root in ("run_body_hand", "run_loco_hoi_track"):
            for deployment in ("sim", "real"):
                with self.subTest(root=root, deployment=deployment):
                    cfg = compose_config(deployment, config_name=root)
                    policy = cfg.policy if root == "run_body_hand" else cfg.hoi
                    self.assertEqual(str(policy.policy_file), "assets/models/body_hand_distill/largebox/policy.onnx")
                    self.assertEqual(str(cfg.motion.file), "assets/motions/largebox_v02.npz")
                    self.assertTrue(cfg.env.enable_odometry)


if __name__ == "__main__":
    unittest.main()
