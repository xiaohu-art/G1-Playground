import unittest

import numpy as np
from omegaconf import OmegaConf

from g1_playground.policy.body_hand import BodyHandObservation, JointAssembler, ReferenceMotion
from g1_playground.utils.math import quat_inv, quat_mul, quat_rotate, quat_to_rot6d
from tests.body_hand_helpers import CONFIG_DIR, hand_joint_names, motion_data, policy_data

GRAVITY_W = np.array([0.0, 0.0, -1.0], dtype=np.float32)
BLOCKS = (
    ("future_joint_pos", 265),
    ("future_anchor_pos", 15),
    ("future_anchor_rot6d", 30),
    ("future_object_pos", 15),
    ("future_object_rot6d", 30),
    ("future_contact", 270),
    ("current_object_pos", 3),
    ("current_object_rot6d", 6),
    ("base_ang_vel", 3),
    ("joint_pos_rel", 53),
    ("joint_vel", 53),
    ("projected_gravity", 3),
    ("last_action", 41),
)


def build_motion(**overrides):
    data = motion_data()
    data.update(overrides)
    return ReferenceMotion(
        data["joint_pos"],
        data["anchor_pos_w"],
        data["anchor_quat_w"],
        data["object_pos_w"],
        data["object_quat_w"],
        data["contact_label"],
        policy_data()["observation"]["future_offsets"],
    )


def block_bounds():
    offset = 0
    result = {}
    for name, size in BLOCKS:
        result[name] = (offset, offset + size)
        offset += size
    return result, offset


class TestReferenceMotion(unittest.TestCase):
    def test_future_window_contains_robot_object_and_contact_data(self):
        motion = build_motion()
        last = motion.num_frames - 1
        self.assertEqual(motion.future_indices(last - 2).tolist(), [last - 2, last, last, last, last])
        joint, anchor_pos, anchor_quat, object_pos, object_quat, contact = motion.future(last - 2)
        self.assertEqual(joint.shape, (5, 53))
        self.assertEqual(anchor_pos.shape, object_pos.shape, (5, 3))
        self.assertEqual(anchor_quat.shape, object_quat.shape, (5, 4))
        self.assertEqual(contact.shape, (5, 54))

    def test_alignment_applies_one_rigid_transform_to_robot_and_object(self):
        motion = build_motion()
        raw_object = motion.raw_object_pos.copy()
        robot_object_distance = np.linalg.norm(raw_object - motion.raw_anchor_pos, axis=1)
        motion.align()

        np.testing.assert_allclose(motion.anchor_pos[0, :2], 0.0, atol=1e-6)
        np.testing.assert_allclose(motion.object_pos, motion.origin.align_pos(raw_object), atol=0.0)
        np.testing.assert_allclose(
            np.linalg.norm(motion.object_pos - motion.anchor_pos, axis=1), robot_object_distance, atol=1e-5
        )

    def test_fixed_reference_shapes_are_enforced(self):
        data = motion_data()
        with self.assertRaisesRegex(ValueError, "contact_label"):
            build_motion(contact_label=data["contact_label"][:, :-1])
        with self.assertRaisesRegex(ValueError, "object_pos_w"):
            build_motion(object_pos_w=data["object_pos_w"][:-1])


class TestJointAssembler(unittest.TestCase):
    def setUp(self):
        config = policy_data()
        self.state_names = list(config["observation"]["joint_names"])
        self.body_names = list(OmegaConf.load(CONFIG_DIR / "robot/g1.yaml").dof.joint_names)
        self.inspire = OmegaConf.load(CONFIG_DIR / "robot/inspire.yaml")
        self.hand_names = hand_joint_names()
        self.assembler = JointAssembler(self.state_names, self.body_names, self.hand_names, self.inspire.mimic)

    def test_positions_follow_joint_names_and_mimic_rules(self):
        body = np.arange(29.0)
        hand = np.arange(100.0, 112.0)
        values = self.assembler.positions(body, hand)
        for source, names in ((body, self.body_names), (hand, self.hand_names)):
            for value, name in zip(source, names, strict=True):
                self.assertEqual(float(values[self.state_names.index(name)]), float(value))
        for follower, mimic in self.inspire.mimic.items():
            driver = values[self.state_names.index(str(mimic.driver))]
            self.assertAlmostEqual(
                float(values[self.state_names.index(follower)]),
                float(driver * mimic.multiplier + mimic.offset),
                places=5,
            )

    def test_velocity_mimic_has_no_position_offset(self):
        values = self.assembler.velocities(np.zeros(29), np.full(12, 2.0))
        for follower, mimic in self.inspire.mimic.items():
            driver = values[self.state_names.index(str(mimic.driver))]
            self.assertAlmostEqual(float(values[self.state_names.index(follower)]), float(driver * mimic.multiplier), 5)

    def test_incomplete_joint_partition_is_rejected(self):
        with self.assertRaises(ValueError):
            JointAssembler(self.state_names, self.body_names, self.hand_names[:-1], self.inspire.mimic)


class TestObservationLayout(unittest.TestCase):
    def setUp(self):
        self.config = policy_data()
        self.motion = build_motion()
        self.observation = BodyHandObservation(self.motion, self.config["observation"]["default_joint_pos"], 787)
        self.anchor_pos = np.array([0.1, -0.2, 0.78], dtype=np.float32)
        self.anchor_quat = np.array([0.9239, 0.0, 0.0, 0.3827], dtype=np.float32)

    def build(self, frame=7):
        default = np.asarray(self.config["observation"]["default_joint_pos"], dtype=np.float32)
        return self.observation.build(
            frame,
            self.anchor_pos,
            self.anchor_quat,
            np.full(3, 0.5, dtype=np.float32),
            default + 0.25,
            np.full(53, -0.75, dtype=np.float32),
            np.linspace(-1.0, 1.0, 41, dtype=np.float32),
        )

    def test_every_block_matches_the_787d_policy_layout(self):
        frame = 7
        observation = self.build(frame)
        spans, total = block_bounds()
        inverse = quat_inv(self.anchor_quat)
        joint, anchor_pos, anchor_quat, object_pos, object_quat, contact = self.motion.future(frame)
        expected = {
            "future_joint_pos": joint.reshape(-1),
            "future_anchor_pos": quat_rotate(inverse, anchor_pos - self.anchor_pos).reshape(-1),
            "future_anchor_rot6d": np.concatenate([quat_to_rot6d(q) for q in quat_mul(inverse, anchor_quat)]),
            "future_object_pos": quat_rotate(inverse, object_pos - self.anchor_pos).reshape(-1),
            "future_object_rot6d": np.concatenate([quat_to_rot6d(q) for q in quat_mul(inverse, object_quat)]),
            "future_contact": contact.reshape(-1),
            "current_object_pos": quat_rotate(inverse, self.motion.object_pos[frame] - self.anchor_pos),
            "current_object_rot6d": quat_to_rot6d(quat_mul(inverse, self.motion.object_quat[frame])),
            "base_ang_vel": np.full(3, 0.5),
            "joint_pos_rel": np.full(53, 0.25),
            "joint_vel": np.full(53, -0.75),
            "projected_gravity": quat_rotate(inverse, GRAVITY_W),
            "last_action": np.linspace(-1.0, 1.0, 41),
        }
        self.assertEqual(total, 787)
        self.assertEqual(observation.shape, (787,))
        for name, (start, stop) in spans.items():
            with self.subTest(block=name):
                np.testing.assert_allclose(observation[start:stop], expected[name], atol=1e-5)

    def test_observation_is_invariant_to_reference_rebase(self):
        arguments = (np.zeros(3), np.zeros(53), np.zeros(53), np.zeros(41))
        before = self.observation.build(20, self.anchor_pos, self.anchor_quat, *arguments)
        self.motion.align()
        after = self.observation.build(
            20,
            self.motion.origin.align_pos(self.anchor_pos),
            self.motion.origin.align_quat(self.anchor_quat),
            *arguments,
        )
        np.testing.assert_allclose(before, after, atol=2e-5)

    def test_invalid_observation_is_rejected(self):
        wrong_width = BodyHandObservation(self.motion, self.config["observation"]["default_joint_pos"], 786)
        with self.assertRaisesRegex(ValueError, "Expected a 786D observation"):
            wrong_width.build(
                0,
                np.zeros(3),
                np.array([1.0, 0.0, 0.0, 0.0]),
                np.zeros(3),
                np.zeros(53),
                np.zeros(53),
                np.zeros(41),
            )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.observation.build(
                0,
                np.array([0.0, 0.0, np.nan]),
                np.array([1.0, 0.0, 0.0, 0.0]),
                np.zeros(3),
                np.zeros(53),
                np.zeros(53),
                np.zeros(41),
            )


if __name__ == "__main__":
    unittest.main()
