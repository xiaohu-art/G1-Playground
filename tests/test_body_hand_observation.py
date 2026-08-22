import unittest

import numpy as np
from omegaconf import OmegaConf

from g1_playground.policy.body_hand import BodyHandObservation, JointAssembler, ReferenceMotion
from g1_playground.utils.math import get_gravity_orientation, quat_inv, quat_rotate, quat_to_rot6d, yaw_quat
from tests.body_hand_helpers import (
    CONFIG_DIR,
    REPO_ROOT,
    hand_joint_names,
    motion_data,
    policy_data,
    session,
    training_golden,
)

GRAVITY_W = np.array([0.0, 0.0, -1.0], dtype=np.float32)


def build_motion():
    data = motion_data()
    config = policy_data()
    return ReferenceMotion(
        data["joint_pos"],
        data["anchor_pos_w"],
        data["anchor_quat_w"],
        config["observation"]["future_offsets"],
    )


def rotation_matrix(quaternion):
    w, x, y, z = (float(value) for value in quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


class TestReferenceMotion(unittest.TestCase):
    def setUp(self):
        self.motion = build_motion()

    def test_future_offsets_clamp_at_the_clip_end(self):
        last = self.motion.num_frames - 1
        self.assertEqual(self.motion.future_indices(last - 2).tolist(), [last - 2, last, last, last, last])
        self.assertEqual(self.motion.future_indices(0).tolist(), policy_data()["observation"]["future_offsets"])

    def test_alignment_puts_the_chosen_frame_at_the_origin(self):
        raw = self.motion.raw_anchor_pos[0].copy()
        self.motion.align()
        aligned_pos, aligned_quat = self.motion.anchor_pos[0], self.motion.anchor_quat[0]
        np.testing.assert_allclose(aligned_pos[:2], [0.0, 0.0], atol=1e-6)
        self.assertAlmostEqual(float(aligned_pos[2]), float(raw[2]), places=6)
        np.testing.assert_allclose(yaw_quat(aligned_quat), [1.0, 0.0, 0.0, 0.0], atol=1e-5)

    def test_alignment_keeps_the_reference_tilt(self):
        raw = self.motion.raw_anchor_quat[0].copy()
        self.motion.align()
        aligned_quat = self.motion.anchor_quat[0]
        np.testing.assert_allclose(
            quat_rotate(quat_inv(aligned_quat), GRAVITY_W), quat_rotate(quat_inv(raw), GRAVITY_W), atol=1e-5
        )

    def test_alignment_is_a_rigid_transform(self):
        before = self.motion.raw_anchor_pos.copy()
        self.motion.align()
        after = self.motion.anchor_pos
        for index in (1, 40, 200, 400):
            self.assertAlmostEqual(
                float(np.linalg.norm(before[index] - before[0])),
                float(np.linalg.norm(after[index] - after[0])),
                places=4,
            )
        np.testing.assert_allclose(before[:, 2], after[:, 2], atol=1e-5)


class TestJointAssembler(unittest.TestCase):
    def setUp(self):
        self.config = policy_data()
        self.state_names = list(self.config["observation"]["joint_names"])
        self.body = list(OmegaConf.load(CONFIG_DIR / "robot/g1.yaml").dof.joint_names)
        self.inspire = OmegaConf.load(CONFIG_DIR / "robot/inspire.yaml")
        self.hand = hand_joint_names()
        self.assembler = JointAssembler(self.state_names, self.body, self.hand, self.inspire.mimic)

    def test_body_hand_and_mimic_tile_the_state_joint_set(self):
        slots = np.concatenate(
            [
                self.assembler.state_to_body.indices,
                self.assembler.state_to_hand.indices,
                self.assembler.state_to_mimic.indices,
            ]
        )
        self.assertEqual(sorted(slots.tolist()), list(range(len(self.state_names))))

    def test_the_hand_state_slots_match_the_published_mapping(self):
        self.assertEqual(
            self.assembler.state_to_hand.indices.tolist(), [36, 37, 35, 34, 48, 38, 31, 32, 30, 29, 43, 33]
        )

    def test_values_land_on_the_joint_they_are_named_for(self):
        values = self.assembler.positions(np.arange(29.0), np.arange(100.0, 112.0))
        for index, name in enumerate(self.body):
            self.assertEqual(float(values[self.state_names.index(name)]), float(index))
        for index, name in enumerate(self.hand):
            self.assertEqual(float(values[self.state_names.index(name)]), float(100 + index))

    def test_mimic_followers_track_their_driver(self):
        values = self.assembler.positions(np.zeros(29), np.linspace(0.1, 1.2, 12))
        for follower, entry in self.inspire.mimic.items():
            expected = values[self.state_names.index(str(entry.driver))] * float(entry.multiplier) + float(entry.offset)
            self.assertAlmostEqual(float(values[self.state_names.index(follower)]), float(expected), places=5)

    def test_mimic_velocity_drops_the_offset(self):
        speeds = self.assembler.velocities(np.zeros(29), np.full(12, 2.0))
        for follower, entry in self.inspire.mimic.items():
            expected = speeds[self.state_names.index(str(entry.driver))] * float(entry.multiplier)
            self.assertAlmostEqual(float(speeds[self.state_names.index(follower)]), float(expected), places=5)

    def test_the_mimic_driver_lookup_tolerates_a_shared_driver(self):
        drivers = [str(entry.driver) for entry in self.inspire.mimic.values()]
        self.assertLess(len(set(drivers)), len(drivers))
        with self.assertRaises(ValueError):
            self.assembler.hand_to_drivers.scatter_into(np.zeros(12), np.zeros(12))

    def test_a_joint_set_that_does_not_tile_is_rejected(self):
        with self.assertRaises(ValueError):
            JointAssembler(self.state_names, self.body, self.hand[:-1], self.inspire.mimic)


class TestObservationLayout(unittest.TestCase):
    def setUp(self):
        self.config = policy_data()
        self.motion = build_motion()
        self.observation = BodyHandObservation(
            self.motion,
            self.config["observation"]["default_joint_pos"],
            int(session().get_inputs()[0].shape[-1]),
        )

    def test_the_builder_emits_the_model_input_dimension(self):
        built = self.observation.build(
            0,
            np.zeros(3),
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.zeros(3),
            np.zeros(53),
            np.zeros(53),
            np.zeros(41),
        )
        self.assertEqual(built.shape, (463,))

    def test_the_declared_dimension_must_match_the_layout(self):
        observation = BodyHandObservation(self.motion, self.config["observation"]["default_joint_pos"], 462)
        with self.assertRaises(ValueError):
            observation.build(
                0,
                np.zeros(3),
                np.array([1.0, 0.0, 0.0, 0.0]),
                np.zeros(3),
                np.zeros(53),
                np.zeros(53),
                np.zeros(41),
            )

    def test_the_previous_action_block_is_last_and_action_sized(self):
        action = np.arange(41, dtype=np.float32)
        built = self.observation.build(
            0,
            np.zeros(3),
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.zeros(3),
            np.zeros(53),
            np.zeros(53),
            action,
        )
        np.testing.assert_array_equal(built[-41:], action)

    def test_rot6d_is_the_first_two_columns_of_the_rotation_matrix(self):
        rng = np.random.default_rng(3)
        for _ in range(200):
            quaternion = rng.normal(size=4).astype(np.float32)
            quaternion /= np.linalg.norm(quaternion)
            np.testing.assert_allclose(
                quat_to_rot6d(quaternion), rotation_matrix(quaternion)[:, :2].reshape(-1), atol=1e-5
            )

    def test_projected_gravity_matches_the_shared_helper(self):
        rng = np.random.default_rng(4)
        for _ in range(200):
            quaternion = rng.normal(size=4).astype(np.float32)
            quaternion /= np.linalg.norm(quaternion)
            np.testing.assert_allclose(
                quat_rotate(quat_inv(quaternion), GRAVITY_W),
                get_gravity_orientation(quaternion[[1, 2, 3, 0]]),
                atol=1e-5,
            )

    def test_a_wrong_sized_previous_action_is_rejected(self):
        with self.assertRaises(ValueError):
            self.observation.build(
                0, np.zeros(3), np.array([1.0, 0, 0, 0]), np.zeros(3), np.zeros(53), np.zeros(53), np.zeros(29)
            )

    def test_non_finite_inputs_are_refused_rather_than_scrubbed(self):
        with self.assertRaises(ValueError) as caught:
            self.observation.build(
                0,
                np.array([0.0, 0.0, np.nan]),
                np.array([1.0, 0.0, 0.0, 0.0]),
                np.zeros(3),
                np.zeros(53),
                np.full(53, np.inf),
                np.zeros(41),
            )
        self.assertIn("non-finite", str(caught.exception).lower())

    def test_a_non_finite_reference_frame_is_refused_when_used(self):
        data = motion_data()
        broken = np.array(data["anchor_pos_w"], dtype=np.float32)
        broken[10, 0] = np.nan
        motion = ReferenceMotion(data["joint_pos"], broken, data["anchor_quat_w"], [0])
        observation = BodyHandObservation(motion, self.config["observation"]["default_joint_pos"], 215)
        with self.assertRaises(ValueError):
            observation.build(
                10,
                np.zeros(3),
                np.array([1.0, 0.0, 0.0, 0.0]),
                np.zeros(3),
                np.zeros(53),
                np.zeros(53),
                np.zeros(41),
            )

    def test_relative_geometry_survives_the_alignment(self):
        arguments = (np.zeros(3), np.zeros(53), np.zeros(53), np.zeros(41))
        robot_pos = np.array([1.0, -2.0, 0.78], dtype=np.float32)
        robot_quat = np.array([0.9239, 0.0, 0.0, 0.3827], dtype=np.float32)
        before = self.observation.build(20, robot_pos, robot_quat, *arguments)

        self.motion.align()
        origin = self.motion.origin
        after = self.observation.build(20, origin.align_pos(robot_pos), origin.align_quat(robot_quat), *arguments)
        np.testing.assert_allclose(before, after, atol=2e-5)


@unittest.skipUnless(training_golden() is not None, "training bundle with golden_frame.npz is not available")
class TestGoldenParity(unittest.TestCase):
    def setUp(self):
        self.config = policy_data()
        self.golden = training_golden()
        self.observation = BodyHandObservation(
            build_motion(),
            self.config["observation"]["default_joint_pos"],
            int(session().get_inputs()[0].shape[-1]),
        )

    def test_the_reference_blocks_rebuild_from_the_staged_motion(self):
        default = np.asarray(self.config["observation"]["default_joint_pos"], dtype=np.float32)
        built = self.observation.build(
            int(self.golden["motion_time_step"]),
            self.golden["robot_anchor_pos_w"],
            self.golden["robot_anchor_quat_w"],
            self.golden["slice/robot_proprio.base_ang_vel"],
            self.golden["slice/robot_proprio.joint_pos"] + default,
            self.golden["slice/robot_proprio.joint_vel"],
            self.golden["slice/robot_proprio.actions"],
        )
        expected = np.asarray(self.golden["observation_463"], dtype=np.float32)
        np.testing.assert_allclose(built[:419], expected[:419], atol=1e-6)
        np.testing.assert_allclose(built[422:], expected[422:], atol=1e-6)
        self.assertLessEqual(float(np.abs(built[419:422] - expected[419:422]).max()), 0.05)

    def test_the_golden_gravity_differs_only_by_training_noise(self):
        clean = quat_rotate(quat_inv(np.asarray(self.golden["robot_anchor_quat_w"], dtype=np.float32)), GRAVITY_W)
        self.assertAlmostEqual(float(np.linalg.norm(clean)), 1.0, places=5)
        difference = np.asarray(self.golden["slice/robot_proprio.projected_gravity"]) - clean
        self.assertTrue(bool(np.all(np.abs(difference) <= 0.05)))


class TestDeploymentAssets(unittest.TestCase):
    def test_only_the_graph_and_its_weights_are_deployed(self):
        model_dir = REPO_ROOT / "assets/models/body_hand_distill/largebox"
        self.assertEqual(sorted(path.name for path in model_dir.iterdir()), ["policy.onnx", "policy.onnx.data"])

    def test_the_contract_and_golden_are_not_deployment_assets(self):
        for name in ("deployment_contract.yaml", "golden_frame.npz", "policy.pt"):
            with self.subTest(name=name):
                self.assertEqual(list((REPO_ROOT / "assets").rglob(name)), [])

    def test_the_old_bundle_directory_is_gone(self):
        self.assertFalse((REPO_ROOT / "assets/models/largebox_body_hand").exists())


if __name__ == "__main__":
    unittest.main()
