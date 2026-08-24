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
    parity_trace,
    policy_data,
    session,
)

GRAVITY_W = np.array([0.0, 0.0, -1.0], dtype=np.float32)

# Observation layout: [deployable_ref_motion_body | robot_proprio] with 53 joints and 5 future offsets.
FUTURE_JOINT_POS = slice(0, 265)
FUTURE_JOINT_VEL = slice(265, 530)
ANCHOR_LIN_VEL_B = slice(530, 545)
ANCHOR_ORI_B = slice(545, 575)
BASE_ANG_VEL = slice(575, 578)
JOINT_POS_REL = slice(578, 631)
JOINT_VEL = slice(631, 684)
PROJECTED_GRAVITY = slice(684, 687)
LAST_ACTION = slice(687, 728)
OBSERVATION_DIM = 728


def build_motion():
    data = motion_data()
    config = policy_data()
    return ReferenceMotion(
        data["joint_pos"],
        data["joint_vel"],
        data["anchor_pos_w"],
        data["anchor_quat_w"],
        data["anchor_lin_vel_w"],
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
        for index in (1, 40, 200, 290):
            self.assertAlmostEqual(
                float(np.linalg.norm(before[index] - before[0])),
                float(np.linalg.norm(after[index] - after[0])),
                places=4,
            )
        np.testing.assert_allclose(before[:, 2], after[:, 2], atol=1e-5)

    def test_alignment_rotates_the_anchor_velocity_without_changing_its_norm(self):
        raw_vel = self.motion.raw_anchor_lin_vel.copy()
        self.motion.align()
        aligned_vel = self.motion.anchor_lin_vel
        np.testing.assert_allclose(np.linalg.norm(aligned_vel, axis=1), np.linalg.norm(raw_vel, axis=1), atol=1e-5)
        origin_quat = self.motion.origin.base_quat
        np.testing.assert_allclose(quat_rotate(origin_quat, aligned_vel), raw_vel, atol=1e-5)


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

    def build(self, frame=0, quat=None, ang_vel=None, pos=None, vel=None, action=None):
        return self.observation.build(
            frame,
            np.array([1.0, 0.0, 0.0, 0.0]) if quat is None else quat,
            np.zeros(3) if ang_vel is None else ang_vel,
            np.zeros(53) if pos is None else pos,
            np.zeros(53) if vel is None else vel,
            np.zeros(41) if action is None else action,
        )

    def test_the_builder_emits_the_model_input_dimension(self):
        self.assertEqual(self.build().shape, (OBSERVATION_DIM,))
        self.assertEqual(int(session().get_inputs()[0].shape[-1]), OBSERVATION_DIM)

    def test_the_declared_dimension_must_match_the_layout(self):
        observation = BodyHandObservation(self.motion, self.config["observation"]["default_joint_pos"], 727)
        with self.assertRaises(ValueError):
            observation.build(0, np.array([1.0, 0, 0, 0]), np.zeros(3), np.zeros(53), np.zeros(53), np.zeros(41))

    def test_the_segments_follow_the_training_layout(self):
        built = self.build(
            frame=0,
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
            ang_vel=np.full(3, 0.11, dtype=np.float32),
            pos=np.full(53, 0.22, dtype=np.float32),
            vel=np.full(53, 0.33, dtype=np.float32),
            action=np.full(41, 0.44, dtype=np.float32),
        )
        default = np.asarray(self.config["observation"]["default_joint_pos"], dtype=np.float32)
        future = self.motion.future_indices(0)
        np.testing.assert_allclose(built[FUTURE_JOINT_POS], self.motion.joint_pos[future].reshape(-1), atol=1e-6)
        np.testing.assert_allclose(built[FUTURE_JOINT_VEL], self.motion.joint_vel[future].reshape(-1), atol=1e-6)
        np.testing.assert_allclose(built[ANCHOR_LIN_VEL_B], self.motion.anchor_lin_vel[future].reshape(-1), atol=1e-6)
        np.testing.assert_allclose(built[BASE_ANG_VEL], 0.11, atol=1e-6)
        np.testing.assert_allclose(built[JOINT_POS_REL], 0.22 - default, atol=1e-6)
        np.testing.assert_allclose(built[JOINT_VEL], 0.33, atol=1e-6)
        np.testing.assert_allclose(built[PROJECTED_GRAVITY], GRAVITY_W, atol=1e-6)
        np.testing.assert_allclose(built[LAST_ACTION], 0.44, atol=1e-6)

    def test_the_anchor_velocity_is_expressed_in_the_robot_anchor_frame(self):
        data = motion_data()
        frame = 10
        robot_quat = np.asarray(data["anchor_quat_w"][frame], dtype=np.float32)
        built = self.observation.build(
            frame, robot_quat, np.zeros(3), np.zeros(53), np.zeros(53), np.zeros(41)
        )
        future = self.motion.future_indices(frame)
        expected = quat_rotate(quat_inv(robot_quat), np.asarray(data["anchor_lin_vel_w"][future], dtype=np.float32))
        np.testing.assert_allclose(built[ANCHOR_LIN_VEL_B], expected.reshape(-1), atol=1e-5)

    def test_the_anchor_orientation_matches_the_first_two_rotation_columns(self):
        data = motion_data()
        frame = 7
        robot_quat = np.asarray(data["anchor_quat_w"][frame], dtype=np.float32)
        built = self.observation.build(
            frame, robot_quat, np.zeros(3), np.zeros(53), np.zeros(53), np.zeros(41)
        )
        future = self.motion.future_indices(frame)
        future_quat = np.asarray(data["anchor_quat_w"][future], dtype=np.float32)
        from g1_playground.utils.math import quat_mul

        relative = quat_mul(quat_inv(robot_quat), future_quat)
        expected = np.concatenate([rotation_matrix(q)[:, :2].reshape(-1) for q in relative])
        np.testing.assert_allclose(built[ANCHOR_ORI_B], expected, atol=1e-5)

    def test_the_previous_action_block_is_last_and_action_sized(self):
        action = np.arange(41, dtype=np.float32)
        built = self.build(action=action)
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
                0, np.array([1.0, 0, 0, 0]), np.zeros(3), np.zeros(53), np.zeros(53), np.zeros(29)
            )

    def test_non_finite_inputs_are_refused_rather_than_scrubbed(self):
        with self.assertRaises(ValueError) as caught:
            self.observation.build(
                0,
                np.array([1.0, 0.0, 0.0, 0.0]),
                np.zeros(3),
                np.zeros(53),
                np.full(53, np.inf),
                np.zeros(41),
            )
        self.assertIn("non-finite", str(caught.exception).lower())

    def test_a_non_finite_reference_frame_is_refused_when_used(self):
        data = motion_data()
        broken = np.array(data["anchor_lin_vel_w"], dtype=np.float32)
        broken[10, 0] = np.nan
        motion = ReferenceMotion(
            data["joint_pos"],
            data["joint_vel"],
            data["anchor_pos_w"],
            data["anchor_quat_w"],
            broken,
            [0],
        )
        observation = BodyHandObservation(motion, self.config["observation"]["default_joint_pos"], OBSERVATION_DIM)
        with self.assertRaises(ValueError):
            observation.build(
                10, np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), np.zeros(53), np.zeros(53), np.zeros(41)
            )

    def test_relative_geometry_survives_the_alignment(self):
        robot_quat = np.array([0.9239, 0.0, 0.0, 0.3827], dtype=np.float32)
        arguments = (np.zeros(3), np.zeros(53), np.zeros(53), np.zeros(41))
        before = self.observation.build(20, robot_quat, *arguments)

        self.motion.align()
        origin = self.motion.origin
        after = self.observation.build(20, origin.align_quat(robot_quat), *arguments)
        np.testing.assert_allclose(before, after, atol=2e-5)


@unittest.skipUnless(parity_trace() is not None, "Isaac parity trace is not available")
class TestParityTrace(unittest.TestCase):
    """Rebuild the network input from the recorded robot state and compare against the Isaac observation.

    The trace is produced by ``scripts/rsl_rl/play.py --parity_trace parity_trace.npz`` on the training
    machine; only the rows that play the deployed clip are checked. The reference-motion segments must
    match exactly, while the corrupted proprio terms are bounded by the training noise ranges.
    """

    def setUp(self):
        self.config = policy_data()
        self.trace = parity_trace()
        deployed = motion_data()
        clip_index = int(np.asarray(deployed["clip_index"]).reshape(-1)[0]) if "clip_index" in deployed.files else 0
        keep = np.flatnonzero(np.asarray(self.trace["clip_ids"]) == clip_index)
        self.assertGreater(len(keep), 1, "the parity trace holds no rows for the deployed clip")
        self.rows = keep
        self.motion = build_motion()
        self.observation = BodyHandObservation(
            self.motion,
            self.config["observation"]["default_joint_pos"],
            int(session().get_inputs()[0].shape[-1]),
        )

    def rebuild(self, row, last_action):
        return self.observation.build(
            int(self.trace["frame_indices"][row]),
            np.asarray(self.trace["base_quat_wxyz"][row], dtype=np.float32),
            np.asarray(self.trace["base_ang_vel"][row], dtype=np.float32),
            np.asarray(self.trace["joint_pos"][row], dtype=np.float32),
            np.asarray(self.trace["joint_vel"][row], dtype=np.float32),
            last_action,
        )

    def test_the_trace_uses_the_deployment_joint_orders(self):
        np.testing.assert_array_equal(
            [str(n) for n in self.trace["joint_names"]], list(self.config["observation"]["joint_names"])
        )
        action_names = list(self.config["action"]["body"]["joint_names"]) + list(
            self.config["action"]["hand"]["joint_names"]
        )
        np.testing.assert_array_equal([str(n) for n in self.trace["action_joint_names"]], action_names)

    def test_the_reference_segments_rebuild_exactly(self):
        for row in self.rows:
            recorded = np.asarray(self.trace["observations"][row], dtype=np.float32)
            # The last-action content is checked separately; seed it from the recording here.
            built = self.rebuild(row, recorded[LAST_ACTION])
            np.testing.assert_allclose(built[: ANCHOR_ORI_B.stop], recorded[: ANCHOR_ORI_B.stop], atol=1e-5)

    def test_the_corrupted_proprio_segments_stay_within_the_training_noise(self):
        for row in self.rows:
            recorded = np.asarray(self.trace["observations"][row], dtype=np.float32)
            built = self.rebuild(row, recorded[LAST_ACTION])
            self.assertLessEqual(float(np.abs(built[BASE_ANG_VEL] - recorded[BASE_ANG_VEL]).max()), 0.2 + 1e-5)
            self.assertLessEqual(float(np.abs(built[JOINT_POS_REL] - recorded[JOINT_POS_REL]).max()), 0.01 + 1e-5)
            self.assertLessEqual(float(np.abs(built[JOINT_VEL] - recorded[JOINT_VEL]).max()), 0.5 + 1e-5)
            self.assertLessEqual(
                float(np.abs(built[PROJECTED_GRAVITY] - recorded[PROJECTED_GRAVITY]).max()), 0.05 + 1e-5
            )

    def test_the_last_action_chains_across_consecutive_frames(self):
        actions = np.asarray(self.trace["raw_actions"], dtype=np.float32)
        frames = np.asarray(self.trace["frame_indices"])
        for previous, row in zip(self.rows[:-1], self.rows[1:], strict=True):
            if frames[row] != frames[previous] + 1:
                continue
            recorded = np.asarray(self.trace["observations"][row], dtype=np.float32)
            np.testing.assert_allclose(recorded[LAST_ACTION], actions[previous], atol=1e-6)
            built = self.rebuild(row, actions[previous])
            np.testing.assert_allclose(built[LAST_ACTION], recorded[LAST_ACTION], atol=1e-6)

    def test_the_deployed_onnx_reproduces_the_recorded_actions(self):
        graph = session()
        observations = np.asarray(self.trace["observations"][self.rows], dtype=np.float32)
        expected = np.asarray(self.trace["raw_actions"][self.rows], dtype=np.float32)
        predicted = graph.run(
            [graph.get_outputs()[0].name], {graph.get_inputs()[0].name: observations}
        )[0]
        np.testing.assert_allclose(predicted.reshape(expected.shape), expected, atol=1e-4)


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
