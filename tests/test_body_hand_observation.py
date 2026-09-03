import unittest

import numpy as np

from g1_playground.policy.body_hand import ReferenceMotion
from g1_playground.policy.body_hand.depth import DepthBodyHandObservation, preprocess_depth
from g1_playground.utils.math import quat_inv, quat_mul, quat_rotate, quat_to_rot6d
from tests.body_hand_helpers import motion_data, policy_data

GRAVITY_W = np.array([0.0, 0.0, -1.0], dtype=np.float32)
REFERENCE_BLOCKS = (
    ("future_joint_pos", 265),
    ("future_anchor_pos", 15),
    ("future_anchor_rot6d", 30),
    ("future_object_pos", 15),
    ("future_object_rot6d", 30),
    ("future_contact", 270),
)


class TestBodyHandObservation(unittest.TestCase):
    def test_depth_layout_matches_the_9994d_training_contract(self):
        config = policy_data()
        data = motion_data()
        motion = ReferenceMotion(
            data["joint_pos"],
            data["anchor_pos_w"],
            data["anchor_quat_w"],
            data["object_pos_w"],
            data["object_quat_w"],
            data["contact_label"],
            config["observation"]["future_offsets"],
        )
        builder = DepthBodyHandObservation(
            motion,
            config["observation"]["default_joint_pos"],
            9994,
            height=72,
            width=128,
            min_distance=0.25,
            max_distance=3.0,
        )
        frame = 7
        anchor_pos = np.array([0.1, -0.2, 0.78], dtype=np.float32)
        anchor_quat = np.array([0.9239, 0.0, 0.0, 0.3827], dtype=np.float32)
        base_ang_vel = np.full(3, 0.5, dtype=np.float32)
        joint_pos = np.asarray(config["observation"]["default_joint_pos"], dtype=np.float32) + 0.25
        joint_vel = np.full(53, -0.75, dtype=np.float32)
        last_action = np.linspace(-1.0, 1.0, 41, dtype=np.float32)
        depth = np.linspace(0.3, 2.9, 72 * 128, dtype=np.float32).reshape(72, 128)

        observation = builder.build(
            frame,
            anchor_pos,
            anchor_quat,
            base_ang_vel,
            joint_pos,
            joint_vel,
            last_action,
            depth,
        )

        inverse = quat_inv(anchor_quat)
        joint, future_anchor_pos, future_anchor_quat, object_pos, object_quat, contact = motion.future(frame)
        expected_reference = {
            "future_joint_pos": joint.reshape(-1),
            "future_anchor_pos": quat_rotate(inverse, future_anchor_pos - anchor_pos).reshape(-1),
            "future_anchor_rot6d": np.concatenate([quat_to_rot6d(q) for q in quat_mul(inverse, future_anchor_quat)]),
            "future_object_pos": quat_rotate(inverse, object_pos - anchor_pos).reshape(-1),
            "future_object_rot6d": np.concatenate([quat_to_rot6d(q) for q in quat_mul(inverse, object_quat)]),
            "future_contact": contact.reshape(-1),
        }
        offset = 0
        for name, size in REFERENCE_BLOCKS:
            np.testing.assert_allclose(observation[offset : offset + size], expected_reference[name], atol=1e-5)
            offset += size

        np.testing.assert_allclose(
            observation[offset : offset + 72 * 128],
            preprocess_depth(depth, height=72, width=128, min_distance=0.25, max_distance=3.0),
        )
        expected_proprioception = np.concatenate(
            [base_ang_vel, np.full(53, 0.25), joint_vel, quat_rotate(inverse, GRAVITY_W), last_action]
        )
        np.testing.assert_allclose(observation[-153:], expected_proprioception, atol=1e-5)
        self.assertEqual(observation.shape, (9994,))


if __name__ == "__main__":
    unittest.main()
