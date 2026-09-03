import numpy as np

from g1_playground.utils.dof import DoFAdapter
from g1_playground.utils.math import TransformAlignment, quat_inv, quat_mul, quat_rotate, quat_to_rot6d

GRAVITY_W = np.array([0.0, 0.0, -1.0], dtype=np.float32)


def reference_features(motion, frame, anchor_pos, anchor_quat) -> tuple[np.ndarray, np.ndarray]:
    anchor_pos = np.asarray(anchor_pos, dtype=np.float32).reshape(3)
    anchor_quat = np.asarray(anchor_quat, dtype=np.float32).reshape(4)
    inverse = quat_inv(anchor_quat)
    joint_pos, future_anchor_pos, future_anchor_quat, object_pos, object_quat, contact = motion.future(frame)
    return (
        np.concatenate(
            [
                joint_pos.reshape(-1),
                quat_rotate(inverse, future_anchor_pos - anchor_pos).reshape(-1),
                np.concatenate([quat_to_rot6d(quaternion) for quaternion in quat_mul(inverse, future_anchor_quat)]),
                quat_rotate(inverse, object_pos - anchor_pos).reshape(-1),
                np.concatenate([quat_to_rot6d(quaternion) for quaternion in quat_mul(inverse, object_quat)]),
                contact.reshape(-1),
            ],
            dtype=np.float32,
        ),
        inverse,
    )


def proprioception_features(default_joint_pos, inverse, base_ang_vel, joint_pos, joint_vel, last_action) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(base_ang_vel, dtype=np.float32).reshape(-1),
            np.asarray(joint_pos, dtype=np.float32).reshape(-1) - default_joint_pos,
            np.asarray(joint_vel, dtype=np.float32).reshape(-1),
            quat_rotate(inverse, GRAVITY_W),
            np.asarray(last_action, dtype=np.float32).reshape(-1),
        ],
        dtype=np.float32,
    )


def observation_vector(blocks, observation_dim: int) -> np.ndarray:
    observation = np.concatenate(blocks, dtype=np.float32)
    if observation.shape != (observation_dim,):
        raise ValueError(f"Expected a {observation_dim}D observation, got {observation.shape}")
    if not np.all(np.isfinite(observation)):
        raise ValueError("Body-hand observation contains non-finite values")
    return observation


class ReferenceMotion:
    def __init__(
        self,
        joint_pos,
        anchor_pos_w,
        anchor_quat_w,
        object_pos_w,
        object_quat_w,
        contact_label,
        future_offsets,
    ):
        self.joint_pos = np.asarray(joint_pos, dtype=np.float32)
        self.raw_anchor_pos = np.asarray(anchor_pos_w, dtype=np.float32)
        self.raw_anchor_quat = np.asarray(anchor_quat_w, dtype=np.float32)
        self.raw_object_pos = np.asarray(object_pos_w, dtype=np.float32)
        self.raw_object_quat = np.asarray(object_quat_w, dtype=np.float32)
        self.contact_label = np.asarray(contact_label, dtype=np.float32)
        self.offsets = np.asarray(future_offsets, dtype=np.int64).reshape(-1)
        self.num_frames = int(self.joint_pos.shape[0])

        for name, values, width in (
            ("joint_pos", self.joint_pos, 53),
            ("anchor_pos_w", self.raw_anchor_pos, 3),
            ("anchor_quat_w", self.raw_anchor_quat, 4),
            ("object_pos_w", self.raw_object_pos, 3),
            ("object_quat_w", self.raw_object_quat, 4),
            ("contact_label", self.contact_label, 54),
        ):
            expected = (self.num_frames, width)
            if values.shape != expected:
                raise ValueError(f"Reference {name} has shape {values.shape}, expected {expected}")
        if self.num_frames == 0:
            raise ValueError("Reference motion has no frames")
        if self.offsets.size == 0:
            raise ValueError("Reference motion has no future offsets")

        self.origin = TransformAlignment(yaw_only=True, xy_only=True)
        self.anchor_pos = self.raw_anchor_pos.copy()
        self.anchor_quat = self.raw_anchor_quat.copy()
        self.object_pos = self.raw_object_pos.copy()
        self.object_quat = self.raw_object_quat.copy()

    def align(self) -> None:
        self.origin.set_base(quat=self.raw_anchor_quat[0], pos=self.raw_anchor_pos[0])
        self.anchor_pos = self.origin.align_pos(self.raw_anchor_pos)
        self.anchor_quat = self.origin.align_quat(self.raw_anchor_quat)
        self.object_pos = self.origin.align_pos(self.raw_object_pos)
        self.object_quat = self.origin.align_quat(self.raw_object_quat)

    def future_indices(self, frame: int) -> np.ndarray:
        return np.minimum(frame + self.offsets, self.num_frames - 1)

    def future(self, frame: int):
        indices = self.future_indices(frame)
        return (
            self.joint_pos[indices],
            self.anchor_pos[indices],
            self.anchor_quat[indices],
            self.object_pos[indices],
            self.object_quat[indices],
            self.contact_label[indices],
        )


class JointAssembler:
    """Assemble the policy state vector from the runtime body and Inspire hand readings."""

    def __init__(self, state_joint_names, body_joint_names, hand_joint_names, mimic_cfg):
        state_joint_names = tuple(state_joint_names)
        hand_joint_names = tuple(hand_joint_names)
        followers = list(mimic_cfg)
        drivers = [str(mimic_cfg[name].driver) for name in followers]
        self.num_joints = len(state_joint_names)

        self.state_to_body = DoFAdapter(state_joint_names, body_joint_names)
        self.state_to_hand = DoFAdapter(state_joint_names, hand_joint_names)
        self.state_to_mimic = DoFAdapter(state_joint_names, followers)
        self.hand_to_drivers = DoFAdapter(hand_joint_names, drivers)
        self.multiplier = np.asarray([float(mimic_cfg[name].multiplier) for name in followers], dtype=np.float32)
        self.offset = np.asarray([float(mimic_cfg[name].offset) for name in followers], dtype=np.float32)

        covered = np.concatenate([self.state_to_body.indices, self.state_to_hand.indices, self.state_to_mimic.indices])
        if sorted(covered.tolist()) != list(range(self.num_joints)):
            raise ValueError("Body, hand and mimic joints do not tile the policy state joints")

    def positions(self, body_values, hand_values) -> np.ndarray:
        return self._fill(body_values, hand_values, self.offset)

    def velocities(self, body_values, hand_values) -> np.ndarray:
        return self._fill(body_values, hand_values, 0.0)

    def _fill(self, body_values, hand_values, mimic_offset) -> np.ndarray:
        body_values = np.asarray(body_values, dtype=np.float32).reshape(-1)
        hand_values = np.asarray(hand_values, dtype=np.float32).reshape(-1)
        values = np.empty(self.num_joints, dtype=np.float32)
        self.state_to_body.scatter_into(body_values, values)
        self.state_to_hand.scatter_into(hand_values, values)
        followers = self.hand_to_drivers.fit(hand_values) * self.multiplier + mimic_offset
        self.state_to_mimic.scatter_into(followers, values)
        return values
