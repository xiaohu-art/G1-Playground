import numpy as np

from g1_playground.utils.dof import DoFAdapter
from g1_playground.utils.math import TransformAlignment, quat_inv, quat_mul, quat_rotate, quat_to_rot6d

GRAVITY_W = np.array([0.0, 0.0, -1.0], dtype=np.float32)


class ReferenceMotion:
    def __init__(
        self,
        joint_pos,
        joint_vel,
        anchor_pos_w,
        anchor_quat_w,
        anchor_lin_vel_w,
        future_offsets,
        terminal_hold_frames=0,
    ):
        joint_pos = np.asarray(joint_pos, dtype=np.float32)
        joint_vel = np.asarray(joint_vel, dtype=np.float32)
        anchor_pos_w = np.asarray(anchor_pos_w, dtype=np.float32)
        anchor_quat_w = np.asarray(anchor_quat_w, dtype=np.float32)
        anchor_lin_vel_w = np.asarray(anchor_lin_vel_w, dtype=np.float32)
        self.source_num_frames = int(joint_pos.shape[0])
        self.terminal_hold_frames = int(terminal_hold_frames)
        if self.source_num_frames < 1 or self.terminal_hold_frames < 0:
            raise ValueError("Reference motion needs at least one source frame and a non-negative terminal hold")
        if self.terminal_hold_frames:
            count = self.terminal_hold_frames
            joint_pos = np.concatenate([joint_pos, np.repeat(joint_pos[-1:], count, axis=0)])
            joint_vel = np.concatenate([joint_vel, np.zeros((count, joint_vel.shape[1]), dtype=np.float32)])
            anchor_pos_w = np.concatenate([anchor_pos_w, np.repeat(anchor_pos_w[-1:], count, axis=0)])
            anchor_quat_w = np.concatenate([anchor_quat_w, np.repeat(anchor_quat_w[-1:], count, axis=0)])
            anchor_lin_vel_w = np.concatenate([anchor_lin_vel_w, np.zeros((count, 3), dtype=np.float32)])

        self.joint_pos = joint_pos
        self.joint_vel = joint_vel
        self.raw_anchor_pos = anchor_pos_w
        self.raw_anchor_quat = anchor_quat_w
        self.raw_anchor_lin_vel = anchor_lin_vel_w
        self.offsets = np.asarray(future_offsets, dtype=np.int64).reshape(-1)
        self.num_frames = int(self.joint_pos.shape[0])

        self.origin = TransformAlignment(yaw_only=True, xy_only=True)
        self.anchor_pos = self.raw_anchor_pos.copy()
        self.anchor_quat = self.raw_anchor_quat.copy()
        self.anchor_lin_vel = self.raw_anchor_lin_vel.copy()

    def align(self) -> None:
        self.origin.set_base(quat=self.raw_anchor_quat[0], pos=self.raw_anchor_pos[0])
        self.anchor_pos = self.origin.align_pos(self.raw_anchor_pos)
        self.anchor_quat = self.origin.align_quat(self.raw_anchor_quat)
        self.anchor_lin_vel = self.origin.align_xyz(self.raw_anchor_lin_vel)

    def future_indices(self, frame: int) -> np.ndarray:
        return np.minimum(frame + self.offsets, self.num_frames - 1)

    def future(self, frame: int):
        indices = self.future_indices(frame)
        return (
            self.joint_pos[indices],
            self.joint_vel[indices],
            self.anchor_pos[indices],
            self.anchor_quat[indices],
            self.anchor_lin_vel[indices],
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


class BodyHandObservation:
    def __init__(self, motion: ReferenceMotion, default_joint_pos, observation_dim: int):
        self.motion = motion
        self.default_joint_pos = np.asarray(default_joint_pos, dtype=np.float32).reshape(-1)
        self.observation_dim = int(observation_dim)

    def build(self, frame, anchor_quat, base_ang_vel, joint_pos, joint_vel, last_action) -> np.ndarray:
        anchor_quat = np.asarray(anchor_quat, dtype=np.float32).reshape(4)
        future_joint_pos, future_joint_vel, _, future_anchor_quat, future_anchor_lin_vel = self.motion.future(frame)
        inverse = quat_inv(anchor_quat)
        relative_quat = quat_mul(inverse, future_anchor_quat)
        anchor_lin_vel_b = quat_rotate(inverse, future_anchor_lin_vel)

        observation = np.concatenate(
            [
                future_joint_pos.reshape(-1),
                future_joint_vel.reshape(-1),
                anchor_lin_vel_b.reshape(-1),
                np.concatenate([quat_to_rot6d(quaternion) for quaternion in relative_quat]),
                np.asarray(base_ang_vel, dtype=np.float32).reshape(-1),
                np.asarray(joint_pos, dtype=np.float32).reshape(-1) - self.default_joint_pos,
                np.asarray(joint_vel, dtype=np.float32).reshape(-1),
                quat_rotate(inverse, GRAVITY_W),
                np.asarray(last_action, dtype=np.float32).reshape(-1),
            ],
            dtype=np.float32,
        )
        if observation.shape != (self.observation_dim,):
            raise ValueError(f"Expected a {self.observation_dim}D observation, got {observation.shape}")
        if not np.all(np.isfinite(observation)):
            raise ValueError("Body-hand observation contains non-finite values")
        return observation
