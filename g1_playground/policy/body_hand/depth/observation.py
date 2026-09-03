import numpy as np

from g1_playground.policy.body_hand.observation import (
    observation_vector,
    proprioception_features,
    reference_features,
)

from .preprocess import preprocess_depth


class DepthBodyHandObservation:
    def __init__(
        self,
        motion,
        default_joint_pos,
        observation_dim: int,
        *,
        height: int,
        width: int,
        min_distance: float,
        max_distance: float,
    ):
        self.motion = motion
        self.default_joint_pos = np.asarray(default_joint_pos, dtype=np.float32).reshape(-1)
        self.observation_dim = int(observation_dim)
        self.height = int(height)
        self.width = int(width)
        self.min_distance = float(min_distance)
        self.max_distance = float(max_distance)

    def build(
        self,
        frame,
        anchor_pos,
        anchor_quat,
        base_ang_vel,
        joint_pos,
        joint_vel,
        last_action,
        depth_m,
    ) -> np.ndarray:
        reference, inverse = reference_features(self.motion, frame, anchor_pos, anchor_quat)
        depth = preprocess_depth(
            depth_m,
            height=self.height,
            width=self.width,
            min_distance=self.min_distance,
            max_distance=self.max_distance,
        )
        proprioception = proprioception_features(
            self.default_joint_pos,
            inverse,
            base_ang_vel,
            joint_pos,
            joint_vel,
            last_action,
        )
        return observation_vector((reference, depth, proprioception), self.observation_dim)
