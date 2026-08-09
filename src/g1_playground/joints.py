from collections.abc import Sequence

import numpy as np

# Depth-first traversal of the kinematic tree
JOINT_DFS_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# Breadth-first traversal of the kinematic tree
JOINT_BFS_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]

assert sorted(JOINT_DFS_NAMES) == sorted(JOINT_BFS_NAMES)


class JointAdapter:
    """Copy joints present in both name lists from src order into tar order.

        obs_adapter    = JointAdapter(JOINT_DFS_NAMES, policy_joint_names)
        dof_pos    = obs_adapter.fit(q)

        action_adapter = JointAdapter(policy_joint_names, JOINT_DFS_NAMES)
        q_target   = action_adapter.fit(q, template=DEFAULT_POS)
    """

    def __init__(self, src_joint_names: Sequence[str], tar_joint_names: Sequence[str]):
        self.src_joint_names = list(src_joint_names)
        self.tar_joint_names = list(tar_joint_names)

        self.src_len = len(self.src_joint_names)
        self.tar_len = len(self.tar_joint_names)

        self.src_indices: list[int] = []
        self.tar_indices: list[int] = []

        for i, name in enumerate(self.src_joint_names):
            if name in self.tar_joint_names:
                self.src_indices.append(i)
                self.tar_indices.append(self.tar_joint_names.index(name))

        assert len(self.src_indices) > 0, "Error fitting src and tar joint names, please check the config."

    def fit(self, data, dim: int = -1, template=None) -> np.ndarray:
        if type(data) is not np.ndarray:
            data = np.asarray(data)

        assert data.shape[dim] == self.src_len, (
            f"Data shape {data.shape} does not match src length {self.src_len} at dim {dim}"
        )

        new_shape = list(data.shape)
        new_shape[dim] = self.tar_len

        if template is None:
            new_data = np.zeros(new_shape, dtype=data.dtype)
        else:
            if type(template) is not np.ndarray:
                template = np.asarray(template, dtype=data.dtype)
            new_data = template.copy()
            assert new_data.shape == tuple(new_shape), (
                f"Template shape {new_data.shape} does not match target shape {tuple(new_shape)}"
            )

        if dim == -1:
            new_data[..., self.tar_indices] = data[..., self.src_indices]
        else:
            tar_index: list = [slice(None)] * len(new_shape)
            src_index: list = [slice(None)] * len(data.shape)
            tar_index[dim] = self.tar_indices
            src_index[dim] = self.src_indices
            new_data[tuple(tar_index)] = data[tuple(src_index)]
        return new_data
