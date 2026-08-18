import mujoco
import numpy as np

from g1_playground.utils.math import TransformAlignment, quat_inv, quat_rotate, quat_to_rot6d

GRAVITY_W = np.array([0.0, 0.0, -1.0], dtype=np.float32)


class TrackObservation:
    def __init__(self, xml_path: str, anchor_body_name: str, default_pos: np.ndarray, num_joints: int):
        self.num_joints = int(num_joints)
        self.default_pos = np.asarray(default_pos, dtype=np.float32).reshape(-1)
        if self.default_pos.shape != (self.num_joints,):
            raise ValueError("Track observation default_pos must contain one value per joint")

        self.model = mujoco.MjModel.from_xml_path(xml_path)  # pyright: ignore[reportAttributeAccessIssue]
        self.data = mujoco.MjData(self.model)  # pyright: ignore[reportAttributeAccessIssue]
        anchor_id = mujoco.mj_name2id(  # pyright: ignore[reportAttributeAccessIssue]
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,  # pyright: ignore[reportAttributeAccessIssue]
            anchor_body_name,
        )
        if anchor_id < 0:
            raise ValueError(f"MuJoCo model has no body named {anchor_body_name!r}")
        self.anchor_id = anchor_id
        self.total_obs_size = self.num_joints * 5 + 6 + 3 + 13

        self._reference_joint_pos = self.default_pos.copy()
        self._reference_anchor_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._reference_anchor_height = np.zeros(1, dtype=np.float32)
        self._reference_gravity = GRAVITY_W.copy()
        self.heading = TransformAlignment(yaw_only=True)
        self._zero_joints = np.zeros(self.num_joints, dtype=np.float32)
        self._zero_vector = np.zeros(3, dtype=np.float32)

    def anchor_pose(self, base_pos, base_quat, joint_pos) -> tuple[np.ndarray, np.ndarray]:
        self.data.qpos[:] = 0.0
        self.data.qpos[0:3] = np.asarray(base_pos, dtype=np.float64).reshape(3)
        quaternion = np.asarray(base_quat, dtype=np.float64).reshape(4)
        self.data.qpos[3:7] = quaternion / max(float(np.linalg.norm(quaternion)), 1e-8)
        count = min(len(joint_pos), self.model.nq - 7)
        self.data.qpos[7 : 7 + count] = np.asarray(joint_pos, dtype=np.float64)[:count]
        mujoco.mj_kinematics(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
        return (
            np.asarray(self.data.xpos[self.anchor_id], dtype=np.float32).copy(),
            np.asarray(self.data.xquat[self.anchor_id], dtype=np.float32).copy(),
        )

    def set_standing_reference(self, base_quat_wxyz: np.ndarray, root_height: float) -> None:
        self.heading.set_base(np.asarray(base_quat_wxyz, dtype=np.float32).reshape(4))
        anchor_pos, anchor_quat = self.anchor_pose(
            np.array([0.0, 0.0, root_height], dtype=np.float32), self.heading.base_quat, self.default_pos
        )
        self._reference_anchor_quat = anchor_quat
        self._reference_anchor_height = anchor_pos[2:3].astype(np.float32)
        self._reference_gravity = quat_rotate(quat_inv(anchor_quat), GRAVITY_W)

    def build(self, joint_pos, joint_vel, base_quat_wxyz, base_ang_vel, last_action) -> np.ndarray:
        joint_pos = np.asarray(joint_pos, dtype=np.float32).reshape(-1)[: self.num_joints]
        joint_vel = np.asarray(joint_vel, dtype=np.float32).reshape(-1)[: self.num_joints]
        base_quat = np.asarray(base_quat_wxyz, dtype=np.float32).reshape(-1)
        base_ang_vel = np.asarray(base_ang_vel, dtype=np.float32).reshape(-1)
        last_action = np.asarray(last_action, dtype=np.float32).reshape(-1)
        if base_quat.shape != (4,) or base_ang_vel.shape != (3,):
            raise ValueError("Track observation requires a wxyz quaternion and a 3D angular velocity")
        if last_action.shape != (self.num_joints,):
            raise ValueError(f"Track observation last_action must be {self.num_joints}D")

        _, robot_anchor_quat = self.anchor_pose(self._zero_vector, base_quat, joint_pos)
        robot_frame = TransformAlignment(robot_anchor_quat)
        relative = robot_frame.align_quat(self._reference_anchor_quat)

        observation = np.concatenate(
            [
                self._reference_joint_pos,
                self._zero_joints,
                quat_to_rot6d(relative),
                base_ang_vel,
                joint_pos - self.default_pos,
                joint_vel,
                last_action,
                quat_rotate(quat_inv(base_quat), GRAVITY_W),
                self._zero_vector,
                self._zero_vector,
                self._reference_gravity,
                self._reference_anchor_height,
            ],
            dtype=np.float32,
        )
        if observation.shape[0] != self.total_obs_size:
            raise ValueError(f"Expected a {self.total_obs_size}D observation, got {observation.shape[0]}")
        return np.where(np.isfinite(observation), observation, np.float32(0.0))
