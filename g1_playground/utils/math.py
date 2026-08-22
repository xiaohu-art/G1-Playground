import numpy as np


def get_gravity_orientation(quaternion):
    qx, qy, qz, qw = quaternion
    return np.array(
        [
            2 * (-qz * qx + qw * qy),
            -2 * (qz * qy + qw * qx),
            1 - 2 * (qw * qw + qz * qz),
        ]
    )


def is_upright(quaternion, max_tilt: float = 1.0) -> bool:
    gravity = get_gravity_orientation(quaternion)
    tilt = float(np.arccos(np.clip(-gravity[2], -1.0, 1.0)))
    return tilt <= max_tilt


def quat_inv(quaternion: np.ndarray) -> np.ndarray:
    inverse = np.asarray(quaternion, dtype=np.float32).copy()
    inverse[..., 1:] = -inverse[..., 1:]
    return inverse


def quat_mul(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = first[..., 0], first[..., 1], first[..., 2], first[..., 3]
    w2, x2, y2, z2 = second[..., 0], second[..., 1], second[..., 2], second[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    ).astype(np.float32)


def quat_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    padded = np.zeros((*vector.shape[:-1], 4), dtype=np.float32)
    padded[..., 1:4] = vector
    return quat_mul(quat_mul(quaternion, padded), quat_inv(quaternion))[..., 1:4]


def yaw_quat(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(quaternion[i]) for i in range(4))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    result = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=np.float32)
    return result / max(float(np.linalg.norm(result)), 1e-8)


def quat_slerp(first: np.ndarray, second: np.ndarray, weight: float) -> np.ndarray:
    first = np.asarray(first, dtype=np.float32).reshape(4)
    second = np.asarray(second, dtype=np.float32).reshape(4)
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second, dot = -second, -dot
    if dot > 0.9995:
        blended = first + weight * (second - first)
        return (blended / max(float(np.linalg.norm(blended)), 1e-8)).astype(np.float32)
    theta = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    sin_theta = float(np.sin(theta))
    blended = np.sin((1.0 - weight) * theta) * first + np.sin(weight * theta) * second
    return (blended / sin_theta).astype(np.float32)


def quat_angular_velocity(previous: np.ndarray, current: np.ndarray, dt: float) -> np.ndarray:
    delta = quat_mul(np.asarray(current, dtype=np.float32), quat_inv(np.asarray(previous, dtype=np.float32)))
    if delta[0] < 0.0:
        delta = -delta
    axis = delta[1:4]
    norm = float(np.linalg.norm(axis))
    if norm < 1e-8 or dt <= 0.0:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * float(np.arccos(np.clip(float(delta[0]), -1.0, 1.0)))
    return (axis / norm * (angle / dt)).astype(np.float32)


def quat_to_rot6d(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    return np.array(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
        ],
        dtype=np.float32,
    )


class TransformAlignment:
    def __init__(self, quat=None, pos=None, yaw_only: bool = False, xy_only: bool = False):
        self.yaw_only = bool(yaw_only)
        self.xy_only = bool(xy_only)
        self.set_base(quat, pos)

    def set_base(self, quat=None, pos=None) -> None:
        if quat is None:
            base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            base_quat = np.asarray(quat, dtype=np.float32).reshape(4)
            base_quat = yaw_quat(base_quat) if self.yaw_only else base_quat.copy()
        self.base_quat = base_quat

        base_pos = np.zeros(3, dtype=np.float32) if pos is None else np.asarray(pos, dtype=np.float32).reshape(3).copy()
        if self.xy_only:
            base_pos[2] = 0.0
        self.base_pos = base_pos

    def align_quat(self, quat: np.ndarray) -> np.ndarray:
        return quat_mul(quat_inv(self.base_quat), np.asarray(quat, dtype=np.float32))

    def align_xyz(self, xyz: np.ndarray) -> np.ndarray:
        return quat_rotate(quat_inv(self.base_quat), np.asarray(xyz, dtype=np.float32))

    def align_pos(self, pos: np.ndarray) -> np.ndarray:
        return self.align_xyz(np.asarray(pos, dtype=np.float32) - self.base_pos)

    def align_transform(self, quat: np.ndarray, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.align_quat(quat), self.align_pos(pos)
