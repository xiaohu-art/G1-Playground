"""Quaternion helpers. Convention: wxyz."""

import math

import numpy as np

GRAVITY_DIR = np.array([0.0, 0.0, -1.0], dtype=np.float32)


def quat_mul(q1, q2) -> np.ndarray:
    """Hamilton product of two wxyz quaternions."""
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def quat_inv(q) -> np.ndarray:
    """Conjugate (inverse for unit quaternions), wxyz."""
    out = np.asarray(q, dtype=np.float32).copy()
    out[1:] = -out[1:]
    return out


def quat_rotate(q, v) -> np.ndarray:
    """Rotate vector ``v`` by unit quaternion ``q`` (wxyz)."""
    v_quat = np.zeros(4, dtype=np.float32)
    v_quat[1:4] = np.asarray(v, dtype=np.float32)
    return quat_mul(quat_mul(np.asarray(q, dtype=np.float32), v_quat), quat_inv(q))[1:4]


def quat_to_rot6d(q) -> np.ndarray:
    """6D rotation representation: the first two columns of R, row-major."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.array(
        [
            1 - 2 * (y * y + z * z),  # r00
            2 * (x * y - w * z),  # r01
            2 * (x * y + w * z),  # r10
            1 - 2 * (x * x + z * z),  # r11
            2 * (x * z - w * y),  # r20
            2 * (y * z + w * x),  # r21
        ],
        dtype=np.float32,
    )


def yaw_quat(q) -> np.ndarray:
    """Yaw-only component of a wxyz quaternion."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    out = np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], dtype=np.float32)
    return out / max(float(np.linalg.norm(out)), 1e-8)


def quat_rotate_inverse(q_wxyz, v) -> np.ndarray:
    """Rotate vector ``v`` by the inverse of unit quaternion ``q`` (wxyz)."""
    w, x, y, z = q_wxyz
    qc = np.array([w, -x, -y, -z])
    return np.array(
        [
            v[0] * (qc[0] ** 2 + qc[1] ** 2 - qc[2] ** 2 - qc[3] ** 2)
            + v[1] * 2 * (qc[1] * qc[2] - qc[0] * qc[3])
            + v[2] * 2 * (qc[1] * qc[3] + qc[0] * qc[2]),
            v[0] * 2 * (qc[1] * qc[2] + qc[0] * qc[3])
            + v[1] * (qc[0] ** 2 - qc[1] ** 2 + qc[2] ** 2 - qc[3] ** 2)
            + v[2] * 2 * (qc[2] * qc[3] - qc[0] * qc[1]),
            v[0] * 2 * (qc[1] * qc[3] - qc[0] * qc[2])
            + v[1] * 2 * (qc[2] * qc[3] + qc[0] * qc[1])
            + v[2] * (qc[0] ** 2 - qc[1] ** 2 - qc[2] ** 2 + qc[3] ** 2),
        ],
        dtype=np.float64,
    )


def projected_gravity(q_wxyz) -> np.ndarray:
    """World -z axis expressed in the base frame: R(q)^T · (0, 0, -1)."""
    return quat_rotate_inverse(q_wxyz, GRAVITY_DIR)
