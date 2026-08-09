"""Quaternion helpers. Convention: wxyz."""

import numpy as np

GRAVITY_DIR = np.array([0.0, 0.0, -1.0], dtype=np.float32)


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
