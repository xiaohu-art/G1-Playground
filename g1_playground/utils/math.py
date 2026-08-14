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
