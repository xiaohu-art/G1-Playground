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


def command_remap(command, new_range, old_range=None):
    command = np.asarray(command, dtype=np.float32)
    old_min, old_mid, old_max = [-1.0, 0.0, 1.0] if old_range is None else old_range
    new_min, new_mid, new_max = new_range
    if abs((command - old_mid) / (old_max - old_min)) < 0.02:
        return np.full_like(command, new_mid, dtype=np.float32)

    scale_neg = (new_mid - new_min) / (old_mid - old_min)
    scale_pos = (new_max - new_mid) / (old_max - old_mid)
    return np.where(
        command < old_mid,
        new_mid + (command - old_mid) * scale_neg,
        new_mid + (command - old_mid) * scale_pos,
    )
