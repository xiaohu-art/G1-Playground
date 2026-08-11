import mujoco
import numpy as np


def _quaternion_matrix(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _arrow_matrix(direction: np.ndarray) -> np.ndarray:
    reference = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(reference, direction)) > 0.99:
        reference = np.array([1.0, 0.0, 0.0])
    x_axis = np.cross(reference, direction)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(direction, x_axis)
    return np.column_stack((x_axis, y_axis, direction))


class MujocoVisualizer:
    def __init__(self, viewer):
        self.viewer = viewer

    def draw_arrow(self, origin, root_quat, vec_local, color, scale=1.0, horizontal_only=False, id=0):
        vec_local = np.asarray(vec_local, dtype=np.float64)
        rotation = _quaternion_matrix(np.asarray(root_quat, dtype=np.float64))
        if horizontal_only:
            yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
            cosine, sine = np.cos(yaw), np.sin(yaw)
            rotation = np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
        vec_world = rotation @ vec_local

        length = np.linalg.norm(vec_world)
        if length > 1e-6:
            direction = vec_world / length
            matrix = _arrow_matrix(direction)
            scaled_length = length * scale
        else:
            matrix = np.eye(3)
            scaled_length = 0.0

        self.viewer.add_marker(
            pos=np.asarray(origin),
            mat=matrix,
            size=np.array([0.02, 0.02, scaled_length]),
            rgba=np.asarray(color),
            type=mujoco.mjtGeom.mjGEOM_ARROW,  # pyright: ignore[reportAttributeAccessIssue]
            id=3000 + id,
        )
