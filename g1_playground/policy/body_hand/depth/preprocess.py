import numpy as np

from g1_playground.camera.depth import resize_depth


def preprocess_depth(
    depth_m,
    *,
    height: int,
    width: int,
    min_distance: float,
    max_distance: float,
) -> np.ndarray:
    if (
        not np.isfinite(min_distance)
        or not np.isfinite(max_distance)
        or min_distance < 0
        or max_distance <= min_distance
    ):
        raise ValueError("Depth distance bounds must be finite and increasing")

    depth = resize_depth(depth_m, height, width)
    valid = np.isfinite(depth) & (depth > min_distance) & (depth < max_distance)
    normalized = np.zeros((height, width), dtype=np.float32)
    normalized[valid] = depth[valid] / np.float32(max_distance)
    return normalized.reshape(-1)
