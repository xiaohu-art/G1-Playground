from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DepthFrame:
    depth_m: np.ndarray
    timestamp: float
    sequence: int


def snapshot_depth(depth_m) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32).copy()
    if depth.ndim != 2:
        raise ValueError(f"Depth image must be two-dimensional, got shape {depth.shape}")
    depth.flags.writeable = False
    return depth


def resize_depth(depth_m, height: int = 72, width: int = 128) -> np.ndarray:
    """Resize a depth image with nearest-neighbor sampling."""

    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2 or depth.shape[0] == 0 or depth.shape[1] == 0:
        raise ValueError(f"Depth image must have a non-empty two-dimensional shape, got {depth.shape}")
    if height <= 0 or width <= 0:
        raise ValueError("Depth output height and width must be positive")

    rows = np.minimum(((np.arange(height) + 0.5) * depth.shape[0] / height).astype(np.intp), depth.shape[0] - 1)
    columns = np.minimum(((np.arange(width) + 0.5) * depth.shape[1] / width).astype(np.intp), depth.shape[1] - 1)
    return depth[np.ix_(rows, columns)]
