import logging
import os
import time
from types import SimpleNamespace

import numpy as np

logger = logging.getLogger(__name__)


def recorder(capacity: int, num_dofs: int = 29):
    if capacity <= 0:
        raise ValueError("Recorder capacity must be positive")
    return SimpleNamespace(
        count=0,
        elapsed=np.zeros(capacity),
        dof_pos=np.zeros((capacity, num_dofs), dtype=np.float32),
        dof_vel=np.zeros((capacity, num_dofs), dtype=np.float32),
        base_quat=np.zeros((capacity, 4), dtype=np.float32),
        base_ang_vel=np.zeros((capacity, 3), dtype=np.float32),
        command=np.zeros((capacity, num_dofs), dtype=np.float32),
        base_pos=np.full((capacity, 3), np.nan, dtype=np.float32),
        base_lin_vel=np.full((capacity, 3), np.nan, dtype=np.float32),
        body_height=np.full(capacity, np.nan, dtype=np.float32),
    )


def record(log, elapsed, state, command, odometry) -> None:
    if log is None:
        return
    index = log.count
    if index >= log.elapsed.shape[0]:
        return
    log.elapsed[index] = elapsed
    log.dof_pos[index] = state.dof_pos
    log.dof_vel[index] = state.dof_vel
    log.base_quat[index] = state.base_quat
    log.base_ang_vel[index] = state.base_ang_vel
    log.command[index] = command
    if odometry is not None:
        log.base_pos[index] = odometry.position
        log.base_lin_vel[index] = odometry.velocity
        log.body_height[index] = odometry.body_height
    log.count = index + 1


def save_recording(log, directory: str, config_text: str = "") -> str | None:
    if log is None or log.count == 0:
        return None
    path = os.path.join(directory, time.strftime("state_%Y%m%d-%H%M%S"))
    os.makedirs(path, exist_ok=True)
    fields = {name: value[: log.count] for name, value in vars(log).items() if name != "count"}
    np.savez_compressed(os.path.join(path, "state.npz"), **fields)
    if config_text:
        with open(os.path.join(path, "config.yaml"), "w", encoding="utf-8") as handle:
            handle.write(config_text)
    odometry = float(np.isfinite(log.base_pos[: log.count, 0]).mean())
    height = float(np.isfinite(log.body_height[: log.count]).mean())
    logger.warning(
        "Wrote %d frames to %s (odometry valid %.1f%%, body height valid %.1f%%)",
        log.count,
        path,
        100.0 * odometry,
        100.0 * height,
    )
    return path
