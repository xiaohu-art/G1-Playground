import argparse
import glob
import importlib
import logging
import os
import time

import mujoco
import numpy as np
from omegaconf import OmegaConf

from g1_playground.utils import resolve_repo_path
from g1_playground.utils.logger import setup_logger
from g1_playground.utils.math import quat_inv, quat_mul, quat_rotate, yaw_quat

logger = logging.getLogger("g1_playground")
RENDER_HZ = 60.0
FALLBACK_HEIGHT = 0.79
DEFAULT_XML = "assets/robots/g1/g1_29dof_rev_1_0.xml"


def latest_recording(directory: str = "logs") -> str:
    candidates = sorted(path for path in glob.glob(os.path.join(directory, "state_*")) if os.path.isdir(path))
    if not candidates:
        raise SystemExit(f"No state_* recording found under {directory}/")
    return candidates[-1]


def base_trajectory(state):
    quaternion = np.asarray(state["base_quat"], dtype=np.float32)[:, [3, 0, 1, 2]]
    frames = len(quaternion)
    heading = quat_inv(yaw_quat(quaternion[0]))
    aligned = np.stack([quat_mul(heading, quaternion[index]) for index in range(frames)])

    position = np.zeros((frames, 3), dtype=np.float64)
    position[:, 2] = FALLBACK_HEIGHT
    if "body_height" in state.files:
        valid = np.isfinite(state["body_height"])
        position[valid, 2] = state["body_height"][valid]

    raw = np.asarray(state["base_pos"], dtype=np.float64)
    if np.isfinite(raw[:, 0]).any():
        first = int(np.argmax(np.isfinite(raw[:, 0])))
        offset = np.where(np.isfinite(raw), raw - raw[first], 0.0)
        offset[:, 2] = 0.0
        position[:, :2] = np.stack([quat_rotate(heading, row.astype(np.float32)) for row in offset])[:, :2]
    return position, aligned


def run(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Kinematic MuJoCo replay of a recorded G1 trajectory")
    parser.add_argument("directory", nargs="?", help="a logs/state_<timestamp> directory; defaults to the newest")
    args = parser.parse_args(argv)

    setup_logger()
    directory = args.directory or latest_recording()
    state = np.load(os.path.join(directory, "state.npz"))
    config_path = os.path.join(directory, "config.yaml")
    cfg = OmegaConf.load(config_path) if os.path.exists(config_path) else None

    model = mujoco.MjModel.from_xml_path(  # pyright: ignore[reportAttributeAccessIssue]
        resolve_repo_path(cfg.robot.xml if cfg is not None else DEFAULT_XML)
    )
    data = mujoco.MjData(model)  # pyright: ignore[reportAttributeAccessIssue]

    joints = np.asarray(state["dof_pos"], dtype=np.float64)
    if joints.shape[1] != model.nu:
        raise SystemExit(f"The model has {model.nu} actuators but the recording stores {joints.shape[1]} joints")
    elapsed = np.asarray(state["elapsed"], dtype=np.float64)
    position, quaternion = base_trajectory(state)

    logger.warning("Replaying %d frames (%.1f s) from %s", len(elapsed), elapsed[-1] - elapsed[0], directory)
    if cfg is not None:
        logger.warning("Recorded on domain %s via %s", cfg.env.domain_id, cfg.env.net_if)

    viewer_module = importlib.import_module("mujoco.viewer")
    with viewer_module.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        with viewer.lock():
            viewer.cam.distance = 3.0
            viewer.cam.elevation = -10.0
            viewer.cam.azimuth = 135.0
        started = time.monotonic()
        for index in range(len(elapsed)):
            if not viewer.is_running():
                break
            data.qpos[:] = 0.0
            data.qpos[0:3] = position[index]
            data.qpos[3:7] = quaternion[index]
            data.qpos[7 : 7 + model.nu] = joints[index]
            mujoco.mj_forward(model, data)  # pyright: ignore[reportAttributeAccessIssue]
            with viewer.lock():
                viewer.cam.lookat[:] = data.qpos[0:3]
            viewer.sync()
            remaining = started + elapsed[index] - elapsed[0] - time.monotonic()
            if remaining > 0:
                time.sleep(min(remaining, 1.0 / RENDER_HZ))


if __name__ == "__main__":
    run()
