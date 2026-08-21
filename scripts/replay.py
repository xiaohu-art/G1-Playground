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
AXIS_LENGTH = 0.5
AXIS_WIDTH = 0.012
PATH_WIDTH = 0.006
PATH_LIFT = 0.01
BEFORE_RGBA = (0.55, 0.55, 0.55, 1.0)
AFTER_RGBA = (0.1, 0.85, 0.3, 1.0)
CAPTURE_RGBA = (1.0, 0.85, 0.0, 1.0)
CAPTURE_RADIUS = 0.05
WORLD_AXES = (
    ((AXIS_LENGTH, 0.0, 0.0), (1.0, 0.0, 0.0, 1.0)),
    ((0.0, AXIS_LENGTH, 0.0), (0.0, 1.0, 0.0, 1.0)),
    ((0.0, 0.0, AXIS_LENGTH), (0.0, 0.0, 1.0, 1.0)),
)


def draw_world_frame(scene) -> None:
    origin = np.zeros(3)
    for tip, rgba in WORLD_AXES:
        geom = add_geom(scene, mujoco.mjtGeom.mjGEOM_ARROW, rgba, np.zeros(3), np.zeros(3))  # pyright: ignore[reportAttributeAccessIssue]
        if geom is None:
            return
        mujoco.mjv_connector(  # pyright: ignore[reportAttributeAccessIssue]
            geom,
            int(mujoco.mjtGeom.mjGEOM_ARROW),  # pyright: ignore[reportAttributeAccessIssue]
            AXIS_WIDTH,
            origin,
            np.asarray(tip, dtype=np.float64),
        )


def add_geom(scene, kind, rgba, size, pos):
    if scene.ngeom >= scene.maxgeom:
        return None
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(  # pyright: ignore[reportAttributeAccessIssue]
        geom,
        kind,
        np.asarray(size, dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.eye(3).flatten(),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1
    return geom


def draw_path(scene, position, active) -> None:
    ground = position.copy()
    ground[:, 2] = PATH_LIFT
    budget = max(scene.maxgeom - scene.ngeom - 2, 0)
    stride = max(len(ground) // max(budget, 1), 1)
    samples = list(range(0, len(ground), stride))
    for start, end in zip(samples, samples[1:], strict=False):
        rgba = AFTER_RGBA if (active is not None and bool(active[end])) else BEFORE_RGBA
        geom = add_geom(scene, mujoco.mjtGeom.mjGEOM_LINE, rgba, np.zeros(3), np.zeros(3))  # pyright: ignore[reportAttributeAccessIssue]
        if geom is None:
            return
        mujoco.mjv_connector(  # pyright: ignore[reportAttributeAccessIssue]
            geom,
            int(mujoco.mjtGeom.mjGEOM_LINE),  # pyright: ignore[reportAttributeAccessIssue]
            PATH_WIDTH,
            ground[start],
            ground[end],
        )
    if active is not None and active.any():
        capture = int(np.argmax(active))
        add_geom(
            scene,
            mujoco.mjtGeom.mjGEOM_SPHERE,  # pyright: ignore[reportAttributeAccessIssue]
            CAPTURE_RGBA,
            (CAPTURE_RADIUS, CAPTURE_RADIUS, CAPTURE_RADIUS),
            ground[capture],
        )


def latest_recording(directory: str = "logs") -> str:
    candidates = sorted(path for path in glob.glob(os.path.join(directory, "state_*")) if os.path.isdir(path))
    if not candidates:
        raise SystemExit(f"No state_* recording found under {directory}/")
    return candidates[-1]


def rebase_origin(state):
    if "rebase_active" not in state.files:
        return None
    active = np.asarray(state["rebase_active"], dtype=bool)
    if not active.any():
        return None
    first = int(np.argmax(active))
    origin = np.asarray(state["rebase_origin_pos"], dtype=np.float64)[first]
    heading = np.asarray(state["rebase_origin_quat"], dtype=np.float32)[first][[3, 0, 1, 2]]
    return origin, heading, active


def base_trajectory(state):
    quaternion = np.asarray(state["base_quat"], dtype=np.float32)[:, [3, 0, 1, 2]]
    frames = len(quaternion)
    stored = np.asarray(state["base_pos"], dtype=np.float64)

    height = np.full(frames, FALLBACK_HEIGHT, dtype=np.float64)
    if "body_height" in state.files:
        valid = np.isfinite(state["body_height"])
        height[valid] = state["body_height"][valid]

    if rebase_origin(state) is not None:
        position = np.where(np.isfinite(stored), stored, 0.0)
        position[:, 2] = np.where(np.isfinite(stored[:, 2]), stored[:, 2], height)
        return position, quaternion

    heading = quat_inv(yaw_quat(quaternion[0]))
    orientation = np.stack([quat_mul(heading, quaternion[index]) for index in range(frames)])
    position = np.zeros((frames, 3), dtype=np.float64)
    position[:, 2] = height
    if np.isfinite(stored[:, 0]).any():
        first = int(np.argmax(np.isfinite(stored[:, 0])))
        offset = np.where(np.isfinite(stored), stored - stored[first], 0.0)
        offset[:, 2] = 0.0
        position[:, :2] = np.stack([quat_rotate(heading, row.astype(np.float32)) for row in offset])[:, :2]
    return position, orientation


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
        rebase = rebase_origin(state)
        active = rebase[2] if rebase is not None else None
        viewer.user_scn.ngeom = 0
        draw_world_frame(viewer.user_scn)
        draw_path(viewer.user_scn, position, active)
        anchor = "the rebase origin" if rebase is not None else "the first recorded frame"
        logger.warning("World frame at %s: x red, y green, z blue, %.2f m", anchor, AXIS_LENGTH)
        if active is not None:
            capture = int(np.argmax(active))
            logger.warning(
                "Path: grey before the rebase, green after; yellow sphere marks the capture at t=%.2f s (frame %d)",
                elapsed[capture],
                capture,
            )
        else:
            logger.warning("Path drawn in grey; this recording was never rebased")
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
