from pathlib import Path

import mujoco
import numpy as np


def compile_mujoco_scene(
    robot_mjcf: str | Path,
    object_mjcf: str | Path | None = None,
    *,
    object_position: np.ndarray | None = None,
    object_quaternion: np.ndarray | None = None,
) -> mujoco.MjModel:
    has_position = object_position is not None
    has_quaternion = object_quaternion is not None
    if has_position != has_quaternion:
        raise ValueError("Object position and quaternion must be provided together")
    if object_mjcf is None and has_position:
        raise ValueError("An object pose requires an object MJCF")

    scene = mujoco.MjSpec.from_file(str(Path(robot_mjcf).resolve()))
    if object_mjcf is not None:
        object_spec = mujoco.MjSpec.from_file(str(Path(object_mjcf).resolve()))
        if has_position:
            position = np.asarray(object_position, dtype=np.float64)
            quaternion = np.asarray(object_quaternion, dtype=np.float64)
            if position.shape != (3,) or quaternion.shape != (4,):
                raise ValueError("Object pose must contain position [3] and quaternion wxyz [4]")
            quaternion_norm = float(np.linalg.norm(quaternion))
            if not np.all(np.isfinite(position)) or not np.all(np.isfinite(quaternion)) or quaternion_norm < 1e-8:
                raise ValueError("Object pose must be finite with a nonzero quaternion")
            object_body = object_spec.body("hoi_object")
            if object_body is None:
                raise ValueError("HOI object MJCF must define body 'hoi_object'")
            object_body.pos = position
            object_body.quat = quaternion / quaternion_norm
        mount = scene.worldbody.add_frame()
        scene.attach(object_spec, frame=mount, prefix="", suffix="")
    model = scene.compile()
    if object_mjcf is not None:
        object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hoi_object")
        freejoint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hoi_object_freejoint")
        if (
            object_id < 0
            or freejoint_id < 0
            or model.jnt_type[freejoint_id] != mujoco.mjtJoint.mjJNT_FREE
            or model.jnt_bodyid[freejoint_id] != object_id
        ):
            raise ValueError("HOI object MJCF must define body 'hoi_object' with freejoint 'hoi_object_freejoint'")
        qpos_address = model.jnt_qposadr[freejoint_id]
        if model.nkey:
            model.key_qpos[:, qpos_address : qpos_address + 7] = model.qpos0[qpos_address : qpos_address + 7]
    return model
