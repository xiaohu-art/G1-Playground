from dataclasses import dataclass
from threading import Lock

import mujoco
import numpy as np


def snapshot(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values).copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class MujocoState:
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    joint_torque: np.ndarray
    base_quaternion_wxyz: np.ndarray
    base_angular_velocity: np.ndarray


class ElasticSupport:
    """Continuously adjustable spring-damper support for a floating-base G1."""

    def __init__(
        self,
        model,
        *,
        body_name: str = "torso_link",
        anchor=(0.0, 0.0, 3.0),
        length: float = 0.0,
        stiffness: float = 200.0,
        damping: float = 100.0,
        scale: float = 0.0,
    ):
        body_id = mujoco.mj_name2id(  # pyright: ignore[reportAttributeAccessIssue]
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_name,  # pyright: ignore[reportAttributeAccessIssue]
        )
        if body_id < 0:
            raise ValueError(f"MuJoCo model has no body named {body_name!r}")
        self.body_id = body_id
        self.anchor = np.asarray(anchor, dtype=np.float64)
        if self.anchor.shape != (3,) or not np.all(np.isfinite(self.anchor)):
            raise ValueError("Elastic support anchor must contain three finite values")
        for name, value in (("length", length), ("stiffness", stiffness), ("damping", damping)):
            if isinstance(value, bool) or not isinstance(value, int | float) or not np.isfinite(value) or value < 0:
                raise ValueError(f"Elastic support {name} must be finite and non-negative")
        self.length = float(length)
        self.stiffness = float(stiffness)
        self.damping = float(damping)
        self.scale = scale

    @property
    def scale(self) -> float:
        return self._scale

    @scale.setter
    def scale(self, value: float) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not np.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError("Elastic support scale must be finite and between 0 and 1")
        self._scale = float(value)

    def apply(self, data) -> np.ndarray:
        if self.scale == 0.0:
            return np.zeros(3, dtype=np.float64)

        displacement = self.anchor - np.asarray(data.qpos[:3])
        distance = float(np.linalg.norm(displacement))
        if distance <= np.finfo(np.float64).eps:
            return np.zeros(3, dtype=np.float64)

        direction = displacement / distance
        velocity_along_band = float(np.dot(data.qvel[:3], direction))
        magnitude = self.stiffness * (distance - self.length) - self.damping * velocity_along_band
        force = self.scale * magnitude * direction
        data.xfrc_applied[self.body_id, :3] += force
        return force

    def remove(self, data, force: np.ndarray) -> None:
        data.xfrc_applied[self.body_id, :3] -= force


class G1MujocoBackend:
    """Pure MuJoCo physics for the floating-base 29-DoF G1 model."""

    def __init__(
        self,
        xml_path: str,
        timestep: float = 0.001,
        *,
        elastic_support_scale: float = 0.0,
        expected_actuators: int = 29,
    ):
        if (
            isinstance(timestep, bool)
            or not isinstance(timestep, int | float)
            or not np.isfinite(timestep)
            or timestep <= 0
        ):
            raise ValueError("MuJoCo backend timestep must be finite and positive")
        self.timestep = float(timestep)
        self.model = mujoco.MjModel.from_xml_path(xml_path)  # pyright: ignore[reportAttributeAccessIssue]
        self.model.opt.timestep = self.timestep
        if self.model.nu != expected_actuators:
            raise ValueError(
                f"G1 MuJoCo backend requires exactly {expected_actuators} actuators, "
                f"model has {self.model.nu}"
            )
        self.data = mujoco.MjData(self.model)  # pyright: ignore[reportAttributeAccessIssue]
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)  # pyright: ignore[reportAttributeAccessIssue]
        self._data_lock = Lock()
        self.elastic_support = ElasticSupport(self.model, scale=elastic_support_scale)
        self.step_mujoco()

    def step_mujoco(self) -> None:
        support_force = self.elastic_support.apply(self.data)
        try:
            mujoco.mj_step(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
        finally:
            self.elastic_support.remove(self.data, support_force)

    def read_unlocked(self) -> MujocoState:
        num_joints = self.model.nu
        return MujocoState(
            joint_pos=snapshot(self.data.qpos[-num_joints:]),
            joint_vel=snapshot(self.data.qvel[-num_joints:]),
            joint_torque=snapshot(self.data.actuator_force),
            base_quaternion_wxyz=snapshot(self.data.qpos[3:7]),
            base_angular_velocity=snapshot(self.data.qvel[3:6]),
        )

    def read(self) -> MujocoState:
        with self._data_lock:
            return self.read_unlocked()

    def copy_data_to(self, destination) -> None:
        """Copy one coherent physics snapshot into another MjData."""

        with self._data_lock:
            mujoco.mj_copyData(destination, self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]

    def step(self, torque, support_scale: float) -> MujocoState:
        torque = np.asarray(torque)
        if torque.shape != (self.model.nu,) or not np.all(np.isfinite(torque)):
            raise ValueError(f"G1 MuJoCo backend requires {self.model.nu} finite actuator torques")
        with self._data_lock:
            self.elastic_support.scale = support_scale
            self.data.ctrl[:] = torque
            self.step_mujoco()
            return self.read_unlocked()
