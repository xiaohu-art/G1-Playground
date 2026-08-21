import logging
import time
from dataclasses import dataclass

import numpy as np
from omegaconf import DictConfig
from unitree_cpp import G1DdsControlEndpoint

from g1_playground.utils.math import TransformAlignment

logger = logging.getLogger(__name__)


def snapshot(values) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32).copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class G1State:
    dof_pos: np.ndarray
    dof_vel: np.ndarray
    base_quat: np.ndarray
    base_ang_vel: np.ndarray


@dataclass(frozen=True)
class G1Odometry:
    position: np.ndarray
    raw_position: np.ndarray
    velocity: np.ndarray
    body_height: float


class G1Env:
    """G1 state and command boundary backed by Unitree DDS."""

    def __init__(
        self,
        *,
        control_dt: float,
        domain_id: int,
        net_if: str,
        lowcmd_topic: str,
        lowstate_topic: str,
        motion_switcher_required: bool,
        dof_cfg: DictConfig,
        enable_odometry: bool = False,
        sport_state_topic: str = "rt/odommodestate",
    ):

        joint_names = list(dof_cfg.joint_names)
        self.num_dofs = len(joint_names)
        if self.num_dofs != 29 or len(set(joint_names)) != self.num_dofs:
            raise ValueError("G1Env requires 29 unique joints")
        stiffness = np.asarray(dof_cfg.stiffness)
        damping = np.asarray(dof_cfg.damping)
        if stiffness.shape != (self.num_dofs,) or damping.shape != (self.num_dofs,):
            raise ValueError("G1Env requires one stiffness and damping value per joint")

        self.control_dt = float(control_dt)
        self.remote_controller_handler = None
        self.born_place_align = False
        self.base_align = TransformAlignment(yaw_only=True, xy_only=True)
        self.control_endpoint = G1DdsControlEndpoint(
            {
                "domain_id": domain_id,
                "net_if": net_if,
                "msg_type": "hg",
                "control_mode": "position",
                "hand_type": "NONE",
                "lowcmd_topic": lowcmd_topic,
                "lowstate_topic": lowstate_topic,
                "enable_odometry": enable_odometry,
                "sport_state_topic": sport_state_topic,
                "control_dt": self.control_dt,
                "num_dofs": self.num_dofs,
                "stiffness": stiffness,
                "damping": damping,
                "motion_switcher_required": motion_switcher_required,
            }
        )

    def activate_commands(self) -> bool:
        return self.control_endpoint.activate_commands()

    def self_check(self) -> None:
        for _ in range(30):
            if self.control_endpoint.self_check():
                logger.info("G1Env self check passed")
                return
            time.sleep(0.1)
        raise RuntimeError("G1Env self check failed: no valid LowState received")

    def set_born_place(self, base_quat_xyzw, base_position) -> bool:
        if self.born_place_align:
            logger.warning("Born place is already set; ignoring repeated request")
            return False

        base_quat_wxyz = np.asarray(base_quat_xyzw, dtype=np.float32)[[3, 0, 1, 2]]
        position = np.asarray(base_position, dtype=np.float32).copy()
        position[2] = 0.0

        self.base_align.set_base(quat=base_quat_wxyz, pos=position)
        self.born_place_align = True
        logger.warning("Odometry rebased at xy=%s, body yaw quaternion=%s", position[:2], self.base_align.base_quat)
        return True

    def read(self) -> G1State:
        robot_state = self.control_endpoint.get_robot_state()
        base_quat = np.asarray(robot_state.imu_state.quaternion, dtype=np.float32)[[1, 2, 3, 0]]
        if self.born_place_align:
            base_quat = self.base_align.align_quat(base_quat[[3, 0, 1, 2]])[[1, 2, 3, 0]]
        state = G1State(
            dof_pos=snapshot(robot_state.motor_state.q),
            dof_vel=snapshot(robot_state.motor_state.dq),
            base_quat=snapshot(base_quat),
            base_ang_vel=snapshot(robot_state.imu_state.gyroscope),
        )
        if self.remote_controller_handler is not None:
            self.remote_controller_handler(robot_state.wireless_remote)
        return state

    def read_odometry(self) -> G1Odometry | None:
        try:
            sport = self.control_endpoint.get_sport_state()
        except RuntimeError:
            return None
        raw_position = np.asarray(sport.position, dtype=np.float32)
        position = np.array([raw_position[0], raw_position[1], sport.body_height], dtype=np.float32)
        if self.born_place_align:
            position = self.base_align.align_pos(position)
        return G1Odometry(
            position=snapshot(position),
            raw_position=snapshot(raw_position),
            velocity=snapshot(sport.velocity),
            body_height=float(sport.body_height),
        )

    def set_gains(self, stiffness, damping) -> None:
        self.control_endpoint.set_gains(
            np.asarray(stiffness, dtype=float).tolist(), np.asarray(damping, dtype=float).tolist()
        )

    def step(self, pd_target) -> None:
        self.control_endpoint.step(np.asarray(pd_target).tolist())

    def shutdown(self) -> None:
        self.control_endpoint.shutdown()
