import logging
import time
from dataclasses import dataclass

import numpy as np
from unitree_cpp import InspireDdsControlEndpoint

from g1_playground.inspire import dof as dof_utils

logger = logging.getLogger(__name__)


def snapshot(values) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class InspireHandState:
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    lost: np.ndarray
    age: float

    @property
    def stale(self) -> bool:
        return not np.isfinite(self.age) or self.age > InspireHandEnv.STALE_SECONDS

    def as_dict(self, joint_names):
        return (
            dof_utils.to_dict(self.joint_pos, joint_names),
            dof_utils.to_dict(self.joint_vel, joint_names),
        )

    def with_mimic(self, joint_names, mimic_cfg):
        position, velocity = self.as_dict(joint_names)
        return (
            dof_utils.expand_mimic(position, mimic_cfg),
            dof_utils.expand_mimic_velocity(velocity, mimic_cfg),
        )


class InspireHandEnv:
    """Inspire hand state and command boundary backed by Unitree DDS."""

    STALE_SECONDS = 0.3

    def __init__(
        self,
        *,
        dof_cfg,
        domain_id: int = 1,
        net_if: str = "lo",
        cmd_topic: str = "rt/inspire/cmd",
        state_topic: str = "rt/inspire/state",
    ):
        self.joint_names = dof_utils.joint_names(dof_cfg)
        self.lower, self.upper = dof_utils.limits(dof_cfg)
        self.num_dofs = dof_utils.NUM_SLOTS
        self.endpoint = InspireDdsControlEndpoint(
            {
                "domain_id": domain_id,
                "net_if": net_if,
                "cmd_topic": cmd_topic,
                "state_topic": state_topic,
            }
        )
        self._last_stroke = np.ones(dof_utils.NUM_SLOTS, dtype=np.float64)

    def self_check(self, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.endpoint.self_check():
                logger.info("InspireHandEnv self check passed")
                return
            time.sleep(0.05)
        raise RuntimeError("InspireHandEnv self check failed: no state received")

    def read(self) -> InspireHandState:
        state = self.endpoint.get_state()
        if not state.valid:
            return InspireHandState(
                joint_pos=snapshot(self.lower),
                joint_vel=snapshot(np.zeros(dof_utils.NUM_SLOTS)),
                lost=np.zeros(dof_utils.NUM_SLOTS, dtype=np.uint32),
                age=float("inf"),
            )
        lost = np.asarray(state.lost, dtype=np.uint32).copy()
        return InspireHandState(
            joint_pos=snapshot(dof_utils.q_to_rad(np.asarray(state.q), self.lower, self.upper)),
            joint_vel=snapshot(dof_utils.dq_to_rad_per_s(np.asarray(state.dq), self.lower, self.upper)),
            lost=lost,
            age=float(state.age_seconds),
        )

    def step(self, joint_pos_target) -> np.ndarray:
        target = np.asarray(joint_pos_target, dtype=np.float64).reshape(-1)
        if target.shape != (dof_utils.NUM_SLOTS,):
            raise ValueError(f"Inspire target must contain {dof_utils.NUM_SLOTS} joint positions")
        if not np.all(np.isfinite(target)):
            raise ValueError(f"non-finite Inspire joint target: {target.tolist()}")

        stroke = dof_utils.validate_q(dof_utils.clip_q(dof_utils.rad_to_q(target, self.lower, self.upper)))
        self.endpoint.step(stroke.tolist())
        self._last_stroke = stroke
        return stroke

    def step_dict(self, targets, missing=None) -> np.ndarray:
        return self.step(dof_utils.from_dict(targets, self.joint_names, missing=missing))

    def open(self) -> np.ndarray:
        return self.step(self.lower)

    def hold(self) -> np.ndarray:
        return self.step(dof_utils.q_to_rad(self._last_stroke, self.lower, self.upper))

    def shutdown(self) -> None:
        self.endpoint.close()
