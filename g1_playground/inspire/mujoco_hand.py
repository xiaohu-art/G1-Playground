import logging

import numpy as np
from unitree_cpp import InspireDdsRobotEndpoint

from g1_playground.inspire import dof as dof_utils

logger = logging.getLogger("g1_playground")

VELOCITY_ALPHA = 0.3
COMMAND_TIMEOUT = 1.0


class InspireMujocoHand:
    """The Inspire hand as the simulator provides it: MuJoCo physics on one side, the rt/inspire wire on the other."""

    def __init__(
        self,
        model,
        dof_cfg,
        mimic_cfg,
        *,
        stiffness: float,
        damping: float,
        torque_limit: float,
        domain_id: int = 1,
        net_if: str = "lo",
        cmd_topic: str = "rt/inspire/cmd",
        state_topic: str = "rt/inspire/state",
        command_timeout: float = COMMAND_TIMEOUT,
    ):
        self.driven_names = dof_utils.joint_names(dof_cfg)
        self.lower, self.upper = dof_utils.limits(dof_cfg)
        self.mimic_cfg = mimic_cfg

        names = dof_utils.actuator_names(model)
        missing = [n for n in self.driven_names if n not in names]
        if missing:
            raise ValueError(f"model has no actuator for {missing}")
        self.driven_index = np.array([names.index(n) for n in self.driven_names], dtype=np.int64)

        self.follower_names = list(mimic_cfg)
        missing = [n for n in self.follower_names if n not in names]
        if missing:
            raise ValueError(f"model has no actuator for mimic follower {missing}")
        self.follower_index = np.array([names.index(n) for n in self.follower_names], dtype=np.int64)
        self.follower_driver = np.array(
            [self.driven_names.index(str(mimic_cfg[n].driver)) for n in self.follower_names], dtype=np.int64
        )
        self.follower_multiplier = np.array(
            [float(mimic_cfg[n].multiplier) for n in self.follower_names], dtype=np.float64
        )
        self.follower_offset = np.array([float(mimic_cfg[n].offset) for n in self.follower_names], dtype=np.float64)
        self.follower_lower = np.array([float(mimic_cfg[n].lower) for n in self.follower_names], dtype=np.float64)
        self.follower_upper = np.array([float(mimic_cfg[n].upper) for n in self.follower_names], dtype=np.float64)

        self.index = np.concatenate([self.driven_index, self.follower_index])
        self.stiffness = float(stiffness)
        self.damping = float(damping)
        self.torque_limit = float(torque_limit)
        self.command_timeout = float(command_timeout)

        self._stroke_rate = np.zeros(dof_utils.NUM_SLOTS, dtype=np.float64)
        self._last_stroke = None

        self.endpoint = InspireDdsRobotEndpoint(
            {
                "domain_id": domain_id,
                "net_if": net_if,
                "cmd_topic": cmd_topic,
                "state_topic": state_topic,
            }
        )

    def joint_targets(self, now: float) -> np.ndarray | None:
        command = self.endpoint.get_command()
        if not command.valid or command.age_seconds > self.command_timeout:
            return None
        driven = dof_utils.q_to_rad(np.asarray(command.q), self.lower, self.upper)
        follower = np.clip(
            driven[self.follower_driver] * self.follower_multiplier + self.follower_offset,
            self.follower_lower,
            self.follower_upper,
        )
        return np.concatenate([driven, follower])

    def torque(self, joint_pos, joint_vel, now: float) -> tuple[np.ndarray, np.ndarray]:
        targets = self.joint_targets(now)
        if targets is None:
            return self.index, np.zeros(self.index.shape[0], dtype=np.float64)
        error = targets - np.asarray(joint_pos)[self.index]
        torque = self.stiffness * error - self.damping * np.asarray(joint_vel)[self.index]
        return self.index, np.clip(torque, -self.torque_limit, self.torque_limit)

    def publish(self, joint_pos, joint_vel, dt: float) -> np.ndarray:
        driven_rad = np.asarray(joint_pos)[self.driven_index]
        stroke = dof_utils.quantize_q(dof_utils.rad_to_q(driven_rad, self.lower, self.upper))
        if self._last_stroke is not None and dt > 0.0:
            raw = (stroke - self._last_stroke) / dt
            self._stroke_rate = VELOCITY_ALPHA * raw + (1.0 - VELOCITY_ALPHA) * self._stroke_rate
        self._last_stroke = stroke

        self.endpoint.publish_state(stroke.tolist(), self._stroke_rate.tolist())
        return stroke

    def shutdown(self) -> None:
        self.endpoint.close()
