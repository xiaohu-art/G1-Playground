import logging
import time
from collections.abc import Callable

import numpy as np

logger = logging.getLogger("g1_playground")

SUPPORT_HOLD_SECONDS = 3.0
SUPPORT_RELEASE_SECONDS = 5.0


class G1MujocoDdsServer:
    """Run the first-party G1 MuJoCo backend behind the Unitree HG DDS wire contract."""

    def __init__(
        self,
        backend,
        robot_endpoint,
        lowstate_factory: Callable,
        torque_limits,
        *,
        body_index=None,
        hand=None,
        command_timeout: float = 0.1,
        clock: Callable[[], float] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ):
        command_timeout = float(command_timeout)
        if not 0.0 < command_timeout < np.inf:
            raise ValueError("DDS command timeout must be finite and positive")
        self.backend = backend
        self.robot_endpoint = robot_endpoint
        self.lowstate_factory = lowstate_factory
        self.torque_limits = np.asarray(torque_limits, dtype=np.float64)
        self.body_index = (
            np.arange(self.torque_limits.shape[0], dtype=np.int64)
            if body_index is None
            else np.asarray(body_index, dtype=np.int64)
        )
        if self.body_index.shape != self.torque_limits.shape:
            raise ValueError("body index must supply one model actuator per torque limit")
        self.hand = hand
        actuators = getattr(getattr(backend, "model", None), "nu", None)
        self.num_actuators = int(actuators) if actuators is not None else int(self.torque_limits.shape[0])
        self.timestep = backend.timestep
        self.command_timeout = command_timeout
        self._clock = time.monotonic if clock is None else clock
        self._sleep = time.sleep if sleeper is None else sleeper
        self._first_command_time: float | None = None
        self._state = backend.read()

    def support_scale(self, now: float) -> float:
        if self._first_command_time is None:
            return 1.0
        active_seconds = now - self._first_command_time
        if active_seconds <= SUPPORT_HOLD_SECONDS:
            return 1.0
        if active_seconds >= SUPPORT_HOLD_SECONDS + SUPPORT_RELEASE_SECONDS:
            return 0.0
        return 1.0 - (active_seconds - SUPPORT_HOLD_SECONDS) / SUPPORT_RELEASE_SECONDS

    def command_torque(self, command, state, now: float) -> np.ndarray:
        joint_pos = np.asarray(state.joint_pos)[self.body_index]
        joint_vel = np.asarray(state.joint_vel)[self.body_index]
        if not command.valid:
            return np.zeros_like(joint_pos, dtype=np.float64)
        if not np.isfinite(command.age_seconds) or command.age_seconds < 0:
            raise RuntimeError("DDS simulator received an invalid command age")
        if command.age_seconds > self.command_timeout:
            raise RuntimeError(
                f"DDS simulator command watchdog expired: {command.age_seconds:.6f}s > {self.command_timeout:.6f}s"
            )
        if self._first_command_time is None:
            self._first_command_time = now - float(command.age_seconds)

        position_target = np.asarray(command.q, dtype=np.float64)
        velocity_target = np.asarray(command.dq, dtype=np.float64)
        feedforward = np.asarray(command.tau, dtype=np.float64)
        stiffness = np.asarray(command.kp, dtype=np.float64)
        damping = np.asarray(command.kd, dtype=np.float64)
        torque = (
            feedforward
            + stiffness * (position_target - joint_pos)
            + damping * (velocity_target - joint_vel)
        )
        return np.clip(torque, -self.torque_limits, self.torque_limits)

    def publish(self, state) -> int:
        snapshot = self.lowstate_factory()
        snapshot.q = np.asarray(state.joint_pos)[self.body_index].tolist()
        snapshot.dq = np.asarray(state.joint_vel)[self.body_index].tolist()
        snapshot.tau_est = np.asarray(state.joint_torque)[self.body_index].tolist()
        snapshot.quaternion = state.base_quaternion_wxyz.tolist()
        snapshot.gyroscope = state.base_angular_velocity.tolist()
        return self.robot_endpoint.publish_lowstate(snapshot)

    def step(self, now: float | None = None) -> int:
        now = self._clock() if now is None else now
        if not np.isfinite(now):
            raise RuntimeError("DDS simulator clock returned a non-finite value")
        command = self.robot_endpoint.get_command()
        body_torque = self.command_torque(command, self._state, now)

        torque = np.zeros(self.num_actuators, dtype=np.float64)
        torque[self.body_index] = body_torque
        if self.hand is not None:
            hand_index, hand_torque = self.hand.torque(self._state.joint_pos, self._state.joint_vel, now)
            torque[hand_index] = hand_torque

        self._state = self.backend.step(torque, self.support_scale(now))
        if self.hand is not None:
            self.hand.publish(self._state.joint_pos, self._state.joint_vel, self.timestep)
        return self.publish(self._state)

    def run(self, stop_event=None) -> None:
        deadline = self._clock()
        try:
            while stop_event is None or not stop_event.is_set():
                self.step(now=self._clock())
                deadline += self.timestep
                remaining = deadline - self._clock()
                if remaining > 0:
                    self._sleep(remaining)
                elif remaining < -self.timestep:
                    missed_deadlines = int(-remaining // self.timestep)
                    deadline += missed_deadlines * self.timestep
                    logger.warning("MuJoCo DDS server missed %d physics deadlines", missed_deadlines)
        finally:
            self.shutdown()

    def shutdown(self) -> bool:
        if self.hand is not None:
            self.hand.shutdown()
        return self.robot_endpoint.close()
