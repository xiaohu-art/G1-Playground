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
        bridge,
        lowstate_factory: Callable,
        torque_limits,
        *,
        command_timeout: float = 0.1,
        clock: Callable[[], float] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ):
        command_timeout = float(command_timeout)
        if not 0.0 < command_timeout < np.inf:
            raise ValueError("DDS command timeout must be finite and positive")
        self.backend = backend
        self.bridge = bridge
        self.lowstate_factory = lowstate_factory
        self.torque_limits = np.asarray(torque_limits, dtype=np.float64)
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
        if not command.valid:
            return np.zeros_like(state.joint_pos, dtype=np.float64)
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
            + stiffness * (position_target - state.joint_pos)
            + damping * (velocity_target - state.joint_vel)
        )
        return np.clip(torque, -self.torque_limits, self.torque_limits)

    def publish(self, state) -> int:
        snapshot = self.lowstate_factory()
        snapshot.q = state.joint_pos.tolist()
        snapshot.dq = state.joint_vel.tolist()
        snapshot.tau_est = state.joint_torque.tolist()
        snapshot.quaternion = state.base_quaternion_wxyz.tolist()
        snapshot.gyroscope = state.base_angular_velocity.tolist()
        return self.bridge.publish_lowstate(snapshot)

    def step(self, now: float | None = None) -> int:
        now = self._clock() if now is None else now
        if not np.isfinite(now):
            raise RuntimeError("DDS simulator clock returned a non-finite value")
        command = self.bridge.get_command()
        torque = self.command_torque(command, self._state, now)
        self._state = self.backend.step(torque, self.support_scale(now))
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
        return self.bridge.close()
