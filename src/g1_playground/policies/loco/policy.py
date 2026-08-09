"""Velocity-tracking locomotion policy (29-DOF, 50 Hz).
"""

import atexit
import collections

import numpy as np
import onnxruntime as ort

from g1_playground.action import JointAction
from g1_playground.command import KeyboardCommander, WirelessRemoteCommander
from g1_playground.joints import JOINT_DFS_NAMES, JointAdapter
from g1_playground.policies.params import NUM_JOINTS, REPO_ROOT
from g1_playground.policies.policy import Policy
from g1_playground.state import G1State
from g1_playground.utils.math import projected_gravity


class LocoPolicy(Policy):
    DT = 0.02  # 50 Hz
    RAMP_DURATION_S = 2.0
    DEFAULT_CONFIG_PATH = REPO_ROOT / "configs/run_loco.yaml"

    def __init__(self, cfg: dict | None = None):
        cfg = self._init_from_config(cfg)
        self._configure(cfg)
        self._session = ort.InferenceSession(
            str(self.params.model_path), providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

        #   obs_adapter:    wire (DFS) -> policy order   (everything entering the policy)
        #   action_adapter: policy order -> wire (DFS)   (everything leaving to the robot)
        self.obs_adapter = JointAdapter(JOINT_DFS_NAMES, self.params.policy_joint_names)
        self.action_adapter = JointAdapter(self.params.policy_joint_names, JOINT_DFS_NAMES)

        # policy-order views of the wire-order params, converted once
        self.default_dof_pos = self.obs_adapter.fit(self.params.default_pos)
        self.action_scale = self.obs_adapter.fit(self.params.action_scale)

        self._cmd = np.zeros(3, dtype=np.float32)
        self.set_command(*self.initial_command)

        self._commander = None
        if self.command_source == "wireless_remote":
            self._commander = WirelessRemoteCommander(self)
        elif self.command_source == "keyboard":
            self._commander = KeyboardCommander(self)
        if self._commander is not None:
            atexit.register(self._commander.stop)  # keyboard: restore terminal

    def _configure(self, cfg: dict) -> None:
        self.ang_vel_scale = float(cfg["obs_scales"]["ang_vel"])
        self.dof_vel_scale = float(cfg["obs_scales"]["dof_vel"])
        self.history_length = int(cfg["history_length"])
        self.max_cmd = np.asarray(cfg["max_cmd"], dtype=np.float32)  # |vx|, |vy|, |wyaw|
        self.term_dims = (3, 3, 3, NUM_JOINTS, NUM_JOINTS, NUM_JOINTS)
        self.obs_dim = self.history_length * sum(self.term_dims)
        
        cmd_cfg = cfg["command"]
        self.command_source = cmd_cfg["source"]
        self.initial_command = (float(cmd_cfg["vx"]), float(cmd_cfg["vy"]), float(cmd_cfg["wyaw"]))

    @property
    def dt(self) -> float:
        return self.DT

    def set_command(self, vx: float, vy: float, wyaw: float) -> None:
        """Set the velocity command (physical units), clipped to the trained ranges."""
        self._cmd = np.clip([vx, vy, wyaw], -self.max_cmd, self.max_cmd).astype(np.float32)

    def reset(self, state: G1State) -> None:
        if self._commander is not None:
            self._commander.start()  # idempotent
        self._t = 0.0
        self._last_action = np.zeros(NUM_JOINTS, dtype=np.float32)
        self._reset_history()
        self._initial_q = np.array(
            [state.body.motor_state[i].q for i in range(NUM_JOINTS)], dtype=np.float32
        )

    def step(self, state: G1State) -> JointAction:
        self._t += self.DT
        p = self.params
        if self._commander is not None:
            self._commander.update(state)

        if self._t < self.RAMP_DURATION_S:
            ratio = min(self._t / self.RAMP_DURATION_S, 1.0)
            q_target = (1.0 - ratio) * self._initial_q + ratio * p.default_pos
            return JointAction(q=q_target, kp=p.kp, kd=p.kd)

        q = np.array([state.body.motor_state[i].q for i in range(NUM_JOINTS)], dtype=np.float32)
        dq = np.array([state.body.motor_state[i].dq for i in range(NUM_JOINTS)], dtype=np.float32)
        quat_wxyz = np.array(state.body.imu_state.quaternion, dtype=np.float32)
        gyro = np.array(state.body.imu_state.gyroscope, dtype=np.float32)

        obs = self._build_obs(q, dq, quat_wxyz, gyro)
        action = self._session.run(None, {self._input_name: obs[None]})[0][0]
        self._last_action = action.astype(np.float32)

        q_policy_target = self.default_dof_pos + self.action_scale * self._last_action
        q_target = self.action_adapter.fit(q_policy_target, template=p.default_pos)
        return JointAction(q=q_target, kp=p.kp, kd=p.kd)

    def _reset_history(self):
        zero_frame = [np.zeros(dim, dtype=np.float32) for dim in self.term_dims]
        self._history = collections.deque([zero_frame] * self.history_length, maxlen=self.history_length)

    def _build_obs(self, q, dq, quat_wxyz, gyro) -> np.ndarray:
        q = np.asarray(q, dtype=np.float32)
        dq = np.asarray(dq, dtype=np.float32)
        gyro = np.asarray(gyro, dtype=np.float32)
        frame = [
            gyro * self.ang_vel_scale,
            projected_gravity(quat_wxyz).astype(np.float32),
            self._cmd,
            self.obs_adapter.fit(q) - self.default_dof_pos,
            self.obs_adapter.fit(dq) * self.dof_vel_scale,
            self._last_action,
        ]
        self._history.append(frame)
        # per-term blocks, frames oldest→newest inside each block
        blocks = [np.concatenate(frames) for frames in zip(*self._history)]
        obs = np.concatenate(blocks)
        assert obs.shape == (self.obs_dim,)
        return obs
