import atexit
import collections
import time
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

from g1_playground.action import JointAction
from g1_playground.joints import JOINT_DFS_NAMES
from g1_playground.policies.params import NUM_JOINTS, REPO_ROOT
from g1_playground.policies.policy import Policy
from g1_playground.state import G1State
from g1_playground.utils.strings import resolve_param
from g1_playground.utils.math import (
    GRAVITY_DIR,
    quat_inv,
    quat_mul,
    quat_rotate,
    quat_to_rot6d,
    yaw_quat,
)

OBS_DIM = 167


class TrackPolicy(Policy):
    DT = 0.02  # 50 Hz
    DEFAULT_CONFIG_PATH = REPO_ROOT / "configs/run_track.yaml"

    FK_XML = Path(__file__).resolve().parent / "g1_29dof_fk.xml"
    ANCHOR_BODY = "torso_link"

    def __init__(self, cfg: dict | None = None):
        cfg = self._init_from_config(cfg)
        self._configure(cfg)

        self._session = ort.InferenceSession(
            str(self.params.model_path), providers=["CPUExecutionProvider"]
        )
        self._obs_name = self._session.get_inputs()[0].name

        # forward kinematics model
        self._mj_model = mujoco.MjModel.from_xml_path(str(self.fk_xml))
        self._mj_data = mujoco.MjData(self._mj_model)
        self._anchor_id = mujoco.mj_name2id(
            self._mj_model, mujoco.mjtObj.mjOBJ_BODY, self.anchor_body
        )
        if self._anchor_id < 0:
            raise ValueError(f"anchor body '{self.anchor_body}' not found in {self.fk_xml}")

        self._dump = None
        if self.dump_path:
            atexit.register(self._write_dump)

    def _configure(self, cfg: dict) -> None:
        self.fk_xml = self.FK_XML.resolve()
        self.anchor_body = self.ANCHOR_BODY
        self.history_length = int(cfg["history_length"])
        self.action_clip = float(cfg["action_clip"])
        self.ref_root_height = float(cfg["reference"]["root_height"])
        self.ramp_duration = float(cfg["ramp_duration"])
        self.ramp_steps = max(1, int(self.ramp_duration / self.DT))
        self.kp_ramp_floor = float(cfg["kp_ramp_floor"])
        self.dump_path = cfg.get("dump")
        
        self.joint_pos_lower = resolve_param(cfg["joint_pos_lower"]["values"], JOINT_DFS_NAMES)
        self.joint_pos_upper = resolve_param(cfg["joint_pos_upper"]["values"], JOINT_DFS_NAMES)

    @property
    def dt(self) -> float:
        return self.DT

    # ---- forward kinematics -------------------------------------------------

    def _fk_anchor(self, base_pos, base_quat, joint_pos) -> tuple[np.ndarray, np.ndarray]:
        """Anchor (torso) world pose for a given root pose + joint configuration."""
        self._mj_data.qpos[:] = 0.0
        self._mj_data.qpos[0:3] = np.asarray(base_pos, dtype=np.float64)
        quat = np.asarray(base_quat, dtype=np.float64)
        self._mj_data.qpos[3:7] = quat / max(np.linalg.norm(quat), 1e-8)
        self._mj_data.qpos[7 : 7 + NUM_JOINTS] = np.asarray(joint_pos, dtype=np.float64)
        mujoco.mj_kinematics(self._mj_model, self._mj_data)
        return (
            np.asarray(self._mj_data.xpos[self._anchor_id], dtype=np.float32).copy(),
            np.asarray(self._mj_data.xquat[self._anchor_id], dtype=np.float32).copy(),
        )

    # ---- lifecycle ----------------------------------------------------------

    def reset(self, state: G1State) -> None:
        self._t = 0.0
        self._ramp_step = 0
        self._last_action = np.zeros(NUM_JOINTS, dtype=np.float32)
        self._history = collections.deque(maxlen=self.history_length)
        self._dump = (
            {k: [] for k in ("t", "wall", "q", "dq", "tau", "quat", "gyro", "action", "q_target")}
            if self.dump_path
            else None
        )

        robot_quat = np.asarray(state.body.imu_state.quaternion, dtype=np.float32)
        self._ref_root_pos = np.array([0.0, 0.0, self.ref_root_height], dtype=np.float32)
        self._ref_root_quat = yaw_quat(robot_quat)
        self._ref_joint_pos = self.params.default_pos.astype(np.float32)
        self._ref_joint_vel = np.zeros(NUM_JOINTS, dtype=np.float32)
        self._ref_anchor_lin_vel_w = np.zeros(3, dtype=np.float32)
        self._ref_anchor_ang_vel_w = np.zeros(3, dtype=np.float32)

    def step(self, state: G1State) -> JointAction:
        self._t += self.DT
        p = self.params

        q = np.array([state.body.motor_state[i].q for i in range(NUM_JOINTS)], dtype=np.float32)
        dq = np.array([state.body.motor_state[i].dq for i in range(NUM_JOINTS)], dtype=np.float32)
        quat = np.asarray(state.body.imu_state.quaternion, dtype=np.float32)
        gyro = np.asarray(state.body.imu_state.gyroscope, dtype=np.float32)

        obs = self._build_obs(q, dq, quat, gyro)
        self._history.append(obs)
        while len(self._history) < self.history_length:  # first step: fill with current
            self._history.appendleft(obs)
        obs_history = np.stack(self._history, axis=0)[None]

        action = self._session.run(
            None, {self._obs_name: obs[None], "obs_history": obs_history}
        )[0].reshape(-1)
        if not np.all(np.isfinite(action)):
            action = np.zeros_like(action)
        self._last_action = action.astype(np.float32)

        scaled = np.clip(self._last_action, -self.action_clip, self.action_clip) * p.action_scale
        q_target = np.clip(
            scaled + p.default_pos, self.joint_pos_lower, self.joint_pos_upper
        ).astype(np.float32)

        factor = min(1.0, self._ramp_step / self.ramp_steps)
        self._ramp_step += 1
        kp = p.kp * (self.kp_ramp_floor + (1.0 - self.kp_ramp_floor) * factor)

        if self._dump is not None:
            tau = np.array(
                [state.body.motor_state[i].tau_est for i in range(NUM_JOINTS)], dtype=np.float32
            )
            for key, value in (
                ("t", self._t), ("wall", time.monotonic()), ("q", q), ("dq", dq), ("tau", tau),
                ("quat", quat), ("gyro", gyro),
                ("action", self._last_action.copy()), ("q_target", q_target.copy()),
            ):
                self._dump[key].append(value)

        return JointAction(q=q_target, kp=kp.astype(np.float32), kd=p.kd)

    # ---- observation --------------------------------------------------------

    def _build_obs(self, q, dq, quat, gyro) -> np.ndarray:
        _, robot_anchor_quat = self._fk_anchor(np.zeros(3), quat, q)
        ref_anchor_pos, ref_anchor_quat = self._fk_anchor(
            self._ref_root_pos, self._ref_root_quat, self._ref_joint_pos
        )
        robot_anchor_inv = quat_inv(robot_anchor_quat)

        obs = np.concatenate(
            [
                self._ref_joint_pos,                                        # 29 absolute
                self._ref_joint_vel,                                        # 29
                quat_to_rot6d(quat_mul(robot_anchor_inv, ref_anchor_quat)), # 6
                gyro,                                                       # 3
                q - self.params.default_pos,                                # 29 relative
                dq,                                                         # 29
                self._last_action,                                          # 29 raw
                quat_rotate(quat_inv(quat), GRAVITY_DIR),                   # 3
                quat_rotate(robot_anchor_inv, self._ref_anchor_lin_vel_w),  # 3
                quat_rotate(robot_anchor_inv, self._ref_anchor_ang_vel_w),  # 3
                quat_rotate(quat_inv(ref_anchor_quat), GRAVITY_DIR),        # 3
                ref_anchor_pos[2:3],                                        # 1
            ],
            dtype=np.float32,
        )
        assert obs.shape == (OBS_DIM,)
        if not np.all(np.isfinite(obs)):
            obs = np.where(np.isfinite(obs), obs, np.float32(0.0))
        return obs

    def _write_dump(self):
        if self._dump and self._dump["t"]:
            np.savez(self.dump_path, **{k: np.asarray(v) for k, v in self._dump.items()})
            print(f"\ndumped {len(self._dump['t'])} steps to {self.dump_path}")
