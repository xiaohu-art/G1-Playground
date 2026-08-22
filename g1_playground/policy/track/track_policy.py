import logging

import numpy as np
import onnxruntime as ort
from omegaconf import DictConfig

from g1_playground.policy.base_policy import BasePolicy
from g1_playground.policy.track.track_observation import TrackObservation
from g1_playground.utils import resolve_repo_path

logger = logging.getLogger(__name__)

NUM_JOINTS = 29


class TrackPolicy(BasePolicy):
    FREQ = 50
    OBS_DIM = 167

    def __init__(self, cfg_policy: DictConfig, device: str, dof_cfg: DictConfig):
        super().__init__(device)
        self.num_dofs = NUM_JOINTS

        self.default_pos = np.asarray(dof_cfg.default_pos)
        self.action_scale = np.asarray(cfg_policy.action_scale)
        self.joint_pos_lower = np.asarray(cfg_policy.joint_pos_lower)
        self.joint_pos_upper = np.asarray(cfg_policy.joint_pos_upper)

        clip_range = np.asarray(cfg_policy.clip_range, dtype=np.float64).reshape(-1)
        self.clip_low, self.clip_high = float(clip_range[0]), float(clip_range[1])

        self.reference_root_height = float(cfg_policy.reference_root_height)
        if not 0.0 < self.reference_root_height < 2.0:
            raise ValueError("Track policy reference_root_height must be a plausible G1 root height")

        policy_file = resolve_repo_path(cfg_policy.policy_file)
        logger.debug("Loading ONNX track policy from %s", policy_file)
        providers = ["CPUExecutionProvider"]
        if str(device).lower().startswith("cuda") and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        self.session = ort.InferenceSession(policy_file, providers=providers)
        self._obs_name, self._history_name, self._history_length = self._resolve_signature()
        self._output_name = self.session.get_outputs()[0].name

        self.observation = TrackObservation(
            resolve_repo_path(cfg_policy.fk_xml),
            str(cfg_policy.anchor_body_name),
            self.default_pos,
            NUM_JOINTS,
        )
        if self.observation.total_obs_size != self.OBS_DIM:
            raise ValueError(
                f"Track policy expects a {self.OBS_DIM}D observation, "
                f"builder produced {self.observation.total_obs_size}D"
            )

        self.reset()

    def _resolve_signature(self) -> tuple[str, str, int]:
        inputs = self.session.get_inputs()
        names = [tensor.name for tensor in inputs]
        if len(inputs) != 2 or names[1] != "obs_history":
            raise ValueError(f"Track policy expects 'obs' and 'obs_history' inputs, got {names}")
        if int(inputs[0].shape[-1]) != self.OBS_DIM or int(inputs[1].shape[-1]) != self.OBS_DIM:
            raise ValueError(f"Track policy expects {self.OBS_DIM}D inputs, got {names}")
        return names[0], names[1], int(inputs[1].shape[1])

    def reset(self) -> None:
        self._last_action = np.zeros(NUM_JOINTS, dtype=np.float32)
        self._history = None
        self._reference_ready = False
        self._last_target = self.default_pos.copy()

    @property
    def standing_target(self) -> np.ndarray:
        return self.default_pos.copy()

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        current = observation[np.newaxis]
        if self._history is None:
            self._history = np.repeat(observation[np.newaxis, np.newaxis], self._history_length, axis=1)
        else:
            self._history = np.concatenate([self._history[:, 1:], observation[np.newaxis, np.newaxis]], axis=1)
        outputs = self.session.run(
            [self._output_name],
            {self._obs_name: current, self._history_name: self._history},
        )
        return np.asarray(outputs[0], dtype=np.float32).reshape(-1)

    def get_observation(self, env_data, control_data) -> np.ndarray:
        base_quat_wxyz = np.asarray(env_data.base_quat, dtype=np.float32)[[3, 0, 1, 2]]
        if not self._reference_ready:
            self.observation.set_standing_reference(base_quat_wxyz, self.reference_root_height)
            self._reference_ready = True
        return self.observation.build(
            env_data.dof_pos,
            env_data.dof_vel,
            base_quat_wxyz,
            env_data.base_ang_vel,
            self._last_action,
        )

    @property
    def last_action(self) -> np.ndarray:
        return self._last_action.copy()

    def set_reference(self, root_height, root_quat, joint_pos, joint_vel, anchor_lin_vel_w, anchor_ang_vel_w) -> None:
        self.observation.set_reference(root_height, root_quat, joint_pos, joint_vel, anchor_lin_vel_w, anchor_ang_vel_w)
        self._reference_ready = True

    def accept_applied_target(self, target) -> None:
        target = np.asarray(target, dtype=np.float64).reshape(NUM_JOINTS)
        target = np.clip(target, self.joint_pos_lower, self.joint_pos_upper)
        action = (target - self.default_pos) / self.action_scale
        self._last_action = np.clip(action, self.clip_low, self.clip_high).astype(np.float32)
        self._last_target = target.copy()

    def act(self, env_data, control_data) -> np.ndarray:
        raw_action = self.get_action(self.get_observation(env_data, control_data))
        if not np.all(np.isfinite(raw_action)):
            logger.critical("Non-finite ONNX action; holding the last commanded target")
            return self._last_target.copy()

        self._last_action = raw_action.copy()
        target = np.clip(raw_action, self.clip_low, self.clip_high).astype(np.float64)
        target = target * self.action_scale + self.default_pos
        self._last_target = np.clip(target, self.joint_pos_lower, self.joint_pos_upper)
        return self._last_target.copy()
