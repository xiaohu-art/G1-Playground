import logging
from collections import deque
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import DictConfig

from g1_playground.policy.base_policy import BasePolicy
from g1_playground.utils import resolve_repo_path
from g1_playground.utils.dof import DoFAdapter
from g1_playground.utils.math import get_gravity_orientation

logger = logging.getLogger(__name__)


class UnitreeWoGaitPolicy(BasePolicy):
    FREQ = 50
    HISTORY_LENGTH = 5
    HISTORY_LAYOUT = (
        ("ang_vel", 3),
        ("gravity", 3),
        ("commands", 3),
        ("dof_pos", 29),
        ("dof_vel", 29),
        ("actions", 29),
    )

    def __init__(self, cfg_policy: DictConfig, device: str, dof_cfg: DictConfig):
        dof = cfg_policy.dof
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(device)

        self.num_dofs = len(dof.joint_names)
        self.default_pos = np.asarray(dof.default_pos)

        policy_file = resolve_repo_path(cfg_policy.policy_file)
        logger.debug("Loading TorchScript policy from %s", policy_file)
        self.model = torch.jit.load(policy_file, map_location=device)
        self.action_scale = cfg_policy.action_scale
        self.last_action = np.zeros(self.num_dofs)

        self.observation_adapter = DoFAdapter(dof_cfg.joint_names, dof.joint_names)
        self.action_adapter = DoFAdapter(dof.joint_names, dof_cfg.joint_names)

        self.obs_scales = cfg_policy.obs_scales
        self.max_cmd = np.asarray(cfg_policy.max_cmd, dtype=np.float32)
        self.reset()

    def reset(self) -> None:
        self.last_action.fill(0.0)
        default_history = [np.zeros(dim, dtype=np.float32) for _, dim in self.HISTORY_LAYOUT]
        self.history_buf = deque([default_history] * self.HISTORY_LENGTH, maxlen=self.HISTORY_LENGTH)

    def get_observation(self, env_data, control_data) -> np.ndarray:
        axes = control_data["axes"]
        commands = np.asarray([axes["LeftY"], -axes["LeftX"], -axes["RightX"]], dtype=np.float32)
        commands[np.abs(commands) < 0.04] = 0.0

        gravity_orientation = get_gravity_orientation(env_data.base_quat)
        obs_current = [
            env_data.base_ang_vel * self.obs_scales.ang_vel,
            gravity_orientation,
            commands * self.max_cmd,
            env_data.dof_pos - self.default_pos,
            env_data.dof_vel * self.obs_scales.dof_vel,
            self.last_action,
        ]
        self.history_buf.append(obs_current)
        history_list = [np.concatenate(items, axis=0) for items in zip(*self.history_buf, strict=True)]
        return np.concatenate(history_list, axis=0)

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        observation_tensor = torch.from_numpy(observation).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            action_tensor = self.model(observation_tensor).cpu()

        action = action_tensor.numpy().squeeze()
        self.last_action = action.copy()
        return action * self.action_scale

    @property
    def standing_target(self) -> np.ndarray:
        return self.action_adapter.fit(self.default_pos)

    def act(self, env_data, control_data) -> np.ndarray:
        policy_state = SimpleNamespace(
            base_quat=env_data.base_quat,
            base_ang_vel=env_data.base_ang_vel,
            dof_pos=self.observation_adapter.fit(env_data.dof_pos),
            dof_vel=self.observation_adapter.fit(env_data.dof_vel),
        )
        observation = self.get_observation(policy_state, control_data)
        policy_target = self.get_action(observation) + self.default_pos
        return self.action_adapter.fit(policy_target)
