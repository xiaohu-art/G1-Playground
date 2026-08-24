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


class LeggedLabPolicy(BasePolicy):
    """Locomotion policy for the LeggedLab G1 checkpoint (TorchScript, 96 -> 29).

    Observation/action semantics mirror ``LeggedLabDeploy/deploy.py::run()`` from
    https://github.com/Hellod035/LeggedLabDeploy (BSD-3-Clause): unit observation
    scales, remote commands clipped to ``command_range`` (no deadzone), observation
    and action clipping to +-``clip_obs``/``clip_action``. The checkpoint is
    recurrent (single-frame observation, LSTM state kept as buffers inside the
    module) and bakes its own input normalization into the graph.
    """

    FREQ = 50
    HISTORY_LENGTH = 1
    HISTORY_LAYOUT = (
        ("ang_vel", 3),
        ("gravity", 3),
        ("commands", 3),
        ("dof_pos", 29),
        ("dof_vel", 29),
        ("actions", 29),
    )
    RESET_WARMUP_STEPS = 50

    def __init__(self, cfg_policy: DictConfig, device: str, dof_cfg: DictConfig):
        dof = cfg_policy.dof
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(device)
        if self.device == "cpu":
            # The 96->256 LSTM gains nothing from intra-op parallelism, while the default
            # thread pool adds multi-ten-millisecond scheduling tails (p99 ~25 ms measured
            # on a busy host) that blow the 50 Hz frame budget and the DDS command watchdog.
            torch.set_num_threads(1)

        self.num_dofs = len(dof.joint_names)
        for field in ("default_pos", "stiffness", "damping"):
            if len(getattr(dof, field)) != self.num_dofs:
                raise ValueError(f"LeggedLabPolicy requires one {field} value for each of {self.num_dofs} joints")
        self.default_pos = np.asarray(dof.default_pos)

        policy_file = resolve_repo_path(cfg_policy.policy_file)
        logger.debug("Loading TorchScript policy from %s", policy_file)
        self.model = torch.jit.load(policy_file, map_location=device)
        # The checkpoint keeps its LSTM hidden/cell state as in-module buffers. Snapshot the
        # saved initial state so reset() can restore it before the warm-up wash below; the
        # zero-input response of the recurrent state does not converge within the warm-up
        # horizon, so restoring the snapshot is the only way to make every session start
        # from the exact state the reference LeggedLabDeploy launcher boots with.
        self._initial_hidden_state = self.model.hidden_state.detach().clone()
        self._initial_cell_state = self.model.cell_state.detach().clone()
        self.action_scale = cfg_policy.action_scale
        self.last_action = np.zeros(self.num_dofs)

        self.observation_adapter = DoFAdapter(dof_cfg.joint_names, dof.joint_names)
        self.action_adapter = DoFAdapter(dof.joint_names, dof_cfg.joint_names)

        self.obs_scales = cfg_policy.obs_scales
        command_range = cfg_policy.command_range
        self.clip_min_command = np.asarray(
            [command_range.lin_vel_x[0], command_range.lin_vel_y[0], command_range.ang_vel_z[0]], dtype=np.float32
        )
        self.clip_max_command = np.asarray(
            [command_range.lin_vel_x[1], command_range.lin_vel_y[1], command_range.ang_vel_z[1]], dtype=np.float32
        )
        self.clip_obs = cfg_policy.clip_obs
        self.clip_action = cfg_policy.clip_action
        self.reset()

    def reset(self) -> None:
        self.last_action.fill(0.0)
        default_history = [np.zeros(dim, dtype=np.float32) for _, dim in self.HISTORY_LAYOUT]
        self.history_buf = deque([default_history] * self.HISTORY_LENGTH, maxlen=self.HISTORY_LENGTH)
        # Restore the checkpoint's saved recurrent state, then re-run the zero-observation
        # warm-up of the reference LeggedLabDeploy launcher. Every reset therefore lands on
        # the identical recurrent state a fresh LeggedLabDeploy boot would control from.
        zero_obs = torch.zeros((1, sum(dim for _, dim in self.HISTORY_LAYOUT)), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            self.model.hidden_state.copy_(self._initial_hidden_state)
            self.model.cell_state.copy_(self._initial_cell_state)
            for _ in range(self.RESET_WARMUP_STEPS):
                self.model(zero_obs)

    def get_observation(self, env_data, control_data) -> np.ndarray:
        axes = control_data["axes"]
        commands = np.asarray([axes["LeftY"], -axes["LeftX"], -axes["RightX"]], dtype=np.float32)
        commands = np.clip(commands, self.clip_min_command, self.clip_max_command)

        gravity_orientation = get_gravity_orientation(env_data.base_quat)
        obs_current = [
            env_data.base_ang_vel * self.obs_scales.ang_vel,
            gravity_orientation,
            commands,
            (env_data.dof_pos - self.default_pos) * self.obs_scales.dof_pos,
            env_data.dof_vel * self.obs_scales.dof_vel,
            self.last_action,
        ]
        self.history_buf.append(obs_current)
        history_list = [np.concatenate(items, axis=0) for items in zip(*self.history_buf, strict=True)]
        return np.concatenate(history_list, axis=0)

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        observation_tensor = torch.from_numpy(observation).unsqueeze(0).float().to(self.device)
        observation_tensor = observation_tensor.clip(-self.clip_obs, self.clip_obs)
        with torch.no_grad():
            action_tensor = self.model(observation_tensor).cpu()

        action = action_tensor.numpy().squeeze()
        action = np.clip(action, -self.clip_action, self.clip_action)
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
