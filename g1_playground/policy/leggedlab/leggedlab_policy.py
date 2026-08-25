import logging
from collections import deque
from types import SimpleNamespace

import numpy as np
from omegaconf import DictConfig

from g1_playground.policy.base_policy import BasePolicy
from g1_playground.policy.tensorrt_runner import TensorRTRunner
from g1_playground.utils import resolve_repo_path
from g1_playground.utils.dof import DoFAdapter
from g1_playground.utils.math import get_gravity_orientation

logger = logging.getLogger(__name__)


class LeggedLabPolicy(BasePolicy):
    """Locomotion policy for the recurrent LeggedLab G1 network (96 -> 29).

    Observation/action semantics mirror ``LeggedLabDeploy/deploy.py::run()`` from
    https://github.com/Hellod035/LeggedLabDeploy (BSD-3-Clause): unit observation
    scales, remote commands clipped to ``command_range`` (no deadzone), observation
    and action clipping to +-``clip_obs``/``clip_action``. The exported TensorRT
    graph takes and returns its recurrent LSTM state explicitly.
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

    def __init__(self, cfg_policy: DictConfig, dof_cfg: DictConfig, runner=None):
        dof = cfg_policy.dof
        super().__init__()

        self.num_dofs = len(dof.joint_names)
        for field in ("default_pos", "stiffness", "damping"):
            if len(getattr(dof, field)) != self.num_dofs:
                raise ValueError(f"LeggedLabPolicy requires one {field} value for each of {self.num_dofs} joints")
        self.default_pos = np.asarray(dof.default_pos)

        policy_file = resolve_repo_path(cfg_policy.policy_file)
        logger.debug("Loading TensorRT locomotion policy from %s", policy_file)
        self.runner = runner or TensorRTRunner(policy_file)
        self._resolve_signature()
        self.action_scale = cfg_policy.action_scale
        self.last_action = np.zeros(self.num_dofs, dtype=np.float32)

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

    def _resolve_signature(self) -> None:
        expected_inputs = ("obs", "hidden_state", "cell_state")
        expected_outputs = ("actions", "next_hidden_state", "next_cell_state")
        if self.runner.input_names != expected_inputs or self.runner.output_names != expected_outputs:
            raise ValueError(
                f"LeggedLab policy requires {expected_inputs} -> {expected_outputs}, "
                f"got {self.runner.input_names} -> {self.runner.output_names}"
            )
        if self.runner.shape("obs") != (1, 96) or self.runner.shape("actions") != (1, self.num_dofs):
            raise ValueError("LeggedLab TensorRT tensor dimensions do not match the policy configuration")
        hidden_shape = self.runner.shape("hidden_state")
        if (
            hidden_shape != self.runner.shape("cell_state")
            or hidden_shape != self.runner.shape("next_hidden_state")
            or hidden_shape != self.runner.shape("next_cell_state")
        ):
            raise ValueError("LeggedLab TensorRT recurrent state shapes disagree")
        self._state_shape = hidden_shape

    def reset(self) -> None:
        self.last_action.fill(0.0)
        default_history = [np.zeros(dim, dtype=np.float32) for _, dim in self.HISTORY_LAYOUT]
        self.history_buf = deque([default_history] * self.HISTORY_LENGTH, maxlen=self.HISTORY_LENGTH)
        self._hidden_state = np.zeros(self._state_shape, dtype=np.float32)
        self._cell_state = np.zeros(self._state_shape, dtype=np.float32)
        zero_obs = np.zeros((1, sum(dim for _, dim in self.HISTORY_LAYOUT)), dtype=np.float32)
        for _ in range(self.RESET_WARMUP_STEPS):
            self._infer(zero_obs)

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
        observation = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        action = self._infer(np.clip(observation, -self.clip_obs, self.clip_obs)).reshape(-1)
        action = np.clip(action, -self.clip_action, self.clip_action)
        self.last_action = action.copy()
        return action * self.action_scale

    def _infer(self, observation: np.ndarray) -> np.ndarray:
        outputs = self.runner.run(
            {"obs": observation, "hidden_state": self._hidden_state, "cell_state": self._cell_state}
        )
        self._hidden_state = outputs["next_hidden_state"]
        self._cell_state = outputs["next_cell_state"]
        return outputs["actions"]

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
