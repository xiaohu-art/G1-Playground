from abc import ABC, abstractmethod

import numpy as np
from box import Box

from robojudo.tools.dof import merge_dof_cfgs
from robojudo.tools.tool_cfgs import DoFConfig

from .env_cfgs import EnvCfg


class Environment(ABC):
    def __init__(self, cfg_env: EnvCfg, device: str = "cpu"):
        self.cfg_env = cfg_env
        self.device = device
        self.update_dof_cfg()

        self._dof_pos = np.zeros(self.num_dofs, dtype=np.float32)
        self._dof_vel = np.zeros(self.num_dofs, dtype=np.float32)
        self._base_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self._base_ang_vel = np.zeros(3, dtype=np.float32)
        self._base_pos: np.ndarray | None = None
        self.visualizer = None

    def update_dof_cfg(self, override_cfg: DoFConfig | None = None):
        dof_config = self.cfg_env.dof
        if override_cfg is not None:
            dof_config = merge_dof_cfgs(dof_config, override_cfg)

        self.dof_cfg = dof_config
        self.joint_names = dof_config.joint_names
        self.num_dofs = dof_config.num_dofs
        self.default_pos = np.asarray(dof_config.default_pos)
        self.stiffness = np.asarray(dof_config.stiffness)
        self.damping = np.asarray(dof_config.damping)
        self.torque_limits = np.asarray(dof_config.torque_limits)
        self.position_limits = np.asarray(dof_config.position_limits)
        self.set_gains(self.stiffness, self.damping)

    @abstractmethod
    def self_check(self):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def update(self):
        raise NotImplementedError

    @abstractmethod
    def step(self, pd_target):
        assert len(pd_target) == self.num_dofs, "pd_target len should be num_dofs of env"
        raise NotImplementedError

    @abstractmethod
    def shutdown(self):
        raise NotImplementedError

    @abstractmethod
    def set_gains(self, stiffness, damping):
        raise NotImplementedError

    @property
    def dof_pos(self):
        return self._dof_pos.copy()

    @property
    def dof_vel(self):
        return self._dof_vel.copy()

    @property
    def base_quat(self):
        return self._base_quat.copy()

    @property
    def base_ang_vel(self):
        return self._base_ang_vel.copy()

    @property
    def base_pos(self):
        return self._base_pos.copy() if self._base_pos is not None else None

    def get_data(self):
        return Box(
            {
                "dof_pos": self.dof_pos,
                "dof_vel": self.dof_vel,
                "base_quat": self.base_quat,
                "base_ang_vel": self.base_ang_vel,
                "base_pos": self.base_pos,
            }
        )
