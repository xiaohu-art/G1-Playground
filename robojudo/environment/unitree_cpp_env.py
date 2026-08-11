import logging
import time

import numpy as np
from unitree_cpp import RobotState, UnitreeController  # type: ignore

from robojudo.environment import Environment, env_registry
from robojudo.environment.env_cfgs import UnitreeCppEnvCfg

logger = logging.getLogger(__name__)


@env_registry.register
class UnitreeCppEnv(Environment):
    cfg_env: UnitreeCppEnvCfg

    def __init__(self, cfg_env: UnitreeCppEnvCfg, device: str = "cpu"):
        self.enabled = cfg_env.act
        super().__init__(cfg_env=cfg_env, device=device)
        self.RemoteControllerHandler = None
        self._dof_idx = cfg_env.joint2motor_idx

        unitree_cfg = cfg_env.unitree.to_dict()
        unitree_cfg["num_dofs"] = self.num_dofs
        unitree_cfg["stiffness"] = self.stiffness
        unitree_cfg["damping"] = self.damping
        self.robot_state: RobotState | None = None
        self.unitree = UnitreeController(unitree_cfg)
        self.self_check()

    def self_check(self):
        for _ in range(30):
            if self.unitree.self_check():
                logger.info("UnitreeCppEnv self check passed")
                return
            time.sleep(0.1)
        raise RuntimeError("UnitreeCppEnv self check failed")

    def reset(self):
        self.update()

    def update(self):
        self.robot_state = self.unitree.get_robot_state()
        motor_state = self.robot_state.motor_state
        if self._dof_idx is None:
            self._dof_pos = np.asarray(motor_state.q, dtype=np.float32)
            self._dof_vel = np.asarray(motor_state.dq, dtype=np.float32)
        else:
            self._dof_pos = np.asarray([motor_state.q[index] for index in self._dof_idx], dtype=np.float32)
            self._dof_vel = np.asarray([motor_state.dq[index] for index in self._dof_idx], dtype=np.float32)

        self._base_quat = np.asarray(self.robot_state.imu_state.quaternion, dtype=np.float32)[[1, 2, 3, 0]]
        self._base_ang_vel = np.asarray(self.robot_state.imu_state.gyroscope, dtype=np.float32)
        if self.RemoteControllerHandler is not None:
            self.RemoteControllerHandler(self.robot_state.wireless_remote)

    def step(self, pd_target):
        assert len(pd_target) == self.num_dofs, "pd_target len should be num_dofs of env"
        if self.enabled:
            self.unitree.step(np.asarray(pd_target).tolist())

    def shutdown(self):
        self.enabled = False
        self.unitree.shutdown()

    def set_gains(self, stiffness, damping):
        if not hasattr(self, "unitree") or not self.enabled:
            return
        self.unitree.set_gains(stiffness, damping)
