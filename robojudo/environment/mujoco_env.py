import mujoco
import mujoco_viewer
import numpy as np

from robojudo.environment import Environment, env_registry
from robojudo.environment.env_cfgs import MujocoEnvCfg
from robojudo.environment.utils.mujoco_viz import MujocoVisualizer


@env_registry.register
class MujocoEnv(Environment):
    cfg_env: MujocoEnvCfg

    def __init__(self, cfg_env: MujocoEnvCfg, device: str = "cpu"):
        super().__init__(cfg_env=cfg_env, device=device)
        self.sim_duration = cfg_env.sim_duration
        self.sim_dt = cfg_env.sim_dt
        self.sim_decimation = cfg_env.sim_decimation
        self.control_dt = self.sim_dt * self.sim_decimation

        self.model = mujoco.MjModel.from_xml_path(cfg_env.xml)  # pyright: ignore[reportAttributeAccessIssue]
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)  # pyright: ignore[reportAttributeAccessIssue]
        mujoco.mj_step(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]

        self.viewer = mujoco_viewer.MujocoViewer(
            self.model,
            self.data,
            width=1200,
            height=900,
            hide_menus=True,
            disable_key_callbacks=True,
        )
        self.viewer.cam.distance = 3.0
        self.viewer.cam.elevation = -10.0
        self.viewer.cam.azimuth = 180.0
        self.visualizer = MujocoVisualizer(self.viewer) if cfg_env.visualize_extras else None
        self.update()

    def reborn(self, init_qpos=None):
        if init_qpos is None:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)  # pyright: ignore[reportAttributeAccessIssue]
        else:
            self.data.qpos[0:7] = init_qpos
            self.data.qvel[:] = 0.0
            self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
        self.update()

    def reset(self):
        self.update()

    def set_gains(self, stiffness, damping):
        assert len(stiffness) == self.num_dofs and len(damping) == self.num_dofs
        self.stiffness = np.asarray(stiffness)
        self.damping = np.asarray(damping)

    def self_check(self):
        return

    def update(self, simple: bool = False):
        self._dof_pos = self.data.qpos.astype(np.float32)[-self.num_dofs :].copy()
        self._dof_vel = self.data.qvel.astype(np.float32)[-self.num_dofs :].copy()
        if simple:
            return

        self._base_quat = self.data.qpos.astype(np.float32)[3:7][[1, 2, 3, 0]].copy()
        self._base_ang_vel = self.data.qvel.astype(np.float32)[3:6].copy()
        self._base_pos = self.data.qpos.astype(np.float32)[:3].copy()

    def step(self, pd_target):
        assert len(pd_target) == self.num_dofs, "pd_target len should be num_dofs of env"

        self.viewer.cam.lookat = self.data.qpos.astype(np.float32)[:3]
        if self.viewer.is_alive:
            self.viewer.render()

        for _ in range(self.sim_decimation):
            torque = (pd_target - self.dof_pos) * self.stiffness - self.dof_vel * self.damping
            self.data.ctrl = np.clip(torque, -self.torque_limits, self.torque_limits)
            mujoco.mj_step(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
            self.update(simple=True)
        self.update()

    def shutdown(self):
        self.viewer.close()
