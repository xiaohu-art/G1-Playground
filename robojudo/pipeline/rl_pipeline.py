import logging
import time

import numpy as np
from box import Box

import robojudo.environment
import robojudo.policy
from robojudo.controller import CtrlManager
from robojudo.environment import Environment
from robojudo.pipeline import Pipeline, pipeline_registry
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg
from robojudo.policy import Policy, PolicyCfg
from robojudo.tools.dof import DoFAdapter
from robojudo.tools.tool_cfgs import DoFConfig
from robojudo.utils.progress import ProgressBar
from robojudo.utils.util_func import get_gravity_orientation

logger = logging.getLogger(__name__)


class PolicyWrapper:
    """Adapt the 29 joints between environment and policy ordering."""

    def __init__(self, cfg_policy: PolicyCfg, env_dof_cfg: DoFConfig, device: str):
        policy_class: type[Policy] = getattr(robojudo.policy, cfg_policy.policy_type)
        self.policy = policy_class(cfg_policy=cfg_policy, device=device)
        self.env_dof_cfg = env_dof_cfg
        self.obs_adapter = DoFAdapter(env_dof_cfg.joint_names, self.policy.cfg_obs_dof.joint_names)
        self.action_adapter = DoFAdapter(self.policy.cfg_action_dof.joint_names, env_dof_cfg.joint_names)

    def get_observation(self, env_data: Box, ctrl_data: Box):
        adapted = env_data.copy()
        adapted.dof_pos = self.obs_adapter.fit(adapted.dof_pos)
        adapted.dof_vel = self.obs_adapter.fit(adapted.dof_vel)
        return self.policy.get_observation(adapted, ctrl_data)

    def get_pd_target(self, obs: np.ndarray) -> np.ndarray:
        action = self.policy.get_action(obs)
        target = action + self.policy.action_default_pos
        return self.action_adapter.fit(target, template=self.env_dof_cfg.default_pos)

    def get_init_dof_pos(self) -> np.ndarray:
        return self.action_adapter.fit(
            self.policy.get_init_dof_pos(),
            template=self.env_dof_cfg.default_pos,
        )

    def __getattr__(self, name):
        return getattr(self.policy, name)


@pipeline_registry.register
class RlPipeline(Pipeline):
    cfg: RlPipelineCfg

    def __init__(self, cfg: RlPipelineCfg):
        super().__init__(cfg=cfg)

        env_class: type[Environment] = getattr(robojudo.environment, cfg.env.env_type)
        self.env = env_class(cfg_env=cfg.env, device=self.device)
        self.ctrl_manager = CtrlManager(cfg_ctrls=cfg.ctrl, env=self.env, device=self.device)
        self.policy = PolicyWrapper(cfg.policy, self.env.dof_cfg, self.device)

        self.env.update_dof_cfg(override_cfg=self.policy.cfg_action_dof)
        self.visualizer = self.env.visualizer
        self.freq = cfg.policy.freq
        self.dt = 1.0 / self.freq
        self.running = True

        self.reset()
        self.self_check()
        self.policy.reset()

    def self_check(self):
        self.env.self_check()
        for _ in range(10):
            self.step(dry_run=True)

    def reset(self):
        logger.info("Pipeline reset")
        self.timestep = 0
        self.env.reset()
        self.policy.reset()
        self.ctrl_manager.reset()

    def safety_check(self):
        if not self.do_safety_check:
            return
        gravity = get_gravity_orientation(self.env.base_quat)
        tilt = np.arccos(np.clip(-gravity[2], -1.0, 1.0))
        if abs(tilt) <= 1.0:
            return
        if hasattr(self.env, "reborn"):
            logger.error("Robot fallen! Resetting simulation.")
            self.env.reborn()
            self.policy.reset()
        else:
            logger.error("Robot fallen! Shutting down for safety.")
            self.shutdown()

    def shutdown(self):
        if not self.running:
            return
        self.running = False
        self.env.shutdown()

    def post_step_callback(self, env_data, ctrl_data, extras):
        self.timestep += 1
        commands = ctrl_data.get("COMMANDS", [])
        for command in commands:
            match command:
                case "[SHUTDOWN]":
                    logger.warning("Emergency shutdown!")
                    self.shutdown()
                case "[SIM_REBORN]":
                    if hasattr(self.env, "reborn"):
                        logger.warning("Reborn simulation environment")
                        self.env.reborn()
                        self.policy.reset()

        self.ctrl_manager.post_step_callback(ctrl_data)
        self.policy.post_step_callback(commands)
        if self.visualizer is not None:
            self.policy.debug_viz(self.visualizer, env_data, ctrl_data, extras)
        self.safety_check()

    def step(self, dry_run: bool = False):
        self.env.update()
        env_data = self.env.get_data()
        ctrl_data = self.ctrl_manager.get_ctrl_data(env_data)
        commands = ctrl_data.get("COMMANDS", [])
        if commands:
            logger.info("Commands: %s", commands)

        obs, extras = self.policy.get_observation(env_data, ctrl_data)
        pd_target = self.policy.get_pd_target(obs)
        if not dry_run:
            self.env.step(pd_target)
        self.post_step_callback(env_data, ctrl_data, extras)

    def _pace(self, last_step_time: float) -> float:
        remaining = last_step_time + self.dt - time.time()
        if remaining > 0:
            time.sleep(remaining)
        elif remaining < -self.dt:
            logger.warning("Control frame dropped during preparation")
        return time.time()

    def _preparation_state(self):
        self.env.update()
        env_data = self.env.get_data()
        ctrl_data = self.ctrl_manager.get_ctrl_data(env_data)
        if "[SHUTDOWN]" in ctrl_data.get("COMMANDS", []):
            logger.warning("Software shutdown requested during preparation")
            self.shutdown()
        self.ctrl_manager.post_step_callback(ctrl_data)
        self.safety_check()
        for controller_name in ("JoystickCtrl", "UnitreeCtrl"):
            if controller_name in ctrl_data:
                ctrl_data[controller_name]["axes"] = {axis: 0.0 for axis in ctrl_data[controller_name]["axes"]}
        return env_data, ctrl_data

    def prepare(self, init_motor_angle=None, prepare_seconds=None):
        """Ramp to the standing pose, then blend into closed-loop policy output."""

        self.env.update()
        initial = np.asarray(self.env.dof_pos, dtype=np.float32)
        desired = (
            np.asarray(init_motor_angle, dtype=np.float32)
            if init_motor_angle is not None
            else np.asarray(self.policy.get_init_dof_pos(), dtype=np.float32)
        )
        ramp_duration = 3.0 if prepare_seconds is None else prepare_seconds
        ramp_steps = max(int(ramp_duration * self.freq), 1)

        logger.warning("Ramping to the policy standing pose over %.1f seconds", ramp_duration)
        progress = ProgressBar("Prepare pose", ramp_steps)
        last_step_time = time.time()
        for step in range(ramp_steps):
            self._preparation_state()
            if not self.running:
                progress.close()
                return
            alpha = (step + 1) / ramp_steps
            self.env.step((1.0 - alpha) * initial + alpha * desired)
            last_step_time = self._pace(last_step_time)
            progress.update()
        progress.close()

        self.reset()
        blend_duration = 5.0 if prepare_seconds is None else prepare_seconds
        blend_steps = max(int(blend_duration * self.freq), 1)
        logger.warning("Blending into closed-loop locomotion over %.1f seconds", blend_duration)
        progress = ProgressBar("Blend policy", blend_steps)
        last_step_time = time.time()
        for step in range(blend_steps):
            env_data, ctrl_data = self._preparation_state()
            if not self.running:
                progress.close()
                return
            observation, _ = self.policy.get_observation(env_data, ctrl_data)
            policy_target = self.policy.get_pd_target(observation)
            alpha = (step + 1) / blend_steps
            self.env.step((1.0 - alpha) * desired + alpha * policy_target)
            last_step_time = self._pace(last_step_time)
            progress.update()
        progress.close()
