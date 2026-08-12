from robojudo.config import cfg_registry
from robojudo.controller.ctrl_cfgs import JoystickCtrlCfg, UnitreeCtrlCfg
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg

from .env.g1_mujuco_env_cfg import G1MujocoEnvCfg
from .env.g1_real_env_cfg import G1RealEnvCfg, G1UnitreeCfg
from .policy.g1_unitree_policy_cfg import G1UnitreeWoGaitPolicyCfg


@cfg_registry.register
class g1(RlPipelineCfg):
    """Unitree G1 29DoF locomotion in MuJoCo."""

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg()
    ctrl: list[JoystickCtrlCfg] = [JoystickCtrlCfg()]
    policy: G1UnitreeWoGaitPolicyCfg = G1UnitreeWoGaitPolicyCfg()


@cfg_registry.register
class g1_real(g1):
    """Unitree G1 29DoF locomotion on real hardware."""

    env: G1RealEnvCfg = G1RealEnvCfg(
        target="hardware",
        unitree=G1UnitreeCfg(
            domain_id=0,
            net_if="enP8p1s0",
        ),
    )
    ctrl: list[UnitreeCtrlCfg] = [UnitreeCtrlCfg()]
    do_safety_check: bool = True
