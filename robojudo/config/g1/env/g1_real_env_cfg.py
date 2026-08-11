from typing import Literal

from robojudo.environment.env_cfgs import UnitreeCppEnvCfg

from .g1_env_cfg import G1EnvCfg


class G1UnitreeCfg(UnitreeCppEnvCfg.UnitreeCfg):
    robot: Literal["g1"] = "g1"
    msg_type: Literal["hg"] = "hg"
    hand_type: Literal["NONE"] = "NONE"
    enable_odometry: bool = False


class G1RealEnvCfg(G1EnvCfg, UnitreeCppEnvCfg):
    env_type: str = "UnitreeCppEnv"
    unitree: UnitreeCppEnvCfg.UnitreeCfg = G1UnitreeCfg(net_if="eth0")
    joint2motor_idx: list[int] | None = None
