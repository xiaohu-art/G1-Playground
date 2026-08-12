from typing import Literal

from pydantic import StrictInt

from robojudo.environment.env_cfgs import UnitreeCppEnvCfg

from .g1_env_cfg import G1EnvCfg


class G1UnitreeCfg(UnitreeCppEnvCfg.UnitreeCfg):
    domain_id: StrictInt = 0
    robot: Literal["g1"] = "g1"
    msg_type: Literal["hg"] = "hg"
    hand_type: Literal["NONE"] = "NONE"
    enable_odometry: bool = False


class G1RealEnvCfg(G1EnvCfg, UnitreeCppEnvCfg):
    env_type: str = "UnitreeCppEnv"
    target: Literal["hardware"] = "hardware"
    unitree: G1UnitreeCfg
    joint2motor_idx: list[int] | None = None
