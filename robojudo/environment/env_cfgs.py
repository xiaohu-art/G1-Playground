from typing import Literal

from pydantic import model_validator

from robojudo.config import Config
from robojudo.tools.tool_cfgs import DoFConfig


class EnvCfg(Config):
    env_type: str
    is_sim: bool = False
    xml: str
    dof: DoFConfig


class MujocoEnvCfg(EnvCfg):
    env_type: str = "MujocoEnv"
    is_sim: bool = True
    sim_duration: float = 60.0
    sim_dt: float = 0.001
    sim_decimation: int = 20
    visualize_extras: bool = True


class UnitreeCppEnvCfg(EnvCfg):
    """Configuration for the Unitree C++ low-level controller."""

    class UnitreeCfg(Config):
        net_if: str = "eth0"
        robot: Literal["g1"] = "g1"
        msg_type: Literal["hg"] = "hg"
        control_mode: Literal["position"] = "position"
        hand_type: Literal["NONE"] = "NONE"
        lowcmd_topic: str = "rt/lowcmd"
        lowstate_topic: str = "rt/lowstate"
        enable_odometry: bool = False
        sport_state_topic: str = "rt/odommodestate"
        control_dt: float = 0.02

    env_type: str = "UnitreeCppEnv"
    act: bool = True
    unitree: UnitreeCfg
    joint2motor_idx: list[int] | None = None

    @model_validator(mode="after")
    def check_joint2motor_idx(self):
        if self.joint2motor_idx is not None and len(self.joint2motor_idx) != self.dof.num_dofs:
            raise ValueError("joint2motor_idx length must match dof.num_dofs")
        return self
