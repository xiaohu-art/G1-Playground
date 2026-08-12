from typing import Literal

from pydantic import StrictInt, model_validator

from robojudo.config import Config
from robojudo.tools.tool_cfgs import DoFConfig


def validate_dds_endpoint(target: str, domain_id: object, net_if: object) -> None:
    if type(domain_id) is not int:
        raise ValueError("DDS domain_id must be an integer")
    if not isinstance(net_if, str) or not net_if or net_if != net_if.strip():
        raise ValueError("DDS net_if must be a non-empty interface name without surrounding whitespace")
    if target == "simulation":
        if domain_id != 1 or net_if != "lo":
            raise ValueError("simulation DDS endpoint must use domain_id=1 and net_if='lo'")
        return
    if target == "hardware":
        if domain_id != 0 or net_if == "lo":
            raise ValueError("hardware DDS endpoint must use domain_id=0 and a non-loopback net_if")
        return
    raise ValueError(f"unsupported DDS target: {target!r}")


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
        domain_id: StrictInt
        net_if: str
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
    target: Literal["simulation", "hardware"]
    act: bool = True
    unitree: UnitreeCfg
    joint2motor_idx: list[int] | None = None

    @model_validator(mode="after")
    def check_environment(self):
        validate_dds_endpoint(self.target, self.unitree.domain_id, self.unitree.net_if)
        if self.joint2motor_idx is not None and len(self.joint2motor_idx) != self.dof.num_dofs:
            raise ValueError("joint2motor_idx length must match dof.num_dofs")
        return self
