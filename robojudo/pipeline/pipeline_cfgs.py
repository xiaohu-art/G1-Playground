from typing import Any

from robojudo.config import Config
from robojudo.controller import CtrlCfg
from robojudo.environment import EnvCfg
from robojudo.policy import PolicyCfg


class PipelineCfg(Config):
    pipeline_type: str
    device: str = "cpu"
    run_fullspeed: bool = False
    do_safety_check: bool = False


class RlPipelineCfg(PipelineCfg):
    pipeline_type: str = "RlPipeline"
    robot: str
    env: EnvCfg | Any
    ctrl: list[CtrlCfg | Any] = []
    policy: PolicyCfg | Any
