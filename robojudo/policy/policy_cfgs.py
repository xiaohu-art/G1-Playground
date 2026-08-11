from pydantic import field_validator, model_validator

from robojudo.config import ASSETS_DIR, Config
from robojudo.tools.tool_cfgs import DoFConfig


class PolicyCfg(Config):
    policy_type: str
    robot: str

    @property
    def policy_file(self) -> str:
        return (ASSETS_DIR / f"models/{self.robot}/PLACEHOLDER.pt").as_posix()

    freq: int = 50
    obs_dof: DoFConfig
    action_dof: DoFConfig

    action_scale: float = 1.0
    action_clip: float | None = None
    action_beta: float = 1.0
    history_length: int = 0

    @property
    def history_obs_size(self) -> int:
        return 0

    @field_validator("action_scale", "action_clip")
    def check_action_scale(cls, value):
        if value is not None and value <= 0:
            raise ValueError("action_scale must be positive")
        return value

    @model_validator(mode="after")
    def check_history(self):
        if self.history_length < 0:
            raise ValueError("history_length cannot be negative")
        if self.history_obs_size < 0:
            raise ValueError("history_obs_size cannot be negative")
        return self


class UnitreeWoGaitPolicyCfg(PolicyCfg):
    class ObsScalesCfg(Config):
        ang_vel: float = 0.2
        gravity: float = 1.0
        dof_pos: float = 1.0
        dof_vel: float = 0.05
        command: list[float] = [1.0, 1.0, 1.0]

    policy_type: str = "UnitreeWoGaitPolicy"
    policy_name: str = "policy_wo_gait"

    @property
    def policy_file(self) -> str:
        return (ASSETS_DIR / f"models/{self.robot}/unitree/{self.policy_name}.pt").as_posix()

    action_scale: float = 0.25
    action_beta: float = 1.0
    history_length: int = 5
    history_obs_dims: dict[str, int] = {}

    obs_scales: ObsScalesCfg = ObsScalesCfg()
    max_cmd: list[float] = [0.8, 0.5, 1.57]
    commands_map: list[list[float]] = [
        [-1.0, 0.0, 1.0],
        [1.0, 0.0, -1.0],
        [1.0, 0.0, -1.0],
    ]
