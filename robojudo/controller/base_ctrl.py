from abc import ABC, abstractmethod

from robojudo.environment import Environment

from .ctrl_cfgs import CtrlCfg


class Controller(ABC):
    """
    Base Controller Module
    """

    def __init__(self, cfg_ctrl: CtrlCfg, env: Environment | None = None, device: str = "cpu"):
        self.cfg_ctrl = cfg_ctrl
        self.env = env  # type: ignore
        self.device = device

        self.triggers: dict = cfg_ctrl.triggers.copy()

    @abstractmethod
    def get_data(self):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        pass

    def post_step_callback(self, commands: list[str] | None = None):
        return

    def process_triggers(self, ctrl_data):
        commands = []
        if len(self.triggers) == 0:
            return ctrl_data, commands
        return ctrl_data, commands
