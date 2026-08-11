import logging

from box import Box

import robojudo.controller
from robojudo.controller import Controller, CtrlCfg

logger = logging.getLogger(__name__)


class CtrlManager:
    """
    A manager that handles multiple controllers and their interactions.
    """

    def __init__(
        self,
        cfg_ctrls: list[CtrlCfg] | None = None,
        env=None,
        device="cpu",
    ):
        self.cfg_ctrls = cfg_ctrls
        self.env = env
        self.device = device

        controllers = {}
        for cfg_ctrl in self.cfg_ctrls or []:
            ctrl_type = cfg_ctrl.ctrl_type
            if ctrl_type in controllers.keys():
                logger.warning(f"Controller type {ctrl_type} already exists, skipping.")
                continue

            ctrl_class: type[Controller] = getattr(robojudo.controller, ctrl_type)
            controller: Controller = ctrl_class(cfg_ctrl=cfg_ctrl, env=self.env, device=self.device)
            controllers[ctrl_type] = {
                "inst": controller,
                "cfg": cfg_ctrl,
                # "triggers": []
            }

        self.controllers = Box(controllers)

    def reset(self):
        """
        Reset all controllers.
        """
        for controller in self.controllers.values():
            controller.inst.reset()

    def post_step_callback(self, ctrl_data: Box):
        """
        Call post step callback for all controllers.
        """
        # self.process_triggers()
        commands = ctrl_data.get("COMMANDS", [])
        for controller in self.controllers.values():
            controller.inst.post_step_callback(commands)

    def get_ctrl_data(self, env_data):
        ctrl_data_all = {}
        ctrl_commands_all = set()
        for ctrl_type, controller in self.controllers.items():
            ctrl_data = controller.inst.get_data()
            ctrl_data_triggered, ctrl_commands = controller.inst.process_triggers(ctrl_data)

            ctrl_data_all[ctrl_type] = ctrl_data_triggered
            ctrl_commands_all.update(ctrl_commands)

        ctrl_data_all["COMMANDS"] = list(ctrl_commands_all)
        return Box(ctrl_data_all)
