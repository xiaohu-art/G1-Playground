import logging
from multiprocessing import Queue

from robojudo.controller import Controller, ctrl_registry
from robojudo.controller.ctrl_cfgs import UnitreeCtrlCfg
from robojudo.controller.joystick_ctrl import JoystickCtrl
from robojudo.controller.utils.joystick import UnitreeRemoteController

logger = logging.getLogger(__name__)


@ctrl_registry.register
class UnitreeCtrl(JoystickCtrl):
    cfg_ctrl: UnitreeCtrlCfg

    def __init__(self, cfg_ctrl: UnitreeCtrlCfg, env=None, device="cpu"):
        # Skip JoystickCtrl initialization
        Controller.__init__(self, cfg_ctrl=cfg_ctrl, env=env, device=device)
        self.unitree_env = env

        self.state_queue = Queue(maxsize=2)  # for axes
        self.event_queue = Queue(maxsize=100)  # for button/dpad events
        self.unitree_remote_controller = UnitreeRemoteController(self.state_queue, self.event_queue)

        self.axes_names = ["LeftX", "LeftY", "RightX", "RightY"]
        self.reset()

        if self.unitree_env is not None:
            self.unitree_env.RemoteControllerHandler = self.unitree_remote_controller.parse
        else:
            logger.warning("No Unitree env, controller not working.")
