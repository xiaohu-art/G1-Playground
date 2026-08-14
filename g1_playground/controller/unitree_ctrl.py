from .base_ctrl import Controller
from .utils.joystick import UnitreeRemoteController


class UnitreeCtrl(Controller):
    def __init__(self, env):
        super().__init__(("LeftX", "LeftY", "RightX", "RightY"))
        remote = UnitreeRemoteController(self.state_queue, self.event_queue)
        env.remote_controller_handler = remote.parse
