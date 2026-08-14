from .base_ctrl import Controller
from .utils.joystick import JoystickThread

JOYSTICK_AXES = ("LeftX", "LeftY", "RightX", "RightY", "LT", "RT")


class JoystickCtrl(Controller):
    def __init__(self, env=None):
        super().__init__(JOYSTICK_AXES)
        self.joystick_thread = JoystickThread(self.state_queue, self.event_queue)
        self.joystick_thread.start()
        self.joystick_thread.ready.wait()
        if self.joystick_thread.startup_error is not None:
            raise RuntimeError("Failed to initialize joystick input") from self.joystick_thread.startup_error
