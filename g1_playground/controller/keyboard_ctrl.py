import time

from .base_ctrl import Controller
from .utils.keyboard import KeyboardThread

KEYBOARD_AXES = ("LeftX", "LeftY", "RightX", "RightY", "LT", "RT")


class KeyboardCtrl(Controller):
    def __init__(
        self,
        env=None,
        forward_axis: float = 0.5,
        backward_axis: float = 0.5,
        strafe_axis: float = 0.4,
        turn_axis: float = 0.35,
        stale_timeout: float = 0.2,
    ):
        super().__init__(KEYBOARD_AXES)
        self.stale_timeout = float(stale_timeout)
        self.keyboard_thread = KeyboardThread(
            self.state_queue,
            self.event_queue,
            KEYBOARD_AXES,
            forward_axis=forward_axis,
            backward_axis=backward_axis,
            strafe_axis=strafe_axis,
            turn_axis=turn_axis,
        )
        self.keyboard_thread.start()
        self.keyboard_thread.ready.wait()
        if self.keyboard_thread.startup_error is not None:
            raise RuntimeError("Failed to initialize keyboard input") from self.keyboard_thread.startup_error

    def read(self) -> tuple[dict, bool]:
        control, shutdown_requested = super().read()
        if time.monotonic() - self.keyboard_thread.last_publish_time > self.stale_timeout:
            return {"axes": dict.fromkeys(KEYBOARD_AXES, 0.0)}, shutdown_requested
        return control, shutdown_requested
