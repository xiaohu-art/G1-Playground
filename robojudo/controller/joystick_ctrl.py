import time
from queue import Empty, Queue

from robojudo.controller import Controller, ctrl_registry
from robojudo.controller.ctrl_cfgs import JoystickCtrlCfg
from robojudo.controller.utils.joystick import JoystickThread


@ctrl_registry.register
class JoystickCtrl(Controller):
    cfg_ctrl: JoystickCtrlCfg

    def __init__(self, cfg_ctrl: JoystickCtrlCfg, env=None, device="cpu"):
        super().__init__(cfg_ctrl=cfg_ctrl, env=env, device=device)

        self.state_queue = Queue(maxsize=2)  # for axes
        self.event_queue = Queue(maxsize=100)  # for button/dpad events
        self.joystick_thread = JoystickThread(self.state_queue, self.event_queue)
        self.joystick_thread.start()

        self.axes_names = self.joystick_thread.config["axis_config"]["axis_map"].keys()
        self.reset()

    def reset(self):
        while not self.state_queue.empty():
            try:
                self.state_queue.get_nowait()
            except Empty:
                break

        while not self.event_queue.empty():
            try:
                self.event_queue.get_nowait()
            except Empty:
                break

        self.last_state = {
            "type": "axes",
            "axes": {name: 0.0 for name in self.axes_names},
            "timestamp": time.time(),
        }

    def get_state(self):
        try:
            state = self.state_queue.get_nowait()
            self.last_state = state.copy()
        except Empty:
            state = self.last_state

        return state

    def get_events(self):
        events = []
        while not self.event_queue.empty():
            try:
                event = self.event_queue.get_nowait()
                events.append(event)
            except Empty:
                break
        return events

    def get_data(self):
        state = self.get_state()
        events = self.get_events()

        return {
            "axes": state["axes"],
            "button_event": events,
        }

    def process_triggers(self, ctrl_data):
        commands = []
        if len(self.triggers) == 0:
            return ctrl_data, commands

        for event in ctrl_data["button_event"]:
            if event["type"] != "button" or not event["pressed"]:
                continue
            command = self.triggers.get(event["name"])
            if command is not None:
                commands.append(command)

        return ctrl_data, commands
