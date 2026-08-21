import logging
import time
from queue import Queue
from threading import Event, Thread

logger = logging.getLogger(__name__)

PUBLISH_HZ = 100.0
SHIFT_KEYS = frozenset({"shift", "shift_l", "shift_r"})
SHUTDOWN_BUTTON = "A"


class KeyboardThread(Thread):
    def __init__(
        self,
        state_queue: Queue,
        event_queue: Queue,
        axes_names,
        forward_axis: float,
        backward_axis: float,
        strafe_axis: float,
        turn_axis: float,
    ):
        super().__init__(name="KeyboardThread", daemon=True)
        self.state_queue = state_queue
        self.event_queue = event_queue
        self.axes_names = tuple(axes_names)
        self.forward_axis = float(forward_axis)
        self.backward_axis = float(backward_axis)
        self.strafe_axis = float(strafe_axis)
        self.turn_axis = float(turn_axis)

        self.pressed_keys: set[str] = set()
        self.running = True
        self.ready = Event()
        self.startup_error: BaseException | None = None
        self.last_publish_time = time.monotonic()

    @staticmethod
    def key_name(key) -> str:
        character = getattr(key, "char", None)
        if character:
            return character
        return str(getattr(key, "name", key))

    def axes(self) -> dict:
        pressed = set(self.pressed_keys)
        shifted = bool(pressed & SHIFT_KEYS)
        values = dict.fromkeys(self.axes_names, 0.0)

        if "up" in pressed:
            values["LeftY"] += self.forward_axis
        if "down" in pressed:
            values["LeftY"] -= self.backward_axis
        if "left" in pressed:
            if shifted:
                values["LeftX"] -= self.strafe_axis
            else:
                values["RightX"] -= self.turn_axis
        if "right" in pressed:
            if shifted:
                values["LeftX"] += self.strafe_axis
            else:
                values["RightX"] += self.turn_axis
        return values

    def publish(self) -> None:
        while self.state_queue.full():
            self.state_queue.get()
        self.last_publish_time = time.monotonic()
        self.state_queue.put({"type": "axes", "axes": self.axes(), "timestamp": self.last_publish_time})

    def run(self):
        try:
            from pynput import keyboard

            listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
            listener.start()
            listener.wait()
            logger.info("[Keyboard] arrows drive locomotion, shift+arrows strafe, esc requests shutdown")
        except BaseException as error:
            self.startup_error = error
            self.running = False
            return
        finally:
            self.ready.set()

        period = 1.0 / PUBLISH_HZ
        try:
            while self.running and listener.is_alive():
                self.publish()
                time.sleep(period)
        finally:
            listener.stop()

    def on_press(self, key) -> None:
        name = self.key_name(key)
        if name == "esc":
            self.event_queue.put({"type": "button", "name": SHUTDOWN_BUTTON, "pressed": True, "timestamp": time.time()})
            return
        self.pressed_keys.add(name)

    def on_release(self, key) -> None:
        self.pressed_keys.discard(self.key_name(key))
