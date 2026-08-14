from queue import Empty, Queue

SHUTDOWN_BUTTON = "A"


class Controller:
    """Consume normalized axes and button edges from one input source."""

    def __init__(self, axes_names):
        self.state_queue = Queue(maxsize=2)
        self.event_queue = Queue(maxsize=100)
        self._axes_names = tuple(axes_names)
        self._last_axes = {name: 0.0 for name in self._axes_names}

    def read(self) -> tuple[dict, bool]:
        """Return the latest axes and whether an A-button press requested shutdown."""

        while True:
            try:
                state = self.state_queue.get_nowait()
            except Empty:
                break
            self._last_axes = state["axes"].copy()

        shutdown_requested = False
        while True:
            try:
                event = self.event_queue.get_nowait()
            except Empty:
                break
            if event["type"] == "button" and event["name"] == SHUTDOWN_BUTTON and event["pressed"]:
                shutdown_requested = True

        return {"axes": self._last_axes.copy()}, shutdown_requested
