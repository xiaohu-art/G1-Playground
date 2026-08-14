import struct
import unittest
from queue import Queue
from types import SimpleNamespace
from unittest.mock import patch

from g1_playground.controller.base_ctrl import Controller
from g1_playground.controller.joystick_ctrl import JOYSTICK_AXES, JoystickCtrl
from g1_playground.controller.unitree_ctrl import UnitreeCtrl
from g1_playground.controller.utils.joystick import JoystickThread


class TestController(unittest.TestCase):
    def test_read_returns_latest_axes_and_consumes_a_press_once(self):
        controller = Controller(("LeftX", "LeftY"))
        controller.state_queue.put({"axes": {"LeftX": 0.1, "LeftY": 0.2}})
        controller.state_queue.put({"axes": {"LeftX": 0.3, "LeftY": 0.4}})
        controller.event_queue.put({"type": "button", "name": "B", "pressed": True})
        controller.event_queue.put({"type": "button", "name": "A", "pressed": False})
        controller.event_queue.put({"type": "button", "name": "A", "pressed": True})

        control, shutdown_requested = controller.read()

        self.assertEqual(control, {"axes": {"LeftX": 0.3, "LeftY": 0.4}})
        self.assertTrue(shutdown_requested)
        self.assertEqual(controller.read(), (control, False))

    def test_joystick_controller_waits_for_device_probe(self):
        calls = []

        class Ready:
            def wait(self):
                calls.append("wait")

        class FakeJoystickThread:
            def __init__(self, state_queue, event_queue):
                self.ready = Ready()
                self.startup_error = None

            def start(self):
                calls.append("start")

        with patch("g1_playground.controller.joystick_ctrl.JoystickThread", FakeJoystickThread):
            controller = JoystickCtrl()

        self.assertEqual(calls, ["start", "wait"])
        self.assertEqual(controller.read(), ({"axes": {name: 0.0 for name in JOYSTICK_AXES}}, False))
        self.assertFalse(hasattr(controller, "env"))
        self.assertFalse(hasattr(controller, "device"))

    def test_joystick_thread_signals_ready_after_no_device_probe(self):
        calls = []

        class Ready:
            def set(self):
                calls.append("ready")

        pygame = SimpleNamespace(
            init=lambda: calls.append("init"),
            joystick=SimpleNamespace(get_count=lambda: calls.append("probe") or 0),
        )
        joystick_thread = JoystickThread(Queue(maxsize=2), Queue(maxsize=100))
        joystick_thread.ready = Ready()

        with patch.dict("sys.modules", {"pygame": pygame}):
            joystick_thread.run()

        self.assertEqual(calls, ["init", "probe", "ready"])
        self.assertFalse(joystick_thread.running)

    def test_joystick_startup_error_is_signaled_and_raised_by_controller(self):
        startup_error = OSError("pygame startup failed")

        class Ready:
            def wait(self):
                pass

        class FakeJoystickThread:
            def __init__(self, state_queue, event_queue):
                self.ready = Ready()
                self.startup_error = startup_error

            def start(self):
                pass

        with (
            patch("g1_playground.controller.joystick_ctrl.JoystickThread", FakeJoystickThread),
            self.assertRaises(RuntimeError) as raised,
        ):
            JoystickCtrl()

        self.assertIs(raised.exception.__cause__, startup_error)

    def test_joystick_thread_signals_startup_error(self):
        calls = []
        startup_error = OSError("pygame startup failed")

        class Ready:
            def set(self):
                calls.append("ready")

        def fail_startup():
            calls.append("init")
            raise startup_error

        joystick_thread = JoystickThread(Queue(maxsize=2), Queue(maxsize=100))
        joystick_thread.ready = Ready()

        with patch.dict("sys.modules", {"pygame": SimpleNamespace(init=fail_startup)}):
            joystick_thread.run()

        self.assertEqual(calls, ["init", "ready"])
        self.assertIs(joystick_thread.startup_error, startup_error)
        self.assertFalse(joystick_thread.running)

    def test_unitree_is_an_independent_standard_queue_input_source(self):
        env = SimpleNamespace(remote_controller_handler=None)
        controller = UnitreeCtrl(env)
        packet = bytearray(40)
        struct.pack_into("<H", packet, 2, 1 << 8)
        struct.pack_into("<f", packet, 4, 0.1)
        struct.pack_into("<f", packet, 8, 0.2)
        struct.pack_into("<f", packet, 12, 0.3)
        struct.pack_into("<f", packet, 20, 0.4)

        env.remote_controller_handler(bytes(packet))
        control, shutdown_requested = controller.read()

        self.assertIsInstance(controller.state_queue, Queue)
        self.assertIsInstance(controller.event_queue, Queue)
        self.assertNotIsInstance(controller, JoystickCtrl)
        self.assertFalse(hasattr(controller, "env"))
        self.assertFalse(hasattr(controller, "device"))
        self.assertTrue(shutdown_requested)
        self.assertAlmostEqual(control["axes"]["LeftX"], 0.1)
        self.assertAlmostEqual(control["axes"]["RightX"], 0.2)
        self.assertAlmostEqual(control["axes"]["RightY"], 0.3)
        self.assertAlmostEqual(control["axes"]["LeftY"], 0.4)


if __name__ == "__main__":
    unittest.main()
