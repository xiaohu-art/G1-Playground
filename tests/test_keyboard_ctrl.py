import time
import unittest
from queue import Queue
from types import SimpleNamespace

from g1_playground.controller.keyboard_ctrl import KEYBOARD_AXES, KeyboardCtrl
from g1_playground.controller.utils.keyboard import KeyboardThread

FORWARD, BACKWARD, STRAFE, TURN = 0.5, 0.5, 0.4, 0.35


def make_thread():
    return KeyboardThread(
        Queue(maxsize=2),
        Queue(maxsize=100),
        KEYBOARD_AXES,
        forward_axis=FORWARD,
        backward_axis=BACKWARD,
        strafe_axis=STRAFE,
        turn_axis=TURN,
    )


def key(name):
    return SimpleNamespace(char=None, name=name)


class TestKeyboardAxes(unittest.TestCase):
    def setUp(self):
        self.thread = make_thread()

    def press(self, *names):
        for name in names:
            self.thread.on_press(key(name))
        return self.thread.axes()

    def test_idle_is_all_zero(self):
        self.assertEqual(self.thread.axes(), dict.fromkeys(KEYBOARD_AXES, 0.0))

    def test_arrows_drive_forward_and_back(self):
        self.assertAlmostEqual(self.press("up")["LeftY"], FORWARD)
        self.thread.on_release(key("up"))
        self.assertAlmostEqual(self.press("down")["LeftY"], -BACKWARD)

    def test_bare_arrows_turn(self):
        self.assertAlmostEqual(self.press("left")["RightX"], -TURN)
        self.thread.on_release(key("left"))
        self.assertAlmostEqual(self.press("right")["RightX"], TURN)

    def test_shifted_arrows_strafe_and_never_turn(self):
        axes = self.press("shift", "left")
        self.assertAlmostEqual(axes["LeftX"], -STRAFE)
        self.assertAlmostEqual(axes["RightX"], 0.0)
        self.thread.on_release(key("left"))
        axes = self.press("right")
        self.assertAlmostEqual(axes["LeftX"], STRAFE)
        self.assertAlmostEqual(axes["RightX"], 0.0)

    def test_opposite_directions_cancel(self):
        axes = self.press("up", "down", "left", "right")
        self.assertAlmostEqual(axes["LeftY"], 0.0)
        self.assertAlmostEqual(axes["RightX"], 0.0)

    def test_release_returns_to_zero(self):
        self.press("up", "right")
        self.thread.on_release(key("up"))
        self.thread.on_release(key("right"))
        self.assertEqual(self.thread.axes(), dict.fromkeys(KEYBOARD_AXES, 0.0))

    def test_escape_is_a_shutdown_button_and_never_an_axis(self):
        self.thread.on_press(key("esc"))
        self.assertEqual(self.thread.axes(), dict.fromkeys(KEYBOARD_AXES, 0.0))
        event = self.thread.event_queue.get_nowait()
        self.assertEqual((event["type"], event["name"], event["pressed"]), ("button", "A", True))

    def test_publish_keeps_only_the_newest_state(self):
        self.press("up")
        for _ in range(5):
            self.thread.publish()
        self.assertLessEqual(self.thread.state_queue.qsize(), 2)
        latest = None
        while not self.thread.state_queue.empty():
            latest = self.thread.state_queue.get_nowait()
        self.assertAlmostEqual(latest["axes"]["LeftY"], FORWARD)


class TestKeyboardCtrlSafety(unittest.TestCase):
    def make_controller(self):
        controller = KeyboardCtrl.__new__(KeyboardCtrl)
        super(KeyboardCtrl, controller).__init__(KEYBOARD_AXES)
        controller.stale_timeout = 0.2
        controller.keyboard_thread = make_thread()
        controller.keyboard_thread.state_queue = controller.state_queue
        controller.keyboard_thread.event_queue = controller.event_queue
        return controller

    def test_fresh_axes_pass_through(self):
        controller = self.make_controller()
        controller.keyboard_thread.on_press(key("up"))
        controller.keyboard_thread.publish()
        control, shutdown = controller.read()
        self.assertAlmostEqual(control["axes"]["LeftY"], FORWARD)
        self.assertFalse(shutdown)

    def test_a_stalled_thread_zeroes_the_command(self):
        controller = self.make_controller()
        controller.keyboard_thread.on_press(key("up"))
        controller.keyboard_thread.publish()
        controller.keyboard_thread.last_publish_time = time.monotonic() - 1.0
        control, _ = controller.read()
        self.assertEqual(control["axes"], dict.fromkeys(KEYBOARD_AXES, 0.0))

    def test_shutdown_survives_a_stalled_thread(self):
        controller = self.make_controller()
        controller.keyboard_thread.on_press(key("esc"))
        controller.keyboard_thread.last_publish_time = time.monotonic() - 1.0
        control, shutdown = controller.read()
        self.assertTrue(shutdown)
        self.assertEqual(control["axes"], dict.fromkeys(KEYBOARD_AXES, 0.0))

    def test_it_is_a_sibling_of_the_other_controllers(self):
        from g1_playground.controller.joystick_ctrl import JoystickCtrl
        from g1_playground.controller.unitree_ctrl import UnitreeCtrl

        self.assertFalse(issubclass(KeyboardCtrl, JoystickCtrl))
        self.assertFalse(issubclass(KeyboardCtrl, UnitreeCtrl))
        self.assertEqual(KeyboardCtrl.__mro__[1].__name__, "Controller")


if __name__ == "__main__":
    unittest.main()
