import importlib.util
import multiprocessing
import os
import sys
import time
import traceback
import unittest
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE_BUILD = REPO_ROOT / "third_party/unitree_cpp/build"
RUN_LOOPBACK = os.environ.get("G1_PLAYGROUND_RUN_DDS_LOOPBACK") == "1"


def _load_native():
    sys.path.insert(0, NATIVE_BUILD.as_posix())
    import unitree_cpp

    return unitree_cpp


def _run_server(ready, stop, results):
    bridge = None
    server = None
    try:
        native = _load_native()
        from g1_playground.simulation import G1MujocoBackend, G1MujocoDdsServer
        from g1_playground.utils import resolve_repo_path

        robot = OmegaConf.load(REPO_ROOT / "configs/robot/g1.yaml")
        backend = G1MujocoBackend(resolve_repo_path(robot.xml), 0.001, elastic_support_scale=1.0)
        bridge = native.G1DdsSimServer(
            {
                "domain_id": 1,
                "net_if": "lo",
                "lowcmd_topic": "rt/lowcmd",
                "lowstate_topic": "rt/lowstate",
                "mode_machine": 5,
            }
        )
        server = G1MujocoDdsServer(
            backend,
            bridge,
            native.DdsLowStateSnapshot,
            robot.dof.torque_limits,
            command_timeout=0.1,
        )
        ready.set()
        server.run(stop)
        stats = bridge.stats
        results.put(
            (
                "server",
                {
                    "accepted_commands": stats.accepted_commands,
                    "crc_errors": stats.crc_errors,
                    "finite_errors": stats.finite_errors,
                    "mode_errors": stats.mode_errors,
                },
            )
        )
    except RuntimeError as error:
        if bridge is not None and "command watchdog expired" in str(error):
            stats = bridge.stats
            results.put(
                (
                    "server_watchdog",
                    {
                        "accepted_commands": stats.accepted_commands,
                        "crc_errors": stats.crc_errors,
                        "finite_errors": stats.finite_errors,
                        "mode_errors": stats.mode_errors,
                        "last_sequence": stats.accepted_commands,
                        "first_command_processed": server is not None and server._first_command_time is not None,
                        "message": str(error),
                    },
                )
            )
        else:
            results.put(("server_error", traceback.format_exc()))
        ready.set()
    except BaseException:
        results.put(("server_error", traceback.format_exc()))
        ready.set()


def _run_state_client(domain_id, samples, results):
    controller = None
    try:
        native = _load_native()
        controller = native.UnitreeController(
            {
                "domain_id": domain_id,
                "net_if": "lo",
                "control_dt": 0.02,
                "msg_type": "hg",
                "control_mode": "position",
                "hand_type": "NONE",
                "lowcmd_topic": "rt/lowcmd",
                "lowstate_topic": "rt/lowstate",
                "enable_odometry": False,
                "sport_state_topic": "rt/odommodestate",
                "num_dofs": 29,
                "stiffness": [0.0] * 29,
                "damping": [0.0] * 29,
            }
        )
        deadline = time.monotonic() + 3.0
        ready = False
        while time.monotonic() < deadline:
            try:
                ready = controller.get_robot_state().tick > 0
            except RuntimeError:
                time.sleep(0.02)
                continue
            if ready:
                break
        ticks = []
        quaternion = None
        joint_count = None
        mode_machine = None
        if ready:
            for _ in range(samples):
                state = controller.get_robot_state()
                ticks.append(state.tick)
                quaternion = list(state.imu_state.quaternion)
                joint_count = len(state.motor_state.q)
                mode_machine = state.mode_machine
                time.sleep(0.02)
        results.put(
            (
                f"client_{domain_id}",
                {
                    "ready": ready,
                    "ticks": ticks,
                    "quaternion": quaternion,
                    "joint_count": joint_count,
                    "mode_machine": mode_machine,
                },
            )
        )
    except BaseException:
        results.put((f"client_{domain_id}_error", traceback.format_exc()))
    finally:
        if controller is not None:
            controller.shutdown()


def _run_command_client(results):
    controller = None
    try:
        native = _load_native()
        controller = native.UnitreeController(
            {
                "domain_id": 1,
                "net_if": "lo",
                "control_dt": 0.02,
                "msg_type": "hg",
                "control_mode": "position",
                "hand_type": "NONE",
                "lowcmd_topic": "rt/lowcmd",
                "lowstate_topic": "rt/lowstate",
                "enable_odometry": False,
                "sport_state_topic": "rt/odommodestate",
                "num_dofs": 29,
                "stiffness": [1.0] * 29,
                "damping": [0.1] * 29,
                "motion_switcher_required": False,
            }
        )
        deadline = time.monotonic() + 3.0
        state = None
        while time.monotonic() < deadline:
            try:
                state = controller.get_robot_state()
            except RuntimeError:
                time.sleep(0.02)
                continue
            if state.tick > 0:
                break
        if state is None or state.tick == 0:
            raise RuntimeError("command client did not receive LowState")

        activation_started = time.monotonic()
        activated = controller.activate_commands()
        activation_seconds = time.monotonic() - activation_started
        controller.step(list(state.motor_state.q))
        time.sleep(0.06)
        results.put(
            (
                "command_client",
                {
                    "activated": activated,
                    "activation_seconds": activation_seconds,
                    "sent": True,
                    "state_tick": state.tick,
                    "joint_count": len(state.motor_state.q),
                },
            )
        )
    except BaseException:
        results.put(("command_client_error", traceback.format_exc()))
    finally:
        if controller is not None:
            controller.shutdown()


@unittest.skipUnless(RUN_LOOPBACK, "set G1_PLAYGROUND_RUN_DDS_LOOPBACK=1 to create local DDS participants")
class TestDdsLoopback(unittest.TestCase):
    def test_domain_one_state_loopback_and_domain_isolation(self):
        if importlib.util.find_spec("unitree_cpp") is None and not tuple(NATIVE_BUILD.glob("unitree_cpp*.so")):
            self.skipTest("vendored unitree_cpp binding is not built")

        context = multiprocessing.get_context("spawn")
        ready = context.Event()
        stop = context.Event()
        results = context.Queue()
        server = context.Process(target=_run_server, args=(ready, stop, results))
        server.start()
        self.assertTrue(ready.wait(10.0), "DDS simulator did not initialize")

        matching = context.Process(target=_run_state_client, args=(1, 12, results))
        mismatched = context.Process(target=_run_state_client, args=(2, 1, results))
        command_client = context.Process(target=_run_command_client, args=(results,))
        try:
            matching.start()
            matching.join(8.0)
            self.assertFalse(matching.is_alive(), "matching DDS client did not finish")
            mismatched.start()
            mismatched.join(6.0)
            self.assertFalse(mismatched.is_alive(), "mismatched DDS client did not finish")
            command_client.start()
            command_client.join(10.0)
            self.assertFalse(command_client.is_alive(), "DDS command client did not finish")
            server.join(3.0)
            self.assertFalse(server.is_alive(), "DDS command watchdog did not stop the simulator")
        finally:
            stop.set()
            server.join(5.0)
            for process in (matching, mismatched, command_client, server):
                if process.is_alive():
                    process.terminate()
                    process.join(2.0)

        messages = {}
        for _ in range(4):
            key, value = results.get(timeout=2.0)
            messages[key] = value
        errors = {key: value for key, value in messages.items() if key.endswith("_error")}
        if errors:
            self.fail("\n\n".join(f"{key}:\n{value}" for key, value in errors.items()))

        matching_result = messages["client_1"]
        self.assertTrue(matching_result["ready"])
        self.assertEqual(len(matching_result["ticks"]), 12)
        self.assertTrue(all(tick > 0 for tick in matching_result["ticks"]))
        self.assertTrue(
            all(
                current > previous
                for previous, current in zip(matching_result["ticks"], matching_result["ticks"][1:], strict=False)
            ),
            matching_result["ticks"],
        )
        self.assertEqual(len(matching_result["quaternion"]), 4)
        self.assertEqual(matching_result["joint_count"], 29)
        self.assertEqual(matching_result["mode_machine"], 5)
        self.assertFalse(messages["client_2"]["ready"])
        self.assertTrue(messages["command_client"]["activated"])
        self.assertTrue(messages["command_client"]["sent"])
        self.assertLess(messages["command_client"]["activation_seconds"], 1.0)
        self.assertGreater(messages["command_client"]["state_tick"], 0)
        self.assertEqual(messages["command_client"]["joint_count"], 29)

        watchdog = messages["server_watchdog"]
        self.assertGreaterEqual(watchdog["accepted_commands"], 1)
        self.assertEqual(watchdog["last_sequence"], watchdog["accepted_commands"])
        self.assertTrue(watchdog["first_command_processed"])
        self.assertEqual(watchdog["crc_errors"], 0)
        self.assertEqual(watchdog["finite_errors"], 0)
        self.assertEqual(watchdog["mode_errors"], 0)
        self.assertIn("command watchdog expired", watchdog["message"])


if __name__ == "__main__":
    unittest.main()
