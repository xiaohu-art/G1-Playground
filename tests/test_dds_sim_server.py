import importlib
import importlib.util
import inspect
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from g1_playground.simulation import G1MujocoBackend
from g1_playground.simulation.mujoco_dds import G1MujocoDdsServer

REPO_ROOT = Path(__file__).resolve().parents[1]


def command(*, valid: bool, age_seconds: float = float("inf")) -> SimpleNamespace:
    return SimpleNamespace(
        valid=valid,
        age_seconds=age_seconds,
        q=np.ones(29),
        dq=np.full(29, -0.5),
        tau=np.full(29, 3.0),
        kp=np.full(29, 100.0),
        kd=np.full(29, 5.0),
    )


def backend_state(*, torque=None) -> SimpleNamespace:
    return SimpleNamespace(
        joint_pos=np.full(29, 0.1),
        joint_vel=np.full(29, 0.2),
        joint_torque=np.zeros(29) if torque is None else np.asarray(torque),
        base_quaternion_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
        base_angular_velocity=np.asarray([0.1, 0.2, 0.3]),
    )


class FakeBackend:
    def __init__(self, clock=None, timestep=0.001):
        self.model = object()
        self.data = SimpleNamespace(qpos=np.full(36, 0.1))
        self.timestep = timestep
        self.state = backend_state()
        self.clock = clock
        self.read_calls = 0
        self.scales = []
        self.torques = []
        self.copy_destinations = []

    def read(self):
        self.read_calls += 1
        return self.state

    def step(self, torque, support_scale):
        torque = np.asarray(torque).copy()
        self.torques.append(torque)
        self.scales.append(support_scale)
        self.state = backend_state(torque=torque)
        if self.clock is not None:
            self.clock.advance(0.0002)
        return self.state

    def copy_data_to(self, destination):
        self.copy_destinations.append(destination)
        destination.qpos[:] = self.data.qpos


class FakeBridge:
    def __init__(self, current_command=None):
        self.command = command(valid=False) if current_command is None else current_command
        self.published = []
        self.close_calls = 0

    def get_command(self):
        return self.command

    def publish_lowstate(self, snapshot):
        self.published.append(snapshot)
        return len(self.published)

    def close(self):
        self.close_calls += 1
        return self.close_calls == 1


class SportStateSnapshot:
    def __init__(self):
        self.position = [0.0, 0.0, 0.0]
        self.velocity = [0.0, 0.0, 0.0]
        self.body_height = 0.0


class LowStateSnapshot:
    def __init__(self):
        self.accelerometer = [0.0, 0.0, 0.0]
        self.rpy = [0.0, 0.0, 0.0]
        self.wireless_remote = bytes(40)


class FakeViewer:
    def __init__(self, *, close_after_renders=None):
        self.close_after_renders = close_after_renders
        self.render_count = 0
        self.close_calls = 0
        self.render_threads = []
        self.alive = True

        self.cam = SimpleNamespace(
            lookat=np.zeros(3),
            distance=0.0,
            elevation=0.0,
            azimuth=0.0,
        )

    def is_running(self):
        return self.alive and (self.close_after_renders is None or self.render_count < self.close_after_renders)

    def sync(self):
        self.render_threads.append(threading.current_thread())
        self.render_count += 1

    def close(self):
        self.alive = False
        self.close_calls += 1

    def lock(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class FakeViewerServer:
    def __init__(self, backend, bridge, *, worker_error=None):
        self.backend = backend
        self.bridge = bridge
        self.worker_error = worker_error
        self.run_thread = None

    def run(self, stop_event):
        self.run_thread = threading.current_thread()
        try:
            if self.worker_error is not None:
                raise self.worker_error
            stop_event.wait(1.0)
        finally:
            self.shutdown()

    def shutdown(self):
        return self.bridge.close()


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.advance(seconds)


def make_server(backend, bridge, **kwargs):
    return G1MujocoDdsServer(
        backend,
        bridge,
        LowStateSnapshot,
        np.full(29, 10.0),
        **kwargs,
    )


def load_launcher():
    source_path = REPO_ROOT / "scripts/simulate.py"
    spec = importlib.util.spec_from_file_location("g1_playground_mujoco_dds_launcher_test", source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load launcher from {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestG1MujocoDdsServer(unittest.TestCase):
    def test_observe_mode_steps_zero_torque_and_publishes_state_without_timeout(self):
        backend = FakeBackend()
        bridge = FakeBridge()
        server = make_server(backend, bridge)

        self.assertEqual(server.step(now=1000.0), 1)

        np.testing.assert_array_equal(backend.torques, np.zeros((1, 29)))
        self.assertEqual(backend.read_calls, 1)
        self.assertEqual(backend.scales, [1.0])
        self.assertEqual(len(bridge.published), 1)
        snapshot = bridge.published[0]
        np.testing.assert_array_equal(snapshot.q, backend.state.joint_pos)
        np.testing.assert_array_equal(snapshot.dq, backend.state.joint_vel)
        np.testing.assert_array_equal(snapshot.tau_est, np.zeros(29))
        self.assertEqual(snapshot.quaternion, [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(snapshot.gyroscope, [0.1, 0.2, 0.3])
        self.assertEqual(snapshot.wireless_remote, bytes(40))

    def test_active_command_uses_full_pd_clip_and_releases_support_at_ramp_end(self):
        backend = FakeBackend()
        bridge = FakeBridge(command(valid=True, age_seconds=0.0))
        server = make_server(backend, bridge)

        for now in (10.0, 11.5, 13.0, 15.5):
            server.step(now=now)

        unclipped = 3.0 + 100.0 * (1.0 - 0.1) + 5.0 * (-0.5 - 0.2)
        np.testing.assert_array_equal(backend.torques, np.full((4, 29), np.clip(unclipped, -10.0, 10.0)))
        self.assertEqual(backend.read_calls, 1)
        np.testing.assert_allclose(backend.scales, [1.0, 0.845, 0.0, 0.0], rtol=0.0, atol=1e-12)
        np.testing.assert_array_equal(bridge.published[-1].tau_est, backend.torques[-1])

    def test_watchdog_allows_threshold_then_fails_closed_above_it(self):
        backend = FakeBackend()
        bridge = FakeBridge(command(valid=True, age_seconds=0.1))
        server = make_server(backend, bridge, command_timeout=0.1)

        server.step(now=1.0)
        bridge.command.age_seconds = 0.100001
        with self.assertRaisesRegex(RuntimeError, "watchdog expired"):
            server.run()

        self.assertEqual(len(backend.torques), 1)
        self.assertEqual(len(bridge.published), 1)
        self.assertEqual(bridge.close_calls, 1)

    def test_command_timeout_must_be_finite_and_positive(self):
        for invalid in (0.0, -0.1, float("nan"), float("inf")):
            with self.subTest(command_timeout=invalid), self.assertRaisesRegex(ValueError, "finite and positive"):
                make_server(FakeBackend(), FakeBridge(), command_timeout=invalid)

    def test_run_uses_absolute_one_millisecond_deadlines_and_closes(self):
        clock = FakeClock()
        backend = FakeBackend(clock)
        bridge = FakeBridge()
        server = make_server(backend, bridge, clock=clock, sleeper=clock.sleep)
        stop_event = SimpleNamespace(is_set=lambda: len(bridge.published) >= 3)

        server.run(stop_event)

        self.assertEqual(len(backend.torques), 3)
        np.testing.assert_allclose(clock.sleeps, [0.0008, 0.0008, 0.0008], rtol=0.0, atol=1e-12)
        self.assertEqual(bridge.close_calls, 1)
        self.assertTrue(server.shutdown() is False)

    def test_server_is_viewer_free(self):
        parameters = inspect.signature(G1MujocoDdsServer).parameters
        backend = FakeBackend(timestep=0.002)
        server = make_server(backend, FakeBridge())

        self.assertNotIn("viewer", parameters)
        self.assertNotIn("render_hz", parameters)
        self.assertNotIn("timestep", parameters)
        self.assertEqual(server.timestep, backend.timestep)

    def test_runtime_has_no_external_simulator_or_subprocess_dependency(self):
        for relative_path in (
            "g1_playground/simulation/mujoco_dds.py",
            "scripts/simulate.py",
        ):
            source = (REPO_ROOT / relative_path).read_text()
            with self.subTest(path=relative_path):
                self.assertNotIn("unitree_mujoco", source)
                self.assertNotIn("subprocess", source)


class TestMujocoDdsServerLauncher(unittest.TestCase):
    def test_runtime_rates_have_one_fixed_owner(self):
        launcher = load_launcher()
        backend_parameters = inspect.signature(G1MujocoBackend).parameters
        server_parameters = inspect.signature(G1MujocoDdsServer).parameters

        self.assertEqual(backend_parameters["timestep"].default, 0.001)
        self.assertEqual(server_parameters["command_timeout"].default, 0.1)
        self.assertEqual(launcher.RENDER_HZ, 60.0)

    def test_build_uses_repo_robot_data_and_native_snapshot_boundary(self):
        launcher = load_launcher()
        captured = {}

        class NativeBridge:
            def __init__(self, endpoint):
                captured["endpoint"] = endpoint
                captured["bridge"] = self

        native = SimpleNamespace(
            G1DdsRobotEndpoint=NativeBridge,
            DdsLowStateSnapshot=LowStateSnapshot,
            DdsSportStateSnapshot=SportStateSnapshot,
        )

        backend = FakeBackend()

        def fake_backend(xml, *, elastic_support_scale, expected_actuators=29):
            captured["backend_args"] = (xml, elastic_support_scale)
            return backend

        class FakeServer:
            def __init__(self, *args, **kwargs):
                captured["server"] = (args, kwargs)

        load_module = Mock(return_value=native)
        with (
            patch.object(launcher.importlib, "import_module", load_module),
            patch.object(launcher, "G1MujocoBackend", side_effect=fake_backend),
            patch.object(launcher, "G1MujocoDdsServer", FakeServer),
        ):
            launcher.build_server()

        self.assertEqual(
            captured["endpoint"],
            {
                "domain_id": 1,
                "net_if": "lo",
                "lowcmd_topic": "rt/lowcmd",
                "lowstate_topic": "rt/lowstate",
                "sport_state_topic": "rt/odommodestate",
                "mode_machine": 5,
            },
        )
        xml, support = captured["backend_args"]
        self.assertEqual(Path(xml), REPO_ROOT / "assets/robots/g1/g1_29dof_rev_1_0.xml")
        self.assertEqual(support, 1.0)
        load_module.assert_called_once_with("unitree_cpp")
        args, kwargs = captured["server"]
        self.assertEqual(args[:3], (backend, captured["bridge"], LowStateSnapshot))
        self.assertEqual(len(args[3]), 29)
        self.assertEqual(
            kwargs,
            {
                "body_index": None,
                "hand": None,
                "sport_state_factory": SportStateSnapshot,
                "sport_publish_hz": 50.0,
            },
        )

    def test_run_always_uses_viewer_path(self):
        launcher = load_launcher()
        server = Mock()
        setup_logger = Mock()

        with (
            patch.dict(sys.modules, {"g1_playground.utils.logger": SimpleNamespace(setup_logger=setup_logger)}),
            patch.object(launcher, "build_server", return_value=server),
            patch.object(launcher, "run_with_viewer") as run_with_viewer,
        ):
            launcher.run([])

        setup_logger.assert_called_once_with()
        server.run.assert_not_called()
        run_with_viewer.assert_called_once_with(server)

    def test_viewer_uses_independent_snapshots_while_server_runs_in_worker(self):
        launcher = load_launcher()
        backend = FakeBackend()
        bridge = FakeBridge()
        server = FakeViewerServer(backend, bridge)
        render_data = SimpleNamespace(qpos=np.zeros(36))
        viewer = FakeViewer(close_after_renders=2)
        launch_passive = Mock(return_value=viewer)
        viewer_module = SimpleNamespace(launch_passive=launch_passive)

        class RecordingEvent:
            def __init__(self):
                self.event = threading.Event()
                self.waits = []

            def set(self):
                self.event.set()

            def is_set(self):
                return self.event.is_set()

            def wait(self, timeout=None):
                self.waits.append((threading.current_thread(), timeout))
                return self.event.wait(timeout)

        stop_event = RecordingEvent()
        threading_api = SimpleNamespace(Event=lambda: stop_event, Thread=threading.Thread)
        main_thread = threading.current_thread()
        with (
            patch.object(launcher, "threading", threading_api),
            patch.object(launcher.importlib, "import_module", return_value=viewer_module) as load_viewer,
            patch.object(launcher.mujoco, "MjData", return_value=render_data) as make_render_data,
        ):
            launcher.run_with_viewer(server)

        load_viewer.assert_called_once_with("mujoco.viewer")
        make_render_data.assert_called_once_with(backend.model)
        launch_passive.assert_called_once_with(
            backend.model,
            render_data,
            show_left_ui=False,
            show_right_ui=False,
        )
        self.assertIsNot(render_data, backend.data)
        self.assertEqual(backend.copy_destinations, [render_data, render_data, render_data])
        np.testing.assert_array_equal(render_data.qpos, backend.data.qpos)
        np.testing.assert_array_equal(viewer.cam.lookat, render_data.qpos[:3])
        self.assertEqual(
            (viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth),
            (3.0, -10.0, 180.0),
        )
        self.assertEqual(viewer.render_count, 2)
        self.assertTrue(all(thread is main_thread for thread in viewer.render_threads))
        self.assertIsNot(server.run_thread, main_thread)
        main_waits = [timeout for thread, timeout in stop_event.waits if thread is main_thread]
        self.assertEqual(len(main_waits), 2)
        self.assertTrue(all(0.0 <= timeout <= 1.0 / launcher.RENDER_HZ for timeout in main_waits))
        self.assertEqual(viewer.close_calls, 1)
        self.assertEqual(bridge.close_calls, 1)

    def test_viewer_propagates_worker_failure_after_closing_resources(self):
        launcher = load_launcher()
        backend = FakeBackend()
        bridge = FakeBridge()
        worker_error = RuntimeError("worker failed")
        server = FakeViewerServer(backend, bridge, worker_error=worker_error)
        render_data = SimpleNamespace(qpos=np.zeros(36))
        viewer = FakeViewer()
        viewer_module = SimpleNamespace(launch_passive=Mock(return_value=viewer))

        with (
            patch.object(launcher.importlib, "import_module", return_value=viewer_module),
            patch.object(launcher.mujoco, "MjData", return_value=render_data),
            self.assertRaises(RuntimeError) as raised,
        ):
            launcher.run_with_viewer(server)

        self.assertIs(raised.exception, worker_error)
        self.assertEqual(viewer.close_calls, 1)
        self.assertEqual(bridge.close_calls, 1)

    def test_viewer_initialization_failure_closes_server_without_starting_worker(self):
        launcher = load_launcher()
        backend = FakeBackend()
        bridge = FakeBridge()
        server = FakeViewerServer(backend, bridge)
        render_data = SimpleNamespace(qpos=np.zeros(36))
        init_error = RuntimeError("viewer init failed")
        viewer_module = SimpleNamespace(launch_passive=Mock(side_effect=init_error))
        thread_type = Mock()
        threading_api = SimpleNamespace(Event=threading.Event, Thread=thread_type)

        with (
            patch.object(launcher, "threading", threading_api),
            patch.object(launcher.importlib, "import_module", return_value=viewer_module),
            patch.object(launcher.mujoco, "MjData", return_value=render_data),
            self.assertRaises(RuntimeError) as raised,
        ):
            launcher.run_with_viewer(server)

        self.assertIs(raised.exception, init_error)
        thread_type.assert_not_called()
        self.assertEqual(bridge.close_calls, 1)


if __name__ == "__main__":
    unittest.main()
