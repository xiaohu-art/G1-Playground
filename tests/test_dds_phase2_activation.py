import hashlib
import importlib
import json
import shutil
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from g1_playground.utils.dof import compose_dof_config
from tests.config_helpers import compose_config, load_pipeline_launcher

REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE2_BOUNDARY_PATH = REPO_ROOT / "tests/fixtures/dds_phase2/observer_activation_boundary.json"
PHASE3_BOUNDARY_PATH = REPO_ROOT / "tests/fixtures/dds_phase3/mujoco_server_boundary.json"
PHASE2_BOUNDARY_SHA256 = "bc360d3450ee086030511e22c0c7dc7f304fac8c70afd054aef6de849bc1133e"


def load_pipeline_for_test():
    logger_module = types.ModuleType("g1_playground.utils.logger")
    logger_module.setup_logger = lambda: None
    policy_module = types.ModuleType("g1_playground.policy")
    policy_module.LeggedLabPolicy = object
    with patch.dict(
        sys.modules,
        {"g1_playground.policy": policy_module, "g1_playground.utils.logger": logger_module},
    ):
        return load_pipeline_launcher()


def sha256_bytes(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def classify_changes(baseline: dict[str, str], current: dict[str, str]) -> dict[str, list[str]]:
    baseline_paths = set(baseline)
    current_paths = set(current)
    return {
        "modified": sorted(path for path in baseline_paths & current_paths if baseline[path] != current[path]),
        "added": sorted(current_paths - baseline_paths),
        "removed": sorted(baseline_paths - current_paths),
    }


class TestDdsPhase2EvidenceBoundary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.boundary = json.loads(PHASE2_BOUNDARY_PATH.read_text())


class TestDdsPhase2NativeLifecycle(unittest.TestCase):
    def test_production_lifecycle_core_without_dds(self):
        compiler = shutil.which("clang++") or shutil.which("c++")
        sdk_include = next(
            (
                path
                for path in (Path("/usr/local/include"), Path("/usr/include"))
                if (path / "unitree/robot/channel/channel_factory.hpp").is_file()
            ),
            None,
        )
        dds_include = next(
            (
                path
                for path in (Path("/usr/local/include/ddscxx"), Path("/usr/include/ddscxx"))
                if (path / "dds/dds.hpp").is_file()
            ),
            None,
        )
        if compiler is None or sdk_include is None or dds_include is None:
            self.skipTest("C++ compiler or Unitree SDK2/CycloneDDS headers are unavailable")

        source = textwrap.dedent(
            r"""
            #include <atomic>
            #include <cassert>
            #include <stdexcept>
            #include <thread>
            #include <vector>

            #include "g1_dds_control_endpoint.hpp"

            using unitree_cpp_detail::DdsControlEndpointLifecycle;
            using unitree_cpp_detail::DdsControlEndpointState;

            int main() {
                DdsControlEndpointLifecycle observer;
                int activate_side_effects = 0;
                int command_writes = 0;
                int receiving_close_writes = 0;
                assert(observer.state() == DdsControlEndpointState::RECEIVING);

                bool command_rejected = false;
                try {
                    observer.RunWhileActive([&]() { ++command_writes; });
                } catch (const std::logic_error&) {
                    command_rejected = true;
                }
                assert(command_rejected);
                assert(command_writes == 0);

                assert(observer.CloseOnce([&](DdsControlEndpointState previous) {
                    assert(previous == DdsControlEndpointState::RECEIVING);
                    if (previous == DdsControlEndpointState::ACTIVE) {
                        ++receiving_close_writes;
                    }
                }));
                assert(observer.state() == DdsControlEndpointState::CLOSED);
                assert(!observer.CloseOnce([&](DdsControlEndpointState) { ++receiving_close_writes; }));
                assert(activate_side_effects == 0);
                assert(command_writes == 0);
                assert(receiving_close_writes == 0);

                bool closed_activation_rejected = false;
                try {
                    observer.ActivateOnce([&]() { ++activate_side_effects; });
                } catch (const std::logic_error&) {
                    closed_activation_rejected = true;
                }
                assert(closed_activation_rejected);
                assert(activate_side_effects == 0);

                DdsControlEndpointLifecycle retryable;
                bool failed = false;
                try {
                    retryable.ActivateOnce([]() { throw std::runtime_error("injected activation failure"); });
                } catch (const std::runtime_error&) {
                    failed = true;
                }
                assert(failed);
                assert(retryable.state() == DdsControlEndpointState::RECEIVING);

                DdsControlEndpointLifecycle active;
                assert(active.ActivateOnce([&]() { ++activate_side_effects; }));
                assert(active.state() == DdsControlEndpointState::ACTIVE);
                assert(!active.ActivateOnce([&]() { ++activate_side_effects; }));
                active.RunWhileActive([&]() { ++command_writes; });
                assert(activate_side_effects == 1);
                assert(command_writes == 1);
                assert(active.CloseOnce([&](DdsControlEndpointState previous) {
                    assert(previous == DdsControlEndpointState::ACTIVE);
                    ++command_writes;
                }));
                assert(command_writes == 2);

                DdsControlEndpointLifecycle concurrent;
                std::atomic<int> concurrent_activations{0};
                std::atomic<int> first_callers{0};
                std::vector<std::thread> threads;
                for (int index = 0; index < 16; ++index) {
                    threads.emplace_back([&]() {
                        if (concurrent.ActivateOnce([&]() { ++concurrent_activations; })) {
                            ++first_callers;
                        }
                    });
                }
                for (auto& thread : threads) {
                    thread.join();
                }
                assert(concurrent_activations == 1);
                assert(first_callers == 1);
                return 0;
            }
            """
        )

        with tempfile.TemporaryDirectory(prefix="g1-playground-dds-phase2-lifecycle-") as temporary_dir:
            temporary_path = Path(temporary_dir)
            source_path = temporary_path / "lifecycle_test.cpp"
            binary_path = temporary_path / "lifecycle_test"
            source_path.write_text(source)
            compile_result = subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-pthread",
                    f"-I{REPO_ROOT / 'third_party/unitree_cpp/src'}",
                    f"-I{sdk_include}",
                    f"-I{dds_include}",
                    source_path,
                    "-o",
                    binary_path,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
            run_result = subprocess.run([binary_path], capture_output=True, text=True, timeout=10)
            self.assertEqual(run_result.returncode, 0, run_result.stderr)

    def test_control_endpoint_routes_side_effects_through_activation(self):
        source_path = REPO_ROOT / "third_party/unitree_cpp/src/g1_dds_control_endpoint.cpp"
        source = source_path.read_text()

        def function_body(signature: str) -> str:
            start = source.index(signature)
            opening_brace = source.index("{", start)
            depth = 0
            for index in range(opening_brace, len(source)):
                if source[index] == "{":
                    depth += 1
                elif source[index] == "}":
                    depth -= 1
                    if depth == 0:
                        return source[opening_brace + 1 : index]
            self.fail(f"unterminated function: {signature}")

        constructor = function_body("G1DdsControlEndpoint::G1DdsControlEndpoint")
        activation = function_body("void G1DdsControlEndpoint::InitializeCommandTransport")
        activate_commands = function_body("bool G1DdsControlEndpoint::activate_commands")
        observer = function_body("void G1DdsControlEndpoint::InitializeObserver")

        for forbidden in (
            "MotionSwitcherClient",
            "lowcmd_publisher_",
            "command_writer_ptr_",
            "handcmd_left_publisher_",
            "handcmd_right_publisher_",
            "handcmd_writer_ptr_",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, constructor)
                self.assertNotIn(forbidden, observer)
                self.assertIn(forbidden, activation)
        self.assertIn("InitializeObserver()", constructor)
        self.assertIn("InitializeCommandTransport()", activate_commands)


class TestDdsPhase2PythonActivation(unittest.TestCase):
    def setUp(self):
        self.module_name = "g1_playground.g1_env"
        self.missing = object()
        self.old_binding = sys.modules.get("unitree_cpp", self.missing)
        self.old_module = sys.modules.pop(self.module_name, None)

        class FakeG1DdsControlEndpoint:
            instances = []

            def __init__(fake_self, cfg):
                fake_self.cfg = cfg
                fake_self.events = ["construct"]
                fake_self.command_writes = 0
                fake_self.lifecycle = "receiving"
                fake_self.robot_state = SimpleNamespace(
                    motor_state=SimpleNamespace(q=[0.0] * 29, dq=[0.0] * 29),
                    imu_state=SimpleNamespace(quaternion=[1.0, 0.0, 0.0, 0.0], gyroscope=[0.0] * 3),
                    wireless_remote=bytes(40),
                )
                fake_self.instances.append(fake_self)

            def self_check(fake_self):
                fake_self.events.append("self_check")
                return True

            def get_robot_state(fake_self):
                fake_self.events.append("get_robot_state")
                return fake_self.robot_state

            def activate_commands(fake_self):
                fake_self.events.append("activate_commands")
                if fake_self.lifecycle == "closed":
                    raise RuntimeError("cannot activate a closed G1DdsControlEndpoint")
                if fake_self.lifecycle == "active":
                    return False
                fake_self.lifecycle = "active"
                fake_self.events.append("activate_commands_effect")
                return True

            def step(fake_self, target):
                if fake_self.lifecycle != "active":
                    raise RuntimeError("G1DdsControlEndpoint command transport is not active")
                fake_self.events.append("step")
                fake_self.command_writes += 1

            def set_gains(fake_self, stiffness, damping):
                fake_self.events.append("set_gains")

            def shutdown(fake_self):
                fake_self.events.append("shutdown")
                if fake_self.lifecycle == "closed":
                    return
                fake_self.lifecycle = "closed"
                fake_self.events.append("shutdown_effect")

        self.fake_controller = FakeG1DdsControlEndpoint
        fake_binding = types.ModuleType("unitree_cpp")
        fake_binding.RobotState = SimpleNamespace
        fake_binding.G1DdsControlEndpoint = FakeG1DdsControlEndpoint
        sys.modules["unitree_cpp"] = fake_binding
        environment_module = importlib.import_module(self.module_name)
        self.env_class = environment_module.G1Env
        self.cfg = compose_config("real")

    def tearDown(self):
        sys.modules.pop(self.module_name, None)
        if self.old_module is not None:
            sys.modules[self.module_name] = self.old_module
        if self.old_binding is self.missing:
            sys.modules.pop("unitree_cpp", None)
        else:
            sys.modules["unitree_cpp"] = self.old_binding

    def make_env(self, deployment="real", **overrides):
        cfg = compose_config(deployment)
        dof_cfg = compose_dof_config(cfg.robot.dof, cfg.policy.dof)
        parameters = {
            "control_dt": 0.02,
            "domain_id": cfg.env.domain_id,
            "net_if": cfg.env.net_if,
            "lowcmd_topic": cfg.env.lowcmd_topic,
            "lowstate_topic": cfg.env.lowstate_topic,
            "motion_switcher_required": cfg.env.motion_switcher_required,
            "dof_cfg": dof_cfg,
        }
        parameters.update(overrides)
        return self.env_class(**parameters)

    def test_endpoint_contract_accepts_sim_and_real_profiles(self):
        expected = {
            "sim": (1, "lo", False),
            "real": (0, "enP8p1s0", True),
        }
        for deployment, (domain_id, net_if, motion_required) in expected.items():
            with self.subTest(deployment=deployment):
                env = self.make_env(deployment)
                native_cfg = self.fake_controller.instances[-1].cfg
                self.assertEqual(native_cfg["domain_id"], domain_id)
                self.assertEqual(native_cfg["net_if"], net_if)
                self.assertIs(native_cfg["motion_switcher_required"], motion_required)
                self.assertNotIn("self_check", self.fake_controller.instances[-1].events)
                env.shutdown()

    def test_endpoint_contract_delegates_validation_to_native_owner(self):
        instances_before = len(self.fake_controller.instances)
        env = self.make_env(
            domain_id=-1,
            net_if="native-owner-if",
            lowcmd_topic="custom/lowcmd",
            lowstate_topic="custom/lowstate",
            control_dt=0.0,
        )
        try:
            self.assertEqual(len(self.fake_controller.instances), instances_before + 1)
            native_cfg = self.fake_controller.instances[-1].cfg
            self.assertEqual(
                {
                    key: native_cfg[key]
                    for key in ("domain_id", "net_if", "lowcmd_topic", "lowstate_topic", "control_dt")
                },
                {
                    "domain_id": -1,
                    "net_if": "native-owner-if",
                    "lowcmd_topic": "custom/lowcmd",
                    "lowstate_topic": "custom/lowstate",
                    "control_dt": 0.0,
                },
            )
        finally:
            env.shutdown()

    def test_read_returns_one_detached_read_only_g1_state(self):
        env = self.make_env()
        fake = self.fake_controller.instances[-1]
        q = np.arange(29, dtype=np.float32)
        dq = q + 100.0
        fake.robot_state.motor_state.q = q.tolist()
        fake.robot_state.motor_state.dq = dq.tolist()
        fake.robot_state.imu_state.quaternion = [1.0, 0.0, 0.0, 0.0]
        fake.robot_state.imu_state.gyroscope = [0.1, 0.2, 0.3]
        fake.robot_state.wireless_remote = bytes(range(40))
        remote_packets = []
        env.remote_controller_handler = remote_packets.append

        try:
            state = env.read()
            self.assertEqual(type(state).__name__, "G1State")
            np.testing.assert_array_equal(state.dof_pos, q)
            np.testing.assert_array_equal(state.dof_vel, dq)
            np.testing.assert_array_equal(state.base_quat, [0.0, 0.0, 0.0, 1.0])
            np.testing.assert_allclose(state.base_ang_vel, [0.1, 0.2, 0.3], rtol=0.0, atol=1e-7)
            self.assertEqual(remote_packets, [bytes(range(40))])
            for values in (state.dof_pos, state.dof_vel, state.base_quat, state.base_ang_vel):
                self.assertFalse(values.flags.writeable)
                with self.assertRaises(ValueError):
                    values[0] = -999.0

            fake.robot_state.motor_state.q[0] = -123.0
            self.assertEqual(state.dof_pos[0], q[0])
            for removed_api in ("reset", "fallen", "dof_pos", "dof_vel", "base_quat", "base_ang_vel"):
                self.assertFalse(hasattr(env, removed_api))
            self.assertEqual(fake.events.count("get_robot_state"), 1)
        finally:
            env.shutdown()

    def test_construct_step_activate_and_active_compatibility(self):
        env = self.make_env()
        fake = self.fake_controller.instances[-1]
        self.assertNotIn("activate_commands", fake.events)
        self.assertNotIn("step", fake.events)
        self.assertNotIn("target", fake.cfg)
        self.assertNotIn("act", fake.cfg)
        self.assertNotIn("role", fake.cfg)
        self.assertFalse(hasattr(env, "_commands_active"))
        self.assertFalse(hasattr(env, "_shutdown"))

        with self.assertRaisesRegex(RuntimeError, "not active"):
            env.step(np.zeros(29))
        self.assertEqual(fake.command_writes, 0)

        self.assertTrue(env.activate_commands())
        self.assertFalse(env.activate_commands())
        self.assertEqual(fake.events.count("activate_commands"), 2)
        self.assertEqual(fake.events.count("activate_commands_effect"), 1)
        target = np.linspace(-0.2, 0.2, 29)
        env.step(target)
        self.assertEqual(fake.command_writes, 1)
        self.assertEqual(fake.events[-1], "step")
        env.shutdown()
        with self.assertRaisesRegex(RuntimeError, "not active"):
            env.step(target)

    def test_receiving_shutdown_is_idempotent_and_has_no_command_write(self):
        env = self.make_env()
        fake = self.fake_controller.instances[-1]
        env.shutdown()
        env.shutdown()
        self.assertEqual(fake.events.count("shutdown"), 2)
        self.assertEqual(fake.events.count("shutdown_effect"), 1)
        self.assertEqual(fake.events.count("activate_commands"), 0)
        self.assertEqual(fake.command_writes, 0)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            env.activate_commands()

    def test_dry_steps_never_activate_or_write(self):
        from g1_playground.controller.unitree_ctrl import UnitreeCtrl

        step = load_pipeline_for_test().step

        class FakePolicy:
            dt = 0.02
            freq = 50
            standing_target = np.zeros(29, dtype=np.float32)

            def act(self, state, control):
                return self.standing_target

            def reset(self):
                return None

        policy = FakePolicy()
        env = self.make_env()
        controller = UnitreeCtrl(env)
        components = env, controller, policy
        fake = self.fake_controller.instances[-1]
        try:
            self.assertNotIn("activate_commands", fake.events)
            for _ in range(10):
                self.assertTrue(step(*components, send_command=False))
            self.assertEqual(fake.command_writes, 0)
            self.assertNotIn("activate_commands", fake.events)
        finally:
            env.shutdown()


class TestDdsPhase2LauncherTeardown(unittest.TestCase):
    def test_tilt_boundary_is_shared_by_all_backends(self):
        from g1_playground.utils.math import is_upright

        self.assertTrue(is_upright(np.asarray([0.0, 0.0, 0.0, 1.0])))
        self.assertFalse(is_upright(np.asarray([1.0, 0.0, 0.0, 0.0])))

    def test_preflight_shutdown_prevents_activation_and_still_tears_down(self):
        launcher = load_pipeline_for_test()

        events = []

        class FakeEnv:
            def self_check(self):
                events.append("self_check")

            def read(self):
                events.append("env.read")
                return SimpleNamespace(
                    dof_pos=np.zeros(29, dtype=np.float32),
                    base_quat=np.asarray([0.0, 0.0, 0.0, 1.0]),
                )

            def activate_commands(self):
                events.append("activate")

            def step(self, target):
                raise AssertionError("preflight shutdown must prevent command writes")

            def shutdown(self):
                events.append("shutdown")

        class FakeController:
            def __init__(self):
                self.reads = 0

            def read(self):
                events.append("controller.read")
                self.reads += 1
                return {"axes": {}}, self.reads == 11

        class FakePolicy:
            freq = 50
            dt = 0.02
            standing_target = np.zeros(29, dtype=np.float32)

            def __init__(self, cfg_policy, device, dof_cfg):
                pass

            def act(self, state, control):
                return self.standing_target

        cfg = compose_config("sim")
        env = FakeEnv()
        controller = FakeController()
        launcher.setup_logger = lambda: None
        launcher.compose_dof_config = lambda robot_dof, policy_dof: object()
        launcher.LeggedLabPolicy = FakePolicy
        launcher.instantiate = lambda component, **kwargs: env if component is cfg.env else controller
        with patch.object(launcher, "is_upright", return_value=True) as is_upright:
            launcher.run(cfg)

        self.assertEqual(controller.reads, 11)
        self.assertEqual(is_upright.call_count, 10)
        self.assertNotIn("activate", events)
        self.assertEqual(events[-1], "shutdown")

    def test_ramp_shutdown_precedes_first_command(self):
        launcher = load_pipeline_for_test()

        events = []

        class FakeEnv:
            def self_check(self):
                events.append("self_check")

            def read(self):
                events.append("env.read")
                return SimpleNamespace(
                    dof_pos=np.zeros(29, dtype=np.float32),
                    base_quat=np.asarray([0.0, 0.0, 0.0, 1.0]),
                )

            def activate_commands(self):
                events.append("activate")

            def step(self, target):
                raise AssertionError("ramp shutdown must precede the first command")

            def shutdown(self):
                events.append("shutdown")

        class FakeController:
            def __init__(self):
                self.reads = 0

            def read(self):
                events.append("controller.read")
                self.reads += 1
                return {"axes": {}}, self.reads == 12

        class FakePolicy:
            freq = 50
            dt = 0.02
            standing_target = np.ones(29, dtype=np.float32)

            def __init__(self, cfg_policy, device, dof_cfg):
                pass

            def act(self, state, control):
                return self.standing_target

        cfg = compose_config("sim")
        env = FakeEnv()
        controller = FakeController()
        launcher.setup_logger = lambda: None
        launcher.compose_dof_config = lambda robot_dof, policy_dof: object()
        launcher.LeggedLabPolicy = FakePolicy
        launcher.instantiate = lambda component, **kwargs: env if component is cfg.env else controller
        launcher.time = SimpleNamespace(
            monotonic=lambda: 100.0,
            sleep=lambda duration: (_ for _ in ()).throw(AssertionError("shutdown must skip pacing")),
        )
        with patch.object(launcher, "is_upright", return_value=True) as is_upright:
            launcher.run(cfg)

        self.assertEqual(controller.reads, 12)
        self.assertEqual(is_upright.call_count, 11)
        self.assertEqual(events.count("activate"), 1)
        self.assertEqual(events[-1], "shutdown")

    def test_startup_keeps_activation_ramp_reset_and_direct_policy_order(self):
        launcher = load_pipeline_for_test()

        events = []
        commanded_targets = []
        policy_controls = []
        sleep_durations = []
        initial = np.linspace(-0.2, 0.2, 29, dtype=np.float32)
        standing = np.ones(29, dtype=np.float32)
        policy_target = np.full(29, 2.0, dtype=np.float32)
        state = SimpleNamespace(
            dof_pos=initial,
            base_quat=np.asarray([0.0, 0.0, 0.0, 1.0]),
        )
        current_frame = 0
        safe_frame = 0

        class FakeEnv:
            def self_check(self):
                events.append("self_check")

            def read(self):
                nonlocal current_frame
                current_frame += 1
                return state

            @staticmethod
            def read_odometry():
                return None

            def activate_commands(self):
                self.assert_safe_frame()
                events.append("activate")

            def step(self, target):
                self.assert_safe_frame()
                events.append("env.step")
                commanded_targets.append(np.asarray(target).copy())

            def shutdown(self):
                events.append("shutdown")

            @staticmethod
            def assert_safe_frame():
                if safe_frame != current_frame:
                    raise AssertionError("every command must use safety checked state from the current frame")

        class FakeController:
            def __init__(self):
                self.reads = 0

            def read(self):
                self.reads += 1
                control = {"axes": {"LeftX": 0.3, "LeftY": 0.4, "RightX": 0.5}}
                return control, self.reads == 10 + 1 + 150 + 2

        class FakePolicy:
            freq = 50
            dt = 0.02
            standing_target = standing

            def __init__(self, cfg_policy, device, dof_cfg):
                events.append("policy.construct")
                self.reset_done = False

            def act(self, state, control):
                axes = control["axes"].copy()
                policy_controls.append(axes)
                events.append("policy.act.active" if self.reset_done else "policy.act.dry")
                return policy_target

            def reset(self):
                events.append("policy.reset")
                self.reset_done = True

        def mark_upright(quaternion):
            nonlocal safe_frame
            safe_frame = current_frame
            return True

        cfg = compose_config("sim")
        env = FakeEnv()
        controller = FakeController()
        effective_dof = object()

        def fake_instantiate(component, **kwargs):
            if component is cfg.env:
                self.assertEqual(kwargs, {"dof_cfg": effective_dof, "control_dt": 0.02})
                return env
            self.assertIs(component, cfg.controller)
            self.assertEqual(kwargs, {"env": env})
            return controller

        launcher.setup_logger = lambda: None
        launcher.compose_dof_config = lambda robot_dof, policy_dof: effective_dof
        launcher.LeggedLabPolicy = FakePolicy
        launcher.instantiate = fake_instantiate
        launcher.is_upright = mark_upright
        launcher.time = SimpleNamespace(monotonic=lambda: 100.0, sleep=sleep_durations.append)
        launcher.run(cfg)

        self.assertEqual(events.count("activate"), 1)
        self.assertEqual(len(commanded_targets), 151)
        self.assertEqual(current_frame, 10 + 1 + 150 + 2)
        self.assertEqual(len(sleep_durations), 151)
        np.testing.assert_allclose(sleep_durations, np.full(151, 0.02), rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(commanded_targets[0], (1.0 - 1 / 150) * initial + standing / 150)
        np.testing.assert_allclose(commanded_targets[149], standing)
        np.testing.assert_allclose(commanded_targets[150], policy_target)

        self.assertEqual(events.count("policy.reset"), 1)
        self.assertLess(events.index("activate"), events.index("env.step"))
        self.assertLess(events.index("env.step"), events.index("policy.reset"))
        self.assertLess(events.index("policy.reset"), events.index("policy.act.active"))
        self.assertEqual(len(policy_controls), 11)
        self.assertTrue(all(any(axes.values()) for axes in policy_controls))
        self.assertEqual(events[-1], "shutdown")

    def test_command_failure_executes_shutdown(self):
        launcher = load_pipeline_for_test()

        events = []

        class FakeEnv:
            def self_check(self):
                events.append("self_check")

            def read(self):
                return SimpleNamespace(
                    dof_pos=np.zeros(29, dtype=np.float32),
                    base_quat=np.asarray([0.0, 0.0, 0.0, 1.0]),
                )

            def activate_commands(self):
                events.append("activate")

            def step(self, target):
                raise RuntimeError("injected command failure")

            def shutdown(self):
                events.append("shutdown")

        class FakePolicy:
            freq = 50
            standing_target = np.zeros(29, dtype=np.float32)

            def __init__(self, cfg_policy, device, dof_cfg):
                events.append("policy.construct")
                self.dt = 0.02

            def act(self, state, control):
                return self.standing_target

        class FakeController:
            def read(self):
                return {"axes": {}}, False

        env = FakeEnv()
        controller = FakeController()
        effective_dof = object()

        def fake_instantiate(component, **kwargs):
            if component is cfg.env:
                self.assertEqual(set(kwargs), {"control_dt", "dof_cfg"})
                self.assertIs(kwargs["dof_cfg"], effective_dof)
                self.assertEqual(kwargs["control_dt"], 0.02)
                events.append("env.instantiate")
                return env
            self.assertIs(component, cfg.controller)
            self.assertEqual(set(kwargs), {"env"})
            self.assertIs(kwargs["env"], env)
            events.append("controller.instantiate")
            return controller

        cfg = compose_config("real")
        launcher.setup_logger = lambda: None
        launcher.compose_dof_config = lambda robot_dof, policy_dof: effective_dof
        launcher.LeggedLabPolicy = FakePolicy
        launcher.instantiate = fake_instantiate
        launcher.is_upright = lambda quaternion: True
        launcher.time = SimpleNamespace(monotonic=lambda: 100.0, sleep=lambda duration: None)
        with self.assertRaisesRegex(RuntimeError, "command failure"):
            launcher.run(cfg)

        self.assertEqual(events[-1], "shutdown")


if __name__ == "__main__":
    unittest.main()
