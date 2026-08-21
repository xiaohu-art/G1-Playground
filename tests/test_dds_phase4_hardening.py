import importlib.util
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
UNITREE_CPP_ROOT = REPO_ROOT / "third_party/unitree_cpp/src"


def cpp_block(source: str, marker: str) -> str:
    start = source.index(marker)
    opening_brace = source.index("{", start)
    depth = 0
    for index in range(opening_brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening_brace + 1 : index]
    raise AssertionError(f"unterminated C++ block: {marker}")


def load_pipeline_launcher():
    source_path = REPO_ROOT / "scripts/pipeline.py"
    spec = importlib.util.spec_from_file_location("g1_playground_phase4_pipeline_test", source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load launcher from {source_path}")
    module = importlib.util.module_from_spec(spec)
    logger_module = SimpleNamespace(setup_logger=lambda: None)
    policy_module = SimpleNamespace(UnitreeWoGaitPolicy=object)
    with patch.dict(
        "sys.modules",
        {"g1_playground.policy": policy_module, "g1_playground.utils.logger": logger_module},
    ):
        spec.loader.exec_module(module)
    return module


class StepFixture:
    def __init__(self, shutdown_requested):
        self.events = []
        self.shutdown_requested = shutdown_requested
        self.state = SimpleNamespace(base_quat="quaternion")
        self.env = SimpleNamespace(read=self.read, step=self.write)
        self.controller = SimpleNamespace(read=self.read_control)
        self.policy = SimpleNamespace(act=self.act)

    def read(self):
        self.events.append("env.read")
        return self.state

    def read_control(self):
        self.events.append("controller.read")
        return "control", self.shutdown_requested

    def act(self, state, control):
        if state is not self.state:
            raise AssertionError("policy must receive the post-safety environment snapshot")
        self.events.append("policy.act")
        return "target"

    def write(self, target):
        self.events.append("env.step")


class TestPipelineCommandGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.launcher = load_pipeline_launcher()

    def test_failed_safety_check_is_handled_before_command_write(self):
        fixture = StepFixture(False)

        def unsafe(quaternion):
            self.assertEqual(quaternion, "quaternion")
            fixture.events.append("is_upright")
            return False

        with patch.object(self.launcher, "is_upright", side_effect=unsafe):
            self.assertFalse(
                self.launcher.step(
                    fixture.env,
                    fixture.controller,
                    fixture.policy,
                )
            )

        self.assertIn("is_upright", fixture.events)
        self.assertNotIn("env.step", fixture.events)
        self.assertEqual(fixture.events, ["env.read", "controller.read", "is_upright"])


class TestNativeStateHardening(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.header = (UNITREE_CPP_ROOT / "g1_dds_control_endpoint.hpp").read_text()
        cls.source = (UNITREE_CPP_ROOT / "g1_dds_control_endpoint.cpp").read_text()
        cls.binding = (UNITREE_CPP_ROOT / "py_binding.cpp").read_text()

    def test_robot_state_owns_mode_and_monotonic_receive_time(self):
        state = cpp_block(self.header, "struct RobotState")
        self.assertRegex(state, r"\buint8_t\s+mode_machine\s*;")
        self.assertRegex(state, r"std::chrono::steady_clock::time_point\s+received_at\s*;")
        self.assertEqual(
            self.binding.count('.def_readwrite("mode_machine", &RobotState::mode_machine)'),
            1,
        )

    def test_lowstate_validation_precedes_single_atomic_commit(self):
        handler = cpp_block(self.source, "void G1DdsControlEndpoint::LowStateHandler")
        validator = cpp_block(self.source, "bool unitree_cpp_detail::IsValidLowStateCandidate")
        commit = "robot_state_buffer_.SetData(robot_state_tmp)"

        self.assertEqual(handler.count(commit), 1)
        commit_index = handler.index(commit)
        for required in (
            "robot_state_tmp.mode_machine = low_state.mode_machine()",
            "robot_state_tmp.received_at = SteadyClock::now()",
            "IsValidLowStateCandidate(robot_state_tmp, previous_state.get(), 5)",
        ):
            with self.subTest(required=required):
                self.assertIn(required, handler)
                self.assertLess(handler.index(required), commit_index)
        for required in (
            "candidate.mode_machine != expected_mode",
            "AllFinite(candidate.motor_state.q)",
            "AllFinite(candidate.motor_state.dq)",
            "AllFinite(candidate.motor_state.tau_est)",
            "AllFinite(candidate.imu_state.quaternion)",
            "TickAdvanced(previous->tick, candidate.tick)",
        ):
            with self.subTest(validator=required):
                self.assertIn(required, validator)

    def test_state_freshness_uses_one_derived_timeout(self):
        timeout = cpp_block(self.source, "double ControlEndpointTimeout")
        validate = cpp_block(self.source, "void G1DdsControlEndpoint::ValidateRobotState")
        read = cpp_block(self.source, "RobotState G1DdsControlEndpoint::get_robot_state")

        self.assertRegex(timeout, r"std::max\(5\.0\s*\*\s*cfg\.control_dt,\s*0\.1\)")
        self.assertIn("IsFreshTimestamp(state.received_at, SteadyClock::now(), ControlEndpointTimeout(cfg_))", validate)
        self.assertIn("Low state data is stale", validate)
        self.assertIn("ValidateRobotState(*robot_state)", read)

    def test_native_constructor_owns_control_period_validation(self):
        constructor = cpp_block(self.source, "G1DdsControlEndpoint::G1DdsControlEndpoint")
        validation = "!std::isfinite(cfg_.control_dt) || cfg_.control_dt <= 0"

        self.assertIn(validation, constructor)
        self.assertLess(constructor.index(validation), constructor.index("InitChannelFactoryOnce(cfg_)"))

    def test_lowstate_candidate_and_freshness_helpers_without_live_dds(self):
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
        library_dir = next(
            (
                path
                for path in (Path("/usr/local/lib"), Path("/usr/lib"))
                if any((path / name).is_file() for name in ("libunitree_sdk2.so", "libunitree_sdk2.a"))
                and (path / "libddscxx.so").is_file()
                and (path / "libddsc.so").is_file()
            ),
            None,
        )
        if compiler is None or sdk_include is None or dds_include is None or library_dir is None:
            self.skipTest("C++ compiler or Unitree SDK2/CycloneDDS development files are unavailable")

        production_source = (UNITREE_CPP_ROOT / "g1_dds_control_endpoint.cpp").as_posix()
        source = textwrap.dedent(
            f"""
            #include <cassert>
            #include <chrono>
            #include <cstdint>
            #include <limits>

            #define main g1_dds_control_endpoint_example_main
            #include "{production_source}"
            #undef main

            int main() {{
                using Clock = std::chrono::steady_clock;
                using unitree_cpp_detail::IsFreshTimestamp;
                using unitree_cpp_detail::IsValidLowStateCandidate;

                RobotState state(29);
                state.tick = 10;
                state.mode_machine = 5;
                state.received_at = Clock::time_point(std::chrono::seconds(7));
                assert(IsValidLowStateCandidate(state, nullptr, 5));

                RobotState previous(29);
                previous.tick = 9;
                previous.mode_machine = 5;
                assert(IsValidLowStateCandidate(state, &previous, 5));
                previous.tick = 10;
                assert(!IsValidLowStateCandidate(state, &previous, 5));
                previous.tick = 11;
                assert(!IsValidLowStateCandidate(state, &previous, 5));

                previous.tick = std::numeric_limits<std::uint32_t>::max();
                state.tick = 0;
                assert(IsValidLowStateCandidate(state, &previous, 5));
                state.tick = 10;

                state.mode_machine = 4;
                assert(!IsValidLowStateCandidate(state, nullptr, 5));
                state.mode_machine = 5;
                state.motor_state.q.at(3) = std::numeric_limits<float>::quiet_NaN();
                assert(!IsValidLowStateCandidate(state, nullptr, 5));
                state.motor_state.q.at(3) = 0.0F;
                state.imu_state.gyroscope.at(1) = std::numeric_limits<float>::infinity();
                assert(!IsValidLowStateCandidate(state, nullptr, 5));

                const auto now = Clock::time_point(std::chrono::seconds(8));
                assert(IsFreshTimestamp(now - std::chrono::milliseconds(99), now, 0.1));
                assert(!IsFreshTimestamp(now - std::chrono::milliseconds(101), now, 0.1));
                assert(!IsFreshTimestamp(now + std::chrono::milliseconds(1), now, 0.1));
                return 0;
            }}
            """
        )

        with tempfile.TemporaryDirectory(prefix="g1-playground-dds-phase4-state-") as temporary_dir:
            temporary_path = Path(temporary_dir)
            source_path = temporary_path / "state_hardening_test.cpp"
            binary_path = temporary_path / "state_hardening_test"
            source_path.write_text(source)
            compile_result = subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    f"-I{UNITREE_CPP_ROOT}",
                    f"-I{sdk_include}",
                    f"-I{dds_include}",
                    source_path,
                    f"-L{library_dir}",
                    f"-Wl,-rpath,{library_dir}",
                    "-lunitree_sdk2",
                    "-lddscxx",
                    "-lddsc",
                    "-ldl",
                    "-pthread",
                    "-o",
                    binary_path,
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
            run_result = subprocess.run([binary_path], capture_output=True, text=True, timeout=10)
            self.assertEqual(run_result.returncode, 0, run_result.stderr)


class TestNativeCommandHardening(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (UNITREE_CPP_ROOT / "g1_dds_control_endpoint.cpp").read_text()
        cls.robot_endpoint_source = (UNITREE_CPP_ROOT / "g1_dds_robot_endpoint.cpp").read_text()

    def test_step_rejects_nonfinite_target_before_buffer_update(self):
        step = cpp_block(self.source, "void G1DdsControlEndpoint::step(")
        finite_gate = step.index("AllRepresentableAsFloat(actions)")
        update = step.index("WriteMotorCommand(motor_command_tmp)")
        self.assertLess(finite_gate, update)
        self.assertRegex(step[finite_gate:update], r"representable")

    def test_periodic_writer_stops_on_stale_state_as_well_as_stale_command(self):
        writer = cpp_block(self.source, "void G1DdsControlEndpoint::LowCommandWriter")
        self.assertIn("command_write_mutex_", writer)
        locked_writer = cpp_block(self.source, "void G1DdsControlEndpoint::LowCommandWriterLocked")
        self.assertIn("CommandExpired() || !HasFreshRobotState()", locked_writer)
        self.assertLess(
            locked_writer.index("CommandExpired() || !HasFreshRobotState()"),
            locked_writer.index("lowcmd_publisher_->Write"),
        )

    def test_expired_command_sends_one_damping_frame_then_stops(self):
        writer = cpp_block(self.source, "void G1DdsControlEndpoint::LowCommandWriterLocked")
        self.assertIn("if (CommandExpired() || !HasFreshRobotState())", writer)
        self.assertEqual(writer.count("command_watchdog_fired_.exchange(true)"), 1)
        self.assertEqual(writer.count("SendDampingCommand()"), 1)
        expired_branch = writer.split("if (CommandExpired() || !HasFreshRobotState())", 1)[1].split(
            "LowCmd_ dds_low_command", 1
        )[0]
        self.assertIn("motor_command_buffer_.Clear()", expired_branch)
        self.assertIn("return;", expired_branch)

        step = cpp_block(self.source, "void G1DdsControlEndpoint::step(")
        self.assertIn("command_watchdog_fired_.load()", step)
        self.assertIn("command watchdog expired", step)

        submit = cpp_block(self.source, "void G1DdsControlEndpoint::WriteMotorCommand")
        self.assertIn("command_write_mutex_", submit)
        self.assertIn("command_watchdog_fired_.load()", submit)
        self.assertNotIn("command_watchdog_fired_.store(false)", submit)

    def test_transport_failures_and_endpoint_initialization_are_not_silent(self):
        writer = cpp_block(self.source, "void G1DdsControlEndpoint::LowCommandWriterLocked")
        damping = cpp_block(self.source, "void G1DdsControlEndpoint::SendDampingCommand")
        for block in (writer, damping):
            self.assertIn("if (!lowcmd_publisher_->Write", block)
            self.assertRegex(block, r'throw std::runtime_error\("Failed to publish DDS')
        self.assertIn("InitializeDdsEndpointOnce(cfg_.domain_id, cfg_.net_if)", self.robot_endpoint_source)
        self.assertNotIn("ChannelFactory::Instance()->Init(cfg_.domain_id", self.robot_endpoint_source)

    def test_motion_switcher_uses_the_remaining_total_deadline(self):
        activation = cpp_block(self.source, "void G1DdsControlEndpoint::InitializeCommandTransport")
        self.assertIn("release_deadline", activation)
        self.assertIn("duration<float>(remaining)", activation)
        self.assertLess(activation.index("SetTimeout"), activation.index("CheckMode"))
        self.assertIn("duration<float>(release_remaining)", activation)
        self.assertIn("release_remaining <= SteadyClock::duration::zero()", activation)
        self.assertLess(activation.index("duration<float>(release_remaining)"), activation.index("ReleaseMode"))
        self.assertIn("sleep_remaining", activation)


if __name__ == "__main__":
    unittest.main()
