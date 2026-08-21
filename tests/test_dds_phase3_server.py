import hashlib
import json
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE3_BOUNDARY_PATH = REPO_ROOT / "tests/fixtures/dds_phase3/mujoco_server_boundary.json"
PHASE3_BOUNDARY_SHA256 = "8a107753a05923c965bf6309c16b8bb96b5c90d8e4f418825c9c8779406c9ebf"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify_changes(baseline: dict[str, str], current: dict[str, str]) -> dict[str, list[str]]:
    baseline_paths = set(baseline)
    current_paths = set(current)
    return {
        "modified": sorted(path for path in baseline_paths & current_paths if baseline[path] != current[path]),
        "added": sorted(current_paths - baseline_paths),
        "removed": sorted(baseline_paths - current_paths),
    }


class TestDdsPhase3EvidenceBoundary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.boundary = json.loads(PHASE3_BOUNDARY_PATH.read_text())

    def test_frozen_phase3_snapshot_is_limited_to_server_files(self):
        vendor = self.boundary["vendor"]
        phase2_files = vendor["phase2_files"]
        phase3_files = vendor["phase3_files"]

        self.assertEqual(classify_changes(phase2_files, phase3_files), vendor["allowed_changes"])
        self.assertEqual(
            vendor["phase3_closure"],
            {
                "file_count": 16,
                "total_bytes": 67930,
                "sha256": "0bde2a4f5be6b11079b7338b0db1a42786ad9aa5f06b1b145bd58bc86d228557",
            },
        )
        self.assertEqual(vendor["phase3_closure"]["file_count"], len(phase3_files))
        self.assertTrue(all(len(digest) == 64 for digest in phase3_files.values()))
        # These legacy paths and bytes are frozen Phase 3 provenance, not the current live-tree names.
        self.assertEqual(
            {
                path: phase3_files[path]
                for path in (
                    "src/dds_sim_server.cpp",
                    "src/dds_sim_server.hpp",
                    "src/unitree_controller.cpp",
                    "src/unitree_controller.hpp",
                )
            },
            {
                "src/dds_sim_server.cpp": "415732e12877fda942bea5fd5bc067ac1b62eb984550be60a58944358f0abd48",
                "src/dds_sim_server.hpp": "c55131fe5b02e99417aa883593cd1243596a8c749d564f16a7f3da28d8f1cf05",
                "src/unitree_controller.cpp": "6aeb1199816fa4830654f439c7d1b169e82ea0dbd0606a58f377806de7abb795",
                "src/unitree_controller.hpp": "5497d917932993c49a54ead4dcb36ad18aa4f7670f6a02fbda33e27e4f2282f3",
            },
        )
        self.assertTrue(
            {
                "src/dds_bridge.cpp",
                "src/dds_bridge.hpp",
                "src/g1_dds_control_endpoint.cpp",
                "src/g1_dds_control_endpoint.hpp",
                "src/g1_dds_robot_endpoint.cpp",
                "src/g1_dds_robot_endpoint.hpp",
            }.isdisjoint(phase3_files)
        )

    def test_phase3_boundary_rejects_unapproved_difference_types(self):
        vendor = self.boundary["vendor"]
        baseline = vendor["phase2_files"]
        mutated = vendor["phase3_files"].copy()
        mutated["LICENSE"] = "changed"
        mutated["src/unapproved_server.cpp"] = "added"
        del mutated["README.md"]
        changes = classify_changes(baseline, mutated)
        violations = {
            change_type: sorted(set(paths) - set(vendor["allowed_changes"][change_type]))
            for change_type, paths in changes.items()
        }
        self.assertEqual(
            violations,
            {
                "modified": ["LICENSE"],
                "added": ["src/unapproved_server.cpp"],
                "removed": ["README.md"],
            },
        )

    def test_current_python_module_exposes_the_minimal_endpoint_surface(self):
        cmake = (REPO_ROOT / "third_party/unitree_cpp/CMakeLists.txt").read_text()
        binding = (REPO_ROOT / "third_party/unitree_cpp/src/py_binding.cpp").read_text()
        package = (REPO_ROOT / "third_party/unitree_cpp/src/unitree_cpp/__init__.py").read_text()

        self.assertIn("src/g1_dds_control_endpoint.cpp", cmake)
        self.assertIn("src/g1_dds_robot_endpoint.cpp", cmake)
        self.assertNotIn("src/dds_bridge.cpp", cmake)
        self.assertNotIn("src/dds_sim_server.cpp", cmake)
        self.assertNotIn("src/unitree_controller.cpp", cmake)
        for symbol in (
            "DdsCommandSnapshot",
            "DdsControlEndpointState",
            "DdsLowStateSnapshot",
            "G1DdsControlEndpoint",
            "G1DdsControlEndpointConfig",
            "G1DdsRobotEndpoint",
            "G1DdsRobotEndpointConfig",
            "G1DdsRobotEndpointStats",
            'def("get_command"',
            'def("publish_lowstate"',
            'def("close"',
        ):
            with self.subTest(symbol=symbol):
                self.assertIn(symbol, binding)
        for legacy_symbol in (
            "ControllerState",
            "DdsBridgeConfig",
            "DdsBridgeStats",
            "DdsSimServerConfig",
            "DdsSimServerStats",
            "G1DdsBridge",
            "G1DdsSimServer",
            "UnitreeConfig",
            "UnitreeController",
        ):
            with self.subTest(legacy_symbol=legacy_symbol):
                self.assertNotIn(legacy_symbol, binding)
                self.assertNotIn(legacy_symbol, package)
        for public_symbol in ("DdsLowStateSnapshot", "G1DdsControlEndpoint", "G1DdsRobotEndpoint"):
            with self.subTest(public_symbol=public_symbol):
                self.assertIn(public_symbol, package)


class TestCurrentNativeDdsEndpoints(unittest.TestCase):
    def test_live_native_tree_has_no_legacy_endpoint_aliases(self):
        source_root = REPO_ROOT / "third_party/unitree_cpp/src"
        native_source = "\n".join(
            path.read_text() for pattern in ("*.cpp", "*.hpp") for path in sorted(source_root.glob(pattern))
        )
        for legacy_symbol in (
            "ControllerState",
            "DdsBridgeConfig",
            "DdsBridgeStats",
            "DdsSimServerConfig",
            "DdsSimServerStats",
            "G1DdsBridge",
            "G1DdsSimServer",
            "UnitreeConfig",
            "UnitreeController",
        ):
            with self.subTest(legacy_symbol=legacy_symbol):
                self.assertNotIn(legacy_symbol, native_source)

    def test_control_and_robot_endpoints_share_only_neutral_dds_utils(self):
        source_root = REPO_ROOT / "third_party/unitree_cpp/src"
        robot_header = (source_root / "g1_dds_robot_endpoint.hpp").read_text()
        robot_source = (source_root / "g1_dds_robot_endpoint.cpp").read_text()
        control_header = (source_root / "g1_dds_control_endpoint.hpp").read_text()
        control_source = (source_root / "g1_dds_control_endpoint.cpp").read_text()

        self.assertIn('#include "dds_utils.hpp"', robot_header + robot_source)
        self.assertIn('#include "dds_utils.hpp"', control_header + control_source)
        for robot_file in (robot_header, robot_source):
            self.assertNotIn("g1_dds_control_endpoint.hpp", robot_file)
        for control_file in (control_header, control_source):
            self.assertNotIn("g1_dds_robot_endpoint.hpp", control_file)

    def test_native_crc_has_no_private_sdk_header_dependency(self):
        unstable_header = "unitree/dds_wrapper/common/crc.h"
        native_sources = [
            *(REPO_ROOT / "third_party/unitree_cpp/src").glob("*.cpp"),
            *(REPO_ROOT / "third_party/unitree_cpp/src").glob("*.hpp"),
            *(REPO_ROOT / "tests").glob("test_*.cpp"),
            *(REPO_ROOT / "tests").glob("test_*.py"),
        ]

        for source_path in native_sources:
            with self.subTest(source=source_path.relative_to(REPO_ROOT)):
                self.assertNotIn(f"#include <{unstable_header}>", source_path.read_text())

    def test_command_validation_and_lowstate_packing_without_live_dds(self):
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

        source = textwrap.dedent(
            r"""
            #include <cassert>
            #include <array>
            #include <cmath>
            #include <cstdint>
            #include <limits>
            #include <stdexcept>

            #include "g1_dds_robot_endpoint.hpp"
            #include "dds_utils.hpp"

            template <typename Message>
            std::uint32_t message_crc(Message& message) {
                return unitree_cpp_detail::Crc32Core(
                    reinterpret_cast<const std::uint32_t*>(&message),
                    static_cast<std::uint32_t>((sizeof(Message) >> 2) - 1));
            }

            int main() {
                using unitree_cpp_detail::DdsCommandValidation;
                using unitree_cpp_detail::FillDdsLowStateForTest;
                using unitree_cpp_detail::ValidateDdsCommandForTest;
                using LowCmd = unitree_hg::msg::dds_::LowCmd_;
                using LowState = unitree_hg::msg::dds_::LowState_;

                const std::array<std::uint32_t, 4> crc_fixture = {
                    0x00000000U, 0x12345678U, 0x9ABCDEF0U, 0xFFFFFFFFU};
                assert(unitree_cpp_detail::Crc32Core(crc_fixture.data(), crc_fixture.size()) == 0x85C09DADU);

                LowCmd command;
                command.mode_pr(0);
                command.mode_machine(5);
                for (std::size_t index = 0; index < 29; ++index) {
                    auto& motor = command.motor_cmd().at(index);
                    motor.mode(1);
                    motor.q(static_cast<float>(index));
                    motor.dq(static_cast<float>(index) + 0.25F);
                    motor.tau(static_cast<float>(index) + 0.5F);
                    motor.kp(100.0F + static_cast<float>(index));
                    motor.kd(2.0F + static_cast<float>(index));
                }
                command.crc(message_crc(command));

                DdsCommandSnapshot accepted;
                assert(ValidateDdsCommandForTest(command, 5, &accepted) == DdsCommandValidation::ACCEPTED);
                assert(accepted.valid);
                assert(accepted.q.size() == 29);
                assert(accepted.q.at(28) == 28.0);
                assert(accepted.dq.at(3) == 3.25);
                assert(accepted.tau.at(4) == 4.5);
                assert(accepted.kp.at(7) == 107.0);
                assert(accepted.kd.at(8) == 10.0);

                LowCmd bad_crc = command;
                bad_crc.crc(bad_crc.crc() ^ 1U);
                assert(ValidateDdsCommandForTest(bad_crc, 5, &accepted) == DdsCommandValidation::CRC_ERROR);

                LowCmd bad_mode = command;
                bad_mode.mode_machine(4);
                bad_mode.crc(message_crc(bad_mode));
                assert(ValidateDdsCommandForTest(bad_mode, 5, &accepted) == DdsCommandValidation::MODE_ERROR);

                LowCmd bad_value = command;
                bad_value.motor_cmd().at(2).q(std::numeric_limits<float>::infinity());
                bad_value.crc(message_crc(bad_value));
                assert(ValidateDdsCommandForTest(bad_value, 5, &accepted) == DdsCommandValidation::VALUE_ERROR);

                DdsLowStateSnapshot state;
                state.q.at(3) = 1.25;
                state.dq.at(4) = -2.5;
                state.tau_est.at(5) = 3.75;
                state.quaternion = {0.5, 0.5, -0.5, -0.5};
                state.gyroscope = {1.0, 2.0, 3.0};
                state.accelerometer = {4.0, 5.0, 6.0};
                state.rpy = {7.0, 8.0, 9.0};
                state.wireless_remote.at(39) = 42;

                LowState low_state;
                const auto crc = FillDdsLowStateForTest(state, 5, 123, &low_state);
                assert(low_state.mode_pr() == 0);
                assert(low_state.mode_machine() == 5);
                assert(low_state.tick() == 123);
                assert(low_state.motor_state().at(3).q() == 1.25F);
                assert(low_state.motor_state().at(4).dq() == -2.5F);
                assert(low_state.motor_state().at(5).tau_est() == 3.75F);
                assert(low_state.imu_state().quaternion().at(2) == -0.5F);
                assert(low_state.imu_state().gyroscope().at(1) == 2.0F);
                assert(low_state.imu_state().accelerometer().at(2) == 6.0F);
                assert(low_state.imu_state().rpy().at(0) == 7.0F);
                assert(low_state.wireless_remote().at(39) == 42);
                assert(low_state.crc() == crc);
                assert(message_crc(low_state) == crc);

                DdsLowStateSnapshot wrong_size;
                wrong_size.q.resize(28);
                bool rejected_size = false;
                try {
                    FillDdsLowStateForTest(wrong_size, 5, 1, &low_state);
                } catch (const std::invalid_argument&) {
                    rejected_size = true;
                }
                assert(rejected_size);

                DdsLowStateSnapshot nonfinite;
                nonfinite.gyroscope.at(0) = std::numeric_limits<double>::quiet_NaN();
                bool rejected_nonfinite = false;
                try {
                    FillDdsLowStateForTest(nonfinite, 5, 1, &low_state);
                } catch (const std::invalid_argument&) {
                    rejected_nonfinite = true;
                }
                assert(rejected_nonfinite);

                DdsLowStateSnapshot too_large;
                too_large.q.at(0) = std::numeric_limits<double>::max();
                bool rejected_narrowing = false;
                try {
                    FillDdsLowStateForTest(too_large, 5, 1, &low_state);
                } catch (const std::invalid_argument&) {
                    rejected_narrowing = true;
                }
                assert(rejected_narrowing);
                return 0;
            }
            """
        )

        with tempfile.TemporaryDirectory(prefix="g1-playground-dds-phase3-robot-endpoint-") as temporary_dir:
            temporary_path = Path(temporary_dir)
            source_path = temporary_path / "g1_dds_robot_endpoint_test.cpp"
            binary_path = temporary_path / "g1_dds_robot_endpoint_test"
            source_path.write_text(source)
            compile_result = subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    f"-I{REPO_ROOT / 'third_party/unitree_cpp/src'}",
                    f"-I{sdk_include}",
                    f"-I{dds_include}",
                    source_path,
                    REPO_ROOT / "third_party/unitree_cpp/src/g1_dds_robot_endpoint.cpp",
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


if __name__ == "__main__":
    unittest.main()
