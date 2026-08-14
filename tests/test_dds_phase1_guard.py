import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UNITREE_CPP_INCLUDE = REPO_ROOT / "third_party/unitree_cpp/src"


class TestDdsEndpointInitGuard(unittest.TestCase):
    def test_guard_semantics_without_initializing_dds(self):
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

            #include "unitree_controller.hpp"

            using unitree_cpp_detail::DdsEndpointInitGuard;

            int main() {
                DdsEndpointInitGuard guard;
                int calls = 0;
                const auto initializer = [&calls](std::int32_t domain_id, const std::string& net_if) {
                    assert(domain_id == 1);
                    assert(net_if == "lo");
                    ++calls;
                };

                assert(guard.InitializeOnce(1, "lo", initializer));
                assert(!guard.InitializeOnce(1, "lo", initializer));
                assert(calls == 1);

                bool domain_conflict = false;
                try {
                    guard.InitializeOnce(0, "lo", initializer);
                } catch (const std::runtime_error&) {
                    domain_conflict = true;
                }
                assert(domain_conflict);

                bool interface_conflict = false;
                try {
                    guard.InitializeOnce(1, "testnic", initializer);
                } catch (const std::runtime_error&) {
                    interface_conflict = true;
                }
                assert(interface_conflict);
                assert(calls == 1);

                DdsEndpointInitGuard retry_guard;
                bool initializer_failed = false;
                try {
                    retry_guard.InitializeOnce(1, "lo", [](std::int32_t, const std::string&) {
                        throw std::runtime_error("injected failure");
                    });
                } catch (const std::runtime_error&) {
                    initializer_failed = true;
                }
                assert(initializer_failed);
                assert(retry_guard.InitializeOnce(1, "lo", [](std::int32_t, const std::string&) {}));

                DdsEndpointInitGuard concurrent_guard;
                std::atomic<int> concurrent_calls{0};
                std::atomic<int> first_callers{0};
                std::vector<std::thread> threads;
                for (int index = 0; index < 16; ++index) {
                    threads.emplace_back([&]() {
                        if (concurrent_guard.InitializeOnce(1, "lo", [&](std::int32_t, const std::string&) {
                                ++concurrent_calls;
                            })) {
                            ++first_callers;
                        }
                    });
                }
                for (auto& thread : threads) {
                    thread.join();
                }
                assert(concurrent_calls == 1);
                assert(first_callers == 1);

                return 0;
            }
            """
        )

        with tempfile.TemporaryDirectory(prefix="g1-playground-dds-guard-") as temporary_dir:
            temporary_path = Path(temporary_dir)
            source_path = temporary_path / "guard_test.cpp"
            binary_path = temporary_path / "guard_test"
            source_path.write_text(source)
            compile_result = subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-pthread",
                    f"-I{UNITREE_CPP_INCLUDE}",
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


if __name__ == "__main__":
    unittest.main()
