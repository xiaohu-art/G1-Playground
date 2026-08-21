import hashlib
import json
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PRE_DDS_CONTRACT_PATH = REPO_ROOT / "tests/fixtures/pre_dds/contract.json"
PHASE1_BOUNDARY_PATH = REPO_ROOT / "tests/fixtures/dds_phase1/unitree_cpp_boundary.json"
PHASE2_BOUNDARY_PATH = REPO_ROOT / "tests/fixtures/dds_phase2/observer_activation_boundary.json"
PRE_DDS_CONTRACT_SHA256 = "d43db515848107099105f3ae2e891097410b7810bfd4ecf14d3a5ba0dbeb240b"
PHASE1_BOUNDARY_SHA256 = "f42ecbf234a9187c685914917212a2231ceab44db8c10a8c47008913158c0def"
EXCLUDED_DIRS = {".git", "__pycache__", "build", "dist"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_file_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.suffix != ".pyc"
        and not any(part in EXCLUDED_DIRS or part.endswith(".egg-info") for part in path.parts)
    }


def source_closure(root: Path) -> dict[str, int | str]:
    files = source_file_hashes(root)
    digest = hashlib.sha256()
    total_bytes = 0
    for relative_path in files:
        path = root / relative_path
        contents = path.read_bytes()
        total_bytes += len(contents)
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(contents)
        digest.update(b"\0")
    return {"file_count": len(files), "total_bytes": total_bytes, "sha256": digest.hexdigest()}


def classify_changes(baseline: dict[str, str], current: dict[str, str]) -> dict[str, list[str]]:
    baseline_paths = set(baseline)
    current_paths = set(current)
    return {
        "modified": sorted(path for path in baseline_paths & current_paths if baseline[path] != current[path]),
        "added": sorted(current_paths - baseline_paths),
        "removed": sorted(baseline_paths - current_paths),
    }


def boundary_violations(changes: dict[str, list[str]], allowed: dict[str, list[str]]) -> dict[str, list[str]]:
    return {change_type: sorted(set(paths) - set(allowed[change_type])) for change_type, paths in changes.items()}


class TestDdsPhase1Boundary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.boundary = json.loads(PHASE1_BOUNDARY_PATH.read_text())
        cls.pre_dds = json.loads(PRE_DDS_CONTRACT_PATH.read_text())

    def test_unitree_cpp_phase1_snapshot_stays_inside_phase1_boundary(self):
        vendor = self.boundary["vendor"]
        baseline = vendor["baseline_files"]
        allowed = vendor["allowed_changes"]
        phase2_boundary = json.loads(PHASE2_BOUNDARY_PATH.read_text())
        phase1_snapshot = phase2_boundary["vendor"]["phase1_files"]
        self.assertEqual(set(phase1_snapshot), set(baseline))
        self.assertEqual(
            {path: phase1_snapshot[path] for path in ("src/unitree_controller.cpp", "src/unitree_controller.hpp")},
            {
                "src/unitree_controller.cpp": "35dd91c38c4638c32384884af19184928d23d8f74357c60a3900cac0fc64a079",
                "src/unitree_controller.hpp": "5872e410bcb43165a0ea479d14d8bf2026d2b8851fbf3ad2106b70d1ba5bc211",
            },
        )
        self.assertTrue(
            {"src/g1_dds_control_endpoint.cpp", "src/g1_dds_control_endpoint.hpp"}.isdisjoint(phase1_snapshot)
        )
        changes = classify_changes(baseline, phase1_snapshot)

        self.assertLessEqual(set(allowed["modified"]), set(baseline))
        violations = boundary_violations(changes, allowed)
        for change_type, paths in violations.items():
            with self.subTest(change_type=change_type):
                self.assertEqual(paths, [], f"unapproved UnitreeCpp {change_type} paths: {paths}")

        self.assertTrue(any(changes.values()))

    def test_boundary_rejects_every_unapproved_difference_type(self):
        baseline = {"example/config.py": "old", "LICENSE": "old", "src/py_binding.cpp": "old"}
        current = {"example/config.py": "new", "LICENSE": "new", "src/domain_helper.cpp": "new"}
        changes = classify_changes(baseline, current)
        self.assertEqual(
            changes,
            {
                "modified": ["LICENSE", "example/config.py"],
                "added": ["src/domain_helper.cpp"],
                "removed": ["src/py_binding.cpp"],
            },
        )
        self.assertEqual(
            boundary_violations(
                changes,
                {"modified": ["example/config.py"], "added": [], "removed": []},
            ),
            {
                "modified": ["LICENSE"],
                "added": ["src/domain_helper.cpp"],
                "removed": ["src/py_binding.cpp"],
            },
        )

    def test_generated_vendor_outputs_do_not_change_the_source_contract(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "kept.cpp").write_text("source")
            for generated_dir, generated_file in (
                ("build", "binding.so"),
                ("dist", "archive.whl"),
                ("__pycache__", "module.pyc"),
                ("package.egg-info", "PKG-INFO"),
            ):
                directory = root / generated_dir
                directory.mkdir()
                (directory / generated_file).write_text("generated")

            self.assertEqual(source_file_hashes(root), {"kept.cpp": sha256_file(root / "kept.cpp")})


if __name__ == "__main__":
    unittest.main()
