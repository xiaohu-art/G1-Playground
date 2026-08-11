import importlib
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from box import Box

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CONFIGS = {"g1", "g1_real"}
EXPECTED_POLICIES = {"UnitreeWoGaitPolicy"}
EXPECTED_PIPELINES = {"RlPipeline"}
EXPECTED_CONTROLLERS = {"JoystickCtrl", "UnitreeCtrl"}
EXPECTED_ENVIRONMENTS = {"MujocoEnv", "UnitreeCppEnv"}
EXPECTED_CHECKPOINT = REPO_ROOT / "assets/models/g1/unitree/policy_wo_gait.pt"
EXPECTED_UNITREE_CPP_FILES = {
    ".gitignore",
    "CHANGELOG.md",
    "CMakeLists.txt",
    "LICENSE",
    "README.md",
    "asset/image.png",
    "example/config.py",
    "example/requirements.txt",
    "example/unitree_cpp_env.py",
    "pyproject.toml",
    "src/py_binding.cpp",
    "src/unitree_controller.cpp",
    "src/unitree_controller.hpp",
    "src/unitree_cpp/__init__.py",
}


class TestFullImports(unittest.TestCase):
    def assert_registry(self, registry, expected_types, import_types=None):
        self.assertEqual(set(registry.types), expected_types)
        for type_name in sorted(expected_types if import_types is None else import_types):
            with self.subTest(type=type_name):
                registered_class = registry.get(type_name)
                self.assertEqual(registered_class.__name__, type_name)

    def test_config_registry_and_29dof_contract(self):
        from robojudo.config import cfg_registry

        self.assert_registry(cfg_registry, EXPECTED_CONFIGS)

        checkpoints = set()
        for type_name in sorted(EXPECTED_CONFIGS):
            with self.subTest(config=type_name):
                cfg = cfg_registry.get(type_name)()
                self.assertEqual(cfg.robot, "g1")
                self.assertEqual(cfg.policy.policy_type, "UnitreeWoGaitPolicy")
                self.assertEqual(cfg.policy.history_length, 5)
                self.assertEqual(sum(cfg.policy.history_obs_dims.values()), 96)

                dof_configs = {
                    "environment": cfg.env.dof,
                    "observation": cfg.policy.obs_dof,
                    "action": cfg.policy.action_dof,
                }
                for label, dof_cfg in dof_configs.items():
                    with self.subTest(config=type_name, dof=label):
                        self.assertEqual(dof_cfg.num_dofs, 29)
                        self.assertEqual(len(set(dof_cfg.joint_names)), 29)

                expected_joints = set(cfg.env.dof.joint_names)
                self.assertEqual(set(cfg.policy.obs_dof.joint_names), expected_joints)
                self.assertEqual(set(cfg.policy.action_dof.joint_names), expected_joints)

                checkpoint = Path(cfg.policy.policy_file)
                self.assertEqual(checkpoint.resolve(), EXPECTED_CHECKPOINT)
                self.assertTrue(checkpoint.is_file(), f"Missing checkpoint: {checkpoint}")
                checkpoints.add(checkpoint.resolve())

        self.assertEqual(checkpoints, {EXPECTED_CHECKPOINT})

        model_root = EXPECTED_CHECKPOINT.parents[1]
        retained_checkpoints = {path.resolve() for path in model_root.rglob("*") if path.suffix in {".onnx", ".pt"}}
        self.assertEqual(retained_checkpoints, {EXPECTED_CHECKPOINT})

    def test_policy_registry(self):
        from robojudo.policy import policy_registry

        self.assert_registry(policy_registry, EXPECTED_POLICIES)

    def test_pipeline_registry(self):
        from robojudo.pipeline import pipeline_registry

        self.assert_registry(pipeline_registry, EXPECTED_PIPELINES)

    def test_controller_registry(self):
        from robojudo.controller import ctrl_registry

        self.assert_registry(ctrl_registry, EXPECTED_CONTROLLERS)

    def test_environment_registry(self):
        from robojudo.environment import env_registry

        self.assert_registry(env_registry, EXPECTED_ENVIRONMENTS, import_types={"MujocoEnv"})

    def test_unitree_cpp_environment_import(self):
        from robojudo.environment import env_registry

        try:
            importlib.import_module("unitree_cpp")
        except ImportError as exc:
            self.skipTest(f"vendored unitree_cpp binding is not built or installed: {exc}")

        environment_class = env_registry.get("UnitreeCppEnv")
        self.assertEqual(environment_class.__name__, "UnitreeCppEnv")

    def test_wogait_torchscript_shape(self):
        from robojudo.config import cfg_registry

        cfg = cfg_registry.get("g1")()
        model = torch.jit.load(cfg.policy.policy_file, map_location="cpu")
        model.eval()
        with torch.no_grad():
            output = model(torch.zeros((1, 480), dtype=torch.float32))

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(tuple(output.shape), (1, 29))

    def test_wogait_observation_and_action(self):
        from robojudo.config import cfg_registry
        from robojudo.policy.unitree_policy import UnitreeWoGaitPolicy

        cfg = cfg_registry.get("g1")().policy
        policy = UnitreeWoGaitPolicy(cfg, device="cpu")
        env_data = Box(
            {
                "base_quat": np.array([0.0, 0.0, 0.0, 1.0]),
                "base_ang_vel": np.zeros(3),
                "dof_pos": np.asarray(cfg.obs_dof.default_pos),
                "dof_vel": np.zeros(29),
            }
        )
        ctrl_data = Box({"JoystickCtrl": {"axes": {"LeftX": 0.0, "LeftY": 0.0, "RightX": 0.0, "RightY": 0.0}}})

        observation, extras = policy.get_observation(env_data, ctrl_data)
        self.assertEqual(observation.shape, (480,))
        self.assertTrue(np.allclose(extras["commands"], 0.0))
        self.assertEqual(policy.get_action(observation).shape, (29,))

    def test_robot_asset_closure(self):
        robot_root = EXPECTED_CHECKPOINT.parents[3] / "robots/g1"
        xml_path = robot_root / "g1_29dof_rev_1_0.xml"
        self.assertEqual(set(robot_root.glob("*.xml")), {xml_path})

        root = ET.parse(xml_path).getroot()
        actuator = root.find("actuator")
        self.assertIsNotNone(actuator)
        self.assertEqual(len(actuator), 29)

        compiler = root.find("compiler")
        mesh_dir = compiler.attrib.get("meshdir", "") if compiler is not None else ""
        referenced_meshes = {
            (robot_root / mesh_dir / mesh.attrib["file"]).resolve()
            for mesh in root.findall("./asset/mesh")
            if "file" in mesh.attrib
        }
        retained_meshes = {path.resolve() for path in (robot_root / "meshes").glob("*.STL")}
        self.assertEqual(retained_meshes, referenced_meshes)

    def test_vendored_dependency_layout(self):
        third_party_root = REPO_ROOT / "third_party"
        unitree_root = third_party_root / "unitree_cpp"
        viewer_root = third_party_root / "mujoco_viewer"

        for obsolete_path in (
            REPO_ROOT / ".gitmodules",
            REPO_ROOT / "submodule_cfg.yaml",
            REPO_ROOT / "submodule_install.py",
            REPO_ROOT / "packages",
            third_party_root / "patches/mujoco_viewer.patch",
        ):
            with self.subTest(obsolete_path=obsolete_path):
                self.assertFalse(obsolete_path.exists())

        self.assertTrue((REPO_ROOT / "scripts/install_third_party.py").is_file())
        self.assertTrue((third_party_root / "README.md").is_file())
        self.assertFalse(any(path.name == ".git" for path in third_party_root.rglob(".git")))
        self.assertFalse(any(third_party_root.rglob("*.egg-info")))

        unitree_files = {
            path.relative_to(unitree_root).as_posix() for path in unitree_root.rglob("*") if path.is_file()
        }
        self.assertEqual(unitree_files, EXPECTED_UNITREE_CPP_FILES)

        for viewer_path in (
            viewer_root / "LICENSE",
            viewer_root / "setup.py",
            viewer_root / "mujoco_viewer/__init__.py",
        ):
            with self.subTest(viewer_path=viewer_path):
                self.assertTrue(viewer_path.is_file())

        viewer_source = (viewer_root / "mujoco_viewer/mujoco_viewer.py").read_text()
        self.assertIn("disable_key_callbacks", viewer_source)
        self.assertIn("_persistent_markers", viewer_source)
        self.assertNotIn("diable_key_callbacks", viewer_source)
        self.assertNotIn("marker_geoms", viewer_source)
