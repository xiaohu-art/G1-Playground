import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from hydra.errors import MissingConfigException
from omegaconf import OmegaConf

from tests.config_helpers import asset_path, compose_config
from tests.runner_helpers import leggedlab_runner

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DEPLOYMENTS = {"sim", "real"}
EXPECTED_POLICIES = {"LeggedLabPolicy"}
EXPECTED_CONTROLLERS = {"KeyboardCtrl", "UnitreeCtrl"}
EXPECTED_CHECKPOINT = REPO_ROOT / "assets/models/leggedlab/g1_policy.onnx"
EXPECTED_COMPONENTS = {
    "sim": (
        "g1_playground.g1_env.G1Env",
        "g1_playground.controller.keyboard_ctrl.KeyboardCtrl",
    ),
    "real": (
        "g1_playground.g1_env.G1Env",
        "g1_playground.controller.unitree_ctrl.UnitreeCtrl",
    ),
}
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
    "src/dds_utils.hpp",
    "src/g1_dds_control_endpoint.cpp",
    "src/g1_dds_control_endpoint.hpp",
    "src/g1_dds_robot_endpoint.cpp",
    "src/g1_dds_robot_endpoint.hpp",
    "src/inspire_dds_endpoint.cpp",
    "src/inspire_dds_endpoint.hpp",
    "src/py_binding.cpp",
    "src/unitree_cpp/__init__.py",
}


class TestFullImports(unittest.TestCase):
    def test_explicit_config_profiles_and_29dof_contract(self):
        checkpoints = set()
        for deployment in sorted(EXPECTED_DEPLOYMENTS):
            with self.subTest(deployment=deployment):
                cfg = compose_config(deployment)
                self.assertEqual(set(cfg.robot), {"xml", "dof"})
                self.assertEqual(
                    (cfg.env._target_, cfg.controller._target_),
                    EXPECTED_COMPONENTS[deployment],
                )
                self.assertNotIn("environment", cfg)
                self.assertNotIn("components", cfg)
                round_trip = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
                self.assertEqual(round_trip, cfg)
                self.assertNotIn("history_length", cfg.policy)
                self.assertNotIn("history_obs_dims", cfg.policy)

                dof_configs = {
                    "environment": cfg.robot.dof,
                    "policy": cfg.policy.dof,
                }
                for label, dof_cfg in dof_configs.items():
                    with self.subTest(deployment=deployment, dof=label):
                        self.assertEqual(len(dof_cfg.joint_names), 29)
                        self.assertEqual(len(set(dof_cfg.joint_names)), 29)

                expected_joints = set(cfg.robot.dof.joint_names)
                self.assertEqual(set(cfg.policy.dof.joint_names), expected_joints)

                checkpoint = asset_path(cfg.policy.policy_file)
                self.assertEqual(checkpoint.resolve(), EXPECTED_CHECKPOINT)
                self.assertTrue(checkpoint.is_file(), f"Missing checkpoint: {checkpoint}")
                checkpoints.add(checkpoint.resolve())

        self.assertEqual(checkpoints, {EXPECTED_CHECKPOINT})

        model_root = EXPECTED_CHECKPOINT.parents[0]
        retained_checkpoints = {path.resolve() for path in model_root.rglob("*") if path.suffix in {".onnx", ".pt"}}
        self.assertEqual(retained_checkpoints, {EXPECTED_CHECKPOINT})

        first = compose_config("sim")
        second = compose_config("sim")
        first.policy.dof.default_pos[0] = 123.0
        self.assertNotEqual(first.policy.dof.default_pos, second.policy.dof.default_pos)
        with self.assertRaises(MissingConfigException):
            compose_config("missing")

        real = compose_config("real")
        self.assertEqual(real.env.domain_id, 0)
        self.assertNotIn("unitree", real.env)
        self.assertIs(real.env.motion_switcher_required, True)
        self.assertNotIn("target", real.env)
        self.assertNotIn("act", real.env)

        sim = compose_config("sim")
        self.assertEqual(sim.env.domain_id, 1)
        self.assertEqual(sim.env.net_if, "lo")
        self.assertIs(sim.env.motion_switcher_required, False)
        self.assertEqual(sim.controller._target_, "g1_playground.controller.keyboard_ctrl.KeyboardCtrl")

        for deployment in EXPECTED_DEPLOYMENTS:
            cfg = compose_config(deployment)
            self.assertNotIn("do_safety_check", cfg)
            self.assertNotIn("run_fullspeed", cfg)

    def test_controller_owns_its_read(self):
        from g1_playground.controller.base_ctrl import Controller

        controller = Controller(("LeftX",))
        controller.event_queue.put({"type": "button", "name": "A", "pressed": True})

        control, shutdown_requested = controller.read()

        self.assertEqual(control, {"axes": {"LeftX": 0.0}})
        self.assertTrue(shutdown_requested)

    def test_leggedlab_observation_and_action(self):
        from g1_playground.policy import LeggedLabPolicy
        from g1_playground.utils.dof import compose_dof_config

        cfg = compose_config("sim")
        effective_dof = compose_dof_config(cfg.robot.dof, cfg.policy.dof)
        policy = LeggedLabPolicy(cfg.policy, dof_cfg=effective_dof, runner=leggedlab_runner())
        env_data = SimpleNamespace(
            base_quat=np.array([0.0, 0.0, 0.0, 1.0]),
            base_ang_vel=np.zeros(3),
            dof_pos=np.asarray(effective_dof.default_pos),
            dof_vel=np.zeros(29),
        )
        ctrl_data = {"axes": {"LeftX": 0.0, "LeftY": 0.0, "RightX": 0.0, "RightY": 0.0}}

        target = policy.act(env_data, ctrl_data)
        self.assertEqual(target.shape, (29,))
        self.assertEqual(policy.standing_target.shape, (29,))

        ctrl_data["axes"].update({"LeftY": 0.039, "LeftX": -0.039, "RightX": 0.04})
        self.assertEqual(policy.act(env_data, ctrl_data).shape, (29,))

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

        installer = REPO_ROOT / "scripts/setup/install_third_party.py"
        self.assertTrue(installer.is_file())
        installer_source = installer.read_text()
        self.assertIn('"unitree_cpp": ROOT_DIR / "third_party/unitree_cpp"', installer_source)
        self.assertNotIn('"mujoco_viewer":', installer_source)
        self.assertFalse(any(path.name == ".git" for path in third_party_root.rglob(".git")))
        self.assertFalse(any(third_party_root.rglob("*.egg-info")))

        unitree_files = {
            path.relative_to(unitree_root).as_posix()
            for path in unitree_root.rglob("*")
            if path.is_file()
            and not any(part in {"build", "dist", "__pycache__"} or part.endswith(".egg-info") for part in path.parts)
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
