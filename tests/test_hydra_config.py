import ast
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from tests.config_helpers import CONFIG_DIR, compose_config

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts/pipeline.py"


class TestHydraConfig(unittest.TestCase):
    def test_configuration_data_lives_in_the_hydra_tree(self):
        yaml_files = {path.relative_to(CONFIG_DIR).as_posix() for path in CONFIG_DIR.rglob("*.yaml")}
        self.assertEqual(
            yaml_files,
            {
                "run_pipeline.yaml",
                "run_track.yaml",
                "run_loco_track.yaml",
                "run_body_hand.yaml",
                "run_loco_hoi_track.yaml",
                "deployment/sim.yaml",
                "deployment/real.yaml",
                "robot/g1.yaml",
                "robot/inspire.yaml",
                "policy/leggedlab_g1.yaml",
                "policy/track.yaml",
                "policy/body_hand_distill_largebox.yaml",
            },
        )
        self.assertFalse((REPO_ROOT / "g1_playground/config").exists())

    def test_root_config_only_composes_config_groups_and_launcher_settings(self):
        root = OmegaConf.load(CONFIG_DIR / "run_pipeline.yaml")
        self.assertEqual(set(root), {"defaults", "recording", "env", "hydra"})

        robot_path = CONFIG_DIR / "robot/g1.yaml"
        policy_path = CONFIG_DIR / "policy/leggedlab_g1.yaml"
        self.assertGreater(robot_path.stat().st_size, 0)
        self.assertGreater(policy_path.stat().st_size, 0)

        robot = OmegaConf.load(robot_path)
        policy = OmegaConf.load(policy_path)
        self.assertEqual(set(robot), {"xml", "dof"})
        self.assertEqual(
            set(robot.dof),
            {"joint_names", "torque_limits"},
        )
        self.assertEqual(
            set(policy.dof),
            {"joint_names", "default_pos", "stiffness", "damping"},
        )
        self.assertEqual(len(robot.dof.joint_names), 29)
        self.assertEqual(len(policy.dof.joint_names), 29)
        for removed_field in ("obs_dof", "action_dof", "freq", "history_length", "history_obs_dims"):
            self.assertNotIn(removed_field, policy)

    def test_track_uses_stable_distinct_inspire_serial_paths(self):
        root = OmegaConf.load(CONFIG_DIR / "run_track.yaml")
        self.assertEqual(set(root.inspire_serial), {"left", "right"})
        self.assertNotEqual(root.inspire_serial.left, root.inspire_serial.right)
        self.assertTrue(root.inspire_serial.left.startswith("/dev/serial/by-path/"))
        self.assertTrue(root.inspire_serial.right.startswith("/dev/serial/by-path/"))

    def test_native_composition_selects_the_runtime_backend(self):
        expected_targets = {
            "sim": {
                "environment": "g1_playground.g1_env.G1Env",
                "controller": "g1_playground.controller.keyboard_ctrl.KeyboardCtrl",
            },
            "real": {
                "environment": "g1_playground.g1_env.G1Env",
                "controller": "g1_playground.controller.unitree_ctrl.UnitreeCtrl",
            },
        }
        for deployment in ("sim", "real"):
            with self.subTest(deployment=deployment):
                cfg = compose_config(deployment)
                self.assertNotIn("unitree", cfg.env)
                self.assertNotIn("environment", cfg)
                self.assertNotIn("components", cfg)
                self.assertEqual(cfg.env._target_, expected_targets[deployment]["environment"])
                self.assertEqual(cfg.controller._target_, expected_targets[deployment]["controller"])
                self.assertIn("_target_", cfg.controller)
                expected_environment_fields = {
                    "_target_",
                    "domain_id",
                    "net_if",
                    "lowcmd_topic",
                    "lowstate_topic",
                    "enable_odometry",
                    "sport_state_topic",
                    "motion_switcher_required",
                }
                self.assertEqual(set(cfg.env), expected_environment_fields)
                self.assertEqual(len(cfg.robot.dof.joint_names), 29)
                self.assertEqual(len(cfg.policy.dof.joint_names), 29)

        sim = compose_config()
        self.assertEqual(sim.env.domain_id, 1)
        self.assertEqual(sim.env.net_if, "lo")
        self.assertIs(sim.env.motion_switcher_required, False)

        real = compose_config("real")
        self.assertEqual(real.env.domain_id, 0)
        self.assertEqual(real.env.net_if, "enP8p1s0")
        self.assertIs(real.env.motion_switcher_required, True)
        for cfg in (sim, real):
            self.assertNotIn("do_safety_check", cfg)
            self.assertNotIn("run_fullspeed", cfg)

    def test_policy_contract_is_owned_by_the_policy(self):
        from g1_playground.policy import LeggedLabPolicy
        from g1_playground.utils.dof import compose_dof_config

        self.assertEqual(LeggedLabPolicy.FREQ, 50)
        self.assertEqual(LeggedLabPolicy.HISTORY_LENGTH, 1)
        self.assertEqual(sum(dim for _, dim in LeggedLabPolicy.HISTORY_LAYOUT), 96)

        mutations = {
            "joint names": lambda cfg: setattr(cfg.policy.dof, "joint_names", list(cfg.policy.dof.joint_names[:-1])),
            "default position": lambda cfg: setattr(
                cfg.policy.dof, "default_pos", list(cfg.policy.dof.default_pos[:-1])
            ),
            "stiffness": lambda cfg: setattr(cfg.policy.dof, "stiffness", list(cfg.policy.dof.stiffness[:-1])),
            "damping": lambda cfg: setattr(cfg.policy.dof, "damping", list(cfg.policy.dof.damping[:-1])),
        }
        for label, mutate in mutations.items():
            with self.subTest(contract=label):
                cfg = compose_config("sim")
                effective_dof = compose_dof_config(cfg.robot.dof, cfg.policy.dof)
                mutate(cfg)
                with self.assertRaises(ValueError):
                    LeggedLabPolicy(cfg.policy, dof_cfg=effective_dof)

    def test_launcher_is_native_hydra_without_argparse(self):
        module = ast.parse(LAUNCHER.read_text(), filename=LAUNCHER.as_posix())
        imported_modules = {alias.name for node in module.body if isinstance(node, ast.Import) for alias in node.names}
        imported_modules.update(
            node.module for node in module.body if isinstance(node, ast.ImportFrom) and node.module is not None
        )
        self.assertNotIn("argparse", imported_modules)
        pipeline_imports = {
            name
            for name in imported_modules
            if name == "g1_playground.pipeline" or name.startswith("g1_playground.pipeline.")
        }
        self.assertEqual(pipeline_imports, set())
        self.assertFalse(any(isinstance(node, ast.ClassDef) for node in module.body))
        function_names = {node.name for node in module.body if isinstance(node, ast.FunctionDef)}
        self.assertNotIn("parse_args", function_names)
        self.assertNotIn("create_components", function_names)
        self.assertNotIn("main", function_names)
        self.assertIn("g1_playground.utils.math", imported_modules)
        self.assertIn("read_frame", function_names)
        self.assertLessEqual(len(LAUNCHER.read_text().splitlines()), 200)

        launcher_identifiers = {node.id for node in ast.walk(module) if isinstance(node, ast.Name)}
        self.assertTrue({"is_real", "domain_id", "do_safety_check", "run_fullspeed"}.isdisjoint(launcher_identifiers))
        launcher_strings = {
            node.value for node in ast.walk(module) if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        self.assertNotIn("domain_id", launcher_strings)

        run = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "run")
        policy_calls = [
            node
            for node in ast.walk(run)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "LeggedLabPolicy"
        ]
        instantiate_calls = sorted(
            (
                node
                for node in ast.walk(run)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "instantiate"
            ),
            key=lambda node: node.lineno,
        )
        self.assertEqual(len(policy_calls), 1)
        self.assertEqual(len(instantiate_calls), 2)
        self.assertLess(policy_calls[0].lineno, instantiate_calls[0].lineno)
        self.assertEqual(ast.unparse(instantiate_calls[0].args[0]), "cfg.env")
        self.assertEqual(ast.unparse(instantiate_calls[1].args[0]), "cfg.controller")
        self.assertEqual({keyword.arg for keyword in instantiate_calls[0].keywords}, {"control_dt", "dof_cfg"})
        self.assertEqual({keyword.arg for keyword in instantiate_calls[1].keywords}, {"env"})

        hydra_decorators = [
            decorator
            for decorator in run.decorator_list
            if isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and isinstance(decorator.func.value, ast.Name)
            and decorator.func.value.id == "hydra"
            and decorator.func.attr == "main"
        ]
        self.assertEqual(len(hydra_decorators), 1)

        for deployment in ("sim", "real"):
            with self.subTest(deployment=deployment):
                result = subprocess.run(
                    [
                        sys.executable,
                        LAUNCHER,
                        "--cfg",
                        "job",
                        "--resolve",
                        f"deployment={deployment}",
                    ],
                    cwd=REPO_ROOT,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                task_cfg = OmegaConf.create(result.stdout)
                self.assertIn("domain_id", task_cfg.env)

    def test_jetson_bootstrap_does_not_import_an_inference_framework(self):
        def import_lines(module: ast.Module, package: str) -> list[int]:
            lines = []
            for node in ast.walk(module):
                if isinstance(node, ast.Import):
                    lines.extend(node.lineno for alias in node.names if alias.name.split(".", 1)[0] == package)
                elif isinstance(node, ast.ImportFrom) and node.module is not None:
                    if node.module.split(".", 1)[0] == package:
                        lines.append(node.lineno)
            return lines

        launcher = ast.parse(LAUNCHER.read_text(), filename=LAUNCHER.as_posix())
        bootstrap_lines = [
            node.lineno
            for node in launcher.body
            if isinstance(node, ast.Import) and any(alias.name == "g1_playground" for alias in node.names)
        ]
        self.assertEqual(len(bootstrap_lines), 1)
        bootstrap_line = bootstrap_lines[0]
        for package in ("hydra", "omegaconf"):
            with self.subTest(package=package):
                lines = import_lines(launcher, package)
                self.assertTrue(lines)
                self.assertLess(bootstrap_line, min(lines))

        package_init = REPO_ROOT / "g1_playground/__init__.py"
        init_module = ast.parse(package_init.read_text(), filename=package_init.as_posix())
        self.assertEqual(import_lines(init_module, "torch"), [])
        self.assertEqual(import_lines(init_module, "onnxruntime"), [])

    def test_cli_rejects_unknown_deployment_and_override(self):
        commands = (
            ["--cfg", "job", "deployment=missing"],
            ["--cfg", "job", "deployment=sim", "missing_option=true"],
        )
        for arguments in commands:
            with self.subTest(arguments=arguments):
                result = subprocess.run(
                    [sys.executable, LAUNCHER, *arguments],
                    cwd=REPO_ROOT,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "HYDRA_FULL_ERROR": "1"},
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertNotEqual(result.returncode, 0)

    def test_hydra_keeps_cwd_and_creates_no_output_files(self):
        raw_cfg = compose_config("real", "env.net_if=testnic", return_hydra_config=True)
        self.assertFalse(raw_cfg.hydra.job.chdir)
        self.assertIsNone(raw_cfg.hydra.output_subdir)
        self.assertEqual(raw_cfg.hydra.run.dir, ".")

        with tempfile.TemporaryDirectory(prefix="g1-playground-hydra-") as temporary_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    LAUNCHER,
                    "--cfg",
                    "job",
                    "deployment=real",
                    "env.net_if=testnic",
                ],
                cwd=temporary_dir,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("net_if: testnic", result.stdout)
            temporary_path = Path(temporary_dir)
            self.assertEqual(list(temporary_path.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
