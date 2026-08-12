import os
import re
import runpy
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

from pydantic import ValidationError

from robojudo.config.config_manager import ConfigManager
from robojudo.config.g1.env.g1_real_env_cfg import G1RealEnvCfg, G1UnitreeCfg
from robojudo.environment.env_cfgs import UnitreeCppEnvCfg

REPO_ROOT = Path(__file__).resolve().parents[1]
UNITREE_CPP_ROOT = REPO_ROOT / "third_party/unitree_cpp"


def strip_cpp_comments(source: str) -> str:
    return re.sub(r"//.*?$|/\*.*?\*/", "", source, flags=re.MULTILINE | re.DOTALL)


class TestDdsPhase1Domain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g1_env = ConfigManager("g1").get_cfg().env

    @classmethod
    def make_endpoint(cls, target, domain_id, net_if):
        return UnitreeCppEnvCfg(
            target=target,
            unitree=UnitreeCppEnvCfg.UnitreeCfg(domain_id=domain_id, net_if=net_if),
            xml=cls.g1_env.xml,
            dof=cls.g1_env.dof,
        )

    def test_domain_id_is_required_and_serialized(self):
        self.assertTrue(UnitreeCppEnvCfg.UnitreeCfg.model_fields["domain_id"].is_required())
        self.assertTrue(UnitreeCppEnvCfg.UnitreeCfg.model_fields["net_if"].is_required())
        self.assertEqual(G1UnitreeCfg.model_fields["domain_id"].default, 0)
        with self.assertRaises(ValidationError) as error:
            UnitreeCppEnvCfg.UnitreeCfg()
        locations = {tuple(item["loc"]) for item in error.exception.errors()}
        self.assertIn(("domain_id",), locations)
        self.assertIn(("net_if",), locations)

        for target, domain_id, net_if in (("simulation", 1, "lo"), ("hardware", 0, "testnic")):
            with self.subTest(target=target):
                endpoint = self.make_endpoint(target, domain_id, net_if)
                self.assertEqual(endpoint.target, target)
                self.assertEqual(endpoint.unitree.to_dict()["domain_id"], domain_id)
                self.assertEqual(endpoint.unitree.to_dict()["net_if"], net_if)

    def test_endpoint_matrix_rejects_crossed_or_malformed_values(self):
        invalid_endpoints = (
            ("simulation", 0, "lo"),
            ("simulation", 1, "testnic"),
            ("simulation", 0, "testnic"),
            ("hardware", 1, "testnic"),
            ("hardware", 0, "lo"),
            ("hardware", 1, "lo"),
            ("hardware", 2, "testnic"),
            ("hardware", -1, "testnic"),
            ("hardware", 0, ""),
            ("hardware", 0, " testnic"),
            ("hardware", 0, "testnic "),
            ("unknown", 0, "testnic"),
            ("simulation", True, "lo"),
            ("simulation", 1.0, "lo"),
            ("simulation", "1", "lo"),
        )
        for target, domain_id, net_if in invalid_endpoints:
            with self.subTest(target=target, domain_id=domain_id, net_if=net_if):
                with self.assertRaises(ValidationError):
                    self.make_endpoint(target, domain_id, net_if)

        with self.assertRaises(ValidationError) as error:
            UnitreeCppEnvCfg(
                unitree=UnitreeCppEnvCfg.UnitreeCfg(domain_id=1, net_if="lo"),
                xml=self.g1_env.xml,
                dof=self.g1_env.dof,
            )
        self.assertIn(("target",), {tuple(item["loc"]) for item in error.exception.errors()})

    def test_g1_real_explicitly_uses_domain_zero(self):
        env = ConfigManager("g1_real").get_cfg().env
        unitree = env.unitree
        self.assertEqual(env.target, "hardware")
        self.assertEqual(unitree.domain_id, 0)
        self.assertEqual(unitree.to_dict()["domain_id"], 0)
        self.assertNotIn("target", unitree.to_dict())

        invalid = env.model_dump()
        for domain_id in (1, False, 0.0, "0"):
            with self.subTest(domain_id=domain_id):
                invalid["unitree"]["domain_id"] = domain_id
                with self.assertRaises(ValidationError):
                    G1RealEnvCfg.model_validate(invalid)

        with self.assertRaises(ValidationError) as error:
            G1RealEnvCfg()
        self.assertIn(("unitree",), {tuple(item["loc"]) for item in error.exception.errors()})

    def test_g1_remains_available_without_unitree_cpp(self):
        script = textwrap.dedent(
            """
            import importlib.abc
            import sys

            import numpy as np

            class BlockUnitreeCpp(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path, target=None):
                    if fullname == "unitree_cpp" or fullname.startswith("unitree_cpp."):
                        raise ModuleNotFoundError("unitree_cpp intentionally blocked")
                    return None

            sys.meta_path.insert(0, BlockUnitreeCpp())

            from robojudo.config.config_manager import ConfigManager
            from robojudo.environment import env_registry
            from robojudo.policy.unitree_policy import UnitreeWoGaitPolicy

            sim = ConfigManager("g1").get_cfg()
            real = ConfigManager("g1_real").get_cfg()
            assert sim.env.env_type == "MujocoEnv"
            assert real.env.unitree.domain_id == 0
            assert "robojudo.environment.unitree_cpp_env" not in sys.modules
            assert env_registry.get("MujocoEnv").__name__ == "MujocoEnv"

            policy = UnitreeWoGaitPolicy(sim.policy, device="cpu")
            action = policy.get_action(np.zeros(480, dtype=np.float32))
            assert action.shape == (29,)
            assert "robojudo.environment.unitree_cpp_env" not in sys.modules

            try:
                env_registry.get("UnitreeCppEnv")
            except RuntimeError as error:
                assert isinstance(error.__cause__, ModuleNotFoundError)
            else:
                raise AssertionError("UnitreeCppEnv unexpectedly imported without its binding")
            """
        )
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_vendor_binding_passes_domain_to_channel_factory(self):
        header = strip_cpp_comments((UNITREE_CPP_ROOT / "src/unitree_controller.hpp").read_text())
        binding = strip_cpp_comments((UNITREE_CPP_ROOT / "src/py_binding.cpp").read_text())
        controller = strip_cpp_comments((UNITREE_CPP_ROOT / "src/unitree_controller.cpp").read_text())
        example = (UNITREE_CPP_ROOT / "example/config.py").read_text()

        struct_body = re.search(r"struct\s+UnitreeConfig\s*\{(?P<body>.*?)\};", header, flags=re.DOTALL)
        self.assertIsNotNone(struct_body)
        self.assertEqual(
            re.findall(r"std::int32_t\s+domain_id\s*=\s*0\s*;", struct_body["body"]), ["std::int32_t domain_id = 0;"]
        )

        self.assertEqual(binding.count('.def_readwrite("domain_id", &UnitreeConfig::domain_id)'), 1)
        dict_constructor = binding.split(".def(py::init([](py::dict cfg_dict) {", 1)[1].split(
            "return new UnitreeController(cfg);", 1
        )[0]
        self.assertEqual(dict_constructor.count('cfg.domain_id = cfg_dict["domain_id"].cast<std::int32_t>();'), 1)

        controller_constructor = controller.split("UnitreeController::UnitreeController", 1)[1].split(
            "UnitreeController::~UnitreeController", 1
        )[0]
        self.assertIn("InitChannelFactoryOnce(cfg_)", controller_constructor)
        self.assertLess(
            controller_constructor.index("InitChannelFactoryOnce(cfg_)"),
            controller_constructor.index("MotionSwitcherClient"),
        )
        all_init_calls = re.findall(r"ChannelFactory::Instance\(\)->Init\((.*?)\)\s*;", controller, re.DOTALL)
        self.assertEqual([re.sub(r"\s+", "", call) for call in all_init_calls], ["domain_id,net_if"])
        self.assertNotIn("ChannelFactory::Instance()->Release", controller)
        self.assertRegex(example, r"domain_id:\s*int\s*=\s*0")

        consumed_keys = re.findall(r'cfg_dict\["([a-z_]+)"\]', binding)
        self.assertEqual(consumed_keys.count("domain_id"), 1)

        example_models = runpy.run_path(UNITREE_CPP_ROOT / "example/config.py")
        self.assertEqual(example_models["RobotConfig"]().unitree.to_dict()["domain_id"], 0)


if __name__ == "__main__":
    unittest.main()
