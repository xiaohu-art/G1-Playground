import ast
import os
import re
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from tests.config_helpers import compose_config

REPO_ROOT = Path(__file__).resolve().parents[1]
UNITREE_CPP_ROOT = REPO_ROOT / "third_party/unitree_cpp"


def strip_cpp_comments(source: str) -> str:
    return re.sub(r"//.*?$|/\*.*?\*/", "", source, flags=re.MULTILINE | re.DOTALL)


class TestDdsPhase1Domain(unittest.TestCase):
    def test_domain_id_is_required_and_serialized(self):
        env = compose_config("real").env
        serialized = OmegaConf.to_container(env, resolve=True)
        self.assertIn("domain_id", serialized)
        self.assertIn("net_if", serialized)
        self.assertNotIn("unitree", serialized)
        self.assertIs(type(serialized["domain_id"]), int)
        self.assertEqual(serialized["domain_id"], 0)
        self.assertTrue(serialized["net_if"].strip())

    def test_sim_and_real_share_one_optional_g1_environment(self):
        script = textwrap.dedent(
            """
            import importlib.abc
            import sys

            import numpy as np

            class BlockOptionalDependencies(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path, target=None):
                    if fullname == "unitree_cpp" or fullname.startswith("unitree_cpp."):
                        raise ModuleNotFoundError("unitree_cpp intentionally blocked")
                    if fullname == "pydantic" or fullname.startswith("pydantic."):
                        raise ModuleNotFoundError("pydantic intentionally blocked")
                    return None

            sys.meta_path.insert(0, BlockOptionalDependencies())

            from pathlib import Path
            from hydra import compose, initialize_config_dir
            from g1_playground.policy.leggedlab import LeggedLabPolicy
            from g1_playground.utils.dof import compose_dof_config
            from tests.config_helpers import load_pipeline_launcher

            launcher = load_pipeline_launcher()

            config_dir = Path.cwd() / "configs"
            with initialize_config_dir(version_base=None, config_dir=config_dir.as_posix()):
                sim = compose(config_name="run_pipeline", overrides=["deployment=sim"])
            with initialize_config_dir(version_base=None, config_dir=config_dir.as_posix()):
                real = compose(config_name="run_pipeline", overrides=["deployment=real"])
            assert sim.env._target_ == "g1_playground.g1_env.G1Env"
            assert sim.controller._target_ == "g1_playground.controller.keyboard_ctrl.KeyboardCtrl"
            assert sim.env.domain_id == 1
            assert sim.env.net_if == "lo"
            assert sim.env.motion_switcher_required is False
            assert "kind" not in sim.env
            assert real.env.domain_id == 0
            assert real.env._target_ == sim.env._target_
            assert "unitree" not in real.env
            assert "g1_playground.g1_env" not in sys.modules
            assert not any(
                name == "g1_playground.environment" or name.startswith("g1_playground.environment.")
                for name in sys.modules
            )
            assert launcher.__name__ == "g1_playground_test_pipeline"

            effective_dof = compose_dof_config(sim.robot.dof, sim.policy.dof)
            policy = LeggedLabPolicy(sim.policy, device="cpu", dof_cfg=effective_dof)
            action = policy.get_action(np.zeros(96, dtype=np.float32))
            assert action.shape == (29,)

            assert "pydantic" not in sys.modules

            try:
                from g1_playground.g1_env import G1Env
            except ModuleNotFoundError:
                pass
            else:
                raise AssertionError("G1Env unexpectedly imported without its required unitree_cpp binding")
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
        header = strip_cpp_comments((UNITREE_CPP_ROOT / "src/g1_dds_control_endpoint.hpp").read_text())
        dds_utils = strip_cpp_comments((UNITREE_CPP_ROOT / "src/dds_utils.hpp").read_text())
        binding = strip_cpp_comments((UNITREE_CPP_ROOT / "src/py_binding.cpp").read_text())
        control_endpoint = strip_cpp_comments((UNITREE_CPP_ROOT / "src/g1_dds_control_endpoint.cpp").read_text())
        example = (UNITREE_CPP_ROOT / "example/config.py").read_text()

        struct_body = re.search(r"struct\s+G1DdsControlEndpointConfig\s*\{(?P<body>.*?)\};", header, flags=re.DOTALL)
        self.assertIsNotNone(struct_body)
        self.assertEqual(
            re.findall(r"std::int32_t\s+domain_id\s*=\s*0\s*;", struct_body["body"]), ["std::int32_t domain_id = 0;"]
        )

        self.assertEqual(binding.count('.def_readwrite("domain_id", &G1DdsControlEndpointConfig::domain_id)'), 1)
        self.assertEqual(binding.count('.def("activate_commands", &G1DdsControlEndpoint::activate_commands)'), 1)
        self.assertEqual(
            binding.count('.def_property_readonly("lifecycle_state", &G1DdsControlEndpoint::lifecycle_state)'), 1
        )
        control_endpoint_binding = binding.split("void bind_G1DdsControlEndpoint", 1)[1]
        dict_constructor = control_endpoint_binding.split(".def(py::init([](py::dict cfg_dict) {", 1)[1].split(
            "return new G1DdsControlEndpoint(cfg);", 1
        )[0]
        self.assertEqual(dict_constructor.count('cfg.domain_id = cfg_dict["domain_id"].cast<std::int32_t>();'), 1)

        endpoint_constructor = control_endpoint.split("G1DdsControlEndpoint::G1DdsControlEndpoint", 1)[1].split(
            "G1DdsControlEndpoint::~G1DdsControlEndpoint", 1
        )[0]
        self.assertIn("InitChannelFactoryOnce(cfg_)", endpoint_constructor)
        self.assertLess(
            endpoint_constructor.index("InitChannelFactoryOnce(cfg_)"),
            endpoint_constructor.index("InitializeObserver()"),
        )
        self.assertNotIn("MotionSwitcherClient", endpoint_constructor)
        self.assertNotIn("lowcmd_publisher_", endpoint_constructor)

        command_transport = control_endpoint.split("void G1DdsControlEndpoint::InitializeCommandTransport()", 1)[
            1
        ].split("bool G1DdsControlEndpoint::activate_commands()", 1)[0]
        self.assertIn("MotionSwitcherClient", command_transport)
        self.assertIn("lowcmd_publisher_", command_transport)
        activate_commands = control_endpoint.split("bool G1DdsControlEndpoint::activate_commands()", 1)[1].split(
            "G1DdsControlEndpoint::lifecycle_state", 1
        )[0]
        self.assertIn("InitializeCommandTransport()", activate_commands)
        all_init_calls = re.findall(r"ChannelFactory::Instance\(\)->Init\((.*?)\)\s*;", dds_utils, re.DOTALL)
        self.assertEqual([re.sub(r"\s+", "", call) for call in all_init_calls], ["selected_domain,selected_if"])
        self.assertIn("InitializeDdsEndpointOnce(cfg.domain_id, cfg.net_if)", control_endpoint)
        self.assertNotIn("ChannelFactory::Instance()->Release", control_endpoint)

        example_tree = ast.parse(example)
        unitree_class = next(
            node
            for node in example_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "G1DdsControlEndpointConfig"
        )
        domain_assignments = [
            node
            for node in unitree_class.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "domain_id"
        ]
        self.assertEqual(len(domain_assignments), 1)
        domain_assignment = domain_assignments[0]
        self.assertIsInstance(domain_assignment.annotation, ast.Name)
        self.assertEqual(domain_assignment.annotation.id, "int")
        self.assertIsInstance(domain_assignment.value, ast.Constant)
        self.assertIs(type(domain_assignment.value.value), int)
        self.assertEqual(domain_assignment.value.value, 0)

        consumed_keys = re.findall(r'cfg_dict\["([a-z_]+)"\]', dict_constructor)
        self.assertEqual(consumed_keys.count("domain_id"), 1)


if __name__ == "__main__":
    unittest.main()
