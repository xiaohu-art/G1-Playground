import hashlib
import importlib
import json
import re
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from box import Box

from robojudo.config.config_manager import ConfigManager
from robojudo.policy.unitree_policy import UnitreeWoGaitPolicy
from robojudo.tools.dof import DoFAdapter, merge_dof_cfgs

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "tests/fixtures/pre_dds/contract.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def asset_closure() -> dict[str, int | str]:
    roots = (REPO_ROOT / "assets/models/g1", REPO_ROOT / "assets/robots/g1")
    files = sorted(path for root in roots for path in root.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    total_bytes = 0
    for path in files:
        contents = path.read_bytes()
        total_bytes += len(contents)
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(contents)
        digest.update(b"\0")
    return {"file_count": len(files), "total_bytes": total_bytes, "sha256": digest.hexdigest()}


def vendor_closure(relative_root: str) -> dict[str, int | str]:
    root = REPO_ROOT / relative_root
    excluded_dirs = {".git", "__pycache__", "build", "dist"}
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix != ".pyc"
        and not any(part in excluded_dirs or part.endswith(".egg-info") for part in path.parts)
    )
    digest = hashlib.sha256()
    total_bytes = 0
    for path in files:
        contents = path.read_bytes()
        total_bytes += len(contents)
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(contents)
        digest.update(b"\0")
    return {"file_count": len(files), "total_bytes": total_bytes, "sha256": digest.hexdigest()}


class TestPreDdsContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expected = json.loads(FIXTURE_PATH.read_text())
        cls.g1 = ConfigManager("g1").get_cfg()
        cls.g1_real = ConfigManager("g1_real").get_cfg()
        cls.effective_dof = merge_dof_cfgs(cls.g1.env.dof, cls.g1.policy.action_dof)

    @staticmethod
    def composition(cfg) -> dict:
        return {
            "pipeline_type": cfg.pipeline_type,
            "env_type": cfg.env.env_type,
            "is_sim": cfg.env.is_sim,
            "controllers": [{"type": controller.ctrl_type, "triggers": controller.triggers} for controller in cfg.ctrl],
            "policy_type": cfg.policy.policy_type,
            "device": cfg.device,
            "run_fullspeed": cfg.run_fullspeed,
            "do_safety_check": cfg.do_safety_check,
        }

    def test_composition_wire_and_joint_contract(self):
        expected = self.expected
        self.assertEqual(expected["schema_version"], 1)
        self.assertEqual(self.composition(self.g1), expected["composition"]["g1"])
        self.assertEqual(self.composition(self.g1_real), expected["composition"]["g1_real"])

        loop = {
            "policy_hz": self.g1.policy.freq,
            "control_dt": 1.0 / self.g1.policy.freq,
            "sim_dt": self.g1.env.sim_dt,
            "sim_decimation": self.g1.env.sim_decimation,
            "sim_duration_configured": self.g1.env.sim_duration,
            "visualize_extras": self.g1.env.visualize_extras,
        }
        self.assertEqual(loop, expected["loop"])

        unitree = self.g1_real.env.unitree
        wire = {
            "robot": unitree.robot,
            "msg_type": unitree.msg_type,
            "control_mode": unitree.control_mode,
            "lowcmd_topic": unitree.lowcmd_topic,
            "lowstate_topic": unitree.lowstate_topic,
            "control_dt": unitree.control_dt,
            "joint2motor_idx": self.g1_real.env.joint2motor_idx,
            "sdk_motor_order_status": "identity_assumption",
            "domain_id_effective": 0,
            "domain_source": "cpp_hardcoded",
            "net_if": "<machine-specific-redacted>",
            "non_loopback_required": True,
        }
        self.assertTrue(unitree.net_if.strip())
        self.assertFalse(unitree.net_if.lower().startswith("lo"))
        self.assertEqual(wire, expected["wire"])

        binding = expected["unitree_binding"]
        constructor_dict = unitree.to_dict()
        self.assertEqual(constructor_dict.pop("domain_id"), 0)
        constructor_dict["net_if"] = "<machine-specific-redacted>"
        constructor_dict["num_dofs"] = self.g1_real.env.dof.num_dofs
        constructor_dict["stiffness"] = self.g1_real.env.dof.stiffness
        constructor_dict["damping"] = self.g1_real.env.dof.damping
        self.assertTrue(self.g1_real.env.act)
        self.assertEqual(binding["act_enabled"], self.g1_real.env.act)
        self.assertEqual(constructor_dict, binding["constructor_dict"])
        binding_source = (REPO_ROOT / "third_party/unitree_cpp/src/py_binding.cpp").read_text()
        consumed_keys = list(dict.fromkeys(re.findall(r'cfg_dict\["([a-z_]+)"\]', binding_source)))
        self.assertIn("domain_id", consumed_keys)
        self.assertEqual([key for key in consumed_keys if key != "domain_id"], binding["cpp_consumed_keys"])
        self.assertEqual(binding["post_construct_gain_source"], "effective_control_env_order")

        env_names = self.g1.env.dof.joint_names
        policy_names = self.g1.policy.action_dof.joint_names
        env_to_policy = DoFAdapter(env_names, policy_names)
        policy_to_env = DoFAdapter(policy_names, env_names)
        joints = expected["joints"]
        self.assertEqual(env_names, joints["environment_order"])
        self.assertEqual(policy_names, joints["policy_order"])
        self.assertEqual(env_to_policy.tar_indices, joints["env_to_policy_target_indices"])
        self.assertEqual(policy_to_env.tar_indices, joints["policy_to_env_target_indices"])

        sentinel = np.arange(29)
        np.testing.assert_array_equal(
            policy_to_env.fit(env_to_policy.fit(sentinel)),
            sentinel,
        )

        expected_control = expected["effective_control_env_order"]
        for name in ("default_pos", "stiffness", "damping", "torque_limits", "position_limits"):
            with self.subTest(control=name):
                np.testing.assert_allclose(
                    np.asarray(getattr(self.effective_dof, name)),
                    np.asarray(expected_control[name]),
                    rtol=0.0,
                    atol=1e-10,
                )

    def test_assets_model_and_policy_golden(self):
        assets = self.expected["assets"]
        for key in ("checkpoint", "xml"):
            with self.subTest(asset=key):
                expected = assets[key]
                path = REPO_ROOT / expected["path"]
                self.assertEqual(path.stat().st_size, expected["bytes"])
                self.assertEqual(sha256_file(path), expected["sha256"])
        self.assertEqual(asset_closure(), assets["closure"])
        viewer = self.expected["vendors"]["mujoco_viewer"]
        self.assertTrue((REPO_ROOT / "third_party/mujoco_viewer/LICENSE").is_file())
        self.assertEqual(vendor_closure("third_party/mujoco_viewer"), viewer["closure"])

        checkpoint = REPO_ROOT / assets["checkpoint"]["path"]
        policy = self.expected["policy"]
        golden_input = policy["golden_input"]
        input_tensor = torch.linspace(
            golden_input["start"],
            golden_input["end"],
            golden_input["steps"],
            dtype=torch.float32,
        ).reshape(1, -1)
        old_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            model = torch.jit.load(checkpoint, map_location="cpu")
            model.eval()
            with torch.inference_mode():
                output = model(input_tensor).cpu().numpy()
        finally:
            torch.set_num_threads(old_threads)

        self.assertEqual(output.shape, (1, policy["output_size"]))
        self.assertEqual(output.dtype, np.float32)
        self.assertTrue(np.isfinite(output).all())
        np.testing.assert_allclose(output[0], policy["golden_output"], rtol=1e-5, atol=1e-6)

        runtime_policy = UnitreeWoGaitPolicy(self.g1.policy, device="cpu")
        processed_action = runtime_policy.get_action(input_tensor.numpy().squeeze())
        expected_raw = np.asarray(policy["golden_output"], dtype=np.float32)
        np.testing.assert_allclose(processed_action, expected_raw * policy["action_scale"], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(runtime_policy.last_action, expected_raw, rtol=1e-5, atol=1e-6)

    def test_field_major_observation_layout(self):
        policy_cfg = self.g1.policy
        expected = self.expected["policy"]
        policy_snapshot = {
            "action_scale": policy_cfg.action_scale,
            "action_beta": policy_cfg.action_beta,
            "action_clip": policy_cfg.action_clip,
            "obs_scales": policy_cfg.obs_scales.model_dump(),
            "max_cmd": policy_cfg.max_cmd,
            "commands_map": policy_cfg.commands_map,
        }
        self.assertEqual(policy_snapshot, {key: expected[key] for key in policy_snapshot})
        self.assertEqual(
            [[name, dim] for name, dim in policy_cfg.history_obs_dims.items()],
            expected["history_fields"],
        )
        self.assertEqual(sum(policy_cfg.history_obs_dims.values()), expected["single_sample_size"])
        self.assertEqual(policy_cfg.history_length * expected["single_sample_size"], expected["input_size"])

        policy = UnitreeWoGaitPolicy(policy_cfg, device="cpu")
        policy.history_buf.clear()
        frames = []
        field_dims = list(policy_cfg.history_obs_dims.values())
        for timestep in range(policy_cfg.history_length - 1):
            frame = [
                np.arange(dim, dtype=np.float32) + 1000 * timestep + 100 * field_index
                for field_index, dim in enumerate(field_dims)
            ]
            frames.append(frame)
            policy.history_buf.append(frame)

        base_ang_vel = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        dof_offset = np.arange(29, dtype=np.float32) * 0.01
        dof_vel = np.arange(29, dtype=np.float32)
        last_action = np.arange(29, dtype=np.float32) + 500.0
        policy.last_action = last_action.copy()
        env_data = Box(
            {
                "base_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                "base_ang_vel": base_ang_vel,
                "dof_pos": policy.obs_default_pos + dof_offset,
                "dof_vel": dof_vel,
            }
        )
        ctrl_data = Box({"JoystickCtrl": {"axes": expected["golden_axes"]}})
        observation, extras = policy.get_observation(env_data, ctrl_data)

        current = [
            base_ang_vel * policy_cfg.obs_scales.ang_vel,
            np.array([0.0, 0.0, -1.0], dtype=np.float32),
            np.asarray(expected["golden_scaled_commands"]),
            dof_offset,
            dof_vel * policy_cfg.obs_scales.dof_vel,
            last_action,
        ]
        frames.append(current)
        expected_observation = np.concatenate(
            [np.concatenate([frame[field_index] for frame in frames]) for field_index in range(len(field_dims))]
        )
        self.assertEqual(expected["packing"], "field-major")
        self.assertEqual(observation.shape, (expected["input_size"],))
        np.testing.assert_allclose(observation, expected_observation, rtol=0.0, atol=1e-7)
        np.testing.assert_allclose(extras["commands"], expected["golden_commands"], rtol=0.0, atol=1e-7)

    def test_pipeline_applies_effective_unitree_gains(self):
        import robojudo.environment as environment_package

        class FakeUnitreeController:
            instances = []

            def __init__(self, cfg):
                self.cfg = cfg
                self.gain_calls = []
                self.events = ["construct"]
                self.state = SimpleNamespace(
                    motor_state=SimpleNamespace(q=[0.0] * 29, dq=[0.0] * 29),
                    imu_state=SimpleNamespace(quaternion=[1.0, 0.0, 0.0, 0.0], gyroscope=[0.0] * 3),
                    wireless_remote=bytes(40),
                )
                self.instances.append(self)

            def self_check(self):
                self.events.append("self_check")
                return True

            def get_robot_state(self):
                self.events.append("get_robot_state")
                return self.state

            def set_gains(self, stiffness, damping):
                self.events.append("set_gains")
                self.gain_calls.append((np.asarray(stiffness).copy(), np.asarray(damping).copy()))

            def step(self, target):
                self.events.append("step")

            def shutdown(self):
                self.events.append("shutdown")

        fake_binding = types.ModuleType("unitree_cpp")
        fake_binding.RobotState = SimpleNamespace
        fake_binding.UnitreeController = FakeUnitreeController
        module_name = "robojudo.environment.unitree_cpp_env"
        missing = object()
        old_binding = sys.modules.get("unitree_cpp", missing)
        old_module = sys.modules.pop(module_name, None)
        old_class = environment_package.env_registry.registered_modules.pop("UnitreeCppEnv", None)
        old_attribute = vars(environment_package).pop("UnitreeCppEnv", missing)
        pipeline = None
        controller = None
        try:
            sys.modules["unitree_cpp"] = fake_binding
            importlib.import_module(module_name)
            from robojudo.pipeline.rl_pipeline import RlPipeline

            invalid_env_cfg = self.g1_real.env.model_copy(deep=True)
            invalid_env_cfg.unitree.domain_id = 1
            with self.assertRaisesRegex(ValueError, "hardware DDS endpoint"):
                environment_package.UnitreeCppEnv(invalid_env_cfg)
            self.assertEqual(FakeUnitreeController.instances, [])

            pipeline = RlPipeline(self.g1_real)
            fake = FakeUnitreeController.instances[-1]
            self.assertEqual(len(fake.gain_calls), 1)
            stiffness, damping = fake.gain_calls[0]
            np.testing.assert_allclose(stiffness, self.effective_dof.stiffness, rtol=0.0, atol=1e-10)
            np.testing.assert_allclose(damping, self.effective_dof.damping, rtol=0.0, atol=1e-10)
            self.assertLess(fake.events.index("construct"), fake.events.index("set_gains"))

            constructor_cfg = fake.cfg.copy()
            self.assertEqual(constructor_cfg.pop("domain_id"), 0)
            constructor_cfg["net_if"] = "<machine-specific-redacted>"
            for key in ("stiffness", "damping"):
                constructor_cfg[key] = np.asarray(constructor_cfg[key]).tolist()
            self.assertEqual(constructor_cfg, self.expected["unitree_binding"]["constructor_dict"])
            controller = pipeline.ctrl_manager.controllers["UnitreeCtrl"].inst
        finally:
            if pipeline is not None:
                pipeline.shutdown()
            if controller is not None:
                for queue in (controller.state_queue, controller.event_queue):
                    queue.close()
                    queue.join_thread()
            sys.modules.pop(module_name, None)
            if old_module is not None:
                sys.modules[module_name] = old_module
            environment_package.env_registry.registered_modules.pop("UnitreeCppEnv", None)
            if old_class is not None:
                environment_package.env_registry.registered_modules["UnitreeCppEnv"] = old_class
            vars(environment_package).pop("UnitreeCppEnv", None)
            if old_attribute is not missing:
                environment_package.UnitreeCppEnv = old_attribute
            if old_binding is missing:
                sys.modules.pop("unitree_cpp", None)
            else:
                sys.modules["unitree_cpp"] = old_binding

    def test_mujoco_pd_and_single_step_golden(self):
        expected_model = self.expected["mujoco"]
        xml_path = REPO_ROOT / self.expected["assets"]["xml"]["path"]
        model = mujoco.MjModel.from_xml_path(xml_path.as_posix())
        self.assertEqual(model.opt.timestep, expected_model["xml_timestep"])
        for name in ("nq", "nv", "nu", "nbody", "njnt", "nsensor", "nsensordata", "nkey"):
            with self.subTest(model_field=name):
                self.assertEqual(getattr(model, name), expected_model[name])

        actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) for index in range(model.nu)]
        self.assertEqual(actuator_names, self.expected["joints"]["xml_actuator_order"])
        sensors = [
            {
                "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, index),
                "dim": int(model.sensor_dim[index]),
                "adr": int(model.sensor_adr[index]),
            }
            for index in range(model.nsensor)
        ]
        self.assertEqual(sensors, expected_model["sensors"])

        model.opt.timestep = expected_model["runtime_timestep"]
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        golden = self.expected["pd_golden"]
        default_pos = np.asarray(self.effective_dof.default_pos, dtype=np.float64)
        q_spec = golden["q_offset_linspace"]
        dq_spec = golden["dq_linspace"]
        target_spec = golden["target_offset_linspace"]
        q = default_pos + np.linspace(q_spec[0], q_spec[1], q_spec[2])
        dq = np.linspace(dq_spec[0], dq_spec[1], dq_spec[2])
        target = default_pos + np.linspace(target_spec[0], target_spec[1], target_spec[2])
        data.qpos[-29:] = q
        data.qvel[-29:] = dq
        mujoco.mj_forward(model, data)

        stiffness = np.asarray(self.effective_dof.stiffness)
        damping = np.asarray(self.effective_dof.damping)
        limits = np.asarray(self.effective_dof.torque_limits)
        raw_torque = (target - q) * stiffness - dq * damping
        clipped_torque = np.clip(raw_torque, -limits, limits)
        np.testing.assert_allclose(raw_torque, golden["raw_torque"], rtol=1e-7, atol=1e-8)
        np.testing.assert_allclose(clipped_torque, golden["clipped_torque"], rtol=1e-7, atol=1e-8)
        np.testing.assert_array_equal(np.abs(raw_torque) > limits, golden["saturation_mask"])

        data.ctrl[:] = clipped_torque
        mujoco.mj_step(model, data)
        np.testing.assert_allclose(data.qpos[-29:], golden["next_q"], rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(data.qvel[-29:], golden["next_dq"], rtol=1e-6, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
