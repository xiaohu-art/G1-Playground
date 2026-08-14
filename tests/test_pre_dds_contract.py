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

from g1_playground.policy.unitree_policy import UnitreeWoGaitPolicy
from g1_playground.utils.dof import DoFAdapter, compose_dof_config
from tests.config_helpers import compose_config

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "tests/fixtures/pre_dds/contract.json"
PRE_DDS_ROBOT_NAME = "g1"
PRE_DDS_CONTROL_DT = 0.02


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
        cls.g1 = compose_config("sim")
        cls.g1_real = compose_config("real")
        cls.effective_dof = compose_dof_config(cls.g1.robot.dof, cls.g1.policy.dof)

    def test_composition_wire_and_joint_contract(self):
        expected = self.expected
        self.assertEqual(expected["schema_version"], 1)
        self.assertEqual(expected["composition"]["g1"]["env_type"], "MujocoEnv")
        self.assertIs(expected["composition"]["g1"]["is_sim"], True)
        self.assertEqual(expected["composition"]["g1_real"]["env_type"], "UnitreeCppEnv")
        self.assertIs(expected["composition"]["g1_real"]["is_sim"], False)
        self.assertEqual(
            expected["loop"],
            {
                "policy_hz": 50,
                "control_dt": 0.02,
                "sim_dt": 0.001,
                "sim_decimation": 20,
                "sim_duration_configured": 60.0,
                "visualize_extras": True,
            },
        )

        # The fixture remains immutable provenance. Current simulation deliberately
        # crosses the same Unitree DDS boundary as hardware instead of recreating it.
        self.assertEqual(self.g1.env._target_, "g1_playground.g1_env.G1Env")
        self.assertEqual((self.g1.env.domain_id, self.g1.env.net_if), (1, "lo"))
        self.assertIs(self.g1.env.motion_switcher_required, False)

        env = self.g1_real.env
        wire = {
            "robot": PRE_DDS_ROBOT_NAME,
            "msg_type": "hg",
            "control_mode": "position",
            "lowcmd_topic": env.lowcmd_topic,
            "lowstate_topic": env.lowstate_topic,
            "control_dt": PRE_DDS_CONTROL_DT,
            "joint2motor_idx": None,
            "sdk_motor_order_status": "identity_assumption",
            "domain_id_effective": 0,
            "domain_source": "cpp_hardcoded",
            "net_if": "<machine-specific-redacted>",
            "non_loopback_required": True,
        }
        self.assertTrue(env.net_if.strip())
        self.assertFalse(env.net_if.lower().startswith("lo"))
        self.assertNotIn("unitree", env)
        self.assertEqual(wire, expected["wire"])

        binding = expected["unitree_binding"]
        constructor_dict = {
            "domain_id": env.domain_id,
            "net_if": env.net_if,
            "robot": PRE_DDS_ROBOT_NAME,
            "msg_type": "hg",
            "control_mode": "position",
            "hand_type": "NONE",
            "lowcmd_topic": env.lowcmd_topic,
            "lowstate_topic": env.lowstate_topic,
            "enable_odometry": False,
            "sport_state_topic": "rt/odommodestate",
            "control_dt": PRE_DDS_CONTROL_DT,
            "num_dofs": len(self.g1_real.robot.dof.joint_names),
            "stiffness": self.effective_dof.stiffness,
            "damping": self.effective_dof.damping,
        }
        self.assertEqual(constructor_dict.pop("domain_id"), 0)
        constructor_dict["net_if"] = "<machine-specific-redacted>"
        expected_constructor = binding["constructor_dict"].copy()
        expected_constructor["stiffness"] = expected["effective_control_env_order"]["stiffness"]
        expected_constructor["damping"] = expected["effective_control_env_order"]["damping"]
        self.assertTrue(binding["act_enabled"])
        self.assertNotIn("act", self.g1_real.env)
        self.assertEqual(constructor_dict, expected_constructor)
        binding_source = (REPO_ROOT / "third_party/unitree_cpp/src/py_binding.cpp").read_text()
        binding_source = binding_source.split("void bind_G1DdsControlEndpoint", 1)[1].split("PYBIND11_MODULE", 1)[0]
        consumed_keys = list(dict.fromkeys(re.findall(r'cfg_dict\["([a-z_]+)"\]', binding_source)))
        self.assertIn("domain_id", consumed_keys)
        self.assertIn('cfg_dict.contains("motion_switcher_required")', binding_source)
        post_baseline_keys = {"domain_id", "motion_switcher_required"}
        self.assertEqual([key for key in consumed_keys if key not in post_baseline_keys], binding["cpp_consumed_keys"])

        env_names = self.g1.robot.dof.joint_names
        policy_names = self.g1.policy.dof.joint_names
        env_to_policy = DoFAdapter(env_names, policy_names)
        policy_to_env = DoFAdapter(policy_names, env_names)
        joints = expected["joints"]
        self.assertEqual(env_names, joints["environment_order"])
        self.assertEqual(policy_names, joints["policy_order"])

        sentinel = np.arange(29)
        historical_policy_order = np.empty(29, dtype=sentinel.dtype)
        historical_policy_order[joints["env_to_policy_target_indices"]] = sentinel
        historical_env_order = np.empty(29, dtype=sentinel.dtype)
        historical_env_order[joints["policy_to_env_target_indices"]] = sentinel
        np.testing.assert_array_equal(env_to_policy.fit(sentinel), historical_policy_order)
        np.testing.assert_array_equal(policy_to_env.fit(sentinel), historical_env_order)
        np.testing.assert_array_equal(
            policy_to_env.fit(env_to_policy.fit(sentinel)),
            sentinel,
        )

        expected_control = expected["effective_control_env_order"]
        self.assertEqual(set(self.effective_dof), {"joint_names", "default_pos", "stiffness", "damping"})
        for name in ("default_pos", "stiffness", "damping"):
            with self.subTest(control=name):
                np.testing.assert_allclose(
                    np.asarray(getattr(self.effective_dof, name)),
                    np.asarray(expected_control[name]),
                    rtol=0.0,
                    atol=1e-10,
                )
        np.testing.assert_allclose(
            np.asarray(self.g1.robot.dof.torque_limits),
            np.asarray(expected_control["torque_limits"]),
            rtol=0.0,
            atol=1e-10,
        )
        self.assertNotIn("position_limits", self.g1.robot.dof)
        self.assertIn("position_limits", expected_control)

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

        runtime_policy = UnitreeWoGaitPolicy(self.g1.policy, device="cpu", dof_cfg=self.effective_dof)
        processed_action = runtime_policy.get_action(input_tensor.numpy().squeeze())
        expected_raw = np.asarray(policy["golden_output"], dtype=np.float32)
        np.testing.assert_allclose(processed_action, expected_raw * policy["action_scale"], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(runtime_policy.last_action, expected_raw, rtol=1e-5, atol=1e-6)

    def test_field_major_observation_layout(self):
        policy_cfg = self.g1.policy
        expected = self.expected["policy"]
        policy_snapshot = {
            "action_scale": policy_cfg.action_scale,
            "max_cmd": policy_cfg.max_cmd,
        }
        self.assertEqual(policy_snapshot, {key: expected[key] for key in policy_snapshot})
        self.assertEqual(policy_cfg.obs_scales.ang_vel, expected["obs_scales"]["ang_vel"])
        self.assertEqual(policy_cfg.obs_scales.dof_vel, expected["obs_scales"]["dof_vel"])
        self.assertEqual(expected["action_beta"], 1.0)
        self.assertIsNone(expected["action_clip"])
        self.assertEqual(expected["obs_scales"]["gravity"], 1.0)
        self.assertEqual(expected["obs_scales"]["dof_pos"], 1.0)
        self.assertEqual(expected["obs_scales"]["command"], [1.0, 1.0, 1.0])

        history_layout = UnitreeWoGaitPolicy.HISTORY_LAYOUT
        self.assertEqual(
            [[name, dim] for name, dim in history_layout],
            expected["history_fields"],
        )
        self.assertEqual(sum(dim for _, dim in history_layout), expected["single_sample_size"])
        self.assertEqual(
            UnitreeWoGaitPolicy.HISTORY_LENGTH * expected["single_sample_size"],
            expected["input_size"],
        )

        policy = UnitreeWoGaitPolicy(policy_cfg, device="cpu", dof_cfg=self.effective_dof)
        policy.history_buf.clear()
        frames = []
        field_dims = [dim for _, dim in history_layout]
        for timestep in range(UnitreeWoGaitPolicy.HISTORY_LENGTH - 1):
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
        env_data = SimpleNamespace(
            base_quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            base_ang_vel=base_ang_vel,
            dof_pos=policy.default_pos + dof_offset,
            dof_vel=dof_vel,
        )
        ctrl_data = {"axes": expected["golden_axes"]}
        observation = policy.get_observation(env_data, ctrl_data)

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

    def test_runtime_constructs_g1_env_with_effective_gains(self):
        class FakeG1DdsControlEndpoint:
            instances = []

            def __init__(self, cfg):
                self.cfg = cfg
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
                raise AssertionError("runtime must provide final gains to the G1DdsControlEndpoint constructor")

            def step(self, target):
                self.events.append("step")

            def shutdown(self):
                self.events.append("shutdown")

        fake_binding = types.ModuleType("unitree_cpp")
        fake_binding.RobotState = SimpleNamespace
        fake_binding.G1DdsControlEndpoint = FakeG1DdsControlEndpoint
        module_name = "g1_playground.g1_env"
        missing = object()
        old_binding = sys.modules.get("unitree_cpp", missing)
        old_module = sys.modules.pop(module_name, None)
        env = None
        controller = None
        try:
            sys.modules["unitree_cpp"] = fake_binding
            environment_module = importlib.import_module(module_name)
            from g1_playground.controller.unitree_ctrl import UnitreeCtrl

            env = environment_module.G1Env(
                control_dt=1.0 / UnitreeWoGaitPolicy.FREQ,
                domain_id=self.g1_real.env.domain_id,
                net_if=self.g1_real.env.net_if,
                lowcmd_topic=self.g1_real.env.lowcmd_topic,
                lowstate_topic=self.g1_real.env.lowstate_topic,
                motion_switcher_required=self.g1_real.env.motion_switcher_required,
                dof_cfg=self.effective_dof,
            )
            controller = UnitreeCtrl(env)
            fake = FakeG1DdsControlEndpoint.instances[-1]
            self.assertNotIn("set_gains", fake.events)

            constructor_cfg = fake.cfg.copy()
            self.assertEqual(constructor_cfg.pop("domain_id"), 0)
            self.assertIs(constructor_cfg.pop("motion_switcher_required"), True)
            constructor_cfg["net_if"] = "<machine-specific-redacted>"
            for key in ("stiffness", "damping"):
                constructor_cfg[key] = np.asarray(constructor_cfg[key]).tolist()
            expected_constructor = self.expected["unitree_binding"]["constructor_dict"].copy()
            self.assertEqual(expected_constructor.pop("robot"), PRE_DDS_ROBOT_NAME)
            expected_constructor["stiffness"] = self.expected["effective_control_env_order"]["stiffness"]
            expected_constructor["damping"] = self.expected["effective_control_env_order"]["damping"]
            self.assertEqual(constructor_cfg, expected_constructor)
            self.assertEqual(type(env).__name__, "G1Env")
            self.assertEqual(type(controller).__name__, "UnitreeCtrl")
        finally:
            if env is not None:
                env.shutdown()
            sys.modules.pop(module_name, None)
            if old_module is not None:
                sys.modules[module_name] = old_module
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
        limits = np.asarray(self.g1.robot.dof.torque_limits)
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
