import hashlib
import importlib
import json
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from g1_playground.policy.leggedlab import LeggedLabPolicy
from g1_playground.utils.dof import compose_dof_config
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
    roots = (REPO_ROOT / "assets/models/leggedlab", REPO_ROOT / "assets/robots/g1")
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
            # Freshly loaded checkpoint (initial saved LSTM state) pins the artifact itself.
            model = torch.jit.load(checkpoint, map_location="cpu")
            model.eval()
            with torch.inference_mode():
                output = model(input_tensor).cpu().numpy()

            self.assertEqual(output.shape, (1, policy["output_size"]))
            self.assertEqual(output.dtype, np.float32)
            self.assertTrue(np.isfinite(output).all())
            np.testing.assert_allclose(output[0], policy["golden_output"], rtol=1e-5, atol=1e-6)

            # The runtime policy is recurrent: its constructor washes the LSTM state via
            # RESET_WARMUP_STEPS zero-observation inferences. Replicate that exact call
            # sequence on the checkpoint to obtain the reference for get_action().
            runtime_policy = LeggedLabPolicy(self.g1.policy, device="cpu", dof_cfg=self.effective_dof)
            washed = torch.jit.load(checkpoint, map_location="cpu")
            washed.eval()
            with torch.inference_mode():
                for _ in range(LeggedLabPolicy.RESET_WARMUP_STEPS):
                    washed(torch.zeros((1, policy["input_size"]), dtype=torch.float32))
                washed_raw = washed(input_tensor.clip(-policy["clip_obs"], policy["clip_obs"])).cpu().numpy()
        finally:
            torch.set_num_threads(old_threads)

        processed_action = runtime_policy.get_action(input_tensor.numpy().squeeze())
        expected_raw = washed_raw[0]
        np.testing.assert_allclose(processed_action, expected_raw * policy["action_scale"], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(runtime_policy.last_action, expected_raw, rtol=1e-5, atol=1e-6)

    def test_single_frame_observation_layout(self):
        policy_cfg = self.g1.policy
        expected = self.expected["policy"]
        policy_snapshot = {
            "action_scale": policy_cfg.action_scale,
            "command_range": {
                "lin_vel_x": list(policy_cfg.command_range.lin_vel_x),
                "lin_vel_y": list(policy_cfg.command_range.lin_vel_y),
                "ang_vel_z": list(policy_cfg.command_range.ang_vel_z),
            },
            "clip_obs": policy_cfg.clip_obs,
            "action_clip": policy_cfg.clip_action,
        }
        self.assertEqual(policy_snapshot, {key: expected[key] for key in policy_snapshot})
        self.assertEqual(policy_cfg.obs_scales.ang_vel, expected["obs_scales"]["ang_vel"])
        self.assertEqual(policy_cfg.obs_scales.dof_pos, expected["obs_scales"]["dof_pos"])
        self.assertEqual(policy_cfg.obs_scales.dof_vel, expected["obs_scales"]["dof_vel"])
        self.assertEqual(expected["action_beta"], 1.0)
        self.assertEqual(expected["action_clip"], 100.0)
        self.assertEqual(expected["obs_scales"]["gravity"], 1.0)
        self.assertEqual(expected["obs_scales"]["command"], [1.0, 1.0, 1.0])
        self.assertNotIn("max_cmd", policy_cfg)
        self.assertNotIn("deadzone", policy_cfg)

        history_layout = LeggedLabPolicy.HISTORY_LAYOUT
        self.assertEqual(
            [[name, dim] for name, dim in history_layout],
            expected["history_fields"],
        )
        self.assertEqual(sum(dim for _, dim in history_layout), expected["single_sample_size"])
        self.assertEqual(
            LeggedLabPolicy.HISTORY_LENGTH * expected["single_sample_size"],
            expected["input_size"],
        )

        policy = LeggedLabPolicy(policy_cfg, device="cpu", dof_cfg=self.effective_dof)
        policy.history_buf.clear()

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

        axes = expected["golden_axes"]
        commands = np.asarray([axes["LeftY"], -axes["LeftX"], -axes["RightX"]], dtype=np.float32)
        clip_min = np.asarray(
            [
                expected["command_range"]["lin_vel_x"][0],
                expected["command_range"]["lin_vel_y"][0],
                expected["command_range"]["ang_vel_z"][0],
            ],
            dtype=np.float32,
        )
        clip_max = np.asarray(
            [
                expected["command_range"]["lin_vel_x"][1],
                expected["command_range"]["lin_vel_y"][1],
                expected["command_range"]["ang_vel_z"][1],
            ],
            dtype=np.float32,
        )
        scaled_commands = np.clip(commands, clip_min, clip_max)
        np.testing.assert_allclose(scaled_commands, np.asarray(expected["golden_scaled_commands"]), rtol=0.0, atol=0.0)

        current = [
            base_ang_vel * policy_cfg.obs_scales.ang_vel,
            np.array([0.0, 0.0, -1.0], dtype=np.float32),
            scaled_commands,
            dof_offset * policy_cfg.obs_scales.dof_pos,
            dof_vel * policy_cfg.obs_scales.dof_vel,
            last_action,
        ]
        expected_observation = np.concatenate(current)
        self.assertEqual(expected["packing"], "single-frame")
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
                control_dt=1.0 / LeggedLabPolicy.FREQ,
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
