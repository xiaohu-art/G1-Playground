"""Parity checks between the integrated LeggedLab locomotion policy and the
reference deployment code it was transcribed from.

Reference values below are copied from LeggedLabDeploy@93736b4
(https://github.com/Hellod035/LeggedLabDeploy, BSD-3-Clause): ``configs/g1.yaml``
and the observation math of ``deploy.py::run()``. They are hardcoded here on
purpose so the checks stay meaningful even if the vendored clone is removed.
"""

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf

from g1_playground.policy.leggedlab import LeggedLabPolicy
from g1_playground.utils.dof import compose_dof_config
from tests.config_helpers import compose_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

# LeggedLabDeploy configs/g1.yaml: policy joint j commands motor joint2motor_idx[j].
JOINT2MOTOR_IDX = [
    0,
    6,
    12,
    1,
    7,
    13,
    2,
    8,
    14,
    3,
    9,
    15,
    22,
    4,
    10,
    16,
    23,
    5,
    11,
    17,
    24,
    18,
    25,
    19,
    26,
    20,
    27,
    21,
    28,
]
REFERENCE_KPS = [
    200,
    200,
    200,
    150,
    150,
    200,
    150,
    150,
    200,
    200,
    200,
    100,
    100,
    20,
    20,
    100,
    100,
    20,
    20,
    50,
    50,
    50,
    50,
    40,
    40,
    40,
    40,
    40,
    40,
]
REFERENCE_KDS = [
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
]
REFERENCE_DEFAULT_POS = [
    -0.2,
    -0.2,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0.42,
    0.42,
    0.35,
    0.35,
    -0.23,
    -0.23,
    0.18,
    -0.18,
    0,
    0,
    0,
    0,
    0.87,
    0.87,
    0,
    0,
    0,
    0,
    0,
    0,
]
REFERENCE_COMMAND_RANGE = {
    "lin_vel_x": [-0.4, 0.7],
    "lin_vel_y": [-0.4, 0.4],
    "ang_vel_z": [-1.57, 1.57],
}
REFERENCE_ACTION_SCALE = 0.25
REFERENCE_CLIP = 100.0


def gravity_from_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    """Reference projected-gravity math of LeggedLabDeploy common/rotation_helper.py (wxyz input)."""
    qw, qx, qy, qz = quat_wxyz
    rotation = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )
    return rotation.T @ np.array([0.0, 0.0, -1.0])


def build_policy():
    cfg = compose_config("sim")
    effective_dof = compose_dof_config(cfg.robot.dof, cfg.policy.dof)
    return cfg, effective_dof, LeggedLabPolicy(cfg.policy, device="cpu", dof_cfg=effective_dof)


class TestLeggedLabParity(unittest.TestCase):
    def test_index_based_motor_mapping_matches_name_based_adapter(self):
        runtime = list(OmegaConf.load(CONFIG_DIR / "robot/g1.yaml").dof.joint_names)
        policy = list(OmegaConf.load(CONFIG_DIR / "policy/leggedlab_g1.yaml").dof.joint_names)
        self.assertEqual(len(JOINT2MOTOR_IDX), len(policy))
        for policy_index, motor_index in enumerate(JOINT2MOTOR_IDX):
            with self.subTest(policy_index=policy_index):
                self.assertEqual(policy[policy_index], runtime[motor_index])

    def test_gains_pose_and_scales_match_reference_config(self):
        dof = OmegaConf.load(CONFIG_DIR / "policy/leggedlab_g1.yaml")
        self.assertEqual(list(dof.dof.stiffness), REFERENCE_KPS)
        self.assertEqual(list(dof.dof.damping), REFERENCE_KDS)
        self.assertEqual(list(dof.dof.default_pos), REFERENCE_DEFAULT_POS)
        self.assertEqual(dof.action_scale, REFERENCE_ACTION_SCALE)
        self.assertEqual(float(dof.obs_scales.ang_vel), 1.0)
        self.assertEqual(float(dof.obs_scales.dof_pos), 1.0)
        self.assertEqual(float(dof.obs_scales.dof_vel), 1.0)
        self.assertEqual(
            OmegaConf.to_container(dof.command_range),
            REFERENCE_COMMAND_RANGE,
        )
        self.assertEqual(float(dof.clip_obs), REFERENCE_CLIP)
        self.assertEqual(float(dof.clip_action), REFERENCE_CLIP)

    def test_observation_matches_deploy_run_math(self):
        cfg, effective_dof, policy = build_policy()
        rng = np.random.default_rng(7)

        # A tilted base orientation, expressed in both conventions (normalized).
        quat_wxyz = np.array([0.9806, 0.0981, 0.0981, 0.1471], dtype=np.float64)
        quat_wxyz /= np.linalg.norm(quat_wxyz)
        quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
        base_ang_vel = rng.normal(size=3).astype(np.float32)
        motor_dof_pos = np.asarray(effective_dof.default_pos, dtype=np.float32) + rng.normal(
            scale=0.05, size=29
        ).astype(np.float32)
        motor_dof_vel = rng.normal(scale=0.5, size=29).astype(np.float32)
        last_action = rng.normal(size=29).astype(np.float32)
        policy.last_action = last_action.copy()
        policy.history_buf.clear()

        env_data = SimpleNamespace(
            base_quat=quat_xyzw,
            base_ang_vel=base_ang_vel,
            # act() remaps runtime order into policy order; do the same via the adapter.
            dof_pos=policy.observation_adapter.fit(motor_dof_pos),
            dof_vel=policy.observation_adapter.fit(motor_dof_vel),
        )
        axes = {"LeftX": -0.9, "LeftY": 0.9, "RightX": 0.9, "RightY": 0.0}
        observation = policy.get_observation(env_data, {"axes": axes})

        # deploy.py::run() reference computation, policy joint order.
        joint_pos = (policy.observation_adapter.fit(motor_dof_pos) - policy.default_pos) * 1.0
        joint_vel = policy.observation_adapter.fit(motor_dof_vel) * 1.0
        command = np.asarray([axes["LeftY"], -axes["LeftX"], -axes["RightX"]], dtype=np.float32)
        command = np.clip(
            command,
            np.asarray([-0.4, -0.4, -1.57], dtype=np.float32),
            np.asarray([0.7, 0.4, 1.57], dtype=np.float32),
        )
        gravity = gravity_from_quat_wxyz(quat_wxyz).astype(np.float32)
        expected = np.concatenate([base_ang_vel * 1.0, gravity, command, joint_pos, joint_vel, last_action])

        self.assertEqual(observation.shape, (96,))
        np.testing.assert_allclose(observation, expected, rtol=1e-6, atol=1e-6)

    def test_commands_have_no_deadzone_and_clip_asymmetrically(self):
        cfg, effective_dof, policy = build_policy()
        env_data = SimpleNamespace(
            base_quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            base_ang_vel=np.zeros(3, dtype=np.float32),
            dof_pos=np.asarray(effective_dof.default_pos, dtype=np.float32),
            dof_vel=np.zeros(29, dtype=np.float32),
        )

        def observe(axes):
            policy.history_buf.clear()
            return policy.get_observation(env_data, {"axes": axes})

        # Below the former 0.04 deadzone: values pass through untouched.
        small = observe({"LeftX": 0.0, "LeftY": 0.03, "RightX": -0.03, "RightY": 0.0})[6:9]
        np.testing.assert_allclose(small, [0.03, 0.0, 0.03], rtol=0.0, atol=1e-7)

        # Beyond the asymmetric lin_vel_x range: clipped to [-0.4, 0.7].
        clipped = observe({"LeftX": 0.0, "LeftY": -1.0, "RightX": 0.0, "RightY": 0.0})[6:9]
        np.testing.assert_allclose(clipped, [-0.4, 0.0, 0.0], rtol=0.0, atol=1e-7)
        clipped = observe({"LeftX": 0.0, "LeftY": 1.0, "RightX": 0.0, "RightY": 0.0})[6:9]
        np.testing.assert_allclose(clipped, [0.7, 0.0, 0.0], rtol=0.0, atol=1e-7)

    def test_reset_wash_makes_inference_deterministic(self):
        cfg, effective_dof, first = build_policy()
        obs = np.linspace(-1.0, 1.0, 96, dtype=np.float32)
        first_action = first.get_action(obs.copy())
        second_action = first.get_action(obs.copy())
        first.reset()
        after_reset = first.get_action(obs.copy())

        # Same session: the recurrent state evolves, so consecutive calls differ...
        self.assertFalse(np.allclose(first_action, second_action))
        # ...but reset() washes the LSTM state back, reproducing the first answer.
        np.testing.assert_allclose(after_reset, first_action, rtol=1e-6, atol=1e-6)

        # A fresh instance follows the identical washed trajectory.
        _, _, second_policy = build_policy()
        np.testing.assert_allclose(second_policy.get_action(obs.copy()), first_action, rtol=1e-6, atol=1e-6)

    def test_action_clipping_applies_around_the_model(self):
        cfg, effective_dof, policy = build_policy()
        obs = np.full(96, 5.0, dtype=np.float32)
        scaled = policy.get_action(obs)
        self.assertTrue(np.all(np.abs(policy.last_action) <= REFERENCE_CLIP + 1e-6))
        np.testing.assert_allclose(scaled, policy.last_action * REFERENCE_ACTION_SCALE, rtol=1e-6, atol=1e-7)

        # Observations beyond clip_obs are clamped before inference.
        huge = np.full(96, 500.0, dtype=np.float32)
        clamped = np.clip(huge, -REFERENCE_CLIP, REFERENCE_CLIP)
        policy.reset()
        action_from_huge = policy.get_action(huge.copy())
        policy.reset()
        action_from_clamped = policy.get_action(clamped.copy())
        np.testing.assert_allclose(action_from_huge, action_from_clamped, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
