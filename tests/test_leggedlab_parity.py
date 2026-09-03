import unittest
from types import SimpleNamespace

import numpy as np

from g1_playground.policy.leggedlab import LeggedLabPolicy
from g1_playground.utils.dof import compose_dof_config
from tests.config_helpers import compose_config
from tests.runner_helpers import leggedlab_runner


class TestLeggedLabPolicy(unittest.TestCase):
    def test_observation_and_action_match_the_deployment_joint_order(self):
        cfg = compose_config("sim")
        dof = compose_dof_config(cfg.robot.dof, cfg.policy.dof)
        policy = LeggedLabPolicy(cfg.policy, dof_cfg=dof, runner=leggedlab_runner())
        runtime_pos = np.asarray(dof.default_pos, dtype=np.float32) + np.linspace(-0.1, 0.1, 29)
        runtime_vel = np.linspace(-1.0, 1.0, 29, dtype=np.float32)
        last_action = np.linspace(-0.5, 0.5, 29, dtype=np.float32)
        policy.last_action = last_action.copy()
        control = {"axes": {"LeftX": -0.9, "LeftY": 0.9, "RightX": 0.9, "RightY": 0.0}}
        policy_state = SimpleNamespace(
            base_quat=np.array([0.0, 0.0, 0.0, 1.0]),
            base_ang_vel=np.array([0.1, 0.2, 0.3]),
            dof_pos=policy.observation_adapter.fit(runtime_pos),
            dof_vel=policy.observation_adapter.fit(runtime_vel),
        )

        observation = policy.get_observation(policy_state, control)
        expected = np.concatenate(
            [
                policy_state.base_ang_vel,
                [0.0, 0.0, -1.0],
                [0.7, 0.4, -0.9],
                policy_state.dof_pos - policy.default_pos,
                policy_state.dof_vel,
                last_action,
            ]
        )
        np.testing.assert_allclose(observation, expected, atol=1e-6)

        policy.reset()
        runtime_state = SimpleNamespace(
            base_quat=policy_state.base_quat,
            base_ang_vel=policy_state.base_ang_vel,
            dof_pos=runtime_pos,
            dof_vel=runtime_vel,
        )
        target = policy.act(runtime_state, control)
        expected_target = policy.action_adapter.fit(policy.default_pos + policy.last_action * policy.action_scale)
        np.testing.assert_allclose(target, expected_target, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
