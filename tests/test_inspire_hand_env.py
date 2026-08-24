import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf

import g1_playground.inspire.hand_env as hand_module
from tests.config_helpers import CONFIG_DIR


class FakeEndpoint:
    def __init__(self, lost_values):
        self.lost_values = [np.asarray(value, dtype=np.uint32) for value in lost_values]
        self.index = 0

    def self_check(self):
        return True

    def get_state(self):
        lost = self.lost_values[min(self.index, len(self.lost_values) - 1)]
        self.index += 1
        return SimpleNamespace(valid=True, q=np.ones(12), dq=np.zeros(12), lost=lost, age_seconds=0.001)

    def close(self):
        return True


class AdvancingEndpoint(FakeEndpoint):
    def __init__(self):
        super().__init__([np.zeros(12, dtype=np.uint32)])

    def get_state(self):
        self.index += 1
        return SimpleNamespace(
            valid=True,
            q=np.ones(12),
            dq=np.zeros(12),
            lost=np.full(12, self.index, dtype=np.uint32),
            age_seconds=0.001,
        )


class ZeroStateEndpoint(FakeEndpoint):
    def __init__(self):
        super().__init__([np.full(12, 100, dtype=np.uint32)])

    def get_state(self):
        self.index += 1
        return SimpleNamespace(
            valid=True,
            q=np.zeros(12),
            dq=np.zeros(12),
            lost=np.full(12, 100, dtype=np.uint32),
            age_seconds=0.001,
        )


def build_env(endpoint):
    inspire = OmegaConf.load(CONFIG_DIR / "robot/inspire.yaml")
    with patch.object(hand_module, "InspireDdsControlEndpoint", return_value=endpoint):
        return hand_module.InspireHandEnv(dof_cfg=inspire.dof)


class TestInspireLinkHealth(unittest.TestCase):
    def test_stable_historical_loss_counters_pass_self_check(self):
        env = build_env(FakeEndpoint([np.full(12, 42, dtype=np.uint32)]))
        with patch.object(hand_module.time, "sleep", return_value=None):
            env.self_check(timeout=0.1)

    def test_advancing_loss_counters_do_not_fail_dds_self_check(self):
        env = build_env(AdvancingEndpoint())
        with patch.object(hand_module.time, "sleep", return_value=None):
            env.self_check(timeout=0.01)

    def test_serial_loss_counters_are_exposed_for_diagnostics_only(self):
        env = build_env(FakeEndpoint([np.zeros(12, dtype=np.uint32), np.ones(12, dtype=np.uint32)]))
        env.read()
        second = env.read()
        np.testing.assert_array_equal(second.lost, 1)
        self.assertFalse(second.stale)

    def test_zeroed_serial_fallback_remains_a_fresh_policy_observation(self):
        env = build_env(ZeroStateEndpoint())
        state = env.read()
        np.testing.assert_allclose(state.joint_pos, env.upper)
        np.testing.assert_allclose(state.joint_vel, 0.0)
        self.assertFalse(state.stale)


if __name__ == "__main__":
    unittest.main()
