import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf

from g1_playground.utils.dof import DoFAdapter, compose_dof_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def runtime_names():
    return list(OmegaConf.load(CONFIG_DIR / "robot/g1.yaml").dof.joint_names)


def policy_names():
    return list(OmegaConf.load(CONFIG_DIR / "policy/leggedlab_g1.yaml").dof.joint_names)


class TestCompleteSetCallers(unittest.TestCase):
    def test_compose_dof_config_owns_the_complete_set_constraint(self):
        robot = OmegaConf.load(CONFIG_DIR / "robot/g1.yaml").dof
        policy = OmegaConf.load(CONFIG_DIR / "policy/leggedlab_g1.yaml").dof
        self.assertEqual(list(compose_dof_config(robot, policy).joint_names), runtime_names())

        trimmed = OmegaConf.create(
            {
                "joint_names": list(policy.joint_names)[:-1],
                "default_pos": list(policy.default_pos)[:-1],
                "stiffness": list(policy.stiffness)[:-1],
                "damping": list(policy.damping)[:-1],
            }
        )
        with self.assertRaises(ValueError) as caught:
            compose_dof_config(robot, trimmed)
        self.assertIn("complete joint set", str(caught.exception))

    def test_the_locomotion_pair_proves_set_equality_on_its_own(self):
        runtime = runtime_names()
        policy = policy_names()
        DoFAdapter(runtime, policy)
        DoFAdapter(policy, runtime)

        renamed = [*policy[:-1], "not_a_joint"]
        with self.assertRaises(ValueError):
            DoFAdapter(runtime, renamed)
        with self.assertRaises(ValueError):
            DoFAdapter(renamed, runtime)

    def test_a_strictly_smaller_policy_layout_is_caught_by_one_of_the_pair(self):
        runtime = runtime_names()
        short = runtime[:-1]
        DoFAdapter(runtime, short)
        with self.assertRaises(ValueError):
            DoFAdapter(short, runtime)


class TestSubsetMode(unittest.TestCase):
    def test_a_target_subset_gathers_in_target_order(self):
        adapter = DoFAdapter(src_joint_names=["a", "b", "c", "d"], tar_joint_names=["c", "a"])
        np.testing.assert_allclose(adapter.fit(np.array([10.0, 20.0, 30.0, 40.0])), [30.0, 10.0])

    def test_scatter_writes_back_to_the_source_layout(self):
        adapter = DoFAdapter(["a", "b", "c", "d"], ["c", "a"])
        out = np.zeros(4)
        adapter.scatter_into(np.array([7.0, 8.0]), out)
        np.testing.assert_allclose(out, [8.0, 0.0, 7.0, 0.0])

    def test_gather_and_scatter_round_trip(self):
        names = [f"j{i}" for i in range(10)]
        target = ["j7", "j2", "j9"]
        adapter = DoFAdapter(names, target)
        values = np.arange(10.0)
        out = np.zeros(10)
        adapter.scatter_into(adapter.fit(values), out)
        for name in target:
            self.assertEqual(out[names.index(name)], values[names.index(name)])

    def test_a_target_outside_the_source_is_rejected(self):
        with self.assertRaises(ValueError):
            DoFAdapter(["a", "b"], ["a", "z"])

    def test_a_repeated_source_is_always_rejected(self):
        with self.assertRaises(ValueError):
            DoFAdapter(["a", "a"], ["a"])

    def test_repeated_targets_gather_but_never_scatter(self):
        adapter = DoFAdapter(["a", "b", "c"], ["c", "c", "a"])
        np.testing.assert_allclose(adapter.fit(np.array([1.0, 2.0, 3.0])), [3.0, 3.0, 1.0])
        with self.assertRaises(ValueError) as caught:
            adapter.scatter_into(np.zeros(3), np.zeros(3))
        self.assertIn("repeated target", str(caught.exception))

    def test_wrong_sized_inputs_are_rejected(self):
        adapter = DoFAdapter(["a", "b", "c"], ["c", "a"])
        with self.assertRaises(ValueError):
            adapter.fit(np.zeros(2))
        with self.assertRaises(ValueError):
            adapter.scatter_into(np.zeros(3), np.zeros(3))
        with self.assertRaises(ValueError):
            adapter.scatter_into(np.zeros(2), np.zeros(4))


class TestBodyHandUsesTheSharedAdapter(unittest.TestCase):
    def test_every_body_hand_mapping_is_a_dof_adapter(self):
        from g1_playground.policy.body_hand import BodyHandPolicy
        from tests.body_hand_helpers import body_joint_names, inspire_cfg, motion_cfg, policy_cfg

        policy = BodyHandPolicy(
            policy_cfg(),
            motion_cfg(),
            device="cpu",
            runtime_body_joint_names=body_joint_names(),
            runtime_hand_joint_names=inspire_cfg().dof.joint_names,
            hand_mimic=inspire_cfg().mimic,
        )
        adapters = SimpleNamespace(
            body_to_runtime=policy.body_to_runtime,
            hand_to_runtime=policy.hand_to_runtime,
            state_to_body=policy.assembler.state_to_body,
            state_to_hand=policy.assembler.state_to_hand,
            state_to_mimic=policy.assembler.state_to_mimic,
            hand_to_drivers=policy.assembler.hand_to_drivers,
        )
        for name, adapter in vars(adapters).items():
            with self.subTest(adapter=name):
                self.assertIsInstance(adapter, DoFAdapter)


if __name__ == "__main__":
    unittest.main()
