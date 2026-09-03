import unittest

import numpy as np

from g1_playground.policy.body_hand import BodyHandPolicy
from tests.body_hand_helpers import body_joint_names, hand_joint_names, inspire_cfg, motion_cfg, policy_cfg
from tests.config_helpers import REPO_ROOT, compose_config
from tests.runner_helpers import body_hand_runner


class TestBodyHandPolicy(unittest.TestCase):
    def test_models_motions_observation_and_actions_form_one_runtime_contract(self):
        policy = BodyHandPolicy(
            policy_cfg(),
            motion_cfg(),
            runtime_body_joint_names=body_joint_names(),
            runtime_hand_joint_names=hand_joint_names(),
            hand_mimic=inspire_cfg().mimic,
            runner=body_hand_runner(9994),
        )

        self.assertEqual((policy.observation_dim, policy.action_dim, policy.freq), (9994, 41, 50))
        self.assertEqual(policy.motion.object_pos.shape, (262, 3))
        raw_action = np.linspace(-2.0, 2.0, 41, dtype=np.float32)
        processed = policy.process(raw_action)
        body, hand = policy.split(processed)
        self.assertEqual(body.shape, (29,))
        self.assertEqual(hand.shape, (12,))

        config = policy_cfg()
        action_names = list(config.action.body.joint_names) + list(config.action.hand.joint_names)
        for values, runtime_names in ((body, body_joint_names()), (hand, hand_joint_names())):
            for value, name in zip(values, runtime_names, strict=True):
                self.assertAlmostEqual(float(value), float(processed[action_names.index(name)]), places=6)

        for hoi in ("depth/largebox", "depth/smalltable"):
            with self.subTest(hoi=hoi):
                cfg = compose_config("sim", f"hoi={hoi}", config_name="run_loco_hoi_track")
                self.assertTrue((REPO_ROOT / str(cfg.hoi.policy_file)).is_file())
                self.assertTrue((REPO_ROOT / str(cfg.motion.file)).is_file())
                self.assertEqual(cfg.hoi.frequency, 50)


if __name__ == "__main__":
    unittest.main()
