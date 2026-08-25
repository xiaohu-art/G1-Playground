import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

import mujoco
import numpy as np
from omegaconf import OmegaConf

from g1_playground.inspire import dof as inspire_dof
from tests.config_helpers import compose_config

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts/replay.py"


def load_launcher() -> ModuleType:
    spec = importlib.util.spec_from_file_location("g1_playground_test_replay", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ArrayState(dict):
    @property
    def files(self):
        return list(self)


class TestReplayGhost(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_launcher()
        cls.cfg = compose_config("sim", config_name="run_loco_hoi_track")
        cls.model = mujoco.MjModel.from_xml_path((REPO_ROOT / cls.cfg.inspire.xml).as_posix())

    def test_recorded_body_and_hand_state_fill_the_inspire_model(self):
        body = np.arange(29, dtype=np.float64).reshape(1, -1)
        hand = np.linspace(0.1, 1.2, 12, dtype=np.float64).reshape(1, -1)
        joints = self.module.recorded_joint_positions(ArrayState(dof_pos=body, hand_pos=hand), self.cfg, self.model)
        names = inspire_dof.actuator_names(self.model)

        self.assertEqual(joints.shape, (1, 53))
        for index, name in enumerate(self.cfg.robot.dof.joint_names):
            self.assertAlmostEqual(joints[0, names.index(name)], body[0, index])
        for index, name in enumerate(self.cfg.inspire.dof.joint_names):
            self.assertAlmostEqual(joints[0, names.index(name)], hand[0, index])
        follower, mimic = next(iter(self.cfg.inspire.mimic.items()))
        driver_value = joints[0, names.index(mimic.driver)]
        expected = np.clip(driver_value * mimic.multiplier + mimic.offset, mimic.lower, mimic.upper)
        self.assertAlmostEqual(joints[0, names.index(follower)], expected)

    def test_ghost_contains_only_robot_geometries_and_is_transparent(self):
        data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, data)
        count = self.module.robot_geom_count(self.model)
        scene = mujoco.MjvScene(self.model, maxgeom=count)

        self.module.draw_ghost(scene, self.model, data)

        self.assertEqual(scene.ngeom, count)
        np.testing.assert_allclose(scene.geoms[0].rgba, self.module.GHOST_RGBA)
        self.assertTrue(all(scene.geoms[index].transparent == 1 for index in range(scene.ngeom)))
        self.assertTrue(all(scene.geoms[index].objid > 0 for index in range(scene.ngeom)))

        torso_geom = next(
            index
            for index in range(self.model.ngeom)
            if mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(self.model.geom_bodyid[index]))
            == "torso_link"
        )
        ghost = next(scene.geoms[index] for index in range(scene.ngeom) if scene.geoms[index].objid == torso_geom)
        self.assertEqual(ghost.dataid, 2 * int(self.model.geom_dataid[torso_geom]))
        np.testing.assert_allclose(ghost.mat, data.geom_xmat[torso_geom].reshape(3, 3))

    def test_reference_uses_the_same_initial_xy_and_yaw_alignment_as_deployment(self):
        names = np.asarray(inspire_dof.actuator_names(self.model))
        yaw_90 = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "motion.npz"
            np.savez_compressed(
                path,
                motion_names=np.asarray(["demo_v02"]),
                motion_lengths=np.asarray([2]),
                joint_names=names,
                joint_pos=np.stack([np.zeros(53), np.ones(53)]).astype(np.float32),
                anchor_pos_w=np.asarray([[2.0, 3.0, 0.8], [3.0, 3.0, 0.8]], dtype=np.float32),
                anchor_quat_w=np.stack([yaw_90, yaw_90]),
                fps=np.asarray([50.0], dtype=np.float32),
            )
            cfg = OmegaConf.create({"motion": {"file": path.as_posix(), "name": "demo_v02"}})
            reference = self.module.reference_trajectory(
                ArrayState(motion_frame=np.asarray([-1, 0, 1])), cfg, self.model
            )

        self.assertIsNotNone(reference)
        frames, joints, position, quaternion = reference
        np.testing.assert_array_equal(frames, [-1, 0, 1])
        np.testing.assert_allclose(joints[1], 1.0)
        np.testing.assert_allclose(position[0], [0.0, 0.0, 0.8], atol=1e-6)
        np.testing.assert_allclose(position[1], [0.0, -1.0, 0.8], atol=1e-6)
        np.testing.assert_allclose(quaternion[0], [1.0, 0.0, 0.0, 0.0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
