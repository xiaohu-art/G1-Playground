import unittest
from types import SimpleNamespace

import mujoco
import numpy as np

from g1_playground.policy.body_hand.motion import aligned_object_frame_zero
from g1_playground.simulation import G1MujocoBackend, G1MujocoDdsServer, compile_mujoco_scene
from g1_playground.utils.dof import compose_dof_config
from tests.config_helpers import REPO_ROOT, compose_config

ROBOT_XML = REPO_ROOT / "assets/robots/g1/g1_29dof_rev_1_0.xml"
INSPIRE_SCENE = REPO_ROOT / "assets/robots/g1_inspire/scene.xml"
SMALLTABLE = REPO_ROOT / "assets/objects/smalltable/object.xml"


class TestMujocoBackend(unittest.TestCase):
    def test_selected_motion_builds_the_dynamic_object_scene(self):
        cfg = compose_config(
            "sim",
            "hoi=depth/smalltable",
            "motion.name=sub17_smalltable_000_v02",
            config_name="run_loco_hoi_track",
        )
        position, quaternion = aligned_object_frame_zero(cfg.motion)
        model = compile_mujoco_scene(
            INSPIRE_SCENE,
            SMALLTABLE,
            object_position=position,
            object_quaternion=quaternion,
        )
        object_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hoi_object")
        object_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hoi_object_freejoint")
        qpos_address = model.jnt_qposadr[object_joint]

        self.assertEqual(model.nu, 53)
        self.assertGreater(model.body_mass[object_body], 0.0)
        self.assertEqual(model.jnt_type[object_joint], mujoco.mjtJoint.mjJNT_FREE)
        np.testing.assert_allclose(model.qpos0[qpos_address : qpos_address + 7], np.r_[position, quaternion])

        backend = G1MujocoBackend(
            INSPIRE_SCENE.as_posix(),
            expected_actuators=53,
            object_mjcf=SMALLTABLE.as_posix(),
            object_position=position,
            object_quaternion=quaternion,
        )
        self.assertEqual(backend.read().joint_pos.shape, (53,))

    def test_elastic_support_reaches_the_floor_when_the_three_second_ramp_finishes(self):
        cfg = compose_config("sim")
        dof = compose_dof_config(cfg.robot.dof, cfg.policy.dof)
        backend = G1MujocoBackend(ROBOT_XML.as_posix(), 0.001, elastic_support_scale=1.0)
        for _ in range(5000):
            backend.step(np.zeros(29), 1.0)

        measured = backend.read().joint_pos
        command = SimpleNamespace(
            valid=True,
            age_seconds=0.0,
            q=measured.copy(),
            dq=np.zeros(29),
            tau=np.zeros(29),
            kp=np.asarray(dof.stiffness),
            kd=np.asarray(dof.damping),
        )

        class Bridge:
            def get_command(self):
                return command

            def publish_lowstate(self, snapshot):
                pass

            def close(self):
                pass

        server = G1MujocoDdsServer(backend, Bridge(), SimpleNamespace, cfg.robot.dof.torque_limits)
        floor = mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        first_floor_contact = None
        for step in range(3100):
            elapsed = step * backend.timestep
            alpha = min((step // 20 + 1) / 150, 1.0)
            command.q = (1.0 - alpha) * measured + alpha * np.asarray(dof.default_pos)
            server.step(now=elapsed)
            if first_floor_contact is None and any(
                backend.data.contact[index].geom1 == floor or backend.data.contact[index].geom2 == floor
                for index in range(backend.data.ncon)
            ):
                first_floor_contact = elapsed

        self.assertIsNotNone(first_floor_contact)
        self.assertGreaterEqual(first_floor_contact, 3.0)
        self.assertLess(first_floor_contact, 3.1)


if __name__ == "__main__":
    unittest.main()
