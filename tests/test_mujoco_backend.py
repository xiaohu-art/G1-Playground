import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np

from g1_playground.simulation import G1MujocoBackend, G1MujocoDdsServer, mujoco_backend
from g1_playground.utils.dof import compose_dof_config
from tests.config_helpers import REPO_ROOT, compose_config

XML_PATH = REPO_ROOT / "assets/robots/g1/g1_29dof_rev_1_0.xml"


class TestG1MujocoBackend(unittest.TestCase):
    def test_model_must_keep_the_29_actuator_contract(self):
        model = SimpleNamespace(opt=SimpleNamespace(timestep=None), nu=28)

        with (
            patch.object(mujoco_backend.mujoco.MjModel, "from_xml_path", return_value=model),
            self.assertRaisesRegex(ValueError, "exactly 29 actuators"),
        ):
            G1MujocoBackend("wrong.xml", 0.001)

    def test_dds_server_keeps_the_existing_twenty_step_pd_path(self):
        cfg = compose_config("sim")
        dof = compose_dof_config(cfg.robot.dof, cfg.policy.dof)
        backend = G1MujocoBackend(XML_PATH.as_posix(), 0.001)
        reference = G1MujocoBackend(XML_PATH.as_posix(), 0.001)
        target = np.asarray(dof.default_pos) + np.linspace(-0.05, 0.05, 29)
        command = SimpleNamespace(
            valid=True,
            age_seconds=0.0,
            q=target,
            dq=np.zeros(29),
            tau=np.zeros(29),
            kp=np.asarray(dof.stiffness),
            kd=np.asarray(dof.damping),
        )

        class Bridge:
            def get_command(self):
                return command

            def publish_lowstate(self, _snapshot):
                return 0

            def close(self):
                return True

        server = G1MujocoDdsServer(
            backend,
            Bridge(),
            SimpleNamespace,
            cfg.robot.dof.torque_limits,
        )

        for index in range(20):
            now = 10.0 + 0.001 * index
            state = reference.read()
            torque = (target - state.joint_pos) * np.asarray(dof.stiffness)
            torque -= state.joint_vel * np.asarray(dof.damping)
            torque_limits = np.asarray(cfg.robot.dof.torque_limits)
            torque = np.clip(torque, -torque_limits, torque_limits)
            server.step(now=now)
            reference.step(torque, server.support_scale(now))

        np.testing.assert_array_equal(backend.data.qpos, reference.data.qpos)
        np.testing.assert_array_equal(backend.data.qvel, reference.data.qvel)
        np.testing.assert_array_equal(backend.data.ctrl, reference.data.ctrl)

    def test_single_step_matches_direct_mujoco(self):
        backend = G1MujocoBackend(XML_PATH.as_posix(), 0.001)
        reference_model = mujoco.MjModel.from_xml_path(XML_PATH.as_posix())
        reference_model.opt.timestep = 0.001
        reference_data = mujoco.MjData(reference_model)
        mujoco.mj_step(reference_model, reference_data)

        torque = np.linspace(-20.0, 20.0, 29)
        backend_state = backend.step(torque, 0.0)
        reference_data.ctrl[:] = torque
        mujoco.mj_step(reference_model, reference_data)

        np.testing.assert_array_equal(backend.data.qpos, reference_data.qpos)
        np.testing.assert_array_equal(backend.data.qvel, reference_data.qvel)
        np.testing.assert_array_equal(backend_state.joint_torque, reference_data.actuator_force)

    def test_state_is_detached_read_only_and_uses_wire_quaternion_order(self):
        backend = G1MujocoBackend(XML_PATH.as_posix(), 0.001)
        state = backend.read()

        np.testing.assert_array_equal(state.base_quaternion_wxyz, backend.data.qpos[3:7])
        for value in (
            state.joint_pos,
            state.joint_vel,
            state.joint_torque,
            state.base_quaternion_wxyz,
            state.base_angular_velocity,
        ):
            self.assertFalse(value.flags.writeable)
            self.assertFalse(np.shares_memory(value, backend.data.qpos))
            self.assertFalse(np.shares_memory(value, backend.data.qvel))
            self.assertFalse(np.shares_memory(value, backend.data.actuator_force))

    def test_render_copy_is_a_detached_coherent_snapshot(self):
        backend = G1MujocoBackend(XML_PATH.as_posix(), 0.001)
        render_data = mujoco.MjData(backend.model)

        backend.copy_data_to(render_data)

        self.assertIsNot(render_data, backend.data)
        self.assertFalse(np.shares_memory(render_data.qpos, backend.data.qpos))
        np.testing.assert_array_equal(render_data.qpos, backend.data.qpos)
        np.testing.assert_array_equal(render_data.qvel, backend.data.qvel)
        np.testing.assert_array_equal(render_data.ctrl, backend.data.ctrl)
        render_qpos = render_data.qpos.copy()

        backend.step(np.linspace(-20.0, 20.0, 29), 0.0)

        np.testing.assert_array_equal(render_data.qpos, render_qpos)
        self.assertFalse(np.array_equal(render_data.qpos, backend.data.qpos))

        backend.copy_data_to(render_data)

        np.testing.assert_array_equal(render_data.qpos, backend.data.qpos)
        np.testing.assert_array_equal(render_data.qvel, backend.data.qvel)
        np.testing.assert_array_equal(render_data.ctrl, backend.data.ctrl)

    def test_disabled_elastic_support_preserves_external_force_and_torque(self):
        backend = G1MujocoBackend(XML_PATH.as_posix(), 0.001)
        support = backend.elastic_support
        backend.data.xfrc_applied[support.body_id] = 1.0

        backend.step(np.zeros(29), 0.0)

        np.testing.assert_array_equal(backend.data.xfrc_applied[support.body_id], np.ones(6))

    def test_elastic_support_handles_zero_distance_and_continuous_scale(self):
        backend = G1MujocoBackend(XML_PATH.as_posix(), 0.001)
        support = backend.elastic_support
        backend.data.qvel[:3] = 0.0
        backend.data.qpos[:3] = support.anchor
        support.scale = 1.0
        zero_force = support.apply(backend.data)
        support.remove(backend.data, zero_force)
        np.testing.assert_array_equal(backend.data.xfrc_applied[support.body_id], np.zeros(6))

        backend.data.qpos[:3] = np.zeros(3)
        support.scale = 0.25
        quarter_force = support.apply(backend.data)
        support.remove(backend.data, quarter_force)
        support.scale = 1.0
        full_force = support.apply(backend.data)
        support.remove(backend.data, full_force)
        np.testing.assert_allclose(full_force, 4.0 * quarter_force)

        for invalid in (-0.1, 1.1, float("nan")):
            with self.subTest(scale=invalid), self.assertRaises(ValueError):
                backend.step(np.zeros(29), invalid)

    def test_enabled_elastic_support_is_applied_to_the_initial_step(self):
        unsupported = G1MujocoBackend(XML_PATH.as_posix(), 0.001)
        supported = G1MujocoBackend(XML_PATH.as_posix(), 0.001, elastic_support_scale=1.0)

        self.assertGreater(supported.data.qvel[2], unsupported.data.qvel[2])
        np.testing.assert_array_equal(
            supported.data.xfrc_applied[supported.elastic_support.body_id],
            np.zeros(6),
        )

    def test_elastic_support_lands_when_locomotion_takes_over(self):
        cfg = compose_config("sim")
        dof = compose_dof_config(cfg.robot.dof, cfg.policy.dof)
        backend = G1MujocoBackend(XML_PATH.as_posix(), 0.001, elastic_support_scale=1.0)
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

            def publish_lowstate(self, _snapshot):
                return 0

            def close(self):
                return True

        server = G1MujocoDdsServer(
            backend,
            Bridge(),
            SimpleNamespace,
            cfg.robot.dof.torque_limits,
        )
        floor = mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        standing = np.asarray(dof.default_pos)
        first_floor_contact = None
        for step in range(3100):
            elapsed = step * backend.timestep
            alpha = min((step // 20 + 1) / 150, 1.0)
            command.q = (1.0 - alpha) * measured + alpha * standing
            server.step(now=elapsed)
            floor_contact = any(
                backend.data.contact[index].geom1 == floor or backend.data.contact[index].geom2 == floor
                for index in range(backend.data.ncon)
            )
            if floor_contact and first_floor_contact is None:
                first_floor_contact = elapsed

        self.assertIsNotNone(first_floor_contact)
        self.assertGreaterEqual(first_floor_contact, 3.0)
        self.assertLess(first_floor_contact, 3.1)


if __name__ == "__main__":
    unittest.main()
