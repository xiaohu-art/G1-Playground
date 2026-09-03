import unittest
from types import SimpleNamespace

import numpy as np

from g1_playground.simulation.mujoco_dds import G1MujocoDdsServer


def state(torque=None):
    return SimpleNamespace(
        joint_pos=np.full(29, 0.1),
        joint_vel=np.full(29, 0.2),
        joint_torque=np.zeros(29) if torque is None else np.asarray(torque),
        base_quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        base_angular_velocity=np.array([0.1, 0.2, 0.3]),
    )


class Backend:
    timestep = 0.001
    model = SimpleNamespace(nu=29)

    def __init__(self):
        self.current = state()
        self.torques = []
        self.support_scales = []

    def read(self):
        return self.current

    def step(self, torque, support_scale):
        self.torques.append(np.asarray(torque).copy())
        self.support_scales.append(support_scale)
        self.current = state(torque)
        return self.current


class Endpoint:
    def __init__(self):
        self.command = SimpleNamespace(
            valid=True,
            age_seconds=0.0,
            q=np.ones(29),
            dq=np.full(29, -0.5),
            tau=np.full(29, 3.0),
            kp=np.full(29, 100.0),
            kd=np.full(29, 5.0),
        )
        self.published = []
        self.closed = False

    def get_command(self):
        return self.command

    def publish_lowstate(self, snapshot):
        self.published.append(snapshot)
        return len(self.published)

    def close(self):
        self.closed = True
        return True


class TestMujocoDdsServer(unittest.TestCase):
    def test_pd_torque_support_release_and_watchdog_are_one_command_boundary(self):
        backend = Backend()
        endpoint = Endpoint()
        server = G1MujocoDdsServer(backend, endpoint, SimpleNamespace, np.full(29, 10.0))

        for now in (10.0, 11.5, 13.0):
            server.step(now=now)

        np.testing.assert_array_equal(backend.torques, np.full((3, 29), 10.0))
        np.testing.assert_allclose(backend.support_scales, [1.0, 0.845, 0.0])
        np.testing.assert_array_equal(endpoint.published[-1].tau_est, np.full(29, 10.0))

        endpoint.command.age_seconds = 0.100001
        with self.assertRaisesRegex(RuntimeError, "watchdog expired"):
            server.run()
        self.assertTrue(endpoint.closed)


if __name__ == "__main__":
    unittest.main()
