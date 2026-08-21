import math
import unittest
from types import SimpleNamespace

import numpy as np

from g1_playground.simulation import G1MujocoBackend, G1MujocoDdsServer
from g1_playground.utils import resolve_repo_path


class FakeSportSnapshot:
    def __init__(self):
        self.position = [0.0, 0.0, 0.0]
        self.velocity = [0.0, 0.0, 0.0]
        self.body_height = 0.0


class FakeRobotEndpoint:
    def __init__(self):
        self.lowstate_writes = 0
        self.sport_states = []

    def get_command(self):
        return SimpleNamespace(valid=False, age_seconds=0.0, q=[], dq=[], tau=[], kp=[], kd=[])

    def publish_lowstate(self, snapshot):
        self.lowstate_writes += 1
        return self.lowstate_writes

    def publish_sport_state(self, snapshot):
        self.sport_states.append((list(snapshot.position), list(snapshot.velocity), snapshot.body_height))
        return len(self.sport_states)

    def close(self):
        return True


def build_server(**kwargs):
    backend = G1MujocoBackend(resolve_repo_path("assets/robots/g1/g1_29dof_rev_1_0.xml"))
    endpoint = FakeRobotEndpoint()
    server = G1MujocoDdsServer(
        backend,
        endpoint,
        SimpleNamespace,
        [200.0] * 29,
        sport_state_factory=FakeSportSnapshot,
        **kwargs,
    )
    return backend, endpoint, server


class TestMujocoRootState(unittest.TestCase):
    def test_root_state_is_a_detached_read_only_snapshot(self):
        backend = G1MujocoBackend(resolve_repo_path("assets/robots/g1/g1_29dof_rev_1_0.xml"))
        state = backend.read()
        for field in ("base_position_world", "base_linear_velocity_world"):
            with self.subTest(field=field):
                array = getattr(state, field)
                self.assertEqual(array.shape, (3,))
                self.assertFalse(array.flags.writeable)
                self.assertFalse(np.shares_memory(array, backend.data.qpos))
                self.assertFalse(np.shares_memory(array, backend.data.qvel))

    def test_root_state_tracks_the_free_joint(self):
        backend = G1MujocoBackend(resolve_repo_path("assets/robots/g1/g1_29dof_rev_1_0.xml"))
        backend.data.qpos[0:3] = [1.25, -0.5, 0.8]
        backend.data.qvel[0:3] = [0.3, 0.0, -0.1]
        state = backend.read()
        np.testing.assert_allclose(state.base_position_world, [1.25, -0.5, 0.8])
        np.testing.assert_allclose(state.base_linear_velocity_world, [0.3, 0.0, -0.1])


class TestSportStatePublishing(unittest.TestCase):
    def test_velocity_is_published_in_the_body_frame(self):
        _, endpoint, server = build_server()
        yaw = math.pi / 2.0
        state = SimpleNamespace(
            base_quaternion_wxyz=np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], dtype=np.float32),
            base_position_world=np.array([2.0, 3.0, 0.78], dtype=np.float32),
            base_linear_velocity_world=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )
        server.publish_sport_state(state)

        position, velocity, height = endpoint.sport_states[-1]
        np.testing.assert_allclose(position, [2.0, 3.0, 0.78], rtol=1e-6)
        np.testing.assert_allclose(velocity, [1.0, 0.0, 0.0], atol=1e-6)
        self.assertAlmostEqual(height, 0.78, places=6)

    def test_body_height_comes_from_the_world_height(self):
        _, endpoint, server = build_server()
        state = SimpleNamespace(
            base_quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            base_position_world=np.array([0.0, 0.0, 0.815], dtype=np.float32),
            base_linear_velocity_world=np.zeros(3, dtype=np.float32),
        )
        server.publish_sport_state(state)
        self.assertAlmostEqual(endpoint.sport_states[-1][2], 0.815, places=6)

    def test_non_positive_height_is_never_published(self):
        _, endpoint, server = build_server()
        for height in (0.0, -0.2, float("nan")):
            with self.subTest(height=height):
                state = SimpleNamespace(
                    base_quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    base_position_world=np.array([0.0, 0.0, height], dtype=np.float32),
                    base_linear_velocity_world=np.zeros(3, dtype=np.float32),
                )
                server.publish_sport_state(state)
        self.assertEqual(endpoint.sport_states, [])

    def test_publishing_is_decimated_to_the_requested_rate(self):
        backend, endpoint, server = build_server(sport_publish_hz=50.0)
        self.assertEqual(server.sport_stride, round(1.0 / (backend.timestep * 50.0)))
        for _ in range(1000):
            server.step(now=0.0)
        self.assertEqual(endpoint.lowstate_writes, 1000)
        self.assertEqual(len(endpoint.sport_states), 50)

    def test_a_disabled_factory_publishes_nothing(self):
        backend = G1MujocoBackend(resolve_repo_path("assets/robots/g1/g1_29dof_rev_1_0.xml"))
        endpoint = FakeRobotEndpoint()
        server = G1MujocoDdsServer(backend, endpoint, SimpleNamespace, [200.0] * 29)
        for _ in range(100):
            server.step(now=0.0)
        self.assertEqual(endpoint.sport_states, [])

    def test_rate_must_be_finite_and_positive(self):
        backend = G1MujocoBackend(resolve_repo_path("assets/robots/g1/g1_29dof_rev_1_0.xml"))
        for rate in (0.0, -50.0, float("inf")):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                G1MujocoDdsServer(backend, FakeRobotEndpoint(), SimpleNamespace, [200.0] * 29, sport_publish_hz=rate)


if __name__ == "__main__":
    unittest.main()
