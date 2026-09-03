import io
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from g1_playground.camera import RealSenseDepthCamera
from g1_playground.policy.body_hand.depth import preprocess_depth, read_depth_preview
from g1_playground.policy.body_hand.depth.preview import encode_depth_preview
from g1_playground.simulation.mujoco_depth import DepthFrameWriter, MujocoDepthCamera


class TestDepthCamera(unittest.TestCase):
    def test_mujoco_frame_reaches_policy_preprocessing_and_preview(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "depth.bin"
            writer = DepthFrameWriter(2, 3, path.as_posix())
            camera = MujocoDepthCamera(path.as_posix())
            try:
                depth = np.array([[0.0, 0.3, 1.5], [2.9, 3.0, np.nan]], dtype=np.float32)
                writer.write(depth, timestamp=12.5)
                camera.self_check(timeout=0.1)
                frame = camera.read()
                sequence, image = read_depth_preview(
                    io.BytesIO(
                        encode_depth_preview(
                            frame.sequence,
                            frame.depth_m,
                            height=2,
                            width=3,
                            min_distance=0.25,
                            max_distance=3.0,
                        )
                    )
                )

                expected = (
                    np.rint(preprocess_depth(depth, height=2, width=3, min_distance=0.25, max_distance=3.0) * 255.0)
                    .astype(np.uint8)
                    .reshape(2, 3)
                )
                self.assertEqual((frame.sequence, frame.timestamp, sequence), (1, 12.5, 1))
                np.testing.assert_array_equal(image, expected)
            finally:
                camera.shutdown()
                writer.close()

    def test_realsense_z16_is_exposed_as_meters_at_60_hz(self):
        depth_data = np.array([[0, 1000, 2000], [3000, 4000, 5000]], dtype=np.uint16)

        class Pipeline:
            def start(self, config):
                self.config = config
                intrinsics = SimpleNamespace(width=3, height=2, fx=10.0, fy=11.0, ppx=1.0, ppy=0.5)
                stream = SimpleNamespace(
                    as_video_stream_profile=lambda: SimpleNamespace(get_intrinsics=lambda: intrinsics)
                )
                sensor = SimpleNamespace(get_depth_scale=lambda: 0.001)
                device = SimpleNamespace(first_depth_sensor=lambda: sensor)
                return SimpleNamespace(get_device=lambda: device, get_stream=lambda _: stream)

            def wait_for_frames(self, timeout_ms):
                time.sleep(0.002)
                return SimpleNamespace(get_depth_frame=lambda: SimpleNamespace(get_data=lambda: depth_data))

            def stop(self):
                pass

        class Config:
            def enable_device(self, serial):
                self.device = serial

            def enable_stream(self, *stream):
                self.stream = stream

        pipeline = Pipeline()
        config = Config()
        rs = SimpleNamespace(
            pipeline=lambda: pipeline,
            config=lambda: config,
            stream=SimpleNamespace(depth="depth"),
            format=SimpleNamespace(z16="z16"),
        )
        with patch("g1_playground.camera.realsense.importlib.import_module", return_value=rs):
            camera = RealSenseDepthCamera(serial="camera", width=3, height=2, frequency=60)
        try:
            camera.self_check(timeout=0.1)
            frame = camera.read()
            np.testing.assert_allclose(frame.depth_m, depth_data.astype(np.float32) * 0.001)
            self.assertEqual(config.stream, ("depth", 3, 2, "z16", 60))
        finally:
            camera.shutdown()


if __name__ == "__main__":
    unittest.main()
