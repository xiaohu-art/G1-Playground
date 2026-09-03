import importlib
import logging
import threading
import time

import numpy as np

from .depth import DepthFrame, snapshot_depth

logger = logging.getLogger(__name__)


class RealSenseDepthCamera:
    """Latest-frame depth source backed by a RealSense Z16 stream."""

    def __init__(
        self,
        *,
        serial: str | None = None,
        width: int = 480,
        height: int = 270,
        frequency: int = 60,
    ):
        try:
            rs = importlib.import_module("pyrealsense2")
        except ImportError as error:
            raise RuntimeError(
                "RealSenseDepthCamera requires pyrealsense2 built for the active Python environment"
            ) from error

        self._rs = rs
        self._pipeline = rs.pipeline()
        config = rs.config()
        if serial is not None:
            config.enable_device(serial)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, frequency)
        profile = self._pipeline.start(config)
        self.depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())

        stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        intrinsics = stream.get_intrinsics()
        self.intrinsics = np.array([intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy], dtype=np.float32)
        self.shape = (int(intrinsics.height), int(intrinsics.width))

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._frame: DepthFrame | None = None
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._capture, name="RealSenseDepthCamera", daemon=True)
        self._thread.start()

    def _capture(self) -> None:
        sequence = 0
        try:
            while not self._stop_event.is_set():
                try:
                    frames = self._pipeline.wait_for_frames(timeout_ms=100)
                except RuntimeError:
                    continue
                depth_frame = frames.get_depth_frame()
                if not depth_frame:
                    continue
                depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale
                sequence += 1
                frame = DepthFrame(snapshot_depth(depth_m), time.monotonic(), sequence)
                with self._lock:
                    self._frame = frame
        except BaseException as error:
            with self._lock:
                self._error = error

    def self_check(self, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                frame, error = self._frame, self._error
            if frame is not None:
                logger.info(
                    "RealSense depth camera ready: %dx%d, scale %.6f m",
                    frame.depth_m.shape[1],
                    frame.depth_m.shape[0],
                    self.depth_scale,
                )
                return
            if error is not None:
                raise RuntimeError("RealSense depth capture failed") from error
            time.sleep(0.01)
        raise RuntimeError("RealSense depth camera self check failed: no frame received")

    def read(self) -> DepthFrame:
        with self._lock:
            frame, error = self._frame, self._error
        if error is not None:
            raise RuntimeError("RealSense depth capture failed") from error
        if frame is None:
            raise RuntimeError("RealSense depth frame is not available")
        return frame

    def shutdown(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=0.2)
        self._pipeline.stop()
        self._thread.join(timeout=1.0)
