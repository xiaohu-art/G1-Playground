import fcntl
import mmap
import os
import struct
import time
from pathlib import Path

import numpy as np

from g1_playground.camera.depth import DepthFrame

MUJOCO_DEPTH_PATH = "/dev/shm/g1_playground-depth.bin"
CAMERA_NAME = "training_d435_depth"
_HEADER = struct.Struct("<QdII")


class DepthFrameWriter:
    def __init__(self, height: int, width: int, path: str = MUJOCO_DEPTH_PATH):
        self.height = int(height)
        self.width = int(width)
        self.path = Path(path)
        self.sequence = 0
        self._size = _HEADER.size + self.height * self.width * np.dtype(np.float32).itemsize
        self._fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        os.ftruncate(self._fd, self._size)
        self._map = mmap.mmap(self._fd, self._size, access=mmap.ACCESS_WRITE)
        _HEADER.pack_into(self._map, 0, 0, 0.0, self.height, self.width)

    def write(self, depth_m, timestamp: float | None = None) -> None:
        depth = np.asarray(depth_m, dtype=np.float32)
        if depth.shape != (self.height, self.width):
            raise ValueError(f"Depth frame shape {depth.shape} does not match {(self.height, self.width)}")
        timestamp = time.monotonic() if timestamp is None else float(timestamp)
        self.sequence += 1
        fcntl.flock(self._fd, fcntl.LOCK_EX)
        try:
            target = np.frombuffer(self._map, dtype=np.float32, count=self.height * self.width, offset=_HEADER.size)
            target[:] = depth.reshape(-1)
            del target
            _HEADER.pack_into(self._map, 0, self.sequence, timestamp, self.height, self.width)
        finally:
            fcntl.flock(self._fd, fcntl.LOCK_UN)

    def close(self) -> None:
        if self._map is None:
            return
        self._map.close()
        self._map = None
        os.close(self._fd)
        self._fd = -1
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


class MujocoDepthCamera:
    """Read the latest MuJoCo depth frame produced by scripts/simulate.py."""

    def __init__(self, path: str = MUJOCO_DEPTH_PATH):
        self.path = Path(path)
        self.shape: tuple[int, int] | None = None
        self._fd = -1
        self._map: mmap.mmap | None = None

    def _open(self) -> None:
        fd = os.open(self.path, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH)
            try:
                header = os.pread(fd, _HEADER.size, 0)
                if len(header) != _HEADER.size:
                    raise RuntimeError("MuJoCo depth frame header is incomplete")
                _, _, height, width = _HEADER.unpack(header)
                size = _HEADER.size + height * width * np.dtype(np.float32).itemsize
                if height == 0 or width == 0 or os.fstat(fd).st_size != size:
                    raise RuntimeError("MuJoCo depth frame has invalid dimensions")
                mapping = mmap.mmap(fd, size, access=mmap.ACCESS_READ)
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        self._map = mapping
        self.shape = (height, width)

    def self_check(self, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                self.read()
                return
            except (FileNotFoundError, RuntimeError):
                time.sleep(0.01)
        raise RuntimeError("MuJoCo depth camera self check failed: no frame received")

    def read(self) -> DepthFrame:
        if self._map is None:
            self._open()
        fcntl.flock(self._fd, fcntl.LOCK_SH)
        try:
            sequence, timestamp, height, width = _HEADER.unpack_from(self._map, 0)
            if sequence == 0:
                raise RuntimeError("MuJoCo depth frame is not available")
            if (height, width) != self.shape:
                raise RuntimeError("MuJoCo depth frame dimensions changed")
            depth = (
                np.frombuffer(self._map, dtype=np.float32, count=height * width, offset=_HEADER.size)
                .reshape(height, width)
                .copy()
            )
        finally:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        depth.flags.writeable = False
        return DepthFrame(depth, timestamp, sequence)

    def shutdown(self) -> None:
        if self._map is not None:
            self._map.close()
            self._map = None
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1


def depth_buffer_to_meters(depth_buffer, model) -> np.ndarray:
    """Convert MuJoCo's reversed-Z depth buffer to camera distance in metres."""

    extent = model.stat.extent
    near = np.float32(model.vis.map.znear * extent)
    far = np.float32(model.vis.map.zfar * extent)
    c_coef = -(far + near) / (far - near)
    d_coef = -(np.float32(2.0) * far * near) / (far - near)
    c_coef = np.float32(-0.5) * c_coef - np.float32(0.5)
    d_coef = np.float32(-0.5) * d_coef
    return (d_coef / (np.asarray(depth_buffer, dtype=np.float64) + c_coef)).astype(np.float32)
