import logging
import socketserver
import struct
import threading
import time

import numpy as np

from .preprocess import preprocess_depth

logger = logging.getLogger(__name__)

_MAGIC = b"G1DP"
_HEADER = struct.Struct("<4sQHH")


def _read_exact(stream, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("Depth preview stream closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_depth_preview(stream) -> tuple[int, np.ndarray]:
    magic, sequence, height, width = _HEADER.unpack(_read_exact(stream, _HEADER.size))
    if magic != _MAGIC or height == 0 or width == 0:
        raise RuntimeError("Invalid depth preview frame")
    image = np.frombuffer(_read_exact(stream, height * width), dtype=np.uint8).reshape(height, width).copy()
    image.flags.writeable = False
    return sequence, image


def encode_depth_preview(
    sequence: int,
    depth_m,
    *,
    height: int,
    width: int,
    min_distance: float,
    max_distance: float,
) -> bytes:
    observation = preprocess_depth(
        depth_m,
        height=height,
        width=width,
        min_distance=min_distance,
        max_distance=max_distance,
    ).reshape(height, width)
    image = np.rint(observation * 255.0).astype(np.uint8)
    return _HEADER.pack(_MAGIC, sequence, height, width) + image.tobytes()


class _PreviewServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class _PreviewHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        server = self.server
        self.request.settimeout(0.5)
        sequence = -1
        try:
            while not server.stop_event.is_set():
                with server.frame_lock:
                    frame = server.frame
                if frame is None or frame.sequence == sequence:
                    time.sleep(0.002)
                    continue
                sequence = frame.sequence
                packet = encode_depth_preview(
                    sequence,
                    frame.depth_m,
                    height=server.height,
                    width=server.width,
                    min_distance=server.min_distance,
                    max_distance=server.max_distance,
                )
                self.request.sendall(packet)
        except (OSError, RuntimeError):
            return


class DepthPreviewServer:
    def __init__(
        self,
        *,
        height: int,
        width: int,
        min_distance: float,
        max_distance: float,
        host: str = "127.0.0.1",
        port: int = 9876,
    ):
        self._server = _PreviewServer((host, int(port)), _PreviewHandler)
        self._server.height = int(height)
        self._server.width = int(width)
        self._server.min_distance = float(min_distance)
        self._server.max_distance = float(max_distance)
        self._server.stop_event = threading.Event()
        self._server.frame_lock = threading.Lock()
        self._server.frame = None
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.05},
            name="DepthPreviewServer",
            daemon=True,
        )
        self._thread.start()
        logger.info("Depth observation preview listening on %s:%d", host, self.port)

    def publish(self, frame) -> None:
        with self._server.frame_lock:
            self._server.frame = frame

    def shutdown(self) -> None:
        self._server.stop_event.set()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()
