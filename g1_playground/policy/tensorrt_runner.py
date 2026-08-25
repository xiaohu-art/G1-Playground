import ctypes
import hashlib
import importlib
import logging
import os
import platform
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / ".cache/tensorrt"


class _CudaRuntime:
    HOST_TO_DEVICE = 1
    DEVICE_TO_HOST = 2
    COMPUTE_CAPABILITY_MAJOR = 75
    COMPUTE_CAPABILITY_MINOR = 76

    def __init__(self):
        self.lib = self._load_library()

        self.lib.cudaGetErrorString.argtypes = [ctypes.c_int]
        self.lib.cudaGetErrorString.restype = ctypes.c_char_p
        self.lib.cudaGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self.lib.cudaGetDevice.restype = ctypes.c_int
        self.lib.cudaDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        self.lib.cudaDeviceGetAttribute.restype = ctypes.c_int
        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaMalloc.restype = ctypes.c_int
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaFree.restype = ctypes.c_int
        self.lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.cudaStreamCreate.restype = ctypes.c_int
        self.lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamDestroy.restype = ctypes.c_int
        self.lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamSynchronize.restype = ctypes.c_int
        self.lib.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.lib.cudaMemcpyAsync.restype = ctypes.c_int

    @staticmethod
    def _load_library():
        machine = platform.machine()
        roots = [Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))]
        roots.extend(sorted(Path("/usr/local").glob("cuda-*"), reverse=True))
        candidates = ["libcudart.so"]
        for root in roots:
            candidates.extend(
                (
                    root / "lib64/libcudart.so",
                    root / f"targets/{machine}-linux/lib/libcudart.so",
                )
            )
        for candidate in candidates:
            try:
                return ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue
        raise RuntimeError("TensorRT requires a CUDA runtime under CUDA_HOME or /usr/local/cuda*")

    def check(self, code: int, operation: str) -> None:
        if code == 0:
            return
        message = self.lib.cudaGetErrorString(code)
        detail = message.decode() if message else f"CUDA error {code}"
        raise RuntimeError(f"{operation} failed: {detail}")

    def compute_capability(self) -> tuple[int, int]:
        device = ctypes.c_int()
        self.check(self.lib.cudaGetDevice(ctypes.byref(device)), "cudaGetDevice")
        major, minor = ctypes.c_int(), ctypes.c_int()
        self.check(
            self.lib.cudaDeviceGetAttribute(ctypes.byref(major), self.COMPUTE_CAPABILITY_MAJOR, device.value),
            "cudaDeviceGetAttribute(compute capability major)",
        )
        self.check(
            self.lib.cudaDeviceGetAttribute(ctypes.byref(minor), self.COMPUTE_CAPABILITY_MINOR, device.value),
            "cudaDeviceGetAttribute(compute capability minor)",
        )
        return major.value, minor.value

    def malloc(self, size: int) -> int:
        pointer = ctypes.c_void_p()
        self.check(self.lib.cudaMalloc(ctypes.byref(pointer), size), "cudaMalloc")
        return int(pointer.value)

    def free(self, pointer: int) -> None:
        self.check(self.lib.cudaFree(ctypes.c_void_p(pointer)), "cudaFree")

    def create_stream(self) -> int:
        stream = ctypes.c_void_p()
        self.check(self.lib.cudaStreamCreate(ctypes.byref(stream)), "cudaStreamCreate")
        return int(stream.value)

    def destroy_stream(self, stream: int) -> None:
        self.check(self.lib.cudaStreamDestroy(ctypes.c_void_p(stream)), "cudaStreamDestroy")

    def copy_to_device(self, device: int, host: np.ndarray, stream: int) -> None:
        self.check(
            self.lib.cudaMemcpyAsync(
                ctypes.c_void_p(device),
                ctypes.c_void_p(host.ctypes.data),
                host.nbytes,
                self.HOST_TO_DEVICE,
                ctypes.c_void_p(stream),
            ),
            "cudaMemcpyAsync(host to device)",
        )

    def copy_from_device(self, host: np.ndarray, device: int, stream: int) -> None:
        self.check(
            self.lib.cudaMemcpyAsync(
                ctypes.c_void_p(host.ctypes.data),
                ctypes.c_void_p(device),
                host.nbytes,
                self.DEVICE_TO_HOST,
                ctypes.c_void_p(stream),
            ),
            "cudaMemcpyAsync(device to host)",
        )

    def synchronize(self, stream: int) -> None:
        self.check(self.lib.cudaStreamSynchronize(ctypes.c_void_p(stream)), "cudaStreamSynchronize")


def _load_tensorrt():
    root = Path(os.environ.get("TENSORRT_ROOT", Path.home() / "TensorRT"))
    for library in ("libnvinfer.so.10", "libnvinfer_plugin.so.10", "libnvonnxparser.so.10"):
        path = root / "lib" / library
        if path.exists():
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    try:
        return importlib.import_module("tensorrt")
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "TensorRT Python bindings are unavailable; install the wheel matching this machine and set TENSORRT_ROOT"
        ) from error


def _model_digest(model_path: Path) -> str:
    digest = hashlib.sha256()
    for path in (model_path, model_path.with_name(f"{model_path.name}.data")):
        if not path.exists():
            continue
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()[:16]


class TensorRTRunner:
    def __init__(self, model_path):
        self.model_path = Path(model_path).resolve()
        if self.model_path.suffix != ".onnx" or not self.model_path.is_file():
            raise ValueError(f"TensorRT requires an ONNX model, got {self.model_path}")

        self.cuda = _CudaRuntime()
        self.trt = _load_tensorrt()
        major, minor = self.cuda.compute_capability()
        cache_name = (
            f"{self.model_path.stem}-{_model_digest(self.model_path)}-"
            f"trt{self.trt.__version__}-{platform.machine()}-sm{major}{minor}.engine"
        )
        self.engine_path = CACHE_DIR / cache_name
        self.trt_logger = self.trt.Logger(self.trt.Logger.ERROR)
        self.runtime = self.trt.Runtime(self.trt_logger)
        self.engine = self._load_engine()
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"TensorRT could not create an execution context for {self.model_path}")

        self.input_names = []
        self.output_names = []
        self.shapes = {}
        self.host_buffers = {}
        self.device_buffers = {}
        self.stream = 0
        try:
            for index in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(index)
                shape = tuple(int(value) for value in self.engine.get_tensor_shape(name))
                if any(value <= 0 for value in shape):
                    raise ValueError(f"TensorRT runner only supports static tensors; {name} has shape {shape}")
                dtype = np.dtype(self.trt.nptype(self.engine.get_tensor_dtype(name)))
                host = np.empty(shape, dtype=dtype)
                device = self.cuda.malloc(host.nbytes)
                if not self.context.set_tensor_address(name, device):
                    raise RuntimeError(f"TensorRT refused the buffer for {name}")
                self.shapes[name] = shape
                self.host_buffers[name] = host
                self.device_buffers[name] = device
                names = (
                    self.input_names
                    if self.engine.get_tensor_mode(name) == self.trt.TensorIOMode.INPUT
                    else self.output_names
                )
                names.append(name)
            self.input_names = tuple(self.input_names)
            self.output_names = tuple(self.output_names)
            self.stream = self.cuda.create_stream()
        except Exception:
            self.close()
            raise

    def _load_engine(self):
        if self.engine_path.is_file():
            engine = self.runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
            if engine is not None:
                logger.debug("Loaded TensorRT engine %s", self.engine_path)
                return engine
            logger.warning("Rebuilding incompatible TensorRT engine %s", self.engine_path)

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Building TensorRT engine from %s", self.model_path)
        started = time.monotonic()
        builder = self.trt.Builder(self.trt_logger)
        network = builder.create_network(1 << int(self.trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = self.trt.OnnxParser(network, self.trt_logger)
        if not parser.parse_from_file(str(self.model_path)):
            errors = "\n".join(str(parser.get_error(index)) for index in range(parser.num_errors))
            raise RuntimeError(f"TensorRT could not parse {self.model_path}:\n{errors}")
        config = builder.create_builder_config()
        config.clear_flag(self.trt.BuilderFlag.TF32)
        plan = builder.build_serialized_network(network, config)
        if plan is None:
            raise RuntimeError(f"TensorRT could not build {self.model_path}")
        engine_bytes = bytes(plan)
        temporary = self.engine_path.with_suffix(".engine.tmp")
        temporary.write_bytes(engine_bytes)
        os.replace(temporary, self.engine_path)
        engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if engine is None:
            raise RuntimeError(f"TensorRT could not load the engine built from {self.model_path}")
        logger.info("Built TensorRT engine %s in %.2f s", self.engine_path, time.monotonic() - started)
        return engine

    def shape(self, name: str) -> tuple[int, ...]:
        return self.shapes[name]

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if set(inputs) != set(self.input_names):
            raise ValueError(f"TensorRT inputs must be {self.input_names}, got {tuple(inputs)}")
        for name in self.input_names:
            values = np.asarray(inputs[name], dtype=self.host_buffers[name].dtype)
            if values.shape != self.shapes[name]:
                raise ValueError(f"TensorRT input {name} must have shape {self.shapes[name]}, got {values.shape}")
            np.copyto(self.host_buffers[name], values)
            self.cuda.copy_to_device(self.device_buffers[name], self.host_buffers[name], self.stream)

        if not self.context.execute_async_v3(stream_handle=self.stream):
            raise RuntimeError(f"TensorRT inference failed for {self.model_path}")
        for name in self.output_names:
            self.cuda.copy_from_device(self.host_buffers[name], self.device_buffers[name], self.stream)
        self.cuda.synchronize(self.stream)
        return {name: self.host_buffers[name].copy() for name in self.output_names}

    def close(self) -> None:
        if getattr(self, "stream", 0):
            self.cuda.destroy_stream(self.stream)
            self.stream = 0
        for pointer in getattr(self, "device_buffers", {}).values():
            self.cuda.free(pointer)
        self.device_buffers = {}

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
