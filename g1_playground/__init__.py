# Fix OMP perfmance issue on ARM platform (Jetson)
import os
import platform

if platform.machine().startswith("aarch64"):
    os.environ["OMP_NUM_THREADS"] = "1"
    # Import torch before NumPy to avoid the Jetson libgomp conflict.
    import torch  # noqa: F401, I001
    import numpy  # noqa: F401, I001

__version__ = "1.5.0"
