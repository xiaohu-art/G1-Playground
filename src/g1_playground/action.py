from dataclasses import dataclass

import numpy as np


@dataclass
class JointAction:
    """One snapshot of motor targets. The robot copies these into LowCmd_ at write time.

    Any field left as None falls back to the robot's defaults.
    """
    q: np.ndarray
    dq: np.ndarray | None = None
    tau: np.ndarray | None = None
    kp: np.ndarray | None = None
    kd: np.ndarray | None = None
