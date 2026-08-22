import numpy as np
from omegaconf import DictConfig

NUM_SLOTS = 12
STROKE_UNITS = 1000


def joint_names(dof_cfg: DictConfig) -> list:
    return list(dof_cfg.joint_names)


def limits(dof_cfg: DictConfig) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(dof_cfg.lower, dtype=np.float64)
    upper = np.asarray(dof_cfg.upper, dtype=np.float64)
    names = joint_names(dof_cfg)
    if len(names) != NUM_SLOTS or len(set(names)) != NUM_SLOTS:
        raise ValueError(f"Inspire DoF config requires {NUM_SLOTS} unique joints")
    if lower.shape != (NUM_SLOTS,) or upper.shape != (NUM_SLOTS,):
        raise ValueError(f"Inspire limits must contain {NUM_SLOTS} values each")
    if np.any(lower >= upper):
        raise ValueError("Inspire lower limits must be strictly below the upper limits")
    return lower, upper


def rad_to_q(rad, lower, upper) -> np.ndarray:
    return (upper - np.asarray(rad, dtype=np.float64)) / (upper - lower)


def q_to_rad(q, lower, upper) -> np.ndarray:
    return upper - np.asarray(q, dtype=np.float64) * (upper - lower)


def dq_to_rad_per_s(dq, lower, upper) -> np.ndarray:
    return -np.asarray(dq, dtype=np.float64) * (upper - lower)


def clip_q(q) -> np.ndarray:
    return np.clip(np.asarray(q, dtype=np.float64), 0.0, 1.0)


def quantize_q(q) -> np.ndarray:
    return np.round(clip_q(q) * STROKE_UNITS) / STROKE_UNITS


def validate_q(q) -> np.ndarray:
    array = np.asarray(q, dtype=np.float64).reshape(-1)
    if array.shape != (NUM_SLOTS,):
        raise ValueError(f"expected {NUM_SLOTS} stroke values, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"non-finite stroke value in {array.tolist()}")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"stroke value outside [0, 1] in {array.tolist()}")
    return array


def expand_mimic(driven_by_name: dict, mimic_cfg: DictConfig) -> dict:
    values = dict(driven_by_name)
    for follower, entry in mimic_cfg.items():
        driver = str(entry.driver)
        if driver not in values:
            continue
        target = values[driver] * float(entry.multiplier) + float(entry.offset)
        values[follower] = float(np.clip(target, float(entry.lower), float(entry.upper)))
    return values


def expand_mimic_velocity(driven_by_name: dict, mimic_cfg: DictConfig) -> dict:
    values = dict(driven_by_name)
    for follower, entry in mimic_cfg.items():
        driver = str(entry.driver)
        if driver in values:
            values[follower] = float(values[driver] * float(entry.multiplier))
    return values


def to_dict(array, names) -> dict:
    values = np.asarray(array, dtype=np.float64).reshape(-1)
    if values.shape != (NUM_SLOTS,):
        raise ValueError(f"expected {NUM_SLOTS} values, got {values.shape}")
    return {name: float(values[index]) for index, name in enumerate(names)}


def from_dict(values: dict, names, missing=None) -> np.ndarray:
    unknown = set(values) - set(names)
    if unknown:
        raise ValueError(f"not a driven Inspire joint: {sorted(unknown)}")
    array = np.empty(NUM_SLOTS, dtype=np.float64)
    for index, name in enumerate(names):
        if name in values:
            array[index] = float(values[name])
        elif missing is None:
            raise ValueError(f"no value for {name}; supply all {NUM_SLOTS} or pass missing=")
        else:
            array[index] = float(missing[index]) if np.ndim(missing) else float(missing)
    return array


def actuator_names(model) -> list:
    import mujoco

    names = []
    for index in range(model.nu):
        name = mujoco.mj_id2name(  # pyright: ignore[reportAttributeAccessIssue]
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,  # pyright: ignore[reportAttributeAccessIssue]
            index,
        )
        names.append((name or "").removesuffix("_motor"))
    return names
