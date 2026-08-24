import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
MODEL_PATH = REPO_ROOT / "assets/models/body_hand_distill/largebox/policy.onnx"
MOTION_PATH = REPO_ROOT / "assets/motions/largebox/sub16_largebox_039_v00.npz"
BUNDLE_ENV = "G1_PLAYGROUND_BODY_HAND_BUNDLE"
DEFAULT_BUNDLE = REPO_ROOT / "hoi/logs/rsl_rl/g1_inspire_hoi_depth/proprio-no-depth-v1/exported"


def session(path=None):
    return ort.InferenceSession(str(path or MODEL_PATH), providers=["CPUExecutionProvider"])


def motion_data():
    return np.load(MOTION_PATH, allow_pickle=False)


def policy_cfg(**overrides):
    cfg = OmegaConf.load(CONFIG_DIR / "policy/body_hand_distill_largebox.yaml")
    for key, value in overrides.items():
        cfg[key] = value
    return cfg


def policy_data() -> dict:
    return OmegaConf.to_container(policy_cfg(), resolve=True)


def motion_cfg(**overrides):
    cfg = OmegaConf.load(CONFIG_DIR / "motion/largebox_039_v00.yaml")
    for key, value in overrides.items():
        cfg[key] = value
    return cfg


def inspire_cfg():
    return OmegaConf.load(CONFIG_DIR / "robot/inspire.yaml")


def body_joint_names():
    return list(OmegaConf.load(CONFIG_DIR / "robot/g1.yaml").dof.joint_names)


def hand_joint_names():
    return list(inspire_cfg().dof.joint_names)


def training_bundle() -> Path | None:
    candidate = Path(os.environ.get(BUNDLE_ENV, DEFAULT_BUNDLE))
    return candidate if (candidate / "parity_trace.npz").is_file() else None


def parity_trace():
    bundle = training_bundle()
    return np.load(bundle / "parity_trace.npz", allow_pickle=True) if bundle is not None else None
