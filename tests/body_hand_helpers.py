import os
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
MODEL_PATH = REPO_ROOT / "assets/models/body_hand_distill/largebox/policy.onnx"
MOTION_PATH = REPO_ROOT / "assets/motions/largebox_v02.npz"
MOTION_NAME = "sub16_largebox_013_v02"
BUNDLE_ENV = "G1_PLAYGROUND_BODY_HAND_BUNDLE"
DEFAULT_BUNDLE = Path(
    "/home/ubuntu/Desktop/IsaacSim51/g1_hoi_learning/logs/rsl_rl"
    "/g1_inspire_body_hand_distill/2026-08-21_22-15-00/exported"
)


def motion_data():
    with np.load(MOTION_PATH, allow_pickle=False) as motions:
        names = [str(name) for name in motions["motion_names"]]
        index = names.index(MOTION_NAME)
        lengths = np.asarray(motions["motion_lengths"], dtype=np.int64)
        start = int(lengths[:index].sum())
        stop = start + int(lengths[index])
        return {
            "joint_names": motions["joint_names"].copy(),
            "joint_pos": motions["joint_pos"][start:stop].copy(),
            "anchor_pos_w": motions["anchor_pos_w"][start:stop].copy(),
            "anchor_quat_w": motions["anchor_quat_w"][start:stop].copy(),
            "fps": motions["fps"].copy(),
        }


def policy_cfg(**overrides):
    cfg = OmegaConf.load(CONFIG_DIR / "policy/body_hand_distill_largebox.yaml")
    for key, value in overrides.items():
        cfg[key] = value
    return cfg


def policy_data() -> dict:
    return OmegaConf.to_container(policy_cfg(), resolve=True)


def motion_cfg(**overrides):
    cfg = OmegaConf.load(CONFIG_DIR / "run_body_hand.yaml").motion
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
    return candidate if (candidate / "golden_frame.npz").is_file() else None


def training_golden():
    bundle = training_bundle()
    return np.load(bundle / "golden_frame.npz", allow_pickle=True) if bundle is not None else None
