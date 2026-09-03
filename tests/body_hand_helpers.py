from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from tests.config_helpers import compose_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
MOTION_PATH = REPO_ROOT / "assets/motions/clips/largebox_v02.npz"
MOTION_NAME = "sub16_largebox_013_v02"


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
            "object_pos_w": motions["object_pos_w"][start:stop].copy(),
            "object_quat_w": motions["object_quat_w"][start:stop].copy(),
            "contact_label": motions["contact_label"][start:stop].copy(),
            "fps": motions["fps"].copy(),
        }


def policy_cfg(**overrides):
    composed = compose_config("sim", "hoi=depth/largebox", config_name="run_loco_hoi_track")
    cfg = OmegaConf.create(OmegaConf.to_container(composed.hoi, resolve=True))
    for key, value in overrides.items():
        cfg[key] = value
    return cfg


def policy_data() -> dict:
    return OmegaConf.to_container(policy_cfg(), resolve=True)


def motion_cfg(**overrides):
    composed = compose_config("sim", "hoi=depth/largebox", config_name="run_loco_hoi_track")
    cfg = OmegaConf.create(OmegaConf.to_container(composed.motion, resolve=True))
    cfg.name = MOTION_NAME
    for key, value in overrides.items():
        cfg[key] = value
    return cfg


def inspire_cfg():
    return OmegaConf.load(CONFIG_DIR / "robot/inspire.yaml")


def body_joint_names():
    return list(OmegaConf.load(CONFIG_DIR / "robot/g1.yaml").dof.joint_names)


def hand_joint_names():
    return list(inspire_cfg().dof.joint_names)
