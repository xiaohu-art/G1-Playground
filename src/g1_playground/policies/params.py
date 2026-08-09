from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from g1_playground.joints import JOINT_BFS_NAMES, JOINT_DFS_NAMES
from g1_playground.utils.strings import resolve_param

NUM_JOINTS = len(JOINT_DFS_NAMES)
REPO_ROOT = Path(__file__).resolve().parents[3]

JOINT_ORDERS = {"bfs": JOINT_BFS_NAMES, "dfs": JOINT_DFS_NAMES}


@dataclass(frozen=True)
class PolicyParams:
    model_path: Path
    policy_joint_names: list[str]
    action_scale: np.ndarray
    default_pos: np.ndarray
    kp: np.ndarray
    kd: np.ndarray

    @classmethod
    def from_dict(cls, cfg: dict) -> "PolicyParams":
        """Build from the ``policy:`` subtree (plain dict — for Hydra pass
        ``OmegaConf.to_container(cfg.policy, resolve=True)``)."""

        def _resolve(block: dict) -> np.ndarray:
            values = dict(block.get("values") or {})
            return resolve_param(values, JOINT_DFS_NAMES, default=float(block["default"]))

        model_path = Path(cfg["model"])
        if not model_path.is_absolute():
            model_path = REPO_ROOT / model_path

        params = cls(
            model_path=model_path,
            policy_joint_names=list(JOINT_ORDERS[cfg["policy_joint_order"]]),
            action_scale=_resolve(cfg["action_scale"]),
            default_pos=_resolve(cfg["default_pos"]),
            kp=_resolve(cfg["kp"]),
            kd=_resolve(cfg["kd"]),
        )
        assert len(params.policy_joint_names) == NUM_JOINTS
        return params

    @classmethod
    def load(cls, path: Path) -> "PolicyParams":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f)["policy"])
