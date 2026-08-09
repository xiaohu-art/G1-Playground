from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import yaml

from g1_playground.action import JointAction
from g1_playground.policies.params import PolicyParams
from g1_playground.state import G1State


class Policy(ABC):
    """Strategy interface for deciding the next action from the current state."""

    DEFAULT_CONFIG_PATH: ClassVar[Path | None] = None

    def _init_from_config(self, cfg: dict | None) -> dict:
        """Load the ``policy:`` subtree and build ``self.params``.
        """
        if cfg is None:
            if self.DEFAULT_CONFIG_PATH is None:
                raise ValueError(f"{type(self).__name__} defines no DEFAULT_CONFIG_PATH")
            with open(self.DEFAULT_CONFIG_PATH) as f:
                cfg = yaml.safe_load(f)["policy"]
        self.params = PolicyParams.from_dict(cfg)
        return cfg

    @property
    @abstractmethod
    def dt(self) -> float:
        """Policy step period in seconds (e.g. 0.1 for 10 Hz)."""

    @abstractmethod
    def reset(self, state: G1State) -> None:
        """Called once after the first state arrives, before any step()."""

    @abstractmethod
    def step(self, state: G1State) -> JointAction:
        """Compute the next action from the current state."""