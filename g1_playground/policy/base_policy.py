from abc import ABC, abstractmethod

import numpy as np


class BasePolicy(ABC):
    FREQ: int

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.freq = self.FREQ
        self.dt = 1.0 / self.freq

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_observation(self, env_data, control_data) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def act(self, env_data, control_data) -> np.ndarray:
        raise NotImplementedError

    @property
    @abstractmethod
    def standing_target(self) -> np.ndarray:
        raise NotImplementedError
