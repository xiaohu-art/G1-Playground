import logging
from abc import ABC, abstractmethod

from .pipeline_cfgs import PipelineCfg

logger = logging.getLogger(__name__)


class Pipeline(ABC):
    def __init__(self, cfg: PipelineCfg):
        self.cfg = cfg
        if cfg.device == "auto":
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = cfg.device
        logger.info("Using device: %s", self.device)

        self.dt = 1.0 / 50
        self.timestep = 0
        self.do_safety_check = cfg.do_safety_check

    @abstractmethod
    def step(self):
        raise NotImplementedError

    @abstractmethod
    def prepare(self):
        raise NotImplementedError

    def safety_check(self):
        return
