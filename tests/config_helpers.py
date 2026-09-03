from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


def compose_config(
    deployment: str = "sim",
    *overrides: str,
    config_name: str = "run_pipeline",
    return_hydra_config: bool = False,
) -> DictConfig:
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR.as_posix()):
        return compose(
            config_name=config_name,
            overrides=[f"deployment={deployment}", *overrides],
            return_hydra_config=return_hydra_config,
        )
