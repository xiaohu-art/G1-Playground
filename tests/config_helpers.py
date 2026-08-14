import importlib.util
from pathlib import Path
from types import ModuleType

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
PIPELINE_LAUNCHER = REPO_ROOT / "scripts/pipeline.py"


def compose_config(deployment: str = "sim", *overrides: str, return_hydra_config: bool = False) -> DictConfig:
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR.as_posix()):
        return compose(
            config_name="run_pipeline",
            overrides=[f"deployment={deployment}", *overrides],
            return_hydra_config=return_hydra_config,
        )


def asset_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def load_pipeline_launcher() -> ModuleType:
    spec = importlib.util.spec_from_file_location("g1_playground_test_pipeline", PIPELINE_LAUNCHER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load launcher from {PIPELINE_LAUNCHER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
