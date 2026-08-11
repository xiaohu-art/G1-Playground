from robojudo.utils.module_registry import Registry

from .base_env import Environment
from .env_cfgs import EnvCfg

env_registry = Registry(package="robojudo.environment", base_class=Environment)

__all__ = [
    "Environment",
    "EnvCfg",
    "env_registry",
]


def __getattr__(name: str) -> type[Environment]:
    try:
        env_class = env_registry.get(name)
    except Exception as e:
        raise AttributeError(f"module {__name__} has no attribute {name}") from e
    globals()[name] = env_class
    return env_class


env_registry.add("MujocoEnv", ".mujoco_env")
env_registry.add("UnitreeCppEnv", ".unitree_cpp_env")
