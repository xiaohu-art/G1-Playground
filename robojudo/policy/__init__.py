from robojudo.utils.module_registry import Registry

from .base_policy import Policy
from .policy_cfgs import PolicyCfg

policy_registry = Registry(package="robojudo.policy", base_class=Policy)

__all__ = [
    "Policy",
    "PolicyCfg",
    "policy_registry",
]


def __getattr__(name: str) -> type[Policy]:
    try:
        policy_class = policy_registry.get(name)
    except Exception as e:
        raise AttributeError(f"module {__name__} has no attribute {name}") from e
    globals()[name] = policy_class
    return policy_class


policy_registry.add("UnitreeWoGaitPolicy", ".unitree_policy")
