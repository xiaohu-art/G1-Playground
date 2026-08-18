from g1_playground.inspire import dof

__all__ = ["InspireHandEnv", "InspireHandState", "dof"]

_LAZY = {
    "InspireHandEnv": "g1_playground.inspire.hand_env",
    "InspireHandState": "g1_playground.inspire.hand_env",
}


def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(_LAZY[name]), name)
