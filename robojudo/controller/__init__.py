from robojudo.utils.module_registry import Registry

from .base_ctrl import Controller
from .ctrl_cfgs import CtrlCfg
from .ctrl_manager import CtrlManager

ctrl_registry = Registry(package="robojudo.controller", base_class=Controller)

__all__ = [
    "Controller",
    "CtrlCfg",
    "CtrlManager",
    "ctrl_registry",
]


def __getattr__(name: str) -> type[Controller]:
    try:
        ctrl_class = ctrl_registry.get(name)
    except Exception as e:
        raise AttributeError(f"module {__name__} has no attribute {name}") from e
    globals()[name] = ctrl_class
    return ctrl_class


ctrl_registry.add("JoystickCtrl", ".joystick_ctrl")
ctrl_registry.add("UnitreeCtrl", ".unitree_ctrl")
