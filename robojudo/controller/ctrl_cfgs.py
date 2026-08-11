from robojudo.config import Config


class CtrlCfg(Config):
    ctrl_type: str
    triggers: dict[str, str] = {}


class JoystickCtrlCfg(CtrlCfg):
    ctrl_type: str = "JoystickCtrl"
    triggers: dict[str, str] = {"A": "[SHUTDOWN]"}


class UnitreeCtrlCfg(JoystickCtrlCfg):
    ctrl_type: str = "UnitreeCtrl"
