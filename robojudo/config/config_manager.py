from robojudo.config import cfg_registry


class ConfigManager:
    def __init__(self, config_name: str):
        self.config_name = config_name
        self.cfg = self.parse_config()

    def get_cfg(self):
        return self.cfg

    def parse_config(self):
        cfg_class = cfg_registry.get(self.config_name)
        return cfg_class()
