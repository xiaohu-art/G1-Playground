from robojudo.environment.env_cfgs import MujocoEnvCfg

from .g1_env_cfg import G1EnvCfg


class G1MujocoEnvCfg(G1EnvCfg, MujocoEnvCfg):
    env_type: str = MujocoEnvCfg.model_fields["env_type"].default
    is_sim: bool = MujocoEnvCfg.model_fields["is_sim"].default
