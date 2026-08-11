import numpy as np

from robojudo.environment.utils.mujoco_viz import MujocoVisualizer
from robojudo.policy import Policy, policy_registry
from robojudo.policy.policy_cfgs import UnitreeWoGaitPolicyCfg
from robojudo.utils.util_func import command_remap, get_gravity_orientation


@policy_registry.register
class UnitreeWoGaitPolicy(Policy):
    cfg_policy: UnitreeWoGaitPolicyCfg

    def __init__(self, cfg_policy: UnitreeWoGaitPolicyCfg, device: str):
        super().__init__(cfg_policy=cfg_policy, device=device)
        self.obs_scales = cfg_policy.obs_scales
        self.max_cmd = cfg_policy.max_cmd
        self.commands_map = cfg_policy.commands_map
        self.reset()

    def reset(self):
        self.last_action.fill(0.0)
        default_history = [np.zeros(dim, dtype=np.float32) for dim in self.cfg_policy.history_obs_dims.values()]
        self._init_history(default_history)

    def post_step_callback(self, commands: list[str] | None = None):
        return

    def _get_commands(self, ctrl_data):
        commands = np.zeros(3)
        for key in ctrl_data.keys():
            if key not in ["JoystickCtrl", "UnitreeCtrl"]:
                continue
            axes = ctrl_data[key]["axes"]
            commands[0] = command_remap(axes["LeftY"], self.commands_map[0])
            commands[1] = command_remap(axes["LeftX"], self.commands_map[1])
            commands[2] = command_remap(axes["RightX"], self.commands_map[2])
            break
        return commands

    def get_observation(self, env_data, ctrl_data):
        commands = self._get_commands(ctrl_data)
        gravity_orientation = get_gravity_orientation(env_data.base_quat)
        obs_current = [
            env_data.base_ang_vel * self.obs_scales.ang_vel,
            gravity_orientation * self.obs_scales.gravity,
            commands * self.obs_scales.command * self.max_cmd,
            (env_data.dof_pos - self.obs_default_pos) * self.obs_scales.dof_pos,
            env_data.dof_vel * self.obs_scales.dof_vel,
            self.last_action,
        ]
        self.history_buf.append(obs_current)
        history_list = [np.concatenate(items, axis=0) for items in zip(*self.history_buf, strict=True)]
        return np.concatenate(history_list, axis=0), {"commands": commands}

    def debug_viz(self, visualizer: MujocoVisualizer, env_data, ctrl_data, extras):
        base_pos = env_data["base_pos"]
        base_quat = env_data["base_quat"]
        command_x, command_y, command_yaw = extras["commands"]

        visualizer.draw_arrow(
            base_pos,
            base_quat,
            [command_x, 0, 0],
            color=[1, 0, 0, 1],
            scale=2,
            horizontal_only=True,
            id=0,
        )
        visualizer.draw_arrow(
            base_pos,
            base_quat,
            [0, command_y, 0],
            color=[0, 1, 0, 1],
            scale=2,
            horizontal_only=True,
            id=1,
        )
        visualizer.draw_arrow(
            base_pos + np.array([0.0, 0.0, 0.6]),
            base_quat,
            [0, command_yaw, 0],
            color=[1, 1, 1, 1],
            scale=2,
            horizontal_only=True,
            id=2,
        )
