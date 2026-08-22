import numpy as np
from omegaconf import DictConfig, OmegaConf


def compose_dof_config(robot_dof: DictConfig, policy_dof: DictConfig) -> DictConfig:
    """Reorder the policy DoF data into the robot runtime joint order."""

    robot_names = list(robot_dof.joint_names)
    policy_names = list(policy_dof.joint_names)
    if len(robot_names) != len(policy_names) or set(robot_names) != set(policy_names):
        raise ValueError("Robot and policy DoFs must be the same complete joint set")

    adapter = DoFAdapter(policy_names, robot_names)
    return OmegaConf.create(
        {
            "joint_names": robot_names,
            "default_pos": adapter.fit(policy_dof.default_pos).tolist(),
            "stiffness": adapter.fit(policy_dof.stiffness).tolist(),
            "damping": adapter.fit(policy_dof.damping).tolist(),
        }
    )


class DoFAdapter:
    """Project values from a named source layout into a target layout."""

    def __init__(self, src_joint_names, tar_joint_names):
        src_joint_names = tuple(src_joint_names)
        tar_joint_names = tuple(tar_joint_names)

        source_index = {name: index for index, name in enumerate(src_joint_names)}
        if len(source_index) != len(src_joint_names):
            raise ValueError("Source joint names must be unique")

        missing = sorted(set(tar_joint_names) - source_index.keys())
        if missing:
            raise ValueError(f"Target joints are missing from source: {missing}")

        self.source_size = len(src_joint_names)
        self.indices = np.asarray([source_index[name] for name in tar_joint_names], dtype=np.int64)
        self._can_scatter = len(set(tar_joint_names)) == len(tar_joint_names)

    def fit(self, source_values) -> np.ndarray:
        values = np.asarray(source_values)
        if values.shape != (self.source_size,):
            raise ValueError(f"Expected one value for each of {self.source_size} source joints, got {values.shape}")
        return values[self.indices]

    def scatter_into(self, target_values, out) -> np.ndarray:
        if not self._can_scatter:
            raise ValueError("Cannot scatter repeated target joints")
        values = np.asarray(target_values)
        out = np.asarray(out)
        if values.shape != self.indices.shape:
            raise ValueError(f"Expected one value for each of {self.indices.size} target joints, got {values.shape}")
        if out.shape != (self.source_size,):
            raise ValueError(f"Expected a {self.source_size}-joint destination, got {out.shape}")
        out[self.indices] = values
        return out
