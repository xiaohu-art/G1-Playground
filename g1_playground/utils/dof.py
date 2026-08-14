import numpy as np
from omegaconf import DictConfig, OmegaConf


def compose_dof_config(robot_dof: DictConfig, policy_dof: DictConfig) -> DictConfig:
    """Reorder the policy DoF data into the robot runtime joint order."""

    adapter = DoFAdapter(src_joint_names=policy_dof.joint_names, tar_joint_names=robot_dof.joint_names)
    return OmegaConf.create(
        {
            "joint_names": list(robot_dof.joint_names),
            "default_pos": adapter.fit(policy_dof.default_pos).tolist(),
            "stiffness": adapter.fit(policy_dof.stiffness).tolist(),
            "damping": adapter.fit(policy_dof.damping).tolist(),
        }
    )


class DoFAdapter:
    def __init__(self, src_joint_names, tar_joint_names):
        src_joint_names = list(src_joint_names)
        tar_joint_names = list(tar_joint_names)
        if len(src_joint_names) != 29 or len(tar_joint_names) != 29 or set(src_joint_names) != set(tar_joint_names):
            raise ValueError("DoFAdapter only supports reordering the complete 29-joint G1 set")
        if len(set(src_joint_names)) != 29:
            raise ValueError("Joint names must be unique")
        self.indices = [src_joint_names.index(name) for name in tar_joint_names]

    def fit(self, data) -> np.ndarray:
        data = np.asarray(data)
        if data.shape != (29,):
            raise ValueError(f"DoFAdapter requires one value for each G1 joint, got shape {data.shape}")
        return data[self.indices]
