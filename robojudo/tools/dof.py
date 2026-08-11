import logging

import numpy as np

from .tool_cfgs import DoFConfig

logger = logging.getLogger(__name__)


def merge_dof_cfgs(base_cfg: DoFConfig, override_cfg: DoFConfig) -> DoFConfig:
    """
    Merge two DoFConfig objects, with override_cfg taking precedence over base_cfg.
    Only non-None values in override_cfg will replace those in base_cfg.

    Returns a new DoFConfig object.
    """
    if not isinstance(base_cfg, DoFConfig) or not isinstance(override_cfg, DoFConfig):
        raise ValueError("Both base_cfg and override_cfg must be instances of DoFConfig")

    merged_cfg = base_cfg.model_copy()

    dof_adapter = DoFAdapter(src_joint_names=override_cfg.joint_names, tar_joint_names=merged_cfg.joint_names)
    for key in override_cfg.prop_keys:
        value_override = getattr(override_cfg, key)
        if key in ["joint_names"] or value_override is None:
            continue
        if key not in merged_cfg.prop_keys:
            raise KeyError(f"Key {key} not in dof_cfg, cannot override")

        value_raw = getattr(merged_cfg, key)
        value_override_fitted = dof_adapter.fit(value_override, dim=0, template=value_raw).tolist()
        setattr(merged_cfg, key, value_override_fitted)
        logger.debug(f"[DoF] override {key} with {value_override_fitted}")
    return merged_cfg


class DoFAdapter:
    def __init__(self, src_joint_names, tar_joint_names):
        self.src_joint_names = src_joint_names
        self.tar_joint_names = tar_joint_names
        self.src_len = len(src_joint_names)
        self.tar_len = len(tar_joint_names)
        if self.src_len != self.tar_len or set(src_joint_names) != set(tar_joint_names):
            raise ValueError("DoFAdapter only supports reordering the complete 29-joint G1 set")
        if len(set(src_joint_names)) != self.src_len:
            raise ValueError("Joint names must be unique")

        self.src_indices = list(range(self.src_len))
        self.tar_indices = [tar_joint_names.index(name) for name in src_joint_names]

    def fit(self, data, dim=-1, template=None) -> np.ndarray:
        if type(data) is not np.ndarray:
            data = np.asarray(data)

        assert data.shape[dim] == self.src_len, (
            f"Data shape {data.shape} does not match src length {self.src_len} at dim {dim}"
        )

        new_shape = list(data.shape)
        new_shape[dim] = self.tar_len

        if template is None:
            new_data = np.zeros(new_shape, dtype=data.dtype)
        else:
            if type(template) is not np.ndarray:
                template = np.asarray(template, dtype=data.dtype)
            new_data = template.copy()
            assert new_data.shape == tuple(new_shape), (
                f"Template shape {new_data.shape} does not match target shape {new_shape}"
            )

        if dim == -1:
            new_data[..., self.tar_indices] = data[..., self.src_indices]
        else:
            indices = [slice(None)] * len(data.shape)
            indices[dim] = self.tar_indices
            src_indices = [slice(None)] * len(data.shape)
            src_indices[dim] = self.src_indices
            new_data[tuple(indices)] = data[tuple(src_indices)]

        return new_data
