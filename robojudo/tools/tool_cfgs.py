from pydantic import computed_field, model_validator

from robojudo.config import Config


class DoFConfig(Config):
    joint_names: list[str]
    default_pos: list[float] | None = None
    stiffness: list[float] | None = None
    damping: list[float] | None = None
    torque_limits: list[float] | None = None
    position_limits: list[list[float]] | None = None

    @computed_field
    @property
    def num_dofs(self) -> int:
        return len(self.joint_names)

    @property
    def prop_keys(self) -> list[str]:
        return [key for key in DoFConfig.model_fields if getattr(self, key) is not None]

    @model_validator(mode="after")
    def check_dof_properties(self):
        length = self.num_dofs
        for key in ("default_pos", "stiffness", "damping", "torque_limits", "position_limits"):
            value = getattr(self, key)
            if value is not None and len(value) != length:
                raise ValueError(f"{key} length {len(value)} does not match num_dofs {length}")

        if self.position_limits is not None:
            for index, limits in enumerate(self.position_limits):
                if len(limits) != 2:
                    raise ValueError(f"position_limits[{index}] must contain [min, max]")
                if limits[0] >= limits[1]:
                    raise ValueError(f"position_limits[{index}] min must be less than max")
        return self
