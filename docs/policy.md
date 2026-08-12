# UnitreeWoGaitPolicy

`UnitreeWoGaitPolicy` is the repository's only policy. It runs the Unitree G1 29DoF velocity controller exported from
[unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab). The checkpoint omits a periodic gait-phase input, so
a zero velocity command does not force the robot to continue stepping.

## Implementation

- Policy implementation: [`unitree_policy.py`](../robojudo/policy/unitree_policy.py)
- Shared policy configuration: [`policy_cfgs.py`](../robojudo/policy/policy_cfgs.py)
- G1 joint order and checkpoint configuration:
  [`g1_unitree_policy_cfg.py`](../robojudo/config/g1/policy/g1_unitree_policy_cfg.py)
- Checkpoint: `assets/models/g1/unitree/policy_wo_gait.pt`

Both the `g1` simulation configuration and the `g1_real` hardware configuration select this policy.

## Observation and Action Contract

Each control frame contributes 96 observation values:

| Field | Size | Description |
| --- | ---: | --- |
| Base angular velocity | 3 | Scaled body-frame angular velocity |
| Projected gravity | 3 | Gravity direction in the body frame |
| Velocity command | 3 | Forward, lateral, and yaw commands |
| Joint position error | 29 | Current position minus the policy default pose |
| Joint velocity | 29 | Scaled joint velocity |
| Previous action | 29 | Previous output after optional beta smoothing, before clipping and action scaling |

The policy stores five samples of each field, producing a 480-value model input. Packing is field-major—not five adjacent
96-value frames: `ang_vel[5] | gravity[5] | commands[5] | dof_pos[5] | dof_vel[5] | actions[5]`. The TorchScript model
returns 29 actions. After action scaling, the pipeline adds the policy default pose to form joint position targets.

The policy and environment may list joints in different orders, but both lists must contain the same 29 G1 joints. The
pipeline adapter is used only to reorder values; missing or uncontrolled joints are not supported.

## Velocity Commands

`JoystickCtrl` in simulation and `UnitreeCtrl` on hardware provide the same normalized axes:

- `LeftY`: forward and backward velocity
- `LeftX`: lateral velocity
- `RightX`: yaw rate

The default maximum command magnitudes are `0.8 m/s`, `0.5 m/s`, and `1.57 rad/s`, respectively. Release both sticks to
request zero velocity. Axis remapping and limits are defined by `UnitreeWoGaitPolicyCfg`; change them only after simulation
validation.

## Model Check

Use this shape check after replacing or re-exporting the checkpoint:

```bash
python - <<'PY'
import torch

model = torch.jit.load("assets/models/g1/unitree/policy_wo_gait.pt", map_location="cpu")
action = model(torch.zeros(1, 480))
assert tuple(action.shape) == (1, 29), action.shape
print("checkpoint shape OK")
PY
```

A successful shape check does not establish safe behavior. Run the complete `g1` MuJoCo configuration and verify standing,
zero-command behavior, command directions, joint limits, and shutdown before using `g1_real`.
