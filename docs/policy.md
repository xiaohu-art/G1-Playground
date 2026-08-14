# UnitreeWoGaitPolicy

[`UnitreeWoGaitPolicy`](../g1_playground/policy/unitree_policy.py) is the only policy. It runs the Unitree G1 29DoF
velocity checkpoint from [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab) for both DDS simulation and
hardware.

## Configuration and class contract

[`configs/policy/unitree_wo_gait.yaml`](../configs/policy/unitree_wo_gait.yaml) contains only values that vary as policy
data:

- checkpoint path;
- one `dof` block with policy joint order, default position, stiffness, and damping;
- action scale, angular/joint-velocity observation scales, and maximum commands.

There is no separate observation/action DoF block. The policy constructs two `DoFAdapter`s around that single list: one
maps runtime state into policy order and the other maps targets back into G1 runtime order. The launcher also uses the
same `policy.dof` once to compose default position and gains for `G1Env`.

Checkpoint-shape invariants are code contracts, not tunable YAML:

```python
UnitreeWoGaitPolicy.FREQ == 50
UnitreeWoGaitPolicy.HISTORY_LENGTH == 5
UnitreeWoGaitPolicy.HISTORY_LAYOUT == (
    ("ang_vel", 3),
    ("gravity", 3),
    ("commands", 3),
    ("dof_pos", 29),
    ("dof_vel", 29),
    ("actions", 29),
)
```

`policy.dt` is therefore `1 / 50 = 0.02` seconds and is injected into `G1Env` by the launcher. Moving frequency, history
length, or field layout into deployment configuration would create a second source of truth for the exported model.

## Observation and action

Each sample has 96 values:

| Field | Size | Meaning |
| --- | ---: | --- |
| angular velocity | 3 | scaled body angular velocity |
| projected gravity | 3 | body-frame gravity direction |
| velocity command | 3 | forward, lateral, yaw |
| joint position error | 29 | policy-order position minus default position |
| joint velocity | 29 | scaled policy-order velocity |
| previous action | 29 | preceding raw model output |

Five samples are packed field-major, not as five adjacent frames:

```text
ang_vel[5] | gravity[5] | commands[5] |
dof_pos[5] | dof_vel[5] | actions[5] = 480 values
```

The TorchScript model returns 29 raw actions. `act(state, control)` stores that raw output as the next previous-action
field, applies `action_scale`, adds the policy default position, maps the result to runtime order, and returns one
29-position target. It does not return diagnostics or an extras dictionary. `standing_target` exposes the default pose in
the same runtime order. Only policy history has a `reset()` lifecycle method.

The control axes become `[LeftY, -LeftX, -RightX]`; magnitudes below `0.04` are zeroed, then values are multiplied by
`max_cmd` (currently `0.8 m/s`, `0.5 m/s`, `1.57 rad/s`).

## Model check

After replacing or re-exporting the checkpoint, verify its fixed shape:

```bash
python - <<'PY'
import torch

model = torch.jit.load("assets/models/g1/unitree/policy_wo_gait.pt", map_location="cpu")
action = model(torch.zeros(1, 480))
assert tuple(action.shape) == (1, 29), action.shape
print("checkpoint shape OK")
PY
```

A shape check is not a behavior or safety test. Revalidate standing preparation, zero command, command directions,
tilt shutdown, and operator shutdown with the complete mandatory-viewer `deployment=sim` path before hardware use.
