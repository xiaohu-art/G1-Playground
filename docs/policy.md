# LeggedLabPolicy

[`LeggedLabPolicy`](../g1_playground/policy/leggedlab/leggedlab_policy.py) is the locomotion policy. It runs the Unitree
G1 29DoF recurrent velocity checkpoint from the [LeggedLab](https://github.com/Hellod035/LeggedLab) training framework
(checkpoint and deployment semantics transcribed from
[LeggedLabDeploy](https://github.com/Hellod035/LeggedLabDeploy) @ `93736b4`, BSD-3-Clause, verified on hardware by the
operator before integration) for both DDS simulation and hardware.

## Configuration and class contract

[`configs/policy/leggedlab_g1.yaml`](../configs/policy/leggedlab_g1.yaml) contains only values that vary as policy
data:

- checkpoint path;
- one `dof` block with policy joint order, default position, stiffness, and damping;
- action scale, observation scales, the command clip range, and observation/action clip magnitudes.

There is no separate observation/action DoF block. The policy constructs two `DoFAdapter`s around that single list: one
maps runtime state into policy order and the other maps targets back into G1 runtime order. The name-based mapping is
equivalent to LeggedLabDeploy's index-based `joint2motor_idx`, which is pinned by
[`tests/test_leggedlab_parity.py`](../tests/test_leggedlab_parity.py) against hardcoded reference values. The launcher
also uses the same `policy.dof` once to compose default position and gains for `G1Env`.

Checkpoint-shape invariants are code contracts, not tunable YAML:

```python
LeggedLabPolicy.FREQ == 50
LeggedLabPolicy.HISTORY_LENGTH == 1
LeggedLabPolicy.HISTORY_LAYOUT == (
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

## Recurrent state

The checkpoint is an LSTM exported with its hidden/cell state as buffers inside the TorchScript module, which is why a
single-frame observation suffices. Input normalization (`EmpiricalNormalization`) is baked into the graph, so the
observation scales in YAML are all `1.0` and no external normalization is applied.

`reset()` restores the checkpoint's saved recurrent state from a snapshot taken at load time, then replays the reference
launcher's zero-observation warm-up (`RESET_WARMUP_STEPS == 50`). The zero-input response of the LSTM does not converge
within that horizon, so the snapshot restore is what makes every control session start from the exact recurrent state a
fresh LeggedLabDeploy boot would control from.

On CPU the constructor pins `torch.set_num_threads(1)`: the tiny LSTM gains nothing from intra-op parallelism while the
default thread pool adds multi-ten-millisecond scheduling tails (p99 ~25 ms measured on a busy host) that blow the 50 Hz
frame budget and the DDS command watchdog.

## Observation and action

Each sample has 96 values:

| Field | Size | Meaning |
| --- | ---: | --- |
| angular velocity | 3 | scaled body angular velocity |
| projected gravity | 3 | body-frame gravity direction |
| velocity command | 3 | forward, lateral, yaw, clipped to `command_range` |
| joint position error | 29 | policy-order position minus default position |
| joint velocity | 29 | scaled policy-order velocity |
| previous action | 29 | preceding clipped raw model output |

`HISTORY_LENGTH == 1`, so the packed observation is this single frame:

```text
ang_vel | gravity | commands | dof_pos | dof_vel | actions = 96 values
```

The full observation is clipped to `+-clip_obs` (100) before inference, and the raw model output is clipped to
`+-clip_action` (100). `act(state, control)` stores that clipped raw output as the next previous-action field, applies
`action_scale` (0.25), adds the policy default position, maps the result to runtime order, and returns one 29-position
target. It does not return diagnostics or an extras dictionary. `standing_target` exposes the default pose in the same
runtime order. Only policy history and the recurrent state have a `reset()` lifecycle method.

The control axes become `[LeftY, -LeftX, -RightX]` with no deadzone, then are clipped elementwise to `command_range`
(currently `vx` in `[-0.4, 0.7] m/s`, `vy` in `[-0.4, 0.4] m/s`, yaw in `[-1.57, 1.57] rad/s`). The asymmetric forward
range and the absent deadzone mirror the reference deployment exactly.

## Model check

After replacing or re-exporting the checkpoint, verify its fixed shape:

```bash
python - <<'PY'
import torch

model = torch.jit.load("assets/models/leggedlab/g1_policy.pt", map_location="cpu")
action = model(torch.zeros(1, 96))
assert tuple(action.shape) == (1, 29), action.shape
print("checkpoint shape OK")
PY
```

A shape check is not a behavior or safety test. Revalidate standing preparation, zero command, command directions,
tilt shutdown, and operator shutdown with the complete mandatory-viewer `deployment=sim` path before hardware use.
