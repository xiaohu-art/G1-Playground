# Controllers

Controllers translate operator input into `ctrl_data` for `UnitreeWoGaitPolicy`. RoboJuDo exposes two controllers with the
same velocity-axis convention:

| Controller | Configuration | Input source | Used by |
| --- | --- | --- | --- |
| `JoystickCtrl` | `JoystickCtrlCfg` | Xbox-compatible controller | `g1` |
| `UnitreeCtrl` | `UnitreeCtrlCfg` | Unitree G1 remote | `g1_real` |

The base interface and controller aggregation live in
[`base_ctrl.py`](../robojudo/controller/base_ctrl.py) and
[`ctrl_manager.py`](../robojudo/controller/ctrl_manager.py).

## Common Data Contract

Both controllers provide an `axes` mapping and a list of button events. A representative payload is:

```json
{
  "axes": {
    "LeftX": 0.0,
    "LeftY": 0.0,
    "RightX": 0.0,
    "RightY": 0.0
  },
  "button_event": [
    {"type": "button", "name": "A", "pressed": true, "timestamp": 1758886189.68}
  ]
}
```

`UnitreeWoGaitPolicy` consumes these axes as follows:

- `LeftY`: forward and backward velocity
- `LeftX`: lateral velocity
- `RightX`: yaw rate
- `A`: emit `[SHUTDOWN]`

The local path expects SDL/pygame to provide stick values in `[-1, 1]`; it applies configured inversion and rounding but no
general clamp. The Unitree path likewise expects the wireless-remote protocol to provide values in that range. The policy
then remaps axes to configured velocity limits. Start every run with the sticks released.

## JoystickCtrl

[`joystick_ctrl.py`](../robojudo/controller/joystick_ctrl.py) reads a locally connected controller through the mappings in
[`controller/utils/joystick.py`](../robojudo/controller/utils/joystick.py). It is the input source for MuJoCo simulation.

Run the simulation with:

```bash
python scripts/run_pipeline.py -c g1
```

If axes are missing or reversed, stop the pipeline and correct the controller mapping before changing policy command
scales. Pressing `A` closes the simulation through the same shutdown command used on hardware.

## UnitreeCtrl

[`unitree_ctrl.py`](../robojudo/controller/unitree_ctrl.py) receives the G1 remote state through `UnitreeCppEnv`. It is used
only by `g1_real`; it does not open a separate desktop controller device.

```bash
python scripts/run_pipeline.py -c g1_real
```

> [!CAUTION]
> `A` is a software shutdown path. Test it with the robot secured and at zero command before any locomotion test, but never
> treat it as the only emergency stop. A trained operator must continuously hold the independent hardware emergency stop
> and keep the operating area clear.

Do not remap `A`, suppress `[SHUTDOWN]`, or add command-producing controller sources to `g1_real` without a dedicated safety
review and a new simulation validation record.
