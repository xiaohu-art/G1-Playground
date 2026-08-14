# Controllers

Controllers are operator-input sources. They do not own the policy, environment lifecycle, or command targets.

| Deployment | Controller | Input source |
| --- | --- | --- |
| `sim` | `JoystickCtrl` | local Xbox-compatible controller |
| `real` | `UnitreeCtrl` | wireless-remote bytes carried by G1 LowState |

[`Controller`](../g1_playground/controller/base_ctrl.py) is a concrete queue consumer. `JoystickCtrl` and `UnitreeCtrl`
are sibling subclasses that only connect different producers to those queues; neither input source inherits from the
other. The deployment profile selects one `controller._target_`, and
[`scripts/pipeline.py`](../scripts/pipeline.py) constructs it with the shared `G1Env`.

## Read contract

`Controller.read()` drains queued updates and returns:

```python
control, shutdown_requested = controller.read()
# control == {"axes": {"LeftX": ..., "LeftY": ..., "RightX": ..., ...}}
# shutdown_requested is bool
```

The newest axis sample is retained. All pending button edges are consumed, and any pressed `A` edge makes the boolean
true for that read. The launcher checks it before policy inference and before the next target write. There is no command
list, trigger registry, controller manager, or post-step callback.

The policy consumes three normalized axes:

- `LeftY`: forward/backward velocity;
- `LeftX`: lateral velocity;
- `RightX`: yaw rate.

The policy owns dead-zone, sign, and maximum-command scaling. Start each run with the sticks released.

## JoystickCtrl

[`JoystickCtrl`](../g1_playground/controller/joystick_ctrl.py) starts the pygame input thread and feeds the base queues.
It is used by the two-process DDS simulation:

```bash
# Terminal 1: mandatory-viewer simulator
python scripts/simulate.py
# Terminal 2: policy client and local joystick
python scripts/pipeline.py deployment=sim
```

If no local controller is connected, the producer keeps all axes at zero. In that case `A` is unavailable; use
`Ctrl+C` to stop the client.

## UnitreeCtrl

[`UnitreeCtrl`](../g1_playground/controller/unitree_ctrl.py) attaches a wireless-remote parser to
`env.remote_controller_handler`; it does not open a desktop device or retain a second environment reference. It is used
only by `deployment=real`:

```bash
python scripts/pipeline.py deployment=real env.net_if=ROBOT_NIC
```

> [!CAUTION]
> `A` is a software shutdown request, not an emergency stop. Verify it with the robot secured and zero commanded
> velocity, while a trained operator continuously guards the independent hardware emergency stop.

Do not remap or suppress the `A` edge on the real profile without a dedicated safety review and new simulation evidence.
