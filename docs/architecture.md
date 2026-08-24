# G1-Playground Sim2Sim / Sim2Real Architecture

G1-Playground deliberately has one policy-facing environment boundary:

```text
sim:  JoystickCtrl -> LeggedLabPolicy -> G1Env -> G1DdsControlEndpoint <-> HG DDS
                                                                              <-> G1DdsRobotEndpoint
                                                                                  <-> G1MujocoDdsServer
                                                                                      <-> G1MujocoBackend

real: UnitreeCtrl  -> LeggedLabPolicy -> G1Env -> G1DdsControlEndpoint <-> HG DDS <-> physical G1
```

The simulator is a separate DDS peer. `G1Env` never selects a backend or calls MuJoCo, so simulation and hardware run the
same client lifecycle, state validation, policy inference, pacing, shutdown checks, and LowCmd writer behavior. Both
profiles construct `unitree_cpp.G1DdsControlEndpoint` inside `G1Env`. Only `scripts/simulate.py` constructs
`unitree_cpp.G1DdsRobotEndpoint`; the real path talks directly to the physical G1 and does not depend on the simulated
robot endpoint.

## Deployment composition

| Profile | Controller | Endpoint | Command activation |
| --- | --- | --- | --- |
| `sim` (default) | `JoystickCtrl` | Domain 1 / `lo` | no MotionSwitcher |
| `real` | `UnitreeCtrl` | Domain 0 / explicit robot NIC | bounded MotionSwitcher release required |

The root configuration owns the one `G1Env` target. Deployment profiles contain only five environment values
(`domain_id`, `net_if`, LowCmd topic, LowState topic, MotionSwitcher requirement) and one controller target. The domain
number does not carry simulator/hardware semantics.

Configuration ownership is intentionally narrow:

| Source | Owned values | Runtime consumers |
| --- | --- | --- |
| `robot/g1.yaml` | XML, runtime joint order, torque limits | DoF composition, simulator backend/server |
| `policy/leggedlab_g1.yaml` | checkpoint, one DoF order/pose/gains, scales, command clip | policy, DoF composition |
| `LeggedLabPolicy` class | 50 Hz, single-frame history, field layout | policy and launcher via `policy.dt` |
| `deployment/*.yaml` | endpoint/topics/motion/controller | `G1Env` and controller construction |

There is no configured position limit. Simulator torque limits remain in robot config; hardware commands do not consume
them. There is also no robot name, joint map, policy observation/action DoF duplication, or deployment control-period
copy.

The simulator has no Hydra root. Its launcher directly loads `robot/g1.yaml` for XML/limits and `deployment/sim.yaml` for
the endpoint/topics. `G1MujocoBackend` owns its 1 ms default timestep, `G1MujocoDdsServer` owns its 0.1-second default
watchdog, and the launcher owns the fixed 60 Hz viewer constant.

## Composition and runtime contracts

[`scripts/pipeline.py`](../scripts/pipeline.py) is the only policy composition root:

```python
dof = compose_dof_config(cfg.robot.dof, cfg.policy.dof)
policy = LeggedLabPolicy(cfg.policy, device=cfg.device, dof_cfg=dof)
env = instantiate(cfg.env, dof_cfg=dof, control_dt=policy.dt)
controller = instantiate(cfg.controller, env=env)
```

The composed DoF contains runtime joint names plus policy default position/stiffness/damping in runtime order. It is
created once; no component mutates gains after construction.

The three public data operations are equally small:

- `G1Env.read() -> G1State`: one native read and one frozen snapshot whose four arrays are detached/read-only;
- `Controller.read() -> (control, shutdown_requested)`: latest axes plus a boolean set by a pressed `A` edge;
- `policy.act(state, control) -> target`: one 29-position target in G1 runtime order.

`Controller` is a concrete queue consumer. `JoystickCtrl` and `UnitreeCtrl` are sibling producers; the Unitree variant
only attaches its parser to `env.remote_controller_handler`. `G1Env` has no reset, fallen state, cached-state API, Python
configuration box, joint map, or recover hook. `read_frame()` applies the pure
`g1_playground.utils.math.is_upright(state.base_quat)` predicate to the same snapshot that will be passed to the policy.

## Shared client lifecycle

Both profiles execute:

```text
self_check
  -> 10 dry read/control/tilt/inference frames (no writes)
  -> one preflight read/control/tilt check
  -> activate command transport
  -> 3 s measured-to-standing ramp
  -> reset policy history
  -> 5 s zero-command policy blend
  -> paced active loop
  -> finally shutdown
```

There is no environment/controller reset step. During ramp, blend, and the active loop, every target write is preceded by
a fresh state read, controller read, shutdown check, and pure tilt check. The policy is the sole owner of history and
therefore the only reset at the ramp/blend boundary.

The policy's model layout is fixed by class constants: one frame of
`ang_vel[3], gravity[3], commands[3], dof_pos[29], dof_vel[29], actions[29]`, yielding 96 inputs and 29 outputs (the
checkpoint is recurrent; its LSTM state lives inside the model). Its
50 Hz period (`policy.dt == 0.02`) is injected into `G1Env`, which passes it to native freshness/writer behavior.

## MuJoCo DDS server

The standalone process has three layers:

```text
scripts/simulate.py                     composition + mandatory viewer
    -> G1MujocoDdsServer               DDS/PD/watchdog/pacing
        +-> G1DdsRobotEndpoint         native simulated-robot HG endpoint
        +-> G1MujocoBackend            locked MuJoCo model/data/support step
```

[`G1DdsRobotEndpoint`](../third_party/unitree_cpp/src/g1_dds_robot_endpoint.cpp) is deliberately small: it validates and
snapshots LowCmd, then builds and publishes LowState from simulator snapshots. It knows nothing about MuJoCo, policy
inference, PD torque, or the viewer. `G1MujocoDdsServer` owns those simulation-side concerns and receives the endpoint as
a dependency.

The server holds the previous `MujocoState`. A tick obtains one coherent LowCmd, computes and clips PD torque using that
state, then calls `backend.step(torque, support_scale)` exactly once. Under one lock the backend updates elastic support,
steps MuJoCo, and returns the only new detached/read-only state. That snapshot is both published and retained for the next
tick. There is no per-tick `backend.read()` and no independent support mutation.

`G1MujocoBackend.timestep` is the physics-period owner, and the server derives its deadline interval from it. The launcher
uses the backend's 1 ms default and does not pass a second timestep into the server.

Python populates joint q/dq/torque, base quaternion, and gyroscope in LowState. The native snapshot's defaults provide
zero accelerometer/RPY/remote fields. The launcher explicitly constructs the native endpoint dictionary and supplies
fixed HG `mode_machine=5`; user YAML contains no mode field.

Before the first valid command the backend advances with zero actuator torque and full elastic support. Support stays
full for 3 seconds from the first command and releases over 5 seconds. A command older than the server's 0.1-second
default timeout terminates the simulator. Torque is clipped only here, using `robot.dof.torque_limits`.

The public launcher always opens the official MuJoCo viewer. Its main thread copies coherent backend data into separate
render data; the background server keeps the absolute physics deadline. Viewer input never mutates physics, and teardown
preserves stop-event, worker join, robot-endpoint close, then viewer close behavior. Tests may instantiate the viewer-free
lower layers directly.

## Native DDS boundaries

| Native class | Constructed by | DDS role | Deployments |
| --- | --- | --- | --- |
| `G1DdsControlEndpoint` | `G1Env` in the policy process | receives LowState; publishes LowCmd after activation | `sim` and `real` |
| `G1DdsRobotEndpoint` | `scripts/simulate.py` | receives LowCmd; publishes LowState built from MuJoCo snapshots | `sim` only |

The implementations live in
[`g1_dds_control_endpoint.cpp`](../third_party/unitree_cpp/src/g1_dds_control_endpoint.cpp) and
[`g1_dds_robot_endpoint.cpp`](../third_party/unitree_cpp/src/g1_dds_robot_endpoint.cpp). The physical G1 supplies the
robot-side DDS endpoint in a real deployment. Consequently, `deployment=real` depends on `G1DdsControlEndpoint`, not the
simulation-only `G1DdsRobotEndpoint`. The native endpoint classes neither construct nor call each other; both
independently use the neutral CRC and endpoint guards in
[`dds_utils.hpp`](../third_party/unitree_cpp/src/dds_utils.hpp).

Construction is receive-only. Native LowState acceptance validates CRC, expected mode, finite motor/IMU values, strictly
advancing tick with wrap, and freshness. `self_check()` requires a nonzero ready tick. Invalid packets neither overwrite
the last good state nor refresh its timestamp.

Activation creates LowCmd transport directly in simulation, or only after MotionSwitcher release on hardware. Targets
must be exactly 29 DDS-float-representable values. When the application command or cached LowState exceeds
`max(5 * control_dt, 0.1 s)`, native code sends one damping command, clears the command buffer, and refuses continued
stale operation. Shutdown is idempotent.

## Safety and evidence boundary

- The hardware path has no configured position clamp or torque-limit enforcement.
- Simulator torque clipping, support, watchdog, client damping, tilt gate, and `A` are software defenses, not certified
  emergency stops.
- No DDS simulator reset/debug channel exists; a fall stops the policy client and requires process restart.
- Manual mandatory-viewer behavior checks and an independently guarded hardware emergency stop remain required before a
  real run.
- Frozen pre-DDS fixtures and Phase 0-5 records are historical evidence. Names such as `G1DdsSimServer` and
  `dds_sim_server.cpp` describe the API at that time and must not be rewritten to resemble the current API.

See the [G1Env contract](environment.md), [policy contract](policy.md), [controller contract](controller.md), and
[real-robot setup](unitree_setup.md).
