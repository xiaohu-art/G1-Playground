# G1Env

[`g1_playground/g1_env.py`](../g1_playground/g1_env.py) is the only policy-facing environment. It exposes one G1 state
snapshot, accepts one 29-joint position target, and adapts that contract to the native Unitree HG DDS control endpoint.

```text
sim:  Policy -> G1Env -> G1DdsControlEndpoint <-> DDS <-> G1DdsRobotEndpoint
                                                              <-> G1MujocoDdsServer <-> G1MujocoBackend
real: Policy -> G1Env -> G1DdsControlEndpoint <-> DDS <-> physical G1
```

There is no environment base class, backend selector, direct `MujocoEnv`, or separate Unitree C++ environment wrapper.
`domain_id` and `net_if` describe a DDS endpoint; they do not select simulator versus hardware behavior.
`G1Env` constructs `unitree_cpp.G1DdsControlEndpoint` for both profiles. The native `G1DdsRobotEndpoint` exists only in
the standalone simulator; `deployment=real` communicates with the physical G1 and never constructs the simulated robot
endpoint.

## Construction and ownership

The root [`configs/run_pipeline.yaml`](../configs/run_pipeline.yaml) owns the single `G1Env` `_target_`. A deployment
profile contributes only endpoint/topics, `motion_switcher_required`, and its controller target. The launcher composes
policy pose/gains into runtime joint order, constructs the policy, then injects its period into the environment:

```python
dof = compose_dof_config(cfg.robot.dof, cfg.policy.dof)
policy = LeggedLabPolicy(cfg.policy, device=cfg.device, dof_cfg=dof)
env = instantiate(cfg.env, dof_cfg=dof, control_dt=policy.dt)
controller = instantiate(cfg.controller, env=env)
```

Thus the 20 ms client period has one policy owner; deployment YAML does not duplicate it. `G1Env` consumes only the
composed joint names/stiffness/damping it needs. It has no position-limit configuration, joint-map option, Python `Box`
configuration wrapper, reset method, fallen-state flag, or cached-state property.

## State and target contract

Every `read()` pulls exactly one native LowState and returns a frozen `G1State` containing:

- `dof_pos`: 29 joint positions;
- `dof_vel`: 29 joint velocities;
- `base_quat`: `[x, y, z, w]`;
- `base_ang_vel`: three angular-velocity values.

All four NumPy arrays are detached copies and read-only. `scripts/pipeline.py` passes the same snapshot to
`is_upright(state.base_quat)` and `policy.act(state, control)`; the tilt check does not ask the environment for another
state.

`step(target)` forwards exactly one 29-position target to native code. The live hardware path does not clamp target
positions. `robot.dof.torque_limits` is consumed by the MuJoCo DDS server only and is not a hardware command barrier.

## Lifecycle and DDS behavior

Construction creates receive resources only. `self_check()` waits for a valid LowState; command transport is created by
`activate_commands()` after dry inference and the final preflight check. The two profiles differ only here:

- `sim`: `motion_switcher_required=false`, because the simulator has no MotionSwitcher service;
- `real`: `motion_switcher_required=true`, requiring bounded successful check/release before LowCmd creation.

The native `G1DdsControlEndpoint` checks CRC, mode, finite values, advancing tick, and freshness. State older than
`max(5 * control_dt, 0.1 s)` is rejected. The writer publishes at `control_dt`; a stale application target or stale
LowState triggers one `Kp=0`, `Kd=5` damping command, clears the cached command, and rejects continued use. `shutdown()`
is idempotent and sends damping only after activation.

Both deployments use the same startup and loop:

1. `self_check()` and 10 dry inference frames without writes;
2. perform one more state/input/shutdown/tilt preflight, then activate commands;
3. ramp measured pose to standing for 3 seconds, then reset policy history;
4. run the paced policy loop directly with shutdown and `is_upright(state.base_quat)` checks before each write;
5. call `shutdown()` from `finally`.

There is no simulator reset channel or `G1Env.reset()`. A tilt failure stops the client; restart the simulation processes
instead of bypassing DDS to mutate backend state.

## Simulator peer

The standalone simulator is a DDS peer, not an environment implementation. Its simulation-only native
[`G1DdsRobotEndpoint`](../third_party/unitree_cpp/src/g1_dds_robot_endpoint.cpp) receives LowCmd and publishes LowState;
it is injected into `G1MujocoDdsServer` and contains no MuJoCo physics. On each 1 ms tick the server computes PD
torque from its cached previous `MujocoState`, clips by `robot.dof.torque_limits`, and calls
`backend.step(torque, support_scale)` once. That locked call sets elastic support, advances physics, and returns the only
new read-only state snapshot for publication and the next tick. `G1MujocoBackend.timestep` is the single physics-period
owner; the server derives its pacing period from the backend.

Python writes only joint position/velocity/torque, quaternion, and gyroscope into each native LowState snapshot. Native
snapshot defaults provide zero accelerometer, RPY, and wireless-remote fields. The fixed simulator
`mode_machine=5` is supplied explicitly when the launcher constructs the native endpoint; it is not user YAML.

The viewer remains observation-only and runs on copied render data. Unit and loopback tests can instantiate the
viewer-free backend/server layers directly; the public simulator launcher intentionally requires a display.

> [!CAUTION]
> Software checks, damping, torque clipping, and simulator behavior do not establish hardware safety. Keep the robot's
> independent emergency stop guarded and its fall/reach envelope clear.
