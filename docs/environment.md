# Environments

An environment supplies robot state to `UnitreeWoGaitPolicy` and applies its joint position targets. RoboJuDo exposes two
G1 29DoF environments:

| Environment | Purpose | Configuration |
| --- | --- | --- |
| `MujocoEnv` | Local simulation | `g1` / `G1MujocoEnvCfg` |
| `UnitreeCppEnv` | Real Unitree G1 | `g1_real` / `G1RealEnvCfg` |

The common interface is defined in [`base_env.py`](../robojudo/environment/base_env.py).

## 29DoF Contract

Both environments expose the same policy inputs:

- `dof_pos`: 29 joint positions
- `dof_vel`: 29 joint velocities
- `base_quat`: base orientation in `[x, y, z, w]` order
- `base_ang_vel`: three-axis base angular velocity

Each call to `step()` receives exactly 29 position targets. The policy configuration supplies its standing pose and tuned
PD gains. The environment configuration declares torque and position limits, but the current runtime only applies torque
clipping in MuJoCo; the UnitreeCpp path neither clamps position targets with `position_limits` nor uses `torque_limits`.
Their joint lists must contain the same 29 names. A different ordering can be adapted, but partial joint control is rejected.

## MujocoEnv

[`mujoco_env.py`](../robojudo/environment/mujoco_env.py) loads
`assets/robots/g1/g1_29dof_rev_1_0.xml` and applies position targets through the configured PD gains. The default simulation
step is `0.001 s` with a decimation of 20, matching the policy's 50 Hz control rate.

Install the viewer and run the simulation from the repository root:

```bash
python scripts/install_third_party.py
python scripts/run_pipeline.py -c g1
```

Use this environment to verify standing behavior, command directions, saturation, falls, and shutdown before every
hardware change.

## UnitreeCppEnv

[`unitree_cpp_env.py`](../robojudo/environment/unitree_cpp_env.py) exchanges low-level state and position commands through
Unitree SDK2 using the [`unitree_cpp`](https://github.com/HansZ8/unitree_cpp) binding. The robot-facing interface is set by
the `g1_real` override in [`g1_cfg.py`](../robojudo/config/g1/g1_cfg.py). Real deployment must explicitly provide the atomic
`hardware / domain 0 / non-lo interface` endpoint; there is no generic `eth0` fallback.

`UnitreeCppEnv` performs an SDK self-check during startup. Its `shutdown()` method disables further actions and invokes the
SDK shutdown path. Installation and network preparation are documented in [Unitree G1 setup](unitree_setup.md).

> [!CAUTION]
> Passing startup checks does not prove that a checkpoint, joint order, gains, or network link is safe. Keep the hardware
> emergency stop under direct operator control, keep people outside the robot's fall and reach envelope, and leave
> `do_safety_check` enabled in `g1_real`.
