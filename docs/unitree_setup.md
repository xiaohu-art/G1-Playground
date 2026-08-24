# Unitree G1 29DoF Real-Robot Setup

`deployment=real` composes `G1Env + UnitreeCtrl` with Domain 0, the configured robot-facing interface, and
`motion_switcher_required=true`. `domain_id` is only a DDS endpoint value; the launcher does not use it to recognize
hardware. DDS simulation and hardware use the same dry frames → preflight → activation → ramp →
paced-loop lifecycle and the same pre-actuation safety checks. `G1Env` uses the native `G1DdsControlEndpoint` to
communicate directly with the physical G1. The simulation-only `G1DdsRobotEndpoint` is not constructed or required by
the real profile.

> [!CAUTION]
> Starting `deployment=real` can command all 29 joints during the shared 3-second standing ramp and subsequent direct
> policy control. Validate the exact checkout with the mandatory-viewer simulator and `deployment=sim`, clear the robot's fall and
> reach envelope, use an approved support arrangement, and assign a trained operator to the independent hardware
> emergency stop.

> [!WARNING]
> The remote `A` button, tilt gate, state watchdog, command damping, and MotionSwitcher checks are software defenses,
> not certified emergency stops. A process, network, controller, or actuator fault can make them ineffective.

> [!WARNING]
> The current configuration has no position-limit field or hardware target clamp. The configured
> `robot.dof.torque_limits` are consumed only by the MuJoCo server, not by hardware LowCmd. Neither is a hardware safety
> barrier.

## 1. Install SDK and binding

Use an Ubuntu host or the G1 onboard computer that can reach the robot's SDK2 network.

1. Install Unitree SDK2 according to Unitree's instructions.
2. From the G1-Playground repository root, build the vendored binding:

   ```bash
   python scripts/setup/install_third_party.py unitree_cpp
   ```

   The build needs CMake, a C++17 compiler, pybind11, scikit-build-core, Unitree SDK2, and CycloneDDS development files.
   It provides the `G1DdsControlEndpoint` used here; its `G1DdsRobotEndpoint` class is only the standalone simulator's
   opposite DDS endpoint.

3. Verify imports without connecting to hardware:

   ```bash
   python -c "import unitree_cpp; from g1_playground.g1_env import G1Env; print(G1Env)"
   ```

Resolve all build, dynamic-library, permission, and import failures before attaching a policy to the robot.

## 2. Configure the endpoint

Identify the exact interface instead of assuming `eth0`:

```bash
ip -brief link
ip -brief address
```

The tracked real profile contains a legacy `enP8p1s0` value. Override it with the interface that actually faces the robot:

```bash
python scripts/pipeline.py --cfg job --resolve deployment=real env.net_if=ROBOT_NIC
```

This command only prints resolved configuration. Confirm Domain 0, the interface name, topics,
`motion_switcher_required=true`, and the intended checkpoint before running without `--cfg`. The 20 ms control period is
not duplicated in deployment YAML: `LeggedLabPolicy` owns 50 Hz, and the launcher injects `policy.dt` into `G1Env`.

The repository convention is Domain 0 / robot NIC for `real` and Domain 1 / `lo` for `sim`. These are explicit profile
choices, not DDS mode semantics. Never point `deployment=sim` at the robot or disable the real profile's
MotionSwitcher requirement as a shortcut.

## 3. Complete preflight

Before every hardware run:

- Start `python scripts/simulate.py`; its viewer is mandatory. In another terminal run
  `python scripts/pipeline.py deployment=sim` with the exact checkout and checkpoint. Verify zero command, x/y/yaw,
  standing preparation, fail-stop fall handling, `A` shutdown, and clean teardown.
- Confirm the robot is the supported G1 29DoF model and inspect joints, battery, remote, cables, and support equipment.
- Verify joint order, starting pose, command directions, policy gains, and expected LowState mode.
- Put both command sticks at zero and follow Unitree's approved procedure for entering low-level control.
- Establish an exclusion zone and stop signal. One operator drives; another guards the hardware emergency stop.
- Test the independent hardware emergency stop according to the manufacturer's procedure.

Do not continue if state delivery is intermittent, a stale/mode/tick/finite check fails, measured pose is unexpected,
MotionSwitcher is unavailable, or software shutdown has not been verified in a secured zero-command test.

## 4. Start sequence

With the robot supported, the exclusion zone clear, and the hardware stop guarded:

```bash
python scripts/pipeline.py deployment=real env.net_if=ROBOT_NIC
```

The exact sequence is:

1. `G1Env` constructs `G1DdsControlEndpoint`, which creates receive resources only.
2. `self_check()` requires valid state.
3. 10 dry frames read state/input, check `A` and tilt/fall, and run policy inference without writing.
4. The launcher performs one more state/input/shutdown/tilt preflight, saves the measured pose, then calls
   `activate_commands()`.
5. Real activation requires MotionSwitcher availability and a bounded successful release; failure raises before LowCmd
   publisher/writer creation.
6. The robot ramps from measured pose to standing over 3 seconds.
7. Policy history resets, then the locomotion policy takes direct closed-loop control.
8. The paced active loop continues with state/input/shutdown/tilt checks before every policy action and command.

Keep both sticks released throughout the ramp and initial policy takeover. Once stable, introduce only a small command and verify its
direction before increasing magnitude.

Native LowState acceptance checks CRC, finite motor/IMU values and expected mode. Initial readiness requires a nonzero
tick; subsequent ticks must strictly advance while allowing normal `uint32` maximum-to-zero wrap. The state freshness
deadline is `max(5 * control_dt, 0.1 s)`. LowCmd targets must be 29 values representable by the DDS `float`
wire type. If the application's latest
target exceeds the same derived age limit, or cached LowState is no longer fresh, the writer sends one Kp=0/Kd=5 damping
command, clears it, and rejects continued use. These mechanisms reduce exposure to stale data; they do not prove hardware
safety.

## 5. Stop and recovery

For a normal software stop, release the sticks and press `A`. The launcher detects it before the next actuation and reaches
`G1Env.shutdown()` through `finally`; an activated control endpoint stops its writer, sends a damping command, and
closes transport. `Ctrl+C` and raised state/command exceptions use the same teardown path.

If the robot does not respond immediately or behaves unexpectedly, use the independent hardware emergency stop. Do not
depend on disconnecting Ethernet as a stopping mechanism. Diagnose and repeat the full simulation/preflight sequence
before restarting.

No new hardware run was performed as part of the DDS simulator and lifecycle hardening work. The final native exact-byte
clean build and local Domain 1/`lo` loopback are transport evidence only, not hardware authorization.
