# Unitree G1 29DoF Real-Robot Setup

The `g1_real` configuration deploys `UnitreeWoGaitPolicy` through `UnitreeCtrl` and `UnitreeCppEnv`. It requires
[Unitree SDK2](https://github.com/unitreerobotics/unitree_sdk2) and the
[UnitreeCpp](https://github.com/HansZ8/unitree_cpp) Python binding.

> [!CAUTION]
> Starting `g1_real` can command all 29 joints during preparation. Complete the MuJoCo test first, clear the robot's full
> fall and reach envelope, use an approved support arrangement, and assign a trained operator to the independent hardware
> emergency stop. Do not perform the first test around people, animals, stairs, glass, or unsecured equipment.

> [!WARNING]
> The remote's `A` button is a software shutdown path, not a certified emergency stop. A process failure, network fault,
> or controller fault can make it unavailable. Keep the hardware emergency stop ready at all times and leave
> `do_safety_check=True` in `g1_real`.

> [!WARNING]
> The current real-robot path does not clamp position targets with `position_limits` and does not use `torque_limits`.
> Those values are not a hardware safety barrier; torque clipping is implemented only in MuJoCo.

## 1. Install the SDK

Use an Ubuntu host or the G1 onboard computer that can reach the robot's SDK2 network.

1. Install Unitree SDK2 by following its official build and installation instructions.
2. From the RoboJuDo repository root, build and install the vendored Python binding:

   ```bash
   python scripts/install_third_party.py unitree_cpp
   ```

   Its source is pinned under `third_party/unitree_cpp/`. The build requires CMake, a C++17 compiler, `pybind11`,
   `scikit-build-core`, Unitree SDK2, and CycloneDDS development libraries.

3. Verify the binding and RoboJuDo environment import:

   ```bash
   python -c "import unitree_cpp; from robojudo.environment import UnitreeCppEnv; print(UnitreeCppEnv)"
   ```

Resolve any SDK library, permission, or import error before connecting the policy to the robot.

## 2. Configure the Network Interface

Connect the host to the robot using the network arrangement required by Unitree SDK2. Identify the exact interface rather
than assuming it is `eth0`:

```bash
ip -brief link
ip -brief address
```

Set `net_if` in [`g1_cfg.py`](../robojudo/config/g1/g1_cfg.py):

```python
class g1_real(g1):
    env: G1RealEnvCfg = G1RealEnvCfg(
        target="hardware",
        unitree=G1UnitreeCfg(
            domain_id=0,
            net_if="eth0",  # replace with the robot-facing interface
        ),
    )
```

The real profile rejects Domain 1 and `lo`; the simulation DDS profile is reserved for Domain 1 on `lo`. Configuration
validation does not prove that the named interface exists, so confirm it with the commands above. Use a dedicated, stable
connection. Do not start actuation until Unitree SDK2 can receive state reliably on that interface.

## 3. Complete the Preflight

Before every hardware run:

- Run `python scripts/run_pipeline.py -c g1` with the same checkout and checkpoint.
- Verify zero-command standing, all three command directions, joint order, joint limits, and `A` shutdown in simulation.
- Confirm the robot is the supported G1 29DoF model and that its joints, battery, remote, and cables pass inspection.
- Follow Unitree's procedure to enter the required low-level control mode; stop if the reported state or model is wrong.
- Place the robot in the approved starting pose with both command sticks released.
- Clear the operating area and agree on a stop signal. One operator controls commands; another guards the hardware stop.
- Test the independent hardware emergency stop according to the manufacturer's procedure.

Do not continue when state updates are intermittent, the measured pose is unexpected, or the software shutdown has not
been verified under a secured, zero-command test.

## 4. Start and Stop

With the exclusion zone clear and the hardware stop guarded, start:

```bash
python scripts/run_pipeline.py -c g1_real
```

Keep both sticks released while preparation first ramps to the standing pose and then blends into closed-loop control;
velocity axes are forced to zero during both phases. After the robot is stable, apply only small commands and verify
direction before increasing speed.

To stop normally, release the sticks and press `A`; RoboJuDo sends `[SHUTDOWN]` to `UnitreeCppEnv`, which disables further
actions and invokes the SDK shutdown path. If the robot does not respond immediately or behaves unexpectedly, use the
independent hardware emergency stop. Diagnose the cause before restarting—do not rely on unplugging the network cable as
a stopping method.

Real-robot deployment is research use at the operator's risk. Software checks reduce risk but cannot make learned control
intrinsically safe.
