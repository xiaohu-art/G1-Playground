# Repository Guidelines

## Scope and Architecture

G1-Playground is limited to Unitree G1 29DoF locomotion. The top-level Hydra deployment group has only `sim` (the
default, UnitreeCpp over Domain 1/`lo`) and `real` (hardware over Domain 0/a robot-facing interface). Both use the
explicit control loop in `scripts/run_pipeline.py`, the single top-level `g1_playground.g1_env.G1Env`, and the same
`UnitreeWoGaitPolicy`; they vary operator input, endpoint, and MotionSwitcher requirement. The runtime boundary is
`Policy → G1Env → DDS → real G1 / MuJoCo server`. Simulation is always the two-process DDS path and its standalone
server always owns an official MuJoCo viewer. Read `docs/architecture.md` before changing this boundary. Do not
reintroduce a direct `MujocoEnv`, a `dds_sim` alias, partial-DoF policies, motion tracking, other robots, or extra
registries without an explicit architecture decision and tests.

## Project Structure

- `configs/run_pipeline.yaml`: Hydra defaults, device, and the one root `G1Env` `_target_`;
  `configs/robot/g1.yaml`: XML, G1 runtime joint order, and simulator torque limits;
  `configs/policy/unitree_wo_gait.yaml`: checkpoint, one `dof` block (joint order/default pose/Kp/Kd), scales, and maximum
  commands; `configs/deployment/`: only endpoint/topics, MotionSwitcher requirement, and controller target for `sim` and
  `real`. Do not duplicate environment target or control period in deployment YAML. `scripts/run_pipeline.py` uses
  `compose_dof_config(cfg.robot.dof, cfg.policy.dof)`, constructs the policy, and injects `policy.dt` into `G1Env` with the
  composed DoF. Do not add robot-side pose/gains, a second policy DoF list, or a post-construction override path.
- `g1_playground/g1_env.py`, `g1_playground/controller/`, and `g1_playground/policy/`: the single G1 DDS environment,
  operator input, and policy implementation. `scripts/run_pipeline.py` is the only policy composition root and spells
  out the complete startup/active lifecycle; do not recreate a transition module, pipeline package, environment hierarchy/package,
  manager, registry, or wrapper class around this loop. Preserve the
  direct policy-runtime data path: `G1Env.read()` returns one frozen `G1State`; `Controller.read()` returns
  `({"axes": ...}, shutdown_requested: bool)`; and `UnitreeWoGaitPolicy.act()`/`standing_target` own runtime DoF mapping
  and target construction. `Controller` is the concrete queue consumer; `JoystickCtrl` and `UnitreeCtrl` are sibling
  input-source subclasses and must not inherit from each other. Do not reintroduce command lists, `BasePolicy`, post-step
  callbacks, or launcher-owned observation/action adapters.
- `g1_playground/simulation/`: `G1MujocoBackend`, elastic startup support, and the viewer-free DDS-server physics loop.
  The backend is the repository-owned MuJoCo physics core and remains independent of policy and DDS; deleting `MujocoEnv`
  does not make the backend optional. Keep the hot path at one cached input state and one
  `backend.step(torque, support_scale)` per tick. That locked call sets support, advances physics, and returns the single
  new detached/read-only snapshot. `G1MujocoBackend.timestep` is the sole physics-period owner; do not add a server copy.
- `assets/robots/g1/` and `assets/models/g1/unitree/`: the 29DoF XML/meshes and sole TorchScript checkpoint.
- `scripts/run_mujoco_dds_server.py`: the standalone simulator assembly root. It reads the existing
  `configs/robot/g1.yaml` and `configs/deployment/sim.yaml` fragments directly; there is no second Hydra root for the
  server. The backend default owns the 1 ms physics/DDS period, the server default owns its 0.1-second watchdog, and the
  launcher constant owns the 60 Hz viewer rate. The main thread always owns official
  `mujoco.viewer.launch_passive()` on separate snapshot data, while pure 1 kHz physics/DDS runs in a background thread.
  Keep backend copy and viewer access locked as implemented and keep the GUI observation-only; viewer interaction must
  never mutate physics. This launcher requires a display and intentionally has no headless option. Tests that do not need
  a GUI must instantiate the lower-level backend/server directly rather than adding a second launcher mode.
  Change a simulation endpoint only in the shared `deployment/sim.yaml`; never apply a one-sided endpoint override to
  the policy client because the standalone server does not consume policy-launcher overrides.
- `scripts/`: the policy launcher, mandatory-viewer simulator entry point, and vendored `unitree_cpp` installer.
- `docs/`: current architecture, component contracts, and hardware safety procedures. Phase and pre-DDS records are
  historical evidence; do not rewrite old commands or results as though the current architecture produced them.
- `tests/test_full_imports.py`: tracked configuration contracts, 29DoF invariants, asset closure, and model-shape tests.
- `third_party/`: tracked vendors, not Git submodules. Preserve recorded revisions/licenses, do not run repository
  formatters over them, and document direct vendor changes in the relevant phase/design record. Keep the pre-DDS
  provenance and source-tree hashes in `tests/fixtures/pre_dds/contract.json` immutable. The retained vendored MuJoCo
  viewer tree is historical/vendor content, not a current runtime dependency.

The runtime uses plain dictionaries/dataclasses, standard `queue.Queue`, and logging; `python-box` and `tqdm` are not
dependencies. Do not restore wrapper/progress abstractions without a concrete consumer.

## Setup, Run, and Validation

```bash
python -m pip install -e ".[dev]"
python scripts/install_third_party.py unitree_cpp
python scripts/run_mujoco_dds_server.py                 # terminal 1; viewer is mandatory
python scripts/run_pipeline.py deployment=sim           # terminal 2
python -m unittest discover -s tests -p 'test_*.py' -v
ruff check g1_playground scripts tests
ruff format --check g1_playground scripts tests
pre-commit run --all-files
```

MuJoCo is pinned to 3.11.0. The simulation launcher requires a working display; the local controller is optional for a
stationary zero-command check, in which case use `Ctrl+C` to stop. Most tests are dependency-free and do not launch the
viewer. The opt-in DDS loopback uses real local participants and a clean-built binding, but it does not exercise hardware,
manual policy scenarios, interface mismatch, or sustained 1 kHz timing/load.

Real deployment requires Unitree SDK2, CycloneDDS, the vendored `unitree_cpp` binding, and the correct robot-facing
interface. Follow `docs/unitree_setup.md`; never infer hardware readiness from an import test.

## Code and Review Conventions

Use Python 3.11+, four-space indentation, modern annotations, `snake_case` names, and `PascalCase` classes. Ruff enforces
120-character lines and E/F/I/B/UP rules. Keep commits focused with short, imperative subjects. Pull requests must
describe the affected contract and exact validation performed.

`tests/test_full_imports.py` intentionally locks the tracked configurations and assets; do not loosen those contracts
merely to make a new component pass. Hydra composes the complete `deployment=sim` or `deployment=real` profile. `run()`
never infers simulation or hardware from `domain_id`; it constructs the policy, then calls
`hydra.utils.instantiate(cfg.env, dof_cfg=dof, control_dt=policy.dt)` followed by
`instantiate(cfg.controller, env=env)`. `G1Env` assembles the native configuration dictionary from explicit endpoint,
topic, motion, period, and DoF inputs. It owns no reset/fallen/cache API, Python `Box`, position limits, or joint map.
Do not reintroduce a
`g1_playground/config` package, Python configuration objects, a direct MuJoCo environment, a pipeline abstraction, or a
compatibility loader. The simulator launcher is an assembly root, not a Hydra application or second schema validator: it
loads the robot XML/limits and simulation DDS endpoint from their existing fragments, then constructs
`G1MujocoBackend`, `G1MujocoDdsServer`, and the native DDS boundary. Do not restore duplicate YAML fields or Python guards
that lock the 1 ms/0.1-second defaults. Keep only owner-boundary invariants: the backend checks its
model/timestep, the server requires a finite positive watchdog timeout, and native code checks the DDS endpoint/wire.

`UnitreeWoGaitPolicy.FREQ=50`, `HISTORY_LENGTH=5`, and `HISTORY_LAYOUT` are checkpoint/class contracts. Keep them out of
Hydra unless the model contract itself changes. `robot.dof.torque_limits` is clipped by the simulator server only; the
live client has neither a position-limit configuration nor a hardware torque clamp.

Keep the control order directly visible in `scripts/run_pipeline.py`: composition, self-check, dry frames, preflight,
activation, 3-second standing ramp, policy-history reset, 5-second zero-command blend, paced active loop, and teardown.
The pure quaternion tilt predicate `is_upright()` belongs in `g1_playground/utils/math.py`; do not recreate a
transition/runner abstraction or move controller/policy-aware lifecycle code into generic utilities. In simulation the
separate server aligns its elastic-support schedule to the first valid LowCmd.
On Jetson, preserve the aarch64 OMP setting and torch-before-numpy import workaround in
`g1_playground/__init__.py` unless validated on target hardware.

Treat joint order, observation packing, action scaling, PD gains, loop timing, DDS topics, shutdown behavior, and network
interfaces as safety-critical. Validate the exact checkout with the viewer server plus `deployment=sim` first. Never run
`deployment=real` as an automated or CI check; it requires explicit operator authorization and a completed preflight.
Software shutdown and tilt checks never replace the independent hardware emergency stop.
