# G1-Playground

G1-Playground is a focused deployment runtime for the **Unitree G1 29DoF**. It serves three policy families:

- the `LeggedLabPolicy` locomotion checkpoint from the [LeggedLab](https://github.com/Hellod035/LeggedLab) training framework
  (transcribed from [LeggedLabDeploy](https://github.com/Hellod035/LeggedLabDeploy), BSD-3-Clause, hardware-verified before integration);
- a whole-body **track** policy for standing reference tracking;
- a **body-hand HOI** policy (`proprio-no-depth-v1`, trained in the sibling `hoi/` Isaac Lab workspace) that tracks a
  clipped reference motion with all 29 body joints plus 12 Inspire-hand joints.

Simulation and hardware use the same policy loop and
the same DDS-facing `G1Env`; the selected endpoint, operator input, and hardware activation requirement differ.

| Hydra deployment | Controller | G1 environment | Endpoint |
| --- | --- | --- | --- |
| `deployment=sim` (default) | `KeyboardCtrl` | `G1Env` | Domain 1 / `lo` → standalone MuJoCo DDS server |
| `deployment=real` | `UnitreeCtrl` | `G1Env` | Domain 0 / robot-facing interface → G1 |

The public runtime path is `Policy → G1Env → DDS → real G1 / MuJoCo server`. `G1Env` does not select or directly
call a physics backend: both deployments cross the same HG LowState/LowCmd wire boundary through
`unitree_cpp.G1DdsControlEndpoint`. Only the standalone simulator constructs `unitree_cpp.G1DdsRobotEndpoint`; on
`deployment=real`, the physical G1 itself fills the robot-endpoint role and no simulated endpoint is created. The two
native classes do not call or own each other; they only share neutral CRC and DDS endpoint helpers inside the extension.

There is no direct in-process `MujocoEnv` path. Simulation is intentionally a two-process DDS system. The standalone
server retains the repository-owned `G1MujocoBackend` physics core and always opens the official MuJoCo viewer.

The policy observes and commands all 29 joints. Its 50 Hz rate, single-frame 96-to-29 layout, and recurrent (LSTM) state
are class-level checkpoint contracts; they are not deployment tuning fields. The checkpoint is stored at
`assets/models/leggedlab/g1_policy.pt`.

## Installation

Python 3.11 or newer is required. MuJoCo is pinned to **3.11.0** so viewer APIs and physics behavior do not drift under an
unreviewed dependency upgrade.

```bash
git clone https://github.com/xiaohu-art/G1-Playground.git
cd G1-Playground
conda create -n g1-playground python=3.11 -y
conda activate g1-playground
pip install -e .
python scripts/setup/install_third_party.py unitree_cpp
```

The simulation and hardware paths both require the vendored `unitree_cpp` binding, Unitree SDK2, and CycloneDDS because
`G1Env` uses its `G1DdsControlEndpoint`. The binding's `G1DdsRobotEndpoint` is an additional simulation-only adapter. The
simulator GUI comes from the official `mujoco.viewer` module installed by the pinned `mujoco` dependency. The retained
`third_party/mujoco_viewer` source tree is vendored historical content and is not used at runtime. The tracked dependency
trees are ordinary directories, not Git submodules.

For development tools, install the optional extra:

```bash
python -m pip install -e ".[dev]"
```

## Configuration

Hydra composes one robot, one policy, and one of two deployment profiles:

```text
configs/
├── run_pipeline.yaml            # locomotion-only client
├── run_body_hand.yaml           # body-hand HOI client
├── run_loco_largebox_track.yaml # loco → body-hand HOI → stand-track handover
├── robot/
│   ├── g1.yaml
│   └── inspire.yaml
├── policy/
│   ├── leggedlab_g1.yaml
│   ├── track.yaml
│   └── body_hand_distill_largebox.yaml
├── motion/
│   └── largebox_039_v00.yaml
└── deployment/
    ├── sim.yaml
    └── real.yaml
```

`run_pipeline.yaml` is the locomotion composition root; `run_body_hand.yaml` and `run_loco_largebox_track.yaml` compose the
body-hand HOI runtime described in [Body-Hand HOI Deployment](#body-hand-hoi-deployment). The simulator launcher reads
`robot/g1.yaml` and
`deployment/sim.yaml` directly, reusing the checked-in XML, torque limits, and default DDS endpoint instead of duplicating
them. It uses the backend's 1 ms default timestep, the server's 0.1-second default watchdog, and a launcher-owned 60 Hz
viewer constant.
The simulator has no Hydra override surface, always opens its viewer, and has no headless switch.

`run_pipeline.yaml` owns the single top-level `g1_playground.g1_env.G1Env` `_target_`. A deployment profile contributes
only DDS endpoint/topics, the MotionSwitcher requirement, and its controller target. The checked-in compositions pair
Domain 1/`lo` plus `JoystickCtrl` with `sim`, and Domain 0/a robot-facing interface plus `UnitreeCtrl` with `real`; `domain_id`
and `net_if` are endpoint data, not backend type tags.

Hydra passes the composed `DictConfig` directly to `scripts/pipeline.py`. Its `run()` function:

1. reorders the single `cfg.policy.dof` pose and gains into robot joint order with
   `compose_dof_config(cfg.robot.dof, cfg.policy.dof)`;
2. constructs `LeggedLabPolicy`;
3. calls `hydra.utils.instantiate(cfg.env, dof_cfg=dof, control_dt=policy.dt)`;
4. calls `instantiate(cfg.controller, env=env)`.

`robot/g1.yaml` owns the XML, G1 runtime joint order, and simulator torque limits. It has no live position-limit field.
`policy/leggedlab_g1.yaml` owns the checkpoint plus one policy DoF block containing joint order, standing pose, and
tuned Kp/Kd, together with the observation scales, command clip range, and clip magnitudes. Frequency, history length, and observation layout stay with the policy class. `G1Env` receives one complete
effective DoF configuration and the policy-owned period at construction; there is no later gain or period override.

To inspect the resolved real configuration without constructing a runtime or connecting to the robot:

```bash
python scripts/pipeline.py --cfg job --resolve deployment=real env.net_if=enP8p1s0
```

## Run the DDS MuJoCo Simulation

Simulation always uses two terminals. Start the simulator first:

```bash
# Terminal 1
python scripts/simulate.py
```

This command requires a working display and always opens the server-owned MuJoCo GUI. The main thread owns official
`mujoco.viewer.launch_passive()` on separate render data, while the physics/DDS loop runs at 1 kHz in a background thread.
Backend stepping and render-data copying share a lock, and viewer interaction never writes back into physics. Closing the
window stops the worker and closes the native `G1DdsRobotEndpoint` and viewer.

Then start the policy client:

```bash
# Terminal 2
python scripts/pipeline.py
# Equivalent explicit Hydra override:
python scripts/pipeline.py deployment=sim
```

The policy client uses `G1Env` on Domain 1/`lo`, exactly as declared by `configs/deployment/sim.yaml`; it does not create
or own a MuJoCo viewer. The simulator's native
[`G1DdsRobotEndpoint`](third_party/unitree_cpp/src/g1_dds_robot_endpoint.cpp) publishes HG `rt/lowstate` and subscribes to
HG `rt/lowcmd`; `G1MujocoDdsServer` consumes its snapshots and advances
`G1MujocoBackend` from the repository-owned G1 XML at 1 kHz. Each tick computes PD torque from the cached previous state,
clips it to `robot.dof.torque_limits`, then makes one locked `backend.step(torque, support_scale)` call. That call advances
physics and returns the only new snapshot for publication and the next tick. The backend is the single owner of the
physics timestep.

If the local simulation endpoint must change, edit `configs/deployment/sim.yaml` before starting both processes. Do not
apply a one-sided `env.domain_id`, `env.net_if`, or topic override to the policy client: the standalone server intentionally
reads the shared file directly.

The sim profile drives with the keyboard: arrow keys command forward/backward and turning, shift+arrows strafe, and `esc`
requests software shutdown. If keyboard input goes stale, the launcher zeroes every axis; use `Ctrl+C` to stop that
terminal.

The server keeps elastic support enabled before the first valid LowCmd, holds full support for the 3-second standing ramp,
then releases it over the 5-second policy blend. A valid command older than the current 0.1-second simulator timeout stops
the server. These are simulation defaults under evaluation, not safety-certified limits or hardware watchdogs.

The simulator is a separate DDS peer outside `G1Env`; it does not load a policy or controller. The complete boundary is:

```text
pipeline.py
  JoystickCtrl → LeggedLabPolicy → G1Env → G1DdsControlEndpoint
                                                                    ⇅ HG DDS
simulate.py
  G1DdsRobotEndpoint → G1MujocoDdsServer → G1MujocoBackend → official MuJoCo viewer
```

## Run on a Real G1

> [!CAUTION]
> A learned policy can fall, accelerate unexpectedly, or command damaging joint motion. Validate the exact checkout and
> checkpoint through the DDS MuJoCo simulation first. Clear the operating area, use an approved support arrangement, keep
> a trained operator at an independent hardware emergency stop, and begin with zero velocity commands.

> [!WARNING]
> `A` is a software shutdown path. It is not a substitute for the robot's independent hardware emergency stop. Safety and
> pacing are unconditional in the launcher and have no runtime bypass.

Configure the robot-facing interface as described in the [G1 setup guide](docs/unitree_setup.md), then start the hardware
profile only after completing its safety checklist:

```bash
python scripts/pipeline.py deployment=real env.net_if=enP8p1s0
```

Replace `enP8p1s0` with the exact robot-facing interface. Startup first receives and validates LowState without a command
publisher. The launcher then performs a final preflight, activates command transport, ramps from the measured pose to
standing over 3 seconds, resets policy history, and blends into zero-command closed-loop control over 5 seconds.
Simulation creates LowCmd transport without MotionSwitcher; real deployment requires MotionSwitcher availability and a
bounded successful release before creating that transport. This path uses `G1Env`'s native `G1DdsControlEndpoint`
directly against the physical G1; it neither constructs nor depends on the simulation-only `G1DdsRobotEndpoint`.

The Unitree remote uses the same velocity axes as the local simulation controller. Keep the independent hardware stop
guarded throughout the run and press `A` for software shutdown at the first sign of instability.

## Body-Hand HOI Deployment

The body-hand runtime deploys `proprio-no-depth-v1`, a proprioception-only whole-body checkpoint distilled in the
sibling `hoi/` Isaac Lab workspace (`logs/rsl_rl/g1_inspire_hoi_depth`). It tracks a staged reference motion clip with
29 body joints and 12 Inspire-hand joints; the intermediate/distal finger joints follow through the mimic table in
`configs/robot/inspire.yaml`.

Contract, pinned by `configs/policy/body_hand_distill_largebox.yaml` and the exported graph:

| Boundary | Value |
| --- | --- |
| Checkpoint | `assets/models/body_hand_distill/largebox/policy.onnx` (+ `policy.onnx.data`), normalization baked into the graph |
| Observation | 728: future joint pos/vel (2×265), future anchor linear velocity in the pelvis frame (15), future anchor orientation rot6d (30), base angular velocity (3), relative joint pos (53), joint vel (53), projected gravity (3), last action (41) |
| Action | 41 position targets = 29 body + 12 hand, scale 1.0 around the default pose |
| Rate | 50 Hz |

The observation carries no anchor *position*, so the policy does not consume odometry; the startup capture still calls
`env.set_born_place(...)` so the IMU yaw and the reference clip share one heading frame. The staged clip is clip 0
(`teacher_ref5_env4_ep1`, 294 source frames, walking) extracted from the packed training set
`hoi/data/train/track_H20_var5_1perclip.npz` (208 clips). To stage a different clip:

```bash
python scripts/prepare_hoi_motion.py --clip <index-or-name> --out-name sub16_largebox_039_v00
```

> [!NOTE]
> The `largebox` file and variable names are kept from the previous checkpoint on purpose; the deployed assets are the
> H20-track clip and the `proprio-no-depth-v1` graph.

Before any DDS run, the in-process MuJoCo smoke test closes the loop around the deployed graph without a viewer:

```bash
python scripts/sim_hoi_smoke.py
```

It plays the deployable source and terminal frames at 50 Hz with PD torques at 1 kHz. It fails if the robot loses the
reference, falls, or reaches the end above the hardware return-handoff velocity limits.

### Body-hand simulation

The hand-aware simulator serves the Inspire topics; start it first, then the pipeline client:

```bash
# Terminal 1
python scripts/simulate.py --inspire
# Terminal 2: body-hand clip only
python scripts/body_hand_pipeline.py deployment=sim
# or the full handover: locomotion → body-hand clip → standing tracker
python scripts/loco_body_hand_pipeline.py deployment=sim
```

`body_hand_pipeline.py` ramps to the first reference pose, captures the heading origin, and starts the dynamic reference
without freezing frame 0. `loco_body_hand_pipeline.py` starts in driven locomotion and, when the operator presses `]`,
keeps locomotion closed-loop at zero command until the base has remained stable for ten consecutive frames. The standing
tracker then walks the live whole-body reference to HOI frame 0 over 4 seconds. The source clip ends in a high-speed phase,
so its deployment config stops at validated frame 275 and appends a two-second zero-velocity terminal reference. The HOI
policy sees the terminal reference through its future window and brakes before handoff. The pipeline then requires ten
consecutive stable frames, preserves the last target actually sent to the robot, and rate-limits the first standing-tracker
command. This keeps the policy histories and physical command continuous. Keyboard input is required in the sim terminal;
the real profile uses the Unitree remote.

`motion.terminal_frame` is clip-specific. If it is omitted, `null`, or `-1`, the loader uses the clip's last source frame;
an explicitly configured non-negative frame overrides that default. Clip-specific values belong in the selected
`configs/motion/*.yaml` fragment rather than in policy or pipeline code.

### Body-hand hardware

> [!CAUTION]
> The same hardware-safety checklist as [Run on a Real G1](#run-on-a-real-g1) applies, plus the Inspire hands. Freeze the
> terminal frame into a quasi-static hold and run that first; only then play the full clip:

```bash
python scripts/prepare_terminal_hold_motion.py \
  --motion hoi/data/train/track_H20_var5_1perclip.npz --clip-name teacher_ref5_env4_ep1
python scripts/body_hand_pipeline.py deployment=real env.net_if=enP8p1s0
```

The hold clip lands next to the staged motion; point `configs/motion/largebox_039_v00.yaml` at it for the first hardware
run, then restore the full clip.

Three additional H20 clips have passed both the checkpoint's clip-wise Isaac evaluation and the deployment MuJoCo smoke
test. Each full-motion config has a matching static preflight config captured at the same validated terminal frame:

| Full motion | Source clip | Terminal frame | Static preflight |
| --- | --- | ---: | --- |
| `motion=h20_clip_006` | `teacher_ref21_env20_ep1` | 276 | `motion=h20_clip_006_hold` |
| `motion=h20_clip_007` | `teacher_ref22_env21_ep1` | 273 | `motion=h20_clip_007_hold` |
| `motion=h20_clip_009` | `teacher_ref24_env23_ep1` | 286 | `motion=h20_clip_009_hold` |
| `motion=h20_clip_014` | `teacher_ref15_env14_ep1` | 274 | `motion=h20_clip_014_hold` |
| `motion=h20_clip_093` | `teacher_ref99_env98_ep1` | 366 | `motion=h20_clip_093_hold` |

For each clip, run the static preflight on hardware before selecting the full motion. The composition roots need no edits:

```bash
python scripts/body_hand_pipeline.py deployment=real motion=h20_clip_006_hold env.net_if=enP8p1s0
python scripts/body_hand_pipeline.py deployment=real motion=h20_clip_006 env.net_if=enP8p1s0
```

Substitute `007`, `009`, `014`, or `093` as needed. The same `motion=...` overrides also work with
`scripts/loco_body_hand_pipeline.py` and `deployment=sim`.

For train↔deploy parity, record an Isaac-side trace with `hoi/scripts/rsl_rl/play.py --parity_trace <path>` and point
`G1_PLAYGROUND_BODY_HAND_BUNDLE` at its directory (default: the checkpoint's `exported/` folder);
`tests/test_body_hand_observation.py::TestParityTrace` then rebuilds the observation from the recorded robot state and
replays the actions through the deployed ONNX.

## Project Layout

- `configs/run_pipeline.yaml`: Hydra policy-client defaults for locomotion.
- `configs/run_body_hand.yaml`, `configs/run_loco_largebox_track.yaml`: body-hand and full-handover composition roots.
- `configs/robot/g1.yaml`: G1 XML, runtime joint order, and simulator torque limits.
- `configs/robot/inspire.yaml`: Inspire-hand scene, runtime joints, limits, and the finger mimic table.
- `configs/policy/leggedlab_g1.yaml`: checkpoint, one policy DoF block, scales, command clip range, and clip magnitudes.
- `configs/policy/body_hand_distill_largebox.yaml`: body-hand graph path, 53-joint observation order, action split, and PD gains.
- `configs/motion/largebox_039_v00.yaml`: the staged body-hand reference clip.
- `configs/deployment/`: endpoint/topics, MotionSwitcher requirement, and controller target for `sim` and `real`.
- `g1_playground/g1_env.py`: the single policy-facing G1 state/target contract and native DDS client adapter.
- `g1_playground/policy/`: `LeggedLabPolicy`, `TrackPolicy`, and `BodyHandPolicy` inference and observation construction.
- `g1_playground/inspire/`: Inspire-hand DDS env, stroke conversion, and the MuJoCo hand adapter.
- `g1_playground/controller/`: Xbox, keyboard, and Unitree remote input.
- `g1_playground/simulation/`: retained `G1MujocoBackend`, elastic support, and the viewer-free DDS-server loop.
- `g1_playground/utils/math.py`: pure quaternion/gravity and tilt calculations.
- `scripts/pipeline.py`: the locomotion Hydra composition root and explicit startup/active policy loop for both deployments.
- `scripts/body_hand_pipeline.py`, `scripts/loco_body_hand_pipeline.py`: body-hand runners for both deployments.
- `scripts/prepare_hoi_motion.py`: packs one training clip into the deployment motion format.
- `scripts/prepare_terminal_hold_motion.py`: freezes one reference frame into a quasi-static hold clip.
- `scripts/sim_hoi_smoke.py`: viewer-free closed-loop MuJoCo smoke test for the body-hand deployment.
- `scripts/simulate.py`: direct-config, mandatory-viewer HG DDS simulator on Domain 1/`lo` for
  `deployment=sim`; `--inspire` loads the hand-aware model.
- `third_party/unitree_cpp/src/g1_dds_control_endpoint.cpp`: native policy-side LowCmd/LowState endpoint used by both
  deployments.
- `third_party/unitree_cpp/src/g1_dds_robot_endpoint.cpp`: native simulated-robot LowCmd/LowState endpoint used only by
  `deployment=sim`.
- `third_party/`: pinned vendor source trees and licenses; the legacy MuJoCo viewer is not a runtime dependency.
- `hoi/`: the sibling Isaac Lab training workspace (checkpoint source); it is not a Python dependency of this runtime.

## Documentation

- [Sim2Sim / Sim2Real deployment architecture](docs/architecture.md)
- [Policy contract](docs/policy.md)
- [Controller mappings](docs/controller.md)
- [G1Env contract](docs/environment.md)
- [UnitreeCpp installation and real-robot safety](docs/unitree_setup.md)

## Development and Validation

Run repository checks from the project root:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
ruff check g1_playground scripts tests
ruff format --check g1_playground scripts tests
pre-commit run --all-files
```

The ordinary suite instantiates the viewer-free backend/server layers directly and does not require the GUI launcher. The
body-hand deployment additionally has the viewer-free closed-loop smoke test:

```bash
python scripts/sim_hoi_smoke.py
```

The opt-in local DDS loopback is:

```bash
G1_PLAYGROUND_RUN_DDS_LOOPBACK=1 python -m unittest tests.test_dds_loopback -v
```

Changes to joint order, observation layout, action scaling, gains, limits, shutdown handling, or network transport are
safety-critical. Document the exact simulation validation performed before proposing a real-robot test.

## License and Citation

The repository is distributed under the
[Creative Commons Attribution-NonCommercial 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/).

```bibtex
@misc{G1Playground,
  author = {Zihan Zhuang, Yi Dong, Peng Li},
  title = {G1-Playground: A deployment framework for the Unitree G1 29DoF},
  url = {https://github.com/xiaohu-art/G1-Playground},
  year = {2025}
}
```

Related projects: [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab),
[Unitree SDK2](https://github.com/unitreerobotics/unitree_sdk2), and
[UnitreeCpp](https://github.com/HansZ8/unitree_cpp).
