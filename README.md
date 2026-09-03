# G1-Playground

G1-Playground deploys a 50 Hz policy sequence on the Unitree G1:

```text
LeggedLab locomotion → Track interpolation → depth HOI → Track default pose
```

Simulation and hardware use the same policy loop and `G1Env` DDS interface. MuJoCo supplies simulated LowState,
odometry, Inspire hand state, and depth; the real deployment uses the G1, Inspire serial service, and a RealSense D435i.
The HOI runtime is depth-only and supports the checked-in `largebox` and `smalltable` models.

## Installation

Python 3.11 or newer is required.

```bash
cd G1-Playground
conda create -n g1-playground python=3.11 -y
conda activate g1-playground
python -m pip install -e .
python scripts/setup/install_third_party.py unitree_cpp
```

Install the architecture-specific TensorRT Python wheel on both the workstation and Jetson:

```bash
export TENSORRT_ROOT="$HOME/TensorRT"
python -m pip install "$TENSORRT_ROOT"/python/tensorrt-*-cp311-none-linux_"$(uname -m)".whl
```

The first run compiles ONNX models into `.cache/tensorrt/`; later runs reuse compatible engines. Do not copy engines
between x86_64 and aarch64 machines.

## Run Depth HOI in MuJoCo

The recommended launcher starts MuJoCo and the policy together in one terminal:

```bash
# Default: largebox depth policy
bash scripts/run_sim_pipeline.sh

# Smalltable depth policy
bash scripts/run_sim_pipeline.sh hoi=depth/smalltable

# Run a specific motion without the selector
bash scripts/run_sim_pipeline.sh \
    hoi=depth/largebox \
    motion.name=sub16_largebox_013_v02
```

If `motion.name` is omitted, use Up/Down and Enter to select a motion. During locomotion:

- Up/Down commands forward/backward velocity.
- Left/Right commands yaw; Shift+Left/Right commands lateral velocity.
- `[` transfers locomotion to Track and interpolates the reference to HOI frame 0.
- `]` starts the HOI policy after Track reaches frame 0.
- Escape requests software shutdown; `Ctrl+C` stops the combined launcher.

To run the same setup in two terminals, give both processes the exact same HOI and motion:

```bash
# Terminal 1
python scripts/simulate.py --inspire \
    hoi=depth/largebox \
    motion.name=sub16_largebox_013_v02

# Terminal 2
python scripts/loco_body_hand_pipeline.py \
    deployment=sim \
    hoi=depth/largebox \
    motion.name=sub16_largebox_013_v02 \
    motion.interactive=false
```

## Locomotion-Only Smoke Test

The smaller pipeline is useful when checking DDS, odometry, keyboard control, or the locomotion checkpoint without HOI:

```bash
# Terminal 1
python scripts/simulate.py

# Terminal 2
python scripts/pipeline.py deployment=sim
```

## Synchronize the G1

Run synchronization from the workstation repository root. Without `--apply`, the command is a dry run:

```bash
bash scripts/setup/sync_unitree_wifi.sh
bash scripts/setup/sync_unitree_wifi.sh --apply
```

The normal apply mode removes remote source/assets that no longer exist locally while preserving the remote Git metadata,
environment, native libraries, TensorRT cache, and logs. Use `--no-delete` only when remote-only source files must remain.

## Run Depth HOI on the Real G1

On the robot:

```bash
ssh unitree-wifi
cd ~/G1-Playground
conda activate g1-playground

# Default largebox depth policy; motion is selected in the terminal
python scripts/loco_body_hand_pipeline.py \
    deployment=real \
    env.net_if=enP8p1s0

# Smalltable instead
python scripts/loco_body_hand_pipeline.py \
    deployment=real \
    env.net_if=enP8p1s0 \
    hoi=depth/smalltable
```

Replace `enP8p1s0` with the robot-facing interface when necessary. The real deployment automatically starts and stops the
repository Inspire service. The Unitree remote controls locomotion; the terminal still owns `[` and `]` policy switching.

Validate the depth observation from another workstation terminal while the real pipeline is running:

```bash
cd G1-Playground
conda activate g1-playground
python scripts/view_depth.py

# Equivalent explicit options
python scripts/view_depth.py --ssh unitree-wifi --port 9876 --scale 4
```

The viewer displays the policy's normalized `128×72` observation through SSH. Invalid pixels are black, with valid depth
shown from dark (near) to bright (far). Escape or closing the window stops only the viewer.

For a real locomotion-only check:

```bash
python scripts/pipeline.py deployment=real env.net_if=enP8p1s0
```

## Inspect Configuration

These commands resolve Hydra configuration without starting DDS or commanding hardware:

```bash
python scripts/loco_body_hand_pipeline.py --cfg job --resolve deployment=sim motion.interactive=false
python scripts/loco_body_hand_pipeline.py --cfg job --resolve deployment=real env.net_if=enP8p1s0 motion.interactive=false
```

The default HOI is `depth/largebox`. Available overrides are:

```text
hoi=depth/largebox
hoi=depth/smalltable
motion.name=<name stored in the selected NPZ bundle>
motion.interactive=false
recording.enabled=false
```

## Replay a Recording

```bash
# Replay the newest logs/state_* recording
python scripts/replay.py

# Replay one recording explicitly
python scripts/replay.py logs/state_20260822-183322
```

## Development Checks

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
ruff check g1_playground scripts tests
ruff format --check g1_playground scripts tests
pre-commit run --all-files
```

The focused suite contains nine hardware-free tests covering locomotion ordering, depth HOI observations/actions, camera
transport, the policy state machine, MuJoCo object initialization/support timing, and the DDS command watchdog.

## Configuration and Assets

- `configs/run_loco_hoi_track.yaml`: depth HOI pipeline defaults and transition timing.
- `configs/hoi/depth/`: largebox and smalltable policy, motion, object, and depth settings.
- `configs/deployment/sim.yaml`: Domain 1/`lo`, MuJoCo depth camera, and keyboard control.
- `configs/deployment/real.yaml`: Domain 0, robot NIC, D435i `480×270@60`, and local depth-preview endpoint.
- `assets/models/body_hand_distill/depth/`: TensorRT source ONNX models and external weights.
- `assets/motions/`: reference motion bundles containing robot, object, and contact trajectories.
- `assets/objects/`: MuJoCo object models selected by the HOI configuration.

## Safety

A learned policy can fall or command unexpected motion. Validate the exact policy, motion, gains, and transition sequence
in MuJoCo before hardware testing. Keep the robot supported, clear the workspace, and keep an independent hardware
emergency stop guarded. Escape and the Unitree remote's `A` button are software shutdown requests, not hardware stops.

## Documentation

- [Deployment architecture](docs/architecture.md)
- [Policy contract](docs/policy.md)
- [Controller mappings](docs/controller.md)
- [G1Env contract](docs/environment.md)
- [Unitree setup](docs/unitree_setup.md)

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
