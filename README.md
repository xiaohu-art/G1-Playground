# RoboJuDo

RoboJuDo is a focused deployment runtime for the **Unitree G1 29DoF** and the
`UnitreeWoGaitPolicy` locomotion checkpoint from
[unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab). The checkpoint and policy contract stay identical in
MuJoCo and on a real G1; input, execution backend, startup lifecycle, and safety configuration differ.

| Configuration | Controller | Environment | Policy |
| --- | --- | --- | --- |
| `g1` | `JoystickCtrl` | `MujocoEnv` | `UnitreeWoGaitPolicy` |
| `g1_real` | `UnitreeCtrl` | `UnitreeCppEnv` | `UnitreeWoGaitPolicy` |

The policy observes and commands all 29 joints. Its five-frame observation history has 480 values, and the model emits
29 joint actions. The checkpoint is stored at `assets/models/g1/unitree/policy_wo_gait.pt`.

## Installation

Python 3.11 or newer is required.

```bash
git clone https://github.com/HansZ8/RoboJuDo.git
cd RoboJuDo
conda create -n robojudo python=3.11 -y
conda activate robojudo
python -m pip install -e .
python scripts/install_third_party.py
```

The last command installs the vendored MuJoCo viewer used by `MujocoEnv`. Both external source trees are tracked directly
under `third_party/`; no Git submodule initialization is required.

For development tools, install the optional extra:

```bash
python -m pip install -e ".[dev]"
```

## Run in MuJoCo

Connect an Xbox-compatible controller, then run the default `g1` configuration:

```bash
python scripts/run_pipeline.py
# Equivalent:
python scripts/run_pipeline.py -c g1
```

The left stick commands forward/backward and lateral velocity. Horizontal movement of the right stick commands yaw.
Press `A` to request shutdown. See [Controller](docs/controller.md) and [Environment](docs/environment.md) for the exact
runtime contract.

## Run on a Real G1

> [!CAUTION]
> A learned policy can fall, accelerate unexpectedly, or command damaging joint motion. Validate the exact checkpoint,
> configuration, and robot model in MuJoCo first. Clear the operating area, use an approved support arrangement, keep a
> trained operator at an independent hardware emergency stop, and begin with zero velocity commands. Never deploy near
> people or fragile equipment.

> [!WARNING]
> `A` sends RoboJuDo's software shutdown command, stops further application commands, and invokes the SDK shutdown path.
> It is not a substitute for the robot's independent hardware emergency stop. Do not disable
> `do_safety_check` in `g1_real`.

Install Unitree SDK2 and the `unitree_cpp` binding, then configure the robot-facing network interface as described in the
[G1 setup guide](docs/unitree_setup.md). Start the real configuration only after completing its safety checklist:

```bash
python scripts/run_pipeline.py -c g1_real
```

The Unitree remote uses the same velocity axes as the simulation controller. Keep your hand on the hardware emergency
stop throughout the run; press `A` for software shutdown at the first sign of instability.

## Project Layout

- `robojudo/config/g1/`: the `g1` and `g1_real` compositions and their 29DoF settings.
- `robojudo/policy/`: `UnitreeWoGaitPolicy` inference and observation construction.
- `robojudo/controller/`: Xbox and Unitree remote input.
- `robojudo/environment/`: MuJoCo and UnitreeCpp execution backends.
- `assets/robots/g1/`: the G1 29DoF MuJoCo model and meshes.
- `assets/models/g1/unitree/`: the locomotion checkpoint.
- `third_party/`: vendored MuJoCo viewer and UnitreeCpp source trees with pinned revisions.
- `scripts/install_third_party.py`: the shared installer for vendored Python packages.
- `scripts/run_pipeline.py`: the single simulation and real-robot launcher.

## Documentation

- [Sim2Sim / Sim2Real deployment architecture](docs/architecture.md)
- [Policy contract](docs/policy.md)
- [Controller mappings](docs/controller.md)
- [Environment contract](docs/environment.md)
- [UnitreeCpp installation and real-robot safety](docs/unitree_setup.md)

## Development and Validation

Run the repository checks from the project root:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
pre-commit run --all-files
```

Changes to joint order, observation layout, action scaling, gains, limits, shutdown handling, or network transport are
safety-critical. Document the exact simulation validation performed before proposing a real-robot test.

## License and Citation

The repository is distributed under the
[Creative Commons Attribution-NonCommercial 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/).

```bibtex
@misc{RoboJuDo,
  author = {Zihan Zhuang, Yi Dong, Peng Li},
  title = {RoboJuDo: A deployment framework for the Unitree G1 29DoF},
  url = {https://github.com/HansZ8/RoboJuDo},
  year = {2025}
}
```

Related projects: [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab),
[Unitree SDK2](https://github.com/unitreerobotics/unitree_sdk2), and
[UnitreeCpp](https://github.com/HansZ8/unitree_cpp).
