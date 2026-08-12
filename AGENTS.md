# Repository Guidelines

## Scope and Architecture

RoboJuDo is limited to Unitree G1 29DoF locomotion. Top-level configurations are `g1` (MuJoCo) and `g1_real` (hardware);
both use `RlPipeline` and `UnitreeWoGaitPolicy`. They vary controller and environment; the policy, 29-joint contract,
50 Hz rate, observation layout, and action semantics stay shared. Read
`docs/architecture.md` before changing this boundary. Do not reintroduce partial-DoF policies, motion tracking, other
robots, or extra registries without an explicit architecture decision and tests.

## Project Structure

- `robojudo/config/g1/`: runtime compositions, joints, gains, limits, and network settings.
- `robojudo/{controller,environment,pipeline,policy}/`: runtime interfaces and implementations.
- `assets/robots/g1/` and `assets/models/g1/unitree/`: the 29DoF XML/meshes and sole TorchScript checkpoint.
- `scripts/`: the launcher and vendored-package installer.
- `docs/`: architecture, component contracts, and hardware safety procedures.
- `tests/test_full_imports.py`: registry allowlists, 29DoF invariants, asset closure, and model-shape tests.
- `third_party/`: tracked vendors, not Git submodules. Preserve recorded revisions/licenses, do not run
  repository formatters over them, and document direct vendor changes in the relevant phase/design record. Keep the
  pre-DDS provenance and source-tree hashes in `tests/fixtures/pre_dds/contract.json` immutable.

## Setup, Run, and Validation

```bash
python -m pip install -e ".[dev]"
python scripts/install_third_party.py
python scripts/run_pipeline.py -c g1
python -m unittest discover -s tests -p 'test_*.py' -v
ruff check robojudo scripts tests
ruff format --check robojudo scripts tests
pre-commit run --all-files
```

The simulation command is a manual check requiring a display and controller, not an automated test. The unittest suite is
a static contract gate; it does not exercise GLFW, the 50 Hz loop, preparation, DDS, or hardware.

Real deployment requires Unitree SDK2, CycloneDDS, the vendored `unitree_cpp` binding, and the correct
robot-facing interface. Follow `docs/unitree_setup.md`; never infer hardware readiness from an import test.

## Code and Review Conventions

Use Python 3.11+, four-space indentation, modern annotations, `snake_case` names, `PascalCase` classes, and `Cfg` suffixes
for configuration models. Ruff enforces 120-character lines and E/F/I/B/UP rules. Keep commits focused with short,
imperative subjects. Pull requests must describe the affected contract and exact validation performed.

`tests/test_full_imports.py` intentionally uses exact registry and asset allowlists; do not loosen them merely to make a new
component pass. On Jetson, preserve the aarch64 OMP setting and torch-before-numpy import workaround in `robojudo/__init__.py`
unless validated on target hardware.

Treat joint order, observation packing, action scaling, PD gains, loop timing, DDS topics, shutdown behavior, and network
interfaces as safety-critical. Validate the exact checkout in `g1` simulation first. Never run `g1_real` as an automated
or CI check; it requires explicit operator authorization and a completed preflight. Software shutdown and tilt checks
never replace the independent hardware emergency stop.
