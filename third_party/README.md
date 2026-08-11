# Vendored Dependencies

The directories in `third_party/` are ordinary tracked source trees, not Git submodules. Install them through the shared
installer from the repository root:

```bash
python scripts/install_third_party.py                  # MuJoCo viewer only
python scripts/install_third_party.py unitree_cpp      # Unitree SDK2 binding
python scripts/install_third_party.py mujoco_viewer unitree_cpp
```

| Directory | Upstream | Vendored revision | License |
| --- | --- | --- | --- |
| `mujoco_viewer/` | https://github.com/rohanpsingh/mujoco-python-viewer | `31af945ef640d276a3bafe5e9b858d80c46c8509` | BSD-2-Clause |
| `unitree_cpp/` | https://github.com/HansZ8/unitree_cpp | `222028cdaa79cdd7c7cda6645ebd2b02d201be0c` (1.0.3) | CC BY 4.0 |

The MuJoCo viewer contains direct local changes previously carried as a patch: optional key callbacks, persistent markers
addressed by ID, MuJoCo-compatible marker initialization, and explicit GLFW context activation before rendering. These
changes are maintained directly in `mujoco_viewer/mujoco_viewer/mujoco_viewer.py`.

`unitree_cpp` is an unmodified snapshot. Building it requires Unitree SDK2, CycloneDDS, CMake, a C++17 compiler,
`scikit-build-core`, and `pybind11`; see its vendored README for details. Preserve both upstream license files when updating
either dependency, and record the new source revision and any local modifications here.
