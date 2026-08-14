# Pre-DDS Baseline

This report freezes the RoboJuDo runtime contract before DDS work begins. It is evidence for regression testing, not a
claim that the current hardware path is production-safe.

> [!NOTE]
> Phase 0 passed its final gate after the operator confirmed the controller baseline in both simulation and real
> deployment. The top-level vendor README deletion is intentional and the stale layout assertion has been removed.
>
> This historical record intentionally retains the former RoboJuDo project/distribution name and the former `-c g1` /
> `-c g1_real` commands. The current G1-Playground entry point uses `deployment=sim` / `deployment=real`; renaming the
> recorded Phase 0 identity or commands would misstate the frozen evidence.
>
> Likewise, “after policy-to-environment merging” below describes the captured implementation. The current runtime keeps
> pose/Kp/Kd only in the policy configuration and composes the effective G1 DoF values once before constructing `G1Env`;
> rewriting the historical wording or fixture would corrupt the baseline evidence.

## Scope and Recovery Point

- Captured: 2026-08-11 (Asia/Shanghai)
- Branch: `release`
- Runtime commit: `9f8bde4fb86c4168b92df14223871e5b2f3e4a85` (`clean`)
- Local recovery tag: `pre-dds` (annotated, points exactly to the runtime commit above)
- External references: `unitree_mujoco@ae6a8403e272733e9996ef59990880330496177f` and
  `G1-Playground@90ec9961a369ec9f1e8e034116cf9b31c1a7bb6e`

The runtime commit is one commit ahead of `origin/release`. No DDS source changes are included. Existing uncommitted
documentation edits are deliberately outside the recovery point and were not staged or rewritten by Phase 0.

## Frozen Contract

The machine-readable contract is in [`tests/fixtures/pre_dds/contract.json`](../tests/fixtures/pre_dds/contract.json),
with executable checks in [`tests/test_pre_dds_contract.py`](../tests/test_pre_dds_contract.py). It locks:

- the exact `g1` and `g1_real` component composition;
- 50 Hz policy/control timing and the MuJoCo 1 ms x 20 substep loop;
- HG `rt/lowstate` and `rt/lowcmd`, current hard-coded Domain 0, and a redacted machine-specific NIC;
- environment, policy, and XML actuator orders for all 29 joints, plus both adapter permutations;
- the effective runtime default pose, Kp, Kd, torque limits, and position limits after policy-to-environment merging;
- the ordered, field-major 5 x 96 observation history and the TorchScript 480-to-29 numerical output;
- a deterministic PD torque/clip calculation and one headless MuJoCo step;
- the checkpoint, XML, complete 37-file model/mesh closure, and both actual vendored source-tree closures; upstream
  revisions remain recorded as provenance.

Key asset fingerprints:

| Item | SHA-256 |
| --- | --- |
| `policy_wo_gait.pt` | `b5861e91bba86cdb35dc10de9764e26336042c817e179ddcebc3427828677414` |
| `g1_29dof_rev_1_0.xml` | `3d06ea42dc1fe4913481de24315c5d50539fbecb5833f284d40234c5c8654465` |
| 37-file asset closure | `d48420d284522462ab662ea9b405937494e9dee0f6b5efd9782865231c07f406` |
| 14-file `mujoco_viewer` closure | `cd06c5cb7e6121aaf3eec1c78260ae30eb9e2d8b1a4f7ce5066e9f9bdb132d36` |
| 14-file `unitree_cpp` closure | `372b53247eff7659a3139fbdd18e740b79d482dfd1369d1d1209e5a8847a1eec` |

## Toolchain Snapshot

- Python 3.11.15; RoboJuDo 1.5.0 editable install
- NumPy 2.4.6; PyTorch 2.13.0+cpu; MuJoCo 3.11.0; Pydantic 2.13.4; pygame 2.6.1
- Ruff 0.12.7; pre-commit 4.3.0
- GCC 11.4.0; CMake 3.31.7; Git 2.34.1; Linux 6.8.0-124-generic x86_64
- Native Unitree SDK2 2.0.0 and CycloneDDS 0.10.2 are installed under `/usr/local`; their Python modules are not
  installed. Vendored `mujoco_viewer` imports successfully, while the `unitree_cpp` Python extension is not built or
  installed.

## Verification Results

| Check | Result |
| --- | --- |
| Phase 0 contract tests | 5 passed |
| Complete unittest discovery | 14 passed, 1 skipped (15 total) |
| Ruff lint and format checks | passed |
| `git diff --check` (staged and unstaged) | passed |
| Headless G1 XML load and deterministic single step | passed |
| Manual `python scripts/run_pipeline.py -c g1` startup | pipeline/model/dry runs passed; stopped with Ctrl+C |
| Controller baseline in `g1` and `g1_real` | passed; reported by the operator |
| Agent-executed `g1_real` or hardware actuation | intentionally not run |

The skipped test requires an installed `unitree_cpp` binding. The agent-run simulation had no `/dev/input` joystick and
therefore exercised zero input only. The operator separately reports that the requested controller baseline passed in
both simulation and real deployment; this is operator evidence and the agent did not actuate hardware.

## Known Baseline Risks

- Domain 0 is hard-coded in vendored C++; it is not yet a typed configuration field.
- `joint2motor_idx=None` is only an identity-order assumption, not hardware evidence.
- `history_obs_size` reports zero although the effective model input is 480; the test freezes actual packing instead.
- The real deployment contains a machine-specific NIC in source. The fixture intentionally records only
  `<machine-specific-redacted>`.
- The real path does not enforce position or torque limits before sending targets, has no state-freshness watchdog, and
  software shutdown is not a hardware emergency stop.

Do not regenerate the numerical golden data after dependency changes without first reviewing the diff and explaining
the physical or numerical reason. Phase 1 must preserve this contract except for the explicitly planned Domain plumbing.
