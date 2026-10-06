# Current Go2-W training

`GO2WCfg` and `GO2WCfgPPO` in
[go2w_config.py](../robot_gym/envs/go2w/go2w_config.py) are the single current
parameter source. Edit that file for the next experiment. Training resolves the
registered task and writes the complete `config.yaml` into its run directory.
No documentation YAML, historical profile, or fine-tune selector supplies parameters.

The old path was inherited Go2 defaults → Go2-W defaults → profile deltas →
fine-tune deltas → a manually maintained recipe YAML → runtime. The current path is
**Go2-W config → runtime → saved run config**. Go2-W configuration no longer inherits
Go2 configuration. The shared robot mechanisms and Dodo/Go2 configurations are unchanged.

The ordinary production command below is **prepared and unexecuted**. Run it from
the repository root with the existing `genesis-gpu` environment active:

```powershell
python -m robot_gym.scripts.train --task go2w --num_envs 4096 --max_iterations 2000 --seed 1 --logger tensorboard --training_diagnostics --headless --rl_device cuda:0
```

Fresh training initializes actor, critic, empirical normalizers, Gaussian std and
Adam from scratch at local iteration zero. The default output is
`logs/go2w/go2w_<timestamp>/`; each run receives its own resolved configuration.
The budget remains 2,000 updates, with saves every 250 and the native final checkpoint.

For continuing a current run, ordinary `--resume --load_run <saved-directory>`
selects its saved config and full native checkpoint state. `--checkpoint` can select
a label explicitly; `--max_iterations` is the number of additional native updates.
The native saved iteration label is retained. Output goes to a new timestamped
directory, never inside the parent run. No optimizer, std, or normalizer reset is
performed on resume. Changes to the current source defaults do not replace saved
numeric values. Missing required saved settings fail instead of inheriting defaults.

## Canonical reference and unchanged behavior

`JOINT_REFERENCE` is the one numeric runtime source for hip 0, thigh 0.70, calf
−1.40 and foot 0 rad. `joint_reference()` expands it identically to FL, FR, RL and RR.
Action-zero positions, observation position errors, the actual-leg pose objective,
reset nominal positions and cached measured-URDF FK all use that reference.
Reference height is derived from the measured wheel cylinders and FK:
0.4277416561558192 m. Adding the 0.003 m reset clearance gives spawn z
0.4307416561558192 m. This is reference geometry, not a loaded equilibrium claim.

| Behavior field group | Comparison with the frozen prepared production config |
| --- | --- |
| Asset, measured URDF and SHA, link/joint order | Exact |
| Four-leg reference, reference height, spawn, corridor FK reference | Exact; now derived from one reference |
| Observations and actions | Exact: 58 inputs, 16 actions; demand-conditioned phase |
| P-leg/V-wheel control, gains, limits, armature, action delay | Exact; leg offsets .30/.35/.40 rad, wheels 18 rad/s, clipping ±1 |
| Required command envelope | Exact: vx [−.30, 1.00], vy [−.30, .30] m/s, yaw [−1, 1] rad/s |
| Training reserve | Exact: vx [−.40, 1.20], vy [−.50, .50] m/s, yaw [−1.5, 1.5] rad/s |
| Command families, tiers, small commands, short/long durations | Exact; sampling config renamed and holds moved under commands |
| Push distribution and duration/interval | Exact: 80/15/5% at 20–50/50–100/100–150 N, 5–10 s intervals, .10–.20 s duration |
| Other domain randomization, reset noise and observation noise | Exact |
| Phase period, diagonal timing, apex and clearance/support | Exact |
| All 21 active reward scales and their parameters | Exact; inactive objectives and knobs removed |
| PPO, networks, normalizers and Gaussian distribution | Exact; fixed LR 3e−4, gamma .995, lambda .95, entropy .003, initial std .40, bounds [.10, .70] |
| Physics/policy timing and rollout | Exact: .005/.020 s, 4096 × 64 |
| Training budget and saving | Exact: 2000 updates, interval 250, native final |

The family order is stand/straight/arc/yaw/precision/lateral/mixed with shares
.15/.20/.07/.20/.10/.20/.08. Ordinary moving families retain .80 core, .10 required
edge and .10 reserve draws; stand and precision stay in core. A reserve command
has only one selected active axis outside core. `commands.sampling` contains all
reserve and precision limits; `commands.holds` contains the retained long holds.
There is one sampler. Seeded commands, duration ticks, family labels, tiers, holds
and the final RNG state match the before version bit for bit.

There is one Go2-W push sampler. One magnitude tier and horizontal direction are
drawn per event, held over its sampled physics ticks and reapplied at base-link COM.
No vertical force or torque is added. The force norm remains at most 150 N;
duration ticks and force impulse remain in diagnostics. Generic Dodo/Go2 pushes
are untouched. No reward, phase, physics or optimization tuning was performed.

## Cleanup and saved-run compatibility

Removed the Go2-W training profile/fine-tune functions and numeric presets for
step recovery, event steps, transfer V1/V2/V3, sensor refinements, navigation
variants, coverage/mobility/precision fine-tunes and the fixed-LR continuation
branch. Their CLI selectors, selective-loading logic and recipe-loader tests are
gone. The documentation recipe YAML and PowerShell wrapper have been removed.
Current phase, sensor, tracking, pose, geometry and contact mechanisms remain;
Gaussian std projection moved from the event-step module into `phase.py`.
The high-rate diagnostic recorder and spectrum helper moved into `diagnostics.py`.
Obsolete event training, inference braking and published-policy preparation
launchers were removed. Shared control and physics files were not edited.

Complete saved current configs are authoritative. A small structural adapter also
accepts the final prepared `reference_tracking_v1` configs with the same sampler,
push and reward mechanisms: it renames sampling/hold fields and carries their
saved numbers. It contains no historical numeric definitions. Incompatible or
incomplete saved configs fail explicitly.

**Intentional limitation:** earlier Go2-W checkpoints requiring the removed
samplers or rewards, including ORIGINAL1499 and final998, need the pre-cleanup
source checkout `e4f622f91f1cdc1a4996650aa031e5ecbf8d6545` for native replay.
They are never silently run with current task defaults. Their weights, saved
configs, reports and evaluation artifacts remain unchanged:

- `logs/go2w_transfer_v3/transfer_v3_seed1_2026-10-03_16-18-34/model_1499.pt`
- `logs/go2w_transfer_v3_navigation_rolling_control_fixed_lr_stage2/lr_only_from499_stage2_seed1_2026-10-06_13-28-54/model_998.pt`

The historical [reference-tracking preparation report](go2w_reference_tracking_v1.md)
remains research evidence. Its old recipe-loader commands are superseded by this
document. No research evidence or historical generated run config was removed.

## Validation evidence, 2026-10-06

The complete production configuration was frozen **before implementation edits**
in [before_config.yaml](../evaluation/go2w_single_config_cleanup/before_config.yaml).
The ordinary command resolves to
[after_config.yaml](../evaluation/go2w_single_config_cleanup/after_config.yaml).
These local evidence snapshots are outputs, never training inputs.

[equivalence.json](../evaluation/go2w_single_config_cleanup/equivalence.json)
lists every compared field: 383 exact matches, nine equal renamed fields,
89 removed inactive/provenance fields, two output-name changes and one added
serialization-version field. There are no changed numerical behavior fields.
The 4,096-command deterministic sample and 1,001 push draws match exactly.
The ten tracking/pose/phase kernel ASTs match exactly. Both other registered task
configs match; 105 protected file hashes and all 3,144 pre-existing log/evaluation
file sizes/timestamps match. This includes `play.py` and both reference checkpoints.

39 focused tests passed (including the corrected diagnostic fixture recheck). They
cover action offsets, clipping/delay, nominal reset/observation/pose
agreement, all command families/tiers/signs and small commands, strong pushes,
contact deduplication, active rewards, dt scaling, phase and sensor transport,
fresh initialization guards, strict saved-config restoration and shared replay.
The retained PPO instrumentation test verifies identical losses, parameters and
RNG with diagnostics enabled. It uses the current signed-reward convention.
[Validation summary](../evaluation/go2w_single_config_cleanup/validation_summary.json)
indexes the test logs; [source checks](../evaluation/go2w_single_config_cleanup/static_checks.json)
record syntax, removed-name and module-import results. `git diff --check` passes.

Exactly one native **4096 × 64, two-update smoke** completed:
[smoke_after.json](../evaluation/go2w_single_config_cleanup/smoke_after.json).
Before updates, checkpoint loading was forbidden, Adam was empty, normalizer
counts and iteration were zero, and std was .40. After two updates, actor/critic
values were finite, Adam had 80 steps, both normalizers counted 524,288 samples,
and LR remained fixed at 3e−4. Native final checkpoint:
`logs/go2w_single_config_smoke/two_updates_2026-10-06_20-05-46/model_1.pt`.

A separate newly constructed CPU runner then restored this checkpoint through
ordinary resume configuration and the native loader:
[resume.json](../evaluation/go2w_single_config_cleanup/resume.json).
Actor, critic, normalizers, learned std, every Adam tensor/step/LR and iteration
label 1 matched exactly. This restore used no further physics ticks or optimizer
updates. No policy-quality evaluation or production training was performed.

This validates configuration and execution equivalence, not locomotion success.
The existing tracking, heading/path, holding/stopping, posture, wheelbase,
clearance, sensor and contact criteria remain the future comparison criteria.
Original1499 and final998 remain the historical comparison artifacts.

## Changed and deleted source files

The following inventory includes adapted tests and moved reusable diagnostics;
it excludes the new ignored local validation evidence and smoke run.

| Status | File |
| --- | --- |
| Added | `docs/go2w_training.md` |
| Added | `tests/test_go2w_current_config.py` |
| Added | `tests/test_go2w_phase.py` |
| Added | `tests/test_go2w_rewards.py` |
| Added | `tests/test_go2w_sensors.py` |
| Deleted | `docs/go2w_reference_tracking_v1.ps1` |
| Deleted | `docs/go2w_reference_tracking_v1.yaml` |
| Deleted | `robot_gym/envs/go2w/native_reference.py` |
| Deleted | `robot_gym/envs/go2w/step_events.py` |
| Deleted | `robot_gym/envs/go2w/zero_command_brake.py` |
| Deleted | `robot_gym/scripts/native_reference.py` |
| Deleted | `tests/go2w_pose_probe.py` |
| Deleted | `tests/test_go2w_contract.py` |
| Deleted | `tests/test_go2w_event_step.py` |
| Deleted | `tests/test_go2w_fresh_recipe.py` |
| Deleted | `tests/test_go2w_navigation_recipe.py` |
| Deleted | `tests/test_go2w_reference_tracking.py` |
| Deleted | `tests/test_go2w_resume.py` |
| Deleted | `tests/test_go2w_rolling_control.py` |
| Deleted | `tests/test_go2w_rolling_placement.py` |
| Deleted | `tests/test_go2w_sensor_smooth.py` |
| Deleted | `tests/test_go2w_symmetry_integration.py` |
| Deleted | `tests/test_go2w_transfer.py` |
| Deleted | `tests/test_go2w_transfer_v2.py` |
| Deleted | `tests/test_go2w_transfer_v3.py` |
| Deleted | `tests/test_native_reference.py` |
| Modified | `README.md` |
| Modified | `robot_gym/envs/go2w/deployment.py` |
| Modified | `robot_gym/envs/go2w/diagnostic_bank.py` |
| Modified | `robot_gym/envs/go2w/diagnostics.py` |
| Modified | `robot_gym/envs/go2w/evaluate.py` |
| Modified | `robot_gym/envs/go2w/go2w_config.py` |
| Modified | `robot_gym/envs/go2w/go2w_env.py` |
| Modified | `robot_gym/envs/go2w/go2w_symmetry.py` |
| Modified | `robot_gym/envs/go2w/phase.py` |
| Modified | `robot_gym/envs/go2w/training_diagnostics.py` |
| Modified | `robot_gym/scripts/train.py` |
| Modified | `robot_gym/utils/diagnostics.py` |
| Modified | `robot_gym/utils/task_registry.py` |
| Modified | `tests/go2w_pose_viewer.py` |
| Modified | `tests/test_diagnostics.py` |
| Modified | `tests/test_fixed_command.py` |
| Modified | `tests/test_inference.py` |
| Modified | `tests/test_shared_pipeline.py` |
