# Current Go2-W training

`GO2WCfg` and `GO2WCfgPPO` in
[go2w_config.py](../robot_gym/envs/go2w/go2w_config.py) are the single current
parameter source. Edit that file for the next experiment. Training resolves the
registered task and writes the complete `config.yaml` into its run directory.
No documentation YAML, historical profile, or fine-tune selector supplies parameters.

**Current run review, 2026-10-08:** training schema 2 adds straight-motion geometric
supervision and a 65-input asymmetric critic; the deployed actor remains 58 inputs.
Yaw tracking is restored to `.8`. See the [task definition and validation below](#geometric-straight-motion-preparation-2026-10-08).
The original budget is now complete: the interrupted run resumed from CK1000
and reached 2000 lineage updates. The [completed-budget review below](#completed-geometric-budget-review-2026-10-08)
rejects final1998 for promotion because holding/stopping regressions outweigh its
gains. Retain geometric CK1000 as the schema2 development baseline; no additional
training or reward change is prepared. Pre-geometric CK1999 remains a comparison.

The old path was inherited Go2 defaults → Go2-W defaults → profile deltas →
fine-tune deltas → a manually maintained recipe YAML → runtime. The current path is
**Go2-W config → runtime → saved run config**. Go2-W configuration no longer inherits
Go2 configuration. The shared robot mechanisms and Dodo/Go2 configurations are unchanged.

The ordinary fresh-training syntax below is retained for reference. The geometric
budget is closed; this syntax is not a recommendation to start another run.
Commands run from the repository root:

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

## Canonical reference and initial cleanup record

`JOINT_REFERENCE` is the one numeric runtime source for hip 0, thigh 0.70, calf
−1.40 and foot 0 rad. `joint_reference()` expands it identically to FL, FR, RL and RR.
Action-zero positions, observation position errors, the actual-leg pose objective,
reset nominal positions and cached measured-URDF FK all use that reference.
Reference height is derived from the measured wheel cylinders and FK:
0.4277416561558192 m. Adding the 0.003 m reset clearance gives spawn z
0.4307416561558192 m. This is reference geometry, not a loaded equilibrium claim.
The following equivalence table records the earlier single-config cleanup. The
schema-2 changes are listed separately below; its critic and active reward count differ.

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

Complete saved current-schema configs are authoritative. Schema 2 rejects older
training/replay schemas before checkpoint loading instead of substituting new
reward or critic semantics. The former structural adapter for prepared
`reference_tracking_v1` snapshots is superseded by this explicit schema boundary.
Use the corresponding historical source checkout for those checkpoints; retain
their raw traces for comparisons. Incomplete current saved configs also fail.

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

## Production policy review, 2026-10-07

**Preserve the final candidate; it is not yet qualified against the frozen
path/heading/holding requirements.** Rolling posture, forward speed and stopping
have improved, but sustained lateral commands still curve substantially. The
single recommended next refinement is described below; this review changed no
training code or settings and launched no training.

The production run is `logs/go2w/go2w_2026-10-07_09-40-23`, initialized fresh and
completed with **2,000 updates**, not 200. Its exact final checkpoint is
`model_1999.pt` (SHA-256
`f1196ee51431ee430f3ed91f2ec0dba9bae1672f9bd3dae93e04677266dfbe58`).
Saved budget and completion record both say 2,000; checkpoint metadata says label
1999, Adam has 80,000 steps, and each normalizer has 524,288,000 samples.
Retained labels are 0, 250, 500, 750, 1000, 1250, 1500, 1750 and 1999. Thus the
earlier preparation command above was subsequently run by the user.

The saved YAML loads through the current strict replay path. It confirms four
identical `0/.70/-1.40/0` leg references, reference/spawn heights
`.4277416561558192/.4307416561558192 m`, fixed-reference P offsets and V-wheel
targets, scales `.30/.35/.40/18`, clipping ±1, 58 observations and 16 actions.
Tracking remains three independent requested-minus-actual body-axis kernels:
`w[1-Huber(e/b)-beta(1-exp(-.5(e/p)^2))]`, with
`w=[1,1,.8]`, `b=[.25,.15,.35]`, `p=[.03,.03,.03]`, `beta=[.25,.25,.25]`.
The actual 12-joint pose objective, demand-conditioned clock, .8 s diagonal
phase/.65 stance/.04 m apex and clearance/support guidance are present. Saved
core/edge/reserve sampling and 80/15/5 push-force tiers match the current design.
Native LR is fixed at `3e-4`; learned std bounds remain `[.10,.70]`. This was a
saved-run inspection, not another source configuration-equivalence audit.

### Retained training diagnostics

| Updates | Measured KL mean / p95 | PPO ratio clipping | Value loss | Tracking x / y / yaw rates |
| --- | ---: | ---: | ---: | --- |
| 1000–1199 | .01331 / .01849 | 24.69% | .01343 | .8996 / .9038 / .5882 |
| 1800–1999 | .01366 / .02032 | 23.83% | .01216 | .9052 / .9138 / .6003 |

Each window has 200 diagnostic records; the first has 199 retained TensorBoard
scalars. Six earlier JSONL labels are missing. All 16 joints have **zero std-floor
occupancy** in both windows; final std is .1054–.1519. Per-joint std, floor
occupancy and sampled/deterministic saturation are in the comparison JSON.
Rear-calf deterministic-mean saturation rises from about 3.76% to 5.50%; most
other joints decrease. Phase-clearance, support, pose and sensor cost rates
improve modestly; this does not establish their causal effects on the policy.

Command-family time shares are similar: late stand/straight/arc/yaw/precision/
lateral/mixed = 17.59/16.51/5.86/22.49/5.30/23.11/9.14%. Long and extended holds
occupy 7.98% and 5.85% of rollout time. Mean per-update push-tier shares are
80.34/14.74/4.93%, duration 30.03 physics ticks (~.150 s), impulse 6.80 N·s.
These are retained event summaries, not pooled event counts. Realized command-tier
time and failure-to-push/reserve associations were not recorded.
The contact-safety raw count averages .00000372 → .00000462 per environment tick.
Termination TensorBoard rates are −.00138 → −.00114, normalized by the configured
60 s episode length and averaged over resetting environments; they are not
failure counts or probabilities. Broad/precision training components were not
logged separately. In particular, yaw's .8 reward ceiling makes .6003 neither
percentage accuracy nor an invertible RMSE.

### Bounded nominal panel

One invocation evaluated final1999 at deterministic policy mean, nominal measured
dynamics, seed 1, no observation noise, pushes or DR, and unchanged saved phase
semantics. Stand is 30 s. Every moving case is **3 s zero + 30 s command + 8 s
zero**, with commands below. Existing 200 Hz capture was enabled only for forward
and positive lateral; comparisons otherwise use 50 Hz. An initial sandbox DLL
import failure occurred before simulation; the native retry completed all cases.

| Case / command | Full mean [vx, vy, body wz] | Full RMSE [vx, vy, body wz] | Full stop endpoint / path, m | Last 2 s stop XY / wz RMS |
| --- | --- | --- | --- | --- |
| Stand [0,0,0] | [.00326,.00005,.00159] | [.00380,.00217,.01337] | — | — |
| Forward [.5,0,0] | [.49579,.00014,.00251] | [.01622,.00064,.00309] | .05749 / .06672 | .00467 / .01725 |
| Lateral+ [0,.3,0] | [.00900,.30049,−.00767] | [.01306,.01531,.04571] | .01707 / .03714 | .00188 / .00164 |
| Lateral− [0,−.3,0] | [.00972,−.30088,.00829] | [.01332,.01441,.04349] | .03208 / .04648 | .00401 / .00370 |
| Yaw+ [0,0,.8] | [.00522,−.00308,.79956] | [.00639,.00991,.04100] | .01125 / .02869 | .00393 / .00797 |
| Yaw− [0,0,−.8] | [.00599,.00373,−.80178] | [.00727,.00976,.03862] | .04422 / .06486 | .00582 / .03192 |

Units are m/s for linear velocity and rad/s for body angular-z. All first-second,
full, late, complete-stop and final-two-second vectors, biases and RMSEs are saved
in [the compact comparison JSON](go2w_policy_review_2026-10-07.json).
No case fell, reset or recorded undesired self/non-wheel contact at 50 Hz. The two
200 Hz captures also have no non-wheel contact above 8 N. This is sampled evidence,
not qualification of the expanded envelope.

Forward meets the retained .45–.55 mean / ≤.04 RMSE speed band, but its
**.50829 m cross-track / 4.242° heading deviation** miss the unchanged
**.05 m / 2°** requirements. ORIGINAL1499 gives .12865 m / 2.916°; final998 gives
1.62938 m / 9.991°. Neither baseline makes the frozen limits optional. Current
forward stop path .06672 m improves on original .07945, final998 .10186 and the
recorded CK499 .1051 m, although its endpoint is farther than either baseline.
It has no backward crossing of the stop-onset line and only .00075 m maximum
backtrack. Residual angular motion remains, especially after negative yaw.

Stand's 20–30 s XY RMS is .00439 m/s, below .005, but body-wz RMS .01117 rad/s
does not improve on the retained CK499 .00670 rotational reference. Full heading
change is +2.735° (maximum deviation 3.049°); 0–5 s endpoint/path is
.00567/.01595 m. Thus translation alone does not qualify holding.

### Lateral drift is continuing curvature plus forward bias

| Case / window | Mean [vx, vy, body wz] | RMSE [vx, vy, body wz] |
| --- | --- | --- |
| +, first second | [.00221,.27860,−.01117] | [.01224,.04313,.07493] |
| +, full 30 s | [.00900,.30049,−.00767] | [.01306,.01531,.04571] |
| +, last 10 s | [.01022,.30278,−.01100] | [.01362,.01330,.04338] |
| −, first second | [.00970,−.28232,.01791] | [.01354,.03905,.04503] |
| −, full 30 s | [.00972,−.30088,.00829] | [.01332,.01441,.04349] |
| −, last 10 s | [.01030,−.30067,.00694] | [.01323,.01285,.04389] |

Bias is mean minus the respective `[0,±.3,0]` request, including zero cross axes.
Using the full quaternion rotation and the horizontal forward/lateral axes at
command onset gives:

| Lateral sign | Pose displacement [initial forward, lateral], m | Forward integral: body-x + rotated body-y + body-z, m | Heading first 1 s / full / last 10 s |
| --- | --- | --- | --- |
| + | [1.61843, 8.83435] | .26509 + 1.34959 + .00489 | −.777° / −18.813° / −7.781° |
| − | [1.75826, −8.81700] | .28594 + 1.46761 + .00608 | +1.133° / +18.176° / +5.391° |

Maximum cross-track from the onset lateral line equals 1.61843/1.75826 m.
Velocity integration agrees with pose displacement within 1.14/1.37 mm on that
axis. About 83% comes from rotating body-y motion, with ~16% direct body-x bias;
tilt/body-z translation is small. The large late heading change rules out a purely
initial orientation error. Oscillatory yaw coexists with directional bias.
Mean Euler heading rates are −.01113/+.01123 rad/s, different from body-wz
−.00767/+.00829 because roll/pitch motion contributes when tilted; heading was
measured from unwrapped pose, not integrated body-wz.

Weighted broad / precision costs per second on full lateral+ are
x `.00136/.02059`, y `.00520/.02361`, yaw `.00682/.09837`; lateral− gives
`.00142/.02145`, `.00460/.02142`, `.00618/.09257`. Late yaw costs remain
`.00615/.09494` and `.00629/.09164`. All-window components are in the JSON.
These instantaneous costs detect error but do not directly enforce a long-horizon
path; their sizes are not policy gradients.

The single allowed earlier check used `model_1000.pt` (1,001 completed updates),
positive lateral only, with the same nominal schedule. It gives **5.44848 m**
cross-track, **−76.940°** heading change, mean body-wz −.04246 and RMSE .06760.
Final1999 improves this to 1.61843 m, −18.813°, −.00767 and .04571. This supports a
persistent learned compromise rather than deterioration confined to late updates.
Matched ORIGINAL1499 lateral traces are substantially straighter (.26958/.55166 m
maximum cross-track); final998 has no retained matched strong 30 s lateral/yaw
panel. Its short ±.1/±.4 cases are not substituted. Historical policies were not
replayed; matched original command/phase/target arrays and nominal plant settings
were verified against the retained artifacts.

### Steps, posture and sensors

Per-wheel values below are FL/FR/RL/RR; physical swing intervals use the existing
6/10 N load hysteresis and ≥2 mm collision-cylinder peak, with boundary events
censored. There are 37 complete desired phase swings per wheel in each moving
stepping window. Desired apex stays 40 mm.

| Case | Completed physical intervals | Median physical peaks, mm | Censored intervals | Unloaded ≤6 N near desired apex, % |
| --- | --- | --- | --- | --- |
| Lateral+ | 33/45/42/37 | 10.87/10.81/8.18/15.87 | 1/1/2/0 | 91.1/89.2/97.3/97.4 |
| Lateral− | 40/37/37/50 | 11.79/9.80/16.70/6.54 | 0/2/1/1 | 93.7/90.8/100/98.9 |
| Yaw+ | 33/39/37/37 | 4.21/17.36/9.49/13.24 | 1/1/1/1 | 82.6/96.2/98.9/98.4 |
| Yaw− | 39/33/37/38 | 18.12/4.62/11.52/7.63 | 1/1/1/0 | 96.8/89.2/97.8/97.9 |

This is shallow clearance despite substantial unloading, not evidence that every
low liftoff/touchdown sample drags. Near desired apex (target ≥32 mm), lateral+
actual means are only 7.27/7.11/6.00/13.48 mm. Original lateral medians were about
20–38 mm. Lateral physical peaks within desired swing windows typically lead the
desired apex by 20–100 ms. Positive-lateral 200 Hz peaks remain shallow, and brief reloads split
intervals: geometric counts become 34/79/87/38, with many additional load-only
events. Counts therefore are not interchangeable with useful gait cycles.
Fewer than two wheels exceed 6 N on .317% of positive-lateral physics ticks,
which policy-rate capture misses. Loaded lateral cylinder-center RMS is
.064–.090 m/s across lateral wheels; this is a sliding surrogate, not tire-patch
slip measurement. Peak distributions, contact duties and timing remain in the
saved traces/analysis.

Positive lateral clips RR-calf action on 39.4% of samples (negative lateral:
RL-calf 37.1%), in repeated bursts up to .24/.18 s. Positive RL-thigh clipping is
10.7%. No sampled actuator force reaches 99% of its limit; positive-lateral
substep control-force peak is 66.2% of limit. Actual-versus-applied rear-calf target
RMSE is .266/.288 rad (+) and .292/.265 rad (−). Issued and applied targets agree
under nominal zero delay; clipping and load-following error remain distinct.

At positive-lateral desired-apex versus all-stance samples, reward rates are:
phase clearance −.1253/−.00655, support −.01428/−.00976, pose −.02194/−.01555,
sensor −.00794/−.00991, roll/pitch-rate −.01343/−.01969, leg action-rate
−.000719/−.001197; tracking x/y/yaw = .9630/.9706/.6963 versus
.9929/.9732/.6965. Other windows and terms are in the JSON. Co-occurrence does not
prove sensor or pose penalties caused the shallow steps. The apex was not raised.

| Late window | Actual leg-reference RMS, rad | Height, m | Wheelbase, m | Front / rear track width, m |
| --- | ---: | ---: | ---: | --- |
| Stand | .05423 | .41701 | .41450 | .40911 / .43665 |
| Forward | .08841 | .41629 | .47904 | .40996 / .43343 |
| Lateral+ | .14729 | .41272 | .34324 | .40166 / .34974 |
| Lateral− | .14736 | .41303 | .34243 | .40138 / .34871 |

Reference geometry is wheelbase .38680 m and both widths .38020 m, distinct
dimensions obtained from cached measured-URDF FK. Pair midpoints and actual joint
errors are retained. Forward height is .41990 m in the first commanded second
and .41629 late: no progressive deep crouch is demonstrated here. Late pose error
is much better than final998's .17992 rad, with height .38160 m. The current pose
is not exact; that alone is not the reason for withholding qualification.

| Case | Camera-body world-vz RMS full / late, m/s | Existing imager world-vz RMS full / late, m/s |
| --- | --- | --- |
| Stand | .005869 / .000238 | .005928 / .000230 |
| Forward | .002697 / .0000063 | .002782 / .0000058 |
| Lateral+ | .09119 / .09095 | .09399 / .09366 |
| Lateral− | .08614 / .08729 | .09555 / .09606 |
| Yaw+ | .05498 / .05759 | .05448 / .05660 |
| Yaw− | .05484 / .05342 | .05874 / .05700 |

`front_realsense_body` and the existing imager use the same authored fixed-frame
transport at 50 Hz. Positive-lateral body-camera RMS is .09384 at 200 Hz versus
.09119 at 50 Hz; the sustained motion is not a one-sample artifact. Original
lateral body-camera RMS was .05649/.05605; original yaw was .07341/.07289.
Thus camera motion improves for rolling/yaw but worsens for lateral motion.
Late lateral signed roll/pitch means are `[.02582,.01271]` and
`[−.02330,.01220]` rad, with standard deviations `[.02587,.03552]` and
`[.02482,.03443]`; body roll/pitch-rate RMS is `[.2794,.2942]` and
`[.2712,.2895]` rad/s. Tilt bias and oscillation are separate observations.
These are rigid-body motion measurements, not jerk estimates or image-quality
measurements; a quiet forward camera does not offset path or stopping errors.

### One next action and validation limits

**Recommendation C:** preserve `model_1999.pt` and, in a future authorized modest
continuation, change only `GO2WCfg.rewards.scales.tracking_yaw: .8 → 1.2`.
Rotation accounts for most lateral drift and also explains the forward path miss.
This proposed 50% weight increase prioritizes the existing command-yaw objective;
it is an engineering hypothesis, not a calibrated optimum or proof of a specific
regularizer conflict. Keep clearance, sensor, posture, action scales, sampling,
PPO, fixed LR and std bounds unchanged. Retain actor, critic, both normalizers,
learned std, Adam moments/steps/LR and native iteration. The earlier comparison
does not justify a lower-LR remedy for a new late regression, or a fresh start.

Ordinary `--resume` restores saved settings, so editing defaults alone would not
apply this proposal. After a future native full-state load and before learning,
the authorized continuation would need to set the resolved
`env.cfg.rewards.scales.tracking_yaw=1.2` and the already prepared runtime
`env.reward_scales['tracking_yaw']=1.2*env.dt` (`.024`), then save that resolved
configuration and provenance in a separate output. Reward preparation must not
run again and multiply dt twice. **No such loading mode, patch or continuation
was implemented or executed in this review.**

[Raw review evidence](../evaluation/go2w_20261007_review/analysis.json) includes
windowed geometry, targets, clipping, costs, censoring and sensor results;
[analysis source](../evaluation/go2w_20261007_review/analyze.py) reuses the existing
sensor/phase/swing helpers. Focused numerical checks pass for full-quaternion
component integration with known rotations, multi-turn heading unwrap, sensor
transport with tilt, and swing boundary censoring. Recorded 50/200 Hz base poses
agree at corresponding endpoints within 1e−6. No optimizer probe, training smoke,
policy bank or full unit suite was run. Training sources, `play.py`, checkpoints,
saved configs and historical evidence remain unchanged.

The [frozen requirements](../evaluation/consolidated_v3_preparation/frozen_targets.json)
remain intact, including holding rotation and complete stopping windows.
Small commands, reverse, arcs/mixed, the wider required/reserve envelope, pushes,
DR and transfer remain **untested by this panel**, with no invented pass criteria.

## Historical yaw-weight preparation (2026-10-07; unexecuted at preparation time)

The production command recorded in this section was subsequently run in
`logs/go2w/go2w_2026-10-07_15-22-01/` and completed all 500 additional updates.
The completed-continuation review below supersedes the preparation-time status;
the preparation evidence itself is retained unchanged.

The exact parent is `logs/go2w/go2w_2026-10-07_09-40-23/model_1999.pt`:
native label 1999, **2000 completed updates**, Adam step 80000 and both normalizer
counts 524288000. Its saved config, checkpoint metadata and retained diagnostics
agree. The current review above and `evaluation/go2w_20261007_review/` remain the
diagnostic basis; this preparation did not repeat policy evaluation.

The only functional change is
`GO2WCfg.rewards.scales.tracking_yaw: .8 -> 1.2` in
`robot_gym/envs/go2w/go2w_config.py`. All other task and learning values match the
parent, including tracking kernels, body angular-z semantics, phase/clearance,
posture/sensor/contact terms, plant, sampling/pushes, 58/16 interface, fixed
LR `3e-4`, gamma `.995`, lambda `.95`, entropy `.003` and std bounds `[.10,.70]`.
The requested additional-update budget and resume bookkeeping are operational
differences, recorded in the [configuration proof](../evaluation/go2w_yaw_continuation_preparation/configuration_proof.json).

`--resume_current_reward_scales` is a Go2-W training-resume opt-in. It snapshots
the central current scales, restores the saved task normally, requires identical
active reward names, and prints/saves the exact old/new scale diff. For this
parent and current source that diff is exactly `tracking_yaw: .8 -> 1.2`.
Scales are applied **before environment reward registration**, so normal dt
weighting occurs once: `1.2 * .02 = .024`. This implements the proposal above
without its suggested post-construction mutation. Ordinary `--resume` and saved
replay still use `.8` (`.016` at runtime). There is no numeric CLI override,
recipe YAML input or second parameter source.

After native loading and before learning, the existing state verifier checks
exact actor/critic, normalizer buffers/counts, learned std, Adam moments/steps/LR
and native iteration equality against the parent. The new hook also checks every
registered reward scale. The separate output's generated `config.yaml` records
the effective values and exact diff under `env_cfg.training_resume`, with source
paths and dynamically computed SHA-256 provenance; no checkpoint hash allowlist
is introduced. `preparation.json` records the before-update verification.
Normalizers remain trainable. The critic initially estimates the previous
objective and must adapt; an initial value-loss change alone is not failure.
This retains learning state, not simulator/RNG state.

Six focused tests pass: ordinary resume/replay, exact one-field diff, active-name
and finite-value guards, opt-in scope, single dt weighting, and verification
after loading. Temporarily changing the central test value to `1.3` produces
runtime `.026`, proving there is no hidden `1.2` override. One native
**4096 x 64, two-update** continuation smoke passed:

| Check | Before updates | After two updates |
| --- | ---: | ---: |
| Native label / completed lineage updates | 1999 / 2000 | 2000 / 2002 |
| Adam step | 80000 | 80080 |
| Actor and critic normalizer counts, each | 524288000 | 524812288 |
| Fixed LR | .0003 | .0003 |
| Runtime yaw scale | .024 | .024 |

Loaded state was exactly equal before learning, both normalizers advanced, and
models/Adam/observations/rewards remained finite. The serialized output config
matched the effective environment. The smoke checkpoint is
`logs/go2w_continuation_smoke/two_updates_2026-10-07_15-10-37/model_2000.pt`;
it is execution evidence only and is **not** the production parent. See the
[before-update proof](../evaluation/go2w_yaw_continuation_preparation/smoke_before.json),
[smoke result](../evaluation/go2w_yaw_continuation_preparation/smoke_after.json)
and [focused test output](../evaluation/go2w_yaw_continuation_preparation/focused_tests_final.txt).
The [final validation summary](../evaluation/go2w_yaw_continuation_preparation/validation_summary.json)
also confirms all 39 protected source-run/review files and `play.py` are unchanged.

The following **500-additional-update command was unexecuted when prepared**;
it records the invocation from the repository root with `genesis-gpu` active:

```powershell
python -m robot_gym.scripts.train --task go2w --resume --resume_current_reward_scales --load_run (Resolve-Path ".\logs\go2w\go2w_2026-10-07_09-40-23").Path --checkpoint 1999 --num_envs 4096 --max_iterations 500 --seed 1 --logger tensorboard --training_diagnostics --headless --rl_device cuda:0
```

The saved rollout remains 64 ticks. Output goes to a separate timestamped
`logs/go2w/go2w_<timestamp>/`. The installed native runner repeats the source
label: 500 updates execute labels 1999 through 2498. With unchanged save interval
250 and native final save, expect `model_2000.pt`, `model_2250.pt`, and final
`model_2498.pt` (**2500 completed lineage updates**, not 2499).

The subsequent decision is bounded to the final checkpoint and the same nominal
six cases: stand (30 s), forward `.5 m/s`, lateral `+/-.3 m/s`, yaw `+/-.8 rad/s`
(moving cases: 3 s zero, 30 s command, 8 s stop). Reuse current1999 raw baselines
under `evaluation/go2w_20261007_review/final1999/`; do not rerun them. Inspect at
most one saved intermediate on a failing case only if final regression warrants
it. Preserve the [frozen requirements](../evaluation/consolidated_v3_preparation/frozen_targets.json).
Prioritize both lateral signs' body-x and body-wz bias/RMSE, pose heading and
path decomposition; forward speed/heading/cross-track; stand displacement/path
and late translation/rotation; and complete stop path/endpoint/reversals/residual
motion. Then check unloading/physical clearance, rear-calf clipping and contacts;
camera-body full/late motion is secondary. Body angular-z is not Euler heading
rate under tilt. Residual direct body-x drift is a separate limitation.

Less heading drift obtained by slowing requested travel is not success. The yaw
reward ceiling rises from `.8` to `1.2`, so compare physical metrics and, if useful,
traces rescored under a common objective rather than total return. Partial
improvement remains an intermediate candidate. Small/reverse/mixed/core-edge
commands, DR and push recovery remain separate untested qualification items.
There is no automatic further training if this bounded continuation is ineffective.

## Completed yaw-weight continuation review (2026-10-07)

**Decision D: retain CK1999 as the development baseline; do not promote final2498
or continue unchanged.** Gains in pure yaw, direct lateral body-x bias and front
clearance do not outweigh forward-path, lateral-stop and holding regressions.
The intermediate is not a better overall candidate. The parent also remains
**unqualified against the frozen targets**. Training defaults and saved replay
semantics are unchanged by this checkpoint selection.

Parent: `logs/go2w/go2w_2026-10-07_09-40-23/model_1999.pt` (2000 updates).
Child: `logs/go2w/go2w_2026-10-07_15-22-01/model_2498.pt` (**500 additional,
2500 lineage updates**). Saved ancestry/hashes, completed-run metadata, final
label 2498, Adam step 100000 and both normalizer counts 655360000 agree. Recorded
before-update state verification was not repeated. Sole functional delta:
`tracking_yaw .8 -> 1.2`, runtime `.024` at dt `.02`; fixed LR `.0003` retained.
Other differences are budget/resume/provenance bookkeeping. Preparation above
is historical; the production run is complete.

[Comparison JSON tables](go2w_yaw_continuation_review_2026-10-07.json) report all
axes and first/full/late/stop windows, per-wheel distributions/censoring/unloading,
clipping versus target error, geometry trends and both sensors. Detailed evidence:
`evaluation/go2w_yaw_continuation_review/`. Parent traces were reused from
`evaluation/go2w_20261007_review/final1999/` without replay.

### Training after episode-accounting startup

| Window | Mean KL | PPO clip fraction | Value loss | Yaw rate / weight | Episode yaw rate / weight |
| --- | ---: | ---: | ---: | ---: | ---: |
| Parent 1800-1999 | .01366 | .2383 | .01216 | .75038 | .74496 |
| Child 2050-2199 | .01349 | .2356 | .01298 | .75900 | .75568 |
| Child 2350-2498 | .01405 | .2396 | .01387 | .76464 | .76176 |

Windows exclude the approximately 47-update startup. Diagnostic 2373 is missing
(148 late records); TensorBoard includes it. Raw yaw `.60031 -> .91757` mostly
reflects rescaling. Normalized scores are not accuracy or invertible RMSE.
Actual LR remains `.0003`; per-joint mean std spans `.1063-.1522 -> .1002-.1441`.
Late floor occupancy: FL/FR calf 8.8/6.1%, RL/RR thigh 27.0/28.4%; other joints zero.
Rear-calf sampled clipping rises about 6.3 -> 6.9%, mean saturation 5.5 -> 6.1%.
Pose cost worsens `.03412 -> .03887`; clearance/support costs modestly improve,
sensor cost stays near `.007`. Contact raw mean rises `4.62e-6 -> 1.69e-5` per tick.
Push summaries remain near 80/15/5, 30 physics ticks and 6.8 N s mean impulse.
Termination counts/causes, command-tier occupancy and event-level associations
are unavailable. Episode means are not failure probabilities; cost magnitudes
are not gradients. Neither total return nor these aggregates select the policy.

### Matched nominal panel

One final-checkpoint invocation used actor mean, seed 1, one environment, nominal
saved dynamics, no noise/DR/pushes or controller/action/phase override. Stand is
30 s; moving cases are 3 s zero, 30 s command, 8 s zero. Each case starts with the
same nominal reset and built-in zero-action warmup, zero added phase offset.
Initial poses, commands, phase and desired targets match the parent exactly.
Analysis is 50 Hz; existing 200 Hz capture covers only forward and lateral+.

| Command | Achieved requested-axis mean, parent -> child | Axis RMSE | Heading change, deg | Max cross-track, m |
| --- | --- | --- | --- | --- |
| Forward .5 m/s | .49579 -> .49970 | .01622 -> .01622 | +4.24 -> -6.83 | .508 -> .972 |
| Lateral +.3 m/s | .30049 -> .30294 | .01531 -> .01758 | -18.81 -> +21.73 | 1.618 -> 1.620 |
| Lateral -.3 m/s | -.30088 -> -.30373 | .01441 -> .01759 | +18.18 -> -21.84 | 1.758 -> 1.826 |
| Yaw +.8 rad/s | .79956 -> .79985 | .04100 -> .03000 | Commanded rotation | See JSON |
| Yaw -.8 rad/s | -.80178 -> -.80221 | .03862 -> .02715 | Commanded rotation | See JSON |

First-second forward speed is `.4667 -> .4656`, lateral `+.2786/-.2823 ->
+.2911/-.2867 m/s`. Travel is not slower. Child final-10-second heading changes
remain forward `-1.87`, lateral `+8.11/-5.56 deg`: curvature persists beyond startup.
Lateral body-x bias improves `+.00900/+.00972 -> -.00064/+.00033 m/s`; body-wz bias
reverses and grows `-.00767/+.00829 -> +.01042/-.01079 rad/s`, despite lower wz RMSE
(`.04571/.04349 -> .03962/.03892`). Pose heading is not integrated body angular-z.

| Lateral initial-forward displacement components, m | Body-x | Rotated body-y | Body-z/tilt |
| --- | ---: | ---: | ---: |
| Parent + | +.2651 | +1.3496 | +.0049 |
| Child + | -.0187 | -1.5942 | -.0066 |
| Parent - | +.2859 | +1.4676 | +.0061 |
| Child - | +.0104 | -1.8295 | -.0056 |

Full-quaternion integration matches lateral pose displacement within 1.5 mm.
Forward cross-track is similarly dominated by rotated forward velocity.
Offline parent-weight rescoring improves lateral yaw scores
`.6948/.7013 -> .7157/.7167` despite worse heading: these are rescored states,
not new rollouts or evidence of better paths.

Stand endpoint/path grows `.0981/.1269 -> .1594/.1686 m`; startup 0-5 s is
`.0057/.0160 -> .0079/.0164 m`. Late XY RMS `.00439 -> .00667 m/s` worsens;
wz RMS `.01117 -> .00457 rad/s` improves, but heading grows `+2.73 -> +7.71 deg`.
Late child wheel targets (FL/FR/RL/RR) are `[-.482,.335,.162,.292] rad/s`, actual
`[.046,.098,.069,.094]`. A deadband could remove the negative FL braking target.
Any future stand mode is a separate integration option requiring braking/correction,
stop/start, small-command and disturbance tests. Zero wheel target is not zero
torque or a rigid lock. The raw policy was evaluated unchanged.

| Full 8 s stop | Endpoint/path m, parent -> child | Final 2 s XY RMS m/s | Final 2 s wz RMS rad/s |
| --- | --- | --- | --- |
| Forward | .0575/.0667 -> .0533/.0597 | .00467 -> .00517 | .01725 -> .00416 |
| Lateral+ | .0171/.0371 -> .0251/.0545 | .00188 -> .00609 | .00164 -> .00780 |
| Lateral- | .0321/.0465 -> .0459/.0590 | .00401 -> .00377 | .00370 -> .00438 |
| Yaw+ | .0112/.0287 -> .0103/.0248 | .00393 -> .00356 | .00797 -> .00550 |
| Yaw- | .0442/.0649 -> .0534/.0597 | .00582 -> .00679 | .03192 -> .01073 |

Opposing translation during forward/lateral+ grows `.75/3.41 -> .94/4.75 mm`;
lateral- decreases, but stop rotation worsens `-1.55 -> -4.61 deg`. Negative-yaw
opposing rotation falls `9.27 -> 3.98 deg`, with `+1.06 deg` still accumulating
in the final 2 s. Endpoints alone hide movement.

### Posture, steps and sensors

Late stand/forward/lateral+/lateral- pose RMS is
`.0542/.0884/.1473/.1474 -> .0535/.0927/.1529/.1536 rad`; child heights are
`.4150/.4152/.4139/.4135 m`, with no demonstrated progressive deep crouch.
The `.427742 m` reference target is unchanged. Wheelbase changes
`.4145/.4790/.3432/.3424 -> .4184/.4884/.3156/.3165 m`, separately measured from
front/rear widths and pair midpoints. JSON tables retain their first/full/late
trends and signed roll/pitch bias/variation. Lateral roll variation decreases;
child late roll/pitch-rate RMS remains `[.239,.299]` / `[.235,.304] rad/s`.

| Case | Median physical peaks mm, FL/FR/RL/RR, parent -> child | Camera-body vz RMS full/late m/s, parent -> child |
| --- | --- | --- |
| Lateral+ | 10.9/10.8/8.2/15.9 -> 18.4/22.5/8.6/15.3 | .0912/.0910 -> .0906/.0898 |
| Lateral- | 11.8/9.8/16.7/6.5 -> 22.5/19.0/14.8/7.9 | .0861/.0873 -> .0899/.0899 |
| Yaw+ | 4.2/17.4/9.5/13.2 -> 14.7/11.6/15.0/4.8 | .0550/.0576 -> .0521/.0518 |
| Yaw- | 18.1/4.6/11.5/7.6 -> 11.6/14.7/6.8/16.4 | .0548/.0534 -> .0548/.0532 |

Apex remains 40 mm. Whole-swing/apex unloading, completed physical events and
0-2 censored intervals are retained. Child lateral apex unloading is 97.4-99.5%,
but yaw+ FR falls 96.2 -> 69.7%, yaw- FL 96.8 -> 66.3% (target >=32 mm).
Low lift-off/touchdown clearance alone is not dragging. The 200 Hz trace confirms
the front-clearance gain but splits short recontacts; event counts depend on rate.
Loaded lateral-center RMS remains `.064-.095 m/s`, a sliding surrogate.
Lateral RR/RL calf clipping falls `39.4/37.1 -> 33.9/34.6%`; actual-minus-applied
rear-calf error remains `.29-.30 rad`. Issued/applied targets agree. No sampled
self/non-wheel contacts, falls, resets or force-limit saturation occur; 200 Hz
positive-lateral peak force is 67.3% of limit. Self-contact sampling is 50 Hz,
not a continuous absence guarantee. Imager full/late data are retained separately;
its lateral+ motion worsens while lateral- improves. Sensor motion is not image
quality/jerk and cannot compensate for tracking/holding regressions.

### Bounded decision

The only intermediate, `logs/go2w/go2w_2026-10-07_15-22-01/model_2250.pt`
(2252 lineage updates), was checked at 50 Hz on stand/forward/lateral+/lateral-.
Forward cross-track/heading is `.585 m/-4.56 deg`; lateral deviation
`2.007/1.675 m`, heading `-20.92/+19.77 deg`; late holding XY/wz RMS
`.00836 m/s/.03464 rad/s`. Its pure-yaw cases were not tested. The checkpoint
comparison does not support another unchanged continuation.

**Keep CK1999 and close this experiment.** No refinement is proposed or executed.
Frozen forward limits remain `.05 m` cross-track, `2 deg` heading, `.45-.55 m/s`
mean and `.04 m/s` RMSE; late holding XY limit `.005 m/s` plus existing rotation
comparisons (CK499 `.00670 rad/s`, full `-11.1 deg`; CK250 `.00030`). Preserve
complete-stop requirements, including CK499 endpoint/path `.0311/.1051 m`.
The [frozen record](../evaluation/consolidated_v3_preparation/frozen_targets.json)
is unchanged. Reverse, small/mixed/core-edge commands, reserve extremes, pushes,
DR, Sim2Sim and hardware remain unqualified. Only new path-analysis checks,
matched protocols and 50/200 Hz pose consistency were validated; no training,
optimizer smoke, full test suite or training-source changes were performed.

## Geometric straight-motion preparation (2026-10-08)

This is **training a velocity-conditioned policy with privileged geometric
supervision**, not deploying a position controller or heading-hold command
adapter. Production was unstarted at preparation time; the interrupted-run review
below supersedes that status. The selected development baseline is
`logs/go2w/go2w_2026-10-07_09-40-23/model_1999.pt` (2000 updates); neither it nor
historical CK2498 initializes this run. Their checkpoint/config/raw comparisons
are preserved. Current yaw tracking returns from 1.2 to the baseline's .8.

### Task and timing

For an issued command `u=(vx,vy,wz)` with `norm(u_xy) > .01 m/s` and
`abs(wz) <= 1e-6 rad/s`, latch horizontal authored-base position `p0` and heading
`psi0` from the quaternion's projected body-forward axis. A projection norm
above `1e-6` is required at latching. Set

```text
d = Rz(psi0) u_xy / norm(u_xy),    n = [-d_y, d_x]
e_cross = n dot (p_xy - p0)
e_heading = atan2(sin(psi-psi0), cos(psi-psi0))
H(z) = .5*z^2 for |z|<=1, otherwise |z|-.5
s = clip(reference_age / .5 s, 0, 1),   a = valid * s^2*(3-2*s)
r_line = -.1*a*H(max(|e_cross|-.005 m,0)/.10 m)
r_heading = -.1*a*H(e_heading/.10 rad)
```

The heading reference is the initial **body** heading, including reverse and
lateral motion; it is not the direction of translation. Full quaternion pose
projection supplies heading, not body angular-z integration. No along-line lag,
`p0+v*t` target, error-based termination, contact gate or action correction is added.
Existing x/y/yaw velocity objectives remain essential: the line cost alone is
zero for ideal, stopped, delayed or backward motion along the same line.

Forward, reverse, lateral and zero-yaw diagonal translation are eligible. Stand,
pure yaw and intentional nonzero-yaw arcs retain their existing objectives. A
fixed `.00001 rad/s` yaw request is already ineligible; the threshold is numerical
zero, not an enlarged yaw deadzone. The sampler's existing deadzones are unchanged;
this task sees the final issued command. It is not a general path-tracking task.
A future nonzero-yaw extension would need distance to a geometric arc/path, not
to a time-synchronized reference point.

Every actual numerical command change, **including speed-only changes**, starts
a new segment at the current boundary. Repeated identical values keep the anchor
and age, even when their sampled hold timer renews. Sampling, fixed-command replay
and evaluation command batches follow this same rule. Pushes, observation getters,
rollout boundaries and PPO updates never relatch or move the anchor.

Physics executes the old command/phase; the task then advances reference age,
scores its ending pose and captures diagnostics. Only afterward does command
resampling install a new anchor, followed by independent episode resets and next
observations. A reset latches the simulator's new pose, not stale terminal base
buffers. Other environments retain their references. The phase/action timing is
unchanged. Native RSL-RL 5.5.1 timeout correction uses the pre-step transition value
with its old reference; reset observations contain the new reference. Nonterminal
rollout-end bootstrapping uses the current reference. No PPO fork or bootstrap
algorithm change was introduced.

### Actor and critic

```text
58 existing noisy/scaled policy inputs -> actor normalizer -> actor -> 16 actions
                       same 58 inputs + 7 relative features
                                      -> critic normalizer -> critic -> value
simulation pose + fixed segment anchor -> geometric rewards and diagnostics
```

The actor's input order, scales, noise, normalizer dimension and P-leg/V-wheel
offset contract are unchanged. The native TensorDict contains `policy: [N,58]`
and `critic: [N,65]`, with `obs_groups={'actor':['policy'],'critic':['critic']}`.
The critic prefix is the same policy observation, including the same noise draw.
Its seven appended features, in order, are:

| Index | Feature before empirical normalization | Units / convention |
| --- | --- | --- |
| 58 | `e_cross / .10 m` | Signed, dimensionless; zero when invalid |
| 59 | `sin(e_heading)` | Signed, dimensionless |
| 60 | `cos(e_heading)` | Dimensionless; one when invalid |
| 61 | Reference valid | 0 or 1 |
| 62 | Smooth activation `a` | 0 to 1 |
| 63 | Reference age / 30 s | Zero when invalid; no early saturation |
| 64 | Remaining command time / 30 s | Policy ticks times dt; -1 for pinned/non-resampled commands |

Both groups retain the existing final observation clip of +/-100. No raw world
coordinates, absolute heading or environment origin enters either network.
Sagittal reflection negates cross-track and heading sine; cosine, validity,
activation and times are unchanged. Existing joint/action reflection and phase
half-cycle signs are retained. Geometry-based tests cover forward, reverse,
both lateral signs and diagonal commands. Only the 58-input actor and its own
normalizer are exported, using the native export wrappers.

The actor remains partially observable: identical policy inputs at opposite
unobserved path offsets produce identical actions. Geometric supervision and a
better-informed value estimate may reduce systematic drift generation; they do
not give the actor arbitrary offset-recovery feedback or guarantee convergence.

### One calibrated parameter set

| Current field | Value | Purpose |
| --- | --- | --- |
| `straight_motion.min_speed_m_s` | .01 m/s | Exclude stand/numerically tiny translation |
| `straight_motion.yaw_zero_rad_s` | 1e-6 rad/s | Exclude intentional yaw |
| `straight_motion.heading_projection_min` | 1e-6 | Finite horizontal heading definition |
| `straight_motion.cross_tolerance_m` | .005 m | Small sway allowance, one tenth of the frozen .05 m path limit |
| `straight_motion.cross_scale_m` | .10 m | Huber quadratic-to-linear transition |
| `straight_motion.heading_scale_rad` | .10 rad | Huber heading transition |
| `straight_motion.ramp_s` | .5 s | One smooth command-change activation ramp |
| `straight_motion.critic_time_scale_s` | 30 s | Relative time normalization only |
| `rewards.scales.straight_cross_track` | -.1 | Cross-track reward-rate coefficient |
| `rewards.scales.straight_heading` | -.1 | Heading reward-rate coefficient |
| `rewards.scales.tracking_yaw` | .8 | Restore selected baseline coefficient |

All runtime numeric choices live in `go2w_config.py`. The helper has formulas and
state transitions, not another parameter preset. Shared reward registration
applies dt exactly once: at .02 s, both new runtime scales are `-.002` and yaw is
`.016`. The termination event remains `-5`, without dt scaling.

[Calibration evidence](../evaluation/go2w_straight_motion_preparation/calibration.json)
uses the existing 50 Hz CK1999/CK2498 raw states and synthetic geometry, with no
new policy rollout. The table shows **positive added cost per second**, summed
across the two new terms; existing states are not claimed to improve.

| Policy / case | First second | Seconds 2-4 | Full 30 s | Late 20-30 s |
| --- | ---: | ---: | ---: | ---: |
| CK1999 forward | .000110 | .00124 | .13261 | .31295 |
| CK2498 forward | .000041 | .00307 | .32218 | .69894 |
| CK1999 lateral+ | .000803 | .00998 | .62359 | 1.30340 |
| CK2498 lateral+ | .000370 | .00517 | .59940 | 1.32286 |
| CK1999 lateral- | .000763 | .01425 | .70773 | 1.44989 |
| CK2498 lateral- | .000258 | .00950 | .72177 | 1.52879 |

For context, seconds 2-4 tracking reward rates sum to 2.65-2.80; pose costs are
about .019 lateral and .124-.137 forward; lateral clearance/support costs are
.033-.045/.016-.028. Thus the new objective starts small but accumulates a
substantial cost on persistent geometric error. Late lateral added cost is
1.30-1.53 against tracking rates near 2.65-2.66. Recorded/reconstructed term
subtotals remain positive on these traces, but are not a complete return.
Cost magnitudes are not gradients, causal conflict evidence or predicted learning.

Synthetic stopped/lagged/ideal/backward on-line paths all have zero geometric
cost. At a .5 m/s command their steady three-axis tracking reward sums are
1.05/2.8/2.8/-.95 respectively (lag has an acquisition transient). A constant
`.01 rad/s` yaw bias at .5 m/s costs `.000168/.00679/.80435` over first/2-4/full
windows; a `.01 m/s` cross-axis bias costs `.000022/.00332/.10151`. A forward-to-
lateral switch with .25 s exponential decay of residual .5 m/s forward velocity
costs .0416 in its first second, then .070: its accumulated .125 m offset is
retained, not erased. This is a synthetic transient, not a measured recovery.
Stand and pure-yaw traces receive exactly zero new cost.

At .05 m cross-track plus 2 degrees heading error, fully active auxiliary cost
is .0162/s. At .5 m/.1 rad it is .495/s; at 1.8 m/.38 rad it is 2.075/s. Huber's
linear tail still distinguishes larger errors: 3 m/1 rad costs 3.895/s, and
5 m/pi costs 8.037/s. For fixed errors, dt/(1-gamma)=4 s gives discounted costs
8.30, 15.58 and 32.15 in the last three examples, versus the unchanged -5 terminal
event. Very large error could therefore create reset-seeking pressure, although
termination also loses tracking reward and incurs other costs. No cap, path
termination or relatch masks this limitation; retained-state arithmetic cannot
predict that behavior. This preparation uses one parameter set, not a sweep.

### Long lateral exposure and unchanged decisions

The existing rare holds now admit `['stand','straight','arc','lateral']` through
`commands.holds.eligible_families`. Probabilities remain .04 for 8-15 s and .015
for 20-30 s; other durations, short exposure, family probabilities, command
ranges/tiers and pushes are unchanged. A deterministic **prospective** 50,000-
segment check (seed 20261008, no simulator or episode truncation) gives:

| Family | Segment share | Duration-weighted share |
| --- | ---: | ---: |
| Stand | 15.20% | 17.10% |
| Straight | 19.74% | 15.96% |
| Arc | 7.19% | 5.57% |
| Pure yaw | 19.59% | 20.59% |
| Precision | 10.25% | 4.99% |
| Pure lateral | 19.85% | 27.01% |
| Mixed | 8.18% | 8.77% |

Lateral positive/negative counts are 4972/4952. They include 662/664 holds <=1 s,
193/186 holds at 8-15 s, 87/63 holds at 20-30 s, and 445/422 commands at
.02-.05 m/s. All commands remain inside reserve bounds, with at most one axis
outside core. Tier segment shares are 85.09/7.51/7.40% overall because stand and
precision remain core-only; ordinary moving draws still use 80/10/10%. These are
prospective complete-hold statistics, not historical occupancy or actual future
episode exposure. Longer lateral holds intentionally increase lateral time share.

The generated [fully resolved production configuration](../evaluation/go2w_straight_motion_preparation/resolved_production.generated.yaml)
is evidence/output, **never an input**. Its complete diff against CK1999 contains
only the new task parameters/two reward scales, critic dimension/group, hold
eligibility and schema version. Against the previous checked-in source it also
contains yaw 1.2 -> .8. All other task/PPO values match: canonical reference and
measured geometry, gains/limits/delay, 4 cm apex/.8 s phase/.65 stance, sensor/
pose/support/safety costs, command envelopes, push mixture, noise/DR, hidden
layers, LR3e-4, gamma.995, lambda.95, entropy.003, std.40 with [.10,.70] bounds,
4096x64, 2000 updates and save interval250.

### Validation and the next comparison

The focused CPU checks cover ideal/lagged lines, error response, translation and
rotation invariance, all-direction reflection, tiny commands/angle wrapping,
reference persistence, speed changes, fixed/evaluation command consistency,
partial resets and native timeout-value semantics, dt weighting, actor/normalizer/
JIT/ONNX isolation, sampler bounds/small commands, the strong push branch,
canonical actions, fresh initialization and saved-schema guards. No full suite
or baseline evaluation campaign was run. All **28 focused tests passed**.

The single native fresh **4096 x 64, two-update smoke passed**, saved separately at
`logs/go2w/straight_motion_smoke_2026-10-08_00-45-43/model_1.pt`. Its
[effective saved config](../logs/go2w/straight_motion_smoke_2026-10-08_00-45-43/config.yaml)
matches the generated production config except for the two-update budget and
smoke output name. [Smoke assertions](../evaluation/go2w_straight_motion_preparation/smoke_validation.json)
verify no checkpoint load, empty initial Adam, initial normalizer counts zero,
std .40 and iteration zero. Both counts advance to 524288, Adam to step80,
and LR remains .0003. The critic consumes 65 inputs and its seven new input
columns change by up to .01943; actor/critic normalizers remain 58/65 dimensions.
All 128 policy boundaries have finite observations/rewards, and final model
tensors/losses are finite. The 121 partial-reset calls (2272 reset events,
including 147 timeouts) keep unreset environments' anchors/ages unchanged and
set reset anchors to the new simulator pose. These are execution observations
from a fresh random policy, not locomotion success or estimated failure rates.
No second smoke or production run was launched.

The [compact preparation record](go2w_straight_motion_preparation_2026-10-08.json)
contains the exact configuration diff, calibration/sampling summaries and smoke
proof. Twenty-five protected checkpoint/config/trace files, including `play.py`,
are hash-unchanged. `git diff --check` passes. Changed source files are
`go2w_config.py`, `go2w_env.py`, new `straight_motion.py`, `go2w_symmetry.py`,
`evaluate.py`, and `diagnostic_bank.py`, all under `robot_gym/envs/go2w/`.
Changed tests are `test_go2w_current_config.py`, `test_go2w_continuation.py` and
new `test_go2w_straight_motion.py`; documentation changes are this guide and the
preparation JSON. No Dodo/Go2/shared runtime, control/physics or `play.py` edits.

Schema 2 is a deliberate training change. Old schema-1 CK1999/CK2498 checkpoints
are rejected explicitly before model loading; their historical checkout and raw
baselines remain authoritative. The pre-change source is commit
`b47458a266a3d4b616ac780bdb58961886dbf2f6` (provenance, not a runtime allowlist).
No legacy parameter reconstruction is added.
Ordinary resume of a new schema-2 run still restores its complete saved task and
learning state. The new production run loads **no** checkpoint, normalizer, std
or Adam state and starts at iteration zero. Saves retain native labels: a fresh
2000-update run ends at `model_1999.pt` (2000 completed updates), in its own
`logs/go2w/go2w_<timestamp>/` directory.

The next physical comparison uses the final new policy with unchanged actor-only
observations. Reuse CK1999/CK2498 raw nominal baselines: stand30s; forward .5,
lateral +/-.3 and yaw +/-.8 with 3s zero,30s command,8s zero. Separate first-second,
full/late movement, holding and complete stops. Retain velocity bias/RMSE on all
axes, achieved travel speed, quaternion heading and anchored path/decomposition,
endpoint/path/reversals/residual motion, contacts and saturation, actual physical
clearance/unloading, reference pose/height/geometry and camera-body/imager motion.
Report both reduced nominal drift generation and the recovery limitation caused
by hidden offsets; no arbitrary path-recovery claim follows from a lower reward.
Reverse and diagonal geometry tests here are not physical qualification; small,
mixed/nonzero-yaw, boundary/reserve, pushes/DR and transfer remain separate.

Frozen targets remain unchanged: forward cross-track <=.05 m and heading <=2 deg
at .5 m/s for30s, with mean .45-.55 m/s and RMSE <=.04 m/s; late stand XY RMS
<=.005 m/s plus the recorded rotation comparisons; complete-stop endpoint/path,
reversal and residual requirements stay in the [frozen record](../evaluation/consolidated_v3_preparation/frozen_targets.json).
Do not accept straighter travel by slowing down or quieter sensors at the expense
of tracking/support/stopping. No automatic tuning or further training is authorized
by this preparation, and the geometric reward does not guarantee convergence.

## Interrupted geometric-run review (2026-10-08)

Historical preparation: the 999-update resume recommended here was subsequently
executed. The completed-budget review below supersedes this next-action decision.

**Decision A: retain CK1000 and finish its original 2000-update lineage with 999
additional native updates, using ordinary full-state resume. No reward, PPO,
sampling, action, physics or exploration change.** This is an intermediate
development candidate, with substantial forward/path and stepping limitations.
Only optional diagnostic aggregation was optimized in this review.

The verified run is `logs/go2w/go2w_2026-10-08_09-57-35/`. Saved labels are
0, 250, 500, 750 and **1000**. `model_1000.pt` contains **1001 completed updates**,
Adam step40040 and both normalizer counts262406144. It is the checkpoint matching
the reported approximately-1000 GUI replay; no GUI session log identifies a more
precise selection. TensorBoard and diagnostics actually reach label1115, so
**1116 updates completed**, beyond the reported console1114. JSONL row254 is
missing; it is unavailable, not a zero measurement. `preparation.json` records
`interrupted_or_failed`; there is no retained final save or exception traceback
that establishes the interruption cause. The later115 updates are not resumable.
This was a planned2000 run, not a completed2000 run.

Saved schema2, actor58/critic65, fixed LR3e-4, yaw weight.8/runtime.016, geometric
runtime scales-.002 each, 4096x64, and the intended reference/action/phase task
are present. Model first layers and both normalizers have the recorded58/65
dimensions. The existing preparation proved critic consumption and actor/export
isolation; no new export or complete state-equivalence campaign was needed.

The [compact comparison/profiling JSON](go2w_geometric_review_2026-10-08.json)
contains all-axis first/full/late errors, stops, joint/geometry measurements,
conditioned costs and profiling data. Large traces and detailed arithmetic remain
under `evaluation/go2w_geometric_review/`. CK1999 raw traces from
`evaluation/go2w_20261007_review/final1999/` were reused, without historical replay.

### Physical comparison

One evaluator invocation used stand30s and moving cases3s zero,30s command,8s
zero: forward(.5,0,0), lateral(0,+/-.3,0), yaw(0,0,+/-.8). Actor mean, nominal
saved dynamics, no noise/DR/pushes, no action/command override. Each case starts
from the existing nominal reset, including its one zero-action settling tick.
Initial pose, command/phase schedules and desired clearances match CK1999 exactly.
All comparisons use50Hz; existing200Hz capture was used only for forward and
positive lateral. There were **no falls, resets or sampled self/non-wheel contacts**
in the six cases. This does not exclude contacts between samples.

| Full30s metric | CK1999 | Geometric CK1000 |
| --- | ---: | ---: |
| Forward mean vx / vx RMSE, m/s | .49579 / .01622 | .49945 / .01780 |
| Forward heading / maximum cross-track | +4.24° / .508m | **+8.55° / 1.024m** |
| Lateral+ mean vy / vy RMSE, m/s | .30049 / .01531 | .30363 / .02036 |
| Lateral+ heading / maximum cross-track | -18.81° / 1.618m | **-1.75° / .217m** |
| Lateral- mean vy / vy RMSE, m/s | -.30088 / .01441 | -.30421 / .02051 |
| Lateral- heading / maximum cross-track | +18.18° / 1.758m | **+8.15° / .677m** |
| Yaw+ achieved wz / RMSE, rad/s | .79956 / .04100 | .80928 / .05381 |
| Yaw- achieved wz / RMSE, rad/s | -.80178 / .03862 | -.80600 / .04934 |
| Stand endpoint / path, m | .09807 / .12685 | .05346 / .05387 |
| Stand late XY / wz RMS | .004385m/s / .011165rad/s | **.001412 / .000374** |

The single permitted additional check was CK750 **forward only**: mean.50118,
RMSE.01783m/s, heading-7.12°, cross-track1.093m. Forward curvature already existed;
its sign changes by CK1000, with little improvement in magnitude. This is not
evidence of a newly collapsing posture, nor evidence that lower LR is required.

Current lateral body-x biases are +.00262/+.00446m/s; body-wz biases are
+.00040/+.00242rad/s, with wz RMSE.05391/.05123. Full-quaternion decomposition
of displacement along the **initial forward axis**, in body-x / rotated body-y /
body-z contributions, gives lateral+ `.0786 + .1453 - .00945m` and lateral-
`.1333 + .5521 - .00894m`. Integration agrees with pose to within3mm at50Hz.
Thus direct x bias remains, while rotation of lateral travel still supplies most
cross-axis displacement. Positive-lateral body-wz bias even has the opposite
sign to its pose heading change: tilted body angular-z is not Euler yaw rate.
Late heading still changes -1.13°/+4.11° over the last10s; forward adds+3.23°.
The errors are not explained solely by acquisition. Forward cross-track is
principally rotation of correctly achieved body-x travel, not lateral velocity.

Stand startup0-5s endpoint/path is .01655/.01660m; the full drift is about5.3cm,
not metres. Late wheel targets FL/FR/RL/RR average
`[-.0950,.0210,.00635,.00351]rad/s`, while actual rates average
`[-.0210,-.0131,-.0153,-.0146]rad/s`. Neither action magnitude nor zero target
alone establishes holding or zero torque. No wheel deadband/stand controller was
added. Full stop endpoint/path for forward, lateral+, lateral-, yaw+, yaw- is
`.0443/.0455`, `.0201/.0396`, `.00473/.0425`, `.0257/.0273`, `.0197/.0215m`.
Their final2s XY RMS is `.00095,.00145,.00172,.00235,.00176m/s`; wz RMS is
`.00609,.00071,.00125,.00110,.00117rad/s`. Endpoint, path, reversals and residual
rotation are separate records. Forward stop endpoint still exceeds frozen
CK499's .0311m, although its path is below .1051m.

Forward actual leg-reference RMS improves .08492→.05879rad. Height averages
.41558m (late.41451); early0-10s to late20-30s changes by2.57mm, then only.24mm
between the first/last2s of the late window. Late roll/pitch is -.88°/-.64°;
late variations are .009°/.008°. Leg RMS rises .00144rad across that late window:
small bounded migration, not progressive collapse. Full/late wheelbase is
.37369/.37441m and front/rear widths .41202/.42333 → .42003/.43126m. Pair
midpoints and each actual joint/target error are in the JSON. The canonical
.427741656m reference height remains unchanged; forward is inside the existing
15mm height tolerance, hence its height cost is zero.

### What the objectives actually measure

`Episode/rew_*` is each reset episode's accumulated contribution divided by the
configured60s, then averaged by the logger. `Train/mean_reward` is accumulated
episode return, not that rate. Diagnostics below instead average the actual
rollout ticks. Do not compare those plotted quantities directly or infer failure
probability from mean episode length/termination reward.

| Recorded state/quantity, full movement | Unscaled cost | Weight | Weighted rate | Per .02s tick |
| --- | ---: | ---: | ---: | ---: |
| Forward actual reference error RMS .05879rad, normalized Huber mean | .17814 | -.5 | -.08907 | -.001781 |
| Forward projected-gravity XY squared sum | .00034054 | -4 | -.001362 | -.00002724 |
| Lateral+ actual pose RMS .15230rad: raw Huber .79214 × relaxation.05 | .039607 | -.5 | -.019803 | -.0003961 |
| Lateral+ cylinder gap minus phase target, Huber(error/.04m), four-wheel mean | .052180 | -1 | -.052180 | -.0010436 |
| Same desired targets, counterfactual actual clearance=0 | .065625 | -1 | -.065625 | -.0013125 |
| Lateral+ loaded axle-lateral center speed squared, four-wheel mean | .0050383m²/s² | -.2 | -.0010077 | -.00002015 |
| Lateral+ phase-weighted normalized support cost | .065241 | -.5 | -.032621 | -.0006524 |
| Lateral+ mirrored-imager world-z velocity squared mean | .013521m²/s² | -1 | -.013521 | -.0002704 |

Pose activation is1 for forward/stand and.05 for these lateral commands; it uses
actual leg positions, not support target offsets or continuous wheel angles.
Step demand is1 throughout commanded lateral/yaw and0 for pure rolling. Clearance
cost has **no load escape gate** and includes stance. At both lateral signs its
cost is about80% of the zero-clearance counterfactual, explaining why a small
global mean does not establish good clearance. The matched CK1999 clearance
costs were .04371/.04337, versus current.05218/.05260.

Completed physical swing median peaks FL/FR/RL/RR are lateral+
`11.45/11.86/10.05/11.63mm`, lateral- `13.61/12.33/11.35/11.79mm` against40mm.
There is no general measured increase in step height. At desired apex, unloaded
fraction(<=6N) is + `[.905,.935,.486,.932]`, - `[.911,.919,.930,.553]`.
One rear wheel often remains loaded; the others can be unloaded yet shallow.
Positive-lateral200Hz desired-window peak leads are typically95/95/105/30ms.
There are21/43/12/35 brief within-swing recontacts, predominantly5ms (some10ms),
which explains excess load-event counts. Completed, load-only and censored events
are retained separately. Low liftoff/touchdown gaps alone were not called dragging.

Loaded lateral center RMS is .084-.099m/s and .084-.094m/s for the two signs:
the current scrub surrogate does capture substantial loaded lateral shuffling.
It averages gated squared speed over **all four wheels**, so it is not tire-patch
slip and does not penalize legitimate longitudinal rolling. Its small weighted
mean is not evidence that PPO ignores it. Clearance, support, pose, sensor and
action-rate costs are conditioned on loaded/unloaded, swing/apex/stance and
early/late windows in the retained analysis. Cost size is not a gradient or proof
of a causal conflict; no coefficient was changed.

Rear-calf positive clipping occurs19.3% on RR for lateral+ and18.5% on RL for
lateral-, but **zero at those calves' desired apex**; it is predominantly support
extension. Actual joints lag the applied -1.0rad targets, distinct from clipping.
All16 joints have zero policy-rate occupancy at99% of force limits; positive
lateral200Hz maximum is68.2% of limit. Offline FK at five recorded clipped states
per affected calf agrees with measured gaps within0.11micrometre; .05rad more
flexion raises that cylinder about7.1-8.3mm with base/other joints fixed. This
does not establish loaded feasibility, but does not support widening the positive
calf limit to fix shallow swings. No actions/gains/limits/apex were changed.

Camera-body vertical RMS improves forward .00270→.00191m/s, but worsens
lateral+ .09119→.10973, lateral- .08614→.10719 and yaw about.055→.074-.075.
Late lateral remains .10909/.10840m/s. Current lateral roll/pitch-rate RMS is
`.247/.289` and `.252/.286rad/s`, so camera motion cannot be attributed simply
to larger angular RMS. Both fixed-frame body and retained imager metrics are
recorded at matching50Hz; positive lateral200Hz is .11190/.11652m/s. These are
rigid-body motion measurements, not measured image quality or jerk.

### Reference, training progress and parameter hypotheses

Reconstructing the single command-onset anchor from recorded poses reproduces
all saved reference errors/age/validity/activation (maximum discrepancy below
3e-8). Translation is eligible100% of its30s command, mean activation.992
(.76 in the first second); stand, yaw and stops are ineligible. No identical
message relatch occurs across the long holds. Geometric cross+heading cost
rates first-second / seconds2-4 / late are forward `.000007/.001343/.71712`,
lateral+ `.000755/.001392/.11523`, lateral- `.000357/.004403/.47469`.
For these eligible windows active-time and all-window means coincide; mixing
in ineligible stand/stops would dilute them. Tracking remains about2.62-2.80/s.
Nominal errors stay below the earlier extreme-cost examples and no resets occur;
training logs do not associate individual large errors, pushes and resets, so
reset-seeking cannot be inferred or ruled out from their averages.

The native profiling warmup checks128 ticks across a rollout boundary and184
partial reset events: unchanged commands preserve anchors/age; reset/changed
commands latch the correct pose independently. No active push fell in those
checked ticks; the unchanged push path and prior focused checks supply the
no-push-relatch mechanism evidence, not an invented event association. The
critic receives seven relative inputs; the actor still cannot distinguish
identical58-input observations at opposite hidden accumulated offsets.

Between labels600-749 and1000-1115, measured KL changes .01503→.01435, PPO
ratio clipping26.81→25.82%, value loss .01615→.01342, geometric heading cost
.00645→.00521/s and cross-track cost .01508→.01453/s. Clearance changes only
.03136→.03102/s. Std remains above its.10 floor for every joint; late means
are approximately.111-.167. There is no evidence here to reset exploration or
raise LR. Family/hold time and push-event summaries are retained; latest family
time is about16% stand,16% straight,6% arc,21% yaw,5% precision,27% lateral,
9% mixed, with approximately11%/8% in selected long/extended holds. Realized
tier occupancy and family-specific long holds are unavailable. Overall training
reference validity/activation averages are43.1%/39.0%; dividing its geometric
rates by the valid fraction gives active-time cost means, without recovering
missing per-event associations.

Sampled `[.001,.004,.005]` becomes `[0,0,0]`: XY vector norm.004123<.01,
and absolute yaw.005<.01. XY uses a **vector-norm** deadzone, not per-axis
zeroing; `[.009,.009,0]` survives. Fixed explicit evaluator commands use their
own unchanged assignment path. Raw network means/samples, normalized clipping,
delayed/scaled wheel targets and PPO likelihood-ratio clipping are different
quantities. Removing small wheel targets can remove braking/correction too.
Gamma.995 has an approximately4s discount time; gamma×GAE-lambda.95 has an
approximately.355s trace time, with value bootstrapping. Dense geometric costs
arrive every eligible tick; neither number means learning sees only one hard
horizon. No gamma/LR/entropy/std/controller changes were made.

### Bounded performance work and next command

RTX4070Ti12GB, driver617.42, Genesis1.4.1, Torch2.9.0+cu130 and RSL-RL5.5.1,
headless4096x64. No other simulation/GUI Python process ran; TensorBoard and
the normal Windows desktop remained. Compilation/startup was excluded. GPU
snapshots show50-67°C, approximately2.8GHz and90-171W against285W; no obvious
power/thermal ceiling appears in these snapshots. Historical competing processes
and continuous throttle flags were not recorded. Checkpoint/TensorBoard writer
I/O is unmeasured; no production checkpoint was written.

| Synchronized wall time per rollout/update | Before | After |
| --- | ---: | ---: |
| Collection, full diagnostics, mean of3×64 ticks | 4.411s | 4.281s |
| Collection, diagnostics disabled, mean of3×64 ticks | 3.904s | 4.021s |
| Increment for full diagnostics | .506s | .260s |
| Native returns + PPO update, one disposable sample | .954s | .943s |

The observed full-diagnostics collection reduction is **2.95%**; the control
condition's variation limits precision. No learning-time speedup is established
by one sample. Starting scene pose, environment/reference buffers and RNG are
restored for on/off comparison. Saved physics is nondeterministic: subsequent
trajectories are not bit-identical, although commands/phase/reset masks match.
This timing comparison is not a bitwise physics-equivalence proof.

Eight instrumented ticks show state/contact refresh and host dispatch dominate
the Python path:13 state updates (including post-reset refresh),21 contact
queries,26 wheel-quaternion getters, and1069→988 CUDA stream synchronizations.
Diagnostic reward aggregation drops65.8→19.2ms host time and per-term recording
62.5→30.3ms. State refresh remains about198ms, physics/control intervals about
65/38ms, versus about5ms for critic-reference features and13ms observation
construction. These nested CPU/CUDA intervals include profiling/host-gap overhead
and cannot be added as exclusive kernel times. Seven critic features are not
demonstrated to be the main slowdown. Both observation groups occupy123MiB of
rollout storage:58MiB duplicates the policy prefix and7MiB is privileged data;
measured add-transition time is about3.2ms over eight instrumented ticks.

Only `go2w/training_diagnostics.py` changed at runtime: batch family reductions,
fixed-size masked slew sums with exact sample counts, and reused reward-term
accumulators without per-tick scalar copies. All64×4096 samples/channels remain;
there is no diagnostic subsampling or phase bias. Reference caching, sampler
constants, diagnostic-only geometry and duplicate state/contact getters were
inspected but left unchanged. Four focused numerical tests pass, including
sample counts, reset exclusion, dt and input/RNG preservation. The two disposable
native updates each load CK1000 and advance Adam40040→40080 at LR3e-4, with finite
losses. No policy was saved. An initial profiling-helper buffer-reset failure
occurred before any optimizer update; its log is retained. `play.py`, CK1000 and
its config pass before/after hash checks; historical artifacts were only read.

The source changes are the diagnostic module, its focused test
`tests/test_go2w_diagnostics_performance.py`, this guide and the comparison JSON.
No current task/learning parameter changed. The frozen .05m/2° forward limits
are still missed substantially; .45-.55m/s and <=.04m/s speed bands are met in
this one forward case. Holding and stopping evidence does not qualify untested
small/reverse/diagonal/mixed/core-edge/reserve/DR/push recovery/transfer conditions.

**Next decision: ordinary full-state resume from CK1000 for999 additional
updates**, preserving actor/critic, both normalizers, learned std, Adam and fixed
LR. Lateral drift/holding and rolling pose have useful gains, with no observed
unsafe contact/collapse in this early panel; the CK750 check does not identify
a better forward checkpoint or a late-only failure. Finish the original lineage
budget before changing another objective. No new reward proposal is justified
by averaged costs alone. This is not exact simulator/RNG continuation and does
not recover the115 unsaved updates. Expected final native label is **1998**,
with **2000 lineage updates** and Adam step80000 (native resume repeats source
label1000). Saves remain every250 plus native final, in a separate timestamped
run. This exact command is **unexecuted**:

```powershell
& 'C:\Users\Liamb\anaconda3\envs\genesis-gpu\python.exe' -m robot_gym.scripts.train --task go2w --resume --load_run go2w_2026-10-08_09-57-35 --checkpoint 1000 --num_envs 4096 --max_iterations 999 --seed 1 --logger tensorboard --training_diagnostics --headless --rl_device cuda:0
```

At that budget, compare the resulting actor-only policy against these retained
CK1000/CK1999 traces under the same frozen tracking/path/holding/stopping,
unloading/contact and sensor requirements. No automatic continuation or tuning
is authorized if it remains ineffective.

## Completed geometric-budget review (2026-10-08)

**Decision D: do not promote final1998 or intermediate1750. Retain
`logs/go2w/go2w_2026-10-08_09-57-35/model_1000.pt` as the schema2 development
baseline, close the original budget, and pause additional training.** The final
policy gains useful lateral coordination, but loses holding/stopping performance
and still curves substantially forward. No reward-scale refinement or training
command is prepared; this decision does not qualify CK1000 either.

The completed child is
`logs/go2w/go2w_2026-10-08_13-31-18/model_1998.pt`. Saved provenance explicitly
names the CK1000 parent and ordinary full native resume. Adam advances
40040→80000, both normalizer counts 262406144→524288000: **1001+999=2000 completed
lineage updates**. Native labels repeat the source label; saved child labels are
1000/1250/1500/1750/1998. Preparation says `completed`, TensorBoard reaches1998,
and the 993-row diagnostic JSONL lacks six rows (1171,1427,1656,1838,1932,1946),
which are unavailable rather than zero. Saved schema2, actor58/critic65, geometric
scales, fixed LR3e-4 and all other task/PPO values match the parent; differences
are resume provenance/load flags and the local999-update budget.

The [compact comparison](go2w_geometric_completed_2026-10-08.json) records all-axis
means/bias/RMSE, first-second/full/late windows, stops, geometry, target/clipping
details and sensor decomposition. Large traces and detailed calculations remain
in `evaluation/go2w_geometric_final_review/`. CK1000 and pre-geometric CK1999 raw
baselines were reused; no historical model was replayed. Any geometric rescoring
of pre-geometric traces is counterfactual, not its original training reward.

### Physical result and selection

One final-panel invocation used stand30s and moving3s zero/30s motion/8s zero:
forward(.5,0,0), lateral(0,±.3,0), yaw(0,0,±.8). Deterministic actor mean,
saved nominal dynamics, noise/DR/pushes disabled, unchanged phase/reset conventions,
including the existing zero-action settling tick. Initial pose, commands and
desired phase/clearance arrays match the retained panels. Primary analysis is50Hz;
only positive lateral has200Hz capture in this review.

| Full movement / holding metric | Pre-geometric1999 | Geometric1000 | Final1998 |
| --- | ---: | ---: | ---: |
| Forward mean vx / RMSE, m/s | .49579/.01622 | .49945/.01780 | .50421/.01677 |
| Forward heading / max initial-line cross-track | +4.24°/.508m | +8.55°/1.024m | **+7.11°/.845m** |
| Lateral+ mean vy / RMSE, m/s | .30049/.01531 | .30363/.02036 | .30076/.02021 |
| Lateral+ heading / max cross-track | −18.81°/1.618m | −1.75°/.217m | **+3.66°/.343m** |
| Lateral− mean vy / RMSE, m/s | −.30088/.01441 | −.30421/.02051 | −.30011/.02006 |
| Lateral− heading / max cross-track | +18.18°/1.758m | +8.15°/.677m | **−1.11°/.126m** |
| Yaw+ / yaw− achieved wz, rad/s | .79956/−.80178 | .80928/−.80600 | .81071/−.81081 |
| Stand endpoint / path, m | .0981/.1269 | .0535/.0539 | **.1759/.1840** |
| Late stand XY / wz RMS | .00439/.01117 | .00141/.00037 | **.00662/.00593** |

Forward curvature remains late (+2.70° in the last10s). Its cross-track integral
is `.84295 + .00237 − .000215m` from rotated body-x, body-y and body-z velocity;
the pose discrepancy is0.20mm. Thus accurate requested-axis speed does not imply
straight travel. Lateral initial-forward displacement decomposes as
`+.01012 − .31051 − .04106m` (+) and `−.00204 − .08238 − .04058m` (−), agreeing
with pose within1.3mm. Direct body-x bias improves to+.000336/−.000069m/s;
body-wz means are+.00580/−.00432rad/s with RMSE.03513/.03377. Pose-derived heading
is authoritative: tilted body angular-z is not Euler heading rate. Late heading
changes +1.19°/+.028° for the two lateral signs; the negative case mostly retains
an earlier offset, whereas forward/positive lateral continue curving.

Final first-second requested-axis means are .46982m/s forward and+.28106/−.28353
lateral; corresponding RMSE is .08697/.04449/.04327m/s. The path improvement is
not obtained by materially slowing sustained travel. Stand startup0–5s endpoint/
path is .0177/.0246m; most of the final17.6cm drift continues afterward.

| Complete8s stop | CK1000 endpoint/path, m | Final endpoint/path, m | Final last2s XY/wz RMS |
| --- | ---: | ---: | ---: |
| Forward | .0443/.0455 | .0824/.0825 | .00667/.00591 |
| Lateral+ | .0201/.0396 | .0599/.0863 | .00812/.00766 |
| Lateral− | .00473/.0425 | .0529/.0676 | .00508/.01395 |
| Yaw+ | .0257/.0273 | .0336/.0441 | .00526/.00586 |
| Yaw− | .0197/.0215 | .0410/.0450 | .00477/.00110 |

Final lateral stop opposing travel is4.0/5.9mm; net/absolute heading travel is
+2.18°/3.33° and−1.92°/6.79°. Yaw stops reverse direction (positive/negative
rotation2.85°/2.12° after yaw+,4.00°/1.89° after yaw−). These are separate from
endpoint displacement; small net values do not establish good stopping.

The one permitted intermediate check used **child1750, stand/forward only**:
1752 lineage updates, Adam70080. It gives forward .50690m/s, RMSE.01702,
cross-track.687m/heading+6.08°, but stand endpoint/path .2601/.2631m and late
XY RMS.00954. Forward stop endpoint/path .1044/.1044m and final XY RMS.00935
are worse again. Its lateral/yaw behavior was not tested; it does not resolve
the higher-priority holding/stopping regression.

### Clearance, body motion and camera

Completed physical swing median peaks FL/FR/RL/RR (mm) change from CK1000
`11.45/11.86/10.05/11.63` to `10.14/13.75/6.20/24.17` for lateral+, and from
`13.61/12.33/11.35/11.79` to `14.91/11.65/24.00/6.32` for lateral−. Complete,
load-only and censored events and peak distributions are retained separately.
Extra clearance is asymmetric; low liftoff/touchdown gaps are not automatically
dragging. Apex unloading improves to `.847/.995/.914/.974` (+) and
`.974/.865/1.000/.958` (−). Positive-lateral200Hz peak lags are
`−20/+20/−90/+20ms`, versus CK1000's `−95/−95/−105/−30ms`. Total brief sampled
recontacts fall111→17 (per wheel21/43/12/35→1/0/16/0; RL increases12→16).
The shallow opposite rear remains early even when
mostly unloaded. Negative lateral mirrors this limitation.

The unchanged smooth envelope `s` specifies desired gap `.04*s` and weights
support cost `s*min(F/Fnom,2)^2 + (1−s)*relu(.2−F/Fnom)^2`; clearance remains
Huber((gap−desired)/.04), averaged over four wheels. Final lateral clearance cost
is57.8%/54.5% of the **same-target zero-clearance counterfactual**, versus79.5%/80.1%
at CK1000. This is a cost ratio, not time spent dragging. Support and loaded
scrub costs also fall; loaded lateral center RMS remains roughly.065–.084m/s.
The surrogate measures lateral wheel-center motion, not tire-patch slip or
legitimate longitudinal rolling. These observations do not establish a timing
bug, defective envelope, or a need to raise the4cm apex.

Ground clearance is not synonymous with articulated leg lift. For positive RR,
mean within-desired-window start-to-peak gap change is11.39mm, decomposed into
base heave+16.54, base rotation+10.82, relative leg motion−13.80 and cylinder
extent−2.17mm. CK1000 gives5.90mm =1.88−.15+3.53+.63mm. Boundaries can already
be elevated; these exact finite contributions are not causal effects or the
final-minus-parent physical peak difference.

| Full lateral + / − | Pre-geometric1999 | Geometric1000 | Final1998 |
| --- | ---: | ---: | ---: |
| Base height std, m | .00387/.00374 | .00320/.00301 | **.00581/.00590** |
| Base world-z velocity RMS, m/s | .05497/.05421 | .03447/.03420 | **.07996/.08327** |
| Camera rotational-lever z RMS, m/s | .09664/.09202 | .09154/.09056 | **.12063/.11914** |
| Camera-body world-z RMS, m/s | .09119/.08614 | .10973/.10719 | **.09953/.09985** |
| Retained imager world-z RMS, m/s | .09399/.09555 | .11413/.11645 | **.10457/.10803** |

Using the existing rigid transforms, `vcamera_z=vbase_z+vrotation_z`. For final
lateral+, the mean-square identity is `.009907=.006393+.014551−.011037m²/s²`;
for lateral−, `.009970=.006935+.014195−.011160`. The negative cross term is
essential: greater base/rotational motion can produce lower camera RMS through
cancellation. Final camera height std is .01221/.01225m, down from CK1000
.01354/.01336m, measured directly from positions. Late camera RMS remains
.10005/.09923m/s; positive200Hz body/imager RMS .10221/.10742 confirms direction.
Pitch-rate RMS increases to .37967/.37483rad/s while roll-rate RMS falls to
.22062/.22157. Signed roll/pitch means are+.03194/+.01440rad and−.03240/+.01422;
oscillatory std is about.0165/.040rad. Base motion is worse; camera motion is
better than CK1000 but worse than pre-geometric1999. No image quality or jerk
claim follows, and this does not justify increasing both sensor/angular costs.

### Posture, safety, logs and closure

Forward actual reference-pose RMS improves .05879→.04679rad, mean height is
.41615m and roll/pitch−.00628/−.00291rad. Early-to-late height changes−1.32mm,
then only−.105mm between the first/last2s of the late window: bounded migration,
not collapse. Final forward wheelbase is.38829m; front/rear widths.40603/.42192m;
front/rear pair midpoints are(.22045,−.00963)/(−.16784,−.00403)m. Lateral mean
height remains.4132–.4134m with early-to-late changes below.4mm; posture RMS
.14815/.14752rad is distinct from necessary support target offsets. Complete
early/late geometry is in the JSON; desired height/reference/gains are unchanged.

No falls, resets or sampled self/non-wheel contacts occur in the final panel.
All joints have zero sampled99%-force-limit occupancy; positive200Hz maximum is
.694 of limit. Positive calf-extension clipping remains predominantly stance;
however FR/FL calf **negative flexion** clips37.8%/34.2% of lateral+/− apex
samples. The persistently shallow opposite rear has neither-sign apex clipping.
Actual/issued/applied targets remain distinct in reporting. This does not support
widening stance-extension limits to solve that rear swing. Contacts between
samples and unsampled conditions remain unknown.

Labels1200–1399 versus1800–1998 show KL.01446→.01366, PPO ratio clipping25.77→24.91%,
value loss.01284→.00969 and fixed LR3e-4. Global cross/heading/clearance cost rates
decline .01254/.00489/.02829→.01120/.00434/.02581; sensor declines.00801→.00759,
but orientation grows.00384→.00433. Family-conditioned term costs by hold/phase
were not recorded. Late std-floor occupancy is about2–3% for front calves and
19–20% for rear thighs; final rear-thigh std=.10. Rear-calf deterministic mean
saturation is about3.94% on training states, distinct from PPO ratio clipping.
None of this warrants resetting exploration. Episode terms divide accumulated
contributions by configured60s; mean return is an episode sum. They are not
interchangeable or failure probabilities. Recorded collection/learning means
are4.209/.855→4.007/.844s; no profiling or bottleneck claim was made here.

Focused numerical checks pass rigid transport/cross-term identities, rotation/
translation invariance, known peak lag, exact wheel-lift decomposition and matched
protocols. A recording-only `--diagnostic_trace_cases` selector limits existing
capture; omitted options preserve its old default. Actual output confirms only
positive lateral has200Hz data. No training parameter changed, no optimizer
update/smoke ran, and checkpoint/config/`play.py` hashes are unchanged.

The frozen30s forward **.05m/2°** targets still fail materially despite passing
the .45–.55m/s / RMSE≤.04m/s speed band. Final stand fails late XY≤.005m/s;
stopping regresses versus CK1000 and retained requirements. Improvements in
negative lateral, clearance and camera cannot offset those losses. The geometric
terms are inactive during stand/stops, so a stronger translating-heading scale
would not directly address this combined failure. No single scale or shape fix
is established by cost magnitudes. The actor still lacks online accumulated
offset; an asymmetric critic does not supply hidden path-recovery feedback.

**Retain CK1000, reject promotion of final1998/1750, and stop training at this
closed budget.** Persistent forward curvature is the retained baseline's concrete
unresolved functional limitation. No command-envelope, reverse/small/mixed,
DR/push, Sim2Sim or hardware qualification is claimed, and no further unchanged
block, fresh start, camera-only refinement or training command is prepared.
