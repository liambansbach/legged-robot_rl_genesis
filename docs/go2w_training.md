# Current Go2-W training

`GO2WCfg` and `GO2WCfgPPO` in
[go2w_config.py](../robot_gym/envs/go2w/go2w_config.py) are the single current
parameter source. Edit that file for the next experiment. Training resolves the
registered task and writes the complete `config.yaml` into its run directory.
No documentation YAML, historical profile, or fine-tune selector supplies parameters.

**Current preparation, 2026-10-08:** training schema 2 adds straight-motion geometric
supervision and a 65-input asymmetric critic; the deployed actor remains 58 inputs.
Yaw tracking is restored to `.8`. See the [task definition and validation below](#geometric-straight-motion-preparation-2026-10-08).
CK1999 remains the development baseline; this new run starts entirely fresh.

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
adapter. Production remains unstarted. The selected development baseline is
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
