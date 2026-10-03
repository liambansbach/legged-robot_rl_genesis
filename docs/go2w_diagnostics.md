# Go2-W diagnostics and reproducibility

Current workflow: [frozen reference and phase-guided V3](#frozen-reference-and-phase-guided-v3-2026-10-03).
Use [saved-config replay](#shared-pipeline-cleanup-2026-10-02) for existing runs.
Older dated entries below preserve historical evidence and commands; their former
selection/export rules are superseded by the current replay section and `--help`.
The older v2 long-run command is historical, not the next training recommendation.
The earlier fixed-target failure is retained as evidence; the owner superseded it
as a prerequisite for the V3 pilot.

## Archived diagnostics — 25 September 2026

The workspace started clean on `testing` at the audited HEAD `3bfb7608b0e3ff75627eb96b2d1c4903c9b7b661`; HEAD remains unchanged. The working changes are this diagnostics pass. No URDF, training reward values, action scales, observations, gains, physics or PPO defaults were changed. CK1598, CK800, CK999, their saved configs and both uploaded ZIP archives retain their original SHA-256 hashes. No generic exports were written.

Saved contracts for all three checkpoints pass the actuator/observation/physics comparison. Differences are retained explicitly: v2.3 reward gates differ from current v2.4, and saved build-time batching flags/run bookkeeping differ from unbuilt defaults. Missing saved configs and important contract mismatches fail before simulation. There is no automatic legacy migration.

Implementation files:

- `robot_gym/utils/diagnostics.py`: provenance, contract checks, summed ground force, finite-cylinder geometry, FK, solver property readback and opt-in physics capture.
- `robot_gym/scripts/evaluate.py` and new `diagnostic_bank.py`: unchanged 31-case nominal suite, explicit checkpoint selection, fresh output directories, and separate fixed bank/equilibrium modes. The default nominal batch is one environment.
- New `robot_gym/utils/training_diagnostics.py`: wrappers around installed RSL-RL rollout, minibatch, log-probability and KL interfaces; no PPO implementation copy. Logs 16-vectors and leg/wheel summaries, raw/effective std and bounds, clipping, target slew and nonterminal reward clipping by sampler family.
- Small opt-in hooks in `legged_robot.py`, `go2w_env.py`, `train.py`, `helpers.py` and `task_registry.py`; focused tests in new `test_diagnostics.py`, plus existing contract/GPU integration tests.

Evidence is in `.migration-audit/diagnostics-20260925/`. `reference_inventory.json`, `reference_contracts.json`, `findings.json` and `analyze.py` retain hashes, comparisons and reproducible postprocessing. Each evaluation records checkpoint/URDF SHA-256, HEAD, tracked diff hash, untracked source hashes, resolved configs, overrides and versions. Runtime: Python 3.11.14, Torch 2.9.0+cu130, Genesis 1.4.1, RSL-RL 5.5.1, RTX 4070 Ti.

| Validation actually run | Result and exit status |
|---|---|
| CPU discovery, including observation/action/symmetry/export tests | 34 passed; one opt-in GPU test skipped; **0** |
| Uninstrumented versus instrumented CK800, 31 cases, one environment, 150 steps/phase | Every original trace field bit-identical; **0 / 0** |
| Instrumented CK800 replay at archive batch size eight | All 31 original traces **and all original metric fields bit-identical**; **0** |
| CK800 fixed 32-condition bank, 600 steps/case | Eight policy cases: no falls; zero-action probe: 1/32 fell; **0** |
| CK999 nominal and two mass/gain corner probes, 600 steps/case | Zero-action and policy stand: no falls; **0** |
| Native 64-environment PPO integration | Two separate fresh smoke runs, exactly two updates each; **0 / 0** |
| Ruff error/undefined checks and `git diff --check` | **0** |

Trace comparison tolerance was `atol=1e-6, rtol=0`; observed differences were zero in both matched-batch comparisons. Changing batch size from eight to one changes floating-point inference enough to diverge in later contact transitions: archived versus one-environment yaw-rate RMSE differences reached 0.01220 rad/s, and instantaneous yaw-rate differences reached 0.35835 rad/s. This prompted the matching eight-environment replay. All 124 cases across the uploaded archives contain eight bit-identical replicas; these are not independent trials.

Main manifests: `nominal-ck800/manifest.json`, `archive-replay-ck800/manifest.json`, `bank-ck800-r2/manifest-with-evidence.json`, and `equilibrium-ck999-r2/manifest-with-evidence.json`. The latter two bind realized conditions, readback and derived `summary.json` files by hash. Raw traces and original runtime metrics are retained separately. Initial native starts hit cache permissions; caches were redirected. Initial bank/probe attempts exited 1 because the CPU FK helper inherited Genesis's CUDA default device; explicit CPU allocation fixed this, a regression test covers it, and the reruns exited 0.

The measured **Genesis control-force API readings** over seconds 3–6 were:

| Checkpoint/case | FL | FR | RL | RR |
|---|---:|---:|---:|---:|
| CK800 nominal stand, Nm | -2.152 | -2.164 | +2.176 | +2.135 |
| CK999 bank nominal-condition stand, Nm | -2.350 | -2.410 | +2.407 | +2.350 |

CK999 uses the explicit bank reset, so it is not an exact replay of its archived stand. These readings agree with separately calculated Kv-error estimates. Genesis's public getter recomputes force from current controller state and clamps it to force limits; it is not a stored integration impulse. The trace also retains the maximum absolute API reading over all four 5 ms physics steps.

CK800 stand wheel loads were `[46.55, 46.61, 49.37, 48.99] N`. At +0.4 yaw, legacy false-support fractions were **15.5% per wheel sample** and **48.0% for any wheel**. Only **1.83%** of wheel samples had geometric gaps above 1 mm; **88.2%** of false-support samples had gaps at or below 1 mm. This distinguishes unloading from substantial lift-off. The aggregate force is invariant to splitting a fixed total among contact points; legacy reward predicates remain unchanged.

Clearance uses the URDF collision center (signed lateral offset ±0.0481 m), full link rotation, radius 0.086 m and half-width 0.026 m:
`zc - r*sqrt(1-u_z**2) - half_width*abs(u_z)`.
It describes the ideal finite-cylinder envelope over z=0, not a rounded hardware tire or the exact solver manifold. Base/wheel quaternions are wxyz. The legacy Euler field is explicitly Genesis intrinsic XYZ, not ROS extrinsic RPY; heading comes from projected body +X. Convention, heading and signed-camber tests pass.

Nominal-joint FK requires base heights `[0.43185, 0.43185, 0.43367, 0.43367] m`, above the 0.415 m reward reference. Late zero-action heights were **0.40468 m nominal**, **0.40881 m light/high-gain**, and **0.38777 m heavy/low-gain**. Loaded sag therefore does not mean the nominal joint angles were attained. In CK800 nominal stand, **all 12 raw deterministic leg means exceed action bounds and all 12 leg targets clip**. This is measured from raw actions/targets, not inferred from joint-angle error. Standing control forces remain below hardware force limits. These observations establish a soft pose/height/target-authority conflict, not a justified replacement height target.

Actual mass, COM, inertia, friction and gains were read before/after selective reset. Both reset and non-reset environments preserved all queried properties exactly. Nominal/light/heavy masses were 19.523/19.023/21.023 kg. Existing payload DR changes base mass/COM while leaving inertia unchanged; this is an approximate payload model. The installed friction rule is `max(wheel coefficient*ratio, ground coefficient*ratio, 0.01)`; ground ratio is one here.

The seeded bank contains saved headings, joint/body perturbations, friction, mass/COM, gains and 0/1-step delays, including fixed corners. Each phase lasts six seconds; sustained holds last twelve. Median/max absolute XY drift during six-second stop holds was **0.171/0.477 m** after forward, **0.262/0.532 m** after positive yaw, **0.282/0.717 m** after negative yaw and **0.305/0.811 m** after the navigation stream. The nominal-condition stand moved 0.329 m longitudinally over its twelve-second trace. The 10 Hz stream adds no command/dynamics filter. Per-condition results retain all velocity axes, quaternion heading, drift, contact and fall outcomes; tracking excludes failed conditions. This small fixed bank is not an IID robustness estimate.

The final training smoke is `logs/go2w_diagnostics_smoke/symmetry_2026-09-25_21-28-03/diagnostics.jsonl`. All 3,072 transitions/update have command-family labels. The second smoke was necessary to verify labels for commands sampled before logger attachment. Paired CPU checks prove unchanged sampler outputs/RNG and unchanged PPO losses, weights, normalizers and RNG with instrumentation.

| Fresh-policy smoke statistic | Update 0 | Update 1 |
|---|---:|---:|
| Sampled action clipping, legs/wheels | 7.297% / 7.365% | 7.365% / 7.585% |
| Effective std, legs/wheels | 0.55013 / 0.55009 | 0.55038 / 0.55018 |
| Scheduler KL, minibatch mean | 0.25515 | 0.02697 |
| PPO ratio clipping fraction | 78.151% | 55.804% |
| Raw nonterminal reward mean, dt-scaled | 0.014391 | 0.012393 |
| Negative/clipped nonterminal fraction | 5.404% | 11.914% |
| Discarded negative magnitude/sample | 0.00013491 | 0.00036266 |
| Mean/sampled leg-target slew RMS, rad/s | 0.557 / 7.314 | 0.824 / 7.328 |
| Mean/sampled wheel-target slew RMS, rad/s² | 51.68 / 659.24 | 78.56 / 671.61 |

These are instrumentation smoke statistics, not CK800/999 performance estimates. LR reached 1e-5; the installed adaptive range is 1e-5–1e-2, independent of initial LR 8e-4. Mirror loss stayed disabled; nonzero symmetry diagnostics do not activate it. CK800/999 each have two raw log-std values above `log(0.8)`, unchanged between checkpoints; effective std clamps at 0.8. CPU tests confirm zero clamp gradient outside the bounds. Lowering entropy alone cannot directly move those parameters back inside.

To reproduce, use explicit experiment `go2w_flat_pilot_v2_4_yaw_mobility`, run `augmentation_seed1_2026-09-25_16-27-23`, checkpoint 800, seed 1, headless and TensorBoard. Run `python -m robot_gym.scripts.evaluate` with `--diagnostic_trace --num_envs 1 --steps 150` for nominal or `--eval_mode bank --num_envs 32 --steps 300` for the bank; choose a new `--output` each time. CK999 probes use `--checkpoint 999 --eval_mode equilibrium --num_envs 3 --steps 300`. Use `--training_diagnostics` only for explicitly requested training. Windows runs used `NUMBA_CACHE_DIR` under this audit directory and `GS_CACHE_FILE_PATH` / `QD_OFFLINE_CACHE_FILE_PATH` under `%TEMP%`, plus `PYTHONIOENCODING=utf-8`.

Recommended next experiment: **only change the x-tracking denominator from 0.25 to 0.09**, paired with an unchanged control from the same complete checkpoint state. Preserve optimizer, raw std and normalizers identically in both branches, and reuse this nominal suite and condition bank. Ignoring +0.1 m/s currently earns 0.960789; the proposed denominator would give 0.894839. No such reward change or long training run was implemented. Hardware tire behavior, broader robustness and whether this isolated incentive improves stopping remain unresolved.

## Paired x-tracking continuation preparation — 26 September 2026

Preparation started on clean `testing` at `18efe5a0db9c3a68b9b9fe64c744bbedf94bcbc5`. Both arms explicitly resume `logs/go2w_flat_pilot_v2_4_yaw_mobility/augmentation_seed1_2026-09-25_16-27-23/model_800.pt` (SHA-256 `d0da829b95c683ce977323c4af88922c2a86ac9e680ff4501af302704c0cfcc1`). The sole behavioral change is `--tracking_sigma_x 0.09` versus control `0.25`; the optional override defaults to `None`, works in training/evaluation, and leaves registered defaults and the product exponential with y denominator `0.04` unchanged. The hypothesis concerns precision/stand/yaw/stop vx errors, not an assumed cure for clipped leg targets or wheel opposition.

Before simulation, continuation checks the saved config beside the explicit checkpoint. Only the explicit x override, run name/resume selection/update budget/logger, diagnostic labels and build-time batching may differ. A 64-environment exception is limited to two-update smokes; seeds, physics, rewards, sampler/reset/DR/noise, model and PPO settings otherwise must match. Missing parents/configs and occupied/parent output folders fail. Native RSL-RL 5.5.1 loading is followed by exact comparisons of actor/critic (including raw std and both normalizers), optimizer tensors/counters/group LRs, algorithm LR and iteration against the parent. CK800's loaded LR is `0.0003844335937499999`, not the configured initial `0.0008`. No state is repaired or reset.

Fresh processes use seed 1 before environment construction. This matches saved learning state and initialization protocol, not the original historical simulator/RNG trajectory. Ordinary `config.yaml`, opt-in `diagnostics.jsonl` and compact `continuation.json` stay inside fresh ignored run folders. Metadata binds checkpoint/config hashes, the saved training git snapshot (HEAD `2c9c186973b9f1c7094e9da273c0a00c439fbfb9` plus its dirty patch), current source identity, verified state/LR, x denominator, seed, paths and planned/completed additional updates. Native labels begin at saved `iter`: two updates end at `model_801.pt`; **300 additional updates end at `model_1099.pt`**, with interval saves at 800, 850, …, 1050. Suffixes are not update counts.

The prepared real runs are sequential, headless, 4096 environments × 48 steps, 300 additional updates, saves every 50, W&B and `--training_diagnostics`, using distinct `xtracking_control_seed1_*` / `xtracking_009_seed1_*` outputs. Neither real run has been launched; both parents remain original CK800. Smoke outputs must never become parents.

Validation: `python -m unittest discover -s tests -v` passed 37 CPU tests with two GPU skips (exit **0**). Exactly one `python -m unittest discover -s tests -p test_go2w_symmetry_integration.py -v` ran per arm in `genesis-gpu`, selected by `GO2W_RESUME_SMOKE_SIGMA=0.25` / `0.09` (exits **0 / 0**). Outputs under the source experiment are `xtracking_control_smoke_seed1_2026-09-26_11-01-41` and `xtracking_009_smoke_seed1_2026-09-26_11-03-56`. Each verified original state, 64 × 48 rollouts, two updates, finite losses/diagnostics/output and native save/reload. Their `continuation.json` and `smoke_validation.json` retain hashes and arguments. These are wiring checks, not learning results. Ruff `--select E9,F63,F7,F82` and `git diff --check` passed (exit **0**); 18 protected checkpoint/config/archive/export hashes are unchanged. Logs and hash evidence remain ignored under `logs/xtracking_preparation_20260926`. No validation run failed.

After both finish, use `python -m robot_gym.scripts.evaluate` with explicit final run paths, `--checkpoint 1099 --num_envs 1 --steps 150 --diagnostic_trace`, the arm's x override, seed 1 and fresh outputs. Compare to the unchanged continuation and `.migration-audit/diagnostics-20260925/nominal-ck800` (one environment, 150 steps/phase), not the eight-identical-replica archive. Only if nominal improvement is useful without regressions, evaluate both with `--eval_mode bank --bank_seed 240925 --num_envs 32 --steps 300 --skip_zero_action_probe`; this omits only the zero-action probe, retaining the existing conditions and all eight policy cases. Reuse CK800's `bank-ck800-r2` evidence.

Judge physical metrics across all velocity axes and final-window residual speeds: stand, +0.1/+0.5 m/s, stopping and both yaw-to-stop signs; preserve lateral signs, sustained yaw, arcs, reverse/braking, fall and non-wheel-contact outcomes. Keep failed-condition counts separate from survivor tracking. Retain bank per-case median/worst XY displacement and path length from the zero-command boundary **including braking**, with heading reported separately. Inspect raw actions, leg target clipping and wheel control forces. Targets `|vx| ≤ 0.02 m/s` at zero command and `|vx−0.10| ≤ 0.02 m/s` for precision are practical goals, not results or safety guarantees. Returns use different reward definitions and are not the comparison criterion. One seed pair cannot establish general superiority; if ambiguous, inspect at most one matched saved pair before proposing further work, without automatic extensions.

## X-tracking outcome and bounded entropy test — 26 September 2026

The completed `xtracking_control_seed1_20260926_111810_2026-09-26_11-20-21` and `xtracking_009_seed1_20260926_111810_2026-09-26_11-36-57` runs have the expected final SHA-256 hashes (`9762ca61…d530b2` / `08a8d400…e04f1`). Both recorded exact original CK800 loading, seed 1, clean source `7dc3e2b`, and 300 additional updates; all Adam step counters increased by 12,000. Saved configs differ from the parent only in documented bookkeeping/diagnostics and the variant's x denominator. The unchanged control is therefore the equal-budget comparator for the entropy test.

Existing one-environment nominal traces were reprocessed without rerunning simulation:

| Final one-second mean | Control 0.25 | Variant 0.09 |
|---|---:|---:|
| Stand vx, m/s | -0.07569504 | -0.07643226 |
| +0.1 actual vx, m/s | 0.02012545 | 0.01813041 |
| +0.5 actual vx, m/s | 0.43861822 | 0.43473228 |
| Forward-to-stop vx, m/s | -0.07839018 | -0.05866090 |
| Yaw-to-stop vx, m/s | 0.02686731 | 0.03081403 |
| Forward-to-precision vx, m/s | 0.01532532 | 0.03820157 |
| Stand base height, m | 0.41878144 | 0.44784880 |

Both have zero recorded falls/non-wheel contacts, but neither passes sustained stopping below 0.05 m/s. The variant's 0.14 s step-relative settling uses approximately 0.074 m/s tolerance and is not a stop pass. Three-second forward-stop displacement, including braking, improves from 0.12959 to 0.09417 m. Secondary improvements do not meet the primary stand/precision goals. Keep x=0.25; this single seed pair does not universally reject tighter tracking. Variant arc front/rear joint RMS is 0.13579/0.18403 rad, while positive-yaw front/rear hip RMS is 0.19561/0.17495 rad. This supports no new front/rear penalty. Loads and small gaps suggest loaded low-clearance motion; absent tangential force recordings prevent attributing it to friction.

The requested diagnostic windows are labels 800–829 and 1070–1099. Local JSONL files contain only 285/288 of 300 records; available window counts are control 26/29 and variant 29/29. TensorBoard also has gaps. No missing values were imputed. Available-row mean effective leg std rises 0.6159→0.6995 (control), 0.6206→0.7421 (variant); sampled leg clipping rises 65.88→72.71% / 66.52→72.16%, and mean saturation 71.10→77.63% / 71.84→77.03%. The two front-thigh raw log-std values already exceed log(0.8); late control also has rear-thigh excursions, variant rear-thigh/calf excursions. Detailed 16-joint raw/effective std, bounds occupancy, slew, clipping, KL and reward-clipping summaries, all physical axes, hashes and missing labels remain in `.migration-audit/entropy-20260926/`. Physical errors by training command family were not logged and are not inferred.

The next intervention is only explicit `--entropy_coef 0.001` from original CK800, preserving x=0.25 and every learned state through native loading. Unset still means 0.005. The strict check permits only the exact explicit entropy request, verifies the effective loaded coefficient, and records it in `continuation.json`. No std projection, optimizer/LR reset or other dynamics change is introduced. Out-of-clamp parameters can have zero local gradient, so lower entropy does not guarantee decreasing every std. The authorized bounds are one two-update smoke, one 300-update run, nominal checkpoints 950/1099, and at most one justified bank candidate.

Preparation validation: the first CPU discovery exited 1 because the existing test file's ending was truncated after formatting; restoring its unchanged ending fixed this. The repeated discovery passed 39 tests with two opt-in GPU skips (exit 0). Ruff error checks and `git diff --check` passed. The single native smoke `entropy_001_smoke_seed1_2026-09-26_12-55-54` completed exactly two 64×48 updates and exited 0, verifying original state, effective entropy 0.001, finite losses/diagnostics/output and native save/reload. Its parent remains original CK800. Detailed logs are ignored under `.migration-audit/entropy-20260926`.

## Bounded entropy result and integration decision — 26 September 2026

The single real run `logs/go2w_flat_pilot_v2_4_yaw_mobility/entropy_001_seed1_2026-09-26_12-58-36` completed **300 additional updates**, 4096×48, seed 1, W&B, exit **0**. Original CK800 was the parent; native actor/critic, raw std, both normalizers and optimizer state/LR matched exactly before updating. Loaded LR was `0.0003844335937499999`; effective entropy was explicitly 0.001, x remained 0.25. All runs/evaluations used clean source `b94221201636b44f94d9eb2e8bdbd203727b37d9`; this result section is a later documentation-only change. Learning initialization/config checks support comparison with the existing unchanged control, without claiming bit-identical simulator trajectories.

`model_950.pt` has SHA-256 `fa53d2b521f48e332624ed306e8879dd51e207e7599cf0736b51ef2f27c6d491`; `model_1099.pt` has `4c894b4b7cc9cc6680aaadba6f41ea923f8298b05b1d855404199f3afc26286b`. Adam counters independently establish **151 / 300** additional updates, respectively (deltas 6,040 / 12,000). Saved tensors and available loss scalars are finite. Diagnostic records cover 273/300 labels, with 29/18 available in the requested first/last 30-update windows. These incomplete windows are descriptive, not imputed full-window estimates.

Available-window effective std, legs/wheels, fell from **0.5932/0.3585 to 0.4780/0.2183**. Sampled clipping changed **66.73/7.10%→69.87/3.52%**, mean saturation **71.88/2.94%→72.73/2.48%**. Mean/sample leg-target slew changed **3.312/4.946→3.218/4.320 rad/s**, wheel slew **193.82/494.94→170.82/334.21 rad/s²**. Scheduler KL was 0.01426→0.01461; PPO ratio clipping 24.32→25.15%; negative reward clipping 0.3486→0.2734%, discarded magnitude/sample 3.489e-5→2.478e-5. FL/FR thigh raw log-std stayed exactly -0.22111896/-0.22163430, above log(0.8)=-0.22314355; effective std remains 0.8. No clamp repair was performed. Full per-joint vectors, bounds occupancy and command-family reward summaries remain in the ignored audit directory.

Only new checkpoints 950/1099 were evaluated. Their failures justified the one authorized v2.3 fallback nominal evaluation, resolved from the protected inventory: `logs/go2w_flat_pilot_v2_3_lateral_exposure/go2w_flat_pilot_v2_3_lateral_exposure_2026-09-25_13-14-38/model_1598.pt`, SHA-256 `087418d168a95f2784ebbb31daa6638101522ade1d08f604dc90ccbd668195e3`. All three nominal processes exited **0**, each 31 cases, one environment, seed 1, 150 steps/phase, no falls/non-wheel contacts. Existing CK800/control/0.09 evaluations were reused.

| Final one-second measurement, m/s | CK800 | Control 1099 | Entropy 950 | Entropy 1099 | v2.3 CK1598 |
|---|---:|---:|---:|---:|---:|
| Stand XY RMS | 0.02074 | 0.07572 | 0.03750 | 0.02132 | 0.00565 |
| +0.1 actual vx | 0.07370 | 0.02013 | 0.05039 | 0.05951 | 0.09641 |
| +0.5 actual vx | 0.46366 | 0.43862 | 0.42793 | 0.41886 | 0.48374 |
| Forward-to-precision actual vx | 0.07483 | 0.01533 | 0.06584 | 0.07333 | 0.10257 |
| Forward-stop XY RMS | 0.02314 | 0.07839 | 0.03012 | 0.01570 | 0.01328 |
| Positive-yaw-stop XY RMS | 0.06454 | 0.02707 | 0.01258 | 0.07316 | 0.01024 |
| Positive-lateral-stop XY RMS | 0.05773 | 0.08290 | 0.06773 | 0.02096 | 0.03154 |
| Negative-lateral-stop XY RMS | 0.07756 | 0.07712 | 0.06362 | 0.01675 | 0.00439 |

Entropy 1099 improves forward/lateral stopping against control and CK800, but misses precision and worsens yaw-stop translation. Its four stop yaw RMS values are 0.00072/0.00904/0.00201/0.00152 rad/s; low yaw residual does not cancel XY drift. Reverse, bilateral lateral, sustained yaw and arcs remain available, but +0.5 speed drops and ±0.4 yaw overshoots to -0.4535/+0.4695 rad/s. Checkpoint 950 is not an equal-budget comparison. CK1598 improves nominal precision substantially, narrowly misses the positive-lateral stop XY goal, and has slower reverse/fast-arc motion. The unchanged nominal suite contains positive yaw-to-stop only; both signs are covered in the bank.

All 12 deterministic stand leg targets still clip in both entropy checkpoints and CK1598. Measured final-second wheel control torques at entropy 1099 are `[-2.319,-2.329,+2.340,+2.289] Nm`; CK1598 has `[-0.880,-0.870,+0.873,+0.883] Nm`. These are Genesis control-force readings, not Kv-error estimates. Arc front/rear joint RMS is 0.12862/0.16339 rad at entropy 1099 versus 0.16102/0.20125 at CK1598. Neither recovers the existing pose criterion after nominal yaw-stop; entropy 1099 recovers after lateral stops in 0.06/0.04 s, CK1598 in never/0.02 s. These observations alone are not aesthetic rejection criteria or evidence of friction causation.

CK1598 alone entered the unchanged bank (seed 240925, 32 identical saved conditions, 300 steps/phase, eight policy cases, zero-action probe omitted). Exit **0**; **0/256 falls and non-wheel-contact conditions**. Original `bank-ck800-r2/summary.json` and traces were reused. Post-stop displacement/path length below cover **six seconds including braking**, not pure steady-state drift; entries are median / worst, metres.

| Bank stop case | CK800 displacement | CK800 path | CK1598 displacement | CK1598 path |
|---|---:|---:|---:|---:|
| Forward | 0.171/0.477 | 0.233/0.535 | 0.126/0.177 | 0.146/0.193 |
| Positive yaw | 0.262/0.532 | 0.286/0.550 | 0.240/0.484 | 0.245/0.490 |
| Negative yaw | 0.282/0.717 | 0.296/0.743 | 0.290/0.489 | 0.298/0.500 |
| Navigation stream | 0.305/0.811 | 0.319/0.814 | 0.384/0.448 | 0.394/0.490 |

CK1598 final-second XY RMS median/worst is 0.01401/0.02325 after forward, 0.04361/0.08618 after positive yaw, 0.04081/0.09356 after negative yaw, and **0.07285/0.11302 after navigation**. Corresponding yaw RMS is 0.00015/0.00104, 0.03085/0.09651, 0.03120/0.07904, 0.04538/0.09763 rad/s. Navigation heading displacement is 0.361/0.722 rad versus CK800 0.048/0.195. No CK1598 condition meets the navigation post-stop XY target. Stand median/worst XY RMS improves to 0.01222/0.02198; precision vx RMSE to 0.00410/0.02648. Bilateral lateral motion survives, with residual cross-axis motion retained. Three isolated hip substep-maximum readings reach torque limits across the two yaw-stop banks; this is not sustained saturation. All per-condition outcomes, six velocity axes and separate heading results remain in `bank_comparison.json`; survivor metrics do not hide failures.

**Decision: retain original CK800 as the reference, without claiming it meets the navigation goals.** CK1598 is the stronger nominal precision fallback but its perturbed yaw/navigation stopping does not justify integration. No candidate meets the requested goals sufficiently; no new export, navigation integration, sim2sim, further bank or additional training was started. All 143 locally present protected reference files remain byte-identical; two historical ZIP paths were already absent at task start. Raw evidence/manifests are under `evaluation/entropy_20260926_125836/`; detailed tables, checks and analysis are under `.migration-audit/entropy-20260926/`. A temporary postprocessor initially selected legacy bank `metrics.json` without contact outcomes; selecting its existing corrected `summary.json` fixed that read-only analysis (exit 1→0), without a simulator rerun or altered checks.

## Optional zero-command wheel braking — 26 September 2026

**Failed bounded controller test; leave the optional brake disabled for integration.** Starting HEAD was clean `b4a250f81a0b8e736d18ed125f095e32726186a5`. The only neural candidate was the protected v2.3 CK1598 above, SHA-256 `087418d168a95f2784ebbb31daa6638101522ade1d08f604dc90ccbd668195e3`. Its saved actuator/observation/physics contract and existing nominal/bank manifests, seeds, lengths and realized conditions were verified. The new resolved evaluation configs match its previous one-environment baseline; the only bookkeeping difference is selecting the same explicit run by name instead of absolute path. Historical reward-only v2.3/v2.4 differences remain recorded; no reward or training default changed.

`--zero_command_brake` enables the shared `go2w/zero_command_brake.py` helper in the existing action path: clip, blend only mapped wheel entries, write issued action history, then ordinary delay/scaling/limits. Complete numerical-zero commands engage in 0.20 s; other commands release in 0.10 s, at 50 Hz. Alpha resets per environment, including the settling step inside a full reset. Legs retain their clipped actor entries. `raw_actions` remains the actor proposal; new `issued_actions` and `zero_command_brake_alpha` distinguish the intervention from post-delay actions and physical targets. Training rejects the flag. Brake playback skips the automatic bare-neural export. This is an inference composite, not improved neural weights or a certified brake.

Five focused command/brake tests plus the existing extended trace test passed (exits **0/0**); Ruff and `git diff --check` passed. Exactly one native nominal process ran, adding only the two prescribed restart sequences (6-second holds, 3-second movement phases; restart-only timeout 25 s). Exit **0**, 31 nominal cases plus two restart traces, **zero falls/non-wheel contacts**. No new sustained effort saturation was observed. In all 24 cases where alpha stayed zero, original trace fields were **bit-identical** to the saved baseline. Stand XY RMS improved 0.00565→0.00063 m/s; push-stand 0.00603→0.00193. Steady +0.1 remained 0.09641 m/s.

Final-second stop measurements, pure policy → composite:

| Nominal stop | XY RMS, m/s | Yaw RMS, rad/s | Absolute heading change, rad |
|---|---:|---:|---:|
| Forward | 0.01328→0.00508 | 0.00016→0.00017 | 0.00105→0.00062 |
| Positive yaw | **0.01024→0.04107** | **0.02720→0.10091** | **0.07780→0.13545** |
| Positive lateral | 0.03154→0.01437 | 0.04932→0.03811 | 0.13818→0.06419 |
| Negative lateral | 0.00439→0.01265 | 0.00111→0.00462 | 0.07203→0.06809 |

Yaw-stop violates both 0.03 m/s and 0.05 rad/s limits. Signed heading changes from +0.07780 to **-0.13545 rad**, with final mean `[vx, vy, yaw] = [-0.02492, -0.01928, -0.09728]`; this is unwanted turning, not a successful stop. Its three-second displacement/path increases 0.02503/0.05415→0.07492/0.15749 m, including braking from the zero-command boundary. All four wheel targets are zero once engaged, yet measured wheel control torques remain `[-0.025,-0.280,-0.294,+0.568] Nm` in the final second. Zero target does not mean zero torque. Stand height changes 0.44684→0.42381 m and front/rear joint RMS 0.10476/0.08751→0.17151/0.21142 rad; pose, clearance and load traces remain explanatory measurements rather than an aesthetic rejection criterion.

Restart +0.1 reaches 0.09902 m/s (vy 0.000004 m/s, yaw 0.000126 rad/s); -0.1 reaches -0.07648 m/s (vx error 0.02352 m/s). Six-second holds after these phases have XY/yaw RMS 0.00036/0.000002 and 0.00031/0.000002. Yaw restart reaches +0.39861/-0.31995 rad/s. Its following holds have XY/yaw RMS 0.00796/0.00808 and **0.03376/0.10064**, respectively; the negative-yaw hold also fails. That hold travels 0.32005 m, displaces 0.11241 m, and changes heading by 0.44240 rad over six seconds including braking. No matched pure-policy restart baseline was rerun.

Per-case/per-phase six-axis velocities, displacement **and** traveled path, signed/absolute heading, heading-increase flags and actuator/pose measurements are retained in `.migration-audit/zero-brake-20260926/physical_metrics.csv` and `nominal_comparison.json`; raw transients and provenance are in `evaluation/zero_brake_20260926/ck1598_nominal/`. These measurements are not an instantaneous-stop model. The preset failure gate stopped work before any new bank or export. No ramp tuning, candidate switch, PPO, sim2sim or hardware work followed. All 191 protected files present at task start remain unchanged; a generic v2.4 export had already changed since the previous inventory and was preserved at its current hash.

## Step-and-recovery preparation — 26 September 2026

Prepared **one fresh candidate**, `--go2w_profile step_recovery_v1`, from clean `testing` at `df39c88b771cc32099931f15a00e3cb39e77a654`. No profile retains the registered baseline; an explicit CPU comparison against that revision found identical commands, timers and RNG states. The profile/version is saved in both ordinary configs. Training, evaluation and playback select it before construction; old-profile checkpoint contracts fail. Same-profile continuation retains strict behavior checks and native full-state verification. The unsuccessful optional wheel brake is disabled and rejected with this profile.

The candidate uses named hip/thigh/calf offsets **0.30/0.35/0.40 rad**, wheel scale **18 rad/s**, and action clipping 1.0. Every target interval retains at least **0.07224 rad** to the actual URDF hard limits, exceeding the required 0.02. The shared URDF FK passes a continuous 4 cm lift, 1 cm horizontal reposition and return path. Fixed-x lift changes agree with front +0.1453/-0.2726 and rear +0.1541/-0.2747 rad. This is unloaded fixed-base kinematics, not a learned gait or validated hardware envelope. Nominal pose, 0.45 m reset height, 0.415 m reward height, gains, physics, bounds and 56/16 interface remain unchanged. Numerical action-rate weights stay unchanged: for a fixed physical target increment, hip/thigh/calf costs become approximately 0.444/0.327/0.250 of their former values. Existing diagnostics retain physical target slew.

Only this profile replaces swing clearance with the finite URDF-cylinder envelope, including collision offset, width and full wxyz rotation. Geometry is cached once, reward state at policy rate, and summed normal loads reuse the single ground-contact query. `loaded_wheels` uses summed force >8 N; historical `foot_contacts`/`legacy_force_support` remain intact. The +0.12 clearance term uses the specified 1 cm activation, 4 cm/2 cm kernel, 0.15 m/s relative reposition factor, unchanged mobility gate and at least two loaded wheels. Unnecessary air retains -0.25 and its existing gate, now with the loaded mask. Genesis 1.4.1's entity velocity getters default to authored link origins, already transporting COM/internal-origin velocities; subtracting base translation and rotation gives body-relative link repositioning. Finite differences and carried-wheel/spin cases pass. Rewards work without PhysicsDiagnostics; evaluation traces additionally retain loaded masks and per-wheel body-relative reposition velocity.

Stand segment probability remains 0.15; 25% of sampled stand segments receive 3–6 s holds. Moving distributions and fixed-command schedules are unchanged. Expected stand time share is **22.95%**, lateral plus mixed **18.13%**. `stand_still` changes only from -0.5 to **-2.0**. JSONL diagnostics now record actual environment-time exposure; the short smoke measured **10.69% stand / 26.97% lateral+mixed**, not a stationary exposure estimate.

The prepared real settings are fresh actor/critic/normalizers/optimizer, Gaussian std **0.35** with existing log parameterization and [0.05,0.8] bounds, entropy **0.001**, seed 1, **4096×48**, **1,500 updates**, saves every **250**, headless, W&B and opt-in diagnostics, experiment `go2w_step_recovery_v1`. Other PPO settings, native augmentation and disabled mirror loss are unchanged; initial LR remains **8e-4** with native adaptive scheduling.

Validation: **47 CPU tests passed across discovery and targeted repair**, with the two GPU tests skipped during CPU checks. The initial CPU import produced no output and was stopped (exit -1); the writable-cache retry exposed a missing synthetic link-index field and a truncated unchanged integration-test tail (exit 1). Restoring the fixture/tail passed the targeted retry (exit 0). Exactly **one fresh GPU smoke** ran: `logs/go2w_step_recovery_v1_smoke/step_recovery_v1_smoke_seed1_2026-09-26_16-42-32`, **64×48×2**, exit **0**, labels 0/1 and exactly two diagnostic records. It verified fresh learning state, finite tensors/rewards/losses, live geometric reward without the physics recorder, dimensions and exact native save/reload. Final `model_1.pt` SHA-256 is `d8154f137f4b68979ae52bc7cdf399e797e8631becec0b79af67a62b009d9b48`; it is not a training parent. Ruff error checks and `git diff --check` pass. A separate read-only postcheck required import/path/Windows subprocess-handle fixes (exits 1→0); it confirmed all 917 protected artifacts unchanged. Logs, resolved differences and hash inventory remain ignored under `.migration-audit/step-recovery-20260926/`. Smoke provenance records the source patch; this report section was added afterward.

**No trained result exists and the real run has not started.** Later evaluate only explicit `model_1000.pt` and `model_1499.pt` at one environment, seed 1, 150 steps/phase with diagnostic traces and this profile. Installed RSL-RL saves these after 1,001/1,500 fresh updates respectively. All 31 cases remain; the existing two longer 6 s hold/3 s restart sequences run automatically as `hold_restart_*`, without braking. Compare physical metrics to CK1598/CK800, acknowledging changed contracts rather than ranking returns or claiming a causal ablation. Retain failures, all velocity axes, bilateral motion and precision; measure stop XY/yaw RMS, heading and displacement/path including braking. From per-limb traces, report completed loaded-to-unloaded-to-loaded intervals and durations, geometric swing peaks and horizontal repositioning, separating front/rear and left/right; partial/rare events must not hide shuffling. Inspect saturation, physical slew, actual control forces/wheel opposition and any hovering, shaking or excessive hopping. Clearer steps alone do not establish navigation readiness. At most one nominally useful candidate may later enter the existing bank; no bank, export, sim2sim or hardware run was performed here.

## CK1499 sustained-command follow-up — 26 September 2026

After the preparation above, the local run `logs/go2w_step_recovery_v1/step_recovery_v1_seed1_20260926_165356_2026-09-26_16-55-18` completed 1,500 fresh updates. This follow-up started on clean `testing` at `b273148c45ac65387526843e693e1a40521ed5a4`. Frozen `model_1499.pt` SHA-256 is `080ac621c286250fa959d269a5e6299761af25a850fbbd449155a6f13f315ae7`; saved config SHA-256 is `916691c276ba5112fe947b00e5c59b527d0390fa68596421e97906c827a501d6`. The saved profile/actuator/observation/physics contract passes. Existing CK1000/1499 nominal manifests also match their checkpoints and source. Readback mass, COM, inertia, friction and gains are identical to the CK1499 nominal reference. The new resolved environment differs from that nominal config only in the evaluation timeout, 20→40 s.

`--no_export` now skips playback's automatic export while retaining the existing brake protection and default behavior. `--load_run` selects the source; `--run_name` only names logging. Ordinary playback still has a **20 s episode timeout**: `--steps 3000` spans resets, not an uninterrupted 60 s trajectory. Fall termination remains enabled. No GUI session was launched in this task.

The explicit `sustained` evaluator reuses the restart rollout path: normal reset before each of nine sequences, then 2 s zero, 25 s movement, 6 s zero, one environment, seed 1, deterministic inference, no noise/DR/push, brake disabled. It prints the entire schedule, captures terminal state before automatic reset, and censors the sequence at its first fall/timeout; later reset samples cannot count as recovery. Source patch SHA-256 is `faaaad1517657d3d75082c0655781f9dc5c0b6f341f48c827686202c9646c71f`, retained alongside the manifest in `evaluation/step_recovery_v1_seed1_20260926_165356/sustained_ck1499/`. This documentation was added afterward. Commands are body-frame m/s, m/s, rad/s; physical velocity transformations use full wxyz quaternions.

Existing short results remain valid: CK1499 stand XY RMS **0.003357 m/s**, +0.1 actual vx **0.099506 m/s**, forward-to-precision **0.094333 m/s**; short forward/yaw-stop XY RMS **0.007164/0.008188 m/s**. Short lateral vy tracking improves over CK1000, but ±0.1 yields only +0.075734/−0.076231 m/s with yaw RMS 0.087743/0.212704 rad/s; this is not a uniformly solved command interface. No historical suite was rerun.

Read-only training review: **1,477/1,500** JSONL rows, with 23 missing labels retained explicitly in `training_summary.json`; first/last 30-label windows contain 28/30 rows. Effective std mean, legs/wheels, changes 0.33683/0.34080→0.08745/0.12544; every recorded raw std remains inside [log(0.05), log(0.8)]. Final per-joint effective std spans 0.06445–0.12866. Sample clipping legs/wheels is 0.397/0.483%→1.316/1.109%; mean saturation 0/0%→1.207/0.882%. Mean/sample physical leg-target slew is 1.155/8.557→2.462/3.252 rad/s; wheel-target slew 47.91/438.33→125.40/216.54 rad/s². Scheduler KL is 0.02046→0.01443, PPO ratio clipping 26.15→23.74%, negative-reward clipping 3.917→0.274%. All 16-vectors and bounds are retained. Available timing scalars (1,442 rows per tag) average collection/learning 2.753/0.691 s per update; missing timing entries are not reconstructed and collection overhead cannot be attributed to diagnostics from these records.

Exactly **one simulation process**, exit **0**, completed all nine 33 s sequences: **zero falls, timeouts, within-sequence resets or non-wheel ground contacts**, and finite trace fields. Moving means below use seconds 20–25 of movement. Stop RMS uses its final second; displacement/path and signed heading span all six stop seconds **including braking**, not drift after settling. Moving heading is unwrapped displacement, not heading error for the commanded yaw cases.

| Case | Late body [vx, vy, yaw], m/s, m/s, rad/s | Moving heading, deg | Stop XY/yaw RMS, m/s / rad/s | Stop displacement/path, m | Stop heading, deg |
|---|---|---:|---|---|---:|
| yaw_p040 | +0.0005, +0.0027, +0.3340 | +497.75 | 0.01282 / 0.00236 | 0.05562 / 0.07861 | -0.598 |
| yaw_n040 | +0.0022, -0.0028, -0.3347 | -506.00 | 0.01412 / 0.00319 | 0.06209 / 0.07486 | +3.069 |
| yaw_p075 | -0.0023, +0.0016, +0.7138 | +1027.87 | 0.01143 / 0.00169 | 0.05506 / 0.07091 | +0.895 |
| yaw_n075 | -0.0060, -0.0020, -0.7211 | -1031.80 | 0.01144 / 0.00481 | 0.05891 / 0.07004 | +1.389 |
| lateral_p020 | -0.0090, +0.1885, -0.0195 | -30.15 | 0.00626 / 0.01098 | 0.02422 / 0.06467 | -3.998 |
| lateral_n020 | -0.0106, -0.1893, +0.0211 | +30.46 | 0.00491 / 0.01061 | 0.02691 / 0.05391 | +1.575 |
| straight_p050 | +0.4727, +0.0006, +0.0001 | +0.32 | 0.00773 / 0.00007 | 0.02145 / 0.07334 | +0.388 |
| diagonal_p010 | +0.6692, +0.0786, -0.0355 | -48.60 | 0.00604 / 0.00098 | 0.01536 / 0.10378 | +0.555 |
| diagonal_n010 | +0.6701, -0.0786, +0.0313 | +47.43 | 0.01054 / 0.01930 | 0.04553 / 0.11287 | -0.304 |

**Confirmed:** low-clearance yaw repositioning, persistent yaw posture distortion, progressive asymmetric straight stance, and substantial unwanted heading accumulation during lateral/diagonal commands. Straight front/rear width grows 0.291/0.293→0.347/0.315 m (0–3 versus 20–25 s); tilt grows 1.68→2.61°, front/rear nominal joint RMS 0.089/0.040→0.112/0.070 rad. Late FL/FR hip angles are +0.026/−0.161 rad; pairwise front/rear mirror RMS is 0.154/0.076 rad, separately from nominal-pose error. Late +0.4 yaw front/rear error is 0.228/0.245 rad; after stopping it remains 0.180/0.119 versus initial stand 0.097/0.032. These are posture changes without a fall or pose collapse. Maximum substep control-force magnitude reaches 77.53% of its joint limit across all traces; no sample reaches 99%. Late mean leg clipping ranges 0–2.97%, including zero in straight rolling; no new authority failure justifies wider scales.

**Refuted in these trajectories:** wrong-sign diagonal **body** vy and inactive wheels. Both diagonals have 0/25 wrong-sign one-second means using sign-normalized vy <−0.01 m/s; ranges are +0.06675…+0.08534 and −0.08943…−0.06590. Their world displacement is [+15.538, −4.708] / [+15.563, +4.759] m; late world velocity is [+0.536, −0.408] / [+0.541, +0.402] m/s. Full-quaternion measurements expose real heading drift rather than dismissing the world path. The level-body illustration `vy_world = vx_body*sin(heading) + vy_body*cos(heading)` changes sign near −8.13° for [0.7,0.1]; it is not the measurement transform. At late +0.4 yaw, measured FL/FR/RL/RR wheel speeds are [−0.735,+1.431,−1.648,+1.064] rad/s, targets [−2.182,+1.839,−2.678,+3.129], and actual control-force readings [−1.447,+0.408,−1.029,+2.064] Nm. No causal percentage of yaw is assigned to wheels versus legs. Body-relative wheel toe remains numerically near zero for this joint geometry; camber follows hip abduction. Full wheel Euler/spin angles are not steering errors. Tangential ground forces were not recorded, so friction causation remains unresolved.

Swing analysis uses **analysis-only** force hysteresis (unload ≤6 N, reload >10 N; initial support >8 N), geometric lift ≥2 mm, and no dwell filter. In case-table order, completed unloading intervals / completed geometric lifts are **287/9, 272/11, 399/30, 399/27, 349/188, 343/182, 0/0, 219/72, 226/66**. Boundary-censored counts are 2,0,0,0,3,3,0,2,1. Completed peak maxima are 4.74,4.19,9.47,9.69,15.33,15.97,none,6.89,6.50 mm: **no completed event reaches 2, 3 or 4 cm** (fractions zero; undefined where no events exist). Counts do not establish useful stepping. Per-limb events, durations, peaks, body-relative displacement and p50/p90/max, including load-only unloading and boundary censoring, remain in `physical_summary.json` and `swing_events.csv`.

The saved objective gives (M,P)=(0.14545,0) at yaw 0.4, (0.65455,0.105) at yaw 0.75, (0.28571,0.28571) at |vy|=0.05, and (1,1) at |vy|≥0.1, including the diagonals; straight/stand gives (0,0). `clearance_air_counterfactual.csv` covers each requested command and 0/5/10/20/30/40 mm with one/two unloaded wheels. It separately records support, height activation, kernel and reposition factors, assuming all unloaded wheels have that height and speed ≥0.15 m/s, with two/three loaded supports. At yaw 0.4 and 40 mm, maximum positive term +0.01745 competes with −0.05568/−0.11136 air cost; at yaw 0.75 the values are +0.07855 and −0.03182/−0.06364. Multiply by dt=0.02; stationary lifted wheels earn zero positive term. Actual +0.4 yaw mean weighted clearance/air/pose/motion/leg-action-rate terms are +0.000030/−0.03109/−0.05255/−0.001426/−0.000353 **before dt**. This is a partial objective comparison, not PPO advantage or proof that stepping is impossible. Earlier lateral activation cannot improve M=1 cases.

**Next proposal, not implemented:** one **sampler-only, 300-additional-update continuation from this exact CK1499**, 4096×48, retaining all learned state, normalizers, raw std, optimizer/LR, actuator contract, rewards and gates. In `_resample_commands`, set yaw exactly zero for half the existing 7% mixed-family samples; keep the other half's full yaw range, existing vx [−0.35,1.10], vy [−0.30,0.30], both signs, and all seven family probabilities. For moving families only, use duration probabilities 70% U(0.5,1), 25% U(1.5,3), 5% U(8,15) seconds; keep stand sampling unchanged. This raises long-moving time exposure without memorizing these nine commands; expected stand time share falls from 22.95% to about 17.7%, so preservation of stopping must be checked. Currently only roughly 0.05% of all segments get mixed-family exact-zero yaw through its ±0.01 deadzone, versus proposed 3.5%, and moving holds never exceed 3 s. Prioritize this coverage/long-hold gap over a global symmetry penalty. Keep the continuation only if both diagonal late vx/vy errors are ≤0.04/0.02 m/s and zero-yaw lateral/diagonal heading change is ≤10° over 25 s, while preserving stand ≤0.02 m/s, precision vx error ≤0.02, stops ≤0.03 m/s / 0.05 rad/s, bilateral yaw/reverse/arc capability and no falls/contact or sustained saturation regressions. These are proposed development criteria, not current results. No std repair, extra reward intervention or new profile was implemented.

Validation: 15 distinct focused CPU tests passed across the repair runs. Initial `python -m unittest tests.test_inference tests.test_fixed_command -v` exited 1 on the pre-existing standalone import cycle (six fixed-command tests passed). Importing task registration first fixed it; `python -m unittest tests.test_inference -v` then passed eight tests and exposed a float32-versus-float64 exact comparison in the new fixture (exit 1). Comparing the expected command at the actual float32 dtype passed the targeted sequence test (exit 0). Ruff `--select E9,F63,F7,F82` and `git diff --check` pass. Read-only postprocessing exits 0. Detailed 0–3/5–10/15–20/20–25 s and one-second six-axis statistics, signed/absolute headings, forces, targets, pose, alignment, actual costs and timing scalars are in the ignored sustained folder. These nine nominal sequences are not independent robustness trials. **Stopping improvement survives; CK1499 is not generally navigation-ready. No bank, training, export or integration followed.**

Actual sustained command (the output is now occupied and protected; use a fresh output only for a separately requested repeat):

```powershell
Set-Location 'C:\Users\Liamb\SynologyDrive\TUM\3_Semester\dodo_alive\legged-robot_rl_genesis'
$Python = 'C:\Users\Liamb\anaconda3\envs\genesis-gpu\python.exe'
$env:NUMBA_CACHE_DIR = "$PWD/.migration-audit/diagnostics-20260925/numba-cache"
$env:GS_CACHE_FILE_PATH = "$env:TEMP/go2w-diagnostics-genesis"
$env:QD_OFFLINE_CACHE_FILE_PATH = "$env:TEMP/go2w-diagnostics-quadrants"
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
$SourceRun = "$PWD/logs/go2w_step_recovery_v1/step_recovery_v1_seed1_20260926_165356_2026-09-26_16-55-18"
& $Python -m robot_gym.scripts.evaluate --task go2w --go2w_profile step_recovery_v1 --experiment_name go2w_step_recovery_v1 --load_run $SourceRun --checkpoint 1499 --eval_mode sustained --num_envs 1 --seed 1 --headless --logger tensorboard --diagnostic_trace --output evaluation/step_recovery_v1_seed1_20260926_165356/sustained_ck1499
```

## Paired coverage and mobility continuation — 26 September 2026

Preparation starts at `4d1c4a1fc803ca9be9b341453dca0f4265c2ca05`. Both arms retain `--go2w_profile step_recovery_v1` and explicitly select `--go2w_finetune coverage` (A) or `coverage_mobility` (B). A shared sampler sets half of mixed-family yaw samples to exactly zero and uses moving durations 70% U(0.5,1), 25% U(1.5,3), 5% U(8,15) seconds. Family probabilities, ranges, deadzones and stand sampling stay unchanged. B alone sets yaw mobility start/full to 0.15/0.60 rad/s and wheel-air relaxation to 0.90; the separate pose gate is unchanged. Unselected configurations retain their original sampler and RNG sequence. Evaluation/playback require the saved selector. Strict continuation permits only these exact requested fields/values plus existing execution bookkeeping.

At yaw 0.4, M changes 0.1454545→0.4444444. For one ideally repositioning unloaded wheel at 4 cm, weighted clearance/air terms change +0.0174545/−0.0556818→+0.0533333/−0.0375; B's two-wheel air cost is −0.075. These are two instantaneous terms before dt=0.02, not PPO advantages. The untruncated proposed stand time share is about 17.7%; 20 s episode resets can truncate long segments. Opt-in diagnostics retain family exposure and add observed long-moving/mixed-zero-yaw time, not inferred completed holds.

Validation: 39 distinct CPU contract/symmetry/diagnostic tests pass. Initial suite exit 1 was a missing resampling flag in the new synthetic fixed-command fixture; its focused repair exited 0 (no experimental setting changed). A read-only preflight compares unselected commands/timers/RNG exactly to the reviewed source and checks both resolved real/smoke configs against the original CK1499 config. Ruff error checks and `git diff --check` exit 0. Exactly one 64×48×2 resume smoke per arm exits 0, each with two diagnostic rows, finite losses/output and native save/reload. Both verify original actor/critic, normalizers, raw std, optimizer and iteration exactly before updates, preserving loaded LR `0.00011390625000000002`. Smoke checkpoints are never parents. Logs and preflight evidence are ignored under `.migration-audit/coverage-20260926/`; shared experiment tag is `20260926_231000`. The two authorized 300-update jobs and final nominal/sustained comparisons follow this implementation commit; no bank or export is part of this experiment.

**Paused at the owner's request after arm A.** Only the real coverage run completed: `logs/go2w_step_recovery_v1/coverage_seed1_20260926_231000_2026-09-26_23-15-39`, from original CK1499, on clean source `a3e79dc8c4656f75cd0aaf0911ac508f47d576da`. Training and read-only completion verification both exit **0**. It completed **300 additional updates** at 4096×48, ending at saved iteration 1798; every optimizer parameter's step counter increased **60000→72000**. Final `model_1798.pt` SHA-256: `02fe2fe814b2b82e35253097b2ff8976cf699b0f42be06dd70c03f202f2eeb25`; config SHA-256: `8f88bb3ab44423e01aa3c2b7a19be9f49e82bd377862315fa47825627bb644d7`. The checkpoint is finite. There are **291 diagnostic rows**; nine missing labels remain missing and are listed in run-local `completion_check.json`, not interpreted as an incomplete run. **Arm B's real training and all new evaluations have not started.** Both smokes are already complete; no additional smoke, training, bank or export followed the pause. No physical improvement or A/B decision can yet be claimed. Continue only on a later owner request, retaining original CK1499 as B's parent and reusing the existing parent evaluations.

### Resumed results — 27 September 2026

The owner authorized continuation from the pause. B completed in `logs/go2w_step_recovery_v1/coverage_mobility_seed1_20260926_231000_2026-09-27_09-56-14`, independently from original CK1499, on clean source `bc6dcf8a2e54114be5a9e99b0d15e77ead98194c` (only the pause documentation differs from A's source). Final `model_1798.pt` SHA-256 is `6d0529edecb6309ec2a95fc8d0524f3c18eeaaba11843d276e096a835d4b41fe`; config SHA-256 is `f91fc0591cd79fec8580bab0d8ae5a31775234d524a0aa65e21e37bdbe0d3508`. Both runs completed **300 additional updates**, saved iteration 1798, optimizer counters **60000→72000**, with training and completion-check exits **0**. Both exact native loaded-state checks match the original parent, including normalizer buffers, raw std, optimizer and loaded LR **0.00011390625000000002**. Actual saved configs differ only in selectors/run names and B's three declared mobility parameters. Parent checkpoint/config hashes remain those recorded above; no parent or historical evaluation was changed.

Exactly four final-checkpoint evaluation processes exited **0**: A/B nominal (31 cases plus the two existing hold/restart sequences) and A/B sustained (nine unchanged 2+25+6 s sequences). All are one environment, seed 1, deterministic, with brake/noise/DR/push disabled and diagnostic traces. Loaded mass/COM/inertia/friction/gains match the parent readback. There were **zero falls or non-wheel contacts**, and all sustained sequences finished without timeouts, within-sequence resets or censoring. No parent replay, bank, export, extra training or GUI session was run. Offline analysis and final evidence checks exit **0**. Initial offline checks exposed intended config-difference records in a whole-parent comparison, nominal versus sequence trace schemas, and older restart metadata; these were corrected in ignored analysis scripts, with original failure logs retained. No runtime checks or experiment values were weakened. Ruff error checks and `git diff --check` exit 0; invoking Ruff through `genesis-gpu` first failed because that environment lacks the module, so the existing system Ruff executable was used without installing anything. For this documentation-only change, the 39 CPU checks and two successful smokes above were not repeated.

Physical comparison below uses final-second nominal measurements and movement seconds 20–25 for sustained velocities. Heading spans the full 25 s movement. Maxima over stop cases are component-wise maxima, not necessarily the same case. Complete six-axis statistics, all 31 cases, restart phases, one-second sustained windows, world paths, per-limb events, stop displacement/path and heading-increase flags remain under `evaluation/coverage_20260926_231000/` in `comparison.json`, the comparison CSVs and each arm's `physical_summary.json`.

| Measurement | Parent CK1499 | A: coverage | B: coverage_mobility |
|---|---:|---:|---:|
| Nominal stand XY RMS, m/s (target ≤0.02) | 0.003357 | 0.010478 | **0.020399, fail** |
| Nominal +0.1 actual vx, m/s | 0.099506 | 0.097636 | 0.109832 |
| Forward-to-precision actual vx, m/s | 0.094333 | 0.096775 | 0.109502 |
| Four nominal stops: max XY/yaw RMS, m/s / rad/s | 0.00819 / 0.00953 | 0.01098 / 0.01522 | 0.02064 / 0.00733 |
| Nine sustained stops: max XY/yaw RMS | 0.01412 / 0.01930 | 0.00952 / 0.02032 | 0.01615 / 0.01014 |
| Lateral +0.2 / −0.2 heading, deg | −30.15 / +30.46 | −17.28 / +16.45 | **−32.37 / +36.64** |
| Diagonal +vy / −vy heading, deg | −48.60 / +47.43 | −16.48 / +20.88 | −15.23 / +15.20 |
| Positive diagonal late vx / vy, m/s | 0.6692 / 0.0786 | 0.6721 / 0.0988 | **0.6536** / 0.0995 |
| Negative diagonal late vx / vy, m/s | 0.6701 / −0.0786 | 0.6717 / −0.0965 | **0.6532** / −0.0986 |
| Straight +0.5 heading, deg | +0.32 | **−9.62** | **−11.53** |
| Nominal −0.25 actual vx, m/s | −0.2333 | −0.2172 | −0.2047 |
| Restart −0.1 actual vx, m/s | −0.1005 | −0.0901 | −0.0776 |

**Functional tracking and stopping:** A reduces lateral heading magnitude by 43–46% and diagonal heading by 56–66%, but **none of the four cases meets 10°**. Both A diagonals meet late vx/vy error goals 0.04/0.02 m/s; B misses vx on both (errors 0.04643/0.04681). All three policies have zero wrong-sign diagonal body-vy one-second means at the declared −0.01 m/s tolerance. Heading still bends the world path: late positive-diagonal world vx/vy is [0.536,−0.408]→[0.676,−0.076]/[0.659,−0.056] m/s for parent→A/B. Body-vy improvement does not erase heading error. B improves short lateral tracking and +0.4 yaw rate, but worsens sustained lateral heading even against the parent and narrowly fails stand. Both lose reverse accuracy; B's −0.1 restart error is 0.02243 m/s. All nominal/sustained/restart post-stop RMS values remain below 0.03 m/s / 0.05 rad/s, but transients still matter: after sustained straight motion, six-second displacement/path is 0.02145/0.07334→0.04880/0.07775 (A), 0.11977/0.12146 m (B), including braking. Signed/absolute stop heading is retained separately. Fast rolling, bilateral yaw/lateral and arcs remain available; A's [1,0,1] arc yaw falls 0.984→0.932 rad/s, while B gives 0.967. These regressions remain visible rather than hidden behind the headline improvement.

**Mobility quality:** no completed sustained event reaches **2, 3 or 4 cm** in parent, A or B. The existing analysis-only 6/10 N hysteresis, 2 mm geometric threshold, no dwell filter and boundary censoring are unchanged. In nine-case order, A completed geometric-lift counts are 16,13,34,44,189,186,0,102,90; B gives 17,18,40,27,190,192,0,102,104. More transitions are not a successful gait. B does show a distribution shift in rear lateral lifts across both signs: RL/RR peak p50/p90 is **10.18/15.57 and 11.04/14.99 mm**, versus A **7.40/10.07 and 7.38/10.60 mm**; median horizontal reposition is about 58 versus 55 mm. Front lateral peaks remain about 4 mm, and yaw repositioning remains low. Per-case/per-limb durations, p50/p90/max height and displacement, load-only transitions and censored events are retained in `swing_comparison.json` and `swing_events.csv`. This partial rear improvement does not qualify B when control regresses.

**Posture and preservation:** B consistently reduces late yaw nominal-pose error; at +0.4, front/rear RMS is parent 0.228/0.245, A 0.215/0.229, B **0.175/0.205 rad**, with similar reductions at the other three yaw commands. Straight late front/rear width is parent 0.347/0.315, A 0.307/0.315, B 0.293/0.328 m; front mirror error improves, but rear error grows 0.076→0.089/0.114 rad. Tilt is 2.61→3.34/2.21°; maximum absolute mean wheel camber is 0.161→0.094/0.118 rad. A better front stance does not mean the entire posture is fixed. Actual control-force readings reach at most 77.53/82.74/78.02% of joint limits across sustained parent/A/B traces; no sample reaches 99%. Whole-sequence raw leg saturation maxima are 2.01/0.49/1.37%. There is no new sustained force saturation pathology. Same-side front/rear wheel opposition remains recorded, not inferred from target signs: over the straight sequence its left/right mean opposing magnitude is 0.578/0.544→0.262/0.226 (A), 0.087/0.045 Nm (B). This joint-force measure is not net ground force or causal wheel contribution to yaw. Detailed physical target slew is retained; no authority change is justified by these results.

Observed training time share is stand **18.42/18.38%** for A/B, lateral+mixed **19.17/19.16%**, long-moving **23.16/23.22%**; exact-zero yaw occupies **50.06/49.76% of mixed time**. These are available environment-step exposures including truncated segments, not segment probabilities or proof every 8–15 s hold completed. A has 291 rows (missing 1533,1561,1562,1608,1609,1656,1657,1663,1747); B has 299 (missing 1543). None were reconstructed. First/last 30-label windows each contain 30 rows. Effective leg/wheel std is A 0.08627/0.12357→0.07938/0.11520, B 0.08630/0.12377→0.07933/0.11728; all recorded raw parameters remain inside the unchanged bounds. Sampled leg/wheel clipping is A 1.289/1.122→1.198/1.101%, B 1.246/1.071→1.249/1.064%; mean saturation A 1.176/0.891→1.084/0.903%, B 1.133/0.850→1.142/0.854%. Scheduler KL stays about 0.014; PPO ratio clipping A 23.48→23.32%, B 23.71→23.49%; negative-reward clipping A 0.220→0.215%, B 0.210→0.208%. Mean/sample leg-target slew falls A 2.436/3.209→2.335/3.018, B 2.437/3.212→2.310/2.986 rad/s. Wheel-target slew falls A 125.14/214.76→117.79/200.21, B 125.07/214.68→117.66/202.66 rad/s². All 16-vectors and genuine timing gaps remain in `training_summary.json`; different collection times are not attributed to a component without evidence.

**Decision:** preserve **A as a functionally improved intermediate**, with CK1499 retained as the reference. Reject **B for promotion** despite its real posture/rear-clearance benefits: its stand, lateral heading, diagonal vx and reverse preservation are worse. Neither meets the full development criteria or is generally navigation-ready. A versus parent includes 300 extra updates and does not isolate sampler causality; A/B is one seed pair, not proof of general superiority. **One recommended next action is owner GUI review of the frozen A/B results with `--no_export`, especially the straight-heading and reverse regressions.** No automatic bank, export, integration or further training follows. The two exact run paths above require profile `step_recovery_v1`, their respective finetune selector and checkpoint 1798; playback retains its 20 s timeout. The completion record, exact executed train/evaluation commands and all large evidence remain ignored under `evaluation/coverage_20260926_231000/` and `.migration-audit/coverage-20260926/`.

## Coverage A closed-loop probe — 27 September 2026

Started from clean `testing` at `e0aee711cd2de7249b07c05d0f5036be2075e786`. Frozen A remains `coverage_seed1_20260926_231000_2026-09-26_23-15-39/model_1798.pt`, profile `step_recovery_v1`, finetune `coverage`: checkpoint SHA-256 `02fe2fe814b2b82e35253097b2ff8976cf699b0f42be06dd70c03f202f2eeb25`, config SHA-256 `8f88bb3ab44423e01aa3c2b7a19be9f49e82bd377862315fa47825627bb644d7`. Existing contracts/manifests, nominal physics/batching and loaded-property readback match. Relevant original files and the local coverage archive were preserved. Source patch SHA-256 during execution was `98682e6c662159a793696688668185d8e0f2ec39834ad3da06b2b4b0da63f962`, saved under `.migration-audit/closed-loop-20260927/`; this result section was added afterward.

The new opt-in `closed_loop` mode leaves legacy evaluator behavior unchanged. Only this mode avoids implicit profile-triggered PhysicsDiagnostics unless `--diagnostic_trace` is requested. It retains profile rewards/state computation, physical failure checks and terminal pre-reset capture. Lightweight traces retain policy-rate control-force readings and cached contact loads, **not substep maxima or geometric swing measurements**. Raw actor output, clipped issued action and post-delay action remain distinct; no action mapping or previous-action observation changed.

Exactly **one simulator process**, exit **0**, ran straight [0.5,0], left [0.5,0.1], right [0.5,−0.1] m/s, each for **20 s / 1,000 policy steps**, with normal resets between sequences. Seed 1, one nominal environment, no noise/DR/push/brake, 30 s episode timeout. The fixed schedule is 2 s exact zero, 12 s trapezoidal path (1 s ramps, integrated multiplier 11 s), 3 s endpoint feedback, 3 s exact zero bypassing feedback. Position/heading are anchored once after settling. Gains are **1.0/1.5 s⁻¹**; feedback updates every fifth 50 Hz policy step (**10 Hz**, 150 updates), with four 5 ms physics steps. World requests are transformed with the full inverse wxyz quaternion. Command bounds are vx [−0.25,0.70], vy [−0.20,0.20] m/s, yaw [−0.50,0.50] rad/s. Nothing was retuned.

All three traces are finite and uninterrupted, with **zero falls, timeouts, within-sequence resets or non-wheel contacts**. The following errors cover **seconds 2–20**, including ramp, endpoint and feedback-off transients; settling has no anchored reference. Cross-track error is perpendicular to the fixed world path. Endpoint error is measured at second 17, immediately before feedback removal.

| Case | Position RMS / p95 / max, mm | Cross-track RMS / p95 / max, mm | Heading RMS / max, deg | Endpoint error, mm |
|---|---:|---:|---:|---:|
| Straight | 30.30 / 45.93 / 53.12 | 0.23 / 0.43 / 0.46 | 0.026 / 0.070 | 7.27 |
| Diagonal left | 27.69 / 39.22 / 48.95 | 11.15 / 23.80 / 25.67 | 0.868 / 2.283 | 14.15 |
| Diagonal right | 29.04 / 44.22 / 48.93 | 12.14 / 29.39 / 31.13 | 1.196 / 3.358 | 14.70 |

During feedback alone (2–17 s), position RMS is 32.26/27.07/27.46 mm and maximum heading error 0.070/1.171/1.123° in table order. **No command clips or touches a bound.** Peak position corrections are 0.0530/0.0484/0.0484 m/s; peak yaw requests 0.00183/0.03064/0.02940 rad/s, at most 6.2% of the yaw bound. Feedback therefore gives bounded tracking here without hiding error through command saturation. Body vx/vy/yaw tracking biases against issued commands are [−0.02650,−0.00021,+0.00059], [−0.02211,−0.00333,−0.01462], [−0.02322,+0.00363,+0.01447]; corresponding RMSE is [0.03308,0.00026,0.00076], [0.02989,0.01500,0.04388], [0.03048,0.01430,0.04270], in m/s, m/s, rad/s. Six-axis velocities, signed reference errors derivable from position/quaternion, requests and corrections remain in the traces; one-second windows remain in `metrics.json`.

Small-command tracking is still weak. In the last two endpoint-feedback seconds, diagonal yaw requests average **+0.02940/−0.02832 rad/s**, but actual body yaw averages only **+0.00061/−0.00074**. Lateral requests +0.01249/−0.01171 m/s coexist with actual vy −0.00177/+0.00245. The endpoint offsets do not vanish. These late holds show no command/velocity sign reversals above the analysis-only 0.01 threshold, with heading ranges only 0.065/0.074°; this is persistent residual error rather than a growing hold oscillation. Moving diagonals have yaw-rate variability around 0.04 rad/s, retained rather than hidden by the small mean. After feedback removal the yaw transient reverses: first/last zero-second mean yaw is −0.01785→+0.01582 (left), +0.03180→−0.01525 rad/s (right).

| Exact-zero phase | Last-second XY / yaw RMS, m/s / rad/s | Three-second displacement / path, mm | Signed heading change, deg |
|---|---:|---:|---:|
| Straight | 0.00536 / 0.00019 | 18.03 / 18.03 | +0.030 |
| Diagonal left | 0.00910 / 0.01589 | 29.40 / 30.59 | +0.645 |
| Diagonal right | 0.00769 / 0.01526 | 33.48 / 34.39 | +0.957 |

Displacement/path include braking from the exact-zero boundary. These **three-second** windows do not replace previous six-second stopping evidence or establish an instantaneous stop. Previous open-loop failures and clearance measurements remain unchanged; this probe supplies different feedback evidence, not a new readiness score. It does not test obstacle avoidance, learned navigation, slopes, healthcare safety, sim2sim or hardware, and three nominal sequences support no robustness claim.

Four distinct focused CPU checks passed (exit **0**); two changed checks were rerun after adding CLI/late-window assertions, also exit **0**. They cover reference integration, transforms/wrapped signs, sample-and-hold, fixed zeros, rate assertions, lightweight selection and fall/timeout/reset/non-finite censoring; existing transform/reset tests were reused. Ruff error checks, `git diff --check` and saved-output validation exit **0**. No simulator failure or rerun occurred. Monotonic timing: startup **43.417 s**, rollout including normal resets **124.522 s**, postprocessing **0.076 s**, shutdown **0.055 s**, internal total **168.089 s** (final JSON write excluded); external process stopwatch **169.177 s**. Output bytes: straight NPZ **415,713**, left **433,953**, right **433,875**, summary **205,245**, manifest **22,434**, one loaded-property snapshot **7,095**; total **1,518,315 bytes**. No per-step CPU dictionary conversion, images or heavy recorder were used. Large evidence remains local under `evaluation/coverage_20260926_231000/closed_loop_A_20260927/`.

**One recommendation:** investigate the observed small-yaw rate bias with a separately authorized, isolated continuation from **A** using a dedicated Go2-W yaw squared-error denominator **0.04 instead of 0.25**, retaining weight **0.8**, all x/y kernels, A's sampler, actuator contract, learned state and PPO settings. This hypothesis is not implemented or launched. The current feedback probe demonstrates bounded, modest-correction tracking, while exposing the low-level rate-interface weakness; no from-scratch run, bank, export or navigation integration follows automatically.

Exact executed command (the output is occupied; do not rerun into it):

```powershell
Set-Location 'C:\Users\Liamb\SynologyDrive\TUM\3_Semester\dodo_alive\legged-robot_rl_genesis'
$Python = 'C:\Users\Liamb\anaconda3\envs\genesis-gpu\python.exe'
$env:NUMBA_CACHE_DIR = "$PWD/.migration-audit/diagnostics-20260925/numba-cache"
$env:GS_CACHE_FILE_PATH = "$env:TEMP/go2w-diagnostics-genesis"
$env:QD_OFFLINE_CACHE_FILE_PATH = "$env:TEMP/go2w-diagnostics-quadrants"
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
$Run = "$PWD/logs/go2w_step_recovery_v1/coverage_seed1_20260926_231000_2026-09-26_23-15-39"
& $Python -m robot_gym.scripts.evaluate --task go2w --go2w_profile step_recovery_v1 --go2w_finetune coverage --experiment_name go2w_step_recovery_v1 --load_run $Run --checkpoint 1798 --eval_mode closed_loop --num_envs 1 --seed 1 --headless --logger tensorboard --output evaluation/coverage_20260926_231000/closed_loop_A_20260927
```

## Precision/clearance refinement — 27 September 2026

This combined candidate starts independently from frozen coverage A CK1798 (`02fe2fe814b2b82e35253097b2ff8976cf699b0f42be06dd70c03f202f2eeb25`), adjacent config `8f88bb3ab44423e01aa3c2b7a19be9f49e82bd377862315fa47825627bb644d7`, on `testing` at `a2caf342e137e7cfde8b6fe2c0089397893b8ecf`. The explicit `step_recovery_v1` / `precision_clearance` selector retains A's sampler, gates, action contract and PPO. The exact seven experimental field changes are dedicated yaw squared-error denominator **0.04** (legacy fallback 0.25), clearance sigma **0.025 m**, activation height **0.040 m**, clearance weight **0.40**, Kp/Kd factor ranges **[0.85,1.15]**, and actuator delay **[0,2] policy ticks**. Target remains 0.04 m, yaw weight 0.8; weights and dt are each applied once. No B mobility changes, std repair, brake or export. Strict continuation accepts only the declared coverage-parent transition; evaluation/playback requires the saved selector.

Nine focused CPU checks passed (exit 0). An added summary-path assertion exposed an integer-axis dtype mismatch; the conversion was fixed and both changed checks passed (exit 0). Ruff error checks and `git diff --check` passed. The native smoke first exited 1 **before any update** because auxiliary-scene destruction left the repository's Genesis initialization flag set. Test-only cleanup was repaired; the same smoke then exited **0**, completing exactly two 64×48 updates from original A, two diagnostic records, finite losses and native save/reload. Exact actor/critic/normalizer/raw-std/optimizer/LR/iteration equality passed before updating; loaded LR was **0.00017085937500000003**. Both failed and successful logs remain under `.migration-audit/precision-20260927/`. No behavioral values were tuned.

The smoke's auxiliary scene was destroyed before rebuilding/reseeding training. Readback verified coherent per-environment gain factors, all delays {0,1,2}, three-slot history and actual delayed actions, preserved friction/mass/COM/inertia on selective reset, and unchanged non-reset gains/delays. Against the protected nominal snapshot, added mass spans −0.44825…+1.47309 kg and maximum absolute COM change 0.014947 m. Geometry friction is wheel 1.0 / ground 0.1; Genesis 1.4.1 combines contacts by `max(wheel*ratio, ground*ratio, 0.01)`, giving the declared effective range [0.6,1.2], sampled here as 0.60090…1.17650. This is actuator delay, not sensor delay; more DR is not a transfer guarantee.

The ignored `objective_table.csv` records 112 counterfactual rows (both objectives, four commands, one/two unloaded wheels, seven heights), saturated body-relative repositioning and at least two supports. At yaw 0.4, one ideal 4 cm wheel gives +0.0581818 clearance / −0.0556818 air before multiplying by 0.02 s. Averaging over unloaded wheels leaves clearance unchanged for two identical swings, while air cost doubles. This is a partial instantaneous objective, not PPO advantage. New lightweight `precision_screen` and `precision_dr` modes preserve terminal pre-reset rows, censor each environment independently and retain policy-rate forces and cached geometry without PhysicsDiagnostics. Fixed schedules and thresholds were implemented before learning; the existing closed-loop mode is unchanged.

### Completed comparison and decision

The successful run is `logs/go2w_step_recovery_v1/precision_clearance_seed1_20260927_132600_2026-09-27_13-26-35`, trained on clean source **`4b1e2fa41a31cf1b277bcbcd3de09409a1291458`**. Final **`model_2397.pt`** SHA-256 is **`be057d1dfb1fea992c5f783de395dfd3c693a3f2013f1e0fe3695c94396c659f`**; config SHA-256 is `0c38d874896555eb9bfbcb3b9bf7718c8a0612a19cfdaac68caed0395eacdd2c`. Native exact loaded-state verification matched original A, including raw std, both normalizers, optimizer counters/group LR and algorithm LR **0.00017085937500000003**. Training exited **0**, completed **600 additional 4096×48 updates**, and all Adam counters increased **72000→96000**. The finite checkpoint's saved iteration is 2397. A first startup (`precision_clearance_seed1_20260927_132100_2026-09-27_13-22-34`) exited **1** at W&B initialization with blocked sandbox networking and **zero updates**. The owner explicitly authorized one identical network-enabled retry; it was the only trained candidate. External successful-process time was **2350.750 s**. No std, optimizer, schedule or experimental values were repaired or retuned.

All **five evaluation processes exited 0**: A/candidate matched screens, candidate unchanged closed loop, and A/candidate eight-condition dynamics checks. Every sequence completed; **no falls, non-wheel contacts, resets, timeouts or non-finite samples** occurred. Screens use one environment, seed 1, five fixed schedules totaling 79 s per policy; dynamics checks use eight environments and the same 26 s sequence. Both have nominal noise/push/brake off, 30 s timeout and policy-rate capture. The eight applied-property snapshots match **exactly**, including common three-slot history and explicit 0/1/2-tick delay conditions; A did not train on 40 ms delay. Existing A closed-loop results were reused. No full nominal/sustained replay, general bank, export or GUI session was run.

The new CPU summary test initially exposed an integer-axis dtype error; its repair passed. Two further analysis-only fixes keep summary tensors on CPU despite Genesis' default device and avoid mutating a NumPy quaternion view while conjugating it. Focused default-device, rotated-frame and candidate-sampler/RNG checks exit **0**. A's already-saved raw screen NPZ was unaffected; its wheel-alignment summary alone was recomputed offline, with every other metric verified unchanged and the original summary retained. No simulator replay was needed. Candidate screen and later evaluations ran on clean **`398b84ffc5722b170969a0dcdeeba3275d363f8e`**; these analysis fixes do not change training. Relevant protected hashes, finite saved arrays and first-failure validity checks pass; Ruff error checks and `git diff --check` exit **0**. All failures and command/exit records remain in `.migration-audit/precision-20260927/`.

| Matched physical measurement | Frozen A | Precision/clearance |
|---|---:|---:|
| Signed small-yaw phases (0.03/0.1), whole-phase yaw RMSE, rad/s | 0.02279 | **0.01243 (−45.5%)** |
| Signed moderate-yaw phases (0.4/0.75), whole-phase yaw RMSE, rad/s | 0.06008 | **0.03193 (−46.8%)** |
| Same groups, final-second yaw RMSE, rad/s | 0.01690 / 0.05094 | **0.01276 / 0.02673** |
| Initial stand XY RMS, m/s | 0.01052 | 0.01384 |
| +0.1 actual vx / absolute error, m/s | 0.09787 / 0.00213 | **0.08496 / 0.01504** |
| +0.5 / −0.25 actual vx, m/s | 0.46357 / −0.22078 | 0.52439 / −0.23073 |
| Five nominal stops: maximum last-second XY / yaw RMS | 0.01154 / 0.01711 | **0.01782 / 0.01790** |
| Closed-loop position RMS: straight / left / right, mm (feedback on) | 32.26 / 27.07 / 27.46 | **25.26** / 27.57 / 26.47 |
| Closed-loop max heading: straight / left / right, deg (feedback on) | 0.070 / 1.171 / 1.123 | 0.077 / **0.640 / 0.607** |
| Endpoint error before feedback removal, mm | 7.27 / 14.15 / 14.70 | **17.13 / 24.65 / 26.40** |
| Completed geometric events / events reaching 2 cm, matched screens | 103 / **0** | 73 / **0** |
| Eight-condition final stop: median / max XY RMS, m/s | 0.00547 / 0.00932 | **0.01126 / 0.01723** |
| Eight-condition final stop: median / max displacement, mm | 24.77 / 43.60 | **50.76 / 67.82** |
| Eight-condition final stop: median / max path length, mm | 36.92 / 55.36 | **73.58 / 85.89** |

**Yaw precision improves, preservation is mixed.** At ±0.4, last-second yaw changes +0.3688/−0.3626→+0.3978/−0.3991 rad/s; at ±0.75, +0.7083/−0.7102→+0.7466/−0.7435. The ±0.03 phases improve but still undertrack (+0.0129/−0.0129→+0.0176/−0.0180); ±0.1 now slightly overshoots. These are history-dependent responses, not a global deadzone measurement. Bilateral lateral motion and reverse remain available. Every nominal stand/precision/zero-speed preservation limit passes, but +0.1 tracking and stopping clearly worsen, not merely by a tiny threshold crossing. In screen order (rolling/reverse, yaw+, yaw−, lateral+, lateral−), final-stop XY RMS A→candidate is **0.00855→0.01782, 0.00362→0.01462, 0.00339→0.01306, 0.00503→0.00605, 0.01154→0.01550 m/s**. Corresponding yaw RMS is 0.00933→0.01350, 0.00719→0.01790, 0.00602→0.01244, 0.01711→0.00537, 0.01613→0.00501 rad/s. After rolling/reverse, three-second displacement/path increases **8.96/36.82→63.36/63.40 mm**; after yaw+, **10.91/15.28→28.43/40.92 mm**, with heading +0.97→+3.36°. All per-phase six-axis means/RMSE, transients, signed heading and displacement/path remain in `phases.csv` and per-process `metrics.json`; displacement includes braking.

Closed-loop gains **1.0/1.5 s⁻¹**, 10 Hz feedback, 50 Hz policy, reference and command bounds were unchanged. No commands clip. Peak translational corrections are A 0.0530/0.0484/0.0484 versus candidate **0.0591/0.0463/0.0457 m/s**; diagonal peak yaw requests fall 0.0306/0.0294→0.0168/0.0157 rad/s. All feedback-on position RMS values remain ≤0.04 m and heading maxima ≤2°. Nevertheless, straight exact-zero XY RMS rises **0.00536→0.02336 m/s**, three-second displacement/path **18.03/18.03→67.89/67.89 mm**, and heading +0.03→+1.34°. Left/right zero-phase XY RMS is 0.00910/0.00769→0.00995/0.01143; heading changes +0.65/+0.96→+3.12/−3.31°. Candidate hold drift leaves larger endpoint errors. Nonzero feedback commands can compensate policy bias; a near-zero achieved hold rate is not by itself proof of an ignored command. These three-second windows do not replace older six-second evidence.

**Clearance did not improve generally.** The unchanged 6/10 N hysteresis, ≥2 mm geometric threshold, no dwell filter and boundary censoring yield no completed 2/3/4 cm events for either policy. Positive-lateral peak p90 in FL/FR/RL/RR order changes **5.10/4.04/8.67/11.62→4.52/7.93/7.82/5.99 mm**; negative-lateral values **4.13/5.13/8.46/8.07→7.80/2.94/4.60/8.05 mm**. This is a redistribution, not a consistent bilateral gain. Counts, rates, duration/repositioning distributions, load-only and censored events are retained per limb in `swings.csv` and raw summaries. Target clipping worsens: candidate front-hip saturation reaches 24% at |vy|=0.1 and 29–30% at 0.2, versus A's maximum 8% at 0.2; rear calf reaches about 16% versus 2%. Longest lateral clipping interval grows 0.08→0.22 s. At |yaw|=0.75, front/rear pose RMS grows about 0.20/0.20→0.245/0.229 rad and roll RMS roughly doubles to 0.08 rad; wheel camber also increases. This is periodic distortion and clipping, not a permanently pinned target or measured force saturation. Maximum policy-rate force/limit ratio rises **0.676→0.777** in the screens and **0.798→0.927** across dynamics conditions; neither records a sample at 99%. Physical slew, actual opposing wheel forces and spin-invariant axle alignment remain available, without claims about substep maxima or causal yaw contribution.

The fixed dynamics comparison preserves every condition, with no survivor filtering. Whole-phase pooled ±0.4 yaw RMSE improves (positive **0.06570→0.03278**, negative **0.05729→0.04782 rad/s**), while median stop yaw RMS grows **0.00292→0.00889 rad/s**. Maximum stop yaw RMS is 0.02331→0.02271; maximum absolute stop heading is 2.65→2.32°, so not every heading measure regresses. Candidate stop translation/path is larger in all eight conditions, including both delays. Exact per-condition all-axis results and applied properties are in `dr_comparison.json`, `dr_phases.csv` and both `*_dr` folders. This tiny common check is not a robustness probability or full sensor-uncertainty validation.

Training has **596 real diagnostic rows**; labels **1838, 1988, 2030, 2134** remain missing. First/last 30-label windows both contain 30 rows. Effective leg/wheel std falls **0.07826/0.11468→0.06411/0.10490**. FL/FR calf and RL/RR thigh raw log-std end below the lower bound ln(0.05), with effective std 0.05; they were preserved, not projected. All 16-vectors remain in run-local `training_summary.json`. Sampled leg/wheel clipping is **1.378/1.154→1.622/0.858%**; deterministic mean saturation **1.266/0.948→1.513/0.708%**. Mean/sample physical leg slew falls 2.311/2.969→2.084/2.582 rad/s and wheel slew 116.45/197.11→107.74/181.40 rad/s². Scheduler KL is 0.01475→0.01407; PPO ratio clipping 24.54→24.19%; negative reward clipping 0.488→0.480%. Observed available-row stand/lateral/mixed time share is **18.31/12.45/6.72%**, including episode truncation. Return across changed objectives is not used to rank controllers.

| Process | Startup / rollout / postprocessing / total wall seconds | Output bytes |
|---|---:|---:|
| A screen | 34.504 / 163.241 / 0.115 / 197.961 | 2,784,745 |
| Candidate screen | 36.044 / 165.844 / 0.098 / 202.094 | 2,797,344 |
| Candidate closed loop | 35.237 / 126.183 / 0.071 / 161.569 | 1,528,031 |
| A eight-condition check | 44.168 / 66.224 / 0.286 / 110.828 | 7,221,645 |
| Candidate eight-condition check | 47.047 / 75.550 / 0.277 / 122.986 | 7,239,852 |

Totals include shutdown, exclude final JSON write; A screen's later offline alignment repair is recorded separately. `evaluation/precision_20260927_132600/completion.json` records actual file sizes, source manifests and unchanged protected inputs. Large data and the full old coverage ZIP remain local.

**Decision: retain frozen coverage A as the recommended transfer candidate.** Preserve CK2397 as a measured yaw-precision improvement, but do not promote it: clearer stepping did not emerge, while low-speed/stop/endpoint behavior, clipping and yaw posture worsened. This combined refinement cannot isolate causality. Passing individual development limits does not erase these continuous regressions; neither policy is hardware- or PhysX-qualified. The next action is a separately instructed Linux IsaacLab transfer task using **A**, with its existing limitations visible. No additional training, checkpoint search, export or transfer was performed here.

Exact commands below are archival (outputs are occupied). The common cache setup is the same as the preceding closed-loop section. The successful retry was explicitly network-enabled; the failed first startup used the identical train command with run name `precision_clearance_seed1_20260927_132100`. Focused CPU commands and individual exits are retained in `.migration-audit/precision-20260927/executed_commands.txt` and the adjacent logs.

```powershell
$Python = 'C:/Users/Liamb/anaconda3/envs/genesis-gpu/python.exe'
$A = "$PWD/logs/go2w_step_recovery_v1/coverage_seed1_20260926_231000_2026-09-26_23-15-39"
$Candidate = "$PWD/logs/go2w_step_recovery_v1/precision_clearance_seed1_20260927_132600_2026-09-27_13-26-35"
$Common = @('--task', 'go2w', '--go2w_profile', 'step_recovery_v1', '--experiment_name', 'go2w_step_recovery_v1', '--seed', '1')
$env:GO2W_RESUME_SMOKE_FINETUNE = 'precision_clearance'
$env:GO2W_COMPARISON_TAG = '20260927_precision'
& $Python -m unittest tests.test_go2w_symmetry_integration.ContinuationIntegrationTests.test_original_checkpoint_two_additional_updates -v
& $Python -m robot_gym.scripts.train @Common --go2w_finetune precision_clearance --load_run $A --checkpoint 1798 --resume --run_name precision_clearance_seed1_20260927_132600 --num_envs 4096 --max_iterations 600 --headless --logger wandb --training_diagnostics
& $Python -m robot_gym.scripts.evaluate @Common --go2w_finetune coverage --load_run $A --checkpoint 1798 --eval_mode precision_screen --num_envs 1 --headless --logger tensorboard --output evaluation/precision_20260927_132600/A_screen
& $Python -m robot_gym.scripts.evaluate @Common --go2w_finetune precision_clearance --load_run $Candidate --checkpoint 2397 --eval_mode precision_screen --num_envs 1 --headless --logger tensorboard --output evaluation/precision_20260927_132600/candidate_screen
& $Python -m robot_gym.scripts.evaluate @Common --go2w_finetune precision_clearance --load_run $Candidate --checkpoint 2397 --eval_mode closed_loop --num_envs 1 --headless --logger tensorboard --output evaluation/precision_20260927_132600/candidate_closed_loop
& $Python -m robot_gym.scripts.evaluate @Common --go2w_finetune coverage --load_run $A --checkpoint 1798 --eval_mode precision_dr --num_envs 8 --headless --logger tensorboard --output evaluation/precision_20260927_132600/A_dr
& $Python -m robot_gym.scripts.evaluate @Common --go2w_finetune precision_clearance --load_run $Candidate --checkpoint 2397 --eval_mode precision_dr --num_envs 8 --headless --logger tensorboard --output evaluation/precision_20260927_132600/candidate_dr
```

Optional owner GUI comparison (not executed; playback retains its 20 s episode timeout). `--load_run` selects the saved policy; `--run_name` does not. Both commands preserve the explicit saved profile/finetune and prohibit automatic export:

```powershell
& $Python -m robot_gym.scripts.play @Common --go2w_finetune coverage --load_run $A --checkpoint 1798 --num_envs 1 --steps 1000 --command_vx 0 --command_vy 0 --command_yaw 0.4 --no_export
& $Python -m robot_gym.scripts.play @Common --go2w_finetune precision_clearance --load_run $Candidate --checkpoint 2397 --num_envs 1 --steps 1000 --command_vx 0 --command_vy 0 --command_yaw 0.4 --no_export
```

## Fresh event-step preparation — 27 September 2026

`event_step_v1` is an opt-in movement-objective redesign from reviewed `testing` HEAD `2479ffd36d9fe04a40e2e1b3ba76cbece973482d`. It replaces the planned transfer work. **Implementation and the fresh learning smoke pass; loaded 5 cm lift feasibility remains unresolved. No long training was launched.** The single prescribed physical check collected valid evidence but its support/controller trajectories did not lift as intended. This is not evidence of insufficient motor power. No gains, limits, height thresholds or experimental settings were retuned. Coverage A remains the comparison/fallback and was never loaded into the new experiment. Its checkpoint/config hashes still match `02fe2fe814b2b82e35253097b2ff8976cf699b0f42be06dd70c03f202f2eeb25` / `8f88bb3ab44423e01aa3c2b7a19be9f49e82bd377862315fa47825627bb644d7`; URDF SHA-256 is `794ad4adaec16bc7a1eebab3d838e1955f37113192689361aff4733e99c9e9c5`.

The explicit profile preserves the plane/URDF, 0.415 m height objective, nominal angles, 50/200 Hz timing, 56 observations, 16 actor-selected actions, offset scales 0.30/0.35/0.40 rad and wheel scale 18 rad/s, normalized clip 1, wheel target limit 20 rad/s, Kp/Kd 40/1, wheel Kv 1 and actual URDF limits. Actor/critic remain [512,256,128] ELU with trained observation normalization and sagittal augmentation, without mirror loss. Cached cylinder geometry and summed ground loads work without PhysicsDiagnostics. Old profile values and sampler RNG paths are preserved, including coverage and precision/clearance. Historical checkpoints are rejected by the new selector and native loading guard.

`G=max(clamp((abs(vy)-.01)/.04,0,1), clamp((abs(yaw)-.10)/.15,0,1))`, independent of vx. The old instantaneous `foot_swing_clearance` and averaged `default_pose` are disabled. Weighted **rate** terms are hip pose `-2*(1-.5G)*mean(error²)`, thigh/calf pose `-.6*(1-.8G)*mean(error²)`, leg motion `-.02*(1-G)*mean(velocity²)`, wheel air `-.25*(1-G)*mean(unloaded)`, prolonged unloading `-.20*mean(clamp((time-.60)/.20,0,1)²)`, and insufficient support `-relu(2-count(load>6))²`. Vertical velocity becomes -1.0 and roll/pitch rate -0.25; orientation -1.2, height -8 and stand -2 remain. Linear tracking remains `exp(-ex²/.25-ey²/.04)`; yaw becomes `.8*(.25*exp(-e²/.25)+.75*exp(-e²/.04))`, ungated. Action rate, acceleration, effort, limits, crossover, collision and termination are unchanged. Generic slide/airtime/survival/wheel-speed bonuses stay off. The new discrete reward alone skips dt; legacy positive clipping and termination scaling remain unchanged.

One vectorized tracker updates after state/contacts/termination and before rewards and command sampling. Summed loads use <=6 / >=10 N hysteresis, .04/.06 s dwell and .12 s confirmed prior support. Initial unknown support must first reach 10 N. Onset pose, command generation and body-relative wheel centers are saved at the first unloading sample; touchdown position/duration use the first reload sample. Valid swings last .10–.60 s, reach at least .008 m actual **and** usable peak, move >=.010 m net, retain >=2 wheels with summed loads >6 N through the sampled swing, and have no failure, timeout, nonfinite state or non-wheel contact. These are modest sampled guards, not stability certification. Command changes censor attempts and credit; identical fixed writes do not. Flicker, invalid/overlong completion and resets consume credit. Reading a reward/observation does not advance the tracker, and repeated reward accumulation is idempotent per transition.

Limb lift reattaches the current base-relative wheel transform to the **frozen takeoff base** and measures the cylinder lower-surface increase. `h_use=min(max(h_actual,0),2*max(h_limb,0))`; factor 2 is explicit in config. This spin-invariant kinematic proxy rejects rigid trunk motion, but does not identify causal ground-force work. Target `h=.025+.025*G_takeoff` is frozen. With `u=clamp((peak_use-.008)/(h-.008),0,1)`, height quality is `(.15+.85*u²)*exp(-(relu(peak_actual-h-.020)/.020)²)`; multiply by `clamp(net_xy/.040,0,1)`. Invalid attempts have zero quality. Each wheel starts with zero credit, accumulates `G*dt` up to 1 s, and consumes it once. The final dimensionless payment is `.15*sum(Q*b)`, with no unloaded-wheel normalization. Consequently total payment cannot exceed `.15*sum_i(credited demand time_i)`. No completion means no payment. This training history is hidden from the unchanged 56-value actor/critic observation; it is not a new robot sensor requirement.

A's seven probabilities, stand/duration mixture, signed vx [-.35,1.10], yaw [-1.4,1.4] and exact zero yaw in half of Mixed are retained. Pure lateral is 80% U(.03,.30), 20% U(.30,high), with balanced signs. `high=.30` through completed update 500, linearly .50 at 1500, then .50. Mixed stays [-.30,.30]. The degenerate initial tail is exactly .30. A native update wrapper increments completed updates; checkpoint `infos.event_step_v1`, diagnostics and preparation metadata save the count and high limit. Resume restores it before resampling; fixed evaluation stays fixed. Native iteration labels are not curriculum counters.

The intended fresh run is 4096×64×2000, seed 1, experiment `go2w_event_step_v1`, saves every 500. Native PPO retains 5 epochs, 8 minibatches, clip .20, lambda .95, desired KL .01, clipped value loss weight 1 and max grad norm 1; gamma is .995, initial LR 3e-4 with the native adaptive scheduler. Rollout duration is 1.28 s. Learned log std starts .40, bounds [.10,.70], entropy .003. Initial physical target std is .12/.14/.16 rad and 7.2 rad/s before clipping. A native Adam post-step hook projects raw log std under no_grad without resetting moments, changing means or drawing randomness; it remains installed on resume. Friction [.6,1.2], base mass delta [-.5,1.5] kg, COM ±.015 m, coherent gain factors [.85,1.15], delay {0,1,2} policy ticks with three-slot action history, physical pushes and existing noise are retained. This actuator delay is not observation latency. These choices do not guarantee event discovery or sim2real success.

The CPU cycle calculation below assumes 1.2 s, G=1, perfect matched tracking, a .4 s swing with 4 cm net repositioning, and prescribed loads leaving 2/3 supports. Hip/sagittal errors during swing are sinusoidal with peaks .12/.05 rad for shuffle and .04/.20 for the step. Bobbing is a rigid 5 cm sin² trunk translation; imposed loads are a mathematical counterexample, not a physically consistent contact simulation. All affected rate integrals are listed; unchanged actuator costs are held equal and excluded. This is neither PPO advantage nor a learned-return forecast.

| Integral / event | Shuffle 1 / 2 legs | 5 cm step 1 / 2 legs | Rigid bob 1 / 2 legs |
|---|---:|---:|---:|
| Linear tracking | 1.2 / 1.2 | 1.2 / 1.2 | 1.2 / 1.2 |
| Yaw tracking | .96 / .96 | .96 / .96 | .96 / .96 |
| Hip pose | -.000720 / -.001440 | -.000080 / -.000160 | 0 / 0 |
| Sagittal pose | -.000015 / -.000030 | -.000240 / -.000480 | 0 / 0 |
| Leg motion; wheel air | 0 / 0 | 0 / 0 | 0 / 0 |
| Prolonged unloading; insufficient support | 0 / 0 | 0 / 0 | 0 / 0 |
| Vertical velocity | 0 / 0 | 0 / 0 | -.030843 / -.030843 |
| Roll/pitch rate; orientation; stand | 0 / 0 | 0 / 0 | 0 / 0 |
| Height | 0 / 0 | 0 / 0 | -.003 / -.003 |
| Removed default-pose; instantaneous clearance | 0 / 0 | 0 / 0 | 0 / 0 |
| Completed event total | .019599 / .039197 | .129000 / .258000 | 0 / 0 |
| Affected total | 2.178864 / 2.197727 | 2.288680 / 2.417360 | 2.126157 / 2.126157 |

The shuffle apex is 1 cm. Height remains graded; full quality at G=1 requires 5 cm and overshoot is penalized. Matched event payments at dt .01/.02/.04 differ only by dwell/credit quantization (at most .006 here). Splitting .8 s credited time into 1/2/4 unit-quality completions always gives .12 per wheel; invalid attempts only discard budget. No claim is made that more frequent events preserve otherwise unused capped credit.

Full URDF FK/IK checks a smooth 5 cm cylinder lift with 2 cm lateral displacement for every leg, with all targets inside the unchanged hard limits. Half the lateral displacement occurs at the apex and the rest by touchdown. Pure-lift front calf changes for 5/6/8/10 cm are -.335131/-.395925/-.513060/-.625218 rad (rear -.337579/-.398630/-.516118/-.628445). Minimum target margins at 5 cm are .064869 front / .062421 rear rad; 6 cm leaves only .004075/.001370. 8 and 10 cm exceed the target envelope; they are read-only probes, not required targets. Full inward 2 cm displacement **at** the 5 cm apex exceeds the calf target by .001151 front / .003462 rear rad. That restrictive pose is recorded explicitly; it is not the tested smooth trajectory and no limit was widened.

Exactly **one nominal GPU feasibility process**, exit **0**, collected four 5 s free-base trials (20.10 s including reset settling steps), gravity/contact/limits active, zero wheel targets, no PPO. Support targets request 25 mm longitudinal and 30 mm lateral trunk transfer away from the swinging limb. Target angles stay within the action envelope. **Collection succeeded; the commanded lift behavior failed.** Planned-swing measurements, excluding initial settling:

| Limb | Cylinder apex, mm | Minimum wheel load, N / loaded wheels | Max roll/pitch, deg | Trunk z range, m | Maximum substep effort / limit |
|---|---:|---|---|---|---:|
| FL | .322 | 0 / 3 | 5.514 / 3.439 | .3818–.4084 | .3641 |
| FR | .322 | 0 / 3 | 5.514 / 3.439 | .3818–.4084 | .3641 |
| RL | -.085 | 13.447 / 4 | 4.551 / 1.822 | .4042–.4161 | .2608 |
| RR | -.085 | 13.443 / 4 | 4.550 / 1.822 | .4042–.4161 | .2608 |

All four reload/retain load afterward; there are no falls or non-wheel contacts. Negative clearance is the small cylinder/solver contact overlap, not a successful lift. Whole-trial target tracking RMS [hip,thigh,calf] is approximately [.02484,.05663,.12776] rad front and [.02659,.03762,.13609] rear. Whole-trial 9.322/7.503 mm maxima occur during initial settling and are **not** evidence of a completed commanded lift. Force maxima use readings after every .005 s substep; all 16 values are retained. Trunk settling/load transfer and PD tracking confound the test, with no measured force saturation. This cannot establish a motor-power limit or confirm useful loaded 5 cm stepping. No physical retest or automatic adjustment followed.

The **one independent fresh 64×64×2 smoke**, exit **0**, saved `logs/go2w_event_step_v1_smoke/event_step_smoke_seed1_20260927_2026-09-27_16-58-18/model_1.pt`, SHA-256 `6e9d7c86d080bad994f5000f40a3793292f50641b427b4caf5410bb2ac8cc57a`. Fresh actor/critic/normalizers/optimizer/std were verified before updating. Exactly 128 transitions and 80 Adam steps had finite rewards, losses, parameters and gradients; event-cache count advanced exactly once per transition. Native save/reload matches all learned state, restores completed count 2 and retains the hook. Initial LR was .0003; native adaptive scheduling reached .00001 in this tiny smoke and was not altered. All 16 std parameters have **0% lower/upper boundary occupancy**; effective [hip,thigh,calf; wheel] std after update 2:

| FL | FR | RL | RR |
|---|---|---|---|
| .400093,.400322,.400254; .399680 | .400417,.400281,.400115; .400115 | .400342,.399783,.400531; .400055 | .400536,.399616,.399945; .399908 |

DR readback spans effective friction .600898–1.176502, mass delta -.448253–1.473088 kg and maximum COM shift .014947 m, with coherent gains and all three delays. Actual delayed actions match history; selective reset preserves the other environments' properties/delays. Genesis is destroyed and its initialization flag cleared. This smoke proves implementation operation, not learned stepping. Its checkpoint must never initialize the intended run.

Validation: **20 focused CPU tests pass**. Six unittest invocations have exits **1,1,0,0,0,0**: initial dtype/fixture/Windows-handle errors were repaired; the second invocation exposed the restrictive inward-apex pose described above; the smooth path then passed. Final review initialized unknown contact as unloaded until a genuine reload threshold is reached, followed by all 19 then-existing CPU checks passing; one additional lightweight summary test checks separate physical and qualified counts. This final reset-edge repair was CPU-tested after the successful smoke, without another GPU process. Ruff error checks, compilation and `git diff --check` pass. Large reports, arrays and every failed log remain in `.migration-audit/event-step-20260927/`. No historical evaluation suite, bank, finetune, checkpoint search, export, navigation/hardware command or transfer was run.

The later `precision_screen` retains common schedules and historical 2/3/4 cm metrics, adds 5 cm physical fractions plus separate command-qualified event records, and adds bilateral .05/.4/.5 lateral schedules explicitly marked expanded envelope without an invented A baseline. It records policy-rate geometry, net repositioning, posture, clipping, stops and terminal failures. Reuse A's saved common screens/closed-loop evidence. Judge repeated per-leg swings across directions alongside quiet stand, low-speed vx, reverse, yaw/lateral tracking and stops; do not promote height alone. Closed-loop/DR follow-up is conditional on a useful nominal candidate and is not part of this preparation.

Exact **unexecuted** intended fresh run (physical lift feasibility remains unresolved). The dated run name is new; no checkpoint, optimizer, normalizer or old std is loaded:

```powershell
Set-Location 'C:\Users\Liamb\SynologyDrive\TUM\3_Semester\dodo_alive\legged-robot_rl_genesis'
$Python = 'C:\Users\Liamb\anaconda3\envs\genesis-gpu\python.exe'
$env:NUMBA_CACHE_DIR = "$PWD/.migration-audit/diagnostics-20260925/numba-cache"
$env:GS_CACHE_FILE_PATH = "$env:TEMP/go2w-diagnostics-genesis"
$env:QD_OFFLINE_CACHE_FILE_PATH = "$env:TEMP/go2w-diagnostics-quadrants"
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
$RunName = 'event_step_v1_seed1_20260927_prepared'
& $Python -m robot_gym.scripts.train --task go2w --go2w_profile event_step_v1 --experiment_name go2w_event_step_v1 --run_name $RunName --num_envs 4096 --max_iterations 2000 --seed 1 --headless --logger tensorboard --training_diagnostics
```

After that run only, select its exact unique name and verify the final completed counter (the fresh smoke verified native label 1 after 2 updates; the intended final label is 1999). The following lightweight evaluation and optional playback are **not executed here**; never use `latest`:

```powershell
$Runs = @(Get-ChildItem -LiteralPath "$PWD/logs/go2w_event_step_v1" -Directory | Where-Object { $_.Name.StartsWith("${RunName}_") })
if ($Runs.Count -ne 1) { throw 'Expected exactly one run with the specified unique name' }
$Run = $Runs[0].FullName
$Meta = Get-Content -LiteralPath "$Run/preparation.json" -Raw | ConvertFrom-Json
if ($Meta.status -ne 'completed' -or $Meta.completed_updates_total -ne 2000 -or $Meta.last_iteration_label -ne 1999) { throw 'Final completed-update counter does not match' }
& $Python -m robot_gym.scripts.evaluate --task go2w --go2w_profile event_step_v1 --experiment_name go2w_event_step_v1 --load_run $Run --checkpoint 1999 --eval_mode precision_screen --num_envs 1 --seed 1 --headless --logger tensorboard --output "evaluation/$RunName/final_screen"
& $Python -m robot_gym.scripts.play --task go2w --go2w_profile event_step_v1 --experiment_name go2w_event_step_v1 --load_run $Run --checkpoint 1999 --num_envs 1 --seed 1 --steps 1000 --command_vx 0 --command_vy 0.05 --command_yaw 0 --no_export
```

Conceptual references: [IsaacLab v2.3.2 first-contact airtime reward](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py), [Walk These Ways](https://arxiv.org/abs/2212.03238), and [Advanced Skills through Multiple Adversarial Motion Priors](https://arxiv.org/abs/2203.14912). This custom event-apex objective reproduces neither their parameter sets, gait clocks nor AMP implementations, and its command gate changes preference rather than proving each step physically necessary.

## Event-step result and rollout overhead — 28 September 2026

Reviewed clean `testing` at `4733792200e10d41660e786f78bb7296bc614405`. Run `logs/go2w_event_step_v1/event_step_v1_seed1_20260927_prepared_2026-09-28_10-30-59` completed 2,000 fresh updates: native label 1999, saved curriculum count 2000/high .50. Checkpoint/config SHA-256 match `eda85fba4f0b3911ed3051e9011b68ddcfb215c9c698f113f109b9b0a9b0ccc2` / `43a949f8084a0b0fb3fafc1928dd01ae13997d204f939b9d6124be6ec232013e`. Its matching evaluation manifest identifies `evaluation/event_step_v1_seed1_20260927_prepared/final_screen`; no latest-run selection was used. All seven original traces, reports, checkpoint/config, URDF and A were preserved and hashed before/after. Scripts, full joint/window tables, reconstructed attempts and timing samples remain local in `.migration-audit/event-review-20260928/`.

The existing screen contains 750/950/950/650/650/850/850 transitions: **113 nominal simulated seconds, no recorded falls, resets, timeouts, nonfinite states or non-wheel contacts**. The initial two-second stand states/actions are identical across files, not seven independent trials. There are **11 completed physical >=5 cm lifts**, including repeated rear-leg lifts, and four >=5 cm reward-qualified completions. Thus the earlier unresolved scripted trial is superseded as a statement about whether the learned policy can lift 5 cm. That trial itself remains unsuccessful scripted control, not evidence of weak motors. These traces do not establish robustness or transfer readiness.

Rolling tracks +.1/+.5/-.25 at final-second vx .102751/.506223/-.242753 m/s. The +.5 posture deteriorates while tracking improves:

| Rolling phase/window | vx, m/s | Base z, m | Front thigh q FL/FR, rad | Front wheel x FL/FR, m | Rear wheel x RL/RR, m |
|---|---:|---:|---|---|---|
| Initial stand, last second | .000394 | .414523 | .7984/.7977 | .1697/.1701 | -.2762/-.2761 |
| +.1, first / last second | .094276 / .102751 | .417681 / .414062 | .7572/.7566 → .6417/.6417 | .1802/.1806 → .2213/.2219 | -.2799/-.2796 → -.2884/-.2899 |
| +.5, first / last second | .477586 / .506223 | .410162 / .376622 | .4584/.4564 → .2397/.2385 | .2818/.2827 → .3558/.3563 | -.2637/-.2651 → -.2352/-.2365 |
| -.25, first / last second | -.139960 / -.242753 | .388681 / .388548 | .5447/.5449 → .9672/.9997 | .2712/.2713 → .1423/.1330 | -.2519/-.2541 → -.2118/-.2103 |
| Final stand, last second | -.012120 | .407328 | .7881/.8001 | .1781/.1785 | -.2710/-.2819 |

Wheel positions above are centers relative to the current base. Exact joint order is `[FL,FR,RL,RR] × [hip,thigh,calf,foot]`; wheel outputs are velocity targets, not joint-position targets. Nominal leg triples are `[0,.70,-1.33]` front and `[0,.75,-1.31]` rear. Targets were reconstructed from **applied** actions and saved scales; applied and issued clipped actions happen to match throughout this nominal rolling trace. At +.5, final-second raw front-thigh means are -1.2103/-1.2221, both saturated for 100% of samples: target .350/.350, actual .2397/.2385, nominal error -.4603/-.4615 and target error -.1103/-.1115 rad. Front calf q is -1.4642/-1.4666 versus targets -1.3126/-1.3141; rear thigh/calf q is [.9820,-1.6229]/[.9903,-1.6293]. Mean wheel loads are [41.65,41.74,53.71,54.42] N. Front-thigh measured policy-rate forces are +4.417/+4.465 Nm; maximum force/limit over all joints is .192. The closest hard joint-limit margin is .6254 rad; front thighs have hard limits [-1.5708,3.4907]. **Action-target saturation, PD state error and mechanical limits are separate.**

Initial stand has rear-thigh raw means 1.0311/1.0319 and 100% positive target saturation in its last second: target 1.100, actual 1.0012/1.0015, nominal .750 rad. Their hard limits are [-.5236,4.5379], and the minimum margin over all leg joints is .5573 rad. Mean wheel loads are [54.49,54.49,41.42,41.20] N; maximum policy-rate force/limit is .258. These are not substep force maxima.

Original reward methods evaluated on recorded states give the following **weighted mean rates**. Multiply by .02 for a policy-step contribution; each one-second integral has the same numeric value. Event payment is discrete, with no extra dt; it is zero in these rolling windows. Slew uses consecutive issued clipped actions, as training does, not q-target tracking error.

| Rate | Initial stand, last second | +.5, first second | +.5, last second | Reverse, last second |
|---|---:|---:|---:|---:|
| Linear / yaw tracking | .999981 / .799999 | .982779 / .799961 | .999832 / .799968 | .999507 / .799733 |
| Sagittal / hip pose | -.014383 / -.000010 | -.018034 / -.000246 | -.057973 / -.000517 | -.037775 / -.002048 |
| Height / orientation | -.000003 / -.001285 | -.000532 / -.000840 | -.011783 / -.000859 | -.005618 / -.000606 |
| Normalized effort | -.000820 | -.000967 | -.000585 | -.001208 |
| Leg / wheel action slew | <.000001 magnitude | -.000446 / -.000324 | <.000001 magnitude | <.000001 magnitude |
| Vertical / roll-pitch velocity | -.000003 / -.000012 | -.000321 / -.001360 | -.000173 / <.000001 | <.000001 / -.000036 |

The stationary folded posture is inexpensive relative to near-maximal tracking, and costs almost no action slew. A's saved `evaluation/precision_20260927_132600/A_screen/rolling_reverse.npz` has the identical command schedule/joint order: at +.5 its final vx/z are .463569/.424165, front-thigh q .6386/.6340 and targets .6080/.6093, with no front-thigh saturation. Event-step improves rolling speed accuracy while losing this posture margin; this cross-policy comparison does not isolate causality. Offline rescoring +.5's final second with G=0 sagittal weights -.6/-1.2/-2.0 gives -.057973/-.115947/-.193244, holding every state and other term fixed. This is reward arithmetic, not a policy rollout, PPO advantage or proof of causal improvement. Per-limb stance/swing relaxation could eventually discourage a dragging stance limb while another swings, but introduces contact-state coupling and more choices. Restoring stronger G≈0 stance retention is the simpler justified first intervention.

Event audit below uses **O/A/C/V**: raw unloading opportunities / eligible reward attempts / completed geometric lifts (existing >=2 mm, no dwell metric) / valid paid completions, assigned to their onset command. Numbers after `;` are final dimensionless `.15*sum(Q*b)` contributions. Physical completion and reward confirmation need not coincide. Rolling and ±.03/±.10 yaw have zero O/A/C/V for every limb; their G is zero.

| Command | FL | FR | RL | RR |
|---|---|---|---|---|
| vy +.10 | 3/3/3/3; .0959 | 26/4/0/0; 0 | 18/4/4/0; 0 | 3/3/3/3; .2408 |
| vy -.10 | 26/4/0/0; 0 | 3/3/3/3; .0982 | 3/3/3/3; .2503 | 21/4/3/0; 0 |
| vy +.20 | 4/4/4/4; .1572 | 29/4/0/0; 0 | 13/4/5/2; .0522 | 4/4/4/4; .5138 |
| vy -.20 | 27/4/0/0; 0 | 4/4/4/3; .1308 | 4/4/4/1; .1380 | 21/5/4/1; .0246 |
| yaw +.40 | 7/7/4/0; 0 | 0/0/0/0; 0 | 15/5/0/0; 0 | 11/7/1/0; 0 |
| yaw -.40 | 0/0/0/0; 0 | 7/7/4/0; 0 | 9/7/1/0; 0 | 16/6/0/0; 0 |
| yaw +.75 | 8/8/8/4; .0529 | 6/5/0/0; 0 | 19/7/0/0; 0 | 12/7/8/0; 0 |
| yaw -.75 | 8/6/0/0; 0 | 8/8/8/4; .0554 | 13/8/8/0; 0 | 11/7/0/0; 0 |
| vy +.05* | 1/1/1/0; 0 | 6/1/0/0; 0 | 6/2/1/0; 0 | 1/1/1/1; .0280 |
| vy -.05* | 5/1/0/0; 0 | 1/1/1/0; 0 | 1/1/1/1; .0282 | 5/2/1/0; 0 |
| vy +.40* | 11/6/6/1; .0109 | 24/5/12/0; 0 | 6/5/6/1; .0235 | 11/6/5/1; .0427 |
| vy -.40* | 17/5/9/0; 0 | 12/5/6/0; 0 | 5/5/5/2; .0952 | 8/5/6/1; .0208 |
| vy +.50* | 22/6/8/0; 0 | 16/5/8/0; 0 | 5/5/5/2; .0460 | 14/5/5/0; 0 |
| vy -.50* | 18/5/13/0; 0 | 21/5/5/0; 0 | 9/5/5/1; .0239 | 5/5/5/3; .0733 |

`*` Expanded-envelope checks without an A baseline. Final-second vy at ±.50 is only +.304760/-.329729 m/s. Across the screen there are 203 completed geometric intervals, 405 completed load-only intervals, 28 physical boundary-censored intervals, and 49 valid reward events totaling 2.202857. Reward attempts separately comprise 161 confirmed completions, 72 flicker cancellations, four overlong cancellations and 12 command cancellations. Prior-support/dwell requirements explain why many raw unloadings never arm. The initial settling transition before recording is unavailable; boundary motion remains censored. No failure cancellation occurs in these traces.

At vy +.20, RR actual/usable peaks are 58.59/47.85/52.54/53.09 mm, duration .24/.24/.26/.26 s and net repositioning 143/129/137/137 mm. Height quality is 1/.9151/1/1, reposition quality 1 throughout, and consumed credits .78/.88/.92/.92 s. FR's sole confirmed attempt reaches only 1.06 mm; its other three attempts flicker. Negative lateral mirrors the front dragging imbalance, but not payment: RL peaks 52.73/51.89/46.40/47.05 mm have three legitimate sampled-support rejections. Each rejected swing contains a sample with only one wheel >6 N (indices 334/427/474); usable height, .26 s duration and 129–141 mm repositioning otherwise pass. Two good-height RL swings at -.40 also fail sampled support. Do not relax this guard to manufacture payment. Support between 50 Hz samples remains unknown.

Both yaw ±.40 already have G=1. Their first FL/FR swings reach 11.104/11.565 mm actual height but only 1.893/2.346 mm usable height, with .18 s duration, 61.94/63.13 mm displacement and valid sampled support. **Usable height alone rejects those two events**; later attempts mostly fail actual/usable height, some duration or net displacement. Across all paid events height quality spans .1500–1, reposition quality .9728–1 and consumed credit .48–1 s. The additive independent-foot objective permits productive limbs to earn credit while another drags; the +.20 result shows this directly. It does not require four useful steps, and no equality/symmetry penalty or guard relaxation was added. Quiet initial stand is retained, but stops are imperfect: final yaw residuals after positive/negative yaw are -.0341/+.0133 rad/s, and after negative lateral +.0514 rad/s with vy +.0148 m/s.

All 2,000 `diagnostics.jsonl` rows exist; cache counts are exactly `1+64*completed_updates`. TensorBoard is missing event label 285; A has 15 missing labels, including 13 in its last 250-label window. No rows were filled. From labels 1500–1749 to 1750–1999, mean episode reward rises 28.3757→29.4955, raw nonterminal reward/transition .028039→.029158 and episode-normalized step-event contribution .005125→.009685. These are stochastic training statistics, not intermediate-checkpoint behavior. Last-250 scheduler KL mean/p95 is .015157/.019403; LR min/mean/max is .00007594/.00034590/.00038443, final .00038443. Native adaptation exceeded the initial .0003 as allowed. Time shares stand/straight/arc/yaw/precision/lateral/mixed are 18.378/19.240/19.155/14.375/9.584/12.539/6.729%. Training logs do not contain cumulative valid-event counts; `censored_attempts_since_reset=13793` is a reset-dependent snapshot, not a total.

Final learned normalized std, matching the checkpoint, is per joint below. All 2,000 logged update-end vectors have 0% lower/upper/outside-bound occupancy; within-update boundary occupancy was not recorded. Mean std alone would obscure the low front-calf exploration.

| Limb | Hip | Thigh | Calf | Wheel |
|---|---:|---:|---:|---:|
| FL | .164007 | .231504 | .113757 | .307831 |
| FR | .164073 | .231396 | .113655 | .307844 |
| RL | .144321 | .130716 | .165026 | .260323 |
| RR | .144353 | .130674 | .164915 | .260122 |

Historical runtime does **not** show a 2× per-transition slowdown. Last-250 event means are 4.13514 s collection + .86501 s learning for 4096×64, or 15.774/3.300 µs per environment transition. A's available 237 rows in labels 1549–1798 give 3.29076 + .77332 s for 4096×48, or 16.738/3.933 µs. Different runs, learning settings and system load prevent attributing that difference to the tracker. Native host timers include action/reward telemetry in collection and diagnostic flush in learning; they are not synchronized kernel profiles. Inter-row residual wall time averages .0413 s event/.0584 s A for adjacent available rows, including logger/save/other overhead. Startup and individual logging/flush costs were not separately instrumented and cannot be recovered exactly.

One baseline and one candidate **pure tensor GPU replay process**, RTX 4070 Ti, batch 4096, reused recorded inputs (no actor, simulator or optimizer). Each component had 32 warmup calls then four blocks of 64, timed with CUDA events and host wall time synchronized before/after each block. Paired candidate-process medians, ms/call:

| Component | Original CUDA / wall | Accepted CUDA / wall |
|---|---:|---:|
| Event update, including command handling | 5.514 / 5.514 | 3.567 / 3.568 |
| Empty tracker reset | .752 / .752 | .00077 / .00107 |
| Reward telemetry | 2.355 / 2.355 | 1.831 / 1.832 |
| Sparse tracker reset | .871 / .871 | .887 / .888 |

The first unchanged-versus-unchanged baseline tracker timings were 4.867/5.050 ms, showing run variability. The accepted changes replace dynamic takeoff/touchdown gathers with fixed-shape masks, reuse clearance refreshed immediately before the one event update, skip empty resets and keep new-profile hip/sagittal indices on-device. Isolated duplicate clearance costs .286 ms; Python-list/CUDA indexing costs .0395/.0113 ms. Same-command handling (.148→.155 ms) and sparse reset show no useful gain alone. An action-telemetry masking candidate (.843→.886 ms wall) was reverted. Gate/finite/transform calculations otherwise remain unchanged: no speculative cache invalidation scheme, telemetry reduction or physics shortcut. **The 35% tracker and 22% reward-telemetry savings are component measurements, not a measured end-to-end training gain.**

Validation: two focused unittest invocations, 3 and 6 tests, exits **0/0** (eight distinct checks, one repeated after reverting action telemetry). Original/optimized tracker state is bit-exact for all 5,650 CPU trace inputs, reset/command/nonfinite/failure fixtures and 96×4096 GPU replay inputs with sparse resets, including credit and censor bookkeeping. Comparing reconstructed CPU events to the saved GPU cache gives exact decisions/durations/censor counts; maximum geometric float difference is 7.1e-8 m and payment difference 3.0e-7 before the .15 scale. Telemetry counts are exact; reordered sum comparisons pass 5e-7 relative/1e-5 absolute tolerance. Both GPU replay processes and the corrected offline analysis exit **0**. Two initial local script launches were stopped during prolonged startup without the documented cache environment (exit **-1/-1**); cached retries succeeded. The first offline reconstruction also exposed an analysis-only redundant command-cancellation call; matching the actual evaluator's direct command writes removed all censor mismatches. Original logs are retained. Compilation and `git diff --check` pass. `python -m ruff` was unavailable (exit **1**); the existing base-environment `ruff.exe` passes E9/F63/F7/F82 checks (exit **0**), without installing anything. No simulation was rerun, no PPO update or checkpoint continuation occurred, and no export/bank/hardware work was performed.

**One next learning recommendation, not implemented or launched:** a single bounded **500-update continuation from this exact model_1999.pt**, retaining its optimizer, normalizers, std, native scheduler and completed curriculum state. Change only sagittal stance retention to `-(2.0*(1-G)+.12*G)*mean(thigh/calf error²)`, restoring -2.0 at G=0 while preserving -.12 at full demand; keep event qualification/payment and all physical settings unchanged. This addresses saturated forward posture while retaining learned rolling and real lifts. Inspect only that continuation's final checkpoint on the same nominal screen: require improved forward height/target margin without worse stand, reverse, bilateral tracking/stops or useful per-leg events. Do not simply extend the unchanged objective, invent a four-foot equality rule or infer deployment readiness. A remains the fallback.

## Sagittal stance continuation — 28 September 2026

**Partial result; keep CK1999 and A as fallbacks.** The single continuation improves rolling posture and repeated bilateral rear-leg lifts, but does not recover front-thigh target reserve. Initial stand drift, low-speed/reverse tracking and the positive-lateral stop also regress. It is a useful development artifact, not an unconditional replacement. No further training or evaluation is launched.

Reviewed clean `a101722119ceb4da34b4687391ff928f8c99c6b6`; implementation, training and evaluation use clean `adc612d67a09c52f967c4b91b8642fdce9208298` (`tighten go2w rolling posture`). The opt-in `--sagittal_stance_weight 2.0` is restricted to `event_step_v1`. Its weighted sagittal **rate** is `-(2*(1-G)+.12*G)*E`, with coefficients -2/-1.06/-.12 at G=0/.5/1 and one .02 s multiplication. The unset path retains the historical calculation. Saved train/evaluation/play contracts reject a missing or different stance selection and unrelated continuation changes. Event rules, full-demand cost, physical interface, tracking, PPO and DR remain unchanged; saved environment configs differ only at this field, and nominal loaded physics readbacks are identical.

Parent: `logs/go2w_event_step_v1/event_step_v1_seed1_20260927_prepared_2026-09-28_10-30-59/model_1999.pt`, with the checkpoint/config hashes recorded above verified exactly. Native resume verification matched actor, critic, both normalizers, raw log-std, Adam tensors and both LR representations before either smoke or real updates. Loaded LR was .00038443359375000017; Adam started at 80000. Completed count 2000 was restored before command sampling, retaining lateral high .50 and the projection hook. The newly seeded simulator/RNG history is not a replay of the parent.

The real run is `logs/go2w_event_step_v1/sagittal_retention_seed1_20260928_172600_2026-09-28_17-27-35`: exactly 4096×64×500 additional updates, seed 1. Native final **model_2498.pt** has completed count **2500**, lateral high **.50**, Adam **100000** and cache count **32001**. Final checkpoint SHA-256 is `4de494271a999c33ca6a8a06913b1facb92eede58c92d7297762671b2b1080e8`; adjacent config is `340a8998d3015b810c46e4d808cfb173837d20aa2a2e1028a6004e198b8ff25d`. Continuation and end-of-run manifests record the original parent and override. The smoke was never a parent.

One final `precision_screen` is saved at `evaluation/sagittal_retention_seed1_20260928_172600/final_screen`. It matches the archived parent's seven schedules, one environment, seed, nominal physics, disabled observation noise/push/brake, 30 s timeout and 50 Hz capture. Both screens contain 113 simulated seconds/5650 ticks, with zero recorded falls, resets, timeouts, nonfinite states or non-wheel contacts. The identical initial stand repeated across each screen is not independent robustness evidence. All 15 protected parent/A/checkpoint/config/URDF/screen files retain their original hashes.

Final-second primary measurements:

| +.5 m/s measurement | Parent | Continuation | Development target |
|---|---:|---:|---|
| Actual vx, m/s | .506223 | .510135 | Error <=.03: met |
| Base height, m | .376622 | .399191 | >=.400: missed by .000809 |
| Front-thigh nominal RMS, rad | .460926 | .277795 | 39.7% reduction; >=25% met |
| FL/FR lower target saturation | 100%/100% | 100%/100% | <=20% each: missed |
| FL/FR target reserve, rad | 0/0 | 0/0 | No improvement |
| Roll/pitch RMS, rad | .00235/.02666 | .00056/.04437 | Pitch increased |

All four thighs below use actual q and delayed-action targets, in radians. Parent→continuation; saturation is final-second target-bound occupancy. Nominals remain .70 front/.75 rear.

| Limb | Initial stand q | Initial target (upper saturation) | +.5 q | +.5 target (lower saturation) | +.5 thigh/calf RMS |
|---|---|---|---|---|---|
| FL | .7984→.7951 | .6507→.6123 (0→0%) | .2397→.4223 | .3500→.3500 (100→100%) | .3390→.2253 |
| FR | .7977→.7945 | .6486→.6127 (0→0%) | .2385→.4221 | .3500→.3500 (100→100%) | .3403→.2244 |
| RL | 1.0012→.8352 | 1.1000→1.0144 (100→0%) | .9820→.8646 | .9495→.9928 (0→0%) | .2754→.1416 |
| RR | 1.0015→.8335 | 1.1000→1.0117 (100→0%) | .9903→.8625 | .9564→.9910 (0→0%) | .2825→.1410 |

Forward raw front-thigh means remain -1.1123/-1.1087, beyond the action bound. Actual-minus-target errors change from about -.111 to +.072 rad; improved state posture does not establish target reserve or a motor-limit change. Initial rear-thigh saturation is removed, with .0856/.0883 rad target reserve. Initial stand height changes .414523→.410465 m; final rolling stand height .407328→.407742 m. Initial stand XY RMS worsens .002197→.007962 m/s, chiefly +vx drift. Final rolling stand vx changes -.012120→+.009424 m/s.

Tracking below is final-second requested-axis velocity, parent→continuation; yaw is rad/s, translation m/s. Expanded lateral checks have no invented A baseline.

| Command | Positive / forward | Negative / reverse |
|---|---|---|
| vx +.10 / -.25 | .102751→.110116 | -.242753→-.228220 |
| yaw ±.03 | .004987→.009024 | -.005107→-.009197 |
| yaw ±.10 | .101707→.093278 | -.103982→-.096027 |
| yaw ±.40 | .402496→.414566 | -.402923→-.412605 |
| yaw ±.75 | .771131→.735427 | -.764931→-.727220 |
| vy ±.05, expanded | .048446→.047120 | -.048034→-.047875 |
| vy ±.10 | .101913→.102354 | -.100391→-.100904 |
| vy ±.20 | .180963→.186364 | -.175478→-.189122 |
| vy ±.40, expanded | .314230→.350974 | -.310948→-.348036 |
| vy ±.50, expanded | .304760→.451616 | -.329729→-.436101 |

Final-stop XY/yaw RMS and maximum of phase-final-second roll/pitch RMS, parent→continuation. These use the existing metrics, not newly tightened acceptance limits.

| Schedule | Stop XY, m/s | Stop yaw, rad/s | Max phase roll/pitch RMS, rad |
|---|---|---|---|
| Rolling/reverse | .01295→.00943 | .01715→.00082 | .0147/.0327→.0006/.0444 |
| Yaw + | .00908→.00989 | .03489→.00419 | .0708/.0423→.0691/.0483 |
| Yaw - | .00747→.00542 | .01400→.00591 | .0743/.0411→.0696/.0490 |
| Lateral + | .01417→.00587 | .00374→.02913 | .0350/.0327→.0632/.0498 |
| Lateral - | .01898→.00732 | .05254→.00757 | .0355/.0327→.0613/.0475 |
| Expanded lateral + | .01161→.01024 | .03717→.03396 | .0340/.0473→.0507/.0301 |
| Expanded lateral - | .02319→.01186 | .03872→.02241 | .0353/.0484→.0578/.0326 |

Physical completed lifts and paid events remain distinct. Full-screen per-limb counts retain all historical thresholds; apex distributions include all completed geometric intervals >=2 mm. Paid amounts below are final dimensionless contributions `.15*sum(Q*b)`, without dt.

| Limb | Physical >=2/3/4/5 cm, parent→new | Apex p50/p90/max, mm, parent→new | Paid count, parent→new | Paid amount, parent→new |
|---|---|---|---|---|
| FL | 6/0/0/0→12/3/0/0 | 5.9/20.2/28.2→10.8/25.7/32.0 | 12→27 | .3169→.6483 |
| FR | 10/0/0/0→14/3/0/0 | 6.0/25.2/29.7→12.3/26.8/31.6 | 10→28 | .2844→.6852 |
| RL | 19/17/15/4→39/27/24/16 | 13.7/48.7/54.3→26.7/60.3/66.6 | 13→34 | .6573→2.1023 |
| RR | 19/17/15/7→42/26/24/14 | 15.0/52.0/58.6→27.7/58.4/65.6 | 14→28 | .9441→1.9080 |

Qualified usable-height medians FL/FR/RL/RR are 19.5/20.4/26.7/28.5→18.4/19.2/27.4/29.5 mm; qualified net-displacement medians are 86.5/83.9/152.1/136.8→66.2/73.9/143.8/149.0 mm. At +.20, RR completes five paid actual/usable peaks 58.6–65.6 mm with 166–174 mm net repositioning; at -.20, RL completes five paid 57.3–66.6 mm peaks with 167–171 mm net repositioning. Each passes sampled support and .26–.30 s duration. The opposite front limb still earns no lateral payment. At yaw ±.40, previously zero paid events become five FL plus one RR / five FR plus three RL events; this does not establish balanced coordination.

Totals: physical geometric intervals 203→257, load-only intervals 405→362, physical boundary-censored intervals 28→28; paid events 49→117, paid amounts 2.202857→5.343807. Physical >=5 cm counts rise **11→30**, paid events at that actual height **4→21**. Of the other nine new >=5 cm lifts, six confirmed attempts fail sampled support, two have no armed attempt, and one is command-censored. Reward completions are 161→206, flicker cancellations 72→81, overlong cancellations 4→0, command cancellations 12→14. Nonexclusive completion-rejection counts actual height/usable height/net displacement/duration/support are 92/99/14/10/19→59/59/13/10/30. Guards were not relaxed. CPU reconstruction matches every saved completion/valid/censor decision; largest geometric difference is 8.2e-8 m and unscaled payment difference 3.6e-7. Substep support and robustness remain unmeasured.

Training wall time was **3018.777 s** including process startup/shutdown: 131,072,000 transitions, 43,419/s overall. Entry-point startup was 76.100 s; learning/logging/final save 2927.349 s. Matched last-250-label windows (parent 1750–1999, continuation 2249–2498) give collection **4.1351→4.5960 s**, learning including diagnostic flush **.8650→.9018 s**, and **52,427→47,681 transitions/s** from mean blocks. Adjacent logged wall intervals give **52,012→47,160/s**, with unassigned logger/save/other residual .0413→.0611 s/update. These are native host/end-to-end measurements, not synchronized kernel timings; different learned states, the changed objective and uncontrolled system load prevent causal attribution. There is **no measured end-to-end speed gain** from the earlier tensor optimization in this comparison. TensorBoard has all 500 labels and finite values; diagnostics has 499 rows, missing **2189**, with all available counters/cache updates correct. No row was filled. Final LR remains .00038443359375000017; all 76 saved tensors are finite, final raw std matches diagnostics, and the 499 recorded update-end vectors have zero bound/outside occupancy. Within-update occupancy is not recorded.

Validation/process record: three focused CPU test methods; first invocation exit **1** because the lightweight rate fixture lacked `reward_scales` (two methods passed), repaired rate-only invocation **0**, and extended progress/contract-only invocation **0**. One 64×64×2 original-parent smoke exited **0** (144.013 s), verified exact initialization, finite state/loss/gradients, rate endpoints, once-only cache updates, projection and native save/reload at count 2002/Adam 80080. One real training process **0**, one final-screen process **0** (317.303 s external wall), no simulator reruns. Offline summary initially exited **1** on an incorrect assumption of 500 diagnostic rows; preserving the missing row and checking available labels repaired it (**0**). Physical reconstruction, summaries, saved-state/config/hash checks and E9/F63/F7/F82/whitespace checks pass (**0**). Failure logs, exact PowerShell launch/exit records and detailed arrays remain ignored in `.migration-audit/sagittal-retention-20260928/`; no large artifacts are committed.

Commands actually used (the unique output already exists; do not repeat the training): CPU selector `-m unittest tests.test_go2w_event_step.SagittalContinuationCPU -v`, then its `.test_weighted_rates_and_legacy_path` and `.test_only_declared_continuation_difference` methods individually. Smoke selector `-m unittest tests.test_go2w_event_step.SagittalContinuationSmoke.test_original_parent_two_updates -v`, with `GO2W_EVENT_GPU=sagittal`. With the documented `genesis-gpu` Python/cache setup, training was `-m robot_gym.scripts.train --task go2w --go2w_profile event_step_v1 --sagittal_stance_weight 2.0 --experiment_name go2w_event_step_v1 --load_run $Parent --checkpoint 1999 --reference_config "$Parent/config.yaml" --resume --run_name sagittal_retention_seed1_20260928_172600 --num_envs 4096 --max_iterations 500 --seed 1 --headless --logger tensorboard --training_diagnostics`; `$Parent` is the exact parent directory above. Evaluation was `-m robot_gym.scripts.evaluate --task go2w --go2w_profile event_step_v1 --sagittal_stance_weight 2.0 --experiment_name go2w_event_step_v1 --load_run $Run --checkpoint 2498 --reference_config "$Run/config.yaml" --eval_mode precision_screen --num_envs 1 --seed 1 --headless --logger tensorboard --output evaluation/sagittal_retention_seed1_20260928_172600/final_screen`.

Optional GUI playback below is **not launched**. It selects the exact saved choice and disables export; it is not an additional qualification screen.

```powershell
Set-Location 'C:\Users\Liamb\SynologyDrive\TUM\3_Semester\dodo_alive\legged-robot_rl_genesis'
$Python = 'C:\Users\Liamb\anaconda3\envs\genesis-gpu\python.exe'
$env:NUMBA_CACHE_DIR = "$PWD/.migration-audit/diagnostics-20260925/numba-cache"
$env:GS_CACHE_FILE_PATH = "$env:TEMP/go2w-diagnostics-genesis"
$env:QD_OFFLINE_CACHE_FILE_PATH = "$env:TEMP/go2w-diagnostics-quadrants"
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
$Run = "$PWD/logs/go2w_event_step_v1/sagittal_retention_seed1_20260928_172600_2026-09-28_17-27-35"
& $Python -m robot_gym.scripts.play --task go2w --go2w_profile event_step_v1 --sagittal_stance_weight 2.0 --experiment_name go2w_event_step_v1 --load_run $Run --checkpoint 2498 --reference_config "$Run/config.yaml" --num_envs 1 --seed 1 --steps 750 --command_vx 0.5 --command_vy 0 --command_yaw 0 --no_export
```

## Rolling-first objective review — 28 September 2026

Reviewed clean `d6afbde8be8749d6013cb68c0de114ea00697ede`. This offline review reuses both exact 113 s screens and their verified event reconstructions. CK2498, CK1999, A, their configs, URDF and screen files retain all 27 checked hashes. Production settings are unchanged; no simulator/profiler or optimizer ran. Detailed arrays and scripts are ignored under `.migration-audit/rolling-objective-20260928`.

**Better state posture has not recovered target headroom.** At +.5 vx, final-second thigh q/target changes `.2397/.3500 -> .4223/.3500` (FL), `.2385/.3500 -> .4221/.3500` (FR). Both lower targets remain saturated for 100% of samples; CK2498 raw means are -1.1123/-1.1087. Actual-minus-target error reverses from about -.111 to +.072 rad. Height improves .376622→.399191 m at vx .506223→.510135 m/s, but a penalty on actual posture does not directly penalize saturated targets or sustained opposing wheel effort.

Full final-second leg q and **applied delayed-action** targets, hip/thigh/calf radians:

| Limb | CK1999 q | CK1999 target | CK2498 q | CK2498 target |
|---|---|---|---|---|
| FL | .0159/.2397/-1.4642 | -.0486/.3500/-1.3126 | .0328/.4223/-1.4863 | .0005/.3500/-1.4112 |
| FR | -.0167/.2385/-1.4666 | .0475/.3500/-1.3141 | -.0357/.4221/-1.4832 | -.0035/.3500/-1.4095 |
| RL | .0175/.9820/-1.6229 | -.0481/.9495/-1.4820 | .0019/.8646/-1.4743 | -.1096/.9928/-1.2458 |
| RR | -.0139/.9903/-1.6293 | .0511/.9564/-1.4861 | -.0010/.8625/-1.4746 | .1108/.9910/-1.2452 |

The joint order is FL, FR, RL, RR, each hip/thigh/calf/wheel, verified against saved names. Mean signed control moments below are `get_dofs_control_force` at 50 Hz, not total contact moments or substep maxima. Loads are summed wheel-ground normal forces.

| Limb | CK1999 hip/thigh/calf/wheel, Nm | CK2498 hip/thigh/calf/wheel, Nm | Mean load, N, old→new |
|---|---|---|---|
| FL | -2.581/4.417/6.059/-.0733 | -1.297/-2.895/3.002/-1.3763 | 41.646→44.657 |
| FR | 2.571/4.465/6.098/-.0689 | 1.291/-2.884/2.950/-1.3629 | 41.744→44.310 |
| RL | -2.630/-1.303/5.631/.0663 | -4.462/5.128/9.142/1.3717 | 53.715→51.159 |
| RR | 2.603/-1.362/5.726/.0733 | 4.473/5.138/9.179/1.3699 | 54.415→51.396 |

The supplied wheel-moment observation is confirmed: summed absolute mean moments increase .282→5.481 Nm (19.5×), while the signed sum stays near zero. Old wheel targets are 5.830/5.843/5.949/5.961 rad/s against actual 5.903/5.912/5.883/5.888; new targets are 4.483/4.498/7.366/7.366 against 5.859/5.860/5.994/5.996. With Kv=1, this is consistent with sustained front braking/rear driving under load, a descriptive preload finding. It does not identify electrical consumption or prove which actuator caused the posture change. The first movement second already has opposing moments, but acceleration makes it unsuitable as a static comparison; its old/new vectors are [-.417,-.409,1.754,1.747]/[-.930,-.926,2.140,2.135] Nm.

Neither mechanical-position nor force saturation explains the final-second target saturation: minimum hard-position margin across all leg samples is .6254→.6354 rad; front-thigh hard limits remain [-1.5708,3.4907], far outside the .35 action target. No sampled force reaches its limit; maximum effort/limit is .1920→.2586, with CK2498 wheels <=.0582. Final-second normalized-effort **rate** is only -.000585→-.000969 versus height -.011783→-.001999, orientation -.000859→-.002362, hip -.000517→-.001179 and sagittal -.057973→-.070530 under their respective saved weights. Tracking is 1.799800→1.799588; action slew is effectively zero. Static bias is cheap under these terms. A stronger actual-angle scalar or higher gains alone offers no demonstrated mechanism for recovering target reserve. The event bonus is already zero at G=0; reducing it cannot directly repair this issue.

**Step discovery and a rolling-first objective differ.** Current full-demand height quality at 2/3/5 cm is .2194/.3832/1, with full credit through 7 cm, assuming usable=actual height and full net repositioning. This is neither an exact-height constraint nor a test of whether lifting was necessary. Physical >=5 cm completions remain 11→30, versus only 4→21 paid events at that height; all-height paid counts are 49→117. Those are capability measurements, not a controller ranking. Existing support, duration and boundary rejections remain unchanged.

CK2498 weighted components integrated over recorded movement families follow. Tracking deficit means `1.8*T - tracking`, not an added penalty. Pose combines hip/sagittal costs; motion combines effort, leg speed, action slew and joint acceleration. Body combines height/orientation/vertical and angular motion/stand; guards include support, unloading, limits, crossover, collisions and termination. Values are before positive-reward clipping. Rates receive dt once; event totals are discrete, with no extra dt. The first .02 s of each trace lacks prior action/velocity and is omitted from rate sums (112.86 s total); physical counts still cover all 113 s.

| Family (moving commands grouped) | Seconds | Tracking / deficit | Pose | Motion | Body / guards | Event |
|---|---:|---|---:|---:|---|---:|
| Stand, including stops | 34.86 | 62.224/.524 | 1.619 | .120 | .916/.249 | 0 |
| Forward +.1/+.5 | 7 | 12.585/.015 | .322 | .010 | .017/0 | 0 |
| Reverse -.25 | 3 | 5.327/.073 | .136 | .009 | .005/.003 | 0 |
| Yaw + | 14 | 24.654/.546 | .493 | .045 | .198/0 | .333 |
| Yaw - | 14 | 24.644/.556 | .497 | .045 | .198/0 | .352 |
| Lateral +.1/+.2 | 8 | 13.914/.486 | .359 | .051 | .316/.132 | 1.552 |
| Lateral -.1/-.2 | 8 | 13.899/.501 | .361 | .051 | .318/.140 | 1.538 |
| Expanded lateral +.05/+.4/+.5 | 12 | 19.775/1.825 | .552 | .113 | .645/.201 | .724 |
| Expanded lateral -.05/-.4/-.5 | 12 | 19.791/1.809 | .556 | .115 | .674/.333 | .845 |

Per-limb entries are **event / pose cost / motion cost**, preserving independent contributions rather than assigning body tracking to particular limbs:

| Family | FL | FR | RL | RR |
|---|---|---|---|---|
| Stand | 0/.385/.026 | 0/.432/.024 | 0/.401/.034 | 0/.400/.036 |
| Forward | 0/.106/.001 | 0/.106/.001 | 0/.055/.004 | 0/.055/.004 |
| Reverse | 0/.039/.002 | 0/.039/.002 | 0/.029/.002 | 0/.029/.002 |
| Yaw + | .306/.094/.014 | 0/.097/.008 | 0/.165/.008 | .026/.137/.016 |
| Yaw - | 0/.099/.008 | .296/.094/.014 | .056/.138/.016 | 0/.166/.007 |
| Lateral + | .223/.037/.014 | 0/.253/.008 | .257/.033/.013 | 1.073/.036/.015 |
| Lateral - | 0/.255/.008 | .230/.037/.014 | 1.079/.036/.015 | .230/.033/.013 |
| Expanded lateral + | .082/.106/.026 | .013/.313/.022 | .279/.047/.025 | .349/.086/.039 |
| Expanded lateral - | .037/.316/.022 | .147/.104/.027 | .432/.089/.041 | .230/.047/.025 |

On ordinary positive/negative lateral segments, bonuses 1.552/1.538 exceed the combined tracking deficits and all penalties 1.345/1.371; none of these samples is affected by positive clipping. RR alone earns 1.073 against its own .051 pose+motion cost while FR receives zero and incurs .262. Thus independent productive limbs can compensate for another limb's dragging/posture cost. This arithmetic is not PPO advantage or evidence that each lift helped propulsion. Parent ordinary lateral payments were 1.060/.642; more event return is not automatically improved usefulness.

Only three mathematical alternatives were compared: current `.15*Q*b`; same shape at `.05*Q*b`; and `.05*Qb*b`, with `Qb=(.15+.85*clamp((h_use-.008)/.017,0,1)^2)*exp(-(relu(h_actual-.05)/.02)^2)*clamp(net/.04,0,1)`. The latter reaches a broad 2.5–5 cm plateau, then falls smoothly. All retain the same validity mask and credited time. Per-wheel payments with b=1 s, G=1, actual=usable apex and net=.04 m:

| Completed apex | Current quality | Current .15 | Same shape .05 | Broad plateau .05 |
|---|---:|---:|---:|---:|
| 2 cm | .2194 | .03291 | .01097 | .02868 |
| 2.5 cm | .2893 | .04339 | .01446 | .05000 |
| 3 cm | .3832 | .05748 | .01916 | .05000 |
| 5 cm | 1 | .15000 | .05000 | .05000 |
| 7 cm | 1 | .15000 | .05000 | .01839 |

The broad alternative is a deliberate reweighting, not uniformly smaller: it modestly increases a clean 2.5 cm event over current payment. On unchanged saved valid events, current/smaller/broad totals are CK1999 **2.20286/.73429/1.39556**, CK2498 **5.34381/1.78127/2.83396**. CK2498 yaw +/- broad totals are .260/.264; ordinary lateral .671/.667; expanded lateral .477/.495. No rejected event becomes payable. These are counterfactual rescoring hypotheses, not rollouts, advantages or predictions of learning.

Seven small CPU tracker cases confirm clean 2 cm and tall 5 cm completions, each consuming .86 s credit, pay .02830/.12900 currently and .02466/.04300 under the broad arithmetic. No-step rolling pays zero at G=0 and G=1 even with capped credit; a 5 cm rigid-body bob has usable height <5e-8 m and pays zero; 4 mm dragging and alternating 20 ms unload/reload chatter also pay zero. Splitting the same .8 s total credit among 1/2/4 hypothetical unit-quality events gives the same .12 current/.04 reduced budget; this is algebra, not an assertion that all spacings pass dwell/duration. The cap bounds frequency incentives, but a capped unused budget can still favor occasional steps over never stepping.

**Command timing is incompatible with naive 10 Hz interpolation.** The current sampler holds commands piecewise constant: moving durations .5–1 s (70%), 1.5–3 s (25%), 8–15 s (5%); stand retains its 25% 3–6 s replacement. Fixed-command writes bypass sampling. `command_changed` checks exact inequality on any component: every different vector increments generation and clears active/confirmed attempts, credit and prior-support time; identical writes do nothing. A CPU fixture with .4 s initial support, a valid .3 s/3 cm/.04 m swing, unchanged vy=.05 and vx refreshed by just +.001 every .1 s gives: identical writes **one paid event**; changes during swing **zero, one censored attempt**; changes throughout support and swing **zero, no attempt armed** because .1 s < .12 s prior support. Old credit is cleared; .06 s after the last change is newly accumulated credit, not retained credit.

For a future correlated stream, the smallest compatible design would freeze quality/demand at takeoff and let same-direction magnitude refreshes retain the attempt and support history; genuine stop or reversal would still cancel and clear credit. Accumulate current G*dt once, capped, without minting credit on refresh. Define reversal/compatibility explicitly before such an experiment. This is a future reward-semantics change, not interpolation added here or a change to frozen-policy inference; it needs no RNN, estimator or actor input.

Runtime attribution remains **unresolved**. Existing matched last-250 blocks give collection/learning 4.1351/.8650 s parent versus 4.5960/.9018 s continuation at 4096×64, with different policy states/settings and uncontrolled load. They cannot separate actor, physics, state reads, events and telemetry; startup and host logging are reported above. No new simulator or microbenchmark was justified or run, and no performance improvement is claimed.

**One prioritized future intervention, not implemented or launched:** make the event bonus a smaller sufficient-clearance auxiliary objective, using the declared broad 2.5–5 cm quality and .05 coefficient. Bound one continuation from this exact CK2498/native state to **250 additional updates, 4096×64, seed 1**, then only its final existing precision screen against the archived results. Retain stance weight 2, PPO, curriculum, physical interface and all event guards. This tests reducing incentives to chase taller lateral lifts while retaining discovered stepping; it is an objective redesign, not a coefficient ablation. It does not claim to fix G=0 preload/headroom. Judge tracking, stand/stops, posture/headroom and repeated useful bilateral lifts; 5 cm is a diagnostic threshold, not universal acceptance.

For that single proposal, keep tracking/stopping, posture/body and actuator/contact limits, completed-touchdown hysteresis/dwell, support/failure checks, frozen-base limb attribution, net displacement and bounded time credit. Replace the demand-scaled apex target and its 5–7 cm full-credit band; old instantaneous clearance/airtime bonuses stay disabled. Keep apex/count thresholds, per-limb imbalance and signed preload measurements diagnostic-only, with no equal-step-count rule. CK1999 and A remain fallbacks. The G=0 target-versus-state mismatch remains a separate unresolved control-objective issue, not grounds for silently raising gains or the sagittal scalar.

Executed with the documented Windows Python/cache setup above: `& $Python .migration-audit/rolling-objective-20260928/review.py` (saved-array rates/decomposition, exit **0**); `& $Python .migration-audit/rolling-objective-20260928/counterfactuals.py` (seven CPU cases and arithmetic, exit **0**); `& $Python -m unittest tests.test_go2w_event_step.EventStepCPU.test_correlated_command_refresh_cancels_swing -v` (one test, three timing cases, exit **0**). `& 'C:\Users\Liamb\anaconda3\Scripts\ruff.exe' check tests/test_go2w_event_step.py --select E9,F63,F7,F82`, `git diff --check`, and the 27-file SHA-256 recheck all exit **0**. No test failed or required a rerun. Only this test and this report are tracked changes; no historical suite, learning smoke or screen was repeated.


## Sufficient-clearance consolidation — 29 September 2026

**Partial consolidation; do not replace frozen CK2498.** Repeated useful bilateral steps survive with smaller rear-leg apices, and initial stand, forward tracking and several stop residuals improve. However, +/-0.05 m/s lateral tracking nearly disappears, reverse and moderate-yaw errors increase, and expanded lateral braking travels farther. This is a useful measured tradeoff, not an overall controller improvement or evidence of energy savings. CK2498, CK1999 and A remain available; no further experiment is launched.

Reviewed clean `d2f40e533aba87145573ede30a3b2e1fd66532be`. Implementation is `bedc414`; the smoke test selection repair is `697694ed6a2ff249210664de626467a012e07473`, the clean source used for training and evaluation. The opt-in `--event_quality_profile sufficient_clearance` requires `event_step_v1`; saved evaluation/play contracts reject a missing/different selector or event scale. The only environment-config differences from CK2498 are `rewards.event_step.quality_profile=sufficient_clearance` and `rewards.scales.step_event=.05`. Its exact quality is the broad formula in the preceding section, with discrete `.05*sum(Q*b)` and no dt. Historical unset arithmetic is bit-exact on the focused replay. Stance weight 2.0, all event decisions/guards/credit, sampler, physics, observations, PPO and std projection remain unchanged. Loaded nominal physics and evaluation settings match the parent exactly.

Parent is the exact `sagittal_retention_seed1_20260928_172600_2026-09-28_17-27-35/model_2498.pt`, checkpoint/config hashes verified against the preceding report. The single real output is `logs/go2w_event_step_v1/sufficient_clearance_seed1_20260929_111000_2026-09-29_11-10-47/model_2747.pt`: SHA-256 **`a56063f324f46145c5baeb293bb89eee6b466fbe16a2c792924dc5b5217614db`**; adjacent config **`9b5615803869ddac2e6d82319ff80ed4688b164aec63bcd6f3610f1d844fdbdd`**. Completed count is **2750**, independently stored from native label **2747**; Adam is **110000**, lateral high remains **.50**. Initialization matched actor, critic, both normalizers, raw std, optimizer tensors and LR exactly at completed count 2500/Adam 100000. Loaded/final LR is .00038443359375. Progress was restored before command sampling. This is newly seeded simulator/RNG history, not historical rollout restoration; the smoke was never a parent.

One final `precision_screen` at `evaluation/sufficient_clearance_seed1_20260929_111000/final_screen` contains all seven unchanged schedules, 5650 ticks/113 s, one environment and seed 1, nominal physics, noise/push/brake disabled, 30 s timeout and 50 Hz recording. There are zero falls, resets, timeouts, nonfinite states or non-wheel contacts; no survivors were filtered. Repeated initial stands are not independent trials. All 27 protected reference hashes remain unchanged. Detailed arrays, whole-phase and cross-axis RMS, timing and reconstruction records stay ignored under `.migration-audit/sufficient-clearance-20260929`.

Below, arrows mean **CK2498→CK2747**. Final-second requested-axis means and RMS errors use m/s or rad/s. Cross-axis means follow [vy,yaw] for vx, [vx,vy] for yaw, [vx,yaw] for vy. `*` retains the existing expanded-envelope designation, without inventing an A baseline.

| Command | Requested mean | Requested RMS error | Cross-axis means (axis order above) |
|---|---:|---:|---|
| vx +0.10 | 0.1101→0.0952 | 0.0101→0.0048 | 0.0000/0.0001→-0.0001/-0.0008 |
| vx +0.50 | 0.5101→0.4966 | 0.0101→0.0034 | 0.0000/0.0002→0.0000/0.0002 |
| vx -0.25 | -0.2282→-0.2140 | 0.0218→0.0360 | -0.0003/-0.0003→0.0000/-0.0002 |
| yaw +0.03 | 0.0090→0.0083 | 0.0210→0.0217 | 0.0097/0.0002→0.0027/0.0000 |
| yaw +0.10 | 0.0933→0.1037 | 0.0100→0.0123 | 0.0053/0.0009→-0.0010/-0.0098 |
| yaw +0.40 | 0.4146→0.4295 | 0.0471→0.0697 | -0.0046/-0.0036→0.0045/-0.0116 |
| yaw +0.75 | 0.7354→0.7586 | 0.0710→0.0654 | -0.0064/0.0096→-0.0083/0.0077 |
| yaw -0.03 | -0.0092→-0.0082 | 0.0208→0.0218 | 0.0096/-0.0002→0.0028/-0.0000 |
| yaw -0.10 | -0.0960→-0.1034 | 0.0080→0.0112 | 0.0058/-0.0013→-0.0034/0.0094 |
| yaw -0.40 | -0.4126→-0.4365 | 0.0515→0.0617 | -0.0004/0.0030→0.0160/0.0151 |
| yaw -0.75 | -0.7272→-0.7596 | 0.0743→0.0665 | -0.0130/-0.0143→-0.0130/-0.0044 |
| vy +0.10 | 0.1024→0.1090 | 0.0230→0.0252 | 0.0021/-0.0007→0.0219/-0.0124 |
| vy +0.20 | 0.1864→0.1968 | 0.0262→0.0234 | -0.0093/0.0010→0.0073/-0.0125 |
| vy -0.10 | -0.1009→-0.1042 | 0.0235→0.0268 | 0.0028/0.0016→0.0210/0.0145 |
| vy -0.20 | -0.1891→-0.1953 | 0.0248→0.0234 | -0.0088/0.0019→0.0006/0.0063 |
| vy +0.05* | 0.0471→0.0082 | 0.0149→0.0419 | 0.0228/-0.0104→0.0032/-0.0178 |
| vy +0.40* | 0.3510→0.3721 | 0.0545→0.0367 | 0.0382/-0.0176→0.0121/-0.0454 |
| vy +0.50* | 0.4516→0.4597 | 0.0730→0.0566 | 0.0178/-0.0396→-0.0010/-0.0798 |
| vy -0.05* | -0.0479→-0.0079 | 0.0153→0.0421 | 0.0219/0.0115→0.0033/0.0180 |
| vy -0.40* | -0.3480→-0.3796 | 0.0579→0.0326 | 0.0272/0.0168→0.0107/0.0486 |
| vy -0.50* | -0.4361→-0.4649 | 0.0803→0.0489 | 0.0429/0.0224→-0.0049/0.0615 |

Initial stand's final-second XY/yaw RMS changes .007962/.000334→.001411/.000337; over the full initial 2 s, displacement/path is .01737/.01746→.00708/.01544 m and signed heading -.000505→.000172 rad. Every final stop lasts 3 s below: the first second includes braking; the last second measures residual motion. Whole-stop displacement/path and signed heading include the entire braking trajectory, not just its final residual. Thus better residual yaw does not automatically mean a shorter or straighter stop.

| Schedule | First-second XY/yaw RMS | Final-second XY/yaw RMS | Whole-stop displacement/path, m | Whole-stop heading, rad |
|---|---|---|---|---|
| rolling_reverse | 0.0341/0.0008→0.0292/0.0001 | 0.0094/0.0008→0.0071/0.0000 | 0.0124/0.0381→0.0200/0.0375 | -0.0017→0.0001 |
| yaw_positive | 0.0280/0.0605→0.0194/0.0620 | 0.0099/0.0042→0.0076/0.0030 | 0.0454/0.0592→0.0335/0.0370 | 0.0038→0.0013 |
| yaw_negative | 0.0339/0.0582→0.0291/0.0488 | 0.0054/0.0059→0.0018/0.0034 | 0.0453/0.0605→0.0312/0.0336 | -0.0047→0.0203 |
| lateral_positive | 0.0288/0.0369→0.0311/0.0389 | 0.0059/0.0291→0.0084/0.0047 | 0.0133/0.0376→0.0253/0.0481 | -0.0658→-0.0622 |
| lateral_negative | 0.0268/0.0473→0.0356/0.0423 | 0.0073/0.0076→0.0063/0.0017 | 0.0195/0.0418→0.0238/0.0437 | 0.0370→0.0646 |
| expanded_lateral_positive | 0.0610/0.0780→0.0835/0.0458 | 0.0102/0.0340→0.0126/0.0107 | 0.0107/0.0751→0.0562/0.0902 | -0.1088→-0.0066 |
| expanded_lateral_negative | 0.0643/0.0389→0.0775/0.0382 | 0.0119/0.0224→0.0153/0.0361 | 0.0146/0.0595→0.0550/0.0897 | -0.0192→-0.0371 |

At +.5 vx, mean height falls **.399191→.391899 m** while front-thigh nominal RMS improves .277795→.208792 rad (24.8%). Front targets acquire .04698/.05186 rad lower-bound reserve and 100%/100% lower saturation becomes 0%/0%. All four final-second thighs have zero target-bound occupancy in initial stand, forward/reverse and final rolling stand. This unexpected G=0 change cannot be attributed to direct event payment, which is zero there, or separated from 250 extra updates in this single-parent comparison.

| Limb | +.5 actual thigh q, rad | Applied thigh target, rad | Thigh/calf nominal RMS, rad |
|---|---|---|---|
| FL | .42234→.49068 | .35000→.39698 | .22529→.19318 |
| FR | .42207→.49174 | .35000→.40186 | .22442→.19168 |
| RL | .86464→.85969 | .99284→1.01860 | .14164→.20396 |
| RR | .86253→.85821 | .99098→1.01841 | .14099→.20352 |

Rear sagittal error worsens despite front-thigh improvement. Signed wheel moments in the same +.5 final second remain opposed: FL/FR/RL/RR **[-1.3763,-1.3629,1.3717,1.3699]→[-1.4680,-1.4163,1.4406,1.4496] Nm**. There is no evidence of eliminating preload, electrical savings or actuator causality. Body motion is also mixed: the next table gives maximum phase-final roll/pitch RMS (including stops), and world-vertical velocity RMS pooled over final-second moving-command windows. World vz avoids mistaking a pitched body's forward velocity component for vertical bobbing. Lower rear apices do not consistently reduce vertical motion; lateral stop pitch increases.

| Schedule | Maximum phase-final roll/pitch RMS, rad | Moving final-window world-vz RMS, m/s |
|---|---|---|
| rolling_reverse | 0.0006/0.0444→0.0006/0.0316 | 0.00036→0.00025 |
| yaw_positive | 0.0691/0.0483→0.0481/0.0233 | 0.03264→0.03514 |
| yaw_negative | 0.0696/0.0490→0.0467/0.0216 | 0.03371→0.03381 |
| lateral_positive | 0.0632/0.0498→0.0364/0.0694 | 0.10513→0.10578 |
| lateral_negative | 0.0613/0.0475→0.0376/0.0716 | 0.11393→0.10903 |
| expanded_lateral_positive | 0.0507/0.0301→0.0519/0.0606 | 0.13294→0.13321 |
| expanded_lateral_negative | 0.0578/0.0326→0.0624/0.0545 | 0.13619→0.13077 |

Physical completions and qualified payments remain separate. Apex quantiles below cover completed geometric intervals >=2 mm; usable/net/duration medians cover qualified events only. Historical 2/3/4/5 cm thresholds are unchanged. Rear p90 apices shrink, while front p90 apices increase; stepping is not uniformly smaller or less frequent.

| Limb | Physical >=2/3/4/5 cm | Actual apex p50/p90/max, mm | Paid count | Usable/net/duration medians, mm/mm/s |
|---|---|---|---|---|
| FL | 12/3/0/0→15/7/1/0 | 10.8/25.7/32.0→11.2/31.4/44.5 | 27→25 | 18.37/66.21/0.18→18.37/63.25/0.20 |
| FR | 14/3/0/0→13/7/0/0 | 12.3/26.8/31.6→11.9/32.6/38.4 | 28→25 | 19.24/73.92/0.18→18.59/64.11/0.20 |
| RL | 39/27/24/16→41/26/16/2 | 26.7/60.3/66.6→24.4/45.1/50.9 | 34→32 | 27.38/143.85/0.28→24.29/98.21/0.24 |
| RR | 42/26/24/14→40/25/19/1 | 27.7/58.4/65.6→24.7/46.4/51.6 | 28→34 | 29.53/148.97/0.28→24.66/101.58/0.24 |

At +.20 vy, RR retains five valid 37.9–41.5 mm actual/usable lifts, 138–147 mm net repositioning and .24–.26 s durations, versus five 58.6–65.6 mm parent lifts. At -.20, RL retains five valid 37.3–39.0 mm lifts, 143–148 mm net and .24–.26 s. At +/- .10, the principal rear limb retains four valid 41.3–49.4 mm lifts per direction. The opposite front limb still receives no ordinary-lateral payment; no equality rule was added. In contrast, +/- .05 has **no recorded unloading intervals or reward attempts** and actual vy only +.00815/-.00790. Its unchanged G=1 does not force a gait or guarantee propulsion. Larger lateral requests track their requested axis better but acquire more parasitic yaw.

Full-screen geometric completions are 257→261, load-only intervals 362→352 and physical boundary-censored intervals 28→28. Qualified events are 117→116; weighted totals 5.343807→2.618288 are not comparable returns under the changed objective. Physical >=5 cm completions are 30→3, paid at that actual height 21→0: two new RL events fail sampled support, and the RR event crosses the stop command and is censored. Completed reward attempts are 206→204, flicker cancellations 81→53, command cancellations 14→17, overlong/failure cancellations zero in both. Nonexclusive completion rejections actual/usable height, net displacement, duration and sampled support are **59/59/13/10/30→40/46/8/4/44**. Initial-support/dwell eligibility and physical load-only intervals remain distinct from armed attempts. Exact reconstructed valid/completed/censor decisions match the saved cache; largest geometric discrepancy is 8.2e-8 m and unscaled payment discrepancy 5.4e-7. Guards were not relaxed. Policy-rate support is not a substep stability certificate.

The real job ran **4096×64×250 = 65,536,000 transitions** in **1514.933 s** process wall time (43,260 transitions/s). Entry-point startup is 117.134 s; learning/logging/final save 1382.125 s; imports/shutdown account for the remaining process time. All 250 update labels exist in TensorBoard and diagnostics; none were filled. Matched settled last-100 windows, parent 2399–2498 versus new 2648–2747, have collection/learning **4.7355/.9154→4.5399/.8900 s**, or **46,390→48,278 transitions/s** from mean blocks. Adjacent logged wall intervals are 5.7118→5.4862 s; unassigned host/logging residuals .0624→.0579 s. These are ordinary native timers with different learned states/settings and uncontrolled system load, not a controlled performance gain. No profiling or microbenchmark ran. Evaluation took 291.886 s process wall (256.199 s rollout, .156 s postprocessing). All 76 checkpoint tensors and available losses are finite. Std projection records 71 update-end rows touching the lower bound, zero upper/outside rows; final front-calf std is .10027/.10019, above the floor. No bound or optimizer setting changed.

Validation/process record: focused CPU invocation `-m unittest tests.test_go2w_event_step.SufficientClearanceCPU -v` had two passes and one Windows stderr-handle error (exit **1**) before legacy replay began. Adding `stderr=PIPE` repaired that test only; `...SufficientClearanceCPU.test_legacy_and_bookkeeping -v` exits **0**, with exact historical output and identical new-selector bookkeeping. Numerical checks cover 2/2.5/3/5/7 cm, dt .01/.02/.04, unit credit, invalid bob/drag/chatter/failure/command cases and narrow config contracts. The first smoke invocation exited **0 but skipped** due to an inherited unittest flag: no scene or updates ran. The one-line selection repair preceded the actual `-m unittest tests.test_go2w_event_step.SufficientClearanceSmoke -v`, exit **0**, 97.179 s process wall: 64×64×2, 128 policy ticks/8192 transitions, 80 Adam steps, counter 2502/Adam 100080, finite state/gradients/losses, once-only event updates, retained hook and native save/reload. Both original logs are preserved. One real training process and one final-screen process exit **0/0**. Checkpoint verification, saved-screen analysis and protected-file verification exit **0**. Targeted Ruff (`E9,F63,F7,F82`) on the six changed Python files and `git diff --check` exit **0**. No behavioral tuning or simulator retry followed a result.

This is not a same-budget unchanged-control ablation. The result supports smaller useful rear steps in common lateral motion, but does not establish a generally better rolling-first policy, balanced coordination, improved energy use or robustness. The low-lateral loss and reverse/moderate-yaw regressions prevent promotion. It does not prove that a fresh policy could never succeed under this objective. No further training, smoke, screen, bank, export, GUI, transfer or hardware activity was run.

Exact executed commands, using the existing cache setup (the real run remains tied to original CK2498):

```powershell
Set-Location 'C:\Users\Liamb\SynologyDrive\TUM\3_Semester\dodo_alive\legged-robot_rl_genesis'
$Python = 'C:\Users\Liamb\anaconda3\envs\genesis-gpu\python.exe'
$env:NUMBA_CACHE_DIR = "$PWD/.migration-audit/diagnostics-20260925/numba-cache"
$env:GS_CACHE_FILE_PATH = "$env:TEMP/go2w-diagnostics-genesis"
$env:QD_OFFLINE_CACHE_FILE_PATH = "$env:TEMP/go2w-diagnostics-quadrants"
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
$env:GO2W_EVENT_GPU = 'sufficient'
& $Python -m unittest tests.test_go2w_event_step.SufficientClearanceSmoke -v
$Parent = "$PWD/logs/go2w_event_step_v1/sagittal_retention_seed1_20260928_172600_2026-09-28_17-27-35"
& $Python -m robot_gym.scripts.train --task go2w --go2w_profile event_step_v1 --sagittal_stance_weight 2.0 --event_quality_profile sufficient_clearance --experiment_name go2w_event_step_v1 --load_run $Parent --checkpoint 2498 --reference_config "$Parent/config.yaml" --resume --run_name sufficient_clearance_seed1_20260929_111000 --num_envs 4096 --max_iterations 250 --seed 1 --headless --logger tensorboard --training_diagnostics
$Run = "$PWD/logs/go2w_event_step_v1/sufficient_clearance_seed1_20260929_111000_2026-09-29_11-10-47"
& $Python -m robot_gym.scripts.evaluate --task go2w --go2w_profile event_step_v1 --sagittal_stance_weight 2.0 --event_quality_profile sufficient_clearance --experiment_name go2w_event_step_v1 --load_run $Run --checkpoint 2747 --reference_config "$Run/config.yaml" --eval_mode precision_screen --num_envs 1 --seed 1 --headless --logger tensorboard --output evaluation/sufficient_clearance_seed1_20260929_111000/final_screen
```

The ignored `train_once.ps1`/`evaluate_once.ps1` wrappers record exits/wall times and refuse occupied run/output paths. Commands above document the completed runs; do not overwrite their outputs. Optional GUI playback, **not launched**, with the same variables/cache setup:

```powershell
& $Python -m robot_gym.scripts.play --task go2w --go2w_profile event_step_v1 --sagittal_stance_weight 2.0 --event_quality_profile sufficient_clearance --experiment_name go2w_event_step_v1 --load_run $Run --checkpoint 2747 --reference_config "$Run/config.yaml" --num_envs 1 --seed 1 --steps 750 --command_vx 0.5 --command_vy 0 --command_yaw 0 --no_export
```


## Frozen feedback candidates and portable inference — 29 September 2026

**Keep P (CK2498) primary for controlled Linux comparison in this tested envelope; retain C (CK2747) as an alternative.** Both finish without failure or an observed hard-position violation. P has lower path/endpoint error with smaller corrections; C reduces moving yaw-rate variability and force peaks and retains straight front-thigh target reserve. C's weak small-lateral response still limits endpoint correction. Neither is an all-command, PhysX or hardware qualification; no controller gains or behavior changed.

Started from clean public 318197eb5bc61c4c8039c30876890687c4fa46bf. Recorder source is clean e08bca9887b8fd65ce995d89221aded0e5bd6e28: only pre-step actor inputs and saved actual position/effort limits were added to the existing lightweight closed-loop record, plus stopping subsequent cases after the first failure. Existing q, policy-rate forces, terminal-before-reset capture and controller calculations are unchanged. P is exactly sagittal_retention_seed1_20260928_172600_2026-09-28_17-27-35/model_2498.pt, SHA-256 4de494271a999c33ca6a8a06913b1facb92eede58c92d7297762671b2b1080e8, config 340a8998d3015b810c46e4d808cfb173837d20aa2a2e1028a6004e198b8ff25d. C is exactly sufficient_clearance_seed1_20260929_111000_2026-09-29_11-10-47/model_2747.pt, SHA-256 a56063f324f46145c5baeb293bb89eee6b466fbe16a2c792924dc5b5217614db, config 9b5615803869ddac2e6d82319ff80ed4688b164aec63bcd6f3610f1d844fdbdd. Both use event_step_v1, stance 2.0; only C selects sufficient_clearance. URDF SHA-256 remains 794ad4adaec16bc7a1eebab3d838e1955f37113192689361aff4733e99c9e9c5. All 39 protected inputs, including CK1999, A and archived screens, remain unchanged.

Cheap replay of the archived pure-lateral arrays confirms **actual** upper calf gaps, not soft penalties or action-target reserve: P FR at +.2, phase 2, 6.48 s has q=-.844524682 versus upper -.837759972, gap **.006764710 rad**; mirrored FL at -.2, 6.48 s has q=-.848728955, gap **.010968983**. C FR at +.2, 6.58 s is -.872869551, gap **.035109580**; FL at -.2, 6.60 s is -.875810862, gap **.038050890**. These minima occur during the transition into +/- .2 after 6 s. No crossing was recorded. Saved hard limits [-2.722700119,-.837759972] were used; 50 Hz samples cannot exclude between-sample violations.

The unchanged experiment is straight [.5,0] and diagonals [.5,+/-.1], each 2 s settle, 12 s trapezoidal path with 1 s ramps, 3 s endpoint feedback and 3 s exact zero. Nominal properties match exactly between P/C; controller/schedule metadata match archived A. A was not rerun. Reference position/heading anchor once after settling; world feedback is transformed with the full inverse wxyz quaternion into body [vx,vy,yaw_rate]. Gains remain 1.0/1.5 per second, 10 Hz sample/hold, 50 Hz policy, four 5 ms physics steps. Bounds remain vx [-.25,.70], vy [-.20,.20], yaw [-.50,.50]. Seed 1, one environment, deterministic actor, noise/DR/push/brake off, 30 s timeout. Event command censoring remains training-reward bookkeeping and does not change frozen actor inference.

Errors below use the established **2–20 s** window, including feedback removal; endpoint is at 17 s. Settling is excluded from anchored errors. No survivor filtering or missing cases: all six have 1000 rows, finite observations/actions/states, zero falls/resets/timeouts/non-wheel contacts. No heavy recorder, video, contact dump or substep trace ran.

| Candidate / case | Position RMS/p95/max, mm | Cross-track RMS/p95/max, mm | Heading RMS/p95/max, deg | Endpoint, mm |
|---|---|---|---|---|
| P straight | 16.28/30.82/35.10 | 1.07/1.77/1.92 | 0.041/0.097/0.153 | 10.87 |
| P diagonal_left | 19.45/35.80/47.53 | 11.85/18.94/19.39 | 1.266/3.520/3.590 | 7.58 |
| P diagonal_right | 19.50/35.80/47.29 | 11.43/18.17/19.15 | 1.256/3.616/3.725 | 8.50 |
| C straight | 19.45/39.27/46.84 | 1.99/3.82/3.96 | 0.081/0.185/0.200 | 14.03 |
| C diagonal_left | 21.04/38.90/48.18 | 12.29/27.95/30.02 | 1.425/3.778/3.871 | 14.91 |
| C diagonal_right | 21.18/38.97/49.71 | 12.18/27.59/29.31 | 1.585/4.199/4.229 | 14.69 |

During feedback alone (2–17 s), maximum heading error is P .153/.866/.783 degrees versus C .200/1.359/1.598. Neither controller clips or touches a command bound. Mean requested/measured body velocities and tracking RMS include ramps and endpoint hold; columns are vx/vy/yaw in m/s, m/s, rad/s. Position/yaw correction peaks use m/s and rad/s respectively.

| Candidate / case | Feedback mean command vx/vy/yaw | Measured mean vx/vy/yaw | Tracking RMS vx/vy/yaw | Peak position/yaw correction |
|---|---|---|---|---|
| P straight | 0.3587/-0.0009/-0.0005 | 0.3671/-0.0000/0.0000 | 0.0130/0.0010/0.0016 | 0.0346/0.0040 |
| P diagonal_left | 0.3668/0.0657/0.0060 | 0.3667/0.0751/-0.0007 | 0.0247/0.0238/0.0336 | 0.0462/0.0227 |
| P diagonal_right | 0.3665/-0.0663/-0.0063 | 0.3668/-0.0758/0.0009 | 0.0246/0.0236/0.0331 | 0.0460/0.0205 |
| C straight | 0.3673/-0.0015/-0.0013 | 0.3675/-0.0001/0.0001 | 0.0142/0.0016/0.0021 | 0.0405/0.0052 |
| C diagonal_left | 0.3559/0.0753/0.0085 | 0.3662/0.0741/-0.0014 | 0.0237/0.0209/0.0304 | 0.0480/0.0356 |
| C diagonal_right | 0.3561/-0.0749/-0.0084 | 0.3663/-0.0739/0.0019 | 0.0235/0.0207/0.0312 | 0.0496/0.0418 |

The established late hold is 15–17 s. P responds weakly to millimetres-per-second lateral corrections; C's signed lateral response is opposite its requested correction in both diagonal late holds. Both also have wrong-sign mean yaw response there. This is a residual correction limit, not proof of an exact deadband threshold. The existing .01 analysis tolerance detects no late-hold command or velocity reversals; it is not an actor deadzone. Moving diagonal yaw standard deviations are P .03689/.03654 versus C .03030/.03063 rad/s, with 131/129 versus 82/77 sign reversals above .01 over 12 s. These sampled oscillations are retained; no growing late-hold oscillation was observed, and no causal attribution to gait or feedback is established.

| Candidate / case | Late-hold command vx/vy/yaw | Measured vx/vy/yaw | Command/velocity reversals above .01 | Heading range, deg |
|---|---|---|---|---|
| P straight | -0.0148/-0.0011/-0.0017 | -0.0048/-0.0001/0.0005 | [0, 0, 0]/[0, 0, 0] | 0.196 |
| P diagonal_left | -0.0115/0.0040/0.0138 | -0.0054/0.0009/-0.0034 | [0, 0, 0]/[0, 0, 0] | 0.400 |
| P diagonal_right | -0.0124/-0.0038/-0.0115 | -0.0053/-0.0008/0.0032 | [0, 0, 0]/[0, 0, 0] | 0.370 |
| C straight | -0.0166/-0.0020/-0.0015 | -0.0040/-0.0001/-0.0001 | [0, 0, 0]/[0, 0, 0] | 0.015 |
| C diagonal_left | -0.0126/0.0117/0.0326 | -0.0077/-0.0016/-0.0027 | [0, 0, 0]/[0, 0, 0] | 0.415 |
| C diagonal_right | -0.0140/-0.0114/-0.0334 | -0.0081/0.0004/0.0053 | [0, 0, 0]/[0, 0, 0] | 0.605 |

Exact-zero summaries retain both the first braking second and last residual second. Whole-stop displacement/path and signed heading span the full 17–20 s interval, starting at the pre-zero boundary. Removing feedback produces several degrees of diagonal heading change even though final yaw-rate RMS is small.

| Candidate / case | First-zero-second XY/yaw RMS | Last-zero-second XY/yaw RMS | Whole-stop displacement/path, mm | Signed heading, deg |
|---|---|---|---|---|
| P straight | 0.00746/0.00065 | 0.00716/0.00077 | 21.98/22.01 | -0.125 |
| P diagonal_left | 0.00471/0.03767 | 0.00517/0.00230 | 14.65/16.41 | -2.665 |
| P diagonal_right | 0.00519/0.03590 | 0.00574/0.00253 | 15.88/17.89 | 3.035 |
| C straight | 0.01131/0.00096 | 0.01064/0.00013 | 32.85/32.89 | 0.085 |
| C diagonal_left | 0.00978/0.03333 | 0.00534/0.00231 | 19.74/21.25 | -2.627 |
| C diagonal_right | 0.00988/0.03364 | 0.00573/0.00099 | 20.84/22.56 | 2.598 |

Actual position margins and force ratios below use all recorded samples. Calf minima are upper-limit gaps; diagonal commands at P minima are [.52307,.09805,.00052] / [.51969,-.10145,-.00891], versus C [.48908,.10520,.00928] / [.49017,-.10443,-.00734]. Thus C's larger margin in pure-lateral screen transitions does **not** generalize to these diagonals: here its smallest gap is .127109 versus P .158113 rad. Both remain positive. Force ratios use the unchanged per-joint effort limits, not action bounds; no sample reaches 99%. They are current-state policy-rate control-force readings, not substep maxima, electrical power or measured ground propulsion. Cruise target statistics use the existing second_12 window (12–13 s, before deceleration).

| Candidate / case | Minimum actual gap: joint, time, rad | Maximum force ratio: joint, time | Cruise front-thigh lower clipping FL/FR | Target reserve FL/FR, rad |
|---|---|---|---|---|
| P straight | RR_calf_joint, 0.02, 0.465064 | 0.4766, FR_thigh_joint, 0.08 | 100.0/100.0% | 0.0000/0.0000 |
| P diagonal_left | FR_calf_joint, 4.00, 0.160551 | 0.7058, FL_hip_joint, 3.84 | 0.0/8.0% | 0.1956/0.2477 |
| P diagonal_right | FL_calf_joint, 4.04, 0.158113 | 0.6891, FR_hip_joint, 10.32 | 10.0/0.0% | 0.2542/0.1991 |
| C straight | RR_calf_joint, 0.04, 0.454322 | 0.4643, FL_thigh_joint, 0.08 | 0.0/0.0% | 0.0488/0.0557 |
| C diagonal_left | FR_calf_joint, 4.58, 0.127109 | 0.6196, FR_thigh_joint, 4.58 | 0.0/4.0% | 0.1907/0.3644 |
| C diagonal_right | FL_calf_joint, 4.60, 0.131800 | 0.6209, FL_thigh_joint, 4.60 | 2.0/0.0% | 0.3534/0.1867 |

Maximum whole-case raw-action clipping fractions are P 49.0% (FL thigh, straight), 13.6% (FL hip, left), 13.4% (FR hip, right); C 0%, 17.6% (FL hip, left), 17.1% (FR hip, right). These target limits are separate from actual mechanical position gaps and force limits. P's straight front-thigh target reserve is still absent. The prior opposing wheel moments and open-loop weaknesses remain applicable; this probe does not erase them.

**Selection limits:** use P first for a controlled, nominal flat-ground Linux reproduction of these .5 m/s straight/+/-.1 diagonal paths with the existing feedback bounds and stop phases. This is a comparison starting point, not a deployment envelope. Do not extrapolate it to pure-lateral .2 transitions (P's archived calf gap is only .006765 rad), stronger yaw, rough terrain, delays, estimator errors or hardware. Keep C to compare its straight target reserve, smaller force peaks and quieter moving yaw against its weaker fine corrections; neither dominates every property. Frozen A/CK1999 remain fallbacks. No cross-policy switching, new heading objective or learned navigation controller was added.

Both failure-free candidates received separate, non-overwriting bundles using existing robot_gym.utils.export.export_policy, offline from the exact checkpoint and saved/read-back interface. Under logs/go2w_event_step_v1/exported/:

- `feedback_P_ck2498_4de494271a99_20260929/` and matching .zip: archive SHA-256 `e3b5b9d689d94c57464310072c19181695f785558ebc4d7cdbe6bfb7af5778d1`; policy_1.pt SHA-256 `cb8454470921a5a44e8a5914a0a2dd301886ff9b7e7b6ef1dffdf5c6b661a35c`.
- `feedback_C_ck2747_a56063f324f4_20260929/` and matching .zip: archive SHA-256 `e1e20870f58509ec27c8f7d2e3c3f9721c19e27a9657da170d4e3cd3f87a8447`; policy_1.pt SHA-256 `ae7c6747cf863aa0ab5cf48a45d3f579afc50465c9060c0414c506edc4125a2b`.

Each bundle contains the frozen TorchScript actor, exact config and URDF bytes, training/evaluation source manifests, hashes, 96-observation parity batch and a detailed separate low-level contract.json. The contract gives 56-vector slices/units/scales, trained normalization embedded exactly once, body axes and wxyz origin conventions, reset/previous-issued-clipped-action semantics, named joint/action order, P legs/V wheels, .30/.35/.40 rad and 18 rad/s scales, 40/1 leg gains and wheel Kv=1, actual limits, .02/.005 s dt/decimation, and three-slot actuator-delay history distinct from observation latency. No heading controller or event tracker is embedded. Simulator base-link linear velocity remains an explicit future hardware **estimator requirement**. Referenced visual meshes and an IsaacLab scene are not supplied; this is experimental inference, not a hardware release.

Parity uses 32 fixed evenly spaced actual inputs per case (96 per policy), frozen buffers and no optimizer. Native CPU raw actor versus recorded GPU maximum absolute differences are **4.77e-7 P / 5.96e-7 C** (declared absolute tolerance 5e-5). Reloaded exported versus native CPU bounded outputs are **exactly equal** (tolerance 2e-6); versus recorded clipped outputs, **4.17e-7 / 5.96e-7**. Every native state tensor and copied normalization buffer is unchanged; counts remain 655360000/720896000. Input maxima 14.61/14.80 stay below the existing observation clip 100. PyTorch-only loading is demonstrated on this Windows CPU; Linux/PhysX compatibility and state estimation remain untested.

Execution: two simulator processes, **exit 0/0**, no retry. P started 12:45:36.552 and ended 12:48:12.734 CEST, wall **156.165 s**; C 12:49:41.876–12:52:22.274, **160.385 s**. Internal startup/rollout/postprocessing/shutdown seconds are P **33.787/121.290/.090/.051**, C **31.795/127.312/.091/.053**. There are 6000 recorded ticks/120 s of scheduled cases. Accounting caveat: retaining the stipulated existing reset behavior also executes one zero-action setup tick and three reset ticks per process, **.16 s combined**; literal total integration is therefore 120.16 s, slightly above the stated 120 s ceiling. No additional case or physical test ran.

Two focused CPU tests (reference/frames; extended recording/first-failure censoring) pass, exit **0**. Synthetic failure rows in that test are not simulator failures. Offline preflight, analysis, table generation and file/ZIP verification exit **0**. First offline export invocation exited **1** on a local-script repository import path before any bundle or scene existed; adding the repository to that script's import path repaired it, and export/parity exited **0**. Both logs are preserved. No simulator rerun, PPO/smoke, precision/sustained/bank repeat, new objective, profiling, package installation, Linux transfer, GUI or hardware command occurred. Raw traces remain at evaluation/feedback_candidates_20260929/{P_ck2498,C_ck2747}; verbose records/scripts are ignored under .migration-audit/feedback-candidates-20260929/. Tracked changes are only the small recorder/test extension and this report.

Targeted Ruff (`E9,F63,F7,F82`) on diagnostic_bank.py/test_inference.py and `git diff --check` exit **0**. No unrelated files changed.

Exact executed evaluation commands (outputs are occupied; do not overwrite). The ignored run_once.ps1 -Candidate P / -Candidate C wrappers only guard existing paths and print/record start, end, exit and process wall time:

```powershell
Set-Location 'C:\Users\Liamb\SynologyDrive\TUM\3_Semester\dodo_alive\legged-robot_rl_genesis'
$Python = 'C:\Users\Liamb\anaconda3\envs\genesis-gpu\python.exe'
$env:NUMBA_CACHE_DIR = "$PWD/.migration-audit/diagnostics-20260925/numba-cache"
$env:GS_CACHE_FILE_PATH = "$env:TEMP/go2w-diagnostics-genesis"
$env:QD_OFFLINE_CACHE_FILE_PATH = "$env:TEMP/go2w-diagnostics-quadrants"
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
$P = "$PWD/logs/go2w_event_step_v1/sagittal_retention_seed1_20260928_172600_2026-09-28_17-27-35"
$C = "$PWD/logs/go2w_event_step_v1/sufficient_clearance_seed1_20260929_111000_2026-09-29_11-10-47"
& $Python -m robot_gym.scripts.evaluate --task go2w --go2w_profile event_step_v1 --sagittal_stance_weight 2.0 --experiment_name go2w_event_step_v1 --load_run $P --checkpoint 2498 --reference_config "$P/config.yaml" --eval_mode closed_loop --num_envs 1 --seed 1 --headless --logger tensorboard --output evaluation/feedback_candidates_20260929/P_ck2498
& $Python -m robot_gym.scripts.evaluate --task go2w --go2w_profile event_step_v1 --sagittal_stance_weight 2.0 --event_quality_profile sufficient_clearance --experiment_name go2w_event_step_v1 --load_run $C --checkpoint 2747 --reference_config "$C/config.yaml" --eval_mode closed_loop --num_envs 1 --seed 1 --headless --logger tensorboard --output evaluation/feedback_candidates_20260929/C_ck2747
& $Python -m unittest tests.test_inference.InferenceTests.test_closed_loop_hold_censoring_and_light_recording tests.test_inference.InferenceTests.test_closed_loop_reference_and_frames -v
& $Python .migration-audit/feedback-candidates-20260929/export_bundles.py
```

The last command documents the completed offline export and now refuses the occupied bundle paths. Detailed analysis ran preflight.py, analyze.py, tables.py and verify_files.py from the same ignored directory with the documented Python. No further training or integration is launched.

## Published P inference snapshot — 29 September 2026

Published the selected P/CK2498 at the stable [ressources/pretrained/go2w](../ressources/pretrained/go2w/README.md) path, starting from clean testing at 4df11998f7d9424090dab1bcfe50020297d67bbd. This is a mutable experimental simulation candidate, not a hardware release. The original policy_1.pt was copied unchanged as policy.pt: **800,038 bytes**, SHA-256 **cb8454470921a5a44e8a5914a0a2dd301886ff9b7e7b6ef1dffdf5c6b661a35c**. Source CK2498 remains **4de494271a999c33ca6a8a06913b1facb92eede58c92d7297762671b2b1080e8**, completed count 2500; the exact saved config remains **340a8998d3015b810c46e4d808cfb173837d20aa2a2e1028a6004e198b8ff25d**. Source ZIP identity/content and every bundled checksum passed before copying. No re-export, different policy or optimizer checkpoint was substituted.

The seven files are policy.pt 800038 B, contract.json 18545 B, config.yaml 13247 B, provenance.json 13361 B, parity_samples.npz 30125 B, check_policy.py 12678 B and README.md 5715 B. The contract extends the existing schema with portable paths, explicit dtype/batch shape, origin transport, limit semantics and separate training/controller/test command coverage. Motor velocity metadata is not misrepresented as a hard leg-speed clamp. The 96 float32 inputs/native CPU bounded references are preserved exactly; recorded GPU raw outputs are separately named numeric arrays. The checker uses only standard Python, PyTorch and NumPy, checks hashes before model loading, and needs no training-package import.

Canonical assets are reused in place: the URDF and all 13 referenced meshes exist with exact case; their DAE files reference no external texture. No geometry, inertia, axis, limit or asset byte was edited. The trained Windows URDF SHA-256 is **794ad4adaec16bc7a1eebab3d838e1955f37113192689361aff4733e99c9e9c5**; its Git LF blob is **495401f076d408858f322dd5227f3166be4fd63e65b8bc648fcbdeffd4c98d44**. They differ only by CRLF/LF. Asset integrity permits only that newline substitution for XML/DAE and requires exact binary mesh bytes. Narrow .gitattributes entries preserve public actor/config/NPZ bytes and LF public documentation/code/JSON; no repository renormalization occurred.

The published command, **python ressources/pretrained/go2w/check_policy.py**, exited **0** once on the final worktree package. A second invocation on a 9,223,263-byte temporary archive of staged tree 9c530f669a2a8ef209caf0dfb4cc6f68fb107dcd plus only required assets also exited **0**, using isolated Python and no ignored-file dependencies. Both checks: 96 samples, **0** maximum CPU reference difference (atol 2e-6), **4.17232513e-7** against clipped GPU-recorded outputs (atol 5e-5), unchanged normalizer buffers. Environment: Windows, Python 3.11.14, PyTorch 2.9.0+cu130, NumPy 2.2.6. Payloads are real Git blobs, not LFS pointers. Targeted Ruff and whitespace checks passed. All 65 protected source/evidence/bundle files remain unchanged.

One local preparation assertion initially exited **1** before copying payloads: it compared the whole saved observation-scale dictionary to the four actor scales, overlooking the unused height_measurements entry. Comparing the actual four interface fields repaired the packaging assertion; preparation exited **0**, with no semantic change. Staged whitespace checking initially exited **2** on the deliberately preserved config CRLF endings; a config-only cr-at-eol attribute fixes that check without altering the snapshot bytes. Logs remain ignored under .migration-audit/publish-go2w-20260929. No simulator, optimizer, new trajectory, installation, transfer or hardware command ran. Linux/PyTorch compatibility and PhysX behavior remain pending. The next bounded Linux task is to fetch and run the public checker first, then reproduce the frozen nominal scene and the existing straight/[.5,+/-.1] feedback paths; do not extrapolate to a deployment envelope.

## Native frozen-Actor reference — 1 October 2026

Started on clean `testing` at `c2c2febed2fc4aea20d874876091a3377f2e9ed2`,
origin/testing, ahead/behind 0/0. The existing `genesis-gpu` environment matches
recorded Python 3.11.14, Torch 2.9.0+cu130, Genesis 1.4.1 and RSL-RL 5.5.1.
The [portable JSON](go2w_native_reference.json) gives all exact code/package/
Actor/asset identities, joint readback and metrics; the [3.7 MB numeric archive](go2w_native_reference_traces.npz)
contains 200 Hz physics states/targets and 50 Hz Actor inputs/actions.

The published P/CK2498 Actor is unchanged and loaded directly, with its embedded
normalizer/clipping. Existing observation/control/reset/command paths are reused.
The complete saved configuration is restored after the existing contract check;
all diagnostic overrides are logged. Saved performance_mode=True,
deterministic=False and batching are retained; normal play overrides the first
two. There is no runner, export, feedback, brake, smoothing, DR, noise or push.

Effective armature is **0.10000000149 kg m² at all 16 joints**, after build/reset.
Active gains after control initialization/reset: legs Kp=40 Nm/rad, Kv=1 Nm s/rad;
wheels Kp=0, Kv=1. Passive stiffness/damping/frictionloss are zero. Bare build
defaults Kp=100/Kv=10 are replaced before inference. Integrator
approximate_implicitfast, Newton 50/50 iterations; .005 s physics/one internal
substep, .02 s policy/decimation four, delay zero, collisions and joint limits on.
No global MuJoCo compatibility option changed. Current readback/source hashes
cannot prove byte identity of the unrecorded historical installation.

Explicit ORIGINAL snapshot `go2w_reference_original.urdf` matches publication
commit `0ca8d747a94341ed94d59075c2899c5348de688a` (LF versus historical CRLF).
MEASURED `go2w_measured_ed8dc93.urdf` exactly matches navigation commit
`ed8dc93b3065a8e2a3a5919f032ed4690529117c`, including Git blob identity.
All 13 shared meshes match its LFS hashes, permitting only DAE CRLF→LF.
Canonical files/publication hashes remain unchanged; the unchanged checker used
an isolated ORIGINAL root. Ambiguous basename fallback now fails.
Total masses are 19.523000/19.683710 kg; merged base masses 6.923000/7.083710 kg.
JSON records COM and inertia in principal and authored base frames. Camera/mount
geometry/inertials and tire radius .086→.09167 m change together.

| Exact-zero 10 s case | Last 2 s planar RMS, m/s | Last 2 s yaw RMS, rad/s | Last 5 s drift, m | 2–10 s leg dq RMS, rad/s | 2–10 s body roll/pitch rate RMS, rad/s |
|---|---:|---:|---:|---:|---:|
| ORIGINAL, source armature | .009971 | .000871 | .051004 | .003605 | .001927 |
| MEASURED, source armature | .010069 | .000793 | .051976 | .003089 | .001728 |
| ORIGINAL, only armature zero | .216402 | .324413 | 1.124184 | 4.388531 | 3.661521 |

All five cases complete with credible support and no fall, timeout, rollout
reset or incomplete interval. The public armature setter verifies zero before
and after fresh reset. Zero-armature joint/body spectra peak at 25 Hz, with
84.4%/93.0% power in 24–26 Hz; Actor peak is its 25 Hz Nyquist bin (86.0%).
Maximum per-joint clipping is 79%; actual minimum leg margin stays .162569 rad.
Source-armature stands pass all three development targets; zero armature fails
all three. Initial 0–2 s and settled 2–10 s results are separate in JSON.

Both fresh sequences use zero 0–3 s, +.2 m/s 3–8 s, zero 8–14 s.
ORIGINAL/MEASURED mean vx over 5–8 s: .186334/.188612 m/s; tracking RMS:
.014213/.011600 m/s; body-forward progress: .960430/.972647 m.
Stop-edge-to-end displacement: .060951/.059478 m; final planar RMS:
.008207/.007924 m/s. No gain/action-scale search ran.

**Budget/checks:** 5 native starts/5 rollouts, 58 s scheduled +.20 s reset
integration, 646.027 s process wall including compilation/GUI. A conservative
600 s allowance for cancelled/completed import-only inspection keeps total below
30 minutes. 30 CPU tests, frozen replay of all 2,900 observations (max 3.10e-6),
integrity/history/command checks, py_compile, existing base Ruff 0.12.0 and
whitespace checks pass. Initial test import order and an offline Windows Git
stderr-handle error were corrected without native reruns. The sole normal viewer
opened 960×720/60 Hz and completed the measured sequence, exit 0; no screenshot
was captured or GUI retry made. Raw logs/evidence:
`evaluation/native_reference_20261001`, `.migration-audit/native-reference-20261001`.

**One next action:** matched PhysX exact-zero ORIGINAL armature A/B, 0.0 versus
**0.1 kg m² at all 16 joints**, fresh 10 s resets, same Actor/gains/limits/
timing/history/ground and verified backend readback. No USD degree conversion
on API gains or armature. Historical measured-asset PhysX calibration is context,
not a matched baseline. This proves a Genesis armature dependency; it does not
prove the sole PhysX cause, hardware parameters, transfer success or training
necessity. Enough evidence exists; the optional measured zero-armature run was skipped.

Playback from the repository root, using the verified existing environment and
fresh outputs; these commands perform no export:

```powershell
$go2wPrefix = (conda env list --json | ConvertFrom-Json).envs | Where-Object { (Split-Path $_ -Leaf) -eq 'genesis-gpu' }
$Python = Join-Path $go2wPrefix 'python.exe'
$env:NUMBA_CACHE_DIR = "$PWD/.migration-audit/diagnostics-20260925/numba-cache"
$env:GS_CACHE_FILE_PATH = "$env:TEMP/go2w-diagnostics-genesis"
$env:QD_OFFLINE_CACHE_FILE_PATH = "$env:TEMP/go2w-diagnostics-quadrants"
$env:PYTHONIOENCODING = 'utf-8'
$Replay = "evaluation/native_reference_replay_$(Get-Date -Format yyyyMMdd_HHmmss)"
& $Python -m robot_gym.scripts.native_reference --candidate P_ck2498 --asset original --case sequence --output "${Replay}_original"
& $Python -m robot_gym.scripts.native_reference --candidate P_ck2498 --asset measured --case sequence --output "${Replay}_measured" --viewer
```

Omit `--viewer` for headless playback. `--case stand` selects 10 s exact zero;
`--zero-armature` is restricted to stand and verifies effective zero.
Default training recipes remain unchanged.

## Fresh measured-model transfer_v1 preparation — 2026-10-01

`--go2w_profile transfer_v1`, experiment `go2w_transfer_v1`, is a **fresh**
random-initialized locomotion recipe. It reuses P/CK2498's event behavior,
commands, rewards and PPO design; it never reads its learning state. Training
rejects resume/load/checkpoint, continuation and inference-dynamics arguments
before scene/runner construction. No long run, Isaac execution or hardware work
was performed. Historical bundles, assets and results remain unchanged.
The checkout started clean on `testing` at `372dd1de5cc8f4cb135b3f475f96de1a38384d58`.
The read-only Isaac integration branch remains at
`d33b88309ed88796e255380988c6dce8bbd10514`; its latest report and actual
`make_articulation_cfg()` were inspected.

The resolved recipe is:

Changed paths stay focused: `go2w_config.py` selects/validates the measured recipe;
`go2w_env.py` maps motor dynamics and shares event behavior; base `legged_robot.py`
passes explicit import/integrator intent and scales mass/inertia through the public
API. `step_events.py`/`task_registry.py` share existing update/std/serialization
hooks; `helpers.py`/`train.py` add fresh-only guards and readback. Existing `play.py`,
`evaluate.py`, `diagnostic_bank.py`, `export.py` and `diagnostics.py` supply explicit
inference settings, bounded panels and nominal-versus-runtime metadata.
`tests/test_go2w_transfer.py` and this section provide focused checks/handoff.

| Item | transfer_v1 |
|---|---|
| Actor / Critic / actions | **56 / 56 / 16**, both consume `policy`; no privileged observations, estimator or estimator history |
| Networks | Separate MLPs and fresh observation normalizers; `[512,256,128]`, ELU |
| Gaussian | Log std, initial .40, projection [.10,.70], learned std; entropy .003 |
| PPO | Initial LR 3e-4, adaptive KL .01, gamma .995, lambda .95, clip .2; value coefficient 1, clipped value loss, 5 epochs, 8 minibatches, gradient bound 1; existing sagittal augmentation |
| Run | Seed 1, 4096 environments, 64 rollout steps/environment, 2000 updates, save every 100 |
| Command/reward behavior | P's complete mixture, 3–6 s long stands, 8–15 s long moving segments, command-gated discrete events; stance weight 2, x/y tracking denominators .25/.04, event weight .15; ordinary rewards retain dt scaling |
| Plant | Measured `go2w_measured_ed8dc93.urdf`, SHA-256 `d298cc7bf4894e869840bdab9ac60f09548d998444018e61854636cff46b1d8c`, navigation source `ed8dc93b3065a8e2a3a5919f032ed4690529117c` |
| Armature | Nominal .01 kg m² at each of 16 mapped motor joints; training uniform [.005,.02], independent leg/wheel group draws per environment, left/right symmetric, constant across episodes; floating base excluded |
| Actuation | Leg Kp/Kd 40/1, wheel Kp/Kv 0/1; passive stiffness/damping/frictionloss zero; existing 23.7/35.55 Nm efforts and speed semantics; leg action scales .30/.35/.40 rad, wheel 18 rad/s; issued actions clipped ±1 |
| Timing | Policy .02 s, physics .005 s, decimation 4, one internal substep; effective `approximate_implicitfast`, Newton, 50 iterations/50 line-search iterations; targets held across physics steps |
| Randomization | P's friction [.6,1.2], gains [.85,1.15], base mass delta [−.5,1.5] kg, separate COM shift ±.015 m, reset noise and pushes retained; action delay 0–2 policy steps = 0–40 ms |

The measured file/resources, joint order and geometry are verified before building.
Wheel radius/contact-height metadata now follows its .09167 m cylinders; the
existing oriented-cylinder helper supplies event/evaluation clearance. Authored
total mass is 19.68371 kg. Nominal-pose FK requires base heights .437519 m front,
.439338 m rear; spawn .45 m leaves 10.66–12.48 mm clearance. Fixed-target loaded
height is .4112–.4114 m, so the **.415 m reward reference and nominal angles remain**.
Merged base mass 7.083710 kg, COM approximately
`[.0280210, −.000002483, −.00278850]` m and principal inertias
`[.12175738, .11458107, .02654729]` kg m² match the retained measured reference.
Sensor/mount mass is already included. Base-mass uncertainty calls the installed
**public** `RigidSolver.set_links_mass(..., scale_inertia=True)` once relative to
the unchanged nominal body; the entity setter alone does not scale inertia.
Native readback verifies proportional scaling and reset persistence. This is a
constant-shape uncertainty approximation; independent COM shifts are also an
uncertainty approximation. Neither reconstructs payload geometry.

Armature .01 is a development preset motivated by the inherited joint defaults
in [Unitree's Go2-W MuJoCo model](https://github.com/unitreerobotics/unitree_mujoco/blob/main/unitree_robots/go2w/go2w.xml).
Its damping/frictionloss and different calf effort are **not imported**. The
randomization range, zero passive terms, sensor/tire approximations and latency
range are not measured hardware identification. The old Actor's .1→0 comparison
established sensitivity, not that .01 fails. The rejected PhysX 40/2 diagnostic is
not a new default. No gain search or timestep change was needed here.

The observation construction, noise, scaling, clipping, action history and
symmetry code remain unchanged. Slices are `0:3` linear velocity, `3:6` angular
velocity, `6:9` gravity, `9:12` commands, `12:24` leg pose error, `24:40` all joint
velocities and `40:56` previous **issued clipped** action. Simulator-derived
linear velocity refers to the authored base-link origin in body axes at the
current policy boundary: installed `get_vel(relative=True)` already performs
origin transport in world axes, then the environment rotates it into body axes.
No second COM transport is applied. Both networks receive the same noisy tensor;
training velocity noise remains .05 m/s. Export still embeds only the Actor's own
normalizer once and produces deterministic `[N,56] → [N,16]` output. New metadata
records nominal dynamics separately from training ranges and actual environment-0
readback; a random training armature never becomes deployment nominal.

Raw preparation evidence is in ignored `evaluation/transfer_v1_preparation_20261001/`.
`preparation_summary.json` is the compact resolved recipe/results handoff. The
unchanged publication checker and original-snapshot CPU Actor parity passed with
zero CPU error; P's bundle and both retained reference assets remain byte-exact.
The plant probe used three isolated environments at .01/.005/.02 kg m², fixed
nominal targets for 5 s, +.01 rad thigh targets for 1 s, then nominal for 3 s.
All 16 armatures, active/passive terms, floating-base exclusion and selective reset
passed. Final planar RMS .004514–.004520 m/s, joint dq RMS .006487–.006516 rad/s,
body roll/pitch-rate RMS .001553–.001561 rad/s, support 193.079 N; minimum finite
cylinder envelope clearance about −1.02 mm. No fall/nonfinite state occurred.
`plant.npz` contains bounded 200 Hz states, and `plant.json` contains readback.
The fresh 64-environment/two-update smoke had 128 rollout steps, 80 optimizer steps,
finite losses/parameters, fresh separate normalizers, empty initial optimizer,
zero initial iteration/curriculum, bounded symmetric armatures and proportional
mass/inertia. Serialization/reload was inference-only. Export parity on 160 new
observations, including actual initial and selective-post-reset observations,
had maximum error **1.073e-6**. This is wiring evidence, not locomotion training.
Its explicitly selected checkpoint is
`logs/go2w_transfer_v1_smoke/transfer_v1_smoke_seed1_20261001_2026-10-01_23-27-24/model_1.pt`;
**never use it as a training parent**. `smoke.json`, TensorBoard, preparation and
manifest files retain details. Relevant CPU regressions, syntax/lint and whitespace
checks are recorded alongside the native process logs. Final CPU run: **91 passed,
two unrelated opt-in GPU checks skipped** (93 discovered). Installed Python
3.11.14, Genesis 1.4.1, Torch 2.9.0+cu130, RSL-RL 5.5.1 and Tensordict .10.0;
the four recorded installed Genesis source hashes still match the frozen-reference
audit. Three native processes used **511.827 s** total, with no long training.
The final evaluator wiring check explicitly loaded smoke CK1 at armature .005 and
delay 2, verified reset persistence, and captured first failures at **4.08 s stand
and 4.20 s forward**. States stayed finite, with no timeout/nonwheel ground
contact; later intervals were censored. Its 200 Hz traces contain 816/840 rows.
These are failures of an untrained smoke Actor, not evidence of learned standing,
transfer success, or a reason to initialize the real run from this checkpoint.

From the repository root, the following owner-controlled command prepares the
verified existing environment and starts the **real fresh run**. It was not executed:

```powershell
$go2wPrefixes = @((conda env list --json | ConvertFrom-Json).envs | Where-Object { (Split-Path $_ -Leaf) -eq 'genesis-gpu' })
if ($go2wPrefixes.Count -ne 1) { throw 'Expected the verified existing genesis-gpu environment' }
$Python = Join-Path $go2wPrefixes[0] 'python.exe'
$env:NUMBA_CACHE_DIR = "$PWD/.migration-audit/diagnostics-20260925/numba-cache"
$env:GS_CACHE_FILE_PATH = "$env:TEMP/go2w-diagnostics-genesis"
$env:QD_OFFLINE_CACHE_FILE_PATH = "$env:TEMP/go2w-diagnostics-quadrants"
$env:PYTHONIOENCODING = 'utf-8'
$RunName = "transfer_v1_seed1_$(Get-Date -Format yyyyMMdd_HHmmss)"
& $Python -m robot_gym.scripts.train --task go2w --go2w_profile transfer_v1 --experiment_name go2w_transfer_v1 --run_name $RunName --num_envs 4096 --max_iterations 2000 --seed 1 --logger tensorboard --training_diagnostics --rl_device cuda:0 --headless
```

The **smoke** used the same training CLI through `tests.test_go2w_transfer.TransferSmoke`,
with `--num_envs 64 --max_iterations 2 --experiment_name go2w_transfer_v1_smoke`;
it additionally verifies serialization, resets and export. This differs from the
owner's continuous 2000-update run. Existing lightweight JSONL diagnostics retain
deterministic versus sampled clipping, physical target slew, motion, Gaussian/KL
and unclamped/clamped reward evidence; no full high-rate training recorder exists.

Review saved checkpoints around labels **300–500** by inference while the same
continuous run proceeds under owner control. Fresh label k follows k+1 completed
updates. The lateral-tail curriculum starts near update 500 and reaches the final
range near 1500; an early screen cannot establish full lateral capability. There
is no automatic stop/resume, finetuning or checkpoint initialization stage.
After the run directory exists, use the same unique `$RunName` from above (or its
exact printed run directory); the following selector refuses multiple matches:

```powershell
$RunMatches = @(Get-ChildItem 'logs/go2w_transfer_v1' -Directory | Where-Object { $_.Name -like "${RunName}_*" })
if ($RunMatches.Count -ne 1) { throw 'Select the exact printed training run directory' }
$Run = $RunMatches[0].FullName
$Review = "evaluation/transfer_v1_ck300_$(Get-Date -Format yyyyMMdd_HHmmss)"
& $Python -m robot_gym.scripts.evaluate --task go2w --go2w_profile transfer_v1 --experiment_name go2w_transfer_v1 --load_run $Run --checkpoint 300 --eval_mode transfer_screen --transfer_armature nominal --transfer_delay 0 --num_envs 1 --seed 1 --rl_device cuda:0 --headless --diagnostic_trace --output "${Review}_nominal"
# Single-factor checks; no Cartesian sweep:
& $Python -m robot_gym.scripts.evaluate --task go2w --go2w_profile transfer_v1 --experiment_name go2w_transfer_v1 --load_run $Run --checkpoint 300 --eval_mode transfer_screen --transfer_cases stand forward --transfer_armature low --transfer_delay 0 --num_envs 1 --seed 1 --headless --diagnostic_trace --output "${Review}_armature_low"
```

For the remaining small sensitivity panel, explicitly select `high` with delay 0,
then `nominal` with delay 1 or 2, using fresh distinct outputs and only `stand forward`.
The nominal panel contains exact stand, forward/start/stop, reverse, both yaw and
lateral signs and a mixed command; each case has a fresh reset and is at most 14 s.
It retains first terminal rows and censors incomplete intervals. Reports include
finite/fall/nonwheel contact, axis tracking errors, final-two-second residual
speed/yaw RMS, five-second drift, post-stop displacement, posture, actual cylinder
clearance, deterministic clipping, target slew and period-two action amplitude.
`--diagnostic_trace` reuses the existing bounded 200 Hz recorder for physical
amplitudes/spectra as well as 50 Hz Actor traces; spectra alone are not a stand test.
Deterministic evaluation disables noise/randomization and selects explicit nominal
or endpoint armature. Ordinary play/export uses the same exact run/checkpoint
selection through `robot_gym.scripts.play`; its existing export machinery writes
the new bundle, never P's historical package.

**Receiving Isaac handoff.** Its current locomotion `asset.py:make_articulation_cfg()`
explicitly sets `armature=0.0`; URDF/USD replacement alone cannot remove that
override. Apply/read back the new bundle's .01 kg m² motor mapping, nominal
40/1 legs and 0/1 wheels, zero passive assumptions, unchanged efforts, mixed
targets, timing and scales. Reuse the existing 56-input/16-output controller and
the **new Actor's embedded normalizer**; verify joint/frame mapping, current
base-link-origin velocity, issued-action/reset history and bundle identities.
Retain the measured embodiment, official Plane, verified principal-inertia import
workaround and authored pre-warmup Joint State. Actuator API gains are SI; raw USD
angular drives use different units, so do not add degree conversion at the API or
to armature. Equal scalar parameters do not imply equal Genesis/PhysX integration.
Keep P and its integration evidence intact. This requires one new-bundle/dynamics
adaptation, not a second player, 53-input adapter or navigation rewrite. The next
transfer and simulation-navigation work continues using simulator velocity.

**Future estimator, documentation only.** The pinned
[basic-locomotion reference](https://github.com/iit-DLSLab/basic-locomotion-isaaclab/blob/a75a480b9ae6a21fbe1ab7f65d0665dbbcccf70c/source/basic_locomotion_isaaclab/basic_locomotion_isaaclab/tasks/custom_observations.py)
uses supervised sensor/action histories and eventually substitutes predictions;
its Go2 option is disabled by default, with five frames and a TCN option. A later
Go2-W estimator can supply only entries 0:3 before existing scaling/clipping and
embedded normalization. Keep Actor width 56; 53 non-velocity features per frame
do not define a 53-input Actor. Use causal histories for all 16 joints/continuous
wheels and issued actions, advancing once per policy tick with isolated episode
resets. True velocity is target/reference only; no true-position shortcut, future
sample or command-as-label. Estimate the same physical origin. Any estimator-only
IMU acceleration needs explicit gravity/frame/offset/timing conventions. First
evaluate supervised/shadow errors, then truth-driven versus estimate-driven
closed-loop stand/stop drift, bias, latency, slip and disturbance response. Low
average MSE does not establish substitution compatibility; separately authorized
estimator-aware policy training may be necessary. Do not copy the reference's
`common_step_counter / 24` clock or fixed update assumptions into this 64-step run.
There is no executable estimator, estimator checkpoint or online learning here,
and simulator-navigation results using truth must be described accordingly.


## Shared pipeline cleanup (2026-10-02)

Replay resolves and prints one checkpoint, safely restores the saved environment
and training class configuration, then applies explicit runtime overrides.
`--load_run` selects the input run; `--checkpoint -1` selects its latest saved model.
For existing owner commands, `--run_name` also selects replay input when
`--load_run` is absent; the CLI prints that compatibility choice.
No historical profile is required. Explicit task/profile/interface conflicts fail
before simulation. Missing optional physical settings retain imported behavior;
Dodo and Go2 do not acquire Go2-W motor overrides. Task rules and evaluation code
now live under `envs/go2w`; historical script/import paths are thin wrappers.

Export is opt-in (`--export`), under the selected run's
`exported/model_<number>/`; `--no_export` is still accepted. Fixed body command axes
are vx/vy in m/s and yaw rate in rad/s: any supplied axis makes omitted axes zero;
no axes preserves command sampling. Command-range curriculum is independent.
`--steps` counts policy ticks: 900 at .02 s means 18 simulated seconds excluding
startup/reset settling. Ticks span resets. `--episode_length_s 30` allows an 18 s
session without an episode timeout, while actual falls still reset.

A separate functional fix refreshes the first observation after checkpoint task
state loading (which can resample commands), then obtains a new policy tensor.
Issued/clipped versus delayed/applied action history and reset semantics are unchanged.
The 200-tick nominal CK1999 cleanup comparison at [.2,0,0] had max state/action
absolute difference 0 (tolerance 1e-4), no terminal reset. This is a short regression,
not locomotion qualification. Existing owner CK300/CK1999 screens remain in
`evaluation/transfer_v1_ck300_20261002_132929` and
`evaluation/transfer_v1_ck1999_20261002_133544`; they are Genesis nominal results.
Cleanup evidence is in ignored `evaluation/transfer_v2_preparation/cleanup_*`.

```powershell
conda activate genesis-gpu
$Run = 'transfer_v1_seed1_20261002_090639_2026-10-02_09-09-17'
python -m robot_gym.scripts.play --task go2w --experiment_name go2w_transfer_v1 --load_run $Run --checkpoint -1 --num_envs 1 --command_vx 0.5 --steps 900 --episode_length_s 30
# Stand: replace --command_vx 0.5 with --command_vx 0.
# Lateral/yaw: use --command_vy 0.3 or --command_yaw 0.8, respectively.
# Omit all command axes for sampled commands; add --export only to export.
python -m robot_gym.scripts.play --task go2w --help
```


## transfer_v2: support-aware stepping candidate (2026-10-02)

`--go2w_profile transfer_v2` opts into one **fresh** recipe, experiment
`go2w_transfer_v2`. It is a prospective engineering candidate, not a qualified
gait or a sim2sim/hardware result. V1 saved recipes/checkpoints remain authoritative
when replayed. No checkpoint, optimizer, normalizer or curriculum initializes v2.

The measured embodiment, nominal .01 and randomized [.005,.02] kg m^2 motor
armature, 40/1 leg and 0/1 wheel active gains, zero passive assumptions, effort and
velocity limits, mass/inertia handling, action scales and timing remain v1's.
Actor/Critic/actions remain **56/56/16**, both networks consume `policy`; separate
fresh normalizers, simulator body/base-link-origin velocity, MLP [512,256,128],
ELU, PPO, log std .40 projected to [.10,.70], entropy .003, LR 3e-4, gamma .995,
64 rollout steps and symmetry augmentation are unchanged. No estimator is added.

The active reward path is below. `mean_L`, `mean_W` and `sum_L/W` mean reductions
over the 12 leg / four wheel joints; hip/sagittal means use four/eight joints.
`e=q-q_nominal`, `G=max(clamp((abs(vy_cmd)-.01)/.04),
clamp((abs(yaw_cmd)-.10)/.15))`, with clamps in [0,1]. `loaded` is wheel normal
load >8 N; the event tracker separately uses its existing 6/10 N hysteresis.
Scales are applied once. **Every continuous term gets policy dt=.02 once**;
`step_event` instead uses its already accumulated gate-time credit. Raw values,
weighted contributions and total reward clipping are different quantities.

| Active term | Actual raw formula / reduction and units | Scale v1 -> v2 | Gate / accounting |
|---|---|---|---|
| tracking_lin_vel | exp(-ex^2/Dx-ey^2/Dy), dimensionless | +1 unchanged; Dx .25 -> .09, Dy .04 | All commands; dt |
| tracking_ang_vel | .25 exp(-ez^2/.25)+.75 exp(-ez^2/.04), dimensionless | +.8 unchanged | All commands; dt |
| orientation | sum(projected_gravity_xy^2), dimensionless, locally quadratic in tilt | -1.2 -> **-4** | All commands; dt |
| base_height | (base_z-.415)^2, m^2 | -8 unchanged | All commands; dt |
| lin_vel_z | vz^2, (m/s)^2 | -1 unchanged | All commands; dt |
| ang_vel_xy | sum(omega_xy^2), (rad/s)^2 | -.25 unchanged | All commands; dt |
| hip_pose | V1: (1-.5G) mean_hip(e^2); v2: mean_hip(C_i e_i^2), rad^2 | -2 -> -1 | V2 C=2 at G=0; at G=1, 1 loaded / .2 unloaded; linear interpolation; dt |
| sagittal_pose | V1: ((2(1-G)+.12G)/.6) mean_sagittal(e^2); v2: mean_sagittal(C_i e_i^2), rad^2 | -.6 -> -1 | V2 C=2 at G=0; at G=1, .6 loaded / .06 unloaded; linear interpolation; dt |
| step_event | sum_W(quality * gate-time credit), seconds of capped credit | +.15 unchanged | Completed valid attempts only; **no extra dt** |
| wheel_swing | G mean_W(eligible * height_score * reposition_score), dimensionless | disabled -> **+.4** | Existing supported-to-unloaded attempt; >=2 other supports; not failed; within .6 s; dt |
| lateral_wheel_scrub | G mean_W(loaded * u_lateral^2), (m/s)^2 | disabled -> **-2** | Loaded cylinder centers, ground-tangent axle projection; dt |
| normalized_effort | mean_L((tau/limit)^2)+mean_W((tau/limit)^2), dimensionless | -.03 unchanged | All commands; instantaneous control effort surrogate; dt |
| leg_acc / wheel_acc | sum_L/W(((dq-dq_previous)/dt)^2), rad^2/s^4 | -2.5e-7 / -1e-7 unchanged | All commands; dt |
| leg_action_rate / wheel_action_rate | sum_L/W((issued_action-previous_issued)^2), normalized action^2 | -.01 / -.005 unchanged | All commands; dt |
| leg_motion | (1-G) mean_L(dq^2), (rad/s)^2 | -.02 unchanged | Vanishes at full step demand; dt |
| stand_still | sum(vxy^2)+yaw_rate^2+.02 mean_W(dq^2), mixed motion surrogate | -2 unchanged | Command norm <1e-6; dt |
| unnecessary_wheel_air | (1-G) mean_W(event_unloaded), dimensionless | -.25 unchanged | Vanishes at full step demand; dt |
| prolonged_unloading | mean_W(clamp((unloaded_time-.6)/.2,0,1)^2), dimensionless | -.2 unchanged | Existing unload clock; dt |
| insufficient_support | relu(2-count_W(load>6 N))^2, dimensionless | -1 unchanged | All commands; dt |
| wheel_crossover | sum_front/rear(relu(.045-yL)^2+relu(.045+yR)^2+.5 relu(.14-yL+yR)^2), m^2 | -2 unchanged | Body-frame wheel origins; dt |
| dof_pos_limits | sum_L(relu(abs(q-mid)-.9*half_range)), rad | -2 unchanged | Finite position-controlled joint limits; dt |
| torque_limits | sum_all(relu(abs(tau)-.9*effort_limit)), Nm | -.5 unchanged | All commands; dt |
| collision | min(nonwheel_contact_count,4), count | -.5 unchanged | All commands; dt |
| termination | reset AND NOT timeout, indicator | -10 unchanged | dt; added **after** total nonterminal clipping |

`only_positive_rewards=True` is retained: sum weighted nonterminal terms, clamp
that sum to >=0, then add termination. Dense `foot_swing_clearance`, default_pose,
point-foot sliding, generic action-rate/acceleration and other zero-scale inherited
terms remain disabled. In particular, the old link-height swing method is not used.
The tracking denominators have units (m/s)^2; for exp(-e^2/D), conventional Gaussian
std is sqrt(D/2). This is unrelated to Gaussian policy exploration std.

V2 maps each wheel's support to its own hip/thigh/calf by names once. Coefficients
are nonnegative and a single negative scale applies afterward. Swing legs have
freedom while loaded legs retain a posture cost; instantaneous left/right actions
are not constrained to match. Nominal angles/root pose and the .415 m height target
are unchanged. Orientation remains the gravity formula, with no extra tilt reward.

The dense height is min(actual oriented-cylinder clearance, frozen-base
limb-contributed clearance), clamped below at zero. Its score rises linearly to
4 cm, remains broad through 4-6 cm (flat until 7 cm), then decays with a 2 cm
upper-tail width. Reposition score is .2+.8 clamp(horizontal frozen-body center
displacement/.04,0,1). Event geometry/attempt age/support/reset state are reused;
only two instantaneous geometry values were exposed. Base rocking, startup/drop,
loaded sliding and wheel spin cannot substitute for limb lift/reposition. Holding
a leg up beyond the existing .6 s maximum earns no dense reward. The existing
8 mm completed-event threshold, 1 cm minimum reposition, .04/.06 s unload/reload
dwells and .12 s prior support remain. Dense feedback is not a successful-step label.

Scrubbing is explicitly a **lateral wheel-center surrogate**, not exact tire
material-point slip. Installed Genesis 1.4.1 `get_links_vel(relative=True)` already
returns authored-origin velocity in world axes. The code adds omega_link cross
collision_offset_world once, then projects onto the normalized horizontal axle.
It preserves ideal aligned rolling and exempts unloaded repositioning. It does
not require perfectly slip-free yaw or disable wheel velocity control.

Family probabilities/ranges and lateral magnitude curriculum are unchanged.
For pure lateral, pure yaw and mixed segments with G>0, probability .70 selects
2-4 s duration; already drawn 8-15 s segments remain. Other durations/families and
long stands retain their previous draws. These are **segment probabilities**, not
fractions of training time. V1 uses no additional random draws. The early-check
curriculum caveat remains: lateral tails start around update 500 and mature at 1500.

The old nominal screens motivate this candidate: CK1999 tracked vx=.2 at mean
.15581 m/s (RMSE .04548), had weak lateral means around +.02025/-.02124 m/s and
near-zero mixed lateral response; mixed FL-calf raw action clipped on 66.8% of the
command phase. This is action saturation, not evidence of torque saturation.
There were zero qualified events and zero completed >=2 mm geometric swings;
startup clearance was excluded. Its stand final planar RMS .00874 m/s and five-second
net drift 2.88 cm improved. These are the existing five-second command-phase
screens, not measurements of the owner's separate .5/.3/.8 replay screenshots.
Recorded training reward clipping was 1.87% at iteration 300 and .96% at 1999;
ten std entries reached .1 at 1999. No entropy schedule was introduced.

Preparation evidence is in ignored `evaluation/transfer_v2_preparation/`.
The native readback confirmed all 16 nominal armatures at .01, leg gains 40/1,
wheel gains 0/1 and zero passive stiffness/damping/friction. Intended nominal q
and zero dq were applied before the first physics step despite Genesis's import
`qpos0` warning. Settled base height was .415873 m, supporting retention of .415.
Offline bounded FK reached 4 cm actual clearance plus 3 cm lateral reposition
at that height for every wheel, with normalized actions no larger than .833.
This is geometric reachability. Four brief single-leg target pulses with the
other legs nominal stayed loaded (no measured positive clearance or failure)
and tilted the base by up to 5.22 degrees. Coordinated load-bearing steps remain
unproven; this one script does not establish infeasibility. No action box, gain,
nominal angle or root-pose change was made to conceal that limitation.

A separate inference-only CK1999 capture used three fresh 300-tick sessions;
these **new 2-6 s windows** differ from the owner's 900-tick sessions and the
earlier five-second nominal screens. No terminal failure occurred:

| Fixed command | Mean requested-axis velocity | Requested-axis RMSE | Signed mean roll / pitch | Roll / pitch RMS | Loaded lateral-center RMS |
|---|---:|---:|---:|---:|---:|
| [.5,0,0] | .44191 m/s | .05810 m/s | -.074 / -4.388 deg | .086 / 4.391 deg | .00124 m/s |
| [0,.3,0] | .08752 m/s | .36661 m/s | -4.651 / 1.311 deg | 5.669 / 2.521 deg | .07168 m/s |
| [0,0,.8] | .83479 rad/s | .03654 rad/s | 1.858 / 4.998 deg | 1.934 / 5.002 deg | .04469 m/s |

Angles come from the actual base quaternion (extrinsic xyz); negative pitch is
nose-up in these axes. No screenshot angle was used. `native_probe.json/npz`
retain the state/support traces. One earlier probe stopped on a test-only CPU/GPU
tensor-device mismatch before target pulses; its `native_probe_setup_error.*`
evidence is retained. Correcting that helper did not modify the simulator stack.

The **one** fresh 64-environment/two-update smoke passed: 128 policy ticks,
8192 environment transitions, 80 optimizer steps, finite losses/gradients,
separate initially empty normalizers, empty optimizer and zero curriculum/iteration,
randomized dynamics and selective-reset isolation, checkpoint serialization and
export parity on 160 observations including actual post-reset observations.
Maximum exported/runtime action error was 8.05e-7. Its run is
`logs/go2w_transfer_v2_smoke/transfer_v2_smoke_seed1_20261002_2026-10-02_16-38-50`;
it is wiring evidence only, with zero qualified completed steps.

Clipping is a material early-learning uncertainty: updates 0/1 clipped 68.97/68.77%
of nonterminal sums, discarding mean negative magnitudes .01798/.02080 per tick.
Among nonterminal states without nonwheel contact and with G>0, 61.72/58.24% were
clipped. Scrub contributions hidden there averaged -2.84e-5/-2.90e-5 per valid
demand tick. Inherited angular-motion costs averaged -.01259/-.01274 and support
costs -.00487/-.00439 per tick; new orientation costs were -.00036/-.00122,
and mean dense swing feedback was only 3.15e-7/2.52e-7. This does not justify
weakening the requested scrub/posture coefficients or changing termination from
two untrained updates. The coefficients and positive-sum clamp are retained;
the early checkpoint review must inspect whether clipping subsides as control
improves. Raw and weighted terms are saved separately in the smoke diagnostics.

All 100 focused CPU checks passed (97 on the combined run, three updated fixture
expectations retested successfully). Ruff, syntax compilation and diff checks
passed. The exact fresh CLI was resolved and validated without constructing a
training environment; the owner's original `--run_name ... --checkpoint -1`
replay command also resolved CK1999 successfully. CPU checks cover saved-config/latest replay for all three registered
robots, fixed axes/duration, legacy observations/action history/symmetry, support
mapping, swing exclusions/cycles, scrubbing reflection/transport, command segment
draws and full reward accumulation/clipping. Logs retain actual results, including
initial fixture fixes. No new GUI, Isaac test or full v2 review panel was run.

From the repository root, the exact **fresh long-run command**, prepared but not executed:

```powershell
conda activate genesis-gpu
python -m robot_gym.scripts.train --task go2w --go2w_profile transfer_v2 --experiment_name go2w_transfer_v2 --run_name transfer_v2_seed1 --num_envs 4096 --max_iterations 2000 --seed 1 --logger tensorboard --training_diagnostics --rl_device cuda:0 --headless
```

The recipe saves every 100 updates. Review checkpoints around 300-500 from this
same continuous run by inference; this does not mean stop/resume or initialize a
second training stage. The smoke checkpoint is never a training parent.
The following review panel is prepared, **not executed during preparation**:

```powershell
$Run = Read-Host 'Exact run directory printed by training (under logs/go2w_transfer_v2)'
python -m robot_gym.scripts.evaluate --task go2w --experiment_name go2w_transfer_v2 --load_run $Run --checkpoint 300 --eval_mode transfer_screen --num_envs 1 --seed 1 --rl_device cuda:0 --headless --diagnostic_trace --output evaluation/transfer_v2_ck300_nominal
# Optional smaller selection uses the same evaluator:
# --transfer_cases stand forward_fast lateral_strong_positive yaw_strong_negative mixed
python -m robot_gym.scripts.play --task go2w --experiment_name go2w_transfer_v2 --load_run $Run --checkpoint -1 --num_envs 1 --command_vy 0.3 --steps 900 --episode_length_s 30
```

The v2 screen adds vx=.5, vy=+/-.3 and yaw=+/-.8 to the retained stand, vx=.2,
reverse, vy=+/-.1, yaw=+/-.4 and [.2,.1,.3] cases, each moving case with a final
six-second stop. Reports include full-phase means/RMSE and counter-command peaks,
signed/RMS base roll/pitch, loaded scrub, actual clearance and per-wheel completed
cycles, action clipping and physical motion amplitudes. `--diagnostic_trace` adds
the existing bounded physics-rate recorder. Normal training only aggregates cheap
scalars: per-family posture, completed per-wheel step heights/counts, scrubbing,
per-term raw/weighted rewards, total clipping and existing action/std diagnostics.

Development review targets are small steady stand/straight tilt bias (roughly
within 2 degrees), improved full-phase lateral tracking without the old large
loaded counter-bursts, and repeated supported/unloaded/repositioned/reloaded
centimeter-scale steps in substantial lateral/yaw demand. Quiet standing alone
cannot qualify stepping, and synthetic or smoke checks cannot qualify learned gait.

Receiving Isaac work still needs the new Actor's own embedded normalizer and
explicit nominal motor dynamics readback (its existing armature=0 override must
be adapted). Keep the measured model, official Plane, verified inertia import and
56/56-to-16 interface/controller. No new Isaac test, navigation training or hardware
execution occurred here. Armature/ranges, passive zeros, mass/COM and tire/sensor
geometry remain development assumptions. Future velocity estimation remains the
separate documented causal-history/shadow/closed-loop project; this run continues
to use simulator velocity and contains no estimator code or checkpoint.

## transfer_v3 preparation blocked — 2026-10-03

**No v3 recipe, 58-input Actor, pilot or main training was released.** The requested
common [.0, .70, -1.40] leg pose failed the prerequisite floating-base fixed-target
stand. One follow-up reducing spawn clearance from 30 mm to 3 mm reproduced it.
The coordinated unloading stage was never reached. Following the requested stop
condition, no gain/armature search, third native trial or learning run followed.
This is a blocker for this preparation, not proof that feedback cannot stabilize
the pose or that the robot cannot step. No loaded reward height was selected.

### Existing CK700 evidence

Read the actual `config.yaml`, `diagnostics.jsonl` and TensorBoard events from
`logs/go2w_transfer_v2/transfer_v2_seed1_2026-10-02_19-40-45`, and the existing
`evaluation/transfer_v2_ck700_nominal`. No historical training or evaluation file
was modified. These are Genesis results, not IsaacLab transfer results.

CK700's 13 nominal cases record no falls or nonwheel ground contacts. Full-command
mean vx is .1897/.4821 m/s for .2/.5 m/s commands. At vy=+.3, mean vy is .01573
m/s; the last second falls to .000352 m/s with undesired vx=.08236 m/s. Negative
lateral demand fails approximately symmetrically. Strong lateral demand ends
with about .7377 m wheelbase and .3589 m base height, versus .3868 m thigh-origin
fore/aft spacing. Eight completed >=2 mm geometric cycles occur across the panel,
mostly at strong-lateral onset; only one event, in mixed motion, qualifies under
the training criteria. Pure yaw +/-.8 has no completed geometric cycles. There is
no sustained lateral stepping, but it would be incorrect to say no wheel lifts.

Actual training rollouts (4096 environments x 64 ticks each):

| Update | Reward sums clipped, all / lateral | Measured minibatch KL mean / max | LR after update | Per-joint std range | Largest deterministic mean-action clipping |
|---|---:|---:|---:|---:|---:|
| 0 | 69.755 / 61.434% | .08258 / .14860 | 1e-5 | .3997-.4004 | 0% |
| 100 | 12.600 / 14.207% | .01523 / .02065 | .0029193 | .2324-.3262 | .0084% |
| 300 | 2.964 / 3.129% | .01614 / .02088 | .0008650 | .1377-.2546 | .7088% |
| 700 | 1.838 / 2.336% | .01378 / .02052 | .0003844 | .1112-.2124 | .7397% |

Reward-sum clipping by sampler family, percent (not action clipping):

| Update | Stand | Straight | Arc | Yaw | Precision | Lateral | Mixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 90.160 | 75.864 | 76.134 | 50.245 | 54.183 | 61.434 | 77.714 |
| 100 | 27.697 | 6.802 | 9.427 | 6.060 | 3.852 | 14.207 | 19.863 |
| 300 | 3.286 | 1.986 | 3.591 | 1.937 | 1.188 | 3.129 | 6.332 |
| 700 | 1.891 | 1.281 | 2.269 | .888 | .589 | 2.336 | 4.182 |

Selected raw reward means -> weighted means per tick (all sampled families):

| Update | Linear tracking | Angular xy motion | Completed event | Dense swing |
|---|---:|---:|---:|---:|
| 0 | .43362 -> .008672 | 2.64010 -> -.013200 | 1.30e-5 -> 1.94e-6 | 7.69e-5 -> 6.15e-7 |
| 100 | .77786 -> .015557 | .75610 -> -.003781 | 1.87e-7 -> 2.80e-8 | 1.43e-5 -> 1.15e-7 |
| 300 | .81275 -> .016255 | .30431 -> -.001522 | 0 -> 0 | 1.48e-5 -> 1.18e-7 |
| 700 | .82567 -> .016513 | .21548 -> -.001077 | 2.91e-6 -> 4.37e-7 | 2.94e-5 -> 2.35e-7 |

All terms, per-joint clipping and family exposure are retained in
`evaluation/transfer_v3_preparation/ck700_evidence.json`. Qualified stochastic
training events by FL/FR/RL/RR at these updates are [12,19,3,5], [0,0,0,1],
[0,0,0,0], [4,2,0,0]; the corresponding >=1 cm qualified counts are
[10,16,2,3], [0,0,0,0], [0,0,0,0], [3,2,0,0]. These are not the deterministic
panel's eight geometric cycles. An ungated geometric cycle count is not logged
for training; it cannot be inferred from the event counter.

At update 700, Gaussian std in hip/thigh/calf/wheel order is FL
[.1149,.2124,.1306,.1707], FR [.1148,.2122,.1306,.1707], RL
[.1112,.1153,.1418,.1655], RR [.1112,.1154,.1418,.1655]. No joint is at the .1
floor in any of the four inspected updates. The largest mean-action clipping
at 700 is RL wheel, .7397%; FL calf is .1457%, averaged over the training rollout.
This does not exclude severe saturation in an individual deterministic command.
At update 700, time exposure is 14.43% stand, 15.99% straight, 14.68% arc,
20.32% yaw, 7.90% precision, 17.13% lateral and 9.56% mixed. These are measured
tick fractions, not configured segment-draw probabilities.

The early large KL and clipping subside; the logs do not establish a persistent
optimizer failure or std-floor collapse. V2 swing shaping still requires an
already-unloaded active attempt. Tiny mean shaping and failed deterministic
lateral coordination support revisiting the objective; aggregate logs cannot
attribute the result to one weight. Three separate quantities remain distinct:
tracking denominators Dx=.09, Dy=.04 have units (m/s)^2; Gaussian std is in
normalized action units; uniform observation noise magnitudes include .05 m/s
linear velocity, .08 rad/s angular velocity, .01 rad position, .2/.5 rad/s
leg/wheel velocity and .02 projected gravity, before the existing observation
scales. Reset noise is separately recorded in the saved config.

The inspected path holds delayed applied targets for four 5 ms substeps, updates
state/events and scores the command that produced the transition, captures the
terminal state, then resamples/resets and builds the next observation. Issued
clipped action history remains distinct from applied delayed action. Continuous
rewards receive .02 s once; the event receives .15 without dt. V2 clips the
nonterminal sum at zero before adding termination. None of these semantics changed.

### Pose calibration and stop

FK uses the measured asset's oriented collision cylinders and link origins.
At hip=0, thigh=.70, calf=-1.40, wheel=0 for all four chains, with a level authored
base frame, the floor-compatible geometric height is **.427741656 m**. All four
floor gaps are zero there; every cylinder center is **+.008632517 m** forward of
its thigh origin. Fore/aft center and thigh spacing are .3868 m. This agrees with
the analytic front-chain estimate, but is constrained geometry, not equilibrium.

Two headless native launches used normal floating-base gravity/contact, fixed
nominal leg position targets and zero wheel velocity targets, no learned policy,
noise, pushes, randomization or external support. Two baseline replicas were
intended for mirrored later target sequences; both failed before those sequences.
Nominal q was applied through the existing reset/control path, with an identity
root quaternion. Neither targets nor torque limits were changed during settling.

| Spawn gap | Spawn base height | First terminal time after reset settling | Terminal height | Signed pitch |
|---|---:|---:|---:|---:|
| 30 mm | .457742 m | 1.80 s | .328143 m | -18.999 deg |
| 3 mm follow-up | .430742 m | 1.80 s | .328651 m | -18.873 deg |

Both cross the unchanged .33 m height threshold, not a timeout. Negative pitch
is nose-up in the actual robot frame. The 3 mm follow-up removes most initial drop
without correcting the collapse. Rear thigh/calf positions reach .3934/-1.7832
rad while targets remain .70/-1.40. Last-second mean front/rear loads are about
21.48/74.27 N per wheel; this is a falling interval, not settled load sharing.
The immediate obstruction is compliant fixed-target posture collapse and rear
loading. No torque saturation was observed: maximum control-force API readback
over the substeps is 53.27% of its joint limit. That getter recomputes controller
force; it is not a recorded solver impulse. No nonwheel ground contacts were
recorded. Nonterminal self-contact queries were also empty; the terminal
self-contact getter follows auto-reset and is unavailable for that state.

Post-reset readback confirms .01 kg m^2 motor armature, leg Kp/Kd=40/1,
wheel Kp/Kv=0/1 and zero passive stiffness/damping/friction. Timing stays .005 s,
one internal substep, four held-target steps per policy tick, with
`approximate_implicitfast`. Those remain development assumptions, not identified
hardware properties. The lower spawn did not expose an import or effort-limit
mismatch. A gravity-compensated target/support calculation needs resolution
before another lift probe; increasing armature or relaxing termination would not
establish the requested pose calibration.

Raw files are in `evaluation/transfer_v3_preparation`: `pose_geometry.json`,
`unloading_probe_drop30mm.*`, `unloading_probe.*`, and `pose_findings.json`.
Use **pose_findings.json** for interpretation: the first report's early-failure
`max_force_fraction` used a unit fallback and is not a valid fraction; its actual
maximum was 15.51 Nm. The corrected summary leaves that fraction unavailable.
Likewise, zero active-stage ticks in the raw report mean **not reached**, not a
measured failed unloading maneuver. Pre-reset q/dq, pose, load, clearance,
issued/applied actions and substep maximum control force remain in the NPZ files.

`tests/go2w_pose_probe.py` retains just this bounded fixed-target calibration
without the unexecuted IK sequence. It refuses an existing output directory and
marks terminal self-contact unavailable. Its FK and syntax were checked; this
cleaned reproduction entry was not launched as a third native trial.

### Maintenance, validation and next execution

Task CLI/fresh-run validation now live with Go2-W configuration; the small
`cli.py`/`training.py` fragments were removed. Runtime metadata belongs to the
environment; explicit legacy continuation checks remain in diagnostics, with
the historical script imports retained. The unused hardcoded measured-asset hash
gate was removed; geometry and interface checks remain. Environment, config,
symmetry, deployment, evaluation and optional diagnostics remain separate because
they have distinct responsibilities. No common robot physics/rewards changed.

Seven focused common-path/fresh-initialization CPU checks passed, including
Dodo/Go2 saved-config/latest selection and old Go2-W replay. The existing measured
geometry regression also passed. Ruff, syntax and diff checks passed. No new
learning, export-parity or phase/symmetry check is claimed: v3 was gated before
implementation. Existing v1/v2 56-input bundles/configs/results remain intact.
The requested v3 design would require 58/58 inputs, appended phase sin/cos with
half-cycle reflection, its own pose and normalizers; no compatible v3 package or
final objective/scales exists yet. No main-training command can truthfully be
provided for an unavailable `transfer_v3` profile. A future main run must still
start fresh, never from the pilot or an old checkpoint.

Exact CK700 replay from the repository root (installed `genesis-gpu`):

```powershell
conda activate genesis-gpu
python -m robot_gym.scripts.play --task go2w --experiment_name go2w_transfer_v2 --load_run transfer_v2_seed1_2026-10-02_19-40-45 --checkpoint 700 --num_envs 1 --command_vy 0.3 --steps 900 --episode_length_s 30 --rl_device cuda:0
```

The other two command axes become zero. For straight/yaw use `--command_vx 0.5`
or `--command_yaw 0.8` instead. `--checkpoint -1` works within this selected run.
900 policy ticks mean 18 simulated seconds, excluding reset settling; the 30 s
episode timeout is separate and falls still reset. Export is opt-in with `--export`
and uses the selected run. `--help` is the complete argument reference.

TensorBoard reports 4.770 s rollout collection plus .914 s learning at update 700
(about 84%/16%, 46,119 environment steps/s). Profile collection/state readback and
logging synchronization first if iteration speed matters. No speed optimization
was implemented or measured; reducing PPO epochs, changing dt or omitting physical
checks cannot be advertised as preserving results without evidence.

**Next action:** resolve the new nominal pose's loaded target/support equilibrium
at the intended plant, then demonstrate a coordinated free-base lift before
implementing and piloting v3. No IsaacLab, navigation, estimator or hardware work
was performed.

## Frozen reference and phase-guided V3 — 2026-10-03

The owner replaced the preceding physical gate: RL may learn support offsets and
feedback around the desired actual posture. The failed fixed-target trial remains
a dynamic observation; it is neither a kinematic failure nor a proof that active
stabilization is impossible. No support controller, gravity compensation, IK action
override or per-tick pose setter was added.

### Frozen viewer and reference

`tests/go2w_pose_viewer.py` builds a bare measured-URDF/Plane scene once. It resolves
the 16 motor DOFs by name, writes q with `set_dofs_position(..., zero_velocity=True)`,
then writes an identity authored-base quaternion and a known height. Installed
Genesis 1.4.1 setters update forward kinematics. Joint state setters do **not** lock
an articulation; zero_velocity clears velocities instantaneously (all entity DOFs
on this API). This tool keeps state frozen by never advancing dynamics. It refreshes
`scene.visualizer.update(force=True)`, allowing the standard camera interaction,
and exits after the wall-clock duration, window close or Ctrl+C.

One headless measurement and a three-second GUI smoke completed. No camera-based
posture measurement or dynamic equilibrium is claimed. All four chains use the
shared reference hip=0, thigh=.70, calf=-1.40 rad; wheel angle zero is reset state,
not a wheel-position objective. Imported joint readback matches the requested
float32 tensor exactly. Authored quaternion [1,0,0,0] and roll/pitch 0/0 are imposed
and read back, not freely attained.

| Wheel | Floor gap at h_ref (m) | Collision center in base axes (m) | Center minus own thigh-origin x (m) |
|---|---:|---|---:|
| FL | 7.45e-9 | [.20203251, .19010000, -.33607170] | .00863251 |
| FR | 7.45e-9 | [.20203251, -.19010000, -.33607170] | .00863251 |
| RL | 7.45e-9 | [-.18476747, .19010000, -.33607170] | .00863253 |
| RR | 7.45e-9 | [-.18476747, -.19010000, -.33607170] | .00863253 |

Imported h_ref is **.427741706 m**; pure URDF FK used by the recipe gives
.427741656 m (50 nm difference). The reward reference is that model geometry,
with a 15 mm height deadband. Reset z is h_ref+.003 = **.430741656 m**, before
existing reset noise. The lowest nonwheel collision-vertex bound is .04185 m
above the plane; exact self-contact was not evaluated without a contact solve.
The import qpos0 warning concerns the import reference; explicit q was reapplied
after build and checked against limits. Nothing was simulated to make it settle.

From the repository root:

```powershell
conda activate genesis-gpu
python -m tests.go2w_pose_viewer --duration 60
python -m tests.go2w_pose_viewer --headless --output evaluation/pose_reference.json
# Optional elevated view; h_ref and the reward target stay unchanged:
python -m tests.go2w_pose_viewer --duration 60 --height-offset 0.003
```

### V3 contract and objective

`--go2w_profile transfer_v3` selects a fresh-only recipe. Old profiles and their
saved poses/rewards remain unchanged. Actor/Critic both consume `policy`, now
**58/58 inputs and 16 actions**, with the existing separate normalizers and
[512,256,128] ELU networks. Slots 0:56 retain the existing quantity/order/scaling/
noise/action-history semantics, but position errors refer to V3 q_ref. Slots
56:58 append noiseless, unscaled sin/cos of the phase before the normalizer.
Simulator velocity still means current body-axis velocity at the authored base
origin. No estimator or privileged critic exists.

At policy boundary t the Actor sees phase p. Four held-target .005 s physics
steps integrate its action; reward and terminal capture use **that same p**.
Then p advances by .02/.8 modulo 1 for the next observation. Getters and command
changes never advance/reset phase. A selected environment's episode reset assigns
uniform p in training and zero in nominal inference; the existing initial reset
includes one zero-action policy tick. Sin/cos both negate under sagittal reflection
(p -> p+.5), with the old velocity/joint/action reflection unchanged. Export records
this timing, reference pose/height, motor dynamics and its own embedded normalizer.
A 56-input Isaac player is not compatible without supplying this phase contract.

FL+RR have offset 0, FR+RL .5. With local phase u=(p+offset)%1, stance fraction
f=.65 and t=clamp((u-f)/(1-f),0,1), E=sin(pi*t)^2 during u>=f, otherwise zero.
There are all-four-stance overlaps, never a prescribed all-foot jump. Command
demand G uses smoothstep of max(clamp((|vy|-.01)/.04), clamp((|yaw|-.10)/.15)).
Desired swing S=G*E and desired cylinder gap h*=.04*S. These smooth phase/demand
envelopes use no observed-contact eligibility. Command steps are not filtered;
phase continues and motor-target transitions remain the Actor's responsibility.

Let H(z)=.5*z^2 for |z|<=1, else |z|-.5. All rows except fall are rates multiplied
by policy dt=.02 **once**. There is no total-positive clipping. Terms use actual
state, never equality of motor targets to q_ref. Continuous wheels have no absolute
angle objective. The complete nonzero reward set is:

| Term | Raw formula / reduction and units | Scale |
|---|---|---:|
| x / y / yaw tracking | Independent 1-H((command-actual)/s); s=.25 m/s, .15 m/s, .35 rad/s | 1 / 1 / .8 |
| Phase clearance | mean over four H((actual oriented-cylinder gap-h*)/.04 m), including loaded wheels and stance | -1 |
| Phase support | mean[S*min(F/F0,2)^2+(1-S)*relu(.2-F/F0)^2]; F is nonnegative wheel normal load, F0=nominal model weight/4 (~48.27 N) | -.5 |
| Orientation | sum(projected_gravity[:2]^2), dimensionless | -4 |
| Height | H(relu(abs(base_z-h_ref)-.015 m)/.05 m) | -1 |
| x corridor | mean H(relu(abs(dx-dx_ref)-(.04+.05*S) m)/.05 m); dx uses cylinder center minus own thigh ORIGIN in BASE axes | -.5 |
| Rolling actual pose | (1-G)*mean of 12 leg joint position-error squares, rad^2 | -.5 |
| Effort | leg mean + wheel mean of (control torque/limit)^2 | -.03 |
| Leg / wheel action rate | sum squared consecutive issued clipped action differences in each group | -.01 / -.005 |
| Vertical / roll-pitch motion | vz^2 (m/s)^2 / sum(omega_xy^2) (rad/s)^2 | -.2 / -.05 |
| Insufficient support | relu(2-count(F>6 N))^2 | -.5 |
| Nonwheel collision | nonwheel-contact count capped at 4 | -2 |
| Joint soft limits | inherited sum of radian excursions outside the soft P-joint range; no wheel angle limit | -2 |
| Effort limits | inherited sum relu(abs(torque)-soft_limit*effort_limit), Nm | -.5 |
| Lateral scrubbing | G*mean(loaded*u_lateral^2), (m/s)^2; axle-projected cylinder-center velocity, not point-foot rolling slip | -.2 |
| Fall | reset AND not timeout; discrete, no dt | -5 |

V3 constructs no WheelStepEvents machine; event, credit, dwell, dense-event swing
and prolonged-unloading rewards are absent. Existing offline geometric cycle
measurement remains available. Stance clearance/support and insufficient-support
costs oppose all-foot heave; actual vertical motion/height are also measured. The
x corridor neither penalizes y/z nor cancels opposing front/rear extensions.
No strong absolute-action penalty penalizes useful constant support offsets.

CPU examples show nonzero cost for a grounded requested swing and reduced cost
after a partial lift/unload, before any completed event. At ideal actual posture
with a .3 m/s lateral command but zero velocity, even the peak missed diagonal
swing/load costs leave a positive .8 reward/s; immediate falling costs -5. Mixed
command ranges are kept narrower so trying is not made intrinsically worse by
extreme combined startup errors. This consistency check does not guarantee PPO
cannot discover termination or motion artifacts; pilot episode duration, returns
and physical evaluation are required.

Segment draw probabilities: stand15%, straight20%, arc7%, pure yaw20%, precision10%,
pure lateral20%, mixed8%. Pure lateral magnitudes are .1-.3 m/s; pure yaw .3-.8
rad/s, with both signs. Straight/arc vx is -.35 to .6 m/s. Mixed ranges are
vx +/-.3, vy +/-.2 m/s, yaw +/-.5 rad/s. Precision retains the small-command
scaling. 80% of lateral/yaw/mixed segment draws use 2-4 s; other draws retain the
short/sustained mixture. Stand retains its 25% chance of a 3-6 s hold. These are
segment probabilities, not time fractions. No lateral-tail curriculum runs in V3.

The plant and PPO remain: measured embodiment, .01 nominal armature with [.005,.02]
training uncertainty, 40/1 legs, 0/1 wheels, zero passive assumptions, unchanged
action/force/velocity limits and mass/inertia randomization, .02/.005 s timing.
Native PPO uses 64 rollout ticks, LR3e-4/adaptive KL.01, gamma.995, lambda.95,
clip.2, gradient bound1, five epochs/eight minibatches, initial log std.4 bounded
[.1,.7] with the existing projection, entropy.003, and sagittal augmentation.
No optimizer, noise, gain or reward sweep was run.

### One fresh pilot and nominal inference

The single authorized pilot completed **200 updates / 4096 environments / 64 ticks**
(52,428,800 transitions), seed 1, in
`logs/go2w_transfer_v3_pilot/transfer_v3_pilot_seed1_2026-10-03_12-23-42`.
Native zero-based checkpoint labels are 0, 100 and **199**. No checkpoint initialized
it and no subsequent learning ran. Fresh actor/critic normalizers and optimizer,
58/58 dimensions, selective phase reset, getter purity, stored armature persistence,
finite state/losses and checkpoint round-trip were checked in that same process.
The deterministic exported Actor agreed within **1.67e-6** on 64 new observations,
including actual post-reset state. Export is under that run's `exported/model_199/`.
The existing plant/API assumptions were retained; this was not a gain calibration.

Available diagnostic labels 0 / 100 / 198 give measured mean minibatch KL
.10277 / .01301 / .01421, LR .00001 / .001946 / .001297, and effective std ranges
[.4000,.4005] / [.2805,.3798] / [.2093,.3456]. No joint reached the .1 floor.
Signed negative-reward fractions were 46.35% / .859% / .083%; total-reward clipping
was **zero** by design. At 198 the largest per-joint deterministic mean clipping
fraction was 3.20%. Clearance raw mean .03418 contributed -.000684/tick; support
.05701 contributed -.000570/tick; three tracking terms together contributed
.051630/tick. Raw kernels, weighted terms, exploration std and sensor/reset noise
are different quantities. No claim about V2's current clipping is inferred here.

The retained diagnostics JSONL has 188 rows (12 labels missing); TensorBoard has
190 collection-time records, last label 198. These gaps are recorded in
`evaluation/transfer_v3_pilot_preparation/pilot_summary.json`, not filled with zeros.
The console, final checkpoint and round-trip checks confirm all 200 updates. The
console's update 199 reports mean episode length 1000 ticks, mean return 47.14 and
4.32 s/update (3.41 collection + .91 learning). The full learn loop took 15m17s,
excluding startup. Collection is the first profiling candidate for speed work;
no speed optimization or claim of an equivalent faster setup was tested.
At diagnostic 198, actual time exposure was stand15.66%, straight12.19%, arc4.48%,
yaw25.99%, precision6.17%, lateral25.70%, mixed9.82%, distinct from segment draws.

One nominal inference process evaluated eight cases with fresh resets, no noise/DR,
.01 armature and zero delay. Stand lasts 10 s; other cases use 3 s zero, 5 s command,
6 s stop. All completed without falls, nonwheel contacts or censored intervals.
The table uses the **entire five-second command window**, not a selected good second.
Cycles require observed unloading/reloading (6/10 N hysteresis) and positive cylinder
clearance; >=2 mm counts have no dwell filter and can include contact chatter.
The >=2 cm column is per wheel in FL/FR/RL/RR order, excluding startup/drop.

| Command | Mean commanded axis (m/s or rad/s) | Axis RMSE | Mean height (m) | Mean roll/pitch (deg) | Completed >=2 cm cycles |
|---|---:|---:|---:|---|---|
| vx=.2 | .1798 | .0251 | .4001 | .14 / 1.71 | 0/0/0/0 |
| vx=.5 | .4248 | .0811 | .3995 | -.23 / 1.28 | 0/0/0/0 |
| vy=.3 | .2596 | .0546 | .4036 | -.83 / 5.15 | 1/0/0/6 |
| vy=-.3 | -.2508 | .0616 | .4028 | 1.21 / 5.40 | 0/1/6/0 |
| yaw=.8 | .7624 | .1088 | .4055 | 1.81 / 2.04 | 0/0/0/0 |
| yaw=-.8 | -.7754 | .1125 | .4048 | -1.74 / 2.00 | 0/0/0/0 |
| [.2,.1,.3] | [.1678,.0875,.2932] | [.0379,.0176,.0939] | .4021 | -.08 / 3.20 | 0/0/0/0 |

The positive-lateral RR wheel completed six 2.02-2.61 cm cycles, five starting more
than a second after command onset. Negative-lateral RL completed six 2.19-2.63 cm
cycles, also five later cycles. Their body-relative vertical rises were generally
1.2-2.1 cm and horizontal repositioning 12-16 cm: these are actual moving limbs,
with some body-motion contribution to ground clearance. The positive FL 4.43 cm
peak was only an onset transient. Other wheels participated at smaller clearances;
>=2 mm counts were 8/12/7/7 and 8/6/6/7. Yaw maxima were 1.60 cm / .75 cm, not
repeated centimeter-scale stepping in all wheels. The 4 cm desired apex was not
achieved as a sustained four-wheel pattern.

Lateral contact duties (load>8 N) were [.688,.700,.680,.632] / [.632,.728,.676,.636].
No sampled all-wheel-unloaded interval occurred; negative lateral had one of 250
command samples with fewer than two loaded wheels. Peak wheel loads reached
344.5/332.9 N. Lateral roll/pitch RMS was 1.44/5.75 and 1.60/5.96 degrees, with
pitch peak-to-peak 7.79/7.86 degrees. Compactness improved relative to the prior
extended V2 pose: mean body-axis front/rear cylinder spacing .409/.413 m, maximum
.457/.471 m; per-wheel |dx| remained below .090 m. Lateral counter-command peaks
in the requested axis were zero at the recorded policy boundaries. Deterministic
action clipping still reached 18.4% FL hip / 15.6% FR hip during lateral motion and
23.6% RL calf / 22.4% RR calf during yaw; this does not establish torque saturation.

Stand final-two-second planar/yaw RMS was .01164 m/s / .04206 rad/s and last-five-
second drift .0304 m, with mean height .4033 m and pitch 1.95 degrees. Across the
seven stop windows final planar RMS ranged .00951-.01679 m/s, yaw .03748-.08257
rad/s, and last-five-second drift .0255-.0406 m. Desired height remains .42774 m;
the learned actual height is still 2-3 cm lower, not a reason to silently redefine
the target. Periodic load transfer is visible even while standing. This pilot
shows useful lateral learning and repeated rear-wheel cycles, with uneven swing
participation, pitch oscillation, impact peaks and incomplete yaw stepping still
unresolved. It is not a finished gait, a Sim2Sim result or hardware validation.

Evidence: `evaluation/transfer_v3_pilot_preparation/{pose_reference.json,pilot_checks.json,pilot_summary.json}`
and `evaluation/transfer_v3_pilot_ck199_nominal/{metrics.json,*.npz,lateral_cycles.png}`.
These ordinary ignored outputs retain the traces and per-wheel events. Existing V1/V2
files and the old failed fixed-target probe were left intact. Focused CPU validation
passed 23 distinct checks across V3, legacy symmetry/replay and common CLI/config paths;
seven V3 tests were rerun after the final evaluation-only geometry capture change.
Syntax and diff checks passed. Ruff passed on the new/core changed files; the two
older evaluation modules retain the same 18 pre-existing import/semicolon findings.
No redundant learning smoke or second pilot ran. The only shared-code addition is
the small saved-interface hook; phase kernels, recipes, rewards, symmetry and
evaluation remain in the Go2-W package.

### Implemented commands after this pilot

Exact inference replay of the pilot (no export unless `--export` is supplied):

```powershell
conda activate genesis-gpu
python -m robot_gym.scripts.play --task go2w --experiment_name go2w_transfer_v3_pilot --load_run transfer_v3_pilot_seed1_2026-10-03_12-23-42 --checkpoint 199 --num_envs 1 --command_vy 0.3 --steps 900 --episode_length_s 30 --rl_device cuda:0
```

Omitted command axes are zero; use `--command_vx .5` or `--command_yaw .8` instead
for the other axes, or `--command_vx 0` for exact stand. `--checkpoint -1` selects
latest **within this selected run**. 900 ticks = 18 simulated seconds, while the
episode timeout is 30 s; falls still reset. The saved V3 config restores the 58-input
interface and new reference without repeating the profile argument.

Prepared fresh main command, **not executed** (2000 updates, save every 100):

```powershell
python -m robot_gym.scripts.train --task go2w --go2w_profile transfer_v3 --experiment_name go2w_transfer_v3 --run_name transfer_v3_seed1 --num_envs 4096 --max_iterations 2000 --seed 1 --logger tensorboard --training_diagnostics --rl_device cuda:0 --headless
```

This starts fresh; the pilot is never a training parent. The next useful action is
to inspect the saved pilot's lateral/yaw replay and load/clearance traces before
authorizing a separate main run. A later Isaac adapter must supply the 58-input
phase timing, new q_ref/height and its own exported normalizer/dynamics, retaining
simulator base-origin velocity. No Isaac, navigation or estimator changes were made.

## V3 sustained sensor-motion baseline — 2026-10-03

The completed main run is `go2w_transfer_v3/transfer_v3_seed1_2026-10-03_16-18-34`;
latest resolved once to `model_1499.pt` (stored iteration 1499, 1500 updates).
This is separate from the earlier 200-update pilot. Its saved recipe is authoritative.

`--eval_mode sensor_sustained` reuses the existing transfer evaluator, transition
capture and cycle counter. Stand is 30 s; moving cases are 3 s zero + 30 s command
+ 8 s zero, with a minimum 60 s episode timeout. The first failure/reset is retained
and later samples are censored. `--transfer_cases` selects a subset. With
`--diagnostic_trace`, only fast forward and positive strong lateral receive the
existing 200 Hz recorder; all cases retain 50 Hz state/action traces. Optional
`--eval_phase_offset .25` shifts only the initial episode phase; normal advancement
and command behavior remain unchanged. Phase-only action comparisons retain all
other recorded observation entries and are not a dynamical test.

The asset's fixed chain gives base-to-`front_realsense` translation
[.33881,.04750,.111] m and identity rotation, describing the left depth-imager
reference. RGB extrinsics, intrinsics and exposure are unknown. `radar` is at
[.28945,0,-.046825] m with fixed pitch 2.8782 rad; its physical LiDAR/TF association
needs eventual verification. Neither frame is an actuated gimbal.

Installed Genesis 1.4.1 `get_pos/get_quat/get_vel(relative=True)` use the authored
base origin; velocity is expressed in world axes, not at COM. The environment
rotates it into body axes. Sensor analysis rotates it back and applies exactly one
transport: p_C=p_B+R*r, v_C=v_B+omega_world cross (R*r). All rigid sensors share
base angular velocity. Metrics use world-up z, not sensor local/optical z. Height
drift, trend, raw variation and detrended variation are reported separately. No
numerical acceleration, filtering, pixel blur or hardware vibration is inferred.

```powershell
conda activate genesis-gpu
python -m robot_gym.scripts.evaluate --task go2w --experiment_name go2w_transfer_v3 --load_run transfer_v3_seed1_2026-10-03_16-18-34 --checkpoint 1499 --eval_mode transfer_screen --num_envs 1 --seed 1 --rl_device cuda:0 --headless --output evaluation/v3_main_nominal_review
python -m robot_gym.scripts.evaluate --task go2w --experiment_name go2w_transfer_v3 --load_run transfer_v3_seed1_2026-10-03_16-18-34 --checkpoint 1499 --eval_mode sensor_sustained --num_envs 1 --seed 1 --episode_length_s 60 --diagnostic_trace --rl_device cuda:0 --headless --output evaluation/v3_main_sustained_review
```

Use a new output directory: existing evidence is never overwritten.

### Measured main-run findings and optional sensor_smooth refinement

No optimizer updates were performed during this preparation. Local evidence is
`evaluation/go2w_sensor_smooth_preparation/summary.json`, with
`sustained_motion.png` and `late_lateral_cycles.png`. Raw traces remain in
`evaluation/transfer_v3_main_ck1499_{nominal,sensor_motion,phase_quarter}/`.
The first sustained process retained a valid 30 s stand trace in
`transfer_v3_main_ck1499_sensor_sustained/stand.npz`, then failed during offline
CPU/GPU tensor postprocessing. That helper was corrected and the stand was
processed offline (`stand_recovered.json`), without repeating or replacing it.

The main diagnostics contain 1414 rows, with 86 missing iteration labels and no
malformed rows. Updates 700–1000 have 290/301 rows; 1400–1499 have all 100.
Mean measured minibatch KL is .01320 / .01326 respectively; mean adaptive LR is
3.835e-4 / 3.628e-4. Mean per-joint std ranges .1000–.1767 / .1000–.1544.
Floor occupancy varies by joint: front thigh/wheel and rear wheel distributions
remain above the minimum; the mean does not imply every joint reached the floor.
The final checkpoint retains nine joints at approximately .1, with the others up
to .1550. Last-window deterministic clipping reaches 11.85% at rear calves and
8.44% at front wheels; sampled clipping reaches 12.45% and 8.37%. This is action
clipping, not torque saturation. V3's signed objective has zero total-positive
clipping. Raw and weighted terms are preserved in the JSON; final mean tracking
contributes +.05454/tick, rolling pose -.0000244/tick, roll/pitch rate
-.000173/tick, and body vertical velocity -.0000151/tick. Mean episode length is
999.55 policy ticks (20 s limit); the logged termination contribution is nonzero,
so training is not described as fall-free. No optimizer pathology is established.

All 13 nominal cases and all eight sustained cases completed without falls,
timeouts, nonfinite states or nonwheel contact. The extra reset-phase forward
case also completed. The following are full **30 s command** means/RMSE; sensor
RMS is the **20–30 s** window. Stand's mean/RMSE includes its initial settling.

| Command [vx,vy,yaw] | Mean [vx,vy,yaw] | RMSE [vx,vy,yaw] | Late left-imager world-vz RMS (m/s) | Late >=2 cm cycles FL/FR/RL/RR |
|---|---|---|---:|---|
| [0,0,0] | [-.0016,-.0003,.0016] | [.0094,.0043,.0166] | .01462 | 0/0/0/0 |
| [.2,0,0] | [.1830,-.0002,.0018] | [.0191,.0061,.0176] | .01508 | 0/0/0/0 |
| [.5,0,0] | [.4638,-.0007,-.0017] | [.0396,.0062,.0103] | .02100 | 0/0/0/0 |
| [0,.3,0] | [-.0028,.2952,.0036] | [.0183,.0215,.0456] | .05800 | 12/12/12/7 |
| [0,-.3,0] | [-.0014,-.2948,-.0068] | [.0178,.0216,.0445] | .05673 | 12/12/8/12 |
| [0,0,.8] | [.0073,.0009,.7984] | [.0118,.0076,.0339] | .07368 | 12/12/12/12 |
| [0,0,-.8] | [.0067,-.0027,-.8026] | [.0114,.0079,.0337] | .07302 | 12/12/12/12 |
| [.2,.1,.3] | [.1990,.0957,.2989] | [.0128,.0142,.0226] | .08122 | 12/12/12/12 |

Cycles require observed unloading <=6 N, reloading >10 N and positive cylinder
clearance; boundary intervals are censored. The 2 cm count adds a height threshold
to the evaluator's 2 mm definition, not the retired training event gate. Limb
horizontal displacement is measured separately. These are repeated late cycles,
not startup clearance. Late lateral peak gaps span 2.47–4.39 cm with contact duties
.486–.568. Lateral action clipping reaches 43.8% in one joint; observed effort-limit
fractions are zero. Positive-lateral sampled load peaks reach 486 N at 200 Hz,
versus 477 N at 50 Hz; these are sampled solver loads, not measured impact impulses.
The recorder does not prove continuous-time torque/force margins.

Fast-forward left-imager vertical RMS grows from .00591 (1–5 s) to .02121
(10–20 s) and .02100 m/s (20–30 s). Late base-origin RMS is .01347 m/s:
the camera lever arm adds meaningful motion. At 200 Hz these are .02158 and
.01355 m/s, so the low-frequency result is not a 50 Hz alias alone. Late camera
height std/peak-to-peak are 2.21/8.45 mm, with +5.14 mm endpoint drift; stand gives
1.52/5.03 mm and +2.49 mm. Drift is not removed from these raw measures. The fitted
1.25 Hz camera-height amplitudes are 2.77 mm forward and 2.04 mm stand (about
80%/90% of detrended variance). A .25-cycle reset offset changes late forward
camera RMS to .02033 m/s (about -3.2%), while full-command vx remains .46416 m/s.
Negating only recorded phase sin/cos changes per-joint actions (RMS .027–.199).
This supports learned phase dependence; it does not identify phase as the sole
cause or justify freezing deployment phase.

Late stand hip signed errors FL/FR/RL/RR are [.1252,-.1519,.0755,-.0035] rad,
versus [.0418,-.0427,.0126,-.0044] in 1–5 s. Fast-forward late errors are
[.1023,-.1173,.0625,.0042] rad. Mean base roll/pitch stays small (fast forward
[-.20,-.56] degrees); level mean attitude alone does not establish quiet motion.
Late maximum absolute wheel-under-thigh x separation stays <=3.36 cm in forward
and <=7.20 cm in stepping cases. After stopping, final-two-second planar RMS is
.0086–.0117 m/s; last-five-second drift is 1.04–5.09 cm. Stand ends at .0108 m/s
and 2.65 cm. Full early/middle/late/stop, hip p95, thigh/calf, sensor/radar, loads,
clipping and cross-axis metrics are in the JSON. One seed is evidence, not a
population estimate. No RGB/depth-quality, hardware-vibration or Sim2Sim claim follows.

The parent sampler really uses .5–1 s / 1.5–3 s timers, 2–4 s discovery holds
(80% of eligible step-demand draws), and some 3–6 s stands. Its legacy 8–15 s
moving branch is inactive for base V3. Last-100-update observed time exposure is
stand/straight/arc/yaw/precision/lateral/mixed =
14.84/12.18/4.37/26.07/6.10/26.00/10.44%. Duration exposure and weak rolling hip
cost are supported refinement hypotheses, not isolated causal ablations.

One optional `--go2w_finetune sensor_smooth` changes only the following settings.
Let G be the existing command step demand, e_j the actual leg q minus q_ref,
and v+ / v- the world-up velocities at the actual left-imager point and its
**virtual mirrored fixed location**. The virtual point is not calibrated RGB or
right-imager geometry. Both points use the asset-derived lever arm. Coefficients
below are reward rates; continuous accumulation multiplies by .02 s exactly once.

| Setting / raw quantity and units | Parent | sensor_smooth |
|---|---|---|
| Rolling pose: (1-G) * sum(w_j * e_j^2)/12, rad² | w_j=1; scale -.5 | Hip weights 3, other leg weights 1; scale -.5 |
| Body-frame vz², (m/s)² | scale -.2 | disabled |
| mean(v+²,v-²), (m/s)², world vertical | absent | scale -1.0 |
| omega_body,x² + omega_body,y², (rad/s)² | scale -.05 | scale -.075 |
| Stand/straight/arc additional timer draw | absent | 4%: 8–15 s; 1.5%: 20–30 s; otherwise existing draw |
| Episode timeout | 20 s | 60 s |
| PPO LR / schedule | initial 3e-4 / adaptive | 5e-5 / fixed |
| Budget / save interval | parent completed 1500 | 300 additional / 50 |

There is no added yaw/acceleration/jerk reward, sensor rendering, action filtering,
phase modification or hard hip lock. Hip weighting vanishes at full step demand;
actual joint angles are scored, not motor target offsets. The existing stance
objective may discourage an uncommanded recovery step; no recovery state machine
is introduced. Tracking, phase clearance/support, corridor, signed accumulation,
discrete -5 fall cost, pose, plant, sensor/reset noise, delay/DR, 58/58 observations,
16 actions, symmetry and .8 s phase remain as saved. Entropy stays .003, with the
same std bounds, native PPO epochs/minibatches and 64 rollout ticks.

On retained late traces the net added cost is 0.10–0.21% of the tracking reward
rate. On lateral/yaw/mixed traces the added hip cost is exactly zero; sensor/rate
costs remain small. This is a conservative shaping budget, not demonstrated
improvement. Segment probabilities are unchanged (.15/.20/.07/.20/.10/.20/.08).
In 10000 timer-only 60 s episodes, added 8–15/20–30 s holds occupy 7.72%/5.94%
of time; lateral/yaw remain 22.96% each. This is **sampler-only** exposure with no
physical falls, not observed refinement-training exposure. The existing scalar
logger now records actual selected-hold time, including truncation. Dispatch
explicitly excludes the legacy event/step-recovery mixed-yaw overrides.

The new opt-in path restores the selected parent's saved YAML before applying
these changes. It loads actor, critic, both normalizers/counts and learned
distribution through installed RSL-RL 5.5.1's native `load_cfg`:
`actor=true, critic=true, optimizer=false, iteration=false`. The changed objective
starts an empty optimizer/local counter at fixed LR 5e-5; the critic remains
trainable and must adapt to different returns. Parent iteration 1499 / 1500 prior
updates are lineage metadata. The original transfer fresh-start guards remain.
Do not pass `--resume`, and do not use this preparation's no-update export as a
training parent.

Validation: 24 focused CPU cases cover fixed geometry/transport/reflection,
saved configuration, selective reset/sampling, reward dt/signs, old shared replay
and V3 phase semantics. The four-environment native **no-update** check verified
empty optimizer, local iteration 0, LR 5e-5, both normalizer counts 393216000 and
all loaded model buffers/parameters. Deterministic parent/loaded/exported outputs
matched on 25 observations, including four actual post-reset observations; five
integration ticks had finite rewards. No `learn` call or optimizer step occurred.
The parent and pretrained bundles remain untouched. The exported check artifact
is parent-equivalent wiring evidence, not a refined policy.

GUI replay, fixed forward for 2050 ticks = 41 s (omitted vy/yaw are zero):

```powershell
conda activate genesis-gpu
python -m robot_gym.scripts.play --task go2w --experiment_name go2w_transfer_v3 --load_run transfer_v3_seed1_2026-10-03_16-18-34 --checkpoint 1499 --num_envs 1 --command_vx 0.5 --steps 2050 --episode_length_s 60 --rl_device cuda:0
```

This GUI command holds vx throughout; the evaluator commands above implement the
3+30+8 s schedule. Replay ticks span physical resets; a longer episode timeout
does not disable falls. Export remains opt-in under the selected run. Ordinary
`--checkpoint -1` works, but 1499 pins this measured baseline.

Prepared command below is **not executed**. It initializes from the explicit
parent checkpoint and writes a timestamped new run below
`logs/go2w_transfer_v3_sensor_smooth/`:

```powershell
conda activate genesis-gpu
python -m robot_gym.scripts.train --task go2w --go2w_profile transfer_v3 --go2w_finetune sensor_smooth --load_run logs/go2w_transfer_v3/transfer_v3_seed1_2026-10-03_16-18-34 --checkpoint 1499 --experiment_name go2w_transfer_v3_sensor_smooth --run_name sensor_smooth_from1499_seed1 --num_envs 4096 --max_iterations 300 --seed 1 --logger tensorboard --training_diagnostics --rl_device cuda:0 --headless
```

Compare approximately +100 (`model_100.pt`, native zero-based label) and +300
(`model_299.pt`) using the **same two evaluator commands**, changing only selected
experiment/run/checkpoint and output directory. Keep both signs and late windows.
Provisional development comparisons, informed by the ~3% alternate-phase forward
variation: seek >=15% lower late camera vertical RMS and hip RMS in stand/straight,
with no material increase in height drift or roll/pitch rate. Require no new falls,
resets or nonwheel contacts; full-phase tracking RMSE should not worsen by more
than max(10%, .005 m/s or .01 rad/s), stop drift by more than 1 cm, or per-wheel
late >=2 cm cycle counts by more than 10% rounded up to one cycle. Check clearance,
contact duty, body/foot posture and action saturation alongside those numbers;
slower motion, fewer steps or an extended wheelbase is not a smoothness improvement.
These are provisional comparison bands from limited variation, not hardware
safety limits or an automatic acceptance rule. Preserve the parent if refinement
fails. Its unchanged 58-input phase bundle can proceed to the separately authorized
first IsaacLab transfer without waiting for perfect camera stabilization.
