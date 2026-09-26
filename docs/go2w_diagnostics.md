# Go2-W diagnostics and reproducibility — 25 September 2026

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
