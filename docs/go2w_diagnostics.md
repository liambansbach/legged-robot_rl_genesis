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
