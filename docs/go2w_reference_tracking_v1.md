Consolidated fresh-training preparation and final configuration — 6 October 2026
======================================================

Close the fixed-LR experiment. Final998 tracks forward speed better than ORIGINAL1499 but curves substantially, remains crouched, has poorer cross-axis tracking during stepping, and does not recover original physical clearance. It is not an overall success. The single proposed next task is **reference_tracking_v1**, a combined engineering redesign, not an isolated causal ablation or a promise of success. There is no proposed unchanged extension.

The [recipe YAML](go2w_reference_tracking_v1.yaml) and [unexecuted PowerShell command](go2w_reference_tracking_v1.ps1) prepare fresh actor, critic, normalizers, Gaussian distribution and Adam. Production training was not started. Work remains on `testing`; `play.py`, historical artifacts and IsaacLab are unchanged. No packages were upgraded and nothing was committed or pushed.

Evidence is in [evaluation/consolidated_v3_preparation](../evaluation/consolidated_v3_preparation/). The [source inventory](../evaluation/consolidated_v3_preparation/sources.json), [measurements](../evaluation/consolidated_v3_preparation/measurements.json), [calibration](../evaluation/consolidated_v3_preparation/calibration.json), [initial preparation configuration](../evaluation/consolidated_v3_preparation/resolved_fresh_config.yaml), and [initial differences](../evaluation/consolidated_v3_preparation/validation.json) contain the detailed arrays and provenance. Implementation evidence is the current repository and actual saved configurations, not uploaded Python snapshots.

The final configuration cleanup first checked local artifacts: the production experiment `logs/go2w_transfer_v3_reference_tracking_v1` did not exist, and the only saved `reference_tracking_v1` configuration belonged to the earlier two-update smoke (`model_0.pt` and `model_1.pt`). The [before/after audit](../evaluation/reference_tracking_v1_cleanup/audit_before.json) records the pre-edit inventory. The [final resolved configuration](../evaluation/reference_tracking_v1_cleanup/resolved_fresh_config.yaml) and [cleanup differences](../evaluation/reference_tracking_v1_cleanup/validation.json) supersede the initial preparation configuration for future production. No production command has run.

The asymmetric `GO2WCfg.init_state` was an **overridden legacy/default inconsistency for V3, not an active V3 reference bug**. It remains active when reconstructing unsaved `event_step_v1` and `step_recovery_v1`: front .70/−1.33, rear .75/−1.31, z=.45 m. Those class values are now explicitly marked legacy; changing them globally would change supported historical recipe reconstruction. `apply_go2w_profile(..., 'transfer_v3')` already replaces them through `v3_joint_reference()`. The fresh entry reads the complete YAML, rejects any differing joint reference, then obtains its mapping from that same function. Historical replay continues to restore saved configurations without this fresh-entry rewrite.

The canonical reference is identical for **FL, FR, RL and RR: hip 0.00, thigh +0.70, calf −1.40, foot 0.00 rad**. Measured-URDF frozen FK gives reference height **.4277416561558192 m**, spawn clearance **.003 m**, and nominal spawn z **.4307416561558192 m**. The height target is the frozen geometry height, not loaded equilibrium. The fresh config validates the exact 16 named references and FK height/spawn consistency; no physical equilibrium experiment was performed.

One `default_dof_pos` tensor supplies action-zero leg targets, observation `q−q_ref`, actual reference-pose reward and reset nominal pose before joint noise. Frozen FK uses `v3_joint_reference()` and its cached geometry supplies the existing height/corridor references. For each leg, `q_target = q_ref + [.30,.35,.40] * clipped/delayed_action` (rad); each wheel uses `qdot_target = 18 * clipped/delayed_action` (rad/s). Normalized clipping stays ±1, and the current per-environment 0–2 policy-tick delay, gains and force limits are unchanged. These are fixed-reference offsets, never absolute outputs, accumulated increments or offsets from the noisy reset pose. Zero leg action is not required to be a static equilibrium.

The completed stage is `logs/go2w_transfer_v3_navigation_rolling_control_fixed_lr_stage2/lr_only_from499_stage2_seed1_2026-10-06_13-28-54/model_998.pt`. Its internal iteration is 998, Adam step is 40040, and both normalizer counts are 262406144 = 1001 × 4096 × 64. Preparation says completed, with 500 additional updates; diagnostics span labels 499–998 with 497 retained rows and fixed LR .0003 throughout. The repeated resume label explains **1001 lineage updates**, not 999. ORIGINAL1499 is `logs/go2w_transfer_v3/transfer_v3_seed1_2026-10-03_16-18-34/model_1499.pt`: iteration 1499, Adam step 60000 and counts 393216000 support 1500 updates. Its saved configuration supplies the original design; its weights are never fresh-run initialization.

Exactly **one evaluator launch**, five nominal deterministic cases and 97 seconds of policy-controlled simulation were used: forward .5 with 3 s startup + 30 s hold + 8 s stop; lateral ±.1 and yaw ±.4 with 3 + 5 + 6 s. All five original command arrays, latent phases and nominal asset/control/reset/physics/noise/DR settings match the retained original traces exactly. Each model retains its own saved clock and reward semantics. Original forward comes from `transfer_v3_main_ck1499_sensor_motion/forward_fast.npz`; signed short cases come from `transfer_v3_main_ck1499_nominal`. No original replay, sweep, robustness bank, repeated geometry audit or static-pose simulation was needed.

The principal matched forward measurements are:

| Quantity | ORIGINAL1499 | Final998 |
|---|---:|---:|
| Full 30 s mean vx / RMSE, m/s | .46378 / .03962 | .48705 / .02467 |
| Maximum cross-track from initial heading line, m | .12865 | **1.62938** |
| Maximum pose-derived heading deviation, degrees | 2.9157 | **9.9914** |
| Height, early 1–5 → late 20–30 s, m | .40962 → .40942 | .37639 → .38160 |
| Late pitch, rad; positive means nose down | −.00969 | +.08839 |
| Late actual 12-joint reference-error RMS, rad | .08675 | .17992 |
| Late wheelbase, m | .36457 | .35616 |
| Late front / rear track width, m | .45197 / .39881 | .44902 / .39044 |
| Full 8 s stop endpoint / path, m | .03156 / .07945 | .01744 / .10186 |
| Stop backward excursion from stop onset, m | 0 | .00633 |
| Final 2 s planar velocity RMS, m/s | .00864 | .00840 |
| Final 2 s yaw RMS, rad/s | .01666 | .00732 |
| Late imager / radar world-z velocity RMS, m/s | .02100 / .01959 | .000094 / .000076 |

Final998's height rises slightly during this hold; these data show a persistent crouch, not progressive lowering during the measured 30 seconds. Posture still migrates: early-to-late wheelbase .31968 → .35616 m, front track .41130 → .44902 m, then mean stop wheelbase .31424 m. The stop endpoint hides reversal and a path 28% longer than original. Very small late sensor motion does not qualify this trajectory.

For short signed cases, the following are full moving-phase means and RMSEs, in vx/vy/yaw order (m/s, m/s, rad/s):

| Command | ORIGINAL1499 mean; RMSE | Final998 mean; RMSE |
|---|---|---|
| Lateral +.1 | −.0003, .0964, −.0013; .0110, .0142, .0247 | −.0386, .0947, −.0049; .0435, .0164, .0877 |
| Lateral −.1 | .0037, −.0947, .0007; .0127, .0132, .0259 | −.0360, −.0900, .0140; .0405, .0186, .0951 |
| Yaw +.4 | .0017, .0024, .3906; .0109, .0103, .0357 | −.0201, −.0079, .3743; .0296, .0213, .0799 |
| Yaw −.4 | .0047, .0000, −.3902; .0113, .0100, .0321 | −.0137, .0159, −.3729; .0233, .0242, .0766 |

Final998's moving-phase wheel clearance peaks, FL/FR/RL/RR in mm, are lateral+ 11.64/6.78/3.84/5.08, lateral− 5.82/19.56/13.32/4.97, yaw+ 7.63/11.71/6.70/5.47, yaw− 9.62/11.83/4.42/9.78. Original peaks span 24–45 mm; requested apex remains 40 mm. These peaks describe the measured short cases, not completed-cycle counts. Existing evaluator metrics retain individual physical swing events. No apex increase is proposed. Full-moving imager/radar RMS is final lateral+ .0844/.0727, lateral− .0839/.0783, yaw+ .0790/.0703, yaw− .0713/.0680 m/s; smoothing does not explain away the poorer tracking and clearance.

There are no falls, resets, non-wheel ground contacts or sampled self-contact pairs in the five final998 cases. Self-contacts were explicitly read, including calf/calf pairs, at 50 Hz. Original retained traces do **not** contain self-contact records; ground-contact zero is not evidence of self-contact zero. Boundary sampling does not establish physics-substep maxima or hardware safety.

Actual late-forward leg poses (hip/thigh/calf, radians) and collision-center positions in authored-body x/y (meters) are:

| Leg | Original actual pose | Final998 actual pose | Original center x/y | Final998 center x/y |
|---|---|---|---|---|
| FL | .1023 / .8029 / −1.4152 | .1996 / .8903 / −1.5812 | .1703 / .2234 | .1721 / .2484 |
| FR | −.1173 / .7878 / −1.3868 | −.0381 / .9285 / −1.7710 | .1700 / −.2286 | .1918 / −.2006 |
| RL | .0625 / .7777 / −1.4861 | .0276 / .6425 / −1.5605 | −.1955 / .2100 | −.1412 / .1985 |
| RR | .0042 / .7976 / −1.5368 | −.0058 / .8640 / −1.5775 | −.1933 / −.1888 | −.2072 / −.1919 |

Reference pose is **0 / .70 / −1.40 for every leg**. Cached measured-URDF reference centers are front (.202033, ±.190100), rear (−.184767, ±.190100): wheelbase .386800 m; both tracks .380200 m. Wheelbase is front-pair mean x minus rear-pair mean x, never a width or image estimate. Late final pair midpoints x/y are (.18197, .02390), (−.17419, .00333); original (.17015, −.00260), (−.19442, .01062).

Neither policy has late-forward action clipping or force saturation. Final per-leg actual-minus-applied target RMSEs are FL .0375/.0585/.1179, FR .0336/.0032/.1775, RL .0647/.0171/.1140, RR .0592/.0107/.1197 rad. Wheel target RMSE is .1752/.1776/.0934/.1160 rad/s. Actual pose can differ materially from its support target without clipping. Raw, clipped-issued and applied targets, signed clipping fractions, loads, clearances, joint errors, height/pitch, heading, path and sensor windows are retained in `measurements.json` and NPZs.

The following is the complete before/after active-objective table. H(z) = z²/2 for |z| ≤ 1, otherwise |z| − 1/2. All entries describe **reward rates**; the accumulator multiplies by policy dt .02 s exactly once. Termination alone is discrete. Coefficients are reward per stated raw unit per second, not policy-gradient magnitudes. `d` is the unchanged V3 stepping demand; `s_j` is its diagonal swing envelope; `u_j=max(Fz_j/F0,0)` uses nominal weight/4. A dash means inactive.

| Objective and raw formula / units | Original1499 coefficient | Final998 coefficient | New coefficient / behavior and purpose |
|---|---:|---:|---|
| Axis tracking: requested − actual e, m/s or rad/s | [1,1,.8] × (1−H(e/[.25,.15,.35])) | Same; yaw scale .07 for pure rolling/stand, .35 otherwise | **[1,1,.8] × [1−H(e/[.25,.15,.35])−.25(1−exp(−.5(e/.03)²))]** on every axis at every command; acquisition plus bounded precision |
| Old rolling pose: (1−d) mean leg angle error², rad² | −.5 | −.5, hip squared errors ×3 | —; replaced, including the extra hip multiplier |
| Actual reference pose: A(c) mean H((q_leg−q_ref)/[.08,.12,.12]), dimensionless | — | — | **−.5**; one normalized actual-joint objective; no wheel angles, action amplitudes or target offsets |
| Old y placement: rolling-only mean H(width excess/.10)+H(midpoint excess/.05), with .04/.015 m deadbands | — | −.05 | —; redundant with actual reference pose |
| Wheel-thigh x corridor: mean H(relu(|dx−dx_ref|−tolerance)/.05), dimensionless; tolerance .04+.05s m | −.5 | −.5 | **−.5d**, stepping reposition guidance only; zero at pure longitudinal/stand |
| Reference height: H(relu(|h−.427741656|−.015)/.05), dimensionless | −1 | −1 | **−1**, unchanged vertical support geometry |
| Level orientation: projected gravity gx²+gy², dimensionless | −4 | −4 | **−4**, unchanged tilt control |
| Phase clearance: mean H((actual gap−.04s)/.04), dimensionless | −1 | −1 | **−1**, unchanged physical lift objective, including while loaded |
| Phase support: mean [s min(u,2)²+(1−s)relu(.2−u)²], dimensionless | −.5 | −.5 | **−.5**, unchanged swing unloading/stance loading |
| Insufficient support: relu(2−count(Fz>6 N))², dimensionless | −.5 | −.5 | **−.5**, avoid fewer than two loaded wheels; distinct from phase allocation |
| Sensor vertical: mean world-z velocity² at left imager and virtual sagittal mirror, m²/s² | — | −1 | **−1**, one transported rigid-sensor objective |
| Base vertical velocity², m²/s² | −.2 | — | —; sensor objective replaces it |
| Body roll/pitch rates: wx²+wy², rad²/s² | −.05 | −.075 | **−.075**, unchanged from final998; dynamic tilt-rate control, no new angular/yaw term |
| Old stand_still: exact-zero × (vx²+vy²+wz²+.02 mean wheel rate²), mixed units | — | −5 | —; duplicated body tracking removed |
| Zero-command wheel rate: exact-zero × mean wheel rate², rad²/s² | — | Included above, effective −.1 | **−.1**, retained separately to discourage idle wheel spinning/slip invisible to body velocity |
| Ground collision: capped non-wheel manifold count | −2 | −2 | —; replaced by contact safety |
| Contact safety: min(unique non-wheel ground links + unique nonadjacent robot self-pairs,4) | — | — | **−2**, includes formerly missed calf/calf contact; 8 N threshold; no point-manifold duplication |
| Normalized effort: mean legs (tau/limit)² + mean wheels (tau/limit)², dimensionless | −.03 | −.03 | **−.03**, actuator effort surrogate |
| Leg action changes: sum (a_t−a_prev)², dimensionless | −.01 | −.01 | **−.01**, target slew regularization |
| Wheel action changes: sum (a_t−a_prev)², dimensionless | −.005 | −.005 | **−.005**, velocity-target slew regularization |
| Loaded lateral wheel-center speed: d mean loaded × speed², m²/s² | −.2 | −.2 | **−.2**, stepping scrub surrogate |
| Leg soft-position violation: sum excess beyond 90% URDF interval, rad | −2 | −2 | **−2**, joint-limit margin; continuous wheels excluded |
| Effort-limit excess: sum relu(|tau|−.9 limit), Nm | −.5 | −.5 | **−.5**, peak actuator-limit margin |
| Fallen/non-timeout termination, indicator | −5 discrete | −5 discrete | **−5 discrete**, unchanged |

No other reward is active. The new identity selects the common kernel before the old yaw-switch branch; the old conditional kernel is not added. Positive rewards are not clipped. All velocities retain the existing authored-base-origin, body-frame convention. No world pose enters actor observations; no estimator, heading controller or inference action override is added.

Pose activation is A = 1 − .95 [1 − (1−S(|vy|/.03))(1−S(|yaw|/.25))], with arguments clipped to [0,1] and S(x)=x²(3−2x). Thus A=1 for pure longitudinal/stand, .2963 at lateral .02, and .05 at lateral .03–.05 or yaw .25 and above. This smooth signed-symmetric union protects small lateral commands and retains a small finite pose preference during stepping. It has no measured-contact or unloading escape gate and no hard lock. The reference and per-joint scales are cached; rewards perform no FK. Diagnostic wheel centers use the existing collision-offset cache.

The single selected calibration uses broad scales [.25,.15,.35], weights [1,1,.8], precision scales [.03,.03,.03], beta [.25,.25,.25], pose scales [.08,.12,.12] rad and pose coefficient −.5. Recorded normalized pose-cost means / 90th percentiles are reference-like original .3218/.3494, final998 crouch .9217/.9239, sensor499 splay .5427/.5513, and failed1000 clench 2.7276/2.7821. Their mean new pose rates are −.1609, −.4609, −.2713 and −1.3638. These are recorded windows, not synthetic evidence that a policy will optimize them; synthetic poses separately test signs and ordering. Exact reference cost is zero even with nonzero wheel angles and nonzero supporting actions.

At final998 late forward, new pose/height/level rates are −.46087/−.19393/−.03126. Original late values are −.17480/−.00250/−.00054. Height and level retain distinct physical purposes and unchanged coefficients. The new zero-wheel term contributes −.01188 reward/s across final998's stop and −.00105 in its last 2 s, versus −.06353/−.03732 on original traces. Its coefficient is the old wheel component, extracted without the duplicate body-velocity/yaw penalty.

For a .03 error, the bounded precision cost is .09837 before axis weight; at .10 it is .24903; its ceiling is .25. For x acquisition or stopping errors ±.5, the total weighted tracking rate is −.75; at ±1 it is −2.75. Broad sensitivity remains after precision saturates. The kernel treats equal requested-minus-actual error identically at start, hold and stop, including zero cross axes during lateral/yaw/mixed commands. Retained acquisition, hold and stop windows have separate broad and precision contributions in `measurements.json`; coefficients were not chosen from aggregate maxima or interpreted as gradients.

Named contact masks are constructed once from actual robot link names and URDF parent/child edges. Same-link and intended adjacent-link pairs are excluded. A self-pair qualifies when any valid manifold point has force norm >8 N; it is counted once, not once per point. Ground uses the existing >8 N world-z force convention, deduplicated by named link. The existing contact getter supplies forces; no net impulse is inferred. Physical self-collision stays enabled. Historical recipes still use their exact old ground-only collision method. This is a narrow addition to the safety objective, not a new collision engine.

The final command-generation choices are all in the YAML's `commands` section. `commands.ranges` now means the required navigation output domain; `reference_tracking_sampling.reserve_ranges` separately records the training reserve. Both are exported in deployment metadata for later Sim2Sim/navigation integration.

| Axis | Required/core envelope | Training reserve | Unit |
|---|---|---|---|
| vx | [−.30, +1.00] | [−.40, +1.20] | m/s |
| vy | [−.30, +.30] | [−.50, +.50] | m/s |
| yaw | [−1.00, +1.00] | [−1.50, +1.50] | rad/s |

The family probabilities remain stand .15, straight .20, arc .07, pure yaw .20, precision .10, pure lateral .20 and mixed .08. Within the ordinary moving families (straight, arc, yaw, lateral and mixed), the segment-tier probabilities are **80% core / 10% required edge / 10% reserve**. Core draws uniformly cover the required intervals on each active axis. Pure lateral uses signed uniform magnitude [.02,.30]; pure yaw uses [.05,1.00], each with balanced signs. Straight has only vx; arcs have vx/yaw; mixed has vx/vy/yaw. The former narrow mixed override no longer caps the new sampler; the serialized compatibility `phase_mixed_ranges` also agrees with core.

An edge draw selects one active axis uniformly and sets it exactly to either required signed limit, with equal sign probability. A reserve draw selects one active axis uniformly and draws its magnitude between that signed required boundary and reserve boundary; all other active axes retain core draws. Thus a reserve command has exactly one axis outside core, and never simultaneous reserve extremes. Most mixed commands remain entirely within the required domain. Stand remains exactly zero. The precision family is exempt from edge/reserve draws and retains its previous small ranges vx [−.063,.108], vy=0, yaw [−.16,.16]. Linear-vector and yaw deadzones remain .01, preserving explicit .02 forward/reverse/lateral and .05 yaw requests. The new command option is absent from historical saved runs, including the earlier reference-tracking smoke; those use their exact old sampler.

Base holds remain 70% .5–1 s and 30% 1.5–3 s. Stand retains its 25% 3–6 s replacement; the existing eligible yaw/lateral/mixed discovery draws retain 80% 2–4 s holds. Eligibility still uses the existing stepping-demand function, so very small yaw can keep shorter holds. Stand/straight/arcs retain 4% 8–15 s and 1.5% 20–30 s holds. No duration policy, phase setting or curriculum phase was added. Dormant event-profile lateral-tail/mixed-zero-yaw fields in this fully resolved YAML do not override the phase-guided sampler.

One [prospective exposure sample](../evaluation/reference_tracking_v1_cleanup/command_exposure.json) used the **actual final `_resample_commands` function**, 100000 segments, CPU seed 20261006, and the sampled integer policy-tick durations. It is an uncensored renewal estimate, **not historical occupancy**; episode resets, failures and finite rollouts can change realized time shares.

| Family | Configured segment share | Sampled segment share | Duration-weighted share | Mean hold |
|---|---:|---:|---:|---:|
| Stand | 15% | 14.76% | 17.54% | 2.75 s |
| Straight/reverse | 20% | 20.18% | 17.15% | 1.97 s |
| Arc | 7% | 6.99% | 6.00% | 1.99 s |
| Pure yaw | 20% | 20.13% | 22.35% | 2.57 s |
| Precision | 10% | 10.08% | 5.24% | 1.20 s |
| Pure lateral | 20% | 19.91% | 22.67% | 2.64 s |
| Mixed | 8% | 7.95% | 9.06% | 2.65 s |

| Sampling tier | All-segment share | Duration-weighted share |
|---|---:|---:|
| Core, including stand/precision | 84.98% | 84.49% |
| Explicit required edge | 7.57% | 7.77% |
| Reserve | 7.45% | 7.73% |

Within ordinary moving segments alone, the realized tier split is 80.01% / 10.08% / 9.92%. Exactly **7452/100000 commands (7.452%)** are outside the required envelope, always on one axis; **zero** exceed reserve or have multiple axes outside core. Explicit negative/positive required-edge counts are vx 1394/1324, vy 1085/1114 and yaw 1332/1323. Observed extrema vx [−.399985,1.199893], vy [−.499507,.499980], yaw [−1.499734,1.499721] remain within the reserve.

Positive/negative shares of all sampled time are vx **26.38%/10.14%**, vy **16.03%/15.70%**, yaw **21.26%/20.92%**; the remainder is zero on that axis. The asymmetric required vx domain intentionally yields more forward exposure; edge and reserve signs themselves are balanced. Pure forward/reverse occupy **12.47%/4.82%** of time (straight-family alone 12.24%/4.70%). Small pure longitudinal magnitudes .02–.05 occupy **.342% forward/.341% reverse**. Pure lateral .02–.05 occupies **1.986%** (.981% positive/1.005% negative); pure yaw .05–.15 occupies **1.841%** (.946% positive/.894% negative).

All eight mixed sign combinations appear (462–1524 segments per octant). Mixed 5th/50th/95th percentiles are vx [−.279,.349,.978], vy [−.291,−.00037,.289], yaw [−.960,0,.964]. Simultaneous |vx|>.1 and |yaw|>.2 occupy **10.57%** of time; adding |vy|>.05 retains **5.38%**. Holds <1.5 s, 1.5–8 s, 8–20 s and ≥20 s occupy **12.77%, 71.99%, 8.60%, 6.64%** respectively. Short changes and long drift/posture exposure are both retained.

Body pushes are changed only through this recipe's explicit `domain_rand.push_force_mixture`. Each event draws one magnitude tier and one horizontal azimuth, independently of command sampling:

| Event probability | Horizontal force magnitude |
|---|---|
| 80% | Uniform [20,50] N |
| 15% | Uniform [50,100] N |
| 5% | Uniform [100,150] N |

Intervals are uniformly sampled integer **250–500 policy ticks = 5–10 s**; duration is **20–40 physics ticks = .10–.20 s**. The nominal uncensored mean interval is 7.5 s: **8 events/minute/environment**, including **.4 strong events/minute** (one 100–150 N event per 150 s on average). This replaces the prior uniform 20–50 N, 3–6 s exposure (13.33 events/minute). Mean active push time is approximately 2%; resets may delay the next event or truncate one. Force and direction stay fixed during each event. The existing physics-step reapplication at the base-link COM remains; no vertical force, external torque, actuator-limit or physics change is made. The vector norm is capped at 150 N, including a floating-point endpoint guard. Strong pushes are not synchronized with reserve commands.

Diagnostics retain `push_event_duration_steps` in physics ticks and `push_event_impulse_ns = ||F|| * duration_steps * sim.dt`. The latter is the full sampled event impulse using the actual tick-quantized duration; an episode reset can truncate delivered impulse. Event-weighted training aggregates record tier fractions, sampled duration ticks and impulse once per event, not once per reapplication; transition captures expose the latest sampled event. Nominal complete-event impulses span 2–30 N·s, with mean 6.825 N·s before reset censoring. Old recipes without this option still use the original generic push function and their saved distribution/timing.

Fresh production is **4096 environments × 64 ticks, 2000 updates, save interval 250 and native final model_1999**, fixed LR 3e-4, gamma .995, lambda .95, initial learned log-Gaussian std .40 bounded [.10,.70]. Separate normalized ELU actor/critic networks remain 512/256/128; Adam, 5 PPO epochs × 8 minibatches, clip .2, entropy .003, value coefficient 1, gradient clip 1, sagittal augmentation and no mirror loss remain unchanged. Desired KL .01 remains saved metadata; fixed scheduling does not adapt LR.

Measured URDF SHA-256 `d298cc7bf4894e869840bdab9ac60f09548d998444018e61854636cff46b1d8c`, reference joint signs, P-leg/V-wheel contract, gains, limits, armature and all DR other than the explicit body-push change are identical to saved ORIGINAL1499. There are 16 actions and 58 current proprioceptive inputs. Policy/physics dt remain .02/.005 s. Demand-conditioned clock semantics are retained from final998; V3 period .8 s, stance fraction .65, half-cycle diagonal offsets, 4 cm apex, clearance and support targets are unchanged. No desired height is lowered. Training uses the retained randomized plant with the explicit body-push distribution above, not the nominal evaluator's disabled DR.

The fresh path reads configuration only, clears checkpoint selectors and both ancestry fields, and fixes this recipe's LR schedule. Ordinary `--resume` remains full-state model/normalizer/Gaussian/Adam/iteration resume and preserves saved semantics. This does not initialize the proposed production run from the two-update smoke. Execute the following only when actually choosing to start the new run; it has **not** been executed:

```powershell
Set-Location 'C:/Users/Liamb/SynologyDrive/TUM/3_Semester/dodo_alive/legged-robot_rl_genesis'
$env:NUMBA_CACHE_DIR = Join-Path (Get-Location) '.cache/numba'
$env:GS_CACHE_FILE_PATH = Join-Path (Get-Location) '.cache/genesis'
$env:QD_OFFLINE_CACHE_FILE_PATH = Join-Path (Get-Location) '.cache/quadrants'
$env:PYTHONIOENCODING = 'utf-8'
& 'C:/Users/Liamb/anaconda3/envs/genesis-gpu/python.exe' -m robot_gym.scripts.train `
  --task go2w --go2w_profile transfer_v3 `
  --go2w_fresh_recipe docs/go2w_reference_tracking_v1.yaml `
  --num_envs 4096 --max_iterations 2000 --seed 1 `
  --logger tensorboard --training_diagnostics --headless --rl_device cuda:0
```

**The initial objective preparation passed fourteen focused tests** (9 new-recipe, 3 historical fresh-entry and 2 historical rolling-control), as does `git diff --check`. They cover requested optima on all axes and zero cross axes; symmetric start/stop costs, bounded precision and large-error broad sensitivity; exact old kernels; actual reference preference over crouch/splay/clench, correct joint signs, smooth relaxation and absence of wheel-position/action-amplitude costs; stepping-only corridor; separate wheel-at-zero cost; dt once; configuration-only fresh state; full-state resume; deduplicated nonadjacent self-contact handling. No exhaustive repository suite was run.

The **earlier preparation smoke** used the intended 4096 × 64 batch with training DR intact. Before updates: checkpoint null, Adam empty, both normalizer counts zero, iteration zero, std .40; runner checkpoint loading was patched to fail if attempted. After updates: Adam step 80, counts 524288, finite actor/critic, LR fixed .0003 and native model_1.pt saved. Named FL-calf/FR-calf is eligible; adjacent thigh/calf is excluded; physical self-collision remains enabled. [Before](../evaluation/consolidated_v3_preparation/smoke_before.json) and [after](../evaluation/consolidated_v3_preparation/smoke_after.json) record this. A smoke establishes execution and state handling, not learned behavior.

The final configuration cleanup passes **13 focused checks**: eight canonical-reference/action/sampling/push checks, three historical fresh-entry checks and two reference-tracking fresh/full-resume checks. The strong 100–150 N branch is exercised by deterministic quantiles, including both tier boundaries and the vector-norm endpoint. The actual action loop is tested across clipping, delays 0/1/2, repeated targets and noisy reset positions. Fresh resolution patches checkpoint loading to fail if attempted. A before/after CPU audit confirms identical historical recipe dictionaries and bit-for-bit sampler output plus RNG state for event-step, step-recovery, base V3 and the earlier reference-tracking smoke. All 89 pre-existing audited files retain their hashes, including saved configurations, selected original/final checkpoints and `play.py`. No policy-quality evaluation, static-equilibrium test or repository-wide suite was run in this cleanup.

Exactly **one final native two-update smoke** then passed at **4096 environments × 64 ticks**. Its output is `logs/go2w_reference_tracking_v1_cleanup_smoke/configuration_cleanup_two_updates_2026-10-06_16-44-45/model_1.pt`. [Before](../evaluation/reference_tracking_v1_cleanup/smoke_before.json): no checkpoint, empty Adam, normalizer counts zero, iteration zero and std .40, with checkpoint loading patched to fail. [After](../evaluation/reference_tracking_v1_cleanup/smoke_after.json): Adam step 80, actor/critic normalizer counts 524288, finite model state and fixed LR .0003. The native metadata contains both navigation envelopes and the push distribution. No pushes occurred within this 2.56 s rollout horizon; deterministic unit tests, not chance in the smoke, verify the strong tier and complete-event reapplication/impulse. This smoke is execution evidence only and will not initialize production. [Executed command record](../evaluation/reference_tracking_v1_cleanup/executed_commands.ps1).

The exact tracked files changed in this cleanup are `robot_gym/envs/go2w/go2w_config.py`, `robot_gym/envs/go2w/go2w_env.py`, `tests/test_go2w_navigation_recipe.py`, `docs/go2w_reference_tracking_v1.yaml`, `docs/go2w_reference_tracking_v1.md` and `docs/go2w_reference_tracking_v1.ps1`. Read-only audits, the single prospective sample, resolved-config evidence and execution evidence are retained separately in [evaluation/reference_tracking_v1_cleanup](../evaluation/reference_tracking_v1_cleanup/). Existing evidence was preserved. Rewards, PPO, phase guidance, action scaling and physics compare exactly with the pre-cleanup YAML.

[Frozen targets](../evaluation/consolidated_v3_preparation/frozen_targets.json) preserve forward .5 for 30 s with ≤.05 m cross-track and ≤2° heading deviation, mean speed .45–.55 and RMSE ≤.04; 20–30 s holding planar RMS ≤.005 m/s with the existing rotational comparison; and unchanged full 8 s stopping path/reversal/residual-motion requirements. Signed stepping and small-command requirements remain. The existing historical holding targets are not claimed to have been measured by this five-case closure. Future comparisons must include **ORIGINAL1499 and final998**, and report actual reference-pose error, wheelbase, both track widths and pair midpoints. Curving, progressive crouching or self-contact disqualifies an otherwise good speed trace; sensor smoothness cannot compensate for poor tracking or stopping.

Production remains unstarted. The wider command envelope is training exposure, not demonstrated policy capability. Remaining uncertainty is learning under the combined objective and new exposure, generalization beyond the one nominal seed, substep contacts not captured by the evaluator, and whether the chosen soft tradeoffs acquire adequate stepping while preserving posture, including small .02–.05 commands. Coefficients are a documented starting set, not proven optima. No further unchanged continuation or automatic learning stage is prescribed.
