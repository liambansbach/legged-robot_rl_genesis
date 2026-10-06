# Interrupted fresh rolling-control diagnosis — 6 October 2026

Stage-two closure: the fixed-LR branch completed at model998 / 1001 lineage updates.
The [consolidated reference_tracking_v1 preparation](go2w_reference_tracking_v1.md)
supersedes the stage-one continuation recommendation below and records the final
matched measurements, one new fresh recipe and its unexecuted command.

The deterministic checkpoint1000 policy has real forward clenching and weak,
uneven yaw stepping. No placement formula, sign, indexing, frame, cache, or
configuration-mutation bug was found. Placement correctly penalizes the invalid
geometry, but its cost is modest relative to other changing objectives. The
available evidence does **not** establish that LR instability, acquisition under
the final precision task, or placement weighting is the primary cause. No reward,
normalizer, optimizer, plant, observation, or training-recipe change is proposed
as an established fix.

| Status | Finding | Evidence / limit |
|---|---|---|
| Confirmed | Correct full task; interrupted before its saved budget | Actual budget 2000, not the earlier prepared 3000; latest checkpoint is label 1000, logs end at 1058 |
| Confirmed | Forward front centers converge inward; rear centers spread | Late widths 0.1521 / 0.5736 m; reference 0.3802 / 0.3802 m |
| Confirmed | Placement is active, correctly signed, and sees that geometry | Late raw 1.20825, weighted rate −0.0604123, per-tick −0.00120825 |
| Confirmed | Self-contact accompanies clenching | FL calf / FR calf at 514 of 750 policy boundaries; maximum reported pair force 38.78 N |
| Confirmed | Yaw demand is fully active; actual lifts are uneven | Demand 1 at ±0.5; 4 cm targets; achieved late yaw +0.118 / −0.296 rad/s |
| Confirmed | Severe forward clenching appeared after checkpoint250 | Its single forward replay has widths 0.4230 / 0.4050 m, no clipping or self-contact |
| Hypothesis | Geometry is too inexpensive relative to useful competing improvements | Corridor improves substantially while height/pose/width deteriorate; reward values alone do not establish a learning gradient or optimal tradeoff |
| Hypothesis | Optimization/exploration contributes | New LR trajectory and persistent std differ, but deterioration is not aligned with the early LR peak |
| Unresolved | Final precision objectives obstruct acquisition | Plausible difference from warm-start refinement, not demonstrated by a causal comparison |

Evidence lives in [evaluation/fresh_1000_diagnosis](../evaluation/fresh_1000_diagnosis/).
`checkpoint1000/analysis.json` contains raw/rate/tick values, named geometry,
joint targets, support and individual cycles; adjacent NPZs retain every sampled
boundary. Scripts there reproduce the bounded replay and offline analyses.

## Actual run, task, and interruption

Run: `logs/go2w_transfer_v3_navigation_rolling_control_fresh/navigation_rolling_control_fresh_seed1_2026-10-06_09-04-36/`.
The checkpoint is `model_1000.pt`, internal `iter=1000`, Adam step **40040**
(40 minibatches per update), and actor/critic normalizer counts **262406144**
(4096 × 64 × 1001). These independently support **1001 completed updates in
the checkpoint**. Saved Adam LR is **0.000576650390625**.

Diagnostics contain 1048 rows over labels 0–1058; missing labels are
137, 266, 311, 317, 327, 566, 795, 833, 876, 901, 1015.
TensorBoard has 1044 value/LR/episode-length records, also ending at 1058.
The native diagnostic flush runs after PPO update, so the last record supports
at least **1059 completed updates**, with **58 later updates absent from the
latest saved model**. There is no final model1999 or model2999, and no retained
termination traceback establishing the exact interruption instruction. The user
reports interruption; the artifacts support an unfinished run. They do not
establish completion of either 2000 or 3000 updates.

Both actual `config.yaml` and `preparation.json` specify **2000** updates.
The older preparation command/document requested 3000. The source recipe is
`logs/go2w_transfer_v3_navigation_rolling_control/navigation_rolling_control_from499_seed1_2026-10-05_11-20-28/config.yaml`.
Its SHA-256 is `743d519d382bcf7f0dc55a2c081cfb37cbfc5f50b5be8b4e3fd11d31e9bca6bb`.
All task settings match that source, aside from provenance/removed ancestry.
The actual active identity is `transfer_v3 / phase_guided /
navigation_rolling_control`, command-demand phase observations, 58 actor and
critic observations, 16 actions. Fresh initialization is explicitly recorded;
no model ancestry/load selectors are inherited.

Native settings: Genesis 1.4.1, Torch 2.9.0+cu130, RSL-RL 5.5.1; 4096 environments,
64 rollout ticks, seed 1; separate normalized ELU MLPs 512/256/128; learned log
Gaussian std initially .40, bounded [.10,.70]; Adam; LR initially 3e-4,
adaptive KL target .01; 5 epochs × 8 minibatches; PPO clip .2; gamma .995,
GAE .95; entropy .003; clipped value loss coefficient 1; gradient clipping
threshold 1; sagittal data augmentation, no mirror loss. Policy dt=.02 s,
four .005 s physics steps. No packages were changed.

### Saved original fresh V3 versus saved new fresh task

Original: `logs/go2w_transfer_v3/transfer_v3_seed1_2026-10-03_16-18-34/config.yaml`,
with historical final model1499 (1500 updates). This comparison uses that file,
not today's base defaults. Full differences are in `config_comparison.json`.

| Material setting | Original saved fresh V3 | New saved fresh rolling-control |
|---|---|---|
| Episode timeout | 20 s | 60 s |
| Phase observation | Unconditional clock | Clock multiplied by command demand |
| Pure lateral magnitude | .10–.30 m/s | .02–.30 m/s |
| Extra sensor holds | Absent | Draw probabilities .04 for 8–15 s, .015 for 20–30 s |
| Rolling/stand yaw error scale | .35 rad/s | .07 rad/s; intentional turning retains .35 |
| Placement rate coefficient | Absent | −.05, with soft width/midpoint corridors |
| Rolling-pose hip multiplier | 1 | 3 |
| Angular xy cost | −.05 | −.075 |
| Base vertical velocity cost | −.2 | 0 |
| Rigid sensor vertical velocity cost | Absent | −1 |
| Exact-stand motion cost | 0 | −5 |
| Planned updates / save interval | 1500 / 100 | Actual 2000 / 250 |
| Native learning settings | Same settings listed above | Same initialization/settings, different realized trajectory |

Plant, gains, action scaling, normalizer architecture, phase timing/apex,
phase clearance/support, and the sagittal wheel-thigh corridor are unchanged
between these saved tasks. The earlier warm-start rolling-control recipe instead
used already trained models/normalizers/std, fresh Adam, and fixed LR 5e-5.
The new run asks a random policy to acquire locomotion under all final precision
terms from its first update. That is a materially different acquisition problem;
it is not evidence that a completed 3000-update experiment would fail.

## Bounded deterministic physical evidence

Five cases at checkpoint1000, each 15 s, one nominal environment, seed 1:
stand, [.5,0,0], [0,.2,0], [0,0,.5], [0,0,−.5]. The user's exact play command
was not available, so .5 forward is explicitly a diagnostic case. One additional
15 s forward-only case used checkpoint250. Total policy-controlled simulation:
**90 seconds**. No bank, sweep, stochastic rollout, robustness campaign, or
optimizer smoke ran. Ordinary evaluator reset/settling is separate from these
timed cases. Setup failures occurred before any diagnostic case; their console
logs remain in the evidence directory.

Replay uses current local saved-config resolution, `configure_nominal`, native
`get_inference_policy`, and existing `rollout_precision`. Nominal evaluation
disables training noise/DR/pushes and uses the standard nominal reset. During
policy-controlled cases no morphology, root/joint state, actions, observation
clock or controller target was corrected. Actor, normalizer and Adam tensors
were identical before/after. `play.py` SHA-256 remained
`16ada3760cc5017869480fb2f8cdfa515d5f1085b48a029fd23d6fb1e0fa6f3e`.

All rows below use the final 5 s; velocity and RMSE order is vx, vy, yaw.

| Command | Achieved mean | Tracking RMSE | Base height | Front/rear width |
|---|---|---|---:|---|
| Stand | −.02321, .00071, .00313 | .02326, .00101, .00604 | .35498 | .29289 / .47763 |
| Forward .5 | .48224, .00052, .01183 | .02194, .00225, .01318 | .36904 | .15211 / .57355 |
| Lateral .2 | .00890, .18787, .01147 | .03802, .02914, .10548 | .38976 | .31557 / .40809 |
| Yaw +.5 | −.02936, .01354, .11796 | .05886, .03527, .44838 | .39930 | .32246 / .37116 |
| Yaw −.5 | −.02170, −.03220, −.29574 | .05878, .04390, .28877 | .39820 | .32475 / .39981 |

No falls, timeouts, or force-thresholded non-wheel **ground** contacts occurred.
Forward has FL-calf/FR-calf self-contact from t=4.56 s, at 514 sampled boundaries,
maximum per-contact force norm 38.776 N. Yaw− has three FR-wheel/RR-wheel
manifold entries at one boundary, t=4.34 s, maximum individual force 45.360 N.
The collision term deliberately counts non-wheel ground contacts, so it remains
zero for these self-contacts. This is its existing scope, not a demonstrated
placement implementation bug. Contacts/force getters are sampled at 50 Hz;
these are not physics-substep maxima or integration impulses. No disputed
boundary geometry required additional substep capture.

### Clenched frame: 15.00 s of pure forward

Exact action/reward-boundary command [.5,0,0], rolling gate **true**, phase demand
**0**, placement coefficient **−.05**, registered coefficient **−.001**.
Named wheel order really is FL, FR, RL, RR:

| Wheel | Local/global link index | Authored link origin, world xyz (m) | Collision center, body xyz (m) |
|---|---|---|---|
| FL | 4 / 5 | 7.39037, .70434, .11540 | .15600, .06958, −.28745 |
| FR | 8 / 9 | 7.42335, .64663, .11088 | .17612, −.08119, −.29495 |
| RL | 12 / 13 | 6.98980, .89142, .07376 | −.20138, .33197, −.23373 |
| RR | 16 / 17 | 7.01460, .35844, .07718 | −.28300, −.28330, −.25416 |

Collision centers in world coordinates are respectively
[7.38432,.74642,.09289], [7.43417,.60254,.09500],
[6.97887,.93369,.09395], [7.02168,.31416,.09460].
Front/rear widths are **.150766 / .615279 m**, pair midpoints
**−.005808 / .024335 m**. References are [.3802,.3802] m and [0,0] m,
width deadband .04 m / scale .10 m, midpoint deadband .015 m / scale .05 m.
The base is **.364381 m** high versus target **.427742 m**; conventional body
roll/pitch are **−.04899 / −.11075 rad**. Ground loads FL/FR/RL/RR are
**63.76 / 36.53 / 42.92 / 49.30 N**; all supports are loaded.

The URDF cylinder-axis vectors in the body frame are approximately
[0,−.9043,.4270], [0,−.9258,−.3780], [0,−.8847,−.4662], [0,−.9480,.3183].
Their absolute camber angles are **25.28°, 22.21°, 27.79°, 18.56°**.
Cylinder-axis sign is arbitrary; joint angles below use actual joint signs.
The finite-cylinder body-y projected **inner-edge gaps** are only
**.029395 / .495719 m**. These are geometric projections, not measured pixel
gaps; tire tilt makes the front visually closer still. The center separation
itself is also far too narrow. Placement sees center widths/midpoints, not camber,
calf clearance or the exact tire-edge envelope.

| Leg | Actual hip / thigh / calf (rad) | Issued = applied P targets (rad) |
|---|---|---|
| FL | −.44115 / 1.15013 / −1.91660 | −.30000 / 1.05000 / −1.80000 |
| FR | .38760 / 1.04456 / −1.87347 | .30000 / 1.05000 / −1.80000 |
| RL | .48496 / .73503 / −1.37310 | .11264 / .89669 / −1.22147 |
| RR | −.32390 / 1.02898 / −1.45177 | .02944 / 1.05000 / −1.34074 |

Raw FL/FR hip actions **−2.3693 / +1.6997** clip to **−1/+1**, commanding
inward P targets. Both front thighs clip positive, both front calves negative,
and RR thigh positive. Those seven components remain saturated throughout the
final 5 s. Actual positions can exceed P targets under load; target clipping is
not a hard position clamp. At this frame FL/FR hip control forces are
+5.658 / −3.503 Nm, opposing their further inward deflection. RL/RR hip forces
are −14.928 / +14.139 Nm. No joint force saturates in the late forward window.
Limits are 23.7 Nm for hips/thighs/wheels and 35.55 Nm for calves; leg kp=40,
kd=1. Nominal delay is zero. Wheel targets are 4.838/6.544/7.239/6.107 rad/s;
actual velocities 6.239/6.110/6.136/5.728 rad/s, below the 20 rad/s target cap.
All joint values, raw targets, clip signs and errors remain in `analysis.json`.

### Formula and runtime checks

The independent reconstruction queries the explicitly named links, applies each
URDF collision offset in its link rotation, subtracts the authored base origin,
and applies one inverse base rotation. Genesis `relative=True` means authored
origin, **not** body-frame coordinates. Offline float64 SciPy rotations agree
with the actual reward input to at most **2.4622e-7 m** at checkpoint1000.
Independent URDF FK from measured joint angles also agrees (forward frame
2.3274e-7 m). Thus no wheel permutation, extra body transform, COM-origin mix-up,
or units mismatch was found.

`capture_precision=False` throughout all six cases: the placement-enabled
training cache updates without precision capture. Instrumentation only copies
its tensors and the original reward functions' returned values. Commands match
at action/reward boundaries. Source YAML and cached nested configuration remain
unchanged through preparation/registration; preparation is idempotent and deep
copy isolation was checked on the actual source recipe. Continuous scales get
dt once; discrete termination retains −5. Reward weighting mismatch is below
1.7e-9, and sum-of-contributions versus actual total below 7.5e-9 per tick.
The independent helper cost differs by at most 1.1e-6 raw from float32 runtime.
Negative rewards are not clipped away (`only_positive_rewards=False`).

New regression in `tests/test_go2w_rolling_placement.py` uses the real helper:

| Controlled widths (front/rear m), centered unless noted | Raw cost | Rate at −.05 |
|---|---:|---:|
| .3802 / .3802 | 0 | 0 |
| .30 / .3802 | .040401 | −.00202005 |
| .20 / .3802 | .451 | −.02255 |
| .10 / .3802 | .951 | −.04755 |
| .10 / .10 | 1.902 | −.09510 |
| Crossed −.10 / −.10 | 3.902 | −.19510 |
| Symmetric widening .6604 / .3802 | .951 | −.04755 |
| Nominal widths, front pair displaced +.065 m | .25 | −.01250 |

Exact pure vx activates all these costs; even partial lateral/yaw commands
exclude them as specified. The test also checks dt weighting and monotonic
narrowing penalties. This is a formula/geometry result, not a policy-gradient
claim. Seven focused placement/fresh-recipe tests passed; actual-source CPU
validation passed; `git diff --check` passed. No behavioral code was changed.

## Actual reward contributions and task conflicts

At the clenched 15 s frame:

| Term | Raw value | Weighted rate | dt-weighted tick |
|---|---:|---:|---:|
| Placement | 1.431277 | −.071564 | −.00143128 |
| Reference height | .467748 | −.467748 | −.00935497 |
| Rolling pose | .248341 | −.124170 | −.00248341 |
| Orientation | .014584 | −.058335 | −.00116671 |
| Wheel-thigh corridor | .167980 | −.083990 | −.00167980 |
| Tracking x | .999533 | +.999533 | +.01999067 |
| Tracking y | .999976 | +.999976 | +.01999951 |
| Tracking yaw | .992639 | +.794111 | +.01588222 |
| Phase clearance | .00001532 | −.00001532 | −.000000306 |
| Phase support | 0 | 0 | 0 |
| Sensor vertical velocity | .00001144 | −.00001144 | −.000000229 |
| Stand still | 0 | 0 | 0 |
| Normalized effort | .085249 | −.002557 | −.00005115 |
| Leg action rate | .00000438 | −.0000000438 | −.000000000877 |
| Collision / insufficient support / torque limits | 0 | 0 | 0 |

Selected **late-window weighted rates** compare the actual failure states:

| Term | Forward | Yaw +.5 | Yaw −.5 |
|---|---:|---:|---:|
| Placement | −.06041 | 0 | 0 |
| Height | −.38310 | −.04082 | −.05037 |
| Rolling pose | −.10972 | 0 | 0 |
| Yaw tracking | +.78582 | +.27121 | +.56118 |
| Phase clearance | −.000016 | −.05063 | −.05139 |
| Phase support | 0 | −.05990 | −.06116 |
| Wheel-thigh corridor | −.03864 | −.19853 | −.11697 |
| Sensor vertical | −.000008 | −.01045 | −.01182 |
| Angular xy | −.000002 | −.00646 | −.01375 |
| Stand still | 0 | 0 | 0 |
| Leg action / effort | −.00000018 / −.00218 | −.000348 / −.00174 | −.00143 / −.00175 |

Placement is **correctly active and relatively modest**, and does not explicitly
measure tire tilt/calf contact. It is not inactive or blind to the center defect.
Height and pose already penalize this posture strongly. At checkpoint250 versus
1000, forward corridor cost improves −.36347 → −.03864, yaw reward
.76021 → .78582 and pair midpoints move nearer zero; height worsens
−.15066 → −.38310, pose −.02774 → −.10972, placement −.00296 → −.06041.
The total late rate is **2.15415 → 2.12631**, slightly worse, not evidence that
clenching is the optimum. These are separate on-policy trajectories, not a
controlled pose perturbation or advantage comparison. Large positive tracking
offsets do not by themselves imply their gradients overpower placement.

For scale only, multiplying placement by four would make the recorded late
forward cost −.24165 and this frame −.28626; a sixfold factor gives −.36247 and
−.42938. These estimates were not selected or applied. They do not establish
retained locomotion, affect no yaw states under the current gate, and cannot
alone explain or repair the shuffling. Neither a weight correction nor an
acquisition curriculum is justified as the primary intervention from this
single interrupted run.

### Stepping and physical sensor motion

Yaw±.5 and lateral .2 have demand **1** throughout: period .8 s, stance fraction
.65, diagonal offsets FL/RR=0, FR/RL=.5, apex **.04 m**. There are 17/18/18/17
complete demanded cycles by wheel in 15 s. Targets exist even while wheels stay
loaded. No apex change is justified by absent lifts.

| Case | Final-5-s actual clearance maxima FL/FR/RL/RR (mm) | Completed ≥2 mm events, whole case | Completed ≥2 cm events, whole case |
|---|---|---|---|
| Lateral .2 | 10.5 / 21.8 / 33.9 / 11.5 | 19 / 20 / 18 / 22 | 5 / 3 / 5 / 0 |
| Yaw +.5 | .47 / 34.8 / .03 / −.15 | 6 / 19 / 10 / 2 | 0 / 11 / 2 / 0 |
| Yaw −.5 | 52.6 / 2.21 / −.09 / 3.32 | 19 / 13 / 8 / 8 | 7 / 5 / 0 / 0 |

These use actual finite-cylinder ground gap, not link height or an unloaded flag.
Small negative gaps are solver penetration. Existing event semantics are 6/10 N
unload/reload hysteresis and ≥2 mm geometric peak, with no dwell filter; multiple
contact events per demanded cycle are possible. Counts are not successful gait
cycles. Per-cycle desired/actual peaks and support are in the JSON.
Mean loads during demanded swing are yaw+ **25.3/7.1/21.9/40.5 N**, yaw−
**2.9/19.3/32.8/26.9 N**. Corresponding unloaded fractions are
**.460/.889/.429/.267** and **.968/.636/.283/.299**. Rear support often fails to
unload on demand, while the leading lifting wheel changes with turn sign.

| Case | Body roll/pitch angular-rate RMS (rad/s) | Imager/radar world-z velocity RMS (m/s) |
|---|---|---|
| Stand | .00243 / .00484 | .00167 / .00156 |
| Forward | .00491 / .00297 | .00306 / .00267 |
| Lateral .2 | .30907 / .28154 | .08422 / .07714 |
| Yaw +.5 | .21281 / .20225 | .10294 / .09314 |
| Yaw −.5 | .33511 / .26640 | .10988 / .09723 |

The strong physical camera motion is reproduced during stepping, whereas late
forward is smooth but malformed. Sensors are rigidly attached: their angular
motion follows body motion. These are physical-frame measurements, not the
renderer's follow-camera movement or optical image stabilization measurements.

Historical results were reused, not rerun. Original fresh model1499 nominal
`evaluation/transfer_v3_main_ck1499_nominal/metrics.json` achieved late yaw
**+.39985/−.38498** for ±.4, with command-phase yaw RMSE **.03572/.03212**,
and six ≥2 cm events per wheel in each turn's recorded episode. Its forward .5
case had late widths **.42952/.38617 m**. Those use 3 s stand / 5 s command /
6 s stop; they establish retained historical capability, not a matched-horizon
or identical-command performance ratio against the new ±.5 cases.

## Native optimization evidence

Only available records contribute to summaries; missing rows are not zeros.
`new_training.json`, `old_training.json`, `*_scalars.json`, `minibatch_lr.json`
and `training_details.json` retain the details.

New after-update LR min/max: **1e-5 / .0098526125**, maximum at label **20**.
LR exceeds .004 at labels 10–57, 59–71 and 75. Replaying the installed native
minibatch scheduler over contiguous recorded KL sequences reproduces all
available endpoints exactly and first reaches the observed peak at 17/minibatch1.
Original after-update min/max: **1e-5 / .0043789389**, maximum at label **21**;
reconstruction reveals an original **intra-update** peak **.0065684084** at
25/minibatch1. These are observed/reconstructible maxima; missing minibatches
cannot exclude a higher unrecorded value. Native code hard-codes floor 1e-5 and
cap **1e-2**, divides/multiplies LR by 1.5 when KL >.02 or 0<KL<.005. There is
no configurable native adaptive-cap argument in this installed PPO. Initial
3e-4 does not mean fixed 3e-4.

| Update window | New/original mean LR | New/original native KL | New/original PPO clipped fraction | New/original value loss |
|---|---|---|---|---|
| 10–49 | .006459 / .003200 | .01405 / .01177 | .2316 / .1928 | .7391 / .06563 |
| 100–249 | .001079 / .001165 | .01524 / .01441 | .2859 / .2559 | .6286 / .01424 |
| 250–499 | .000628 / .000658 | .01524 / .01392 | .2907 / .2474 | .5202 / .00105 |
| 500–749 | .000476 / .000418 | .01530 / .01338 | .2843 / .2419 | .4932 / .00084 |
| 750–999 | .000420 / .000385 | .0151 / .01329 | .2764 / .2390 | .4541 / .00074 |
| 1000–1058 | .000445 / .000371 | .0148 / .01319 | .2669 / .2373 | .3551 / .00074 |

New measured KL equals native scheduler KL in these adaptive records. The
original has native KL but no separate measured-KL field; no values were invented.
New update-mean KL range is .007430–.096320; original native .007030–.096628.
Both maxima are update 0. PPO clipped-fraction ranges are .16686–.71980 and
.15580–.71164, also with startup maxima. Native KL is distribution KL, not the
same quantity as clipped action fraction or measured tracking error. The PPO
ratio diagnostic includes the existing symmetry-augmented pass; measured KL
uses original samples. **Gradient norms were not logged**: threshold 1 is a
setting, not an observed norm. Different reward scales/episode lengths make
value-loss magnitudes imperfect cross-task comparators.

At checkpoint1000, effective std by leg (hip/thigh/calf/wheel):

| Leg | New fresh | Original fresh at the same label |
|---|---|---|
| FL | .35348 / .52779 / .28362 / .26254 | .10000 / .13779 / .10000 / .17391 |
| FR | .35340 / .52783 / .28374 / .26234 | .10000 / .13787 / .10000 / .17394 |
| RL | .34142 / .24408 / .33113 / .24174 | .10006 / .10000 / .10060 / .15076 |
| RR | .34155 / .24403 / .33123 / .24184 | .10004 / .10000 / .10053 / .15076 |

New stochastic leg-action clipping is **17.53%**, deterministic means saturated
on those same stochastic rollout states **12.98%**; original **3.32% / 3.00%**.
Wheel values are new **.240% / .00181%**, original **4.46% / 4.27%**.
Those are distinct from the deterministic nominal forward trajectory, whose
front actions themselves saturate. Higher exploration helps explain noisy
training traces but cannot dismiss a directly reproduced deterministic defect.

Episode length reaches 3000 ticks (60 s) near update47; initial episode ages are
randomized while reward/logger sums start at zero, so the startup ramp includes
accounting effects. The largest post-startup logged dip is **2261.75 ticks
(45.235 s)** at labels233–234, with LR .00086498, KL .01551 and clipping .29483
at233: well after the early LR peak. Late windows average roughly 59 s; final
logged mean is **2807.03 ticks (56.141 s)**. Checkpoint250's forward geometry
remains un-clenched after that dip. Deterioration in sampled height/placement and
mean saturation continues while LR declines. Across 938 matched records at
100–1058, LR correlation with placement/height raw cost is **−.476/−.544**;
leg-mean saturation correlation is **+.679/+.858**. These are nonstationary,
contemporaneous associations, not causal attribution. Current logs lack
historical per-family wheel widths, so exact onset cannot be dated within
250–1000.

There are 905 episode-termination scalar records, 213 nonzero; this is **not**
213 falls. Episode reward terms average ended environments' sums divided by the
configured maximum episode seconds (60 new, 20 original). Exact per-update
fall/timeout counts and a complete terminal-event ledger were not logged. For
example −.083333 is consistent with −5/60 for ended fallen episodes, not an
environment-time failure rate. Final zero termination scalar and a shortened
rolling mean episode length are not contradictory.

Empirical normalizers start adapting immediately, with count262144 by checkpoint0
and262406144 by checkpoint1000; both models' moments are retained. Startup KL
includes changing normalization as well as policy updates. Installed native
normalization uses cumulative sample moments; no specific defect was found,
and nothing was frozen or changed. During deterministic replay, state parity
confirms no normalizer adaptation or optimizer update occurred.

### Why aggregate tracking_yaw is misleading here

The raw reward is `1 - Huber(yaw_error / scale)`, weighted by .8 and dt=.02.
Scale is **.07** for exact rolling/stand, **.35** for intentional turning or
lateral motion. For a .35 rad/s error these give raw **−3.5** versus **+.5**,
weighted rates **−2.8** versus **+.4**. Mixing them obscures failure families.
The deterministic traces separate them: late stand/forward weighted rates
**.79702/.78582** at .07; yaw+/yaw− **.27121/.56118** and lateral **.76367** at
.35. Historical logs have family total reward/exposure/posture, but reward-term
sums are aggregate. They do **not** contain historical per-family tracking_yaw
statistics, and family totals cannot reconstruct that split. Aggregate new yaw
rate improves from −.486 (10–49) to +.1008 (1000–1058), without proving good
intentional turning. Low-demand partial commands are likewise not all equivalent
to fully activated stepping.

## Single next action / missing discriminator

Keep production paused and retain both checkpoints. No A/B/C/D training package
or changed recipe was implemented: the measured geometry weakness is real, but
the evidence does not isolate the cause of late acquisition/posture deterioration.
In particular, stronger placement cannot address yaw under its existing gate;
the turning states have zero rolling-pose and stand-still cost; their sensor
cost is smaller than clearance/support/corridor losses. This does not demonstrate
that the newly added precision terms obstruct acquisition.

The single missing discriminator is an **LR-only comparison from saved
checkpoint250**, before the measured clenching: retain actor, critic, learned std,
both empirical normalizers **and all Adam moments/step counters**; retain the
complete saved task; explicitly override native schedule to **fixed 3e-4** and
set both native `alg.learning_rate` and every Adam parameter-group LR **after
checkpoint loading**. Do not reset or freeze normalizers. Budget **750 additional
updates**, reaching 1001 total lineage updates, with a separate output and a
review after 250 additional updates. This would be a changed optimization
comparison, **not an exact continuation**. Native resume reuses label250, so
750 additions end at label999; lineage, not filename alone, establishes total
updates. This tests whether the established policy avoids late deterioration
under controlled LR; it does not test fresh-from-random acquisition or guarantee
fixed LR will help.

That comparison is specified as the missing evidence, not selected as a proven
remedy or prepared for execution. No runnable training command is supplied:
the existing CLI has no validated LR-after-load override, and inventing one or
silently retaining the checkpoint LR would mislabel the experiment. No PPO fork,
monkey-patch, optimizer smoke, production run, push, artifact deletion, historical
policy edit, package upgrade, or IsaacLab change occurred.

## LR-only branch preparation — 6 October 2026

The subsequent request authorizes preparation of the discriminator above.
**Production has not started.** The first stage is limited to **250 additional
updates**, with the remaining **500 conditional on review and explicit further
authorization**. No random initialization or reward alternative is prepared.

Source is this report's original fresh run,
`navigation_rolling_control_fresh_seed1_2026-10-06_09-04-36/model_250.pt`.
Its internal label is **250**, every Adam counter is **10040**, and both
normalizer counts are **65,798,144**, independently confirming **251 completed
updates**. SHA256 is
`a9b3675c4fb376ae83998da23e300d1870aa08ac72bbfa20f9d704d3f585e3e8`.
Checkpoint1000 remains a failed comparison, never the initialization.

The small opt-in `--go2w_resume_fixed_lr 0.0003` extends the existing full-state
V3 loading path. Native loading restores actor, critic, both empirical
normalizers, learned action std, all Adam moments/counters and iteration first.
The task's post-load hook then sets native `schedule="fixed"`,
`alg.learning_rate=3e-4` and **every Adam group LR=3e-4**, before diagnostics and
config saving. It does not modify RSL-RL. Ordinary resume and historical
selective initialization retain their existing loading behavior.

| Check | Verified result |
|---|---|
| Actual LR change | Saved Adam LR **0.0008649755859375 → 0.0003**; adaptive → fixed. Source YAML already says initial LR .0003, so the YAML algorithm diff is schedule only |
| Before first update | Exact actor/critic state equality, including normalizer buffers/counts and learned std; exact Adam equality except group LR; internal iteration retained |
| Task preservation | Complete saved environment equal apart from continuation provenance; actor/critic architecture, all other PPO settings, seed, batch, observations, sampler, rewards, control, physics, exploration bounds and normalizer behavior retained |
| Source preservation | Source checkpoint, saved YAML and local `play.py` hashes unchanged; source recipe is copied, with no fine-tuning delta or grandparent reload |
| Focused tests | **14 passed**: resume/config validation, all-group LR override, ordinary fixed-LR resume, existing fresh/selective initialization and placement checks |
| Single native smoke | **4096 × 64**, exactly **2** updates; **128** policy ticks and **80** Adam steps checked for fixed LR; observations/actions/rewards and final learning state finite |
| Saved smoke states | Labels **250/251** mean **252/253** lineage updates; Adam steps **10080/10120**; normalizer counts **66,060,288 / 66,322,432** in both models; moments advance, LR stays .0003 |
| Ordinary reload | Native CPU full-state reload of smoke251, **without** the opt-in flag, exactly matches its saved learning state and retains fixed .0003; no further updates |

The smoke's native learning loop and checkpoint saves completed. Its final
read-only probe then raised a CUDA-versus-CPU `torch.equal` error while checking
changed Adam moments. Both saved states and the ordinary reload were verified
on CPU afterward, with **zero additional simulation or optimizer updates**.
The original console traceback and run metadata (`interrupted_or_failed`,
label251) are preserved; this is a probe failure after two completed updates,
not a third update or a reported clean process exit.

Preparation evidence is in
[lr_only_preparation](../evaluation/fresh_1000_diagnosis/lr_only_preparation/):
[resolved config](../evaluation/fresh_1000_diagnosis/lr_only_preparation/config.yaml),
[source and exact differences](../evaluation/fresh_1000_diagnosis/lr_only_preparation/preparation.json),
[pre-update load proof](../evaluation/fresh_1000_diagnosis/lr_only_preparation/pre_update_load_state.json),
[smoke verification](../evaluation/fresh_1000_diagnosis/lr_only_preparation/smoke_validation.json),
[ordinary reload proof](../evaluation/fresh_1000_diagnosis/lr_only_preparation/resume_validation.json)
and [focused test output](../evaluation/fresh_1000_diagnosis/lr_only_preparation/focused_tests.txt).
The separate smoke run is
`logs/go2w_transfer_v3_navigation_rolling_control_fixed_lr_smoke/lr_only_from250_two_update_smoke_2026-10-06_11-38-30/`.
Its checkpoints are validation artifacts, not production initialization.

### Prepared command and lineage

[production.ps1](../evaluation/fresh_1000_diagnosis/lr_only_preparation/production.ps1)
contains the **one exact, unexecuted PowerShell production command**, including
the existing Python interpreter and local cache settings. It selects source250,
seed1, 4096 environments, the saved 64-step rollout and remaining PPO settings,
and 250 additional updates. Its distinct experiment is
`go2w_transfer_v3_navigation_rolling_control_fixed_lr`, with run name
`lr_only_from250_stage1_seed1` and a new timestamped directory.

Native RSL-RL 5.5.1 repeats the loaded label. The first stage runs labels
**250–499**, then saves **model_499.pt at 501 lineage updates**. The final native
save occurs even though label499 is not a periodic save boundary. Saved config,
full checkpoint and `training_resume` provenance support ordinary full-state
resume from that immediate branch; fixed LR must remain saved and loaded.
The next generation preserves previous-generation provenance and uses
`source_total_updates + (saved_label - source_label + 1)` for completed lineage.

If subsequently authorized, a separate 500-update stage from label499 would
end at **label998 / 1001 lineage updates**. The earlier label999 estimate applies
to one uninterrupted 750-update call, not this staged plan. No second-stage
command or automatic continuation is prepared. Simulator/RNG state is not
restored bit-for-bit: this is a **changed-optimization branch**, not a perfectly
paired causal experiment or an exact continuation.

### First-stage review, not final qualification

The [review plan](../evaluation/fresh_1000_diagnosis/lr_only_preparation/review_plan.json)
is frozen before production: evaluate only the stage's final checkpoint, using
the same **15-second nominal deterministic forward vx=.5, yaw+.5 and yaw−.5**
cases and existing measurement conventions. Preserve full, after-first-second
and late-five-second windows, action/reward boundary capture, terminal/reset
events and named contact records. Reuse source250 forward and failed1000
results; source250 has no historical yaw traces to invent. The original
**adaptive model_500.pt exists**, with internal label500, Adam step20040 and
**501 updates**. Allow only one extra forward case on it during the future
review for equal training age; nothing was evaluated now.

Prioritize named collision-center widths and pair midpoints, height, actual
joint angles/targets, persistent clipping, self-contact counts and named pairs,
achieved velocities/yaw errors/cross-axis motion, and each wheel's desired versus
actual clearance, unloading/support and complete/censored cycles. Retain wheel
tilt and sensor/body motion context. Yaw .5 already has full phase demand.
The collision reward remains a **non-wheel ground-contact** count; recurring
calf/self-contact is a **failed candidate property** even if that reward is zero.
Do not disable self-collision, correct policy states/actions or filter away
contacts/resets.

Less clenching alone cannot justify continuation. Require useful functional
progress in forward tracking and both yaw signs/stepping, alongside acceptable
contact/target behavior; fixed LR may simply slow acquisition. Neither aggregate
return nor a smaller loss suffices. All existing final functional targets are
copied unchanged into the review plan: forward speed/path/heading, late stand,
stopping, both lateral/yaw signs including partial commands, and placement with
support/tracking. These three short cases are an early diagnostic review, not
that final qualification. A positive review still requires explicit authorization
before spending any of the conditional remaining 500 updates.

## Completed fixed-LR stage-one review — 6 October 2026

**Recommendation: continue unchanged for the planned remaining 500 updates.**
The policy makes useful functional progress without renewed severe clenching,
recurring self-contact or progressive height collapse in these short cases.
Continuation was **not launched**. Height, path accuracy and stepping clearance
remain unfinished; this is an acquisition decision, not final qualification.

The actual run is
`logs/go2w_transfer_v3_navigation_rolling_control_fixed_lr/lr_only_from250_stage1_seed1_2026-10-06_11-54-29/`.
Its completed `model_499.pt` is **501 lineage updates**, after 250 additions to
original fresh source250. Full-load flags, first/final Adam counters
**10080/20040** and both final normalizer counts **131,334,144** confirm learning
state continuity. Production has no separate pre-update tensor dump; exact
pre-update parity remains the preparation proof. Functional task and other PPO
settings match the source. Config, native Adam and all 250 recorded LRs are
**fixed .0003**. No historical499 is used.

### Later learning windows

Both windows have all 50 records, after the **46.875-update** episode-accounting
ramp (60 s / .02 s / 64). Early resumed return growth is not sufficient evidence.
Empty native scheduler-KL arrays mean no adaptive decisions, **not zero KL**.

| Metric; weighted costs are reward rates | Labels300–349 | Labels450–499 |
|---|---:|---:|
| Measured KL mean / maximum minibatch | .00690 / .01056 | .00767 / .01334 |
| PPO ratio clipping fraction | 18.73% | 19.18% |
| Value loss | .41776 | .30275 |
| Height / rolling pose / placement | −.05478 / −.00631 / −.00055 | −.07226 / −.00805 / −.00089 |
| Insufficient support / phase support | −.02980 / −.03584 | −.01813 / −.02648 |
| Phase clearance / sensor vertical velocity | −.03374 / −.03039 | −.03144 / −.02300 |
| Stand-still / body angular motion | −.26745 / −.09100 | −.19933 / −.06997 |

Per-joint vectors are in `training_review.json`. Std falls (rear thigh
.263→.219), while rear thigh/calf mean-action saturation on stochastic states
rises from 1.43%/1.01% to 2.44%/2.35%. These are not deterministic outcomes.

### Four agreed cases and retained comparisons

Exactly **four 15 s cases** ran: fixed499 forward/yaw+/yaw− and adaptive500
forward, also **501 updates**. Nominal initial pose, phase, seed1 and 50 Hz
conventions match retained traces. All three requested windows, joint targets
and contacts are saved. Below: **late-five-second** values; heading/cross-track
cover the entire 15 s case.

| Forward vx=.5 | Fixed499 | Equal-age adaptive500 | Retained source250 | Retained failed1000 |
|---|---:|---:|---:|---:|
| Achieved vx / vx RMSE, m/s | .5124 / .01242 | .4444 / .05575 | .4864 / .01368 | .4822 / .02194 |
| vy bias / yaw bias, m/s and rad/s | −.00197 / −.00469 | +.00215 / −.00960 | −.00238 / −.02203 | +.00052 / +.01183 |
| Front / rear width, m | .4209 / .3990 | .4269 / .4760 | .4230 / .4050 | .1521 / .5736 |
| Front / rear pair midpoint y, m | +.0233 / −.0050 | −.0727 / +.0848 | +.0357 / −.0277 | −.0035 / +.0108 |
| Height, m | .3823 | .3797 | .3853 | .3690 |
| Signed pitch, degrees | +6.37 | −3.88 | +6.26 | −6.84 |
| Full heading / max cross-track | −6.24° / .556 m | +24.68° / 2.364 m | −20.94° / 1.355 m | +11.56° / .655 m |
| Self-contact boundaries / 750 | 0 | 0 | 0 | 514, FL calf / FR calf |

Fixed499's front/rear widths widen by **18.7/7.1 mm**
from 1–5 s to 10–15 s, rather than clenching. Height changes **.38319→.38225 m**
(−0.94 mm; after-1s slope −0.095 mm/s), versus adaptive500's **−10.96 mm**.
The remaining **45.5 mm** deficit to the unchanged .42774 m height reference
is real. Pitch settles from +5.01° early to +6.37° late, with **.055° late
standard deviation**; roll is −.777° ± .072° standard deviation. A numerical
rotation check verifies positive pitch as nose-down (body +X toward world −Z).
The late bias is largely static.

Forward sensor world-z velocity RMS (front camera/radar) is **.0300/.0286 m/s**
over the full case, **.00154/.00121** after 1 s, and **.000260/.000251** late.
Late body angular RMS x/y/z is **.00136/.000795/.00469 rad/s**. The 15 s path
still drifts substantially; it does not qualify the unchanged 30 s target.

Fixed499 has no sustained target clipping: isolated **20 ms** events in forward
and yaw+, none in yaw−. Adaptive500's **RL calf** clips in the positive direction
for **12.04 s**, including every late sample. Fixed499's late FL/FR calves are **−1.703/−1.806 rad**
versus issued/applied **−1.462/−1.582 rad**, without clipping. No case reaches
99% of a force limit at sampled boundaries;
maximum control-force ratios are **.540/.450/.435** for fixed forward/yaw+/yaw−,
and **.583** for adaptive forward. Target clipping is not force saturation.

Yaw+/yaw− achieve **+.5092/−.5197 rad/s** late, with yaw RMSE **.0792/.0853**,
vx bias **+.00449/+.00408 m/s**, vy bias **−.0195/+.0205 m/s**, and final XY
displacement **.0475/.0730 m**. Retained failed1000 achieved late **+.118/−.296**;
there is no
equal-age yaw comparison or source250 yaw trace. The user's lateral observation
remains qualitative.

| Yaw quantity; wheel order FL, FR, RL, RR | +.5 command | −.5 command |
|---|---|---|
| Late actual peak clearance, mm (desired 40 each; demand=1) | 8.68, 4.34, 5.63, 14.35 | 5.12, 10.88, 18.49, 7.88 |
| Late desired-swing unloading below6 N | 62%, 74%, 85%, 81% | 81%, 45%, 73%, 78% |
| Late mean normal load, N | 50.8, 48.9, 46.4, 49.7 | 46.3, 50.6, 49.9, 46.3 |
| Completed unloading events / events reaching2 mm | 30/14, 38/18, 31/25, 22/16 | 39/19, 28/11, 26/16, 30/20 |
| Left / right-censored events | 0/1, 0/0, 0/0, 0/1 | 0/1, 0/0, 0/0, 0/1 |

Unloading can fragment within the 17–18 complete demanded phase cycles; these
are not all clean steps. No completed yaw+ event reaches2 cm; only one yaw−
RL event does. FR remains loaded through much of demanded yaw− swing. Late
camera/radar world-z RMS is **.0647/.0595** for yaw+ and **.0616/.0588 m/s** for
yaw−; body roll/pitch angular RMS is **.208/.101** and **.207/.118 rad/s**.
Clearance, support timing and sensor motion need further acquisition.

All four new cases have **zero falls, resets, non-wheel ground contacts and
self-contact boundaries**, with no named self-contact pairs. This conclusion
uses explicit self-contact capture, not the zero ground-collision reward.
Counts/force ratios cover sampled policy boundaries, not all substeps.
No states/actions/contacts were corrected or filtered.

### Decision and unexecuted continuation

Equal-age adaptive500 is not yet severely clenched either; its poorer tracking,
larger offsets and sustained calf clipping are the useful comparison. This one
resumed branch does not establish LR as the sole cause. Final goals remain
unchanged while the planned **500 remaining updates** allow further acquisition.

[continuation_unexecuted.ps1](../evaluation/fresh_1000_diagnosis/lr_stage1_review/continuation_unexecuted.ps1)
contains the **one exact unexecuted command**: ordinary full-state resume of
stage499 into a separate experiment. Configuration resolution verifies saved
fixed3e-4 without an LR override, unchanged task and all load flags. No fine-tune
or fresh-recipe flag is present. Installed native counting gives **labels499–998 /
1001 lineage updates**. No training or optimizer smoke ran during this review.

Evidence: [compact comparison](../evaluation/fresh_1000_diagnosis/lr_stage1_review/comparison.json),
[all physical windows/targets/cycles](../evaluation/fresh_1000_diagnosis/lr_stage1_review/physical_details.json),
[training windows and source verification](../evaluation/fresh_1000_diagnosis/lr_stage1_review/training_review.json),
[exact evaluation commands](../evaluation/fresh_1000_diagnosis/lr_stage1_review/evaluation_commands.ps1),
[continuation resolution](../evaluation/fresh_1000_diagnosis/lr_stage1_review/continuation_resolution.json)
and [calculation checks](../evaluation/fresh_1000_diagnosis/lr_stage1_review/calculation_validation.json).
Only new numerical calculations were checked. The initial relative-path lookup
failed before simulation; the retained console records it. Exactly four cases
subsequently completed, with no training. Source policies/configs and `play.py`
hashes are unchanged.
