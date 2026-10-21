# Optional base-velocity estimation

This path replaces the actor's three simulator-velocity inputs with a learned
estimate. It is optional and uses the existing task registry and train/play/evaluate
entry points. Missing or null `velocity_estimator` in an old saved configuration
means disabled. Rewards, critic inputs and supervised labels keep simulator truth.
`env.base_lin_vel` is never overwritten.

## Current decision and preserved artifacts

The selected frozen pair is now the self-contained
[current package](../ressources/pretrained/go2w/README.md), including the full
checkpoint/config, privileged parent reference, measured assets and portable
history/ONNX checks. This is a publication milestone only: the failed stopping
comparisons below remain. No new training was performed for publication.

The October 21 integration merges `85045a4` with `testing@3fe238c` using a normal
merge. It retains the existing `--manual_control` window and source-yaw comparison.
There is no second keyboard controller. No new training or weight replacement was
performed during integration. The original 200-collection warmup (199 retained),
100 PPO updates and 4,784 retained estimator optimizer steps already happened.
Their reported combined wall time, including the interrupted first collection,
was 17m45s. The saved pair records 931.64 s, excluding that interrupted attempt.

**Retain the frozen pair as the conservative development baseline.** The fine-tuned
pair improves nominal estimation accuracy, but its longer quiet-standing drift
and stressed actuation slew are worse. Both pairs fail parts of the expanded stop
screen, so neither is qualified for hardware. Keep both unchanged for independent
sim2sim. Do not start a longer PPO run from these reward curves. A sensor-modeling
trial should follow the hardware signal audit, with a fixed comparison panel;
there is not enough evidence to blame one sensor or the estimator alone.

All paths below are relative to this repository; generated evidence stays ignored.

| Artifact | Local path | SHA256 |
| --- | --- | --- |
| Parent full checkpoint | `ressources/pretrained/go2w/reference/model_1999.pt` | `2f8f3a96ad7fdf83c2966b44d10fc76ac274d9d81130a470af881f81a5980bf6` |
| Frozen actor + estimator | `ressources/pretrained/go2w/checkpoint/model_2197.pt` | `c35c5b3861a0f7d0de1975a96e80dc6d45831ecad225b5a82d8031e63e95d9bf` |
| Published frozen ONNX | `ressources/pretrained/go2w/policy.onnx` | `5117e7a57c39426a1da6d6ceedb2c242d6642e6598ca80663b669e1619fc9c60` |
| Fine-tuned pair | `logs/go2w_velocity/finetune_2026-10-21_13-00-42/model_2296.pt` | `8c06103a914084bd99870e63f5496a2d9ab768e11d57fc6f56b5920f5403c639` |
| Local fine-tuned ONNX (not the default) | `logs/go2w_velocity/finetune_2026-10-21_13-00-42/exported/model_2296/policy.onnx` | `077575ccbafe456cdf6b89f688ca5d5fa6bf488cd7d2c99b7e8b81ad3e58d66c` |

The parent actor/critic/std/normalizers/native Adam state equals the frozen
checkpoint bitwise. The fine-tuned actor and estimator differ as expected. Native
labels 1999/2197/2296 are not estimator stage counters. The packaged reference's
inference-only actor hash remains `639a443f3da3d9fa15d9e532b85959df55c7bd5618564797740adb66b1d2bfcb`;
it is distinct from the retired 56-input pretrained package. The original local
warmup and fine-tuning runs remain intact. Old publication/handoff assets can be
retrieved with matching source at commit `7d1f87c87fe1b190b4278fe870c69d6c652a3ce5`.

## Architecture and contract

Shared code is in `utils/velocity_estimator.py`, `estimator_ppo.py` and
`deployment_history.py`. The Go2-W task declares feature selection, units, joint
conventions and reflection. `inference_policy(...)` composes estimation and actor
inference for play and evaluation, including bank/equilibrium. Reports record the
selected velocity source; estimated mode never falls back to simulator velocity.

The MLP has five frames, hidden layers `[128,128,128]`, ReLU and a linear three-value
output. These remain starting hyperparameters. Its target is body-frame `(vx,vy,vz)`
in m/s at the **authored base-link origin**, not the COM or IMU origin. The internal
actor remains 58-input/16-action; the privileged critic has 65 inputs. Policy time
is .02 s and physics time .005 s. Five samples span 80 ms, oldest to newest.

| Frame slots | Signal before saved scaling | Units |
| --- | --- | --- |
| 0:3 | Body gyro | rad/s |
| 3:6 | Projected gravity | unit vector |
| 6:9 | vx, vy, yaw command | m/s, m/s, rad/s |
| 9:21 | Twelve leg offsets from trained pose | rad |
| 21:37 | Sixteen joint speeds, including wheels | rad/s |
| 37:53 | Previous issued clipped actions | dimensionless |
| 53:55 | Demand-conditioned sine/cosine phase | dimensionless |

There are no absolute wheel angles, acceleration, simulator velocity/position,
contacts, critic geometry, physical labels or future samples in deployment inputs.
Zero commands do not force estimates to zero. Fixed feature mean/std (initialized
from the parent actor statistics, including epsilon and a .01 denominator floor)
standardize scaled features once. Estimated m/s is scaled once before the actor's
own embedded normalizer. Normalization and action bounds are not duplicated.

`SensorHistory` acquires one coherent simulated measurement at a policy boundary.
Delay and holding affect sensors together; command, issued action and phase are
current controller state. Getters/manual changes refresh only those newest slots,
without advancing phase/history or drawing noise. A same-tick selective reset
acquires and repeats the first real measurement only for invalidated rows; other
rows do not age or resample. Bias is drawn per episode. Explicit full reset retains
the existing zero-action settling tick; automatic reset timing is unchanged.
Estimated inference uses estimates from startup, without a truth bootstrap.

The standalone `PolicyHistory` mirrors this lifecycle: `append` at boundaries,
`refresh` for controller updates, and `reset(ids)` followed by a first real frame
via `refresh` for same-boundary resets. The phase period is .8 s, with demand axes
`[1,2]` and thresholds `[[.01,.05],[.10,.25]]`. The manifest supplies these values.
Previous actions are the issued clipped actions, not delayed applied actions.

Warmup runs the deterministic parent and freezes actor, critic, Gaussian std,
both normalizers and PPO optimizer. Fine-tuning uses fully estimated inputs,
retains native Adam moments and explicitly changes PPO to fixed LR `3e-5`.
`EstimatorPPO` clones history/clean current labels before `env.step`, stores the
exact effective actor observations and delegates PPO to RSL-RL 5.5.1. Estimator
weights stay fixed through rollout and PPO consumption; supervised Adam updates
run at the iteration boundary. The next action recomputes the estimate even when
the runner retains its last observation. Symmetry includes every history frame
and target, with polar velocity/axial gyro signs and task-specific joint swaps.

Full checkpoints retain native learning state plus estimator weights, Adam,
normalization, feature/history configuration, stage and counters. Initialization
from an actor requires `--resume --estimator_init --estimator_stage warmup`.
Resuming a pair omits `--estimator_init`; stage changes require explicit
`--estimator_stage finetune`. `--training_seconds` is a cumulative saved-stage
budget, not fresh time per resume. A future authorized run must subtract the
saved 931.64 s (and account separately for the interrupted pilot); do not reset
counters. Simulator/RNG trajectories are not checkpointed.

## Matched integration evidence

The installed stack is Python 3.11.14, Torch 2.9.0+cu130, Genesis 1.4.1 and
RSL-RL 5.5.1 on an RTX 4070 Ti. It was retained. New evidence is under
`evaluation/velocity_integration_20261021`; the original `velocity_frozen` and
`velocity_finetuned` screens and model/export files were left intact.

The new panel uses 12 whole 26-second sequences per candidate/condition, seeds
2843/2844, identical initial physical states, nominal physics and 10 Hz commands
with a 50 Hz policy. It covers both yaw/lateral signs, forward/reverse, mixed and
low-speed commands, six- and ten-second exact-zero windows and stop/resume.
Stress adds saved synthetic uniform sensor noise, 20 ms delay, 40 ms holding and
one 30 N push at 20.4 s; these are plausible sensitivity settings, not measured
hardware uncertainties. Ordinary stops, pre-push stops and post-push recovery are
separate. All 93,600 transitions were pre-reset and valid; no falls or undesired
contacts occurred. The panel took 366.64 s; CPU batch-one composed inference was
0.176 ms. This is a compact screen, not a robustness proof.

Tracking RMSE columns are `(vx m/s, vy m/s, yaw rad/s)`:

| Candidate | Nominal tracking | Stress tracking | Ten-second nominal stand path | Stress leg / wheel target slew RMS |
| --- | --- | --- | --- | --- |
| Parent + truth | (.0238,.0086,.0240) | (.0249,.0108,.0428) | 3.4 mm | .936 rad/s / 20.95 rad/s² |
| Frozen + estimator | (.0247,.0095,.0240) | (.0288,.0126,.0435) | 36.1 mm | .935 rad/s / 21.14 rad/s² |
| Fine-tuned + estimator | (.0245,.0101,.0236) | (.0285,.0134,.0454) | 59.4 mm | 1.029 rad/s / 24.08 rad/s² |

Fine-tuning improves the worst nominal ordinary-stop path (.1491 to .1251 m), but
worsens the stressed counterpart (.1568 to .1735 m). The final nominal stand
planar-speed RMS is .0003/.0036/.0064 m/s for parent/frozen/fine-tuned. Across all
cases, worst final wheel-speed RMS is .025/.048/.082 rad/s nominal and
.152/.187/.214 rad/s after the stressed push: persistent residual spin, not a
large runaway. Saturation stays below .29% nominal and .18% stressed. Fine-tuned
maximum tilt is .068/.065 rad nominal/stress, versus frozen .083/.073 rad.
Vertical-speed RMS is .0352/.0289 m/s fine-tuned versus .0363/.0284 frozen.

| Pair | Estimation RMSE xyz, nominal / stress (m/s) | Stress p95 absolute xyz (m/s) |
| --- | --- | --- |
| Frozen | (.0137,.0065,.0064) / (.0215,.0134,.0181) | (.0372,.0253,.0411) |
| Fine-tuned | (.0070,.0051,.0063) / (.0209,.0155,.0201) | (.0378,.0298,.0414) |

Frozen nominal x bias is -.00830 m/s, fine-tuned -.00071 m/s. Stress maximum
absolute xyz errors are (.265,.243,.184) and (.208,.213,.191) m/s respectively;
tails remain important. The coupled fine-tuning stage introduced the additional
slew/drift tradeoff; this three-way screen does not isolate its actor changes from
its estimator changes. It does not imply that training RMSE predicts deployment.

The existing proposed tolerance panel was reused without loosening thresholds.
Only stop-path comparisons fail: frozen 6/12 nominal and 3/12 stressed cases;
fine-tuned 7/12 and 6/12. The old short pilot passed for the fine-tuned pair, which
is compatible with these longer-window failures. Historical
`final_stop_drift_rms_m_s` remains `sqrt(mean(v_xy**2))` over time and components.
The new `final_stop_planar_speed_rms_m_s` is `sqrt(mean(vx**2+vy**2))`; thresholds
were not reinterpreted. Signed heading drift uses unwrapped projected body +X
heading, with the incoming boundary included. Final yaw RMS, residual planar
speed and settling are reported separately in `stop_windows`.

All nominal cases settle by the diagnostic rule (planar <=.02 m/s and
abs(yaw-rate)<=.03 rad/s continuously to window end for >=.25 s). None of the
stressed cases, including the truth parent, meets that strict remaining-quiet
rule. Maximum final/recovery yaw RMS is .039/.039/.043 rad/s for
parent/frozen/fine-tuned. That sensitivity is not evidence of one unique cause.
The unchanged pilot traces were also reanalyzed in `pilot_stop_reanalysis.json`;
their original metrics were not overwritten.

## What the training plots say

Raw TensorBoard event files confirm final warmup/fine-tuning RMSE xyz of
(.01419,.01318,.01080)/(.02652,.01994,.01592) m/s, reward 140.36/126.12 and episode
length 2992.5/2980.1 of 3000. These are different on-policy distributions.
Fine-tuning first-ten to last-ten mean RMSE falls from
(.0508,.0306,.0205) to (.0271,.0200,.0152). The reward/length ramp includes the
initial episode/log-window fill and is not evidence for another 1000–2000 updates.

Late-ten mean leg-action-rate penalty is -.00107/-.00524; wheel-action-rate
-.000044/-.000598; zero-command wheel-rate -.00600/-.02413; sensor vertical
velocity -.00320/-.00703; roll/pitch-rate -.00582/-.02936. Phase support/clearance
also worsen, while reference-pose improves (-.03675 to -.03194). Final tracking
reward x/y/yaw is (.893,.906,.656)/(.865,.867,.601). All PPO loss scalars are finite.
Policy mean std falls only .11332 to .11313 (about .17%), with effective per-action
std near .10–.15; this is not collapse. Differential entropy remains finite around
-12.29. Late throughput is about 7060/5968 transitions/s, collection 2.27/2.34 s
and learning .057/.412 s per iteration. Distribution shift is a plausible
contributor, not an established sole cause. Matched deterministic evaluation
above, not these training scalars alone, informs the selection.

## Native IMU feasibility and hardware limits

The production path currently derives body gyro from simulator angular motion
and projected gravity from the ideal root quaternion. It does **not** read a
native Genesis IMU. Defaults remain delay=0, hold=1, bias_fraction=0; training
uses synthetic uniform observation noise. No acceleration channel is present.

The measured URDF retains Unitree's authored `imu` fixed link at
`[-.02557,0,.04232]` m from `base_link`, with zero mounting rotation. The normal
loader merges fixed links, and the analytic path has no sensor attachment.
The temporary probe retained `imu` explicitly and attached before `scene.build`.
This is an authored mounting specification, not a hardware measurement of the
particular robot. Ideal gyro is the same anywhere on that rigid body after axis
rotation; moving the linear-velocity target to the IMU would change the contract.

Installed Genesis 1.4.1 supports the documented
[IMU API](https://genesis-world.readthedocs.io/en/v1.4.0/user_guide/sensing/imu.html).
A two-environment/four-second probe recorded 800 physics boundaries. Transformed
native gyro differs from analytic body gyro by at most 1.79e-7 rad/s; zero-noise
measured and truth reads agree exactly. Configured 10 ms delay plus known bias
matches a two-step shift within 2.79e-8 rad/s. Repeated reads are identical.
Standing accelerometer z averages 9.812 m/s²: specific force, not velocity.
One batched sensor read averaged .080 ms; the four-second instrumented probe took
8.95 s. This is not a large-batch training-overhead benchmark. The same probe was
rerun once after fixing a CPU/GPU mismatch in its offline summary, without changing
sensor parameters. Production code and input signatures were not changed.

Keep synthetic channels for the preserved comparisons. A separately authorized
native-gyro trial is feasible within the same 55-feature interface, after auditing
mounting, timestamps, bias and filtering. Apply noise once, either in native IMU
or the history acquisition layer. Native IMU does not provide the filtered
orientation/projected gravity required by the actor; an orientation-source audit
is still necessary. Acceleration would need lever-arm, gravity, frame, filtering
and reset treatment and a new trained contract; it must not be added blindly.

## Commands and portability

PowerShell from the actual checkout. Install the optional export and manual-control
dependencies normally; no private cache directory is part of the interface:

```powershell
Set-Location 'C:/Users/Liamb/SynologyDrive/TUM/3_Semester/dodo_alive/legged-robot_rl_genesis'
$py = 'C:/Users/Liamb/anaconda3/envs/genesis-gpu/python.exe'
$env:PYTHONIOENCODING = 'utf-8'
& $py -m pip install -r requirements-export.txt pygame
$parent = (Resolve-Path 'ressources/pretrained/go2w/reference').Path
$warm = (Resolve-Path 'ressources/pretrained/go2w/checkpoint').Path
$pair = (Resolve-Path 'logs/go2w_velocity/finetune_2026-10-21_13-00-42').Path
& $py -m robot_gym.scripts.play --task go2w --load_run "$warm" --checkpoint 2197 --num_envs 1 --velocity_source estimated --manual_control
# To inspect the fine-tuned pair, use --load_run "$pair" --checkpoint 2296.
$out = "evaluation/velocity_review_$(Get-Date -Format yyyyMMdd_HHmmss)"
& $py -m robot_gym.scripts.evaluate --task go2w --load_run "$pair" --checkpoint 2296 --estimator_screen --comparison_checkpoint "$parent/model_1999.pt" --frozen_checkpoint "$warm/model_2197.pt" --headless --output "$out"
& $py -m robot_gym.scripts.play --task go2w --load_run "$warm" --checkpoint 2197 --num_envs 1 --velocity_source estimated --steps 1 --headless --export --onnx --export_dir "$out/export" --parity_histories "$out/nominal_frozen_estimated.npz"
& $py "$out/export/deployment_history.py" "$out/export"
```

Manual control starts at zero before reset, preserves adjustable directional
limits, Shift slow, Space stop, focus-loss release gating and safe window cleanup.
It is unlimited unless `--steps` is explicit. Estimated input is the default for
pair checkpoints; `--velocity_source truth` is an explicit diagnostic comparison.
Estimated `sensor_sustained` is rejected before construction because its actor-slot
phase counterfactuals have no defined history intervention. Use `transfer_screen`,
`precision_dr` or the matched estimator panel. `source_yaw` remains the immutable
truth-input reference and rejects an estimated source.

Exports refuse to overwrite an existing directory. Interface 3 remains stateless
float32 `[batch,5,55] -> clipped_actions[batch,16], velocity_m_s[batch,3]`.
The bundle includes embedded normalization, actor-only comparison TorchScript,
composed TorchScript/ONNX, source/checkpoint/config identities, contract and real
parity histories. No critic, simulator truth or training-log dependency is present.
History/phase bookkeeping remains outside the graph and is reproducible with
NumPy. Actions map to P-leg position/V-wheel speed targets, not final motor torques.
Do not invent a Unitree permutation: policy order interleaves wheels; the official
SDK example groups twelve leg motors followed by four wheels.

## Checks and next work

Eight focused estimator tests pass, as do six manual-control and four source-yaw
tests. The final full suite has 110 tests, two failures and five errors, exactly
matching an isolated unchanged `origin/testing@3fe238c` baseline. Existing failures
are `test_resume_restores_complete_saved_config` (stale schema-1 expectation) and
`test_urdf_nominal_and_actuator_conventions` (stale .2 action-scale expectation).
Existing errors are `test_phase_boundary_reset_and_observation`,
`test_unique_vectors_joints_and_actions`,
`test_closed_loop_hold_censoring_and_light_recording`,
`test_precision_schedules_and_terminal_alignment`, and
`test_sustained_schedule_and_censoring` (incomplete test fixtures). Assertions were
not weakened. The native eight-condition `precision_dr` run passed with estimated
inputs, 1300 ticks each and no failures, in 180.04 s. Same-tick selective reset,
repeated getters/RNG, full state loading and privileged-input isolation are checked.
The original portable package still reconstructs history exactly and matches ORT
on 5600 recorded inputs (max action error 6.56e-7, velocity 2.98e-7 m/s).

A narrow replay fix also restores the missing `GaussianDistribution` name in the
shipped early Dodo/Go2 configs only when their saved scalar `std_param` identifies
it; all saved numeric settings remain unchanged. The same omission fails on
unchanged testing. Both full pretrained checkpoints then passed two-tick native
replay with the estimator disabled. The supported `EstimatorPPO` diagnostic path
passed collection and native minibatch/KL hooks with zero optimizer steps and
unchanged learned state/counters. The existing manual-control/Genesis windows
completed a 50-tick estimated replay; key and focus-loss behavior was covered by
tests, not a human visual gait review.

The fresh bundle at `evaluation/velocity_integration_20261021/export` contains the
updated portable reset helper and expanded screen (marked **failed**). Its ONNX
hash is identical to the original. Eager/ORT and isolated portable replay pass
15,600 recorded histories, 12 startup/reset samples and 426 clipped-action histories:
max action error 8.94e-7, velocity 3.58e-7 m/s, history reconstruction error zero.
The isolated checker imports no Genesis, RSL-RL or repository package.

No IsaacLab adapter is available in this repository. In that separate project:

1. Import the complete ONNX plus manifest and implement scaled feature ordering,
   oldest-first history, reset padding, demand phase and previous issued clipped
   actions. Check real-recording parity before closed-loop replay.
2. Use the same measured URDF, trained pose, joint order, nominal dynamics, P/V
   targets, .005/.02 s rates and matched initial states/command traces. Keep
   base-origin truth only for metrics and the parent comparison.
3. Run parent+truth and both composed pairs on the recorded panel. Record actuator
   target/response, wheel loads/contact forces, stopping and yaw. The October 11
   source report shows good Genesis yaw but PhysX undershoot with truth inputs;
   resolve contact/actuation response independently of velocity estimation.

The required yaw envelope is +/-1.0 rad/s, navigation's handoff asks +/-1.25, and
training reserve reaches +/-1.5. Reserve samples are not a tested envelope. This
integration neither edits the navigation project nor qualifies +/-1.25.

On the separate ROS2/Jetson project audit actual Go2-W LowState/SDK2 topic/type,
IMU mount/axis/quaternion conventions, orientation filter and gravity convention,
gyro bias/noise/sample age, joint and wheel encoder signs/order/units/filtering,
source timestamps/jitter/dropouts, 10 Hz command versus 50 Hz policy timing,
actuator delay/holding, and the low-level motor permutation. Official
[LowState](https://github.com/unitreerobotics/unitree_ros2/blob/master/cyclonedds_ws/src/unitree/unitree_go/msg/LowState.msg)
contains IMU/motor fields but no base linear velocity;
[SportModeState](https://github.com/unitreerobotics/unitree_ros2/blob/master/cyclonedds_ws/src/unitree/unitree_go/msg/SportModeState.msg)
has a separate velocity field whose continuing validity under custom low-level
control must be checked on the robot. Nothing here verifies those hardware signals.

## Another robot and reference method

Another task implements `velocity_estimator_spec()` with its velocity/feature/live
slots, units/scales and reference point, assembles measurements through
`measured_observations(clean)`, keeps truth critic inputs separate and extends its
symmetry mapping to history/labels if used. Opt in through the same saved estimator
configuration and entry points. No Go2-W edits are needed. Dodo/Go2 stay disabled
until they declare a contract; the compact command panel is Go2-W-specific.

The principle follows [basic-locomotion-isaaclab at dd92b31](https://github.com/iit-DLSLab/basic-locomotion-isaaclab/tree/dd92b3116f89caf8b083490d5d815fdbff8ac665): supervised history-to-velocity learning followed
by estimated actor inputs. Its default-disabled Go2 method includes acceleration.
This implementation copies no upstream source, callback fitting loop, thresholds,
TCN or twelve-joint indexing. Its COM-velocity alias is not our base-origin target.
Upstream is BSD-3-Clause, copyright 2025 DLS Lab at Istituto Italiano di Tecnologia.
