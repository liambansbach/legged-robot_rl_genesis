# Optional velocity estimator

The estimator predicts body-frame `(vx, vy, vz)` in m/s at the **authored base-link
origin**, matching the existing actor. It does not predict COM velocity. It is an
opt-in path through the existing registry, train, play, evaluate and export entry
points. `env_cfg.velocity_estimator: null`, including a missing field in older
saved configurations, means disabled. No parent package is overwritten.

## Baseline and scope

The implementation started from clean local `testing` at
`11072c4789fedd3aead08499628a4e0634699fb2`, not the public review's `3fe238c`.
No applicable AGENTS.md was found. The Windows desktop has an RTX 4070 Ti;
the working interpreter is `C:/Users/Liamb/anaconda3/envs/genesis-gpu/python.exe`:
Python 3.11.14, Torch 2.9.0+cu130, Genesis 1.4.1, RSL-RL 5.5.1. The stack was
retained. Genesis imports stalled in the restricted execution sandbox but worked
outside it. Optional ONNX Runtime 1.31.0 was installed in ignored `.cache/onnx-runtime`.

The latest accepted-for-comparison local source remains the experimental handoff:

* Full parent: `ressources/handoffs/go2w_2026-10-09/runs/go2w_2026-10-08_23-45-24/model_1999.pt`
* Full checkpoint SHA256: `2f8f3a96ad7fdf83c2966b44d10fc76ac274d9d81130a470af881f81a5980bf6`
* Corresponding `inference/go2w/policy.pt` SHA256:
  `639a443f3da3d9fa15d9e532b85959df55c7bd5618564797740adb66b1d2bfcb`

Actor 58, critic 65, actions 16; policy period .02 s, physics period .005 s.
The saved pose, normalizers, P-leg/V-wheel mapping, issued-action convention,
phase and reset timing remain authoritative. The older pretrained package is a
different policy. Both it and the accepted handoff remain unchanged.

The latest local review is October 9: lateral drift and apex clipping remain
limitations. No newer local sim2sim result or IsaacLab adapter was found. The
October 11 source-yaw report in the task describes a Genesis/PhysX difference
even with ground-truth inputs; this estimator does not resolve that contact or
actuator mismatch. Required export yaw is +/-1.0 rad/s, the navigation handoff
uses +/-1.25, and reserve training reaches +/-1.5. These are different claims;
reserve samples do not establish a tested envelope. The navigation repository
was not edited.

## Feature and sensor contract

Five chronological frames at 50 Hz span **80 ms**, oldest to newest. For the
current Go2-W task each frame contains 55 scaled actor features:

| Frame slots | Signal | Raw units |
| --- | --- | --- |
| 0:3 | Body gyro | rad/s |
| 3:6 | Projected gravity | unit vector |
| 6:9 | Body command vx, vy, yaw | m/s, m/s, rad/s |
| 9:21 | Twelve leg joint offsets from trained reference | rad |
| 21:37 | All sixteen joint speeds, including wheels | rad/s |
| 37:53 | Previous **issued clipped** action | dimensionless |
| 53:55 | Demand-conditioned sine/cosine phase | dimensionless |

The task declares selection and scaling; shared code does not assume 55 features
or twelve joints. Absolute wheel angles remain excluded. There is no simulator
velocity, position, contact truth, geometry, mass, friction label or future sample
in deployable inputs. Zero command does not zero the prediction.

The MLP defaults to `[128,128,128]`, ReLU, and a linear three-component output.
Feature statistics are initialized once from the parent's trained actor statistics
for those feature slots, with its epsilon and a .01 denominator floor. They remain
fixed and are saved separately from the actor's own normalization. Features enter
in scaled actor units; the estimator standardizes them once, predicts m/s, and the
composition scales velocity once before the actor's embedded normalizer. The
original actor normalizer is never applied twice.

One measured frame is cached per real policy boundary. Sensor noise, optional
episode bias, integer-tick delay and sample holding are coherent across channels.
Current commands, issued actions and phase are controller state and are not
sensor-delayed. Repeated getters, checkpoint loading and manual command changes
do not shift history, redraw noise or advance phase. Manual updates refresh only
the newest controller slots. Automatic resets invalidate only affected rows.
Their first measured frame pads all history slots; explicit full reset still
includes the existing zero-action settling tick. Estimated playback uses the
estimator from its first action, without a truth-velocity bootstrap.

The default keeps saved sensor noise and adds no bias, delay or sample holding
beyond one policy tick. `sensor_bias_fraction`, `sensor_delay_steps` and
`sensor_hold_steps` are explicit sensitivity settings, not measured hardware
models. The supervised target is clean **current** velocity, even with delayed
measurements. Rewards and critic use simulator truth; `env.base_lin_vel` is never
overwritten. Auxiliary TensorDict groups are excluded from actor/critic inputs.

Acceleration is deliberately absent. This feature set is a hypothesis. If slip
or transient failures dominate, assess longer history or a matched accelerometer
before changing gait rewards. An accelerometer model would need proper acceleration
at the actual IMU, gravity/frame rotation, lever arm, filtering and reset handling;
differencing body-frame velocity is insufficient.

## Training and replay

Commands below are PowerShell from the repository root with `genesis-gpu` active.
Absolute run directories avoid the existing resolver treating them as log labels.
If `python` resolves the desktop's Python 3.14, use the full 3.11 interpreter path
listed above instead. Supported saved Go2-W schema 2/3 behavior is preserved;
older incompatible reward/critic schemas retain their existing historical-source
requirement and are not silently rebuilt from current defaults.

```powershell
$parent = (Resolve-Path 'ressources/handoffs/go2w_2026-10-09/runs/go2w_2026-10-08_23-45-24').Path
python -m robot_gym.scripts.train --task go2w --resume --load_run "$parent" --checkpoint 1999 --estimator_init --estimator_stage warmup --experiment_name go2w_velocity --run_name warmup --num_envs 256 --max_iterations 200 --seed 23 --logger tensorboard --headless --training_seconds 1800
```

Warmup runs the deterministic parent with clean simulator velocity and measured
nonvelocity channels. Actor, critic, Gaussian std, both normalizers and native
Adam state stay bitwise unchanged; completion checks this against the full parent.
Only the estimator's persistent Adam optimizer learns. It performs 16 vectorized
minibatches per collection, outside environment callbacks. Saved physical
randomization, reward design and command distribution are retained: these ranges
are configured assumptions, not hardware uncertainty measurements.

Set `$warm` to the resulting absolute run directory. Screen entire held-out
command sequences and perturbation seeds before deciding whether PPO is useful:

```powershell
$warm = (Resolve-Path 'logs/go2w_velocity/warmup_2026-10-21_12-45-46').Path
python -m robot_gym.scripts.evaluate --task go2w --load_run "$warm" --checkpoint -1 --estimator_screen --headless --output evaluation/velocity_frozen
```

If adaptation is needed, initialize from that full pair; omit `--estimator_init`:

```powershell
python -m robot_gym.scripts.train --task go2w --resume --load_run "$warm" --checkpoint -1 --estimator_stage finetune --experiment_name go2w_velocity --run_name finetune --num_envs 256 --max_iterations 100 --seed 29 --logger tensorboard --headless --training_seconds 1650
```

Fine-tuning retains native Adam moments but intentionally changes PPO to fixed
LR `3e-5`. Existing symmetry stays enabled; labels and every history frame are
reflected with correct polar/axial signs and joint swaps. Policy storage contains
the actual estimated-input observations used for actions and old log probabilities.
Estimator weights/statistics stay fixed during rollout and PPO consumption, then
supervised updates run at the iteration boundary. The next `act()` recomputes its
estimate from the runner's retained history without advancing it. Inputs are fully
estimated throughout fine-tuning; there is no implicit transition curriculum.

Resume either stage with `--resume --load_run <pair-directory> --checkpoint -1`,
omitting the stage to retain its saved value. Checkpoints include full native
learning state plus estimator weights, Adam, feature statistics, architecture,
history spec and explicit stage counters. Native iteration labels remain native;
they do not select estimator stages. Simulator/RNG state is not saved, so resuming
starts new episodes and does not promise an identical simulator trajectory.

The wall-time budget accumulates across saved estimator stages. Use fewer updates
or a larger explicitly chosen future budget only after reviewing the pilot. No
automatic sweep, repeated fit loop or checkpoint-dependent threshold is used.

Play a pair with the shared optional keyboard controller (`pip install pygame`):

```powershell
python -m robot_gym.scripts.play --task go2w --load_run "$warm" --checkpoint -1 --num_envs 1 --pygame --steps 10000
python -m robot_gym.scripts.play --task go2w --load_run "$warm" --checkpoint -1 --velocity_source truth --command_vx .5 --steps 500
```

W/S forward, A/D lateral, Q/E yaw, Space stop, Esc quit. Estimated-input mode is
printed explicitly and is the default for pair checkpoints. The live checkout
had no shared pygame controller; this small optional controller uses the existing
fixed-command entry point. No pygame dependency is needed for headless workflows.

## Export and portable replay

```powershell
python -m pip install -r requirements-export.txt
python -m robot_gym.scripts.play --task go2w --load_run "$warm" --checkpoint -1 --headless --num_envs 1 --steps 1 --export --onnx --parity_histories evaluation/velocity_frozen/nominal_estimated.npz
```

The selected run receives `exported/model_<label>/policy.onnx`,
`estimated_policy.pt`, actor-only `policy_1.pt`, `contract.json`, runtime config,
dynamics, portable `deployment_history.py` and real `parity_samples.npz`.
The **version 3** stateless graph takes float32 `[batch,5,55]` history and returns
bounded actions plus m/s velocity. It exports no critic and needs no truth input.
Feature normalization, actor normalization and action bounds are embedded once.
No training log is required after exporting. Run its portable checker with:

```powershell
python '<export-directory>/deployment_history.py' '<export-directory>'
```

The checker uses only NumPy, Torch and ONNX Runtime. `PolicyHistory` reproduces
oldest-first append, selective resets, repeated-first-frame padding and controller
refreshes. `phase_features` accepts task-specified axes and thresholds; Go2-W uses
axes `[1,2]`, thresholds `[[.01,.05],[.10,.25]]`, period .8 s. At action boundary p,
hold the action for four .005 s physics steps, then advance p by .02/.8 modulo 1.
Automatic reset replaces phase before forming the next observation. Match the
explicit-reset settling convention when replaying recorded simulator startup.

The manifest records checkpoint/tensor/graph hashes, saved configuration, source
HEAD/dirty status, rates, reference point, sensors, history semantics and bounds.
Actuator outputs become position targets for legs and speed targets for wheels;
they are not final motor torques. Never infer a hardware motor permutation from
policy order: this actor interleaves each wheel, while the
[official SDK example](https://github.com/unitreerobotics/unitree_sdk2/blob/main/example/go2w/go2w_stand_example.cpp)
groups twelve legs before four wheels.

## Validation and remaining work

The native GPU replay smoke passed. Seven focused CPU/native-PPO tests cover old
saved configs, causal history/getters/resets, privileged-input isolation, full
history symmetry, frozen state, checkpoint reload and TorchScript/ONNX parity.
The complete suite has the same three failures and twelve errors as untouched
HEAD (stale legacy fixtures); no additional failures remained in the final
99-test run. Compile checks and `git diff --check` also passed.

The frozen candidate is `logs/go2w_velocity/warmup_2026-10-21_12-45-46/model_2197.pt`,
SHA256 `c35c5b3861a0f7d0de1975a96e80dc6d45831ecad225b5a82d8031e63e95d9bf`.
It contains 199 completed collections, 3,184 supervised Adam steps and zero PPO
updates. A first attempt completed one collection before a native logging error;
it was repaired and the remaining budget reduced to 199. Total pilot expenditure
was 200 collections and 674.75 seconds including both setups. The retained run
took 541.10 seconds. Both runs verified the full parent learning state unchanged.
Final training-rollout RMSE was `(0.0142, 0.0132, 0.0108)` m/s; this is not held-out
or closed-loop qualification. The saved native label 2197 is not an estimator
stage count. This recovery is not a second hyperparameter experiment.

The frozen pair passed all eight nominal comparisons. Aggregate held-out estimator
RMSE was `(0.01913, 0.01009, 0.00907)` m/s nominal and
`(0.02884, 0.01977, 0.02551)` m/s under sensor stress. All four panels had zero
falls and zero abnormal-contact ticks. Nominal startup histories and velocity
targets matched exactly between parent and pair. The real lifecycle checks passed
(getters/phase/RNG, selective reset and first-frame padding). Eager CPU batch-one
combined inference averaged .194 ms; the screen took 196.38 seconds including setup.

One stress comparison failed the predeclared stop-path tolerance: negative-lateral
second stop was .091565 m versus parent's .048091 m; the limit was
`1.25 * .048091 + .03 = .090114 m`. This is a 1.45 mm screening miss, not a
hardware safety threshold. Its velocity bias was `(-.01278,-.00116,-.00361)` m/s,
per-axis p95 absolute error `(.04659,.05088,.06421)` m/s, and coarse lag
`(.08,.04,.04)` s. Nominal negative-lateral lag was `(.04,.02,0)` s. Full per-case
tracking, drift, tilt, height and saturation metrics are in
`evaluation/velocity_frozen/metrics.json`; full temporal records remain alongside it.
A single conservative 100-update fine-tuning pilot was selected to assess this
estimated-input distribution shift, keeping rewards and randomization fixed.

The frozen ONNX graph SHA256 is
`5117e7a57c39426a1da6d6ceedb2c242d6642e6598ca80663b669e1619fc9c60`.
Its estimator tensor SHA256 is
`58d8c942a33b47bddb545ed254906ca44fb845696783aefa273ae9af514199c0`.
CPU eager/ONNX Runtime parity on 5,600 real histories had maximum action error
`6.855e-7` and velocity error `2.981e-7` m/s, below the declared `2e-5` tolerance.
An isolated `python -I` process loaded the copied package with no Genesis, RSL-RL
or repository imports. Portable reconstruction had exactly zero history error,
including eight startup/reset samples and 379 histories with clipped actions.

### Retained simulation candidate

The single 100-update fine-tuning pilot passed **all nominal and sensor-stress
comparison criteria** against the immutable parent. Retain
`logs/go2w_velocity/finetune_2026-10-21_13-00-42/model_2296.pt`, SHA256
`8c06103a914084bd99870e63f5496a2d9ab768e11d57fc6f56b5920f5403c639`.
The full checkpoint contains warmup counter 199, fine-tune counter 100, supervised
Adam step 4784 and native Adam step 84040. Native LR is explicitly `3e-5`, with
retained moments. The original parent and frozen pair are unchanged fallbacks.
Fine-tuning took 390.57 s; total training, including the failed first warmup and
all training setup, was **1065.32 s (17 min 45 s)**. No further training was run.

| Final pair estimator error, m/s | Nominal | Sensor stress |
| --- | --- | --- |
| Bias x/y/z | .00065 / .00060 / .00258 | .00025 / .00085 / .00268 |
| RMSE x/y/z | .00990 / .00697 / .00841 | .02670 / .01950 / .02728 |
| p95 absolute x/y/z | .02073 / .01572 / .01188 | .04816 / .03693 / .05969 |
| Maximum absolute x/y/z | .10594 / .05240 / .16856 | .23829 / .21646 / .22522 |

These aggregate whole held-out episodes, including startup and transitions.
Negative-lateral coarse correlation lag x/y/z is `.02/0/0` s nominal and
`.04/.04/.04` s under stress. It is a 20 ms-resolution correlation diagnostic,
not a calibrated estimator latency. Stress tails remain substantially larger
than nominal errors.

| Control metric: original truth parent -> final estimated pair | Nominal | Sensor stress |
| --- | --- | --- |
| Tracking RMSE vx/vy/yaw (m/s, m/s, rad/s) | .03954/.01276/.03536 -> .03956/.01426/.03507 | .04037/.01495/.05223 -> .04201/.01789/.05629 |
| Maximum stop path across cases, m | .11444 -> .11793 | .11969 -> .11935 |
| Maximum final-stop XY RMS speed, m/s | .00649 -> .00769 | .00732 -> .01456 |
| Maximum roll/pitch norm, rad | .07679 -> .06610 | .07424 -> .07262 |
| Minimum base height, m | .39763 -> .39656 | .40096 -> .39783 |
| Action saturation fraction | .661% -> .661% | .420% -> .482% |
| Falls / abnormal-contact ticks | 0 / 0 -> 0 / 0 | 0 / 0 -> 0 / 0 |

The formerly failing negative-lateral stress stop is now .06677 m, below its
.09011 m criterion (parent .04809 m, frozen pair .09157 m). Improved nominal
estimation does not imply every metric improved: drift and some tracking errors
remain higher than privileged truth. All predeclared relative tolerances passed.
Cross-axis motion and every individual case are retained in
`evaluation/velocity_finetuned/metrics.json`, with temporal arrays and a compact
summary. Eager CPU batch-one estimator+actor averaged .198 ms; the matched screen
took 192.93 s including setup. This qualifies only this simulation screen, not
the navigation application's stricter limits or real hardware.

Final ONNX SHA256:
`077575ccbafe456cdf6b89f688ca5d5fa6bf488cd7d2c99b7e8b81ad3e58d66c`.
Final actor/estimator tensor SHA256 values are respectively
`fb91378d9981a9f4fa230c89b7ffa247e447374d7963c929d1ae965edeb3ba37` and
`8fc3659fdbd27940fb4f78a79cc3b50fb5d87fc2e9148ffc97d1114bfc95d1b7`.
The final export passed eager/ORT and isolated TorchScript/ORT checks on 5,600
real histories: maximum action error `6.557e-7`, velocity error `2.981e-7` m/s,
exact portable history reconstruction, eight reset/startup samples and 354
clipped-action histories. The package includes its passing screen definition,
metrics, checkpoint/config identity, runtime identity and a source-tree hash.

Ready-to-run commands for the retained pair:

```powershell
$parent = (Resolve-Path 'ressources/handoffs/go2w_2026-10-09/runs/go2w_2026-10-08_23-45-24').Path
$pair = (Resolve-Path 'logs/go2w_velocity/finetune_2026-10-21_13-00-42').Path
python -m robot_gym.scripts.play --task go2w --load_run "$pair" --checkpoint 2296 --num_envs 1 --pygame --steps 10000
python -m robot_gym.scripts.evaluate --task go2w --load_run "$pair" --checkpoint 2296 --estimator_screen --comparison_checkpoint "$parent/model_1999.pt" --headless --output evaluation/velocity_pair_recheck
python -m robot_gym.scripts.play --task go2w --load_run "$pair" --checkpoint 2296 --num_envs 1 --steps 1 --headless --export --onnx --parity_histories evaluation/velocity_finetuned/nominal_estimated.npz
# Optional future continuation; not needed for the completed pilot:
python -m robot_gym.scripts.train --task go2w --resume --load_run "$pair" --checkpoint 2296 --experiment_name go2w_velocity --run_name continued --num_envs 256 --max_iterations 100 --seed 31 --logger tensorboard --headless --training_seconds 1650
```

Pygame 2.6.1 was tested using SDL dummy video/audio drivers, including moving
commands and exact Space-stop. No visible control window or hardware was opened;
interactive Genesis-viewer use remains untested. Optional dependencies were
installed only into ignored cache directories during verification.

The screen predeclares proposed simulation tolerances in `screen_definition.json`.
It separates velocity bias/RMSE/tails/coarse transient lag from tracking, stopping
path/drift, cross-axis motion, tilt/height, saturation, falls, abnormal contacts and
inference runtime. It uses eight whole command sequences at 50 Hz, 10 Hz command
changes, and nominal versus delayed/held/noisy sensor conditions with a stop push.
It is not a hardware safety envelope, broad robustness claim or PhysX validation.

No local IsaacLab adapter was available. Replay handoff: use interface 3 ONNX,
build scaled features in manifest order, maintain history and phase exactly as
above, verify zero-input/real-recording parity, then run the same declared command
sequences/initial conditions in PhysX for both parent truth and estimated pair.
Use authored base-origin truth only for comparison metrics, not the combined
graph. Resolve the known yaw dynamics difference before attributing drift to the
estimator. Keep motor order and transport in the separate deployment repository.

Official [LowState](https://github.com/unitreerobotics/unitree_ros2/blob/master/cyclonedds_ws/src/unitree/unitree_go/msg/LowState.msg)
contains IMU and motor measurements, without a base-linear-velocity field.
[SportModeState](https://github.com/unitreerobotics/unitree_ros2/blob/master/cyclonedds_ws/src/unitree/unitree_go/msg/SportModeState.msg)
has a separate velocity field. Its availability, timestamp, reference frame and
continued validity during custom low-level control require an actual robot signal
audit; neither availability nor unavailability is assumed here. No ROS2 transport
or hardware operation was added.

## Another robot

Declare `velocity_estimator_spec()` in its task: the three velocity slots,
deployable feature slots, live controller slots, units/scales and reference point.
Return a separate truth critic group and route observation assembly through
`measured_observations(clean)` (the shared LeggedRobot path already does this).
Provide its task-specific reflection for auxiliary history/target groups if it
uses symmetry. Opt in with the same estimator configuration and CLI. No Go2-W
code changes are necessary. Dodo/Go2 remain disabled until their task explicitly
provides this contract; the compact evaluation command panel is Go2-W-specific.

## Reference method

The principle follows [basic-locomotion-isaaclab at dd92b31](https://github.com/iit-DLSLab/basic-locomotion-isaaclab/tree/dd92b3116f89caf8b083490d5d815fdbff8ac665):
supervised history-to-velocity learning followed by estimated actor inputs. Its
Go2 config disables estimation by default and includes acceleration. This version
does not copy its implementation, TCN, fitting callback, fixed 6000/7000 thresholds,
CWD checkpoints, bounded velocity output or twelve-joint deployment indexing.
The reference's IsaacLab target aliases COM velocity; this project uses its saved
actor's authored-origin target. Upstream code is BSD-3-Clause, copyright 2025 DLS
Lab at Istituto Italiano di Tecnologia; no upstream source was copied here.
