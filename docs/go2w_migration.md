# Go2-W flat-locomotion migration and training guide

Reward settings below include the small [v2 pilot refinement](go2w_v2.md). Migration and smoke measurements in this document describe the original baseline; the v2 report records the real 200-update pilot diagnosis and subsequent validation.

This implements the preparation milestone on `testing`: a velocity-conditioned, proprioceptive MLP with 16 mixed P/V actuators. The software and short training/inference paths have been exercised locally. **The smoke checkpoint is not a trained locomotion controller.** Full training, checkpoint selection, and IsaacLab integration remain subsequent steps; no full training was started.

## A. Environment migration

| Component | Before | Tested target |
|---|---|---|
| Python | 3.11.14 | 3.11.14 |
| PyTorch | 2.9.0+cu130 | 2.9.0+cu130 |
| torchvision | 0.24.0+cu130 | 0.24.0+cu130 |
| PyTorch CUDA runtime | 13.0, available | 13.0, available |
| Genesis | 0.4.6 | 1.4.1 |
| RSL-RL | 5.2.0 | 5.5.1 |
| Quadrants | 0.6.2 | 1.3.0 |
| Pydantic / core | 2.12.3 / 2.41.4 | 2.13.5 / 2.46.5 |
| PyAV | absent | 18.1.0 |
| gstaichi | 2.6.0 | removed |

Hardware: NVIDIA RTX 4070 Ti, 12 GB VRAM. The existing `genesis-gpu` Conda environment was upgraded in place. `pip check` passed before and after migration. The resolver preserved Torch/CUDA. `gstaichi` was removed only after checking that no installed package required it and that the repository did not import it. Rendering dependencies and Conda's older CUDA toolkit packages were retained: removing those without proving their full dependency/use relationships would be unnecessary risk to the working environment. The clean environment specification omits that old toolkit baggage and uses the CUDA 13.0 Torch wheels.

Recovery/debug snapshots are local, ignored files under `.migration-audit/`: `versions-before.txt`, `pip-freeze-before.txt`, `conda-list-before.txt`, `pip-check-before.txt`, `install-plan.json`, and corresponding `*-after.txt` snapshots. They are not a portable lockfile. `conda_env.yaml` is now a short supported-environment specification.

Upstream references checked during migration: [Genesis 1.4.1 on PyPI](https://pypi.org/project/genesis-world/1.4.1/), [Genesis release](https://github.com/Genesis-Embodied-AI/genesis-world/releases/tag/v1.4.1), [RSL-RL releases](https://github.com/leggedrobotics/rsl_rl/releases), [5.3 Beta distribution](https://github.com/leggedrobotics/rsl_rl/releases/tag/v5.3.0), and [5.4 logging migration](https://github.com/leggedrobotics/rsl_rl/releases/tag/v5.4.0). Installed release source was also inspected directly for the APIs used here.

Migration fixes:

- `set_links_mass` and `set_links_COM` replace removed mass/COM shift APIs; COM variation is restricted to the base.
- Public entity-local joint indices replace an assumption that global and local DOF indices coincide.
- Batched link state getters replace repeated foot-link getter calls.
- `get_contacts(..., is_padded=True)` avoids trimming contacts through a device-to-host count synchronization. Ground contacts are filtered and processed in batched Torch operations.
- `ViewerOptions.refresh_rate` replaces deprecated `max_FPS`, which failed during modern scene construction.
- External body wrenches replace joint-torque disturbances, preserving position and velocity control.
- Scene reset restores build-time friction ratios; resets explicitly restore the sampled floor condition.
- Rewards now score the command actually used for the transition. Reset observations refresh physical state, action history clears, and timeout extras update every step.
- RSL-RL uses its supported `WandbLogWriter` configuration. The old global W&B monkey patch and obsolete PPO adapter were removed. `--logger tensorboard` supports local smoke runs without credentials.
- Obsolete global Torch JIT profiling overrides caused trouble on this stack and were removed. Environment steps use ordinary no-grad tensors inside RSL-RL's inference-mode rollouts; this fixes a TorchScript reset error. Final validation includes process exit status, not just printed assertions.
- Export embeds the trained observation normalizer, observation clipping, and action clipping. Nonfinite rewards are no longer silently replaced by zeros.

## B. Physics configuration

| Setting | Baseline |
|---|---|
| Physics timestep / substeps | 0.005 s / 1 |
| Control decimation / policy frequency | 4 / 50 Hz |
| Gravity | (0, 0, -9.81) m/s² |
| Solver | Newton |
| Solver iterations / line-search iterations | 50 / 50 maximum |
| Integrator | Genesis default `approximate_implicitfast` |
| Friction cone | pyramidal |
| Contact resolution | convex |
| Constraint time constant | 0.01 s, twice the physics timestep |
| Multi-contact | enabled |
| Rolling / torsional friction | disabled; both material coefficients explicitly 0 |
| Collisions / joint limits / self-collision | enabled |
| Adjacent and neutral-pose self-collisions | Genesis defaults, disabled |
| Terrain | true plane |
| Ground sliding coefficient | 0.1 |
| Robot nominal sliding coefficient | 1.0 |
| Training performance mode | enabled |
| Deterministic evaluation | performance mode off; deterministic Genesis algorithms enabled |

The ground coefficient is deliberately low because Genesis combines pair friction with a maximum. Wheel coefficients 0.6–1.2 therefore remain effective; a ground coefficient of 1.0 would mask the low end. This choice is about the pair rule, not a claim that the floor itself is slippery.

Pyramidal/convex is the conservative baseline. Elliptic/Signorini gives an isotropic cone and avoids sliding increasing normal force, but the installed solver documentation warns of extra work. All three tested variants permit normal straight rolling. Scripted yaw/arc tests did not show a decisive stability improvement sufficient to justify the extra cost. The comparison includes an elliptic variant with rolling coefficient 0.0001 m and torsional coefficient 0.001 m; these are effective moment-arm lengths, not dimensionless Coulomb coefficients. They remain configurable but disabled for the first training run. No uncalibrated tire-resistance randomization is enabled.

The benchmark uses 128 environments, warmup, and 200 measured policy steps each for straight (0.5 m/s), pure yaw (1 rad/s), and an arc (1 m/s, 1.25 rad/s), with ideal wheel-rate feedforward and nominal leg targets. This is an open-loop physics test. Its falls do not measure a learned policy's performance, and its small-batch throughput does not predict throughput at 4096 environments. Variants ran sequentially on the same GPU with performance mode off.

| Variant | Straight env-policy-steps/s | Yaw env-policy-steps/s | Arc env-policy-steps/s | Falls: straight / yaw / arc |
|---|---:|---:|---:|---|
| Pyramidal / convex | 4180 | 3994 | 4124 | 0 / 256 / 768 |
| Elliptic / Signorini | 3718 | 2404 | 2006 | 0 / 256 / 640 |
| Elliptic + rolling/torsional | 4119 | 2334 | 1973 | 0 / 256 / 640 |

All variants tracked straight feedforward at approximately 0.516–0.518 m/s. Open-loop yaw averaged only 0.319–0.325 rad/s against a 1 rad/s target and suffered repeated falls; those counts include resets. This demonstrates the need for learned posture/control feedback, not successful yaw tracking. Elliptic cost about 40–51% throughput in the turning tests. Adding the small rolling/torsional terms cost another 3% in yaw and 2% in arcs; the straight timings varied enough that no isolated straight-line overhead claim is justified. Logs and the local benchmark script are under `.migration-audit/bench-final-*.txt` and `.migration-audit/physics_bench.py`.

## C. Actuator and deployment contract

The actor's Gaussian samples and deterministic means are clipped to [-1, 1] before control and before storage as previous action. Exported actions have that same bound. Leg targets are `q_nominal + 0.2 * action` radians. Wheel targets are `18 * action` rad/s, with a second physical target clamp at ±20 rad/s. The normalized policy normally reaches ±18, leaving a separate safety cap for future scale changes. The URDF wheel speed limit is 30.1 rad/s; it is a model limit, not independently certified hardware data.

Action order follows the URDF, **interleaving each wheel with its leg**, rather than putting all wheels last:

| Action | Joint | Mode | Scale | Nominal q (rad) | Effort limit (Nm) |
|---:|---|:---:|---:|---:|---:|
| 0 | FL_hip_joint | P | 0.2 rad | 0 | 23.7 |
| 1 | FL_thigh_joint | P | 0.2 rad | 0.70 | 23.7 |
| 2 | FL_calf_joint | P | 0.2 rad | -1.33 | 35.55 |
| 3 | FL_foot_joint | V | 18 rad/s | excluded | 23.7 |
| 4 | FR_hip_joint | P | 0.2 rad | 0 | 23.7 |
| 5 | FR_thigh_joint | P | 0.2 rad | 0.70 | 23.7 |
| 6 | FR_calf_joint | P | 0.2 rad | -1.33 | 35.55 |
| 7 | FR_foot_joint | V | 18 rad/s | excluded | 23.7 |
| 8 | RL_hip_joint | P | 0.2 rad | 0 | 23.7 |
| 9 | RL_thigh_joint | P | 0.2 rad | 0.75 | 23.7 |
| 10 | RL_calf_joint | P | 0.2 rad | -1.31 | 35.55 |
| 11 | RL_foot_joint | V | 18 rad/s | excluded | 23.7 |
| 12 | RR_hip_joint | P | 0.2 rad | 0 | 23.7 |
| 13 | RR_thigh_joint | P | 0.2 rad | 0.75 | 23.7 |
| 14 | RR_calf_joint | P | 0.2 rad | -1.31 | 35.55 |
| 15 | RR_foot_joint | V | 18 rad/s | excluded | 23.7 |

Nominal leg Kp/Kd is 40 Nm/rad and 1 Nm·s/rad; wheel Kp/Kv is 0 and 1 Nm·s/rad. Raising leg Kp from 20 reduces static sag so the ±0.2 rad action interval has useful posture authority. Zero-action standing settles around 0.410 m in the nominal pyramidal model. The v2 height target is 0.415 m, close to that posture and the pilot's stable 0.414 m stand. Wheels are never position controlled.

Wheel-scale derivation: radius r=0.086 m; nominal half-track is 0.0465 + 0.0955 + 0.0481 = 0.1901 m. The training envelope's ideal outside wheel speed is `(1.10 + 1.4*0.1901)/0.086 = 15.89 rad/s`. Scale 18 supplies approximately 13% headroom. The navigation envelope needs about 14.39 rad/s under the same idealization. Skid steering adds slip, so this is a sizing calculation, not a no-slip tracking guarantee.

Observation contract (56 floats, body axes x forward/y left/z up):

| Slice | Quantity |
|---|---|
| 0:3 | simulator base linear velocity, body frame |
| 3:6 | base angular velocity, body frame |
| 6:9 | projected unit gravity |
| 9:12 | body-frame vx, vy, yaw-rate command |
| 12:24 | 12 leg position errors, FL/FR/RL/RR hip-thigh-calf order |
| 24:40 | 16 joint velocities in action order |
| 40:56 | previous clipped actor action, before optional actuator delay |

All configured observation scales are 1.0; observation clipping is ±100 before empirical normalization. Wheel angles are absent, not merely masked zero slots. The actor and critic each have their own empirical normalizer. Export `contract.json` records order, scales, nominal angles, rates, and bounds. A downstream adapter must reproduce this contract, use P/V modes and effort limits, and run at 50 Hz while holding each 10 Hz navigation command for five policy ticks. No navigation geometry or artificial response lag enters the actor.

## D. Commands

Training ranges: vx [-0.35, 1.10] m/s; vy [-0.30, 0.30] m/s; yaw [-1.4, 1.4] rad/s. Navigation ranges remain vx [-0.25, 1.00], vy [-0.25, 0.25], yaw [-1.25, 1.25].

| Family | Probability | Sampling |
|---|---:|---|
| Stand | 15% | exact zeros |
| Straight | 25% | vx only, forward and reverse |
| Arc | 25% | vx + yaw |
| Yaw dominant | 15% | yaw only |
| Precision | 10% | vx range multiplied by 0.18; yaw by 0.2; vy=0 |
| Lateral | 7% | vy only |
| Mixed | 3% | vx + vy + yaw |

In v2.1, intervals are sampled independently per environment and independently of command family: 70% use 25–50 policy ticks inclusive (0.5–1.0 s), and 30% use 75–150 ticks inclusive (1.5–3.0 s). Each mode is uniform over its integer tick range. These are probabilities per sampled segment, not fractions of elapsed time. Linear deadzone is a 0.01 m/s XY vector norm; yaw deadzone is 0.01 rad/s. Stand detection uses command norm <1e-6. Deadzones add a small number of extra zero commands beyond the explicit 15%. About 90% of commands have zero lateral demand. Independent successive family draws cover stops, reversals, yaw-sign changes, arcs, and precision transitions; evaluation also tests those transitions explicitly. Go2-W velocity curriculum is disabled. The generic optional curriculum is invoked on reset before clearing accumulated rewards, at most once per episode-length interval.

## E. Rewards

Coefficients below are the exact config values. As in the existing environments, every coefficient, including termination, is multiplied by policy dt=0.02. `only_positive_rewards=True` is retained: the nonterminal sum is clipped at zero, then termination is added. This preserves the established baseline; a two-iteration smoke is not evidence for changing the clipping strategy.

| Active term | Coefficient | Meaning |
|---|---:|---|
| tracking_lin_vel | 1.0 | exp(-vx error² / 0.25 - vy error² / 0.04) |
| tracking_ang_vel | 0.8 | exp(-yaw-rate squared error / 0.25) |
| lin_vel_z | -0.15 | squared vertical body velocity |
| ang_vel_xy | -0.12 | roll/pitch angular velocity squared |
| orientation | -1.2 | projected gravity XY squared |
| base_height | -8.0 | (height - 0.415 m)² |
| normalized_effort | -0.03 | mean squared normalized leg effort + mean squared normalized wheel effort |
| leg_acc | -2.5e-7 | summed squared leg acceleration |
| wheel_acc | -1e-7 | summed squared wheel acceleration |
| leg_action_rate | -0.01 | summed squared normalized leg action differences |
| wheel_action_rate | -0.005 | summed squared normalized wheel action differences |
| termination | -10.0 | falls/invalid base contact; timeouts excluded |
| dof_pos_limits | -2.0 | leg excursion into outer 10% of finite joint ranges |
| torque_limits | -0.5 | summed effort above 90% of actuator limits |
| stand_still | -0.5 | at zero command: XY velocity² + yaw rate² + 0.02*mean wheel velocity² |
| foot_swing_clearance | 0.08 | gait-allowance-weighted clearance, target 0.03 m, sigma 0.015 m |
| default_pose | -1.0 | mean squared leg position error |
| leg_motion | -0.02 | mean squared leg joint velocity |
| unnecessary_wheel_air | -0.25 | wheel air fraction times (1 - 0.75*gait allowance) |
| wheel_crossover | -2.0 | side clearance <0.045 m or left/right separation <0.14 m |
| collision | -0.5 | non-wheel ground-contact slots, capped at four |

The lateral gate ramps from zero at |vy|=0.03 to one at |vy|=0.10. The yaw gate ramps from zero at |yaw|=0.60 to one at |yaw|=1.10. Combined allowance is max(lateral gate, 0.80*yaw gate). Low/moderate yaw prefers skid steering; high yaw may use posture assistance or stepping. Pose and leg-velocity penalties retain 30% strength at full lateral demand and 44% at full yaw demand. Wheel-air penalties retain 25% and 40%, respectively. Clearance is weighted by the combined allowance. Non-wheel contacts use a vertical-force threshold of 8 N; base contact above this threshold terminates. Base height <0.33 m or roll/pitch >30 degrees also terminates.

Disabled: raw `torques`, generic `dof_vel`, generic `dof_acc`, generic `action_rate`, `dof_vel_limits`, `feet_air_time`, `feet_slide`, positive `wheel_contact`, `survive`, and `feet_stumble`. Removed helper logic: forward-drive gating, pose-hold fade, and overlapping positive wheel-contact reward. Wheel speed itself is penalized only at standstill. The effort term uses the public clipped instantaneous P/V control-force getter as a controller-effort surrogate, not as measured mechanical energy or hardware power. No additional actuator-strength DR is layered on top of gain uncertainty.

## F. Domain randomization and resets

| Quantity | Exact range / behavior |
|---|---|
| Effective wheel-ground sliding friction | uniform [0.6, 1.2], one sample shared across four wheels per environment |
| Rolling / torsional friction | 0, disabled; no DR |
| Base mass addition | uniform [-0.5, +1.5] kg; nominal total robot mass 19.523 kg |
| Base COM offset | each local axis uniform [-0.015, +0.015] m |
| Leg Kp | nominal 40 times uniform [0.9, 1.1], shared across joints per environment |
| Leg Kd / wheel Kv | nominal 1 times a separate uniform [0.9, 1.1], shared across joints per environment |
| Wheel Kp | exactly zero |
| Action delay | uniformly 0 or 1 policy ticks (0 or 20 ms), sampled on reset |
| Linear velocity noise | uniform ±0.05 m/s |
| Angular velocity noise | uniform ±0.08 rad/s |
| Projected gravity noise | uniform ±0.02 per component |
| Leg position noise | uniform ±0.01 rad |
| Leg velocity noise | uniform ±0.2 rad/s |
| Wheel velocity noise | uniform ±0.5 rad/s |
| Command / previous-action noise | none |
| Push force | magnitude uniform [20, 50] N, direction uniform in world XY |
| Push duration | 20–40 physics ticks inclusive, 0.10–0.20 s |
| Push onset interval | 150–300 policy ticks inclusive, 3–6 s |
| Push application point | current base COM; no added torque in baseline |
| Optional torque | configurable component range, baseline [0, 0] Nm |

Mass, COM and floor conditions are sampled at construction and retained across episodes; controller gains and latency are resampled on reset. Base mass changes leave the specified link inertia unchanged, representing moderate centrally located payload/model uncertainty. Other link COMs remain unchanged. Push state is per environment, reset on episode reset, and reapplied at every physics step for its duration, including across policy boundaries. Pushes are enabled from the first training rollout; no push curriculum is used.

For mass 19.523 kg, 20–50 N gives approximately 1.0–2.6 m/s² free-body acceleration and 0.10–0.51 m/s impulse velocity change over the chosen durations. Ground reaction and actuator forces modify the measured response. A deterministic 40 N, 0.15 s forward push gave approximately 0.335 m/s change and 0.028 m displacement in the physical smoke test. The test checks that the duration expires and that public controller forces still match P/V equations afterwards.

Reset perturbations: leg angles ±0.03 rad, all joint velocities ±0.05 rad/s, base roll/pitch ±0.02 rad and yaw ±0.05 rad, base linear/angular velocities ±0.03 m/s and ±0.03 rad/s. Spawn position is (0,0,0.45) m. These are small posture variations, not fallen-state recovery. Evaluation disables these variations and all stochastic DR/noise, applying only its prescribed deterministic pushes.

## G. PPO and policy

Actor and critic: separate MLPs with [512,256,128] hidden units, ELU, empirical observation normalization, 56 inputs. Actor: 16 Gaussian means, learned log-standard-deviation parameters initialized to log(0.55), clamped standard deviation [0.05,0.8]. Critic: one value. RSL-RL 5.5.1 Beta was inspected: it is bounded and exportable but doubles the output parameter head and changes exploration/optimization. Gaussian with explicit control/export bounds preserves the conventional locomotion baseline without adding a new optimization variable to this migration.

| PPO / runner setting | Value |
|---|---|
| Class | PPO / OnPolicyRunner |
| Clip / value-loss coefficient | 0.2 / 1.0 |
| Clipped value loss | true |
| Entropy coefficient | 0.005 |
| Learning epochs / minibatches | 5 / 8 |
| Initial learning rate | 0.0008 |
| Schedule / desired KL | adaptive / 0.01 |
| Gamma / lambda | 0.99 / 0.95 |
| Max gradient norm | 1.0 |
| Per-minibatch advantage normalization | false |
| RND / symmetry | none / none |
| Mixed precision / Torch compile | false / none |
| Parallel environments | 4096 default |
| Steps per environment per rollout | 48 |
| Rollout samples / minibatch samples at default size | 196608 / 24576 |
| Maximum iterations | 2000 |
| Save interval | 50 iterations, plus final save |
| Seed | 1, CLI overridable |
| Actor / critic observation group | policy / policy |
| Logger | W&B by inherited default, or `--logger tensorboard` |
| W&B project / experiment name | go2w-locomotion / go2w |
| Resume default / load_run / checkpoint | false / -1 / -1 |

No network enlargement, learned estimator, symmetry mapping, mixed precision, or Torch compilation was introduced. Symmetry would require a separately verified physical mirror transform, and there is no demonstrated need for it in the first baseline.

## H. URDF changes

Only the four wheel collision geometries changed. Visual meshes, inertias, masses, axes, and joint limits remain intact. Each collision is a cylinder of radius 0.086 m and width 0.052 m. The mesh bounds are approximately X/Z ±0.085964 m and left Y [0.022185,0.074000] m, with mirrored right Y. Collision centers are therefore Y=+0.0481 m on the left and -0.0481 m on the right. Rotation `(pi/2,0,0)` aligns canonical cylinder Z with the joint's Y axis. Axis sign is immaterial for a cylinder. Wheel joint axes remain `(0,1,0)` on all four wheels. At level wheel contact, wheel-link origin height is approximately 0.086 m.

The source mesh has rounded sidewalls/tread; a cylinder preserves its bounding envelope, not the exact curved tire contact patch. This approximation is intentional for simpler transferable collision geometry. The existing wheel COM at Y=±0.06 m and inertia tensor were not replaced with uniform-cylinder estimates.

## I. Validation and interpretation

The checks include three registered robot constructions/finite tensors, GPU batched rigid-scene stepping, controller mapping/bounds, friction persistence through selective reset, timed body-force response with P/V equations intact, PPO optimization, checkpoint save/load, deterministic inference, export parity, and all 28 evaluation cases. Existing supplied Go2/Dodo actor checkpoints load in RSL-RL 5.5.1. Their long-run performance after the simulator upgrade has not been re-established.

| Check | Result |
|---|---|
| Dependency consistency | `pip check` passes |
| Minimal GPU rigid scene | Four batched environments step successfully |
| Dodo | Four environments, 100 steps, finite 36-value observations/rewards/torques; clean exit |
| Go2 | Four environments, 100 steps, finite 48-value observations/rewards/torques; clean exit |
| Go2-W physics | Eight environments, 56 observations, correct 16-action mapping; all four wheels contact; clean exit |
| Effective friction | Shared across each robot's four wheels, varies between environments (observed 0.616–1.143); selective reset preserves it |
| Timed base push | 40 N for 0.15 s: delta vx 0.33465 m/s, displacement 0.02841 m; expires; P/V equations still match |
| PPO smoke | 64 environments × 48 rollout steps × 2 iterations; finite losses, checkpoint saved, exit code 0 |
| Play / viewer | Final checkpoint loads, exports, runs 30 viewer steps, and exits with code 0; earlier headless play also passes |
| Export parity | 256 random observations: maximum absolute error 0 against deterministic checkpoint inference; normalizer contains 6144 samples; actions bounded |
| Existing checkpoints | Supplied Dodo and Go2 actor state dictionaries load strictly in RSL-RL 5.5.1 |
| Evaluation | All 28 cases complete on final checkpoint, two environments, 25 steps per segment; JSON and 28 trace files saved; exit code 0 |
| Contract tests | Six tests pass: controller/bounds, command mixture/timers/yaw gate, push duration, observations, normalized export, response measurement |
| Static checks | Ruff F/E9 checks pass for changed Python code; `git diff --check` passes |

Final smoke checkpoint: `logs/go2w_smoke/final_2026-09-21_09-05-40/model_1.pt`. The value losses were 0.0154 and 0.0199. Export: `logs/go2w_smoke/exported/policies/policy_1.pt`, with `contract.json` including nominal gains, effort limits, and the source checkpoint. Evaluation: `.migration-audit/evaluation-final/metrics.json`. This is only an integration check. Logs are retained locally under `.migration-audit/`; no large run outputs are committed.

Genesis warns that the URDF's all-zero neutral qpos exceeds calf limits while building the scene. Every runtime reset sets the configured valid nominal bent-leg posture before stepping; all finite/contact/controller assertions pass. This warning is not suppressed. Play/evaluation use the current repository configuration, so retain the training run's saved `config.yaml` and reproduce its actuator/observation settings when loading older checkpoints.

Go2-W's observation contract changed from 60 to 56 and its exploration parameterization changed; old Go2-W checkpoints are not compatible with this new baseline. Use a new experiment directory. Evaluation writes per-case `.npz` traces and `metrics.json`, including RMSE, fall counts/fractions, roll/pitch RMS, height variance, undesired contact steps, wheel/torque saturation, action changes, wheel acceleration, leg motion, stopping behavior, and step-response crossings. Metrics describe the actor actually loaded. Failed episodes retain raw pre-reset traces but do not contaminate surviving-episode RMSE; fall fraction must always be considered alongside RMSE.

The 63.2% crossing is a descriptive response-time estimate, not a validated first-order fit. Rise/settling/crossing values can be null when a response does not converge or an episode falls. Use the full traces for subsequent navigation-proxy identification. Do not copy a time constant from the two-iteration smoke checkpoint into the navigation controller.

## J. Deferred work

- Full Go2-W training, quantitative checkpoint comparison, and navigation-loop integration.
- IsaacLab/PhysX actuator/observation adapter and cross-simulator validation; this repository does not contain the target navigation application.
- Velocity estimation, system identification, tire/contact and actuator calibration, hardware latency calibration, sim-to-real testing, ROS 2 deployment.
- Rough terrain, perception, stand-up recovery, symmetry augmentation, and alternative action distributions.

## K. Windows / Conda commands

Run from the repository root. `--no-capture-output` streams progress; the first build may compile kernels for several minutes.

Check the environment and GPU:

```powershell
conda run -n genesis-gpu python -m pip check
conda run -n genesis-gpu python -c "import sys, torch, importlib.metadata as m; print(sys.version); print('Torch', torch.__version__, 'CUDA', torch.version.cuda, 'available', torch.cuda.is_available()); print('Genesis', m.version('genesis-world'), 'RSL-RL', m.version('rsl-rl-lib'))"
conda run --no-capture-output -n genesis-gpu python -m unittest discover -s tests -v
conda run --no-capture-output -n genesis-gpu python -m robot_gym.scripts.smoke --task go2w --num_envs 8 --headless
```

Small headless training smoke:

```powershell
conda run --no-capture-output -n genesis-gpu python -m robot_gym.scripts.train --task go2w --num_envs 64 --max_iterations 2 --headless --logger tensorboard --experiment_name go2w_smoke --run_name smoke
```

Intended full training (not started automatically):

```powershell
conda run --no-capture-output -n genesis-gpu python -m robot_gym.scripts.train --task go2w --num_envs 4096 --max_iterations 2000 --seed 1 --headless --logger tensorboard --experiment_name go2w_flat --run_name baseline
```

Use `--logger wandb` after `wandb login` if desired. These commands keep all logging local by default.

Play the latest full-training checkpoint and export the bounded normalized actor:

```powershell
conda run --no-capture-output -n genesis-gpu python -m robot_gym.scripts.play --task go2w --experiment_name go2w_flat --load_run -1 --checkpoint -1 --num_envs 1 --steps 1000
```

Add `--headless` for no viewer. Replace `--load_run -1` with the exact run-directory name and `--checkpoint -1` with an iteration number to choose a checkpoint. For the smoke checkpoint, use `--experiment_name go2w_smoke`. Exports go to `logs/<experiment>/exported/policies/policy_1.pt` and `contract.json`.

Deterministic evaluation, with 3 seconds before and after each command transition:

```powershell
conda run --no-capture-output -n genesis-gpu python -m robot_gym.scripts.evaluate --task go2w --experiment_name go2w_flat --load_run -1 --checkpoint -1 --num_envs 8 --steps 150 --seed 1 --headless --output evaluation/go2w_flat
```

Longer `--steps` increases each segment duration. To validate the evaluator itself against the smoke checkpoint, use `--experiment_name go2w_smoke --steps 25`. That abbreviated run is not a locomotion-quality assessment. Dodo/Go2 regressions use the same smoke module with `--task dodo` or `--task go2`.
