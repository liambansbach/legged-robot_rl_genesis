# Go2-W v2: reward refinement after the 200-update pilot

This is a reward/evaluation patch on `testing`, followed by the v2.1 command-duration refinement below. Physics, P/V control, geometry, DR, pushes, observations, actions, Gaussian distribution, PPO/network settings, command values/family probabilities, episode length, and terrain are unchanged. No long training is started by this task.

## Pilot evidence

Checkpoint: `logs/go2w_flat_pilot/gaussian_seed1_2026-09-21_12-06-08/model_199.pt`.
Original evaluation: `evaluation/go2w_flat_pilot_gaussian/` (28 cases, eight environments, six seconds per case).
The TensorBoard event file and all original traces were inspected before changing rewards. The frozen checkpoint was then replayed with additional diagnostics under the original reward configuration into `evaluation/go2w_pilot_diagnostics_v1/`. These results describe v1, not a trained v2 policy.

TensorBoard first-20 / last-20 update means:

| Metric | First 20 | Last 20 |
|---|---:|---:|
| Episode length, policy steps | 484.67 | 999.29 |
| Weighted linear tracking | 0.3363 | 0.9191 |
| Weighted yaw tracking | 0.1631 | 0.5211 |
| Default-pose penalty | -0.00511 | -0.01256 |
| Base-height penalty | -0.00429 | -0.00484 |
| Leg-motion penalty | -0.00417 | -0.00507 |
| Wheel-air penalty | -0.01718 | -0.03447 |
| Undesired contact penalty | -0.00000067 | 0 |

The aggregate yaw score includes the numerous zero/small-yaw command segments; it does not establish sustained high-yaw performance. Commands lasted at most one second in the v1 pilot, whereas the fixed evaluations expose sustained behavior. The v2 reward patch initially retained that timing; the subsequent v2.1 change below adds sustained segments.

Yaw diagnosis (rad/s): “late” is seconds 3–6; “final window” is seconds 5–6; “last” is the final sample. All eight deterministic environments follow the same trajectory, so they are not independent trials.

| Command | First-second mean | Late mean | Late std | Final-window mean | Last sample | Correct sign in late window |
|---:|---:|---:|---:|---:|---:|---:|
| -1.25 | -1.262 | -0.224 | 1.204 | -0.070 | -1.237 | 64.7% |
| -1.00 | -1.008 | -0.179 | 1.046 | -0.315 | -1.049 | 64.7% |
| -0.40 | -0.455 | -0.212 | 0.098 | -0.228 | -0.153 | 93.3% |
| +0.40 | +0.393 | +0.013 | 0.027 | +0.008 | -0.007 | 61.3% |
| +1.00 | +0.920 | +0.260 | 0.784 | +0.250 | +1.062 | 64.0% |
| +1.25 | +1.140 | +0.520 | 0.896 | +0.397 | +0.592 | 72.0% |

Yaw initially approaches the correct target, then degrades. At ±1/±1.25 it oscillates through both signs. A correct final sample can therefore be misleading. At +0.4 it instead settles near zero after about two seconds. Negative and positive yaw are asymmetric. The -0.8/+0.8 arcs initially achieve approximately -0.846/+0.845, then average -0.126/-0.045 late, despite vx remaining approximately 0.51 m/s. None of these cases falls or makes undesired ground contact.

No sign/frame/controller bug was found. All four authored wheel axes are +Y; P/V indexing and positive scales are correct. Genesis returns world angular velocity, which the environment correctly rotates into the body frame. A numerical heading-rate check on the replay correlates above 0.9994 with measured body yaw rate in the four high-yaw cases (small roll/pitch explains some difference). Mean right-minus-left wheel speed has the commanded sign in all six pure-yaw cases.

Wheel order below is FL, FR, RL, RR. Actions are normalized; speeds are rad/s. Values are means over seconds 3–6.

| Command | Wheel actions | Wheel speeds | Right-minus-left speed |
|---:|---|---|---:|
| -1.25 | [0.106, 0.026, 0.172, -0.171] | [1.621, 0.560, 1.100, -1.161] | -1.661 |
| -1.00 | [0.082, -0.016, 0.169, -0.160] | [1.158, 0.075, 0.687, -1.036] | -1.403 |
| -0.40 | [0.032, 0.027, 0.383, -0.276] | [1.317, 0.574, 2.814, -2.470] | -3.013 |
| +0.40 | [0.038, 0.085, -0.108, 0.127] | [0.579, 0.788, 0.303, 0.600] | +0.253 |
| +1.00 | [0.081, 0.081, -0.162, 0.248] | [1.072, 1.785, -1.136, 2.562] | +2.206 |
| +1.25 | [0.123, 0.096, -0.209, 0.368] | [1.563, 2.324, -1.795, 4.527] | +3.542 |

For example, at +1 rad/s the RR action falls from a first-second mean of 0.754 to roughly 0.18–0.28 later. Tracking degradation accompanies changes in the policy's commands/posture, with little saturation; there is no evidence that a sign inversion or hard actuator limit explains it.

| Command | Per-wheel contact fractions | Any wheel below contact threshold | Leg speed RMS (rad/s) | Hip error RMS / maximum (rad) |
|---:|---|---:|---:|---|
| -1.25 | [0.747, 0.827, 0.960, 0.787] | 60.7% | 0.643 | 0.264 / 0.485 |
| -1.00 | [0.833, 0.813, 1.000, 0.820] | 44.7% | 0.517 | 0.260 / 0.433 |
| -0.40 | [0.987, 0.993, 0.853, 0.527] | 55.3% | 0.127 | 0.422 / 0.622 |
| +0.40 | [1.000, 1.000, 1.000, 1.000] | 0% | 0.058 | 0.341 / 0.451 |
| +1.00 | [0.827, 1.000, 0.820, 0.827] | 41.3% | 0.528 | 0.275 / 0.442 |
| +1.25 | [0.687, 0.993, 0.833, 0.693] | 58.7% | 0.580 | 0.254 / 0.429 |

High yaw already contains dynamic leg assistance and intermittent unloading/reloading. At +1, individual FL/RL/RR contact flags switch 18–22 times per environment in three seconds, while FR remains loaded; at -1 the RL wheel stays loaded and the others switch. This is oscillatory contact behavior, not evidence of a clean deliberate swing gait. The existing 8 N contact threshold detects unloading as well as actual loss of geometric contact, so flags alone cannot prove stepping. No prescribed gait is added.

Lateral commands remain largely ignored: the final-window vy is approximately -0.00009, -0.00132, -0.00660, -0.00166 m/s for commands -0.10, -0.25, +0.10, +0.25. At +0.25, vx is instead approximately +0.13 m/s and leg speed is only 0.011 rad/s.

The pilot stands at approximately 0.4136 m; nominal zero-action standing was approximately 0.410 m. During ±0.4 yaw, height drops to approximately 0.360 m while hip error becomes large. Thus the old 0.433 m target is above the natural stand, but these data do not prove it caused splay. Moving the target to 0.415 m removes unnecessary pressure to extend; stronger nominal-pose regularization addresses persistent deviation without introducing a stance controller.

## Exact v2 reward settings

With body-frame errors `ex = vx_command - vx` and `ey = vy_command - vy`:

`tracking_lin_vel = exp(-ex²/0.25 - ey²/0.04)`, coefficient **1.0**.

The denominators have units (m/s)²; they are not standard deviations. Longitudinal tolerance is unchanged. Ignoring vy=0.10 now gives 0.7788 rather than 0.9608; ignoring vy=0.25 gives 0.2096 rather than 0.7788. A 0.02 m/s lateral error still gives 0.9900 at zero longitudinal error. This provides an incentive for explicit lateral motion without giving centimeter-per-second noise a large cost.

Yaw tracking remains `exp(-yaw_error²/0.25)`, coefficient **0.8**, up from 0.6. This is a 33% increase, with maximum yaw reward still below the linear-tracking maximum of 1.0.

Define `ramp(x, lo, hi) = clamp((x-lo)/(hi-lo), 0, 1)`:

- `lateral_gate = ramp(abs(vy_command), 0.03, 0.10)`.
- `yaw_gate = ramp(abs(yaw_command), 0.60, 1.10)`.
- `gait_allowance = max(lateral_gate, 0.80*yaw_gate)`.

The continuous yaw ramp leaves ±0.4 and all commands up to ±0.6 fully regularized. At ±1.0, allowance is 0.64; at ±1.1 and above, it is 0.80. This allows assistance where the traces show high-rate oscillation, without automatically relaxing ordinary low-rate rolling.

| Term | Coefficient | Multiplier | Retained at full lateral / full yaw |
|---|---:|---|---|
| default_pose | **-1.0** | `1 - 0.7*gait_allowance` | 30% / 44% |
| leg_motion | -0.02 | `1 - 0.7*gait_allowance` | 30% / 44% |
| unnecessary_wheel_air | -0.25 | `1 - 0.75*gait_allowance` | 25% / 40% |
| foot_swing_clearance | +0.08 | `gait_allowance` | activation 100% / 80% |

Default-pose regularization is full strength at stand/straight/low yaw and partially relaxed for lateral/high yaw. Contact preference now retains a nonzero floor even during lateral stepping. Clearance remains the existing optional swing reward (3 cm target); it does not prescribe a gait. Effort, acceleration, action rate, joint limits, orientation, termination, and other rewards retain their original formulas and coefficients. Base-height target is **0.415 m**, with coefficient **-8.0**. Reward dt scaling and positive clipping are unchanged.

## Evaluator additions

Every case now includes a `diagnostics` object. The fixed final window is the last min(1 second, post-transition duration); it is explicitly not an automatic claim of stability. It reports mean achieved [vx, vy, yaw], RMSE, velocity std, last-sample velocity, mean base height, and signed error from the configured target. Post-transition metrics add leg-position RMS, hip-abduction RMS/max, body-frame wheel-link lateral positions/stance width, each wheel's contact fraction, counts/fractions for 0–4 simultaneous contacts, any-wheel-unloaded fraction, per-wheel mean actions/speeds, right-minus-left wheel speed, and leg speed RMS. These are available for every case, including yaw/arc tests. Existing falls, contacts, action/torque saturation, transient metrics, and the original 19 trace columns remain.

New `.npz` arrays retain the 12 leg errors, four contact flags, four wheel actions/velocities, four body-frame wheel-link positions, wheel-link world heights, and base roll/pitch/yaw. Hip indices are mapped from actuator order into the 12-leg observation order. Link origins are reported explicitly; their height is not a tire-clearance measurement for a tilted cylinder. Environments that fall are excluded from diagnostic aggregates, with their count reported and all pre-reset samples retained; all-fallen cases return null aggregates. No new observations or training-time diagnostic work is introduced.

## Validation and fresh pilot

| Check | Result |
|---|---|
| Unit/contract tests | All nine pass, including anisotropic tracking, continuous/symmetric gait allowance, retained penalties, hip-index metrics, failed episodes, P/V bounds, observations and export |
| Go2-W GPU smoke | Eight environments; finite tensors, all four nominal wheel contacts, friction/reset checks, action bounds and physical pushes pass; exit 0 |
| Known push | 40 N for 0.15 s: delta vx 0.33465 m/s; displacement 0.02841 m; normal P/V control intact |
| PPO smoke | 64 environments, two iterations; finite value losses 0.0172 / 0.0217; checkpoint saved; exit 0 |
| Original pilot replay | All 28 original 19-column traces reproduce exactly, maximum absolute difference 0 |
| Extended evaluator smoke | All 28 cases complete; new array dimensions and finite contents verified; JSON saved; exit 0 |
| Static checks | Ruff F/E9 and `git diff --check` pass |

Smoke checkpoint: `logs/go2w_v2_smoke/reward_v2_2026-09-24_13-18-55/model_1.pt`. Local test logs and evaluator smoke outputs are under `.migration-audit/v2-*.txt` and `.migration-audit/evaluation-v2-smoke/`. These checks validate the implementation, not whether v2 has learned better lateral/yaw behavior. That requires the fresh pilot and the same deterministic evaluation. The 400-update run was not started.

## v2.1 command durations

Go2-W overrides only `_reset_command_timer`. Whenever a new command is sampled, each selected environment independently chooses a duration mode: **70% short, uniform 0.5–1.0 s**, or **30% sustained, uniform 1.5–3.0 s**. Sampling uses integer policy ticks, inclusive: 25–50 or 75–150 at 50 Hz. The three config fields are `short_command_duration_range`, `sustained_command_duration_range`, and `sustained_command_probability`. Duration selection is independent of family/value selection; Dodo and Go2 keep the generic timer unchanged.

The percentages apply to sampled segments, not elapsed time. Expected sampled duration is 1.2 s. Longer segments expose sustained tracking while short segments remain the majority. Accepted v2 rewards/gait gates and evaluator diagnostics are unchanged. The inexpensive duration test samples 20,000 commands, checks both modes/ranges and the sustained fraction within 1.5 percentage points, checks stand/lateral prevalence in both modes, and checks selective timer resets.

v2.1 validation: all 10 unit/contract tests pass; the eight-environment Go2-W GPU smoke passes with clean exit; the 64-environment, two-iteration PPO smoke passes with finite value losses 0.0164 / 0.0225, checkpoint save, and exit code 0. Checkpoint: `logs/go2w_v2_1_smoke/mixed_duration_2026-09-24_17-38-17/model_1.pt`. Local logs are `.migration-audit/v2_1-{unit,gpu-smoke,ppo-smoke}.txt`. Ruff F/E9 and `git diff --check` pass. The 400-update pilot was not started.

From the repository root, start a fresh 400-update v2.1 run (no `--resume`):

```powershell
conda run --no-capture-output -n genesis-gpu python -m robot_gym.scripts.train --task go2w --num_envs 4096 --max_iterations 400 --seed 1 --headless --logger tensorboard --experiment_name go2w_flat_pilot_v2_1 --run_name gaussian_seed1
```

Then evaluate its final checkpoint under the same 3-second segments:

```powershell
conda run --no-capture-output -n genesis-gpu python -m robot_gym.scripts.evaluate --task go2w --experiment_name go2w_flat_pilot_v2_1 --load_run -1 --checkpoint 399 --num_envs 8 --steps 150 --seed 1 --headless --output evaluation/go2w_flat_pilot_v2_1
```

Compare survival, stand/straight/braking/push performance as well as lateral/yaw tracking and pose/contact diagnostics. The v2 observation/action/model contract remains checkpoint-compatible with v1, but the proposed pilot intentionally trains from scratch.
