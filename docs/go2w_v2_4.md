# Go2-W v2.4: yaw mobility and pose recovery

This study separates permission to move/unload the legs from permission to depart from nominal pose. No reward coefficient, PPO parameter, physics/control setting, observation, command exposure/timing, or symmetry configuration changes. The primary study starts from fresh actor/critic initialization.

## Diagnosis before editing

Inspected the original and extended v2.3 TensorBoard logs and the checkpoint 1598 deterministic traces. Over the last 100 updates of the extended run, weighted linear/yaw rewards average 0.93544/0.75798, versus 0.93195/0.75075 at the end of the original 800-update run. Pose, leg-motion and wheel-air terms average -0.01303, -0.00389 and -0.02615; swing clearance averages +0.00747. Mean action std rises from 0.5946 to 0.6836. This supports the requested objective refinement without changing PPO/entropy.

At yaw -0.4/+0.4, hip RMS is 0.2250/0.2169 rad and any-wheel-unloaded fraction is 42.7%/39.3%. At -1/+1, hip RMS is 0.1821/0.1835 rad and unloading is 84.0%/79.3%. Yaw-to-stop ends with hip RMS 0.14448 rad despite final yaw rate 0.02770 rad/s. These measurements motivate allowing moderate-yaw mobility while preserving nominal-pose pressure.

## Exact gates

Let `ramp(a,l,h) = clamp((a-l)/(h-l),0,1)`:

```text
L  = ramp(abs(vy_command), 0.03, 0.10)
Ym = ramp(abs(yaw_command), 0.30, 0.85)
Yp = ramp(abs(yaw_command), 0.60, 1.10)

mobility_gate        M = max(L, 0.80 * Ym)
pose_relaxation_gate P = max(L, 0.35 * Yp)
```

Old yaw mobility thresholds were **0.60 / 1.10 rad/s**. New thresholds are **0.30 / 0.85 rad/s**. The mobility weight stays 0.80. Pose retains the old activation thresholds but reduces its yaw weight from 0.80 to **0.35**. Thus moderate yaw gets mobility without extra pose relaxation, and fully active yaw retains **75.5%** of nominal-pose pressure. Lateral-only gates and their reward effects are identical to v2.3 across the whole lateral range.

| Reward | Coefficient (unchanged) | Gate application |
|---|---:|---|
| default_pose | -1.0 | `(1 - 0.7*P) * mean(leg_position_error²)` |
| leg_motion | -0.02 | `(1 - 0.7*M) * mean(leg_velocity²)` |
| unnecessary_wheel_air | -0.25 | `(1 - 0.75*M) * mean(not contact)` |
| foot_swing_clearance | +0.08 | Existing clearance kernel multiplied by `M` |

The Go2-W `_gait_gate` method remains only as the documented hook for Go2's inherited swing-clearance kernel; it returns `_mobility_gate`. Pose and leg/wheel penalties call their explicit helpers. No base Go2/Dodo reward behavior changes.

For pure commands (the other axis is zero):

| Command | Mobility M | Pose P | Default-pose retention | Leg-motion retention | Wheel-air retention |
|---|---:|---:|---:|---:|---:|
| yaw 0.4 | 0.145455 | 0 | **100%** | 89.818% | 89.091% |
| yaw 1.0 | 0.80 | 0.28 | **80.4%** | 44% | 40% |
| yaw 1.25 | 0.80 | 0.35 | **75.5%** | 44% | 40% |
| vy 0.25 | 1.0 | 1.0 | **30%** | 30% | 25% |
| stand / straight | 0 | 0 | **100%** | 100% | 100% |

Both signs behave identically. All gates remain continuous piecewise-linear ramps. The extra yaw pose pressure is moderate in absolute reward units: on the saved +1 yaw trajectory, its approximately 0.02767 leg-position MSE would contribute -0.02225 instead of -0.01528 before time scaling. Tracking priority remains unchanged. This calculation is not a prediction of the retrained policy.

Command probabilities remain `[stand .15, straight .20, arc .20, yaw .15, precision .10, lateral .13, mixed .07]`. Pure-lateral magnitudes remain uniform 0.10–0.30 m/s with balanced signs. Durations remain 70% 0.5–1.0 s and 30% 1.5–3.0 s. Lateral thresholds, height target 0.415 m, tracking/action-rate/effort rewards and 3 cm swing target are unchanged. Native symmetry augmentation remains on, mirror loss off, coefficient 0.0.

## Fixed-command play

The local active range overrides in `play.py` are replaced by `--command_vx`, `--command_vy` and `--command_yaw`. Supplying any one enables fixed mode, with omitted axes set to zero. Explicit zero therefore differs from omitting all command flags. The startup message prints all three requested components; the viewer receives the same `env.commands` tensor (its existing velocity arrows visualize linear velocity).

Generic `LeggedRobot.set_fixed_command` installs a finite three-component vector, disables periodic resampling and immediately rebuilds observations. Both generic and Go2-W samplers check it before any sampling/masking, including episode resets. Play configures it before its reset/first policy call. No training range edits are needed. Omitting all flags preserves ordinary play sampling; training does not activate the mode. `set_fixed_command(None)` restores the previous resampling-enabled setting.

These examples use the existing v2.3 checkpoint so they can be run immediately. After training v2.4, change the experiment/run/checkpoint to the desired v2.4 model:

```powershell
$playArgs = @('--task', 'go2w', '--experiment_name', 'go2w_flat_pilot_v2_3_lateral_exposure', '--load_run', 'go2w_flat_pilot_v2_3_lateral_exposure_2026-09-25_13-14-38', '--checkpoint', '1598', '--num_envs', '1', '--steps', '1000', '--logger', 'tensorboard')
conda run --no-capture-output -n genesis-gpu python -m robot_gym.scripts.play @playArgs --command_yaw 0.4
conda run --no-capture-output -n genesis-gpu python -m robot_gym.scripts.play @playArgs --command_yaw 1.0
conda run --no-capture-output -n genesis-gpu python -m robot_gym.scripts.play @playArgs --command_vy 0.25
conda run --no-capture-output -n genesis-gpu python -m robot_gym.scripts.play @playArgs --command_vx 0.5 --command_yaw 0.8
```

Use `--command_vx 0` for a fixed zero command. Add `--headless` for non-viewer execution. Existing play export behavior is retained.

## Evaluation

All 31 cases and existing metrics, mirror pairs, lateral diagnostics and yaw-to-stop recovery remain. Yaw cases (including yaw reverse and yaw-to-stop) gain `yaw_mobility`: per-wheel contact transition counts and airborne clearance mean/p90/max over the post-transition window, excluding fallen environments. These reuse the lateral diagnostic kernel. Clearance remains explicitly a nominal wheel-link-height proxy and false force-threshold contacts may mean unloading. Existing `diagnostics.post_transition` supplies hip/leg RMS and any-wheel-airborne fraction; `diagnostics.final_window` supplies velocity. No combined score is introduced.

## Validation

- All **23 CPU tests** pass, including gate continuity/sign symmetry, exact pure-lateral preservation and the four requested pose-retention values. Fixed-command tests exercise all six requested commands on Dodo, Go2 and Go2-W through actual sampler guards, selective reset, observation rebuilding and viewer command handoff; physics alone is stubbed in these CPU tests. Zero/omitted CLI axes, invalid inputs and disabling fixed mode are covered.
- Eight-environment Dodo, Go2 and Go2-W construction/smoke regressions pass with clean exits. Go2-W friction/contact/reset, P/V controls, action bounds and push checks pass.
- The real **64-environment, two-iteration PPO integration** passes with native augmentation on, mirror loss off, and TensorBoard `Loss/symmetry` **0.0136339394 / 0.0108394511**. The same live environment then passes all six fixed commands with full/selective resets and correct same-step policy observations.
- The reused integration test keeps its historical smoke experiment name. Checkpoint for this v2.4 validation: `logs/go2w_v2_2_smoke/symmetry_2026-09-25_16-15-36/model_1.pt`.
- Short deterministic evaluator smoke passes all 31 cases with two environments and 25 steps per command phase, using the existing v2.3 checkpoint 1598. All raw trace arrays are finite; eight yaw mobility summaries, seven mirror pairs and yaw-to-stop pose recovery are present. Recomputing all four archived lateral cases after extracting the shared helper reproduces their previous diagnostic dictionaries exactly. Output: `.migration-audit/evaluation-v2_4-smoke`; verification: `.migration-audit/v2_4-eval-verification.txt`. This checks evaluator plumbing, not sustained v2.4 performance.
- Ruff undefined/error checks and `git diff --check` pass.
- Logs are `.migration-audit/v2_4-unit.txt`, `v2_4-{dodo,go2,go2w}-smoke.txt`, `v2_4-ppo-smoke.txt` and `v2_4-diagnosis.txt`. Run CPU tests with `python -m unittest discover -s tests -v` in `genesis-gpu`; set `GO2W_GPU_TESTS=1` and select `test_go2w_symmetry_integration.py` for the opt-in GPU test.

## Fresh study

From the repository root:

```powershell
conda run --no-capture-output -n genesis-gpu python -m robot_gym.scripts.train --task go2w --num_envs 4096 --max_iterations 1000 --seed 1 --headless --logger tensorboard --experiment_name go2w_flat_pilot_v2_4_yaw_mobility --run_name augmentation_seed1
```

No resume flag: actor/critic start fresh. Save interval stays 50. Checkpoint labels are zero-based: evaluate 600 and 800 around the requested intermediate updates, and 999 after exactly 1000 updates:

```powershell
conda run --no-capture-output -n genesis-gpu python -m robot_gym.scripts.evaluate --task go2w --experiment_name go2w_flat_pilot_v2_4_yaw_mobility --load_run -1 --checkpoint 600 --num_envs 8 --steps 150 --seed 1 --headless --logger tensorboard --output evaluation/go2w_v2_4_yaw_600
conda run --no-capture-output -n genesis-gpu python -m robot_gym.scripts.evaluate --task go2w --experiment_name go2w_flat_pilot_v2_4_yaw_mobility --load_run -1 --checkpoint 800 --num_envs 8 --steps 150 --seed 1 --headless --logger tensorboard --output evaluation/go2w_v2_4_yaw_800
conda run --no-capture-output -n genesis-gpu python -m robot_gym.scripts.evaluate --task go2w --experiment_name go2w_flat_pilot_v2_4_yaw_mobility --load_run -1 --checkpoint 999 --num_envs 8 --steps 150 --seed 1 --headless --logger tensorboard --output evaluation/go2w_v2_4_yaw_999
```

Admission requires preserved velocity/stability performance, less sustained moderate-yaw hip deformation, and faster yaw-to-stop pose recovery. Useful unloading or low-clearance stepping is allowed; exaggerated stepping frequency is not a target. Smoke tests establish implementation correctness, not learned behavioral improvement. No long run has been started.
