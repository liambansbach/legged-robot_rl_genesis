# Go2-W v2.2: sagittal symmetry augmentation

The only learning change from v2.1 is native RSL-RL 5.5.1 mini-batch symmetry augmentation. Rewards, gait gates, command families/ranges/duration mixture, observations/noise, physics/URDF, DR/pushes, actuator settings, Gaussian distribution, network and PPO parameters remain unchanged. Dodo and Go2 retain `symmetry_cfg=None`.

The existing v2.1 checkpoint 600 evaluation confirms final-window vy of -0.230834 m/s for -0.25, versus -0.029059 m/s for +0.25. This experiment tests whether augmentation improves that learned asymmetry without further gait or reward tuning.

## Native extension configuration

```yaml
symmetry_cfg:
  data_augmentation_func: robot_gym.envs.go2w.go2w_symmetry:sagittal_augmentation
  use_data_augmentation: true
  use_mirror_loss: false
  mirror_loss_coeff: 0.0
```

The function accepts keyword arguments `env`, `obs`, `actions`; either data argument may be `None`. Each supplied batch becomes `[original B; mirrored B]`, with no input mutation. The current observation is a TensorDict containing only `policy: [B,56]`; both actor and critic use that group. Unexpected groups/layouts/joint sets fail explicitly.

Installed `rsl_rl/extensions/symmetry.py` resolves the callable and augments observations/actions, while repeating native PPO targets. It computes actor-mean consistency MSE on mirrored observations and detaches the result when `use_mirror_loss=False`. The PPO/logger path reports the raw value as `Loss/symmetry`, without a contribution to gradients. Enabling mirror loss later requires only the flag and an explicitly chosen coefficient. No inference-time mirroring or averaging is applied.

## Physical reflection and runtime mapping

Reflection is body/world Y -> -Y. Zero-based observation blocks are:

| Block | Indices | Output from input |
|---|---|---|
| Body linear velocity | 0–2 | `[vx, -vy, vz]` |
| Body angular velocity (axial) | 3–5 | `[-wx, wy, -wz]` |
| Projected gravity (polar) | 6–8 | `[gx, -gy, gz]` |
| Command | 9–11 | `[vx, -vy, -yaw]` |
| Leg position errors | 12–23 | Swap FL/FR and RL/RR; negate hip only |
| All joint velocities | 24–39 | Same side swap; negate hip only |
| Previous actions | 40–55 | Same side swap; negate hip only |

The code constructs permutations from `env.joint_names` and `env.leg_action_indices`, the same runtime orders used by `compute_observations` and the P/V controller. The tests also build observations through the actual environment method, verify URDF axes/default angles/action scales, and deliberately reorder joints.

Verified runtime actuator order is FL, FR, RL, RR, each containing hip, thigh, calf, foot (wheel). Each output selects the following zero-based input index:

```text
16 actions / velocities / previous actions:
permutation = [4,5,6,7, 0,1,2,3, 12,13,14,15, 8,9,10,11]
sign        = [-1,1,1,1, -1,1,1,1, -1,1,1,1, -1,1,1,1]

12 leg position errors (FL/FR/RL/RR hip, thigh, calf):
permutation = [3,4,5, 0,1,2, 9,10,11, 6,7,8]
sign        = [-1,1,1, -1,1,1, -1,1,1, -1,1,1]
```

Hip axes are +X, thigh/calf/wheel axes are +Y, with zero joint-origin rotation. Nominal left/right angles reflect consistently. All wheel actions use the same positive 18 rad/s scale and V control. Existing v2.1 forward evaluation at vx=0.5 has positive wheel velocities `[5.375,5.383,5.545,5.617]` rad/s and achieved vx=0.468 m/s, confirming the common forward sign. Wheel reflection therefore swaps sides without negation.

## Evaluation additions

All 28 existing cases/metrics/raw arrays remain. `mirror_pairs` adds seven final-window velocity comparisons: yaw ±0.4/±1.0/±1.25, vy ±0.10/±0.25, arcs vx=0.5 with yaw±0.8 and vx=1.0 with yaw±1.0. Each reports both source means and component-wise `abs(v_minus - [1,-1,-1]*v_plus)`, with m/s and rad/s identified separately. An all-fallen member produces null mismatch. No overall score mixes these units.

Three appended cases hold yaw +1.0, lateral +0.25, or lateral -0.25 before stopping: `yaw_to_stop`, `lateral_positive_to_stop`, `lateral_negative_to_stop`. As with existing transitions, `--steps 150` gives 3 seconds before removal and 3 seconds afterward. Raw body velocities, 12 leg errors and four contact flags are retained.

`pose_recovery` compares per-environment leg RMS and hip RMS against that environment's final one-second stand reference, each plus **0.05 rad**. Both must remain within threshold through the end, with at least **0.25 s** of recorded samples. Time is the first qualifying post-removal sample time. Null means a fall/invalid stand reference or no observed recovery within the horizon. Thresholds, eligible counts and recovered fraction are recorded. This measures return of error magnitudes toward the stand reference, not exact joint-by-joint pose equality or a new reward.

## Validation

CPU tests: `conda run --no-capture-output -n genesis-gpu python -m unittest discover -s tests -v`.

Real runner integration (64 environments, two PPO iterations; all ordinary Go2-W learning settings retained):

```powershell
$env:GO2W_GPU_TESTS = '1'
conda run --no-capture-output -n genesis-gpu python -m unittest discover -s tests -p test_go2w_symmetry_integration.py -v
Remove-Item Env:GO2W_GPU_TESTS
```

The GPU test checks runtime joint/DOF ordering, native callable resolution, 64->128 observation/action augmentation, two actual PPO updates, finite TensorBoard `Loss/symmetry` entries at steps 0 and 1, and checkpoint save. It is opt-in so ordinary CPU test discovery does not build a GPU simulation.

Validation on 2026-09-24:

- All 18 CPU tests pass, including arbitrary float32/float64 observation/action involution, unique-value signs/permutations, original-first 2B shapes, both `None` paths, nominal pose, runtime reordering, and evaluator diagnostics. The opt-in GPU test passes separately.
- Dodo, Go2 and Go2-W each pass the existing eight-environment construction/smoke regression with clean exit. Go2-W friction, reset, contact, mixed P/V, action saturation and external push assertions pass.
- The real 64-environment PPO runner completes two iterations. TensorBoard `Loss/symmetry` is **0.0154915452** at step 0 and **0.0104287835** at step 1, with augmentation enabled and mirror loss disabled. Checkpoint: `logs/go2w_v2_2_smoke/symmetry_2026-09-24_21-39-28/model_1.pt`.
- The extended evaluator replays v2.1 checkpoint 600 with eight environments and 3-second segments. All 31 cases complete; all 28 original cases have **exactly identical metrics and every raw array**. All new arrays have the expected shape and finite values. Results: `.migration-audit/evaluation-v2_2-v2_1-600/metrics.json`; verification: `.migration-audit/v2_2-eval-verification.txt`.
- Ruff F/E9 and `git diff --check` pass. Local logs: `.migration-audit/v2_2-unit.txt`, `v2_2-{dodo,go2,go2w}-smoke.txt`, and `v2_2-ppo-smoke.txt`.

The replay establishes a lateral-component mirror mismatch of **0.259893 m/s** for the ±0.25 pair. In each new stop case, all eight deterministic environments survive, but none returns inside both pose thresholds within the three-second recovery window (leg RMS threshold 0.145259 rad, hip RMS threshold 0.081341 rad). These identical deterministic replicas are not independent statistical trials. The null recovery times describe this finite window and threshold, not an inability to recover later. No pose/reward tuning follows from this diagnostic in v2.2.

The full 600-update training has not been started.

## Fresh 600-update study

From the repository root (no `--resume`):

```powershell
conda run --no-capture-output -n genesis-gpu python -m robot_gym.scripts.train --task go2w --num_envs 4096 --max_iterations 600 --seed 1 --headless --logger tensorboard --experiment_name go2w_flat_pilot_v2_2_symmetry --run_name augmentation_seed1
```

The unchanged save interval is 50. RSL-RL labels iterations from zero: the intermediate checkpoint is `model_300.pt` (301 completed updates), and the final checkpoint after exactly 600 updates is `model_599.pt`. This run does not produce `model_600.pt`.

```powershell
conda run --no-capture-output -n genesis-gpu python -m robot_gym.scripts.evaluate --task go2w --experiment_name go2w_flat_pilot_v2_2_symmetry --load_run -1 --checkpoint 300 --num_envs 8 --steps 150 --seed 1 --headless --logger tensorboard --output evaluation/go2w_v2_2_symmetry_300
conda run --no-capture-output -n genesis-gpu python -m robot_gym.scripts.evaluate --task go2w --experiment_name go2w_flat_pilot_v2_2_symmetry --load_run -1 --checkpoint 599 --num_envs 8 --steps 150 --seed 1 --headless --logger tensorboard --output evaluation/go2w_v2_2_symmetry_600_updates
```

`--load_run -1` selects the latest run within this new experiment; use its exact timestamped folder for later comparisons if more runs are added. Admit the policy based on correct-direction lateral tracking for both signs, reduced paired mismatch, preserved stand/straight/reverse/precision/braking and sustained yaw performance, and no meaningful increase in falls, non-wheel contacts or saturation. Stepping frequency is not a target. The smoke tests establish implementation correctness, not these learned-performance outcomes.
