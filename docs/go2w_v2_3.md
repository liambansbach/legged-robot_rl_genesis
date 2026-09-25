# Go2-W v2.3: lateral command exposure

The only learning changes from v2.2 are command-family probabilities and the pure-lateral magnitude distribution. All rewards/gait gates, symmetry settings, command timing, physics/URDF, DR/pushes, observations/noise, control settings, network/distribution and PPO settings remain unchanged. There is no curriculum or inference-time action modification.

## Evidence and range decision

The archived v2.2 checkpoint 599 traces confirm first-second vy of -0.18536/+0.18000 m/s for commands -0.25/+0.25, followed by final-second vy +0.00258/-0.00075 m/s. Recorded world-Y displacement is approximately -0.22735/+0.21466 m over the six-second case. The final height is 0.368/0.369 m, versus 0.419 m standing, and mean lateral stance width is approximately 0.348 m versus 0.263 m standing. These support testing sustained lateral exposure without changing gait allowance.

Keep the maximum at **0.30 m/s**, including pure lateral. The traces do not establish that ±0.50 would improve step discovery, and the current failure already occurs at ±0.25. Widening the range would additionally change command difficulty. With the unchanged lateral tracking kernel, zero lateral velocity yields only exp(-0.50²/0.04) ≈ 0.00193 at a 0.50 command, so a larger target does not automatically provide a better early learning signal. A wider range can be tested separately if later evidence supports it.

## Exact sampler

| Family | Probability |
|---|---:|
| Stand | 0.15 |
| Straight | 0.20 |
| Arc | 0.20 |
| Yaw | 0.15 |
| Precision | 0.10 |
| Pure lateral | 0.13 |
| Mixed | 0.07 |
| Total | 1.00 |

Probabilities are fractions of all sampled segments. Lateral-bearing families total **20%**; the existing deadzone can remove a negligible number of near-zero mixed commands.

`pure_lateral_magnitude_range = [0.10, 0.30]` m/s. Pure lateral sets vx=yaw=0 and chooses an equiprobable sign independently of a uniform magnitude in that range. The implementation reuses the existing uniform Y draw: for `u ~ Uniform[0,1)`, set `s = 2u-1`, `m = low + (high-low)*abs(s)`, and `vy = -m if s<0 else +m`. This preserves random draw count and leaves mixed commands on their existing uniform [-0.30,+0.30] range. Initialization rejects magnitude ranges that do not fit both signs of the configured command range.

Timing stays independent of family: **70% uniform 0.5–1.0 s**, **30% uniform 1.5–3.0 s** (inclusive integer ticks 25–50 / 75–150 at 50 Hz).

Symmetry remains exactly:

```yaml
symmetry_cfg:
  data_augmentation_func: robot_gym.envs.go2w.go2w_symmetry:sagittal_augmentation
  use_data_augmentation: true
  use_mirror_loss: false
  mirror_loss_coeff: 0.0
```

## Lateral diagnostics

The same 31 evaluator cases, existing metrics, mirror pairs and pose recovery remain. Each fixed vy ±0.10/±0.25 case gains a `lateral` object:

- Mean body vy in the first, second and final second of the full case, with actual window durations reported for short smoke traces.
- Mean signed net displacement along world Y, measured from the initial position to the final position. New `.npz` files additionally retain `initial_world_y` so the first policy tick is included. When processing older traces without it, the explicitly reported interval starts at the first recorded sample (`dt`). All existing raw arrays are preserved.
- Mean/p90/maximum of `max(0, wheel_link_z - contact_height)` over wheel samples with false contact flags, plus the sample count. This is **height above nominal contact height**, a proxy rather than the actual ground gap of a tilted wheel cylinder. False force-threshold contact flags can indicate unloading; no such samples produces null statistics.
- Contact-flag transition count per wheel, averaged across surviving environments, in the existing FL/FR/RL/RR order. These include both contact loss and regain; they are not automatically classified as steps.

Environments that fall anywhere in the case are excluded. All-fallen cases return null aggregates. The metrics describe transient displacement, height and contact changes without adding any reward or claiming that contact transitions alone prove deliberate stepping.

## Validation

CPU contracts cover exact probability normalization/settings, approximately 20% lateral exposure, balanced pure-lateral signs and uniform non-trivial magnitudes, configurable magnitudes, mixed full-range sampling, unchanged duration parameters/fractions and timing independence. Diagnostic tests cover window boundaries, signed displacement, clearance quantiles, contact transitions, no-airborne samples and fallen environments.

Validation commands:

```powershell
conda run --no-capture-output -n genesis-gpu python -m unittest discover -s tests -v
conda run --no-capture-output -n genesis-gpu python -m robot_gym.scripts.smoke --task dodo --num_envs 8 --headless --seed 1
conda run --no-capture-output -n genesis-gpu python -m robot_gym.scripts.smoke --task go2 --num_envs 8 --headless --seed 1
conda run --no-capture-output -n genesis-gpu python -m robot_gym.scripts.smoke --task go2w --num_envs 8 --headless --seed 1
$env:GO2W_GPU_TESTS = '1'
conda run --no-capture-output -n genesis-gpu python -m unittest discover -s tests -p test_go2w_symmetry_integration.py -v
Remove-Item Env:GO2W_GPU_TESTS
```

Results on 2026-09-25:

- All **20 CPU tests** pass; the opt-in GPU integration test passes separately.
- Dodo, Go2 and Go2-W each pass the eight-environment construction/smoke checks with clean exit. Go2-W contact, friction/reset, P/V control, action bounds and pushes pass; the known push produces delta vx 0.334654 m/s.
- The **64-environment, two-iteration PPO smoke** passes with value losses 0.0145/0.0177 and TensorBoard `Loss/symmetry` 0.0136168869/0.0110032512. Augmentation remains on and mirror loss off. The existing integration test retains its old log experiment name; this v2.3 validation checkpoint is `logs/go2w_v2_2_smoke/symmetry_2026-09-25_10-49-57/model_1.pt`.
- The four archived v2.2 lateral traces pass independent cross-checks for velocity windows and displacement. A short evaluator smoke (two environments, 25 steps per segment) completes all **31 cases**, preserving seven mirror pairs and writing four lateral summaries plus the initial-position metadata. Shapes and finite arrays pass; a missing second-second window correctly reports null in this one-second smoke.
- Ruff F/E9 and `git diff --check` pass. Local logs/results are under `.migration-audit/v2_3-*.txt`, `.migration-audit/v2_3-lateral-baseline.json`, and `.migration-audit/evaluation-v2_3-smoke/`.

## Fresh 800-update study

From the repository root, with fresh actor and critic (no `--resume`):

```powershell
conda run --no-capture-output -n genesis-gpu python -m robot_gym.scripts.train --task go2w --num_envs 4096 --max_iterations 800 --seed 1 --headless --logger tensorboard --experiment_name go2w_flat_pilot_v2_3_lateral_exposure --run_name augmentation_seed1
```

Save interval remains 50. Checkpoint labels are zero-based: models 400 and 600 are the requested intermediate evaluations; model 799 is the final checkpoint after 800 updates. Evaluate the latest run in this new experiment, or replace `--load_run -1` with its exact timestamped folder once available:

```powershell
conda run --no-capture-output -n genesis-gpu python -m robot_gym.scripts.evaluate --task go2w --experiment_name go2w_flat_pilot_v2_3_lateral_exposure --load_run -1 --checkpoint 400 --num_envs 8 --steps 150 --seed 1 --headless --logger tensorboard --output evaluation/go2w_v2_3_lateral_400
conda run --no-capture-output -n genesis-gpu python -m robot_gym.scripts.evaluate --task go2w --experiment_name go2w_flat_pilot_v2_3_lateral_exposure --load_run -1 --checkpoint 600 --num_envs 8 --steps 150 --seed 1 --headless --logger tensorboard --output evaluation/go2w_v2_3_lateral_600
conda run --no-capture-output -n genesis-gpu python -m robot_gym.scripts.evaluate --task go2w --experiment_name go2w_flat_pilot_v2_3_lateral_exposure --load_run -1 --checkpoint 799 --num_envs 8 --steps 150 --seed 1 --headless --logger tensorboard --output evaluation/go2w_v2_3_lateral_799
```

Admission requires sustained, correctly signed and comparable lateral velocities at ±0.25, with repeated foot repositioning instead of collapse into a static splayed stance. Preserve standing, forward/reverse tracking, braking, sustained yaw/arcs and push recovery without material increases in falls, non-wheel contacts or saturation. Implementation smoke tests do not establish those learned outcomes. If exposure alone fails, report it; further gait-specific regularization/shaping requires a separate study.

The long run has not been started.
