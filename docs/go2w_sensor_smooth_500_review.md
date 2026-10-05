# Go2-W sensor refinement reviews

Current decision (2026-10-05): **prepare navigation_rolling_control from zero-hold model_499.pt**, combining the prepared placement cost with conditional yaw precision; retain 250 as a straighter fallback. See [the combined preparation](#combined-rolling-control-preparation-2026-10-05). Earlier selections, placement-only proposals and cleanup results below are historical.

## Sensor smoothing: verified 500-update review

Retain **model_499.pt as the sensor-motion development parent** and prepare one 200-update phase-observation refinement. The final model materially improves rigid-body sensor motion while preserving both directions of lateral/yaw stepping. It does not improve all physical qualities: front splay and hip RMS increased, fast-forward stopping worsened, and some yaw errors/contact peaks increased. Do not qualify this result for deployment or IsaacLab transfer yet. Midpoint 250 offers narrower hips and better moderate-forward tracking/stopping, but final 499 has the strongest sensor improvement and retained clearance cycles.

## Checkpoints and evidence

- Original: `logs/go2w_transfer_v3/transfer_v3_seed1_2026-10-03_16-18-34/model_1499.pt` (1,500 completed updates).
- Actual refinement: `logs/go2w_transfer_v3_sensor_smooth/sensor_smooth_from1499_seed1_2026-10-04_10-42-42/`.
- Midpoint: `model_250.pt` (251 completed local updates; no 249 checkpoint).
- Final/selected: `model_499.pt` (500 completed local updates; cumulative lineage 2,000).
- Compact machine-readable review: `evaluation/sensor_smooth_500_review/summary.json`. Detailed `training_summary.json`, `physical_summary.json`, `nominal_summary.json`, `phase_dependency.json`, `matching.json`, original manifests, metrics and NPZ traces remain in that directory or their referenced evaluation directories. A shorter decision summary is saved there as `summary.md`.

Saved `config.yaml`, `preparation.json`, checkpoints, 498 diagnostic rows and TensorBoard events agree on the source and local update count. Checkpoint `infos` is missing (`None`), not fabricated metadata. Native selective initialization retained actor, critic, empirical normalizers and learned std, and reset optimizer/local iteration. Checkpoint optimizer steps reach 20,000 (40 minibatches/update); normalizer counts reach 524,288,000. Both observation groups have 58 entries; actions have 16. The original parent's adaptive final optimizer LR was approximately 0.0003844; refinement LR is fixed **0.00005** in configuration, every observed diagnostic and checkpoint.

The recipe actually changed episode duration 20 to 60 s; angular xy cost -0.05 to -0.075; base-z cost -0.2 to zero; added sensor world-z velocity cost -1; rolling-pose hip weighting 3 versus other legs 1 within coefficient -0.5; and sparse selected long/extended holds (4% 8–15 s, 1.5% 20–30 s for stand/straight/arc). No reference/plant change: each limb qref `(0, .70, -1.40, 0)`, reference/spawn heights .4277417/.4307417 m, P legs Kp 40/Kd 1, V wheels Kp 0/Kd 1, scales .3/.35/.4/18, clipping +/-1, wheel target limit 20. Nominal armature .01 kg m², training DR .005–.02 with delay 0–2 ticks; passive terms zero. Policy/physics .02/.005 s, decimation 4, one substep, same implicitfast solver. Gait .8 s, stance .65, apex .04 m, diagonal offsets unchanged. Gamma .995, GAE .95 and 512/256/128 ELU networks unchanged. Training observation noise is distinct from policy action std and reward kernel scales.

## Logging and observed learning diagnostics

`init_at_random_ep_len=True` randomizes initial episode lengths while environment reward sums and logger return/length accumulators start at zero. Episode reward terms divide by the full configured episode duration. At 60 s/.02 s = 3,000 ticks, a full episode spans 3,000/64 = 46.875 updates. TensorBoard return/length are 1.64/31 ticks at update 0, 87.20/1,627 at 25, 159.27/2,971 at 46, and 160.74/3,000 at 47. The first-to 47 timestamps span about 205 s. The simultaneous first-50-update ramp is strongly consistent with startup accounting; updates occur during that time, so it is not an exclusively logging explanation. Compare steady windows. `Train/mean_reward` is episode return: 160 over 60 s cannot be compared directly with 54 over 20 s.

Yaw is **`.8 * (1 - Huber((command_yaw - measured_yaw_rate)/.35))`**. Huber is .5e² inside unit magnitude and |e|-.5 outside; weighted rate ceiling .8. It is not Gaussian. A ~.764 weighted yaw rate is not a direct yaw RMSE or grounds for tightening its scale.

| Window | Observed/missing diagnostic rows | Mean episode return | PPO likelihood-ratio clipped fraction | Historical KL |
|---|---|---:|---:|---|
|75–125|50 / 125 missing|160.847|.09336|unavailable|
|225–275|51 / none|160.885|.09670|unavailable|
|400–499|100 / none|160.904|.09891|unavailable|

Iteration 317 is also missing. RSL-RL 5.5.1 calls native `get_kl_divergence` for its scheduler only with `schedule == 'adaptive'` and non-null `desired_kl`. Fixed-run KL arrays are empty, not zero, and parent's adaptive KL cannot substitute. `desired_kl` is not a hard trust-region bound. The added diagnostic measurement uses old/current distribution tensors from the existing pass, original sample prefix before symmetry augmentation, no stochastic actor call, gradients, scheduler changes or PPO fork. It supplies future values only.

Selected signed weighted rates (early/middle/final): sensor vertical -.00574/-.00523/-.00498; angular xy -.01195/-.01167/-.01144; rolling pose -.00255/-.00285/-.00294; phase clearance -.00794/-.00807/-.00832; phase support -.00595/-.00597/-.00608; effort -.00142/-.00147/-.00148; leg action -.00518/-.00512/-.00505. Tracking x/y/yaw remain approximately .987/.981/.764. Raw and signed weighted values for every term are in the JSON, without treating absent rows as zero.

Per-command time exposure (early/middle/final): stand .173/.170/.173; straight .164/.167/.166; arc .058/.058/.056; yaw .230/.229/.231; precision .052/.053/.052; lateral .231/.229/.230; mixed .092/.094/.092. Long-hold exposure is ~.077/.077/.078 and extended ~.058/.057/.058 of actual environment time, not draw probabilities or completed-episode counts. JSON includes seconds and all joint clipping/std-floor vectors. Final sampled clipping FL/FR/RL/RR, each hip/thigh/calf/wheel: [.0048,.0454,.0280,.0791] / [.0049,.0452,.0279,.0791] / [.0114,.0042,.1488,.0002] / [.0115,.0042,.1483,.0002]. Many leg std parameters occupy the .1 floor 80–100% of recorded updates; wheels remain above it. Likelihood-ratio clipping ~10% does not establish small or safe KL.

## Matched physical comparison

Parent nominal traces were reused from `evaluation/transfer_v3_main_ck1499_nominal`; full sustained traces from `evaluation/v3_main_sustained_review`. The other parent sensor_sustained directory was incomplete. Midpoint/final outputs are `ck250_nominal`, `ck499_nominal`, `ck250_sustained`, `ck499_sustained` under this review. All share seed 1, nominal plant, noise/DR off, deterministic actor, identical actual command arrays and latent phases. Matching proof checks resolved control/asset/init/sim/noise/DR/gait settings and sampling. No falls, resets, timeouts or nonfinite states in any candidate panel; terminal pre-reset state is therefore absent. Rollouts preserve pre-reset states and do not splice restarted episodes into holds.

Nominal stand is 10 s, moving 3 s zero +5 s command +6 s stop; sustained stand 30 s, moving 3+30+8 s. Early 1–5, middle 10–20 and late 20–30 s are relative to command onset. Means/RMSE below are full command (vx/vy in m/s, yaw in rad/s); off-command axes expose cross-axis motion. Detailed response times, early/middle/late values, roll/pitch means/rates, signed hips/RMS/p 95, load/contact/effort/clipping arrays, actual heights/clearances and limb repositioning are in `physical_summary.json`.

| Command (vx, vy, yaw) | Parent means; RMSE | +251 means; RMSE | +500 means; RMSE |
|---|---|---|---|
| stand 0.00/0.00/0.00 | -0.0016/-0.0003/0.0016; 0.0094/0.0043/0.0166 | 0.0031/0.0001/-0.0019; 0.0063/0.0040/0.0095 | -0.0032/0.0001/-0.0018; 0.0091/0.0029/0.0077 |
| forward 0.20/0.00/0.00 | 0.1830/-0.0002/0.0018; 0.0191/0.0061/0.0176 | 0.1894/0.0002/-0.0011; 0.0121/0.0040/0.0092 | 0.1871/0.0000/-0.0013; 0.0153/0.0040/0.0112 |
| forward_fast 0.50/0.00/0.00 | 0.4638/-0.0007/-0.0017; 0.0396/0.0062/0.0103 | 0.4732/0.0001/0.0007; 0.0303/0.0073/0.0167 | 0.4781/-0.0000/0.0013; 0.0266/0.0072/0.0159 |
| lateral_strong_positive 0.00/0.30/0.00 | -0.0028/0.2952/0.0036; 0.0183/0.0215/0.0456 | 0.0023/0.2966/0.0090; 0.0184/0.0174/0.0462 | 0.0067/0.2944/0.0039; 0.0224/0.0182/0.0452 |
| lateral_strong_negative 0.00/-0.30/0.00 | -0.0014/-0.2948/-0.0068; 0.0178/0.0216/0.0445 | 0.0039/-0.2965/-0.0096; 0.0189/0.0172/0.0459 | 0.0087/-0.2951/-0.0062; 0.0216/0.0179/0.0457 |
| yaw_strong_positive 0.00/0.00/0.80 | 0.0073/0.0009/0.7984; 0.0118/0.0076/0.0339 | 0.0072/-0.0007/0.7945; 0.0119/0.0069/0.0390 | 0.0028/0.0028/0.7972; 0.0104/0.0097/0.0455 |
| yaw_strong_negative 0.00/0.00/-0.80 | 0.0067/-0.0027/-0.8026; 0.0114/0.0079/0.0337 | 0.0075/0.0003/-0.7950; 0.0120/0.0066/0.0364 | -0.0005/-0.0053/-0.8004; 0.0093/0.0097/0.0409 |
| mixed 0.20/0.10/0.30 | 0.1990/0.0957/0.2989; 0.0128/0.0142/0.0226 | 0.2052/0.0975/0.2965; 0.0130/0.0104/0.0309 | 0.1960/0.0985/0.2989; 0.0143/0.0077/0.0312 |


Actual left-imager/radar **world-z velocity RMS**, m/s, not optical-axis or acceleration estimates. Stop displacement in meters uses the preceding command boundary through all 8 s of stopping, including its first tick. Sensor-window endpoint displacement uses a different sample boundary; both are labeled separately in JSON and never added.

| Command | Parent imager/radar | +251 imager/radar | +500 imager/radar | Parent/+251/+500 stop displacement |
|---|---:|---:|---:|---:|
| stand | 0.01663/0.01555 | 0.01206/0.01201 | 0.01126/0.01081 | n/a |
| forward | 0.01147/0.01035 | 0.00724/0.00697 | 0.00675/0.00588 | 0.0515/0.0649/0.0686 |
| forward_fast | 0.02159/0.01980 | 0.01066/0.01003 | 0.00796/0.00737 | 0.0316/0.0890/0.1150 |
| lateral_strong_positive | 0.06119/0.05507 | 0.06235/0.05668 | 0.05908/0.05569 | 0.0433/0.0417/0.0260 |
| lateral_strong_negative | 0.05584/0.05455 | 0.05181/0.05396 | 0.04937/0.05218 | 0.0712/0.0595/0.0501 |
| yaw_strong_positive | 0.07322/0.07333 | 0.03690/0.03854 | 0.03126/0.03400 | 0.0628/0.0391/0.0654 |
| yaw_strong_negative | 0.07368/0.07301 | 0.03726/0.03814 | 0.03365/0.03561 | 0.0796/0.0529/0.0501 |
| mixed | 0.08108/0.07735 | 0.05270/0.05201 | 0.04285/0.04384 | 0.0929/0.0637/0.0678 |


Final fast-forward imager RMS improves ~63% (.02159 to .00796), with faster travel (.464 to .478 for .5 command); stand improves ~32%, forward .2 ~41%, yaw +/- ~57%/~54%, mixed ~47%. Strong positive lateral is nearly flat at 200 Hz (.06234 to .06195), rather than a convincing smoothness gain. The previous 15% target is descriptive. Hip RMS and stopping targets are missed. Fast-stop endpoint displacement .0316/.0890/.1150 m and final 2 s planar RMS .00864/.00896/.01054 m/s worsen. Strong-yaw+ RMSE .0339/.0390/.0455 increases despite mean yaw being close to .8. Nominal yaw+/-.4 and lateral+/-.1 stepping remain, as do reverse and mixed locomotion; all 13 nominal cases completed.

| Command | Parent >=2 cm FL/FR/RL/RR | +251 | +500 |
|---|---|---|---|
| stand | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 |
| forward | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 |
| forward_fast | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 |
| lateral_strong_positive | 37/37/37/20 | 37/37/37/37 | 37/37/37/37 |
| lateral_strong_negative | 37/37/23/37 | 37/37/36/37 | 37/37/37/37 |
| yaw_strong_positive | 37/37/37/36 | 37/37/37/37 | 37/37/37/37 |
| yaw_strong_negative | 37/37/37/37 | 37/37/37/37 | 37/37/37/37 |
| mixed | 37/37/37/37 | 37/37/37/37 | 37/37/37/37 |


Event definition remains 6/10 N unload/reload hysteresis with 2 mm geometric lift and separately counted >=2 cm peaks. Censored boundary events are excluded from completed counts; windows are not summed into extra cycles. All candidates retain ~37 completed lift cycles per stepping limb over 30 s. Rear lateral 2 cm counts increase from 20/23 to 37 on final, while nominal low lateral/yaw retain 6 per wheel. Midpoint nominal strong yaw- has one 5-versus 6 threshold count; continuous peak-clearance distributions and censored events are saved, so a fixed threshold is not equated to gait loss.

Front stance widening is systematic outward splay, not primarily left/right asymmetry: fast-forward full signed FL/FR hips .094/-.106 -> .156/-.153 -> .175/-.174 rad. Front/rear track .445/.392 -> .483/.385 -> .497/.387 m; wheelbase .362 -> .343 -> .335 m. Final late front track .517 m, hip biases .202/-.204; late front-track drift 4.46 mm across 10 s. Final stand also settles outward (early/middle/late widths .413/.480/.491 m), with late drift 3.64 mm. A modest width change is not automatically unsafe, but these data do not support improved hip posture or no progressive settling. Actual base/camera mean heights remain near .4105/.5272 m in final fast rolling. Late camera height .52632 m, endpoint drift -.969 mm, fitted trend -.0394 mm/s and detrended oscillation RMS .308 mm are separate, overlapping descriptions; they are not additive.

Existing 200 Hz recording resolves fast-forward imager RMS .02195/.01061/.00786 (late .02158/.00633/.00401). Full transient wheel-load maximum increases from 100.5 to 122.6 N; positive-lateral maximum increases 499.7 to 542.1 N. P 95 loads, duty and peak arrays are recorded per wheel. Full fast-forward largest joint effort p 95 decreases 11.24 to 8.98 Nm at 50 Hz, while 200 Hz control peaks reach 20.84 Nm. Readbacks do not establish integrated impulses or hardware vibration. No extra seeds or long campaign were run; this is one matched seed, with run-to-run uncertainty unmeasured.

## Rolling phase diagnosis

The latent clock keeps advancing, raw sin/cos actor inputs rotate, and phase-conditioned targets are separate. Pure vx/stand demand is zero: desired swing/clearance do not periodically force a gait. Play normally skips reward accumulation; deleting a reward cannot instantly alter a fixed replay actor.

Using recorded physical states, one consistent joint fit of a linear trend and sin/cos harmonics 1–4 (1.25,2.5,3.75,5 Hz) shows final stand imager amplitudes .166/.486 mm at 1.25/2.5 Hz, block phase-lock stability .812/.964. Final fast rolling amplitudes .261/.182 mm, stability .947/.727. Joint harmonic explained variance is .357 stand/.478 forward for the imager and .154 forward for the base. These are 1–30 s fits, including settling; their fractions are not added to separately fitted harmonic metrics. Residual content includes nearby 1.2–1.3/2.45–2.55 Hz bands and nonstationarity. Leg actions also have coherent clock content, e.g. final fast hip fundamental amplitudes about .148/.151/.172/.180 raw action. See the recorded phase-analysis arrays for all axes/joints.

On 12 retained observations per stand/forward/fast case and candidate, varying only raw phase over 0/.25/.5/.75 before the saved normalizer changes final fast hip action with phase std .0749/.0759/.0787/.0823. RMS ranges correspond to ~.057–.066 rad of hip target variation. All 56 physical/history/command entries stayed fixed. This is a counterfactual dependency test, not a new valid trajectory or proof of sole causation; native same-input legacy parity is exactly 0.

One isolated paired diagnostic on final 499 zeroed raw phase at zero stepping demand, restoring the original clock inputs at positive demand. Latent phase and targets remained unchanged. Late imager RMS fell from .00604 to .00000161 m/s at stand, and .00388 to .0000377 at .5 forward; no falls/resets. Full forward sensor RMS changed little because transients dominate. The diagnostic worsened 8 s stop displacement .1150 to .1478 m and final 2 s planar RMS .01054 to .01370 m/s. A brief 3 s zero/6 s roll/5 s lateral/6 s roll/4 s stop preserved lateral mean .2937 m/s and actual .033–.039 m clearances, but showed braking lag (stop mean vx .0387). It is an untrained changed-input diagnostic, never published as a trained improvement. Previous-action feedback/dynamics can sustain motion without clock input.

## One prepared refinement, no updates executed

Route A is supported by the dependency and closed-loop evidence. `sensor_phase_conditioned` restores the explicit selected 499 recipe, then changes only `phase_observation_mode` from `unconditional` to `command_demand`. LR remains fixed 5e-5; all reward coefficients, phase support/clearance targets, motor/reference/plant, bounds, gamma/GAE, estimator inputs, std settings and network dimensions remain unchanged. The existing sensor objective/sampler is inherited once, without duplicate weighting or legacy sampling branches. Prior generation lineage is retained. Actor/critic/normalizers/learned std are loaded natively; optimizer and local counter reset.

The 58-entry contract changes: raw slots 56:58 become `demand(command)*[sin(2*pi*p), cos(2*pi*p)]` before the embedded empirical normalizer. The existing symmetric smooth demand is max of clipped ramps |vy| .01–.05 m/s and |yaw| .10–.25 rad/s, then `d*d*(3-2*d)`. High-demand stepping inputs are unchanged; zero demand gives raw `[0,0]`, not normalized zero. Retained normalizer maps that to approximately `[2.964e-6,-4.778e-6]`; removing unit-circle observations is still a distribution shift. Reflection p->p+.5 negates both clock entries while demand is invariant. Old checkpoints default to unconditional encoding. Export metadata explicitly records the mode and demand. No IsaacLab code was edited.

Saved proposed settings are `evaluation/sensor_smooth_500_review/next_refinement/config.yaml`; `preparation.json` proves exact native actor/critic/normalizer/std retention, empty new optimizer, local counter 0, LR 5e-5,58/58/16 and same-legacy-input parity 0. The model load uses an observation stub, not a training smoke. Budget 200 local updates, save interval 50; review `model_100.pt` (101 local updates), final `model_199.pt`. The midpoint review is a required human development review, not an automatic training pause. Reconsider the recipe if stopping, splay or either stepping direction regresses; do not treat a lower sensor RMS as sufficient.

Run these commands only after separately authorizing learning. No training or optimizer update ran in this task:

```powershell
$env:NUMBA_CACHE_DIR = Join-Path (Get-Location) '.cache/numba'
$env:GS_CACHE_FILE_PATH = Join-Path (Get-Location) '.cache/genesis'
$env:QD_OFFLINE_CACHE_FILE_PATH = Join-Path (Get-Location) '.cache/quadrants'
$env:PYTHONIOENCODING = 'utf-8'
$py = 'C:/Users/Liamb/anaconda3/envs/genesis-gpu/python.exe'
& $py -m robot_gym.scripts.train --task go2w --go2w_profile transfer_v3 --go2w_finetune sensor_phase_conditioned --load_run logs/go2w_transfer_v3_sensor_smooth/sensor_smooth_from1499_seed1_2026-10-04_10-42-42 --checkpoint 499 --experiment_name go2w_transfer_v3_sensor_phase_conditioned --run_name sensor_phase_conditioned_from499_seed1 --num_envs 4096 --max_iterations 200 --seed 1 --logger tensorboard --training_diagnostics --rl_device cuda:0 --headless
```

After that authorized run, resolve its actual timestamped output (no invented run directory), then evaluate midpoint/final with the same existing panels and fresh output directories:

```powershell
$phaseRun = (Get-ChildItem -LiteralPath logs/go2w_transfer_v3_sensor_phase_conditioned -Directory | Where-Object Name -Like 'sensor_phase_conditioned_from499_seed1_*' | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
& $py -m robot_gym.scripts.evaluate --task go2w --load_run $phaseRun --checkpoint 100 --eval_mode transfer_screen --output evaluation/sensor_phase_conditioned_midpoint_nominal --num_envs 1 --seed 1 --headless --rl_device cuda:0
& $py -m robot_gym.scripts.evaluate --task go2w --load_run $phaseRun --checkpoint 100 --eval_mode sensor_sustained --diagnostic_trace --output evaluation/sensor_phase_conditioned_midpoint_sustained --num_envs 1 --seed 1 --headless --rl_device cuda:0
& $py -m robot_gym.scripts.evaluate --task go2w --load_run $phaseRun --checkpoint 199 --eval_mode transfer_screen --output evaluation/sensor_phase_conditioned_final_nominal --num_envs 1 --seed 1 --headless --rl_device cuda:0
& $py -m robot_gym.scripts.evaluate --task go2w --load_run $phaseRun --checkpoint 199 --eval_mode sensor_sustained --diagnostic_trace --output evaluation/sensor_phase_conditioned_final_sustained --num_envs 1 --seed 1 --headless --rl_device cuda:0
```

The new named experiment creates timestamped outputs; existing models and evaluations remain intact. These are prepared commands, not actions taken here. Selected unchanged 499 replay/export:

```powershell
& $py -m robot_gym.scripts.play --task go2w --experiment_name go2w_transfer_v3_sensor_smooth --load_run sensor_smooth_from1499_seed1_2026-10-04_10-42-42 --checkpoint 499 --num_envs 1 --command_vx .5 --steps 2000 --episode_length_s 60 --rl_device cuda:0 --export
```

For a later separately authorized receiving simulator, match 16 joint order and P-leg/V-wheel controls, exact reference/scales/bounds,58 observation order/scales (body velocities at the authored base origin, wxyz orientation, previous clipped actions), the saved embedded normalizer, and explicit encoding mode. Preserve .02 s policy/.005 s physics boundary timing and advancing latent clock even at zero demand; transfer gait period/offsets and sign reflection consistently. Exporting legacy 499 must remain unconditional. A phase-conditioned future checkpoint requires the smooth demand before normalization. Matching these inputs and motor semantics is a receiving-side requirement, not evidence that Sim2Sim already works.

## Code and validation

Maintenance commit `f2498a8` moves 30 reward methods into one final `# Reward functions` block, active V3 first and legacy compactly labeled. Lifecycle, observations, resets, commands, termination, helpers and `compute_reward` stay above it. A one-off AST comparison found 66 method bodies/decorators unchanged, no duplicates or reward decorator dependencies. No permanent hash gate or old simulation campaign was added.

The separate diagnostics/recipe change adds fixed-schedule KL observation, the bounded inference override and opt-in saved phase encoding/selective refined-parent loading. Forty-six focused tests pass (sensor smoothing, V3, symmetry, contract and event refactor); no optimizer-running tests or new learning smoke were used. Native no-update loading/parity/shape/finite-observation checks pass. The broader `SagittalContinuationCPU.test_only_declared_continuation_difference` and `SufficientClearanceCPU.test_selector_and_contract` assertions encountered saved-config omissions (armature/inertia/capture fields); they are outside these changes and were not replaced by invented successful results. All trained artifacts and original evaluations were preserved. Work stays on `testing`; no packages upgraded, training started, optimizer updates executed or changes pushed.

## Phase-conditioned policy: actual 200-update review (2026-10-04)

**Keep checkpoint 199 as an intermediate navigation candidate.** It is the strongest new screen result and substantially quiets late rolling, but small lateral transitions lose physical lifts and fast stopping still trails original V3. Prepare one exposure change: pure lateral magnitude [.10,.30] -> [.02,.30] m/s, both signs. Defer freezing for diagnostic Sim2Sim until this deficit is reviewed. Keep original V3/1499 as the tracking/stopping fallback and retain sensor/250 and sensor/499.

The exact run is `logs/go2w_transfer_v3_sensor_phase_conditioned/sensor_phase_conditioned_from499_seed1_2026-10-04_16-11-11`. Its saved files are `model_0.pt`, `model_50.pt`, `model_100.pt`, `model_150.pt`, **`model_199.pt`**. Checkpoint labels 0-199 mean **200 completed local updates**, not 500; no model_200 exists. Adam steps reach 8,000; actor/critic normalizer counts reach 576,716,800. The saved preparation and TensorBoard tail agree. All 200 diagnostic rows exist; TensorBoard scalar rows 72 and 158 are missing, with diagnostics present at both. Checkpoint `infos=None` is unavailable metadata, not a missing update.

The explicit parent is `logs/go2w_transfer_v3_sensor_smooth/sensor_smooth_from1499_seed1_2026-10-04_10-42-42/model_499.pt`; its parent is `logs/go2w_transfer_v3/transfer_v3_seed1_2026-10-03_16-18-34/model_1499.pt`. Cumulative lineage is 2,200 updates. Native initialization retained actor, critic, normalizers and learned std; optimizer/local counter reset. The sole recipe change from sensor/499 was unconditional -> command_demand clock observations. Rewards, command sampling, 60 s episodes, gains, armature/delay/physics, action contract, randomization, networks and fixed LR 5e-5 match the saved sensor parent. No unexpected differences were found. Each reference was replayed with its own unconditional semantics.

Evidence: `evaluation/sensor_phase_200_review/comparison.json`, `training.json`, `matching.json`, `sensor_frames.json`, individual `*_analysis.json`, native manifests/metrics and retained NPZ traces. The JSON indexes reused reference directories and all newly created outputs; no old output was overwritten. Analysis scripts in this ignored evaluation directory are one-off postprocessing, not a new evaluator.

### Learning interpretation

The real logger starts return/length and reward accumulators without a preceding full episode while `init_at_random_ep_len=True` randomizes initial episode lengths. Terms divide by full 60 s duration. Thus 60/.02/64 = **46.875 updates** explains a strongly consistent startup ramp: update 0 return/length 1.648/31.10 ticks, 46 159.351/2,970.55, 47 160.961/2,999.79; first-to-47 timestamps span about 214 s. Optimization also occurs then, so accounting is not the exclusive explanation. Mean reward is episode return; 160/60 s cannot be compared directly with 54/20 s.

Yaw remains `.8 * (1 - Huber(yaw_error/.35))`, with .8 ceiling, quadratic inner and linear outer error cost. It is not the legacy Gaussian. Tracking scales (.25/.15/.35), learned policy std and uniform observation noise are different quantities. The ~.765 yaw contribution does not justify tightening the yaw scale.

| Steady window | Rows | Mean return | Measured KL mean/p95 | PPO ratio clip | Value loss | Transitions/s |
|---|---:|---:|---:|---:|---:|---:|
| 60-99 | 40 | 160.941 | 0.00372/0.00507 | 0.0977 | 0.0006813 | 57237 |
| 100-149 | 50 | 160.895 | 0.00369/0.00500 | 0.0967 | 0.0006909 | 57166 |
| 150-199 | 50 | 160.926 | 0.00375/0.00508 | 0.0988 | 0.0006989 | 59562 |

These are observed diagnostic KL values from old/current tensors in the existing minibatch pass, using the original sample prefix before symmetry augmentation. There is no extra stochastic actor call, optimizer/scheduler change or PPO fork. Installed RSL-RL 5.5.1 calls native scheduler KL only when schedule is adaptive; scheduler arrays here are empty, not zero. Historical sensor/499 KL remains unavailable. desired_kl=.01 is not a hard trust-region bound.

Final-window sampled clipping FL/FR/RL/RR (hip/thigh/calf/wheel) is [.0049,.0478,.0323,.0864] / [.0048,.0476,.0318,.0861] / [.0098,.0039,.1613,.00009] / [.0097,.0039,.1617,.00009]. Many hip/calf/rear-thigh std parameters occupy the .1 floor 76-100% of updates; front thighs/wheels do not. Final checkpoint std is approximately [.1,.115,.1,.138] front and [.1,.1,.1,.125] rear. Full per-joint clipping, deterministic saturation, scaled-target slew, floor occupancy and signed raw/weighted rates are saved in training.json; floor occupancy does not establish optimizer failure.

| Signed weighted rate | 60-99 | 100-149 | 150-199 |
|---|---:|---:|---:|
| sensor_vertical_velocity | -0.00485 | -0.00476 | -0.00481 |
| ang_vel_xy | -0.01140 | -0.01130 | -0.01136 |
| rolling_pose | -0.00326 | -0.00321 | -0.00335 |
| phase_clearance | -0.00749 | -0.00755 | -0.00730 |
| phase_support | -0.00588 | -0.00591 | -0.00586 |
| tracking_yaw | 0.76428 | 0.76454 | 0.76441 |

Final actual time exposure: stand 0.173, straight 0.166, arc 0.057, yaw 0.230, precision 0.053, lateral 0.229, mixed 0.093; selected long/extended holds 0.083/0.056. These are time fractions, not draw probabilities. Historical transition counts and conditioned partial-demand exposure were not logged and remain unavailable.

### Bounded physical comparison

New checkpoints 50 (51 updates), 150 (151) and 199 (200) each completed the existing 13-case transfer_screen. Only strongest 199 received the eight-case sustained panel. Both references reused raw nominal/sustained results after matching resolved physical/noise/DR settings, seed, actual commands, latent phases and sampling. The intentional actor phase-mode difference is preserved. Nominal stand is 10 s; moving schedule is 3 s zero +5 s command +6 s stop. Sustained stand is 30 s; moving schedule is 3+30+8 s. One additional 41 s transition case was run for both references and 199, with no observation override. No new seed campaign.

| New checkpoint | Fast stop endpoint/path (6 s), m | +.4 yaw stop endpoint, m | -.4 yaw stop endpoint, m |
|---|---:|---:|---:|
| ck50 | 0.0279/0.0852 | 0.0959 | 0.1389 |
| ck150 | 0.0672/0.1208 | 0.0466 | 0.0402 |
| ck199 | 0.0106/0.0645 | 0.0257 | 0.0145 |

Endpoint cancellation can hide motion; path length is retained. One 5-versus-6 >=2 cm wheel count on the short strong-yaw- screen is a threshold/boundary observation, not deleted gait: the full sustained test retains 37 >=2 cm cycles per limb.

Full-command results below use body vx/vy and body yaw rate; all-axis means/RMSE expose cross-axis motion. Original/sensor means and RMSE remain in the preceding historical tables and current JSON. New full-window values:

| Command | 199 mean vx/vy/yaw | 199 RMSE vx/vy/yaw |
|---|---:|---:|
| stand | -0.0068/0.0000/0.0001 | 0.0090/0.0000/0.0003 |
| forward | 0.1858/0.0000/0.0000 | 0.0158/0.0000/0.0002 |
| forward_fast | 0.4786/0.0000/0.0000 | 0.0266/0.0001/0.0004 |
| lateral_strong_positive | 0.0073/0.2969/0.0069 | 0.0165/0.0173/0.0459 |
| lateral_strong_negative | 0.0088/-0.2967/-0.0083 | 0.0170/0.0173/0.0470 |
| yaw_strong_positive | 0.0049/0.0018/0.7916 | 0.0111/0.0088/0.0418 |
| yaw_strong_negative | 0.0070/-0.0008/-0.7884 | 0.0121/0.0086/0.0414 |
| mixed | 0.2025/0.0961/0.2953 | 0.0109/0.0089/0.0385 |

| Command | Imager vz RMS original/sensor/199, m/s | Stop endpoint original/sensor/199, m | 199 stop path, m | 199 settling, s |
|---|---:|---:|---:|---:|
| stand | 0.01663/0.01126/0.00963 | n/a | n/a | n/a |
| forward | 0.01147/0.00675/0.00228 | 0.0515/0.0686/0.0150 | 0.0150 | 0.18 |
| forward_fast | 0.02159/0.00796/0.00765 | 0.0316/0.1150/0.0759 | 0.0760 | 0.22 |
| lateral_strong_positive | 0.06119/0.05908/0.05016 | 0.0433/0.0260/0.0777 | 0.0954 | 0.96 |
| lateral_strong_negative | 0.05584/0.04937/0.04419 | 0.0712/0.0501/0.0395 | 0.0503 | 1.76 |
| yaw_strong_positive | 0.07322/0.03126/0.02465 | 0.0628/0.0654/0.0100 | 0.0166 | 0.22 |
| yaw_strong_negative | 0.07368/0.03365/0.02470 | 0.0796/0.0501/0.0209 | 0.0633 | 0.44 |
| mixed | 0.08108/0.04285/0.03394 | 0.0929/0.0678/0.0233 | 0.0427 | 0.8 |

Settling is the first stop time after which planar speed stays <=.02 m/s and |yaw rate| <=.03 rad/s for the entire remainder, with at least 1 s remaining. It can coexist with displacement; it is not a certified stopping threshold. Immediate 0-.2 s response, first-second camera motion, early 1-5 s, middle 10-20 s, late 20-30 s and final 2 s RMS/displacement are separate in the JSON. Fast-stop final XY/yaw RMS for 199 is .00384 m/s/.00231 rad/s. Historical original fast endpoint .0316 and sensor .1150 m are protocol comparisons, not universal stopping distances.

199 improves most stops against sensor/499, but positive-lateral endpoint worsens .0260 -> .0777 m, fast stop still exceeds original .0316 -> .0759 m, and mixed yaw RMSE worsens .0312 -> .0385 rad/s (original .0226). Stand startup drift worsens: 30 s endpoint .0479/.0960/.2042 m for original/sensor/199; its late motion settles, so this is not steady rolling oscillation. An existing original fast-stop quarter-cycle trace gives .0337 m versus .0316 at default phase; no extra phase bank was needed for the clear partial-lateral deficit. Fast mean vx .4786 versus sensor .4781 and original .4638 shows that smoothness was not bought by slower fast travel. No falls, resets, nonfinite states or non-wheel contacts occurred in these nominal panels; no terminal failure state exists to report. Native recording retains terminal rows if failures occur and never splices restarted holds.

Full-demand lateral +/- .3 and yaw +/- .8 retain 37 >=2 cm cycles per wheel over 30 s, with additional shallow unload/reload events kept separate. The unchanged event definition uses 6/10 N hysteresis, 2 mm geometric lift, separate >=2 cm counts and censored boundaries. Positive-lateral physical peak cylinder clearances are .0339/.0362/.0367/.0288 m; completed-cycle medians include shallow unloads (.0315/.0358/.0334/.0231 m). Per-wheel duty, p95/peak loads, body-frame repositioning and clearance distributions remain in JSON. At 200 Hz, fast movement peak wheel load is 109.8 N versus sensor 122.6 N; Positive-lateral 200 Hz peak is 528.7 N (original 499.7, sensor 542.1), rather than the aliased 50 Hz peak 358.4 N. Deterministic action clipping reaches ~48% on a stepping joint; issued targets are bounded and readback effort-limit occupancy is zero at policy rate. This is not a hardware actuator assessment.

Front splay remains systematic: fast full front/rear widths .445/.392 original, .497/.387 sensor, .491/.380 m new; late new width .515/.380, front hips +.201/-.201 rad. Late front widening is 7.70 mm between the first/last one-second means within 20-30 s. Pair midpoints are only .051/.066 mm from center, with .030/-.005 mm late drift: this is symmetric front splay, not major left/right displacement. New mean base height .40848 m and roll/pitch -.00021/-.01709 rad stay bounded in this test. Width improvement is slight, not a solved posture problem, and small stable width changes alone are not a safety failure.

### Partial-demand transitions determine the next action

The sole added schedule is 3 s zero, 5 s vx .5, 3 s vy +.03, 4 s vy +.1, 4 s vx .5, 3 s yaw +.175, 4 s yaw -.175, 3 s vy -.03, 4 s vx .2, 8 s zero. Clock and saved encoding remain native. Selected partial lateral demand is .5.

| Transition | Original mean active axis; 2 mm cycles FL/FR/RL/RR | Sensor parent | 199 |
|---|---:|---:|---:|
| vy +.03 | 0.0254; 1/3/3/1 | 0.0256; 1/2/1/1 | 0.0181; 0/0/0/0 |
| vy -.03 | -0.0310; 2/2/1/2 | -0.0319; 2/2/2/2 | -0.0127; 0/0/0/0 |
| yaw +.175 | 0.1807; 3/3/4/2 | 0.1716; 3/4/4/3 | 0.1668; 2/0/0/3 |
| yaw -.175 | -0.1830; 4/5/4/4 | -0.1684; 5/5/5/5 | -0.1680; 0/4/4/0 |

199 partial-lateral cylinder peaks are only .18-1.54 mm across the two signs, below the 2 mm event lift criterion; this is not merely a 2 cm threshold loss. Full-demand +.1 still averages .0950 m/s and yields 4/5/5/4 >=2 cm cycles. Partial yaw tracks .1668/-.1680 versus +/-.175 requested, but lifts are distributed unevenly and all are below 2 cm. Roll-after-step cross motion and each transition camera window are retained in the JSON. These checks support a partial-demand deficiency associated with the changed clock contract and sparse matching exposure; they do not prove encoding is its sole cause.

Saved pure-lateral sampling is [.10,.30], so pure lateral [.02,.05] is missing even though mixed/precision contexts can contain partial demand. One targeted exposure change retains both signs, full-demand commands and existing mixture/duration/long-hold logic. The interval [.02,.05] will occupy about 10.7% of uniformly drawn pure-lateral magnitudes, not 10.7% of all training time. No hidden brake, action override, step-objective removal or hip lock is introduced. Legacy stand_still is inactive (coefficient zero); V3 already tracks zero vx/vy/yaw with its active Huber terms. Zero-command time exposure is ~17.3%, but actual command-to-zero transition counts were not recorded. Stopping and splay remain review metrics, not additional simultaneous training changes.

### Camera reference, sampling and robustness

The measured URDF fixed transforms from authored base_link use meters and x forward/y left/z up. front_realsense (left depth-origin proxy) is [.33881,.0475,.111], front_realsense_body (rear-face center) [.31736,0,.111], both identity rotation. Radar is [.28945,0,-.046825], rotation rows [-.96551223,0,.26035769], [0,1,0], [-.26035769,0,-.96551223]. Exact fixed-joint chains/matrices are saved in sensor_frames.json. Point world position is p_base+R_base*r and velocity R_base*(v_body+omega_body cross r), transporting the authored-origin velocity once. Vertical means world-up z, not sensor optical axis. Angular velocity is common to rigid frames; JSON records both body-frame axes and R_base*omega_body in common world axes. Base quaternion uses wxyz; reported roll/pitch use extrinsic xyz.

The active sensor reward still measures the left imager and its virtual sagittal mirror. Adding rear-face diagnostics does not alter it, fixed-link merging, masses, geometry, constraints or actor observations. Roll/pitch-rate -.075 and gravity-orientation -4 already price angular motion. A quiet rear-face point cannot establish rotation stability. Fast rear-face full RMS is .00750 m/s; late .0000679 (200 Hz .0000647); corresponding late imager .0000703 (200 Hz .0000671), versus sensor-parent imager ~.00401 at 200 Hz. Late rear-face mean height .52480 m, endpoint change -.539 mm, trend -.05395 mm/s and detrended height RMS .0163 mm describe overlapping properties separately and are not added. Late world angular RMS [.0000073,.000122,.0000848] rad/s confirms quiet axes in this particular roll window. Full-command transient motion remains. Rigid-body motion is not measured image quality, vibration or Sim2Sim performance.

Existing substep capture supplies 200 Hz (.005 s) on fast forward and positive lateral; other traces are 50 Hz (.02 s). The eight-condition existing precision_dr check ran 26 s per condition with saved uniform noise and 20-50 N, .1-.2 s pushes every 3-6 s, no torque. Conditions: nominal; friction .6/+90-degree heading; friction 1.2/-90; added base mass 1.5 kg/COM x+.015 m; gains .85; gains 1.15; delays1/2 policy ticks. All eight completed without falls/resets/non-wheel contacts/nonfinite state. Actual push peaks 33.8-48.0 N and active exposure .56-.92 s per condition were recorded. Policy-rate contact maximum reached 668.5 N in delay2, so no blanket robustness claim. The mass corner holds inertia nominal; training instead scales base inertia with mass. Armature stays .01 here. This fixed bank is not an IID test or a sweep of combined extremes. Increasing pushes is deferred: no measured failure currently supports combining that change with the partial-command repair.

### Transfer-relevant contract

Actor/critic 58, actions 16. Joint/action order is FL,FR,RL,RR, each hip/thigh/calf/foot_joint. Observation order: body linear velocity 3, body angular 3, projected gravity 3, command 3, leg q-qref 12, all joint dq 16, previous issued clipped action 16, clock 2. Physical observation scales are 1 and raw clipping 100; the saved empirical normalizer is inside the actor and must be carried once. Previous issued action precedes delay, so it differs from applied action during delayed steps. Simulator linear velocity is authored base-link-origin velocity rotated to body coordinates, not COM velocity; a later estimator must supply that origin/frame/timing. No estimator or camera input was added.

Per-limb reference is hip 0/thigh .7/calf -1.4/wheel 0. P legs Kp 40/Kd 1, V wheels Kp 0/Kd 1, action scales .3/.35/.4/18, clip +/-1, wheel target cap20 rad/s. Policy .02/physics .005 s, decimation 4, one substep, implicitfast; nominal armature .01 kg m2 and passive stiffness/damping/frictionloss zero. Clock period .8 s, stance.65, apex.04 m, offsets FL0/FR.5/RL.5/RR0. Latent p always advances after action/physics/reward at the old p. Training reset p uniform; deterministic replay reset 0 then one zero tick yields initial p=.025. A getter or command assignment does not advance p.

Selected mode is command_demand: clock=d*[sin(2*pi*p),cos(2*pi*p)], d=smoothstep(max(clamp((|vy|-.01)/.04),clamp((|yaw|-.10)/.15))). Multiplication occurs before the saved normalizer; zero raw clock becomes approximately [+4.18e-6,-5.11e-6] normalized. Reflection p->p+.5 negates clock, demand is sign invariant. Original1499/sensor499 retain unconditional mode. A receiver must check **58 versus legacy56**, saved mode, exact input order/scales, embedded normalization, frame/origin, issued-action history, phase reset/boundary timing, command signs, joint order, reference/limits and P-leg/V-wheel gains/armature before a separately authorized diagnostic transfer. This is not hardware-ready.

Active saved training DR: base added mass -.5..1.5 kg with proportional base inertia scaling; COM each axis +/-.015 m; wheel friction ratios .6..1.2 with effective max-rule contact combination (ground .1/wheel1 geometry); gains .85..1.15; symmetric leg/wheel armature groups .005..02 sampled at construction, held across resets; delay 0..2 ticks; pushes/noise as above. Uniform noise amplitudes: body linear .05, body angular .08, leg q .01, leg dq .2/wheel dq .5, gravity .02; commands/history/clock noiseless. Training preparation armature readback spans .005001..019994. Nominal and stress JSON readbacks include mass/inertia, COM, friction, gains, armature/passive terms and actual push force.

The actual nominal URDF totals 19.68371 kg, merged base 7.08371 kg, COM [.0280210,-.000002483,-.0027885] m. Base 6.921 plus two .001 head links and camera base .01103/two flanges .00821/mount .01254/body .12072 are included; radar and IMU mass 0. No guessed payload mass was added. Runtime base mass 7.083710 matches; principal solver inertia diagonal [.121757,.114581,.026547] requires its frame transform before comparison with URDF tensors. Real payload/mount stiffness and unmodeled hardware remain assumptions.

### One prepared exposure refinement and validation

`navigation_partial_lateral` restores the explicitly selected full 199 training checkpoint and applies only commands.pure_lateral_magnitude_range [.10,.30] -> [.02,.30]. Keep phase encoding, all rewards, fixed LR 5e-5, learned std/normalizers, physics/gains/reference, resets, gamma/GAE, networks and pushes unchanged. It is selective initialization with a fresh optimizer/local counter, not true resume. Previous-generation lineage is retained. The new output is go2w_transfer_v3_navigation_partial_lateral, bounded initial budget 150, save every 50, review 50/100 and final 149. `evaluation/sensor_phase_200_review/next_partial_lateral_refinement/config.yaml` and preparation.json prove exact native retention, empty optimizer/counter 0 and unchanged observation contract. The earlier unrun stopping-cost draft is marked superseded and retained locally as design provenance; it is not a second proposal.

The authorized execution smoke used 64 environments for **two local updates (80 Adam minibatch steps, 8,192 transitions)** in `logs/go2w_transfer_v3_navigation_partial_lateral_smoke/navigation_partial_lateral_from199_seed1_smoke_2026-10-04_19-14-13`, final model_1. All model tensors are finite; normalizer count 576,724,992 and LR 5e-5 confirm retained state then native updates. Value losses .000247/.000431; measured KL mean .0171/p95 .0241/max .0279 and PPO clipping .298 are higher in this small batch than the 4096-env training run, not silently reported as small or safe. No production run started and no efficacy claim follows from this smoke.

Fifty focused tests passed: 49 sensor/V3/symmetry/contract/event-refactor/terminal-alignment checks and one shared replay check covering Dodo, Go2, Go2W. The broader exploratory event suite still encounters the two previously documented old continuation-fixture omissions (armature/inertia/capture and default phase fields); they were not rewritten or called successful. One new optional push-readback fixture compatibility error was fixed and its terminal-alignment test passes. Changes stay in Go2W; the owner play.py modification remains untouched. Prior behavior-preserving reward-organization commit f2498a8 remains separate. No packages upgraded, IsaacLab edits, artifact deletion or push occurred.

PowerShell setup and executable commands reproducing the recorded inference runs (original CLI arguments are retained in each manifest; reruns require fresh output directories because existing outputs are protected):

```powershell
$env:NUMBA_CACHE_DIR = Join-Path (Get-Location) '.cache/numba'
$env:GS_CACHE_FILE_PATH = Join-Path (Get-Location) '.cache/genesis'
$env:QD_OFFLINE_CACHE_FILE_PATH = Join-Path (Get-Location) '.cache/quadrants'
$env:PYTHONIOENCODING = 'utf-8'
$py = 'C:/Users/Liamb/anaconda3/envs/genesis-gpu/python.exe'
$phaseRun = 'logs/go2w_transfer_v3_sensor_phase_conditioned/sensor_phase_conditioned_from499_seed1_2026-10-04_16-11-11'
$sensorRun = 'logs/go2w_transfer_v3_sensor_smooth/sensor_smooth_from1499_seed1_2026-10-04_10-42-42'
$originalRun = 'logs/go2w_transfer_v3/transfer_v3_seed1_2026-10-03_16-18-34'
foreach ($ck in @(50,150,199)) {
    & $py -m robot_gym.scripts.evaluate --task go2w --load_run $phaseRun --checkpoint $ck --eval_mode transfer_screen --output "evaluation/sensor_phase_200_review/ck${ck}_nominal" --num_envs 1 --seed 1 --headless --rl_device cuda:0
}
& $py -m robot_gym.scripts.evaluate --task go2w --load_run $phaseRun --checkpoint 199 --eval_mode sensor_sustained --diagnostic_trace --output evaluation/sensor_phase_200_review/ck199_sustained --num_envs 1 --seed 1 --headless --rl_device cuda:0
foreach ($item in @(@($originalRun,1499,"original_transition"),@($sensorRun,499,"sensor_transition"),@($phaseRun,199,"ck199_transition"))) {
    & $py -m robot_gym.scripts.evaluate --task go2w --load_run $item[0] --checkpoint $item[1] --eval_mode sensor_sustained --eval_phase_transition --output "evaluation/sensor_phase_200_review/$($item[2])" --num_envs 1 --seed 1 --headless --rl_device cuda:0
}
& $py -m robot_gym.scripts.evaluate --task go2w --load_run $phaseRun --checkpoint 199 --eval_mode precision_dr --eval_noise_pushes --output evaluation/sensor_phase_200_review/ck199_robustness --num_envs 8 --seed 1 --headless --rl_device cuda:0
```

The following production command is **prepared, not run**. It needs separate learning authorization; the two-update smoke used this same command with experiment suffix _smoke, run suffix _smoke, num_envs64 and max_iterations2.

```powershell
& $py -m robot_gym.scripts.train --task go2w --go2w_profile transfer_v3 --go2w_finetune navigation_partial_lateral --load_run $phaseRun --checkpoint 199 --experiment_name go2w_transfer_v3_navigation_partial_lateral --run_name navigation_partial_lateral_from199_seed1 --num_envs 4096 --max_iterations 150 --seed 1 --logger tensorboard --training_diagnostics --rl_device cuda:0 --headless
```

After an authorized run, select its actual printed directory. This selector refuses ambiguity instead of choosing an unrelated latest run. Review checkpoint50 and final 149 with the same panels, especially partial lifts, stopping/path, full-demand steps and splay; model100 is also saved.

```powershell
$refinedRuns = @(Get-ChildItem -LiteralPath logs/go2w_transfer_v3_navigation_partial_lateral -Directory | Where-Object Name -Like 'navigation_partial_lateral_from199_seed1_*')
if ($refinedRuns.Count -ne 1) { throw 'Select the exact directory printed by training.' }
$refinedRun = $refinedRuns[0].FullName
foreach ($ck in @(50,149)) {
    & $py -m robot_gym.scripts.evaluate --task go2w --load_run $refinedRun --checkpoint $ck --eval_mode transfer_screen --output "evaluation/navigation_partial_lateral_ck${ck}_screen" --num_envs 1 --seed 1 --headless --rl_device cuda:0
    & $py -m robot_gym.scripts.evaluate --task go2w --load_run $refinedRun --checkpoint $ck --eval_mode sensor_sustained --eval_phase_transition --output "evaluation/navigation_partial_lateral_ck${ck}_transition" --num_envs 1 --seed 1 --headless --rl_device cuda:0
}
& $py -m robot_gym.scripts.evaluate --task go2w --load_run $refinedRun --checkpoint 149 --eval_mode sensor_sustained --diagnostic_trace --output evaluation/navigation_partial_lateral_ck149_sustained --num_envs 1 --seed 1 --headless --rl_device cuda:0
# Replay/export the retained, unmodified intermediate:
& $py -m robot_gym.scripts.play --task go2w --load_run $phaseRun --checkpoint 199 --num_envs 1 --command_vx .5 --steps 2000 --episode_length_s 60 --rl_device cuda:0 --export
```

### Local storage cleanup dry-run

`evaluation/sensor_phase_200_review/storage_manifest.json` contains exact keep/archive/delete-candidate paths and companion provenance paths for archive copies. Enumeration measured local file sizes without following descendant links, traversing external directories or hydrating Synology placeholders; 605 entries were skipped and excluded from savings. GetCompressedFileSizeW provides a local compressed-size estimate, alongside logical bytes; it is not a promise of NTFS free-space gain. Installed environment caches outside this repository were not measured.

| Local artifact class | Files | MiB |
|---|---:|---:|
| small_provenance_results | 390 | 11.06 |
| large_results_and_other | 174 | 70.15 |
| generated_export | 52 | 13.98 |
| events_and_diagnostics | 82 | 146.15 |
| checkpoint | 251 | 1124.15 |
| research_archive | 9 | 155.09 |
| trace | 1121 | 494.24 |
| cache | 962 | 124.61 |

Measured scopes (MiB): logs/ 1278.79, evaluation/ 725.19, .cache/ 122.49, .ruff_cache 0.10, ressources 10.96; additional Python caches are listed individually. Generated exports are a separate protected artifact class in the manifest.

| Exact manifest set | Files | Estimated local MiB | Action now |
|---|---:|---:|---|
| keep | 1980 | 1566.98 | Retain; dry-run only |
| archive_candidates | 100 | 447.88 | Retain; dry-run only |
| delete_candidates | 961 | 124.59 | Retain; dry-run only |

Keep original 1499, sensor 250/499, all five phase-run checkpoints, every current comparison/fallback and smoke artifact, published bundles, documented historical labels, every run-final checkpoint, configurations/preparation/lineage, TensorBoard/diagnostics, reported results/raw traces and generated exports. The 100 explicit archive candidates are undocumented intermediates: copy with their listed provenance companions and verify externally before any separately reviewed local removal. Potential local reduction is447.88 MiB after such an archive; it is not a research-artifact deletion recommendation. Only 961 explicit compiler/Python-cache paths are delete candidates (~124.59 MiB), after all jobs stop and separate review. No generic age rule, git clean -fdx, directory-wide deletion or deletion command was used.

No obsolete source-code deletion is supported. transfer_v3 still uses transfer_v1/event_step_v1 profile builders, legacy replay methods and their tests; names alone do not make them obsolete. A later cleanup must use a reviewed explicit file manifest, recheck attributes and repository containment, and preserve cloud placeholders. This task deleted nothing.

## Navigation partial lateral: 200-update review (2026-10-05)

**Keep the final checkpoint as an intermediate and the parent as a fallback. Prepare one zero-command holding objective before qualifying navigation transfer.** Small lateral commands now produce physical lifts and better negative-direction response, and strong stepping is retained. Late stand/stop creep and cross-axis motion remain functional deficiencies. A width-only objective is deferred because it would not directly address these errors. No checkpoint comparison demonstrates ongoing same-recipe physical progress. No midpoint was needed for this intermediate/new-objective decision; an untested midpoint may offer a different trade-off.

Selected checkpoint: `logs/go2w_transfer_v3_navigation_partial_lateral/navigation_partial_lateral_from199_seed1_2026-10-04_23-48-18/model_199.pt`.

Parent/fallback: `logs/go2w_transfer_v3_sensor_phase_conditioned/sensor_phase_conditioned_from499_seed1_2026-10-04_16-11-11/model_199.pt`. Original reference remains `logs/go2w_transfer_v3/transfer_v3_seed1_2026-10-03_16-18-34/model_1499.pt`.

Local machine-readable decision: [summary.json](../evaluation/navigation_partial_200_review/summary.json). Training/provenance, native metrics/manifests/NPZ, comparison analysis, load proofs and proposed settings remain in `evaluation/navigation_partial_200_review/`. Large artifacts are ignored by Git.

### Verified recipe and learning evidence

There is one non-smoke run in the named experiment. Saved files are `model_0.pt`, `model_50.pt`, `model_100.pt`, `model_150.pt`, `model_199.pt`. Preparation reports completion of 200 updates; final iteration is 199 and Adam step is 8,000 (40 minibatches/update). Both empirical normalizer counts are 629,145,600. Shape is 58 actor inputs / 58 critic inputs / 16 actions. Checkpoint `infos` is absent (`None`). Source is the explicit phase-parent 199 above; cumulative lineage is 2,400 updates.

The sole behavioral delta is `commands.pure_lateral_magnitude_range: [.10,.30] -> [.02,.30]`. Rewards, `command_demand` phase encoding, references, control, simulation, observation noise, DR, networks and PPO settings equal the saved phase recipe. LR is fixed 5e-5, entropy .003, learned log-std floor .1. DR still includes 20–50 N pushes, delay 0–2 ticks, nominal armature .01 with randomized .005–.02, mass/COM/friction/gain randomization. No larger push range was introduced. Each candidate was replayed with its own saved semantics; original V3 remains unconditional.

TensorBoard contains all 200 steps 0–199; JSONL contains 199 rows, with **diagnostic row 103 missing**, not a missing update. Random initial episode lengths and empty reward accumulators explain the approximately 47-update accounting ramp: 60/.02/64 = 46.875. Return/length rise from 1.65/31 ticks at 0 to 159.60/2,971 at 46 and 161.12/3,000 at 47, across about 222 seconds. Learning also occurs during this period; startup accounting is not the sole possible cause of every early change. Compare later windows, and distinguish episode return from reward rate and from differently sized episodes.

| Local window | Diagnostic rows | Measured minibatch KL mean / p95 | PPO ratio clipping | Value loss | Episode return | FPS |
|---|---:|---:|---:|---:|---:|---:|
|60–99|40|.003779 / .005120|10.26%|.000632|161.144|58,067|
|100–149|49; 103 missing|.003785 / .005146|10.20%|.000634|161.121|57,700|
|150–199|50|.003793 / .005151|10.15%|.000622|161.118|57,810|

KL is the existing diagnostics measurement of old/current distributions on original minibatch samples before symmetry augmentation. Native scheduler KL arrays are empty: installed RSL-RL 5.5.1 takes that branch only for adaptive scheduling. Empty is not zero; `desired_kl` is not a hard bound. Surrogate loss is approximately -.00069 to -.00078. Final std by FL/FR/RL/RR, each hip/thigh/calf/wheel: `[.1,.11269,.1,.13623] / [.1,.11267,.1,.13623] / [.1,.1,.1,.12330] / [.1,.1,.1,.12333]`. Floor occupancy in the final window is 84–100% for the ten floor-bound leg parameters, zero for the other six. These are nonzero std values, independent of plotted axis offsets, tracking kernels and observation noise. Final sampled clipping reaches ~16.9% for rear calves and 8.6% for front wheels; no LR/exploration reset is justified by these observations alone.

V3 yaw remains `.8 * (1 - Huber(yaw_error/.35))`, with ceiling .8; it is not Gaussian. The ~.765 weighted contribution does not justify a tighter yaw scale. Aggregate averages also reflect the changed command distribution. Final recorded time exposure is 17.18% stand and 23.13% pure lateral. Partial-magnitude time exposure was **not logged**. An offline call to the actual sampler (32,768 draws, seed 7) found 6,620 pure-lateral draws, magnitudes .02004–.29997, 10.45% in [.02,.05), including 378 positive / 314 negative draws. The .01 linear deadzone preserves these commands. This estimates draws, not the actual training history or time weighting.

### Bounded matched physical comparison

Only final 199 was evaluated: the existing 41 s transition sequence once, sustained stand/forward_fast once, and transfer_screen once. No midpoint, robustness bank, extra seed, phase bank or full sustained panel was run. References were reused from `sensor_phase_200_review/ck199_transition`, `ck199_sustained`, `ck199_nominal`, `original_transition`, and their original directories `v3_main_sustained_review` / `transfer_v3_main_ck1499_nominal`. Relevant control/asset/init/sim settings, applied physical readbacks, seed 1, actual command arrays, latent phase arrays and sampling match. Nominal noise/DR/pushes are off; actor inference is deterministic. Stand is 30 s; fast movement is 3 s zero + 30 s .5 m/s + 8 s stop. Screen holds are shorter (3+5+6 s; stand 10 s).

Means/RMSE use the full command segment. “Late” partial response is its last 1 s; steady rolling/stand is 20–30 s after onset. Camera means actual left-imager world-z velocity RMS. Units are m, m/s, and rad/s.

| Measurement | Original V3/1499 | Phase parent/199 | Navigation final/199 |
|---|---:|---:|---:|
|+.03 lateral: mean / late vy|.0254 / .0292|.0181 / .0220|.0227 / .0225|
|-.03 lateral: mean / late vy|-.0310 / -.0214|-.0127 / -.0051|-.0200 / -.0178|
|Partial vy RMSE: positive / negative|.0126 / .0123|.0143 / .0203|.0128 / .0131|
|Uncommanded vx in partials: positive / negative|.0099 / .0071|.0142 / .0073|.0232 / .0152|
|Stand: 0–5 s endpoint / late XY RMS|.0699 / .01056|.0900 / .00178|.0160 / .01682|
|Stand: total 30 s endpoint|.0479|.2042|.3039|
|Fast forward: mean vx / RMSE|.4638 / .0396|.4786 / .0266|.4746 / .0287|
|Fast stop: 8 s endpoint / path|.03156 / .07945|.07593 / .07595|.05995 / .12580|
|Fast stop: final 2 s XY RMS|.00864|.00384|.01501|
|Forward front / rear center width|.4455 / .3924|.4909 / .3800|.5052 / .3772|
|Front widening: last minus first second|.0461|.0894|.1157|
|Forward camera: full / late, 200 Hz|.02195 / .02158|.00755 / .000067|.00602 / .000124|

The navigation policy's late stand endpoint changes .168 m in the final 10 s; startup improved but creeping persisted. Its transition-sequence final 8 s endpoint/path are .1073/.1262 m versus parent's .0240/.0480. Fast-stop immediate first-.2-s vx is .1261 (parent .1262; original .0936). Settling is .24 s (parent .22; original unavailable) using the last excursion above XY speed .02 or |yaw rate| .03, followed by at least 1 s below both. This definition admits the observed .015 m/s creep; it does not certify a stationary hold or a stopping distance. Endpoint cancellation conceals path length and reversal.

Partial-command completed >=2 mm lift counts FL/FR/RL/RR are `[3,2,1,3]` and `[1,3,3,1]`, versus parent's all zeros. Completed peak clearances span roughly .5–5.6 mm, with 0 cycles >=2 cm; limb repositioning means are about 9–24 mm. Event definition remains unload <=6 N / reload >10 N; boundary intervals are censored. Loaded cylinder-center lateral velocity RMS is .004–.015 m/s, comparable in scale to references, but is a scrub surrogate rather than material tire slip. Late partial vx is still .0162/.0202 m/s: improved lifts and vy do not establish clean lateral navigation.

The 5 s strong screen retained mean lateral +.2906/-.2933 (RMSE .0238/.0235) and yaw +.7934/-.7915 (RMSE .0453/.0423). All wheels completed about six >=2 cm cycles; negative yaw's RR has five, also present in the parent. All new cases had no falls, resets, nonfinite states or non-wheel contacts. Maximum deterministic action saturation is 50.8% on strong negative lateral, versus parent 49.2%; this is a target bound, not a force reading. Stand has none, fast movement max .2%, stop max 1.25%; no stand/fast effort readback reaches 99% of its force limit. Matched 200 Hz forward load maxima are ~105 N new / 110 N parent / 101 N original. Stand and transitions use 50 Hz; fast camera/contact comparisons use all four .005 s physics samples per .02 s policy tick.

Full stand camera RMS is .01033 (parent .00963), despite very quiet late values .000028/.000015. Partial-transition camera RMS increased with the new lifts: positive/negative full .0292/.0222 versus .0175/.0091; first-second .0406/.0173 versus .0287/.0110. These rigid-point measurements do not measure image quality or hardware vibration.

### Geometry and one next objective

The existing frozen viewer's `inspect_pose()` imposes joint positions without stepping. Saved reference/FK output in `transfer_v3_preparation/pose_geometry.json` and `transfer_v3_pilot_preparation/pose_reference.json`, together with current `v3_reference_geometry()` / cylinder helpers, gives authored-base collision-center y = +/- .19010 m for both pairs: **front/rear reference width .38020/.38020 m**. This is imposed reference geometry, not loaded equilibrium or a requirement to balance at zero action. No frozen-viewer launch or policy-rollout setter was needed.

New forward mean widths .5052/.3772 m are systematic front splay. Late front width reaches .5379 m, while pair midpoint means stay within .5 mm and midpoint changes within .5 mm; rear separation changes -.0024 m. Mean base height is .40797 m (parent .40848; reference .42774); there is no fall. Late front actual hip angles are +.2313/-.2326 rad, but issued/applied targets are +.0342/-.0344, with target-error RMS .1973/.1983. The target is outward, yet much of the observed splay is loaded actuator tracking error; it cannot be described as an equally large learned outward target. Gains/reference/spawn/URDF were preserved.

The selected next objective activates **the existing `stand_still` coefficient from 0 to -5**, under new opt-in `navigation_zero_hold`. Its unchanged raw cost is `vx² + vy² + yaw_rate² + .02*mean(wheel_joint_rate²)` in body coordinates, gated by `norm(command) < 1e-6`. It is inactive for both partial lateral signs, partial yaw, and moving commands. Existing zero-command tracking alone loses only .00226 rate in late stand; stand exposure is already substantial. On the unmodified traces the added signed rate would be -.00491 in late stand (current rolling pose -.01126), -.02743 over the full fast stop (current x tracking error cost .00666 and rolling pose -.01567), and -.00395 in its final 2 s. This is meaningful objective strength, not predicted learning success. It does not directly price moving partial cross-axis drift or guarantee narrower hips.

No width corridor, extra sensor/angular cost, changed tracking scale, controller override, phase change, observation, sampler change, PPO change or plant change accompanies this proposal. The saved parent recipe is restored first; only that coefficient is activated once. Both models/normalizers/std are retained exactly; a deliberately new objective uses the existing selective path with **fresh Adam/local counter**, preserving the full previous lineage. Proposed settings/proof are in `next_zero_hold_refinement/`. One coherent **500-update** experiment saves every 50, with midpoint `model_250.pt` (251 local updates) and final `model_499.pt`. Production command, prepared but not run:

```powershell
$env:NUMBA_CACHE_DIR = Join-Path (Get-Location) '.cache/numba'
$env:GS_CACHE_FILE_PATH = Join-Path (Get-Location) '.cache/genesis'
$env:QD_OFFLINE_CACHE_FILE_PATH = Join-Path (Get-Location) '.cache/quadrants'
& "$env:USERPROFILE/anaconda3/envs/genesis-gpu/python.exe" -m robot_gym.scripts.train `
  --task go2w --go2w_profile transfer_v3 --go2w_finetune navigation_zero_hold `
  --load_run logs/go2w_transfer_v3_navigation_partial_lateral/navigation_partial_lateral_from199_seed1_2026-10-04_23-48-18 `
  --checkpoint 199 --experiment_name go2w_transfer_v3_navigation_zero_hold `
  --run_name navigation_zero_hold_from199_seed1 --num_envs 4096 --max_iterations 500 `
  --seed 1 --logger tensorboard --training_diagnostics --headless --rl_device cuda:0
```

### Genuine continuation and checks

The new small same-recipe path uses `--task go2w --go2w_profile transfer_v3 --resume --load_run <explicit saved run directory> --checkpoint <label> --max_iterations <additional updates>`, omitting `--go2w_finetune`. Saved settings are authoritative; it explicitly replaces inherited selective load flags with actor/critic/optimizer/iteration all true. It retains Adam, LR, models, normalizers/std, seed, environment count and lineage, validates recipe compatibility, and writes a fresh separate output (default experiment suffix `_resume`). It neither reapplies a delta nor walks back to a grandparent. The selected source was checked **before any update**: exact actor/critic/normalizer/std/Adam parity, Adam step 8,000, LR 5e-5 and runner iteration 199. Simulator/RNG state is not restored bit-for-bit.

Installed runner starts its next loop at the loaded label. From 199, N additional updates execute labels 199 through 199+N-1; e.g. two updates finish at 200, not 201. Saved labels therefore overlap the source even though Adam and actual update counts advance. This upstream convention was preserved, with explicit lineage metadata accounting for it. Separate output prevents source overwrite. Historical replay keeps its saved phase contract and does not require a training output directory.

Edits are confined to Go2-W config/metadata/resume handling and focused tests; the existing reward body is unchanged. Eight focused tests passed for saved-recipe restoration, true-load flags/counter lineage, incompatible overrides, inference replay, historical selective recipes, actual sampling and zero-only gating. No-update native load checks passed. Exactly one 64-environment **two-update** native objective smoke completed at `logs/go2w_transfer_v3_navigation_zero_hold_smoke/zero_hold_execution_smoke_2026-10-05_01-07-29/`, with finite active reward/diagnostic rows and labels 0/1; it validates execution, not physical improvement. No other optimizer updates ran in this review.

The actual three inference commands used the same `--task go2w --experiment_name go2w_transfer_v3_navigation_partial_lateral --load_run navigation_partial_lateral_from199_seed1_2026-10-04_23-48-18 --checkpoint 199 --num_envs 1 --seed 1 --headless --rl_device cuda:0` selection. Their additional flags were respectively `--eval_mode sensor_sustained --eval_phase_transition --output evaluation/navigation_partial_200_review/final_transition`; `--eval_mode sensor_sustained --transfer_cases stand forward_fast --diagnostic_trace --output evaluation/navigation_partial_200_review/final_rolling`; and `--eval_mode transfer_screen --output evaluation/navigation_partial_200_review/final_screen`, run through `python -m robot_gym.scripts.evaluate` in genesis-gpu. Outputs were fresh and historical baselines unchanged.

No production training, push, package upgrade, deletion, receiving-repository edit or new storage task occurred. `play.py`, Go2 and Dodo remain untouched. This is one nominal seed and a bounded screen, with no new robustness/transfer qualification. The selected model retains its 58/16 command-demand contract, including the two raw clock entries and previous-issued-action semantics; a later receiving simulator must preserve its saved normalizer, joint/action mapping and clock timing rather than use a 56-input or unconditional-phase receiver.

## Navigation zero hold: 500-update review (2026-10-05)

**Decision: keep zero-hold 499 as an intermediate development parent and prepare one rolling-placement objective.** Retain 250 as a straighter fallback. Neither warrants a Sim2Sim freeze yet: 499 improves translation at zero command but develops asymmetric rolling geometry and continuing yaw drift; 250 still creeps substantially. Geometry improvement is not guaranteed to correct curvature.

Selected checkpoint: `logs/go2w_transfer_v3_navigation_zero_hold/navigation_zero_hold_from199_seed1_2026-10-05_08-46-00/model_499.pt`. Fallback: `model_250.pt` in the same directory. Exact parent: `logs/go2w_transfer_v3_navigation_partial_lateral/navigation_partial_lateral_from199_seed1_2026-10-04_23-48-18/model_199.pt`. All remain unmodified.

Compact results: [review.json](../evaluation/navigation_zero_hold_500_review/review.json). Native traces, manifests, detailed windows/per-joint measurements, reflection residuals, training inspection and prepared config are under `evaluation/navigation_zero_hold_500_review/`. [commands.ps1](../evaluation/navigation_zero_hold_500_review/commands.ps1) records executable evaluation/replay commands and the separate, unexecuted production proposal.

### Run and logs

The actual production run saved labels 0, 50, 100, 150, 200, 250, 300, 350, 400, 450 and 499. Iteration metadata agrees: **250 represents 251 local updates**, Adam step 10,040; **499 represents 500**, Adam step 20,000 (40 minibatches/update). Preparation is completed; TensorBoard ends at 499. Diagnostic rows **438/448** and TensorBoard row **406** are missing records, not missing updates. Normalizer counts at 250/499 are 694,943,744/760,217,600.

The only saved behavioral delta is `stand_still: 0 -> -5`. Go2-W's active method is `vx_body² + vy_body² + angular_z_body² + .02*mean(wheel_rate²)`, gated by the complete three-axis command norm below 1e-6. Registration applies coefficient and policy dt once. Ordinary play skips reward accumulation. The 58/16 interface, command-demand encoding, sampler, other rewards, normalizers, dynamics, observation noise, training DR and PPO settings otherwise match the parent. Symmetry uses augmentation, with mirror loss disabled and coefficient zero.

Random initial episode counters accompany empty reward/length accumulators. Episode reward entries divide by the full 60 s duration. A full episode is `60/.02/64 = 46.875` updates: logged length rises from 31 ticks at update 0 to approximately 3,000 at 47, while return rises from 1.62 to 157.36 (06:46:06–06:49:33 UTC). This strongly supports startup accounting as a major contributor, without excluding simultaneous learning. Total return across different objectives/durations is not used to rank checkpoints.

| Window | Diagnostic rows | Measured KL mean/p95 | PPO clipping | Value loss | Steps/s |
|---|---:|---:|---:|---:|---:|
| 75–125 | 51 | .00427/.00576 | .1093 | .00333 | 56,120 |
| 225–275 | 51 | .00426/.00569 | .1123 | .00231 | 56,817 |
| 400–499 | 98 | .00421/.00562 | .1097 | .00201 | 56,659 |

The existing wrapper measures KL on original minibatch samples using old/current distributions, excluding augmentation copies. Scheduler KL arrays are empty: installed RSL-RL 5.5.1 calls that branch only for an adaptive schedule. Fixed LR is **5e-5**, also verified in checkpoint Adam groups; desired_kl is not a hard bound.

Signed weighted rates in these windows respectively: stand_still **-.04869/-.03924/-.03641**; sensor vertical **-.00476/-.00484/-.00467**; reference height **-.00460/-.00593/-.00577**; phase clearance **-.00711/-.00745/-.00762**; wheel_corridor **-.000894/-.000992/-.001275**. These stochastic aggregates do not establish deterministic holding. `wheel_corridor` prices fore-aft wheel-under-thigh x offsets, not lateral width. Yaw remains `.8*(1-Huber(error/.35))`, ceiling .8, not a Gaussian; the approximately .765 contribution does not justify tighter tracking scales.

Late effective action std is approximately .1000 except front thighs (.1236). Floor occupancy percentages in hip/thigh/calf/wheel order: FL **49/0/83/97**, FR **48/0/87/96**, RL **40/90/22/98**, RR **43/89/26/97**. Full per-joint std, clipping and raw/weighted rates are in `training.json`. The .1 floor is nonzero—1.8 rad/s for wheels before clipping—and distinct from observation noise or reward-error scales. No LR/exploration change is proposed.

### Physical decision

Only 250 and 499 received fresh 30 s stand and 3 s zero + 30 s vx=.5 + 8 s stop tests. One nominal environment, seed 1, deterministic actor, no noise/pushes/delay. Relevant asset/control/physics/initial-state settings, schedules and sampling match the reused baseline traces (`baseline_compatibility.json`); original V3 retains unconditional phase semantics. No sustained case, transition or screen case fell, reset or made non-wheel contact. Terminal failures were not spliced into successful holds.

| Candidate | Stand endpoint/path m | Late stand XY/yaw RMS (m/s, rad/s) | 8 s stop endpoint/path m | Final stop XY RMS m/s | Forward heading change ° | Max cross-track m |
|---|---:|---:|---:|---:|---:|---:|
| Original V3/1499 | .0479/.2838 | .01056/.02336 | .0316/.0795 | .00864 | -2.92 | .1287 |
| Phase parent/199 | .2042/.2046 | .00178/.00012 | .0759/.0760 | .00384 | +.04 | .0208 |
| Navigation parent/199 | .3039/.3475 | .01682/.00010 | .0600/.1258 | .01501 | +.52 | .1041 |
| Zero hold/250 | .2399/.2402 | .01324/.00030 | .0425/.0789 | .00625 | -.62 | .0066 |
| **Zero hold/499** | **.0526/.1003** | **.00257/.00670** | **.0311/.1051** | **.00544** | **+12.60** | **.5076** |

Late stand is 20–30 s; final stop is its last 2 s. Startup 0–5 s stand displacement is .0332/.0352 m for 250/499, versus .0160 m for the immediate parent. At 499, reduced translation accompanies -11.1° stand heading change. Its stop endpoint resembles original V3, but the longer path and reversal matter: final mean vx/vy is -.00467/+.00265 m/s and yaw RMS .00605 rad/s. At 250 final backward speed is .00620 m/s. Neither endpoint alone nor the older settling threshold establishes a quiet hold.

For vx=.5, 250 mean vx/vy/body-angular-z is **.50489/-.000012/-.000360**, RMSE **.02204/.00240/.01289**; 499 mean is **.47110/-.000696/+.007372**, RMSE **.03189/.00331/.02224**. Heading is derived from pose (unwrapped world projection of body +X), not integrated body angular-z. Cross-track uses the movement-start pose and heading. At 499 the initial heading offset is only .080°; drift is -4.78° in 0–5 s and +6.86° during 20–30 s. Endpoint cross-track is +.421 m, below the .508 m maximum: continuing curvature changes direction.

The supplied GUI sequence—499, direct vx=.6, vy=yaw=0, 1,000 ticks/20 s—was reproduced in **one additional native rollout**. Mean vx is .5697 m/s, yet max cross-track is **.4636 m**, endpoint -.1879 m, and heading goes from -2.46° at 5 s to +7.68° at 20 s. Final-five-second yaw rate averages +.01972 rad/s. Good body vx tracking can therefore coexist with curved travel. Current `play.py` fixes the axes, restores the correct model/config, uses nominal deterministic inference and the same phase timing. No replay/indexing/control bug was found; it remains untouched. No second extra rollout was needed.

### Geometry and symmetry diagnosis

Cached reference output and existing FK/cylinder helpers agree on **.380199997 m** front/rear widths and zero pair midpoints. This is imposed authored geometry, not loaded equilibrium. No static-pose simulation or rollout state setter was used.

| Late vx=.5 quantity | 250 | 499 |
|---|---:|---:|
| Wheel-center y, FL/FR/RL/RR m | .28799/-.28746/.18734/-.18639 | .31286/-.22527/.22222/-.18819 |
| Front/rear width m | .57545/.37372 | .53814/.41041 |
| Front/rear midpoint m | .00026/.00047 | .04379/.01702 |
| Actual front hips FL/FR rad | +.28375/-.28201 | +.35266/-.10875 |
| Applied front hips FL/FR rad | +.03064/-.02960 | +.16930/+.01678 |
| Front hip tracking-error RMS rad | .25471/.25386 | .18340/.12558 |
| Base height m | .40405 | .40376 |
| Roll/pitch mean rad | -.00091/-.03330 | -.00948/-.01064 |

Issued and delayed-applied targets agree exactly; post-step positions are aligned with that tick's held targets. Unlike the historical symmetric loaded hip error, **499 has both asymmetric targets and substantial loaded tracking errors**. Front width grows .00932 m/s in 1–5 s and .00048 m/s in 20–30 s; rear width changes -.00044 m/s late. The GUI sequence ends with front width .5445 m and midpoint +.0391 m.

Late wheel targets/actual rates at 499 (FL/FR/RL/RR) are **4.761/6.116/5.444/4.906** versus **5.439/5.319/5.217/5.214 rad/s**. Current-state control forces average **-.678/+.796/+.227/-.308 Nm**; front hip forces -7.33/+5.02 Nm. These do not establish ground yaw moment. FL calf action clips on **69.6%** of forward ticks (250: both front calves approximately 32.7%). At 200 Hz there is no 99%-force-limit occupancy; peak force ratios are .613/.617 for 499/250. Peak wheel loads at 499 are 102.4/104.4/98.7/98.4 N. Runtime readbacks show matched left/right gains, armature .01, friction and joint mappings; commands are exact. No further physics audit was made.

Full/late forward imager world-z velocity RMS at matched 200 Hz: **250 .00900/.00424**, **499 .00961/.000111**, immediate parent **.00602/.000124**, original **.02195/.02158 m/s**. Stand 50 Hz full/late: 250 .01316/.000154; 499 .01272/.000191. A quiet late point does not erase worsening transitions/geometry and does not establish image quality or hardware vibration performance.

The single transition sequence retains ±.03 lateral response: full mean vy **+.02467/-.02742**, last-2-s **+.02671/-.02694**; unwanted vx is -.00455/-.01555 (late -.01732/-.01531). Completed >=2 mm lift counts FL/FR/RL/RR are **3/3/2/0** and **3/3/0/1**, none >=2 cm. Existing 6/10 N event hysteresis and censored-window handling are retained; clearance distributions and loaded lateral-motion metrics are in native outputs. Small-command control remains imperfect; neither reward credit nor a clearance threshold alone establishes success. Transition-stop endpoint/path improves to .0275/.0301 m versus parent's .1073/.1262.

The 13-case screen retains lateral ±.3 means **+.29356/-.29681** (axis RMSE .02379/.02420), yaw ±.8 means **+.77947/-.78002** (RMSE .05362/.04790). Completed >=2 cm cycles are 6/6/6/6 for both lateral signs and positive yaw, 6/6/6/5 for negative yaw.

Reflection tests use 150 recorded forward observations per sustained candidate, 100 for GUI, mirrored **before the saved normalizer**. Deterministic CPU action parity with recorded outputs is within 6e-7; normalizers remain unchanged. At 499, hip residual RMS in normalized actions is FL/FR/RL/RR **.002686/.001080/.002376/.000713**, physically **.000806/.000324/.000713/.000214 rad**. Largest leg residual is .00245 rad (RR calf); wheel residuals .0418/.0333/.0399/.0547 rad/s. All joints' signed means/RMS/p95 are in `*_symmetry.json`. This does not suggest a large mirror-mapping failure on these states, but proves neither plant symmetry nor closed-loop stability. An equivariant actor can respond asymmetrically to asymmetric states. Geometry's causal role in curvature remains unisolated.

### One prepared refinement

Opt-in **navigation_rolling_placement** adds only `rewards.scales.rolling_placement=-.05`, with saved physical parameters. Per front/rear pair, let `width=y_left-y_right`, `midpoint=(y_left+y_right)/2`:

`raw = mean_pairs[Huber(relu(abs(width-width_ref)-.04)/.10) + Huber(relu(abs(midpoint)-.015)/.05)]`.

It is gated by `norm([command_vy,command_yaw]) < 1e-6`: stand/longitudinal rolling only, protecting partial lateral and intentional turning. No contact gate, hidden controller or rigid pose lock. Width deadband is 4 cm; midpoint deadband 1.5 cm. Geometry is actual collision-center placement, not PD support offsets. References are derived once in preparation, saved and cached; no FK in reward steps or added observations.

On recorded 499 late forward motion, estimated width/midpoint rates are **-.01698/-.00417**, total **-.02115**, versus rolling_pose -.02576 and x/y/yaw tracking-error costs .00454/.000009/.000474. Late stand estimate -.00393 versus rolling_pose -.00853 and stand_still -.000478. This establishes objective scale, not expected learning success. No simultaneous mirror-loss, tracking-scale, hip/gain, action-limit, phase, sensor, sampler, LR or exploration change is prepared.

The selected saved recipe is restored first, with exact actor/critic/normalizer/std retention. The new objective deliberately uses **fresh Adam/local counter zero**, retaining lineage at 2,900 prior updates; this is not same-recipe resume. Existing true resume remains intact. Budget **500 updates**, saves every 50, midpoint review at label 250 (251 updates), final label 499. Saved config/proof: `next_rolling_placement_refinement/` in the review output. **Production command, not executed:**

```powershell
python -m robot_gym.scripts.train `
  --task go2w --go2w_profile transfer_v3 --go2w_finetune navigation_rolling_placement `
  --load_run logs/go2w_transfer_v3_navigation_zero_hold/navigation_zero_hold_from199_seed1_2026-10-05_08-46-00 `
  --checkpoint 499 --experiment_name go2w_transfer_v3_navigation_rolling_placement `
  --run_name navigation_rolling_placement_from499_seed1 `
  --num_envs 4096 --max_iterations 500 --seed 1 `
  --logger tensorboard --training_diagnostics --headless --rl_device cuda:0
```

Edits are confined to Go2-W config, one optional cached geometry/reward calculation and focused tests. Six tests passed: `python -m unittest tests.test_go2w_rolling_placement tests.test_go2w_resume`. They cover measured geometry, deadbands, reflection, partial-command exclusion, saved recipe/history and existing resume behavior. Native no-update loading proves exact model/normalizer/std parity and fresh optimizer/counter. One **64-environment, two-update native smoke** completed at `logs/go2w_transfer_v3_navigation_rolling_placement_smoke/placement_execution_smoke_2026-10-05_10-14-06/`: label 1, Adam step 80, finite models and active finite reward. It validates execution only; no other optimizer updates ran in this review.

No production training, push, package upgrade, deletion, receiving-repository edit, cleanup, full test suite, robustness bank, checkpoint sweep or second extra rollout occurred. `play.py`, Go2 and Dodo are untouched. The 58/16 command-demand contract, normalization, simulator-derived body velocity and previous-issued-action semantics are unchanged. Future diagnostic transfer must preserve these; no hardware or Sim2Sim qualification is claimed.

## Combined rolling-control preparation (2026-10-05)

**Decision: prepare one 500-update warm start, `navigation_rolling_control`, from zero-hold CK499.** This supersedes the unstarted placement-only production proposal. It has two deliberate behavioral deltas: actual rolling placement and conditional yaw precision. This is a combined engineering refinement; improvement cannot later be attributed to either delta independently without an ablation. No new policy performance is claimed here.

Explicit parent: `logs/go2w_transfer_v3_navigation_zero_hold/navigation_zero_hold_from199_seed1_2026-10-05_08-46-00/model_499.pt`. Keep `model_250.pt` there as the straighter fallback and preserve all earlier references. Saved configurations under `logs/` confirm that placement-only has **only the earlier two-update smoke**, with no production run. Branch `testing` was clean at the start of this preparation; `play.py` remains untouched.

Local artifacts are under [rolling_control_preparation](../evaluation/navigation_zero_hold_500_review/rolling_control_preparation/): [fully resolved config](../evaluation/navigation_zero_hold_500_review/rolling_control_preparation/config.yaml), [load proof](../evaluation/navigation_zero_hold_500_review/rolling_control_preparation/preparation.json), exact `config_delta.json`, [cost comparison](../evaluation/navigation_zero_hold_500_review/rolling_control_preparation/costs.json), `action_range.json`, and [frozen selection protocol](../evaluation/navigation_zero_hold_500_review/rolling_control_preparation/selection_protocol.json). Large checkpoints/traces remain outside Git.

### Two changes, one fixed recipe

The existing collision-center placement implementation is reused unchanged. Cached authored-base front/rear reference widths are **.380199997/.380199997 m**, pair midpoints zero. Per-pair dimensionless cost is the sum of Huber width and midpoint deviations after deadbands, averaged over the two pairs: width deadband/scale **.04/.10 m**, midpoint **.015/.05 m**, coefficient **-.05**. There is no contact gate, pose lock, new input, state setter, or FK inside a reward step. Loaded support offsets remain permissible.

V3 yaw remains **`.8 * (1 - Huber((command_yaw - body_angular_z) / scale))`**. Only an opt-in `rewards.phase_objective.rolling_yaw_scale=.07` is added. The saved general scale stays **.35 rad/s**, weight/ceiling **.8**. Both new objectives share the existing gate **`norm([command_vy, command_yaw]) < 1e-6`**: measured motion does not choose the gate; vx of either sign does not disable it. Equality and larger norms use the old yaw scale and exclude placement. Subthreshold requests pass, and their actual commanded yaw is still tracked. This is a hard threshold discontinuity, including immediately after turn-to-zero switches; no command-state machinery or ramp is added. Partial lateral/turning commands retain their previous objectives. This gate is distinct from the smooth phase-demand function.

Old configs omit the new key and retain their exact yaw kernel and replay semantics. Stand_still **-5**, partial lateral **[.02,.30]**, command-demand clock/phase guidance, sensor/angular/orientation rewards, normalizers, learned std and bounds, 58/16 architecture, action/history/reference contract, timing, plant and DR all remain saved-parent values. LR **5e-5 fixed**, entropy **.003**, gamma/GAE **.995/.95**, 64 rollout ticks, and every other PPO setting are unchanged. No heading controller, estimator, mirror-loss change, or inference action modification is introduced.

### Costs on retained trajectories

All entries below are **signed weighted error rates per second**, evaluated at the retained **50 Hz reward boundaries**, not 200 Hz contact peaks. Yaw columns exclude the unchanged +.8 ceiling; total yaw reward equals .8 plus that cost. Early is first 1 s of the named segment; late is 20–30 s for sustained cases, 15–20 s for GUI, and the final 2 s otherwise. Full means the entire command or stop segment. `costs.json` includes each companion term for every full/early/late window, plus first-5-s windows and peaks.

| Retained case | Old yaw full | New yaw full | New first 1 s | New late |
|---|---:|---:|---:|---:|
| 250 stand | -.00002 | -.00039 | -.00001 | -.00001 |
| 499 stand | -.00018 | -.00457 | -.00001 | -.00366 |
| 250 vx=.5 | -.00054 | -.01357 | -.00101 | -.03351 |
| 499 vx=.5 | -.00161 | -.04037 | -.00027 | -.01184 |
| 499 GUI vx=.6 | -.00212 | -.05306 | -.00015 | -.03179 |
| 250 fast stop | -.00006 | -.00158 | -.00689 | -.000005 |
| 499 fast stop | -.00141 | -.03528 | -.06707 | -.00299 |
| 499 lateral-to-roll 1 | -.00153 | -.03816 | -.10329 | -.01269 |
| 499 lateral-to-roll 2 | -.00158 | -.03947 | -.09742 | -.01535 |
| 499 transition stop | -.00012 | -.00289 | -.00206 | -.00227 |
| 499 positive yaw stop | -.00767 | -.18920 | -.18408 | -.09604 |
| 499 negative yaw stop | -.00099 | -.02320 | -.05708 | -.03239 |
| 499 strong positive yaw stop | -.00331 | -.06106 | -.21180 | -.00477 |
| 499 strong negative yaw stop | -.00316 | -.05211 | -.18764 | -.04087 |

Companion costs over the full segments, with the prospective placement coefficient applied once:

| Retained case | Placement | Existing rolling pose | Existing stand_still | Existing x+y tracking error |
|---|---:|---:|---:|---:|
| 250 stand | -.00247 | -.00569 | -.00195 | -.00067 |
| 499 stand | -.00184 | -.00568 | -.00079 | -.00016 |
| 250 vx=.5 | -.01747 | -.02335 | 0 | -.00396 |
| 499 vx=.5 | -.01522 | -.02010 | 0 | -.00833 |
| 499 GUI vx=.6 | -.01318 | -.01775 | 0 | -.01186 |
| 250 fast stop | -.01934 | -.01975 | -.00895 | -.00781 |
| 499 fast stop | -.01380 | -.01761 | -.00966 | -.00712 |
| 499 lateral-to-roll 1 | -.00235 | -.00918 | 0 | -.03286 |
| 499 lateral-to-roll 2 | -.00002 | -.00384 | 0 | -.00948 |
| 499 transition stop | -.00089 | -.00429 | -.00157 | -.00049 |

The selected **.07 rad/s** makes errors within its quadratic region 25 times as costly; outside it Huber grows linearly, so the ratio is less than 25. At 499's first lateral-to-roll transition, first-second yaw cost -.1033 accompanies x+y error -.0965, placement -.00132 and pose -.00806. First-second fast-stop yaw -.0671 accompanies stand_still -.05065, x+y -.05042 and pose -.02136. Late forward yaw -.01184 accompanies placement -.02115 and pose -.02576. Thus the change materially prices curvature and rotation in stand without making steady posture costs irrelevant.

Turn-stop transients are more demanding: strong-turn first-second yaw costs are -.212/-.188, versus stand_still -.0292/-.0310. Worst instantaneous yaw-error rate is **-2.98** (total yaw reward -2.18; error contribution -.0596 per policy tick); the moderate positive-yaw stop also retains a substantial full cost -.189. These are meaningful penalties, not negligible peaks. They do not clearly justify moving away from .07 before a learning test, but they make braking/transition retention a required check. Neither these counterfactual costs nor a 25x local curvature imply learning success. The retained phase sequence supplies lateral-to-roll transitions, **not a direct turn-to-roll trajectory**; existing turn-to-zero screen traces exercise the same gate switch. No new rollout was launched to fill that distinction. Active lateral/turning yaw rewards remain numerically unchanged on those traces.

### Action range and exposure

CK499 FL calf clips **69.6%** of forward ticks entirely in the positive direction: its reference -1.4, scale .4 and clip +/-1 give target **[-1.8,-1.0] rad**. The actor asks for further extension, not flexion. Late actual/applied calf means are **-1.0417/-1.0000**, control force +1.67 Nm; GUI positive clipping is 62.4%. FL hip actual/applied is **+.3527/+.1693**, within a target interval +/- .3. Issued/applied targets agree. Matched forward substeps have zero 99%-force-limit occupancy, peak ratio .613. Target clipping is not torque saturation. More calf flexion and inward hip target range remain available; the data establish an extension limit but **no clear action-range blocker to the intended placement correction**. They do not prove a stable, narrower solution is feasible. No action range, force, gain or geometry change is made.

Actual diagnostic rows cover 498 updates (438/448 missing records). Logged time fractions are **17.24% stand, 16.40% straight**: those families necessarily activate the new gate, providing at least **33.64%** coverage. Exact gate occupancy was not logged. Selected 8–15/20–30 s hold ticks account for **7.83%/5.51%** pooled exposure; the code allows these holds for stand/straight/arc, but the logs have no family-by-hold cross-tab. Late-window fractions are similar. Partial lateral draws remain uniform absolute [.02,.30], both signs, above the .01 linear deadzone. Partial-magnitude time exposure is unavailable. The phase-guided sampler's existing sensor-hold path is active; the legacy moving-long branch is not. No sampler campaign, new exposure statistics, or mixture change is introduced.

### Warm start and fixed decision protocol

Preparation restores the explicit parent's complete saved recipe first and applies the two deltas once. Native no-update loading verified **exact actor/critic/normalizer/std parity**, normalizer counts 760,217,600 each, fresh Adam state, LR 5e-5 and local iteration zero. Prior lineage remains 2,900 updates. This is a **warm start, not training from scratch or true resume**. The existing unchanged-recipe true-resume path remains intact. The resolved config is the fixed rewards/sampler/observations/physics target for a future fresh-start comparison; no fresh-training branch is added.

One budget: **500 updates**, save every 50; installed runner labels midpoint **250 after 251 updates**, final **499 after 500**. Prepared production command, **not executed** (from this checkout; no `--resume`):

```powershell
$env:NUMBA_CACHE_DIR = Join-Path (Get-Location) '.cache/numba'
$env:GS_CACHE_FILE_PATH = Join-Path (Get-Location) '.cache/genesis'
$env:QD_OFFLINE_CACHE_FILE_PATH = Join-Path (Get-Location) '.cache/quadrants'
& "$env:USERPROFILE/anaconda3/envs/genesis-gpu/python.exe" -m robot_gym.scripts.train `
  --task go2w --go2w_profile transfer_v3 --go2w_finetune navigation_rolling_control `
  --load_run logs/go2w_transfer_v3_navigation_zero_hold/navigation_zero_hold_from199_seed1_2026-10-05_08-46-00 `
  --checkpoint 499 --experiment_name go2w_transfer_v3_navigation_rolling_control `
  --run_name navigation_rolling_control_from499_seed1 `
  --num_envs 4096 --max_iterations 500 --seed 1 `
  --logger tensorboard --training_diagnostics --headless --rl_device cuda:0
```

The selection protocol was saved before the execution smoke. Its targets are development goals, **not hardware safety limits**:

- At vx=.5 for 30 s: maximum cross-track **<=.05 m**, absolute pose-derived heading change **<=2 degrees**, measured from the moving-segment start line/heading. Report achieved vx and RMSE; slowing or stopping is not success. Declared speed comparison band: mean .45–.55 m/s, RMSE <=.04 m/s.
- Stand: 20–30 s XY RMS **<=.005 m/s**. Also require improved late yaw RMS and full stand heading change over CK499's .00670 rad/s and -11.1 degrees, retaining CK250's .00030 rad/s comparison. Translation alone cannot qualify holding. Separate startup displacement/path.
- No material increase in full 8 s stop path/reversal/residual motion, or loss of retained stepping. Compare endpoint and path separately against CK499, CK250 and original V3. Use prior approximately 15% comparison bands descriptively, considering absolute magnitude and censoring, rather than turning them into safety thresholds.
- Placement must approach the corridor without impairing support, speed, stopping or necessary corrective actions. Retain both lateral/yaw signs and inspect physical cycles/clearance distributions. Small-lateral cross-axis drift remains unqualified; this recipe does not directly resolve it.

Future evaluation is limited to **250 and 499: existing sustained stand/forward_fast; the better candidate: one phase-transition and transfer_screen; at most one vx=.6 reproduction if needed**. Match nominal conditions/windows/sampling and reuse all historical outputs. Do not relax targets after seeing results. If neither checkpoint meets the combined goals, recommend a **separately authorized fresh-start comparison using this same fixed target recipe**, with appropriate initial exploration and training budget, instead of adding a third reward or another renamed fine-tune.

### Validation and limits

Nine focused tests passed: `python -m unittest tests.test_go2w_rolling_control tests.test_go2w_rolling_placement tests.test_go2w_resume`. They cover old kernels/replay, conditional scale with the real commanded yaw target, threshold boundary, reflection, x/y preservation, measured geometry, reward/dt weighting once, exact recipe restoration and true-resume behavior. Native full-parent load proof is saved separately. Code changes stay in Go2-W's existing config/kernel and focused tests; no evaluator or loading framework was added.

Exactly one **64-environment, two-update native smoke** completed at `logs/go2w_transfer_v3_navigation_rolling_control_smoke/rolling_control_execution_smoke_2026-10-05_11-08-17/`. Saved label 1 has Adam step 80, LR 5e-5, finite models/optimizer and parent-normalizer count advanced by 8,192. Both rewards have finite diagnostics; placement is active, with raw means .00105/.00467, and coefficient/dt weighting matches once. The resolved behavioral config matches the production target; only the requested execution/output budget differs. [smoke_validation.json](../evaluation/navigation_zero_hold_500_review/rolling_control_preparation/smoke_validation.json) and [commands.ps1](../evaluation/navigation_zero_hold_500_review/rolling_control_preparation/commands.ps1) record proof and actual commands. These are the only optimizer updates in this preparation; they validate execution, not improvement.

This preparation uses existing nominal traces only; no fresh behavioral evaluation, robustness bank, production training, push, artifact deletion, stack upgrade, receiving-side edit or broad test campaign was performed. The combined objective has no demonstrated behavioral improvement yet and is not a hardware/Sim2Sim qualification.
