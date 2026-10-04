# Sensor smoothing: verified 500-update review

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


Final fast-forward imager RMS improves ~63% (.02159 to .00796), with faster travel (.464 to .478 for .5 command); stand improves ~32%, forward .2 ~41%, yaw +/- ~57%/~54%, mixed ~47%. Strong positive lateral is nearly flat at 200 Hz (.06234 to .06195), rather than a convincing smoothness gain. The previous 15% target is descriptive. Hip RMS and stopping targets are missed. Fast-stop endpoint displacement .0316/.0890/.1150 m and final 2 s planar RMS .00864/.00896/.01054 m/s worsen. Strong-yaw+ RMSE .0339/.0390/.0455 increases despite mean yaw being close to .8. Nominal yaw+/-.4 and lateral+/-.1 stepping remain, as do reverse and mixed locomotion; all 14 nominal cases completed.

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
