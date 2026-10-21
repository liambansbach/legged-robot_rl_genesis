# Go2-W frozen velocity-estimator baseline

**Development baseline for diagnostic Sim2Sim; stopping limitations unresolved;
not hardware-qualified.** Copy this entire directory. Inference needs no local
training logs, Genesis, RSL-RL or `robot_gym`.

The single default is **`policy.onnx`**, the frozen original actor composed with
its trained warmup estimator. `estimated_policy.pt` is the same composition in
TorchScript. `reference/policy.pt` is the identical original actor with privileged
body velocity inputs, for diagnostics only. The fine-tuned pair is not published
as a second default. Nothing was trained or retuned for this publication.

| Identity | SHA256 of file bytes |
| --- | --- |
| `checkpoint/model_2197.pt` | `c35c5b3861a0f7d0de1975a96e80dc6d45831ecad225b5a82d8031e63e95d9bf` |
| Frozen combined `policy.onnx` | `5117e7a57c39426a1da6d6ceedb2c242d6642e6598ca80663b669e1619fc9c60` |
| `reference/model_1999.pt` | `2f8f3a96ad7fdf83c2966b44d10fc76ac274d9d81130a470af881f81a5980bf6` |
| Original `reference/policy.pt` | `639a443f3da3d9fa15d9e532b85959df55c7bd5618564797740adb66b1d2bfcb` |
| Measured URDF | `d298cc7bf4894e869840bdab9ac60f09548d998444018e61854636cff46b1d8c` |

`manifest.json` lists every payload's role, size and hash, plus model tensor hashes
and export runtime/source identity. Tensor hashes are not file hashes. The
publication commit is the Git commit containing this directory, not a fabricated
self-reference inside the manifest. Checkpoint/config files were copied byte for
byte. The selected actor, critic, Gaussian std, normalizers and native Adam state
equal the parent bitwise. The estimator has 199 retained warmup collections and
3,184 supervised steps. These labels do not imply a new PPO update.

## Verify before importing physics

From any copy of this directory:

```sh
python -m pip install numpy onnxruntime
python -I check_policy.py
# Optional cross-runtime and privileged-reference checks:
python -m pip install torch
python -I check_policy.py --torchscript
```

The checker verifies all file hashes, case-sensitive URDF/mesh/texture closure,
history reconstruction, phase timing and ORT against saved eager outputs. Optional
PyTorch checks both scripted models. No simulator or repository imports occur.
The main fixture is a declared contiguous six-second subset of **frozen-pair**
trials: all 12 cases in nominal and sensor-stress conditions, 7,200 histories.
The 300-history native lifecycle fixture also exercises a selective same-tick
reset. Startup, command refresh and clipped previous actions are included.
Reference tolerances remain `atol=rtol=2e-5` for composed inference and `1e-6`
absolute for history; reference-actor tolerances are recorded separately.

Published checks passed: 7,200 recorded histories plus 300 native lifecycle
histories; maximum ORT action difference `7.16e-7`, velocity difference `3.58e-7`
m/s, history reconstruction error zero. An isolated NumPy/ORT-only copy passed.
Native checks comprised a three-second two-environment replay, a 50-tick estimated
GUI replay and one eight-second relocated truth-reference yaw case. The GUI run
was automated, without owner visual gait approval. The normal suite ran once:
104 passed, with the same two failures/five errors recorded at `7d1f87c` (listed in
`validation/native_smoke.json`); no new test regressions were found.

`validation/fixtures.json` binds each source recording to its checkpoint and
actor/estimator tensors. The earlier three-way screen's outer checkpoint was the
fine-tuned model: it is **not** this default. Only its `*_frozen_estimated` traces
and `*_frozen` comparisons were extracted here. Original full evidence was not
modified. The compact reference fixture preserves its original expected outputs.

## Inputs, state and outputs

`contract.json` is the machine-readable contract. ONNX takes float32
`history[batch,5,55]`, oldest frame first, and returns float32 clipped normalized
`clipped_actions[batch,16]` and `velocity_m_s[batch,3]`. Five 50 Hz samples span
80 ms. The internal actor still has 58 inputs; the 65-input critic is not exported.
The estimate is vx/vy/vz at the authored **base_link origin in body axes**, not
COM velocity or velocity at the IMU mount. Body axes are +X forward, +Y left, +Z up.

| Frame slice | Values before saved physical scaling |
| --- | --- |
| 0:3 | Body gyro, rad/s |
| 3:6 | Projected unit gravity |
| 6:9 | Body command vx, vy, yaw rate; m/s, m/s, rad/s |
| 9:21 | Twelve leg position offsets from the trained reference, rad |
| 21:37 | All sixteen joint velocities, rad/s |
| 37:53 | Previous issued clipped normalized action |
| 53:55 | Demand-conditioned phase sine/cosine |

All saved physical scales happen to be 1 in this pair. Apply them once. The graph
contains fixed estimator feature normalization, the MLP, insertion of the estimate
into the actor vector, observation clipping, saved actor normalization and action
clipping. Do not normalize again outside it. Wheel angles, true velocity, position,
contacts, critic features and randomization labels are excluded. Never zero the
estimate at zero command or bootstrap/fall back to simulator velocity.

Use `deployment_history.PolicyHistory` and `phase_features`. Acquire one coherent
sensor frame and append once at each genuine 20 ms policy boundary. Reset only
affected rows, repeating their first acquired frame across the history. A fresh
same-tick reset row must be reacquired; unaffected rows retain their sample/age.
Between-boundary command refresh uses `refresh`, updating only the latest live
command/phase/issued-action slots. It does not acquire sensors or advance time.

Phase period is .8 s. Demand is the smoothstep of the maximum clipped lateral
(.01 to .05 m/s) and yaw (.10 to .25 rad/s) ramps. Forward speed alone does not
activate phase observations. Explicit Genesis reset initializes phase0/actions0,
then integrates one zero-action policy tick: first policy phase is **.025**.
Automatic/selective reset starts the affected row at phase0 without adding that
explicit settling tick. Getters and command refresh never advance phase. The
stored prior action is the issued clipped action, even if actuation is delayed.

Current simulated gyro comes from base-link angular motion; gravity comes from
the ideal base quaternion with optional synthetic noise/delay/holding. No native
IMU or acceleration input was added. The authored `imu` fixed link is at
`(-.02557,0,.04232)` m, zero RPY, but the production observation path does not
read a sensor attached there. Hardware orientation filtering, mounting axes,
timestamps and joint-speed filtering remain unverified. Native IMU gyro alone
does not supply the actor's filtered orientation/projected gravity.

## Robot and actuation

Load `robot/urdf/go2w_measured_ed8dc93.urdf` with the adjacent `dae/` assets.
These are byte-exact snapshots of the canonical measured model. Relative paths
and case are significant. No package URI, symlink or external texture is required.
Use this same measured model for the composed policy and privileged parent actor.

Policy joint order is FL, FR, RL, RR, each hip/thigh/calf/wheel. Legs use P position
targets `q_ref + action * scale`, with reference `[0,.70,-1.40]` rad and scales
`[.30,.35,.40]` rad per leg. Wheels use V speed targets
`clip(18*action, -min(20, joint_limit), +min(20, joint_limit))` rad/s. Leg targets
need no additional clipping; the allowed intervals already lie inside hard limits.
Kp is 40 on legs and 0 on wheels; active Kd is 1 on all joints. These are actuator
targets, not final motor torques. Actual effort limits and joint limits are in the
contract. Do not invent a Unitree motor permutation: the SDK example groups 12
legs before four wheels, whereas this policy interleaves them.
Nominally applied and issued actions coincide. A modeled actuator delay changes
the applied targets; the observation history still contains the issued actions.

`validation/nominal_dynamics.json` contains **actual nominal loaded** post-reset
mass/COM/inertia, material, joint and solver readbacks from Genesis 1.4.1. It also
records the reference pose, initial state, link merging and import settings.
Merged base mass is 7.083710 kg; total authored mass is 19.68371 kg. Armature is
.01 kg m² on every joint; passive stiffness/damping/frictionloss are zero.
These actuator/passive settings are development assumptions, not identified
hardware parameters. COMs are in authored link frames; inertia matrices are in
Genesis principal inertial frames, with the corresponding quaternions provided.
Transform tensors before comparing to IsaacLab's readbacks.

Wheel collision cylinders have radius .09167 m, width .052 m and authored lateral
offsets/axes. Radius is inferred from the owner's approximate loaded tread
circumference, not a calibrated rolling radius. Vendor wheel inertia is unchanged.
Camera/mount masses and some poses were measured; COM/inertias use uniform-density
STL estimates and documented adapter assumptions. Preserve the URDF qualifications.
Fixed-link merging is enabled with wheel links retained and `align=False`.

## Existing comparison and limitations

`validation/replay.json` specifies the 12 cases, exact 10 Hz command times/gains,
reset rules, seeds, zero windows and optional world-frame 30 N push at base COM.
Use 50 Hz policy and reference 200 Hz physics. A smaller physics dt must change
decimation to preserve policy/history/phase timing. Nominal physics was used in
both conditions; stress adds synthetic noise, 20 ms delay, 40 ms holding and one
push. This is a small simulation screen, not measured hardware uncertainty.

| Whole-panel metric | Parent + truth | Frozen + estimator |
| --- | ---: | ---: |
| Nominal vx/vy/yaw tracking RMSE | .0238 / .00858 / .0240 | .0247 / .00950 / .0240 |
| Stressed vx/vy/yaw tracking RMSE | .0249 / .0108 / .0428 | .0288 / .0126 / .0435 |
| Nominal max ordinary stop path, m | .1186 | .1491 |
| Stressed max ordinary stop path, m | .1440 | .1568 |
| Nominal ten-second stand path, m | .0034 | .0361 |
| Stressed max post-push recovery path, m | .0564 | .1053 |

Frozen estimator RMSE vx/vy/vz is .0137/.00651/.00638 m/s nominal and
.0215/.0134/.0181 m/s stressed. No falls or undesired contacts occurred in this
panel. **Six nominal and three stressed cases fail the existing stop-path
comparison tolerances.** No overall pass is claimed. Full per-case stop drift,
heading/yaw, wheel spin, slew, tilt, saturation and settling results are in
`validation/comparison.json`. Ordinary stops and post-push recovery are separate.
Historical component RMS `sqrt(mean(v_xy**2))` is retained; planar-speed RMS
`sqrt(mean(vx**2+vy**2))` is an additional, differently named metric. These
thresholds are proposed diagnostic criteria, not hardware safety limits.

The fine-tuned local pair improved nominal estimation but worsened long quiet
drift and stressed actuation slew. That motivated this conservative frozen
baseline; publication does not resolve stopping. No further training is proposed
solely from training reward curves.

## Receiving IsaacLab project

1. Run isolated recorded-input/history parity before any physics rollout.
2. Import the measured model and verify runtime wheel geometry, mass, COM/inertia
   after merging, joint permutation, active/passive gains, armature and limits
   against the nominal readbacks. Keep collision/material semantics explicit.
3. Implement the **composed estimator + actor + history/phase** pipeline at 50 Hz,
   with 10 Hz command holds. Run the declared panel, then the privileged reference
   under matched conditions. Use true base-origin velocity only for metrics and
   the explicit reference actor; log wheel targets/speeds, slip/contact response
   and stop trajectories separately.
4. Review import/control equivalence and numerical convergence. Genesis is not
   physical ground truth. Do not use per-command tuning, yaw multipliers, invented
   inertia, inflated torque limits or disabled collisions to match a curve.

The historical truth-input Genesis/PhysX yaw mismatch remains. The small
`validation/source_yaw_summary.json` cites the unchanged original report:
Genesis late yaw error was <=.00815 rad/s at +/- .6/.8 while matched PhysX
undershot by .26285–.29624 rad/s. An estimator does not resolve that difference.
Required yaw is +/-1.0, navigation requested +/-1.25, and training reserve is
+/-1.5 rad/s. Reserve exposure does not certify either requested envelope.

No navigation retraining/replacement, ROS2 work or hardware trial occurred here.
After satisfactory scoped Sim2Sim, a separately authorized hardware audit must
check actual Go2-W topics, IMU axes/quaternion/filter/sample age, joint/wheel
encoder signs/order/rates/filtering, and command/actuator timing. LowState provides
IMU and motor fields, not base linear velocity. SportModeState's separate velocity
field needs its validity checked during custom low-level control.

## Native replay in this repository

With the repository's Genesis environment active (`pip install pygame` for manual
control), from the repository root in PowerShell:

```powershell
$run = (Resolve-Path 'ressources/pretrained/go2w/checkpoint').Path
python -m robot_gym.scripts.play --task go2w --load_run "$run" --checkpoint 2197 --num_envs 1 --velocity_source estimated --manual_control
python -m robot_gym.scripts.play --task go2w --load_run "$run" --checkpoint 2197 --num_envs 1 --velocity_source estimated --command_vx 0 --steps 500
python -m robot_gym.scripts.play --task go2w --load_run "$run" --checkpoint 2197 --num_envs 1 --velocity_source estimated --command_vx .2 --command_yaw .3 --steps 250
```

Manual replay retains Shift slow mode, Space stop, adjustable directional limits
and focus-loss handling. It runs until closed unless `--steps` is explicit.
The printed source identifies estimated inference; viewer velocity arrows show
simulator truth. A short automated GUI smoke is not human gait approval.

Full checkpoint/config pairs are for deliberate native replay/resume, not required
for ONNX inference. Saved configs retain original absolute paths **as provenance**;
explicit `--load_run` resolves the packaged checkpoint. They are not portable
runtime asset dependencies. Do not reset cumulative training counters to bypass
a future budget. No training is authorized by these playback commands.

This publication replaces the old 56-input P/CK2498 package and retires the dated
handoff/model1000 fallback. Retrieve those with their matching source/assets at
[`7d1f87c`](https://github.com/liambansbach/legged-robot_rl_genesis/tree/7d1f87c87fe1b190b4278fe870c69d6c652a3ce5).
Never load an old policy against a different URDF just because a filename resolves.
The estimator principle follows IIT DLS Lab's BSD-3-Clause
[basic-locomotion-isaaclab](https://github.com/iit-DLSLab/basic-locomotion-isaaclab/tree/dd92b3116f89caf8b083490d5d815fdbff8ac665);
the implementation is local, with no copied upstream implementation. Original
asset attribution remains in the XML/mesh files.
