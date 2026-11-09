# Go2-W locomotion

Copy this entire directory to use the published frozen actor and warmup velocity estimator. It includes the measured robot, two native runs, the composed policy and recorded simulation evidence.

| File | Use |
| --- | --- |
| `policy.onnx`, `estimated_policy.pt` | Equivalent ONNX and TorchScript graphs combining the estimator and actor |
| `checkpoint/model_2197.pt`, `checkpoint/config.yaml` | Full frozen pair for native Genesis replay or resume |
| `reference/model_1999.pt`, `reference/config.yaml` | Full parent run with simulator velocity |
| `reference/policy.pt` | Truth-input parent actor for controlled comparisons |
| `contract.json`, `manifest.json` | Input/output contract, asset dependencies, file hashes and saved learning-state identities |
| `robot/`, `validation/` | Portable URDF and meshes, nominal loaded dynamics, replay schedule, comparison and lifecycle fixtures |

## Check the copy

From this directory, run:

```sh
python -m pip install numpy onnxruntime
python -I check_policy.py
```

With PyTorch installed, `python -I check_policy.py --torchscript` also checks both scripted models. The checker verifies all payload hashes, case-sensitive mesh references, recorded history reconstruction, phase timing and inference against saved outputs. It uses 7,200 histories from twelve command cases in nominal and stressed conditions and 300 native lifecycle histories including a selective reset. Simulation setup and training logs are unnecessary for these checks.

## Controller interface

The graph takes float32 history shaped `[batch,5,55]`, oldest frame first. It returns clipped normalized actions `[batch,16]` and velocity `[batch,3]` in m/s at the authored base-link origin. Body axes are +X forward, +Y left and +Z up. Output units, physical scales and each action slot are specified in [contract.json](contract.json).

| Frame slots (end exclusive) | Values before saved physical scaling |
| --- | --- |
| 0–3 | Body gyro, rad/s |
| 3–6 | Projected unit gravity |
| 6–9 | Command vx and vy in m/s, yaw rate in rad/s |
| 9–21 | Twelve leg position offsets, rad |
| 21–37 | Sixteen joint velocities, rad/s |
| 37–53 | Previous issued clipped actions |
| 53–55 | Demand-conditioned phase sine and cosine |

All saved physical feature scales in this pair are 1. The graph applies learned feature normalization, velocity insertion, observation clipping, actor normalization and action clipping. Feed the measured history directly, including at startup and at zero command.

Use `deployment_history.PolicyHistory` to append one coherent sensor frame at each 20 ms policy boundary. Repeat the first acquired frame for reset rows. A same-tick command change refreshes only the newest controller slots. Explicit Genesis reset integrates one zero-action tick, so the first phase is .025 cycles. Selective reset starts the affected row at phase zero. The .8 s clock's observed amplitude depends on lateral and yaw demand. The helper reproduces these conventions. The previous action is the issued clipped target even under actuator delay.

The policy expects FL, FR, RL, RR order and hip/thigh/calf/wheel order within each leg. It offsets leg targets from `[0,.70,-1.40]` rad by actions scaled by `[.30,.35,.40]` rad. Wheel actions are speed targets scaled by 18 rad/s and limited to the smaller of 20 rad/s and the joint limit. These are mixed position/velocity targets. Leg Kp is 40, wheel Kp is zero and Kd is 1 for all joints. Verify any external motor permutation explicitly.

Load `robot/urdf/go2w_measured_ed8dc93.urdf` with its adjacent meshes. Preserve the authored fixed-link transforms, retained wheel links, armature, active/passive joint properties, material and limits when comparing simulators. The loaded mass, COM, principal inertias and reference pose are in [nominal_dynamics.json](validation/nominal_dynamics.json). The .09167 m wheel radius comes from approximate loaded tread circumference. Wheel inertia is vendor data. Camera and mount inertias are geometric estimates. The gyro and projected gravity originate from simulated base motion. Hardware orientation filtering, mounting axes, encoder ordering and timing require separate verification.

## Simulation evidence

The parent actor, critic, normalizers, Gaussian std and Adam state in checkpoint 2197 match checkpoint 1999 bitwise. The estimator has 199 retained warmup collections and 3,184 supervised steps. The extended replay panel used 50 Hz policies, 200 Hz physics and 10 Hz command holds. Sensor stress added synthetic noise, 20 ms delay, 40 ms holding and a world-frame push at the base COM. [Replay protocol](validation/replay.json)

| Whole-panel metric | Parent + truth | Frozen + estimator |
| --- | ---: | ---: |
| Nominal tracking RMSE vx / vy / yaw | .0238 / .00858 / .0240 | .0247 / .00950 / .0240 |
| Stressed tracking RMSE vx / vy / yaw | .0249 / .0108 / .0428 | .0288 / .0126 / .0435 |
| Nominal maximum ordinary stop path, m | .1186 | .1491 |
| Stressed maximum ordinary stop path, m | .1440 | .1568 |
| Nominal ten-second stand path, m | .0034 | .0361 |
| Stressed maximum post-push recovery path, m | .0564 | .1053 |

No falls or undesired contacts occurred in this panel. Six nominal and three stressed cases exceeded stop-path comparison tolerances. The fine-tuned candidate had more quiet-standing drift and stressed actuation slew. The frozen pair is the published simulation baseline. Stopping, cross-simulator yaw response and hardware performance remain open. See the [per-case results](validation/comparison.json) and [truth-input yaw comparison](validation/source_yaw_summary.json).

## Native replay

In the repository's active Genesis environment, set the absolute run path from the repository root:

| Shell | Command |
| --- | --- |
| Bash | `run="$(pwd)/ressources/pretrained/go2w/go2w_locomotion/checkpoint"` |
| PowerShell | `$run = (Resolve-Path 'ressources/pretrained/go2w/go2w_locomotion/checkpoint').Path` |

```sh
python -m robot_gym.scripts.play --task go2w --load_run "$run" --checkpoint 2197 --velocity_source estimated --manual_control
```

Install pygame for manual control. For a fixed forward command replace `--manual_control` with `--command_vx 0.2 --num_envs 1 --steps 250`. The viewer's velocity arrows show simulator truth. The saved configs retain their original source paths as provenance.

The retired 56-input actor and matching robot can be recovered at the [earlier publication](https://github.com/liambansbach/legged-robot_rl_genesis/tree/7d1f87c87fe1b190b4278fe870c69d6c652a3ce5). The estimator approach draws on the BSD-3-Clause [IIT DLS Lab reference](https://github.com/iit-DLSLab/basic-locomotion-isaaclab/tree/dd92b3116f89caf8b083490d5d815fdbff8ac665). Original asset attribution is retained in the XML and mesh files.
