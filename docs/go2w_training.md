# Go2-W locomotion

The wheeled quadruped learns body-frame forward, lateral and yaw velocity commands on a flat plane. `GO2WCfg` and `GO2WCfgPPO` define fresh training. Each run saves its resolved configuration for replay and continuation. [Task configuration](../robot_gym/envs/go2w/go2w_config.py)

## Robot and timing

The measured URDF includes the camera assembly and four cylindrical wheels. Each policy tick spans four physics steps of 5 ms. Joint order is FL, FR, RL, RR with hip, thigh, calf and wheel in each leg. The twelve leg actions offset the reference pose. The four interleaved wheel actions command angular speed. The current actor uses 58 inputs including twelve leg offsets, all sixteen joint speeds and two demand-conditioned phase features. Its 65-input critic also observes geometric reference state. [Robot asset notes](../ressources/robots/go2w/README.md)

## Train and evaluate

```sh
python -m robot_gym.scripts.train --task go2w --num_envs 4096 --max_iterations 2000 --seed 1 --logger tensorboard --training_diagnostics --headless
```

Training diagnostics are opt-in. For replay, select an absolute run directory and its numbered checkpoint. For continuation add `--resume --load_run` and `--checkpoint` to the training command. `--max_iterations` counts additional updates. The runner restores saved learning state and begins new simulator episodes.

The published package contains both the estimated policy and its truth-driven parent. From the repository root, set `pretrained` as in the [replay quickstart](../README.md#replay-a-published-policy). To compare the pair in simulation:

```sh
python -m robot_gym.scripts.evaluate --task go2w --estimator_screen --load_run "$pretrained/go2w/go2w_locomotion/checkpoint" --checkpoint 2197 --comparison_checkpoint "$pretrained/go2w/go2w_locomotion/reference/model_1999.pt" --headless --output evaluation/go2w_estimator
```

The published comparison used an extended 12-case panel with longer zero-command windows. Its command schedules, disturbances, checkpoint identities and per-case results travel with the package. The ordinary two-way screen above uses the shorter panel. Use `python -m robot_gym.scripts.evaluate --task go2w --help` for task-specific evaluation modes. [Published replay protocol](../ressources/pretrained/go2w/go2w_locomotion/validation/replay.json)

## Why this checkpoint

- The earlier 56-input actor used another robot/configuration combination. Retrieve it together with its matching assets. [Earlier publication](https://github.com/liambansbach/legged-robot_rl_genesis/tree/7d1f87c87fe1b190b4278fe870c69d6c652a3ce5)
- Geometric reference tracking and the subsequent objective refinement produced the 58-input parent at checkpoint 1999. Its full checkpoint and truth-input TorchScript actor remain in `reference/`. [Refinement measurements](go2w_refinement_transfer_2026-10-09.json)
- The selected checkpoint 2197 pairs that frozen actor with the supervised warmup estimator. Subsequent joint fine-tuning improved nominal velocity estimation but increased long quiet-standing drift and stressed actuation slew. On the published screen, six nominal and three stressed cases exceeded the frozen pair's stop-path comparison tolerances. The evidence covers simulation and export. Cross-simulator turning response, stopping and hardware behavior remain open. [Frozen package and results](../ressources/pretrained/go2w/go2w_locomotion/README.md)

The dated experiment narratives, reconstruction notes and original diagnostic commands are available at their publication revision. Recorded measurements retain the source paths and checkpoint identities from those runs. [Experiment archive at 9c9bda3](https://github.com/liambansbach/legged-robot_rl_genesis/tree/9c9bda3dabbf8f5807afcceca23340c8b91ba9cf/docs)
