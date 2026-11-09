# Adding a task and an optional velocity estimator

## Another robot

1. Place the URDF and its referenced meshes under `ressources/robots/<robot>/`. Use the URDF joint names for the reference pose, gains, action scales and actuator types. Set the base and foot/contact links appropriate to this model. The loader resolves asset paths and reads movable joints. Verify their order and the foot links against the model.
2. Create an environment and configs under `robot_gym/envs/<robot>/`. Use `LeggedRobot` for stepping, selective resets, command ownership and the canonical observations. Define the asset, spawn pose, joint/control parameters, observation and action dimensions, episode settings, rewards and PPO config. Implement task-specific rewards or geometry in the task environment. [Dodo example](../robot_gym/envs/dodo/dodo_config.py)
3. Import and register the task with `task_registry.register("my_robot", MyEnv, MyCfg(), MyCfgPPO())`. The registered name becomes the `--task` argument for train and play. [Registration](../robot_gym/envs/__init__.py)
4. Train with `python -m robot_gym.scripts.train --task my_robot --headless`. Replay with `python -m robot_gym.scripts.play --task my_robot --load_run RUN --checkpoint N`, replacing `RUN` with the absolute saved directory and `N` with a saved checkpoint number. Keyboard replay uses the configured asymmetric vx, vy and yaw limits. [Quickstart](../README.md)

## Estimator inputs

Ordinary policies take body velocity from the simulator. An explicitly enabled estimator predicts body-frame base linear velocity in m/s from measured features and controller history. The critic, rewards and supervised target still use clean simulator state. Only Go2-W ships a trained estimator.

For the canonical layout, actor slots 0–2 are base linear velocity. The rest are body angular velocity, projected gravity, three command values, joint offsets, joint speeds and the previous issued clipped action. Feature sizes follow the task's joint and action dimensions. Dodo has 33 features per frame, Go2 has 45. Go2-W uses a specialized 55-feature frame with twelve leg positions, all wheel speeds and demand-conditioned phase. [Go2-W frame and actuator contract](../ressources/pretrained/go2w/go2w_locomotion/contract.json)

For a different layout, override `velocity_estimator_spec()`. Identify the three actor velocity slots, the ordered remaining actor slots, the live controller slots within that feature vector, physical scales, units and reference point. The two sets of actor indices must partition the actor vector. Keep clean critic inputs at the parent critic's width. If a task uses symmetry augmentation, supply the matching transformations for its measured history and supervised target. [Shared history and model](../robot_gym/utils/velocity_estimator.py)

Each policy boundary contributes one coherent measurement. Reset pads only affected rows with the first measured frame. Within a tick, a changed command refreshes the latest controller features. Delay and holding apply to sensors. History always contains the last *issued, clipped* action, including when actuation is delayed. The default MLP reads five frames, oldest first. Feature normalization lives in the estimator. The actor's existing normalizer runs once after velocity insertion.

## Initialize, resume and replay

Start from a full native checkpoint and its adjacent `config.yaml`. Warmup collects trajectories with the frozen parent actor and trains the estimator against simulator truth. For example, with `pretrained` set as in the [replay quickstart](../README.md#replay-a-published-policy):

```sh
python -m robot_gym.scripts.train --task dodo --resume --load_run "$pretrained/dodo/dodo_walking" --checkpoint 1499 --estimator_init --estimator_stage warmup --max_iterations 1 --num_envs 32 --logger tensorboard --headless
```

One collection exercises the software path. Policy quality requires learning and evaluation. Use the resulting run's full checkpoint for ordinary `--resume` or replay. Omit `--estimator_init` when resuming a pair. Explicit `--estimator_stage finetune` switches the actor to estimated inputs for PPO. Replay defaults to the saved estimator from the first tick. Select `--velocity_source truth` only for a diagnostic comparison. [Estimator PPO and saved state](../robot_gym/utils/estimator_ppo.py)

## Compare and export

Use `python -m robot_gym.scripts.evaluate --task TASK --estimator_screen --load_run RUN --checkpoint N --comparison_checkpoint PARENT --headless --output OUTPUT` with a saved estimator run, its full parent checkpoint file and a new results directory. Canonical tasks use their command ranges, action clipping and policy timestep. Go2-W retains its velocity, phase, wheel, stop and disturbance panel. [Go2-W evaluation](go2w_training.md)

Replay `--export` writes an actor and a composed `estimated_policy.pt`. For ONNX add `--onnx --parity_histories FILE` and select a fresh `--export_dir`. Use recorded histories from the same task and input contract. For the published pair:

```sh
python -m robot_gym.scripts.play --task go2w --load_run "$pretrained/go2w/go2w_locomotion/checkpoint" --checkpoint 2197 --velocity_source estimated --headless --num_envs 1 --steps 1 --export --onnx --export_dir evaluation/go2w_export --parity_histories "$pretrained/go2w/go2w_locomotion/parity_samples.npz"
```

The composed graph returns clipped actions and estimated velocity. Supply scaled history in contract order and let the graph apply the saved normalization. History acquisition, reset padding, controller refresh and any task-specific phase clock run outside the graph. The standalone NumPy helper and recorded comparisons accompany the export. [Published portable checker](../ressources/pretrained/go2w/go2w_locomotion/README.md)

The history-to-velocity approach draws on `basic-locomotion-isaaclab` (BSD-3-Clause, copyright 2025 DLS Lab at Istituto Italiano di Tecnologia). [Reference method](https://github.com/iit-DLSLab/basic-locomotion-isaaclab/tree/dd92b3116f89caf8b083490d5d815fdbff8ac665)
