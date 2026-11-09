# Legged robot locomotion in Genesis

Train and replay PPO locomotion policies for the Dodo biped, Unitree Go2 quadruped and Go2-W wheeled quadruped. The three tasks share a Genesis environment step, task registry, RSL-RL runner and keyboard controller. Each task supplies its own robot model, actuator settings and rewards.

| Published policy | Actor inputs / actions | Control |
| --- | --- | --- |
| [Dodo walking](ressources/pretrained/dodo/dodo_walking/README.md) | 36 / 8 | Joint position targets |
| [Go2 walking](ressources/pretrained/go2/go2_walking/README.md) | 48 / 12 | Joint position targets |
| [Go2-W locomotion](ressources/pretrained/go2w/go2w_locomotion/README.md) | 58 / 16 | Leg position and wheel speed targets |

The Go2-W package includes a learned base-velocity estimator, its frozen actor and an independently runnable ONNX graph. These policies were evaluated in simulation.

## Install

From the repository root, with Conda and a GPU driver compatible with the CUDA 13.0 PyTorch build:

```sh
conda env create -f conda_env.yaml
conda activate genesis-gpu
```

The tested stack is Python 3.11.14, PyTorch 2.9.0+cu130, Genesis 1.4.1 and RSL-RL 5.5.1. TorchScript export uses this environment. Install pygame 2 for keyboard replay with `python -m pip install pygame`. For ONNX export and CPU parity checks, install `python -m pip install -r requirements-export.txt`.

## Replay a published policy

Set the absolute path to the packages once from the repository root.

| Shell | Command |
| --- | --- |
| Linux (Bash) | `pretrained="$(pwd)/ressources/pretrained"` |
| Windows (PowerShell) | `$pretrained = (Resolve-Path 'ressources/pretrained').Path` |

The following commands work in either shell. Each opens Genesis and a small keyboard window.

```sh
python -m robot_gym.scripts.play --task dodo --load_run "$pretrained/dodo/dodo_walking" --checkpoint 1499 --manual_control
python -m robot_gym.scripts.play --task go2 --load_run "$pretrained/go2/go2_walking" --checkpoint 1499 --manual_control
python -m robot_gym.scripts.play --task go2w --load_run "$pretrained/go2w/go2w_locomotion/checkpoint" --checkpoint 2197 --velocity_source estimated --manual_control
```

Focus the keyboard window. Hold **W/S** for forward/reverse, **A/D** for left/right and **Q/E** for counterclockwise/clockwise yaw. Axes combine, opposite keys cancel, Shift gives 30% speed and Space requests a stop. Releasing the keys or losing focus requests zero. Release held keys after refocusing. Tab selects a limit axis and +/− adjusts its speed for the session. Each direction starts at its own saved command limit. Esc exits.

Remove `--manual_control` for sampled commands. For a constant body-frame command, replace it with e.g. `--command_vx 0.2 --num_envs 1 --steps 250`. Omitted axes are zero in fixed-command mode. Linear speeds are m/s and yaw rate is rad/s. Add `--headless` to run without the viewer.

## Train and export

```sh
python -m robot_gym.scripts.train --task dodo --num_envs 512 --max_iterations 100 --logger tensorboard --headless
python -m robot_gym.scripts.play --task dodo --load_run "$pretrained/dodo/dodo_walking" --checkpoint 1499 --num_envs 1 --steps 1 --headless --export --export_dir evaluation/dodo_export
```

Training writes the resolved `config.yaml` and checkpoints to a timestamped run under `logs/<experiment>/`. To continue a run, use `train --resume` with its absolute `--load_run` directory and checkpoint number. Replay and resume restore the selected configuration, including the robot asset. The published Dodo uses `dodo_daimao.urdf`. Fresh training uses `dodo_daimao2.urdf`.

The two published walking runs use immediate actions and uniform command sampling with a dead zone. Their saved reward terms and proportional joint reset noise carry into continued training. Dynamics and transition accounting use the current Genesis pipeline, so continued trajectories can differ from those recorded with Genesis 0.4.6.

Export to a fresh directory. The bounded TorchScript actor includes its observation normalizer. Add `--onnx` for ONNX. Estimator runs also export the composed history-to-action model and require recorded histories for ONNX parity. [Estimator and task extension guide](docs/velocity_estimator.md)

## Demonstrations

**Dodo**

https://github.com/user-attachments/assets/8276639f-ffef-4358-a578-7138b7d5190d

<img width="1658" height="715" alt="image" src="https://github.com/user-attachments/assets/655c3ed9-ef01-461f-8c61-9e362383d8f7" />
<img width="2469" height="1260" alt="image" src="https://github.com/user-attachments/assets/0e6ed3f3-12a9-408f-8176-a8bbe92f4dab" />

**Go2**

https://github.com/user-attachments/assets/afaef573-847f-4afc-a96d-eb92982bcb41

<img width="1657" height="677" alt="image" src="https://github.com/user-attachments/assets/d4dcaaed-5765-42fa-9461-c76f04ecce89" />
<img width="2469" height="1258" alt="image" src="https://github.com/user-attachments/assets/2ecb48ee-5110-4de5-927c-b43a35b0307a" />

## Repository map

- [robot_gym/envs](robot_gym/envs) contains the base environment, registered tasks and their training configs.
- [robot_gym/scripts](robot_gym/scripts) contains the training, replay and evaluation entry points. [robot_gym/utils](robot_gym/utils) holds shared run loading, keyboard input, estimation and export.
- [ressources](ressources) contains robot models and pretrained runs. [tests](tests) contains regression fixtures and checks.
- [Go2-W training and evaluation](docs/go2w_training.md) describes the measured model, command panel and published evidence.

The task/config/registry structure draws on [Unitree RL Gym](https://github.com/unitreerobotics/unitree_rl_gym/tree/main). Contact: liam.bansbach@tum.de.
