# RL GYM – Reinforcement Learning for Legged Robots (Genesis)

This repository provides a **modular reinforcement learning pipeline** for training legged robots using the **Genesis physics engine**.

It is inspired by the structure of Unitree RL pipelines but fully adapted to:

- **Genesis (GPU accelerated physics)**
- **Custom robots (URDF / MJCF)**
- **PPO (rsl-rl)**

---

## Features

- GPU-accelerated RL training with **Genesis**
- PPO implementation via **rsl-rl**
- Clean **task + config + registry system**
- Automatic **URDF parsing (joints + feet)**
- Supports **Train → Play → Export → Deployment**
- Integrated logging via **Weights & Biases**
- Multiple environments (flat + uneven terrain)
- Bounded TorchScript export with embedded observation normalization

---
## Results 

Following Results were achieved using this setup:
- Intel Core i5-13600K (14 cores, 20 threads)
- 32 GB RAM (DDR5 - 6000 MT/s)
- NVIDIA GeForce RTX 4070 Ti
- 4096 Envs, 1500 iterations -> runtime: 45min

Dodo:

https://github.com/user-attachments/assets/8276639f-ffef-4358-a578-7138b7d5190d

<img width="1658" height="715" alt="image" src="https://github.com/user-attachments/assets/655c3ed9-ef01-461f-8c61-9e362383d8f7" />
<img width="2469" height="1260" alt="image" src="https://github.com/user-attachments/assets/0e6ed3f3-12a9-408f-8176-a8bbe92f4dab" />


GO2:

https://github.com/user-attachments/assets/afaef573-847f-4afc-a96d-eb92982bcb41

<img width="1657" height="677" alt="image" src="https://github.com/user-attachments/assets/d4dcaaed-5765-42fa-9461-c76f04ecce89" />
<img width="2469" height="1258" alt="image" src="https://github.com/user-attachments/assets/2ecb48ee-5110-4de5-927c-b43a35b0307a" />


---

## Pipeline Overview

- **Train**: Learn policy using PPO in simulation  
- **Play**: Visualize trained policy
- **Export**: Export a trained policy in order to deploy it (Sim2Sim or Sim2Real)

---

## Project Structure

```text
robot_gym/
│
├── envs/
│ ├── base/
│ ├── dodo/
│ ├── go2/
│ ├── go2w/
│
├── scripts/
│ ├── train.py
│ ├── play.py
│
├── utils/
│ ├── task_registry.py
│ ├── helpers.py
│ ├── debug.py
│ ├── math.py
│ ├── terrain.py
│ ├── urdf_reader.py
│
├── ressources/
│ ├── robots/
│ ├── pretrained/
logs/

```

-> The main structure and many design choices are based on this repository: https://github.com/unitreerobotics/unitree_rl_gym/tree/main

---

## Installation

### 1. Clone

SSH example:

```bash
git clone git@github.com:liambansbach/legged-robot_rl_genesis.git
cd legged-robot_rl_genesis
```

### 2. Setup environment

```bash
conda env create -f conda_env.yaml
conda activate genesis-gpu
```

---

## Usage

### Training

```bash
python -m robot_gym.scripts.train --task dodo --experiment_name dodo_walking_test --num_envs 4096 --max_iterations 1000
```

Go2-W uses one current configuration in
[go2w_config.py](robot_gym/envs/go2w/go2w_config.py): a flat plane, 16 mixed P/V
actions and 58 observations including the demand-conditioned phase clock.
Use the ordinary `--task go2w` command; each run saves its complete resolved
`config.yaml`. See [current Go2-W training](docs/go2w_training.md) for the production
command, saved-run compatibility and behavior-equivalence evidence.

The historical [experimental Go2-W simulation policy](ressources/pretrained/go2w/README.md)
retains its published weights, interface and portable CPU check. Its older interface
is separate from the current task. Historical reports and artifacts remain available;
source recipe selectors have been removed.

Training pipeline:

- PPO via rsl-rl
- parallel environments on GPU
- config-driven setup via TaskRegistry
  - If you want to train your own robot, simply add your URDF file to "ressources/robots/", create a new config and env file in "robot_gym/envs" and also register the new task in "envs/__init__.py".

### Play (Evaluation)

```powershell
python -m robot_gym.scripts.play --task dodo --experiment_name dodo_walking --load_run RUN_DIRECTORY --checkpoint -1
```

Replace `RUN_DIRECTORY` with the selected directory under `logs/dodo_walking`.
Replay restores that run's saved configuration and prints its resolved checkpoint.
`--checkpoint -1` selects the latest checkpoint within that run; explicit numbers
also work. `--run_name` remains a replay-selector alias for older commands.

Supplying any of `--command_vx`, `--command_vy` (m/s), or `--command_yaw` (rad/s)
fixes the command in body axes; omitted axes become zero. With none supplied,
commands are sampled. `--steps` counts policy ticks across resets: 900 ticks at
Go2-W's .02 s policy period means 18 simulated seconds, excluding reset settling.
`--episode_length_s` separately changes the timeout; falls still reset.
Use `python -m robot_gym.scripts.play --task go2w --help` for all Go2-W options.

#### Keyboard replay

Add `--manual_control` to a successful replay command for any registered robot
(Dodo, Go2 or Go2-W). It shows Genesis and a small pygame 2 input window. Install
the optional dependency in your active Genesis environment with
`python -m pip install pygame`; ordinary replay, training and help do not need it.
Keep the input window focused while Genesis remains visible. Losing focus sends
zero commands on the next policy tick; release keys before moving again.

Hold W/S for forward/reverse vx, A/D for left/right vy, and Q/E for
counterclockwise/clockwise yaw. Commands use body +X forward, +Y left, +Z up;
linear speeds are m/s and yaw rate is rad/s. Opposite keys cancel, axes combine,
either Shift multiplies speeds by 0.3, and held Space requests exact zero.
Releasing movement keys requests zero; this asks the policy to stop and does not
force the robot's physical velocity. Grey arrows turn orange while their keys are
held, including when opposite keys cancel or Space overrides them.

Initial limits come from the selected saved/runtime `commands.ranges`, using each
direction's own bound. Tab cycles the highlighted limit axis: vx, vy, yaw. Press
`+` (also `=` or numpad `+`) / `-` (also numpad `-`), or click the panel's `+` / `-`
buttons, to change both directional speed magnitudes for that axis by 0.1 m/s
(vx/vy) or 0.1 rad/s (yaw). Each magnitude stops at zero when decreasing. Increasing
can exceed the training range or enable a direction initially at zero. Changes
apply on the next policy inference and last for this replay session, including
resets; they do not modify the saved configuration.

Manual replay starts at zero, defaults to one environment, and broadcasts to all
environments if `--num_envs` is supplied. Esc or closing either window exits.
It runs until exit unless `--steps` is explicit, retaining the policy timestep and
episode resets. Remove fixed `--command_vx/vy/yaw` arguments and `--headless`.
For the current handoff's selected refinement candidate (native CK1999, saved
configuration and nominal replay dynamics; no historical profile required):

```powershell
$run = (Resolve-Path 'ressources/handoffs/go2w_2026-10-09/runs/go2w_2026-10-08_23-45-24').Path
python -m robot_gym.scripts.play --task go2w --experiment_name go2w --load_run "$run" --checkpoint 1999 --num_envs 1 --seed 1 --rl_device cuda:0 --no_export --manual_control
```

### Export Policy

Replay exports only when `--export` is supplied. The existing exporter embeds the
Actor normalizer once and writes under the selected run:

```bash
logs/<experiment>/<run>/exported/model_<checkpoint>/policy_1.pt
```

`--no_export` remains accepted for compatibility; export is disabled by default.

### Observations

Typical observation vector:

- base linear velocity
- base angular velocity
- projected gravity
- command velocities
- joint positions
- joint velocities
- previous actions

### Reward System

Modular reward design:

- Base rewards
- velocity tracking
- orientation stability
- base height
- smoothness penalties
- Robot-specific rewards
- foot swing clearance
- flat feet
- torso pitch
- hip penalties (avoid clinching legs together)
- survival reward

Implemented in LeggedRobot env and robot specific env (e.g.: DodoEnv)

### Configuration System

Hierarchical config system:

- EnvCfg → simulation + robot
- RewardCfg → reward shaping
- TrainCfg → PPO

Configs auto-instantiate recursively

### Automatic Robot Parsing

- extracts joint names
- resolves paths automatically

Implemented via URDFReader

You can easily use your own URDF robot file for training your own locomotion policy. Just make sure that its consistent with the provided pipeline and that the URDF is optimized. Optimizing can include "simplifying collisions", by using collision-boxes or cylinders instead of the actual meshes. This will reduce training time by a lot.

### Logging

- Weights & Biases integration OR Tensorboard (as given by RSL-RL)
- reward breakdown
- training metrics

When using Weights & Biases, authenticate once. `WANDB_USERNAME` is optional
if your default account entity is appropriate:

```powershell
wandb login
$env:WANDB_USERNAME="YOUR_WANDB_ENTITY"
```

To make the entity persistent for future PowerShell sessions:

```powershell
[Environment]::SetEnvironmentVariable("WANDB_USERNAME", "YOUR_WANDB_ENTITY", "User")
```

---

## Example Commands

```bash
# Full training
python -m robot_gym.scripts.train --task=dodo --num_envs 4096
# Debug run
python -m robot_gym.scripts.train --task=dodo --num_envs 512 --max_iterations 50
# Go2W visual smoke test
python -m robot_gym.scripts.train --task=go2w --experiment_name go2w_first_visual --run_name go2w_spawn_check --num_envs 16 --max_iterations 50
# Play model
python -m robot_gym.scripts.play --task=dodo
```

---

## Future Work

This pipeline is still at an early stage and will be extended from time to time with the following functionalities:

- two-stage training (Teacher–Student Learning with Privileged Information)
- further domain randomization
- curriculum learning
- sim2real (ROS2)

---

## Contact

If you encounter any issues, feel free to contact me:

mail: liam.bansbach@tum.de
