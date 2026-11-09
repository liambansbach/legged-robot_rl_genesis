# Learning to walk and roll in Genesis

Dodo balances on two legs, Unitree Go2 walks on four, and Go2-W has a wheel at the end of each leg. Ask all three to move forward at 0.3 m/s and each has a different balance problem to solve. This project uses the Genesis physics simulator and RSL-RL's implementation of Proximal Policy Optimization (PPO) to train their controllers. All three trained policies are included, so you can drive a robot before training one yourself.

[Features](#features) · [Install](#install) · [Try a policy](#replay-a-published-policy) · [How it works](#how-a-command-becomes-motion) · [Train your own](#train-your-own-policy)

## Features

- **GPU-parallel PPO training** in Genesis, with TensorBoard or Weights & Biases logging.
- **Flat and procedural terrain** with configurable randomization of friction, base mass and joint gains.
- **Custom robot tasks** with URDF assets, joint settings and robot-specific rewards.
- **Policy replay** with keyboard control, fixed commands or sampled velocities.
- **Body velocity estimation** from sensor and controller history, with a trained estimator included for Go2-W.
- **TorchScript and ONNX export** with observation normalization and action limits built into the exported policy.

## Supported robots

| Robot | Type | Actor inputs → actions | Published run |
| --- | --- | --- | --- |
| Dodo | Biped | 36 → 8 | [Dodo walking](ressources/pretrained/dodo/dodo_walking/README.md) |
| Go2 | Quadruped | 48 → 12 | [Go2 walking](ressources/pretrained/go2/go2_walking/README.md) |
| Go2-W | Wheeled quadruped | 58 → 16 | [Go2-W locomotion](ressources/pretrained/go2w/go2w_locomotion/README.md) |

**Dodo in simulation**

https://github.com/user-attachments/assets/8276639f-ffef-4358-a578-7138b7d5190d

https://github.com/user-attachments/assets/765ad1f3-c007-4130-8f2c-b92aab3ce22f

**Go2 in simulation**

https://github.com/user-attachments/assets/afaef573-847f-4afc-a96d-eb92982bcb41

https://github.com/user-attachments/assets/c59d69cd-e617-4aad-8bf4-92024ff59d88

**Go2-W in simulation**

https://github.com/user-attachments/assets/44d97ef7-f679-4b17-906f-78b17df7704e

<details>
<summary>See the training figures</summary>

**Dodo training**

<img width="1658" height="715" alt="Dodo training results, first figure" src="https://github.com/user-attachments/assets/655c3ed9-ef01-461f-8c61-9e362383d8f7" />
<img width="2469" height="1260" alt="Dodo training results, second figure" src="https://github.com/user-attachments/assets/0e6ed3f3-12a9-408f-8176-a8bbe92f4dab" />

**Go2 training**

<img width="1657" height="677" alt="Go2 training results, first figure" src="https://github.com/user-attachments/assets/d4dcaaed-5765-42fa-9461-c76f04ecce89" />
<img width="2469" height="1258" alt="Go2 training results, second figure" src="https://github.com/user-attachments/assets/2ecb48ee-5110-4de5-927c-b43a35b0307a" />

</details>

### Training hardware

The published Dodo run completed 1,500 PPO updates with 4,096 parallel environments in about 45 minutes on this machine.

- **CPU** — Intel Core i5-13600K (14 cores, 20 threads)
- **GPU** — NVIDIA GeForce RTX 4070 Ti
- **Memory** — 32 GB DDR5 at 6000 MT/s

## Install

You'll need Git, Conda and an NVIDIA driver compatible with the CUDA 13.0 build of PyTorch.

**Tested software**

- Python 3.11.14
- PyTorch 2.9.0+cu130
- Genesis 1.4.1
- RSL-RL 5.5.1

### 1. Clone the repository

```sh
git clone https://github.com/liambansbach/legged-robot_rl_genesis.git
cd legged-robot_rl_genesis
```

If you're reading this guide on the `testing` branch, run `git switch testing` in the clone.

### 2. Create the Conda environment

```sh
conda env create -f conda_env.yaml
conda activate genesis-gpu
```

## Replay a published policy

From the repository root, set the path to the pretrained runs once in your shell.

| Shell | Command |
| --- | --- |
| Bash | `pretrained="$(pwd)/ressources/pretrained"` |
| PowerShell | `$pretrained = (Resolve-Path 'ressources/pretrained').Path` |

**First walk.** Open the viewer and ask Dodo to walk forward at a constant velocity of 0.3 m/s. The window closes after 250 policy steps, or five simulated seconds.

```sh
python -m robot_gym.scripts.play --task dodo --load_run "$pretrained/dodo/dodo_walking" --checkpoint 1499 --num_envs 1 --command_vx 0.3 --steps 250
```

**Take the controls.** Install the optional keyboard dependency with `python -m pip install pygame`, then run one of these commands.

```sh
python -m robot_gym.scripts.play --task dodo --load_run "$pretrained/dodo/dodo_walking" --checkpoint 1499 --manual_control

python -m robot_gym.scripts.play --task go2 --load_run "$pretrained/go2/go2_walking" --checkpoint 1499 --manual_control

python -m robot_gym.scripts.play --task go2w --load_run "$pretrained/go2w/go2w_locomotion/checkpoint" --checkpoint 2197 --velocity_source estimated --manual_control
```

Click the small keyboard window to give it focus. You can combine movement keys.

- **W / S** — move forward or backward.
- **A / D** — move left or right.
- **Q / E** — turn left or right.
- **Shift** — hold for 30% speed.
- **Tab** — select a speed limit, then adjust it with **+** or **−**.
- **Esc** — end the replay.

Release the keys, hold **Space** or switch away from the input window to request zero velocity.

Omit the command and keyboard flags to let the task randomly sample velocities. Add `--headless` for replay without a viewer.

## How a command becomes motion

When you hold **W**, you set a desired forward speed. Dodo's policy reads that command together with the robot's state and sets a target angle for each joint. Genesis runs the joint controllers for four physics steps before the policy makes its next decision.

<details>
<summary>Inside Dodo's walking policy · observations, actions and rewards</summary>

**Observations** — 36 values in the order the actor receives them

- 3 linear velocities and 3 angular velocities, expressed in the robot's body frame
- 3 components of gravity in that frame, which tell the policy how the body is tilted
- 3 commands for forward speed, sideways speed and yaw rate
- 8 joint angles relative to the reference pose, 8 joint velocities and the 8 previous action values

**Actions** — 8 joint targets updated every 20 ms

- The outputs follow the hip, upper leg, lower leg and foot joints on the right, then the same four joints on the left.
- A zero action requests the reference angle. On the published policy, a clipped action of +1 shifts that joint's target by +0.25 radians.

**Rewards** — what shaped the published gait during training

- Track the requested forward and sideways speeds and the requested turning rate.
- Lift each swinging foot clear of the ground, keep the feet level and maintain the target body height and forward torso pitch.
- Use smooth actions and limit torque. Falling incurs a penalty.

[Saved configuration for the published run](ressources/pretrained/dodo/dodo_walking/config.yaml)

</details>

Go2-W can roll while repositioning its legs. Its policy uses a gait phase and issues position targets for twelve leg joints and speed targets for four wheels. The published ONNX model combines the velocity estimator and actor, so it can turn five frames of sensor and controller data into the next action. [Go2-W model, checks and simulation results](ressources/pretrained/go2w/go2w_locomotion/README.md)

## Train your own policy

Start a short Dodo run, then watch the latest checkpoint from that experiment.

```sh
python -m robot_gym.scripts.train --task dodo --experiment_name dodo_first_run --num_envs 512 --max_iterations 100 --logger tensorboard --headless

python -m robot_gym.scripts.play --task dodo --experiment_name dodo_first_run --load_run -1 --checkpoint -1 --num_envs 1 --command_vx 0.3
```

To follow learning, activate `genesis-gpu` in a second terminal at the repository root and run `tensorboard --logdir logs/dodo_first_run`. Weights & Biases is also available with `--logger wandb`.

The task config sets the terrain, rewards and training conditions. For example, Dodo trains on a plane with randomized friction and joint gains. Each run saves its resolved `config.yaml` beside its checkpoints. To continue training, pass `--resume`, the run directory with `--load_run` and its checkpoint number with `--checkpoint`. Replay also restores the saved robot model. For the published Dodo this is `dodo_daimao.urdf`. Fresh runs use `dodo_daimao2.urdf`.

### Export a policy

This writes Dodo's bounded TorchScript actor, including its trained observation normalizer, to `evaluation/dodo_export/policy_1.pt`. Choose a fresh export directory for each run.

```sh
python -m robot_gym.scripts.play --task dodo --load_run "$pretrained/dodo/dodo_walking" --checkpoint 1499 --num_envs 1 --steps 1 --headless --export --export_dir evaluation/dodo_export
```

For ONNX, install the extras with `python -m pip install -r requirements-export.txt` and add `--onnx`. Exporting a combined estimator and actor to ONNX also requires recorded input histories for the parity check. [Estimator and export guide](docs/velocity_estimator.md#compare-and-export)

## Explore the repository

```text
legged-robot_rl_genesis/
├── robot_gym/
│   ├── envs/
│   │   ├── __init__.py        # Registers the three tasks
│   │   ├── base/              # Shared physics step, observations and resets
│   │   ├── dodo/              # Biped config and rewards
│   │   ├── go2/               # Quadruped config
│   │   └── go2w/              # Wheeled task, gait phase and evaluation
│   ├── scripts/
│   │   ├── train.py           # Train or resume a policy
│   │   ├── play.py            # Replay, keyboard control and export
│   │   └── evaluate.py        # Run evaluation on a saved policy
│   └── utils/                 # Run loading, velocity estimation and export
├── ressources/
│   ├── robots/                # Robot descriptions and meshes
│   └── pretrained/            # Published checkpoints and saved configs
├── docs/                      # Task and estimator guides
├── tests/                     # Physics, replay and policy regression checks
├── conda_env.yaml             # Training and replay environment
└── requirements-export.txt    # Optional ONNX dependencies
```

To add a robot, start with its URDF and meshes, define its task config and register a name. That name works with the same train and play commands. [Add a robot](docs/velocity_estimator.md#another-robot) · [Go2-W training and evaluation](docs/go2w_training.md)

The task, config and registry structure draws on [Unitree RL Gym](https://github.com/unitreerobotics/unitree_rl_gym/tree/main). For questions about the project, contact [Liam Bansbach](mailto:liam.bansbach@tum.de).
