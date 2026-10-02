"""Shared CLI, configuration conversion, checkpoint selection and seeding."""

import os
import torch
import numpy as np
import random
import argparse

def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if callable(val):
            continue
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def update_class_from_dict(obj, values):
    """Restore nested config objects; authored dictionaries replace dictionaries."""
    for key, value in values.items():
        if key.startswith("_"):
            raise ValueError(f"Private configuration key is not supported: {key}")
        current = getattr(obj, key, None)
        if isinstance(value, dict) and current is not None and hasattr(current, "__dict__"):
            update_class_from_dict(current, value)
        else:
            setattr(obj, key, value)


def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_load_path(root, load_run=-1, checkpoint=-1):
    if not os.path.isdir(root) and not (load_run is not None and os.path.isabs(str(load_run))):
        raise ValueError(f"No runs in this directory: {root}")

    if str(load_run) == "-1":
        load_run = -1
    if str(checkpoint) == "-1":
        checkpoint = -1

    # ------------------------------------------------------------------
    # Select run directory
    # ------------------------------------------------------------------
    if load_run == -1:
        runs = [
            os.path.join(root, run)
            for run in os.listdir(root)
            if run != "exported" and os.path.isdir(os.path.join(root, run)) 
        ]

        if len(runs) == 0:
            raise ValueError(f"No runs in this directory: {root}")

        # Robust across month changes: use filesystem modification time
        runs.sort(key=os.path.getmtime)
        load_run = runs[-1]
    else:
        load_run = os.path.join(root, load_run)

    if not os.path.isdir(load_run):
        raise ValueError(f"Run directory does not exist: {load_run}")

    # ------------------------------------------------------------------
    # Select checkpoint
    # ------------------------------------------------------------------
    if checkpoint == -1:
        models = [
            file
            for file in os.listdir(load_run)
            if file.startswith("model") and file.endswith(".pt")
        ]

        if len(models) == 0:
            raise ValueError(f"No model checkpoints found in: {load_run}")

        # Prefer model_final.pt if it exists
        if "model_final.pt" in models:
            model = "model_final.pt"
        else:
            def model_iteration(filename):
                stem = os.path.splitext(filename)[0]  # model_100 -> model_100
                try:
                    return int(stem.split("_")[-1])
                except ValueError:
                    return -1

            models.sort(key=model_iteration)
            model = models[-1]
    else:
        model = f"model_{checkpoint}.pt"

    load_path = os.path.join(load_run, model)

    if not os.path.isfile(load_path):
        raise ValueError(f"Checkpoint does not exist: {load_path}")

    return load_path

def update_cfg_from_args(env_cfg, cfg_train, args):
    from robot_gym.utils.task_registry import task_registry
    if env_cfg is not None and env_cfg.asset.name != task_registry.env_cfgs[args.task].asset.name:
        raise ValueError(f"Configuration asset {env_cfg.asset.name!r} conflicts with task {args.task!r}")
    task_registry.get_task_class(args.task).configure(env_cfg, cfg_train, args)
    # seed
    if env_cfg is not None:
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        if getattr(args, "logger", None):
            cfg_train.runner.logger = args.logger
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint

    return env_cfg, cfg_train

def get_args():
    parser = argparse.ArgumentParser(description="RL Policy")

    custom_parameters = [
        {"name": "--no_export", "action": "store_true", "help": "Compatibility: keep replay export disabled (the default)"},
        {"name": "--training_diagnostics", "action": "store_true", "help": "Opt-in RSL-RL and unclipped reward JSONL diagnostics"},
        {"name": "--output", "default": "evaluation", "help": "Evaluation output directory"},
        {"name": "--logger", "choices": ["tensorboard", "wandb"], "help": "Override training logger"},
        {"name": "--steps", "type": int, "default": 1000, "help": "Replay policy ticks; 900 at dt=.02 is 18 s, excluding reset settling; spans episode resets"},
        {"name": "--episode_length_s", "type": float, "help": "Replay episode timeout in seconds; session length is still --steps"},
        {"name": "--export", "action": "store_true", "help": "Opt-in replay export under the selected run/exported/model_<checkpoint>"},
        {"name": "--command_vx", "type": float, "help": "Fixed play vx in m/s; omitted axes default to zero"},
        {"name": "--command_vy", "type": float, "help": "Fixed play vy in m/s; omitted axes default to zero"},
        {"name": "--command_yaw", "type": float, "help": "Fixed play yaw rate in rad/s; omitted axes default to zero"},
        {"name": "--task", "type": str, "default": "dodo", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--resume", "action": "store_true", "default": False, "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str, "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str, "help": "Training output label; compatibility replay selector when --load_run is absent"},
        {"name": "--load_run", "type": str, "help": "Saved run directory for replay or explicit training resume; -1 selects the latest run"},
        {"name": "--checkpoint", "type": int, "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": "Device used by the RL algorithm, (cpu, cuda, cuda:0, cuda:1 etc..)"},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
    ]

    for param in custom_parameters:
        param = param.copy()
        name = param.pop("name")
        parser.add_argument(name, **param)

    from robot_gym.utils.task_registry import task_registry
    selection = argparse.ArgumentParser(add_help=False)
    selection.add_argument("--task", default="dodo")
    task_name = selection.parse_known_args()[0].task
    task_registry.get_task_class(task_name).add_arguments(parser)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.episode_length_s is not None and (not np.isfinite(args.episode_length_s) or args.episode_length_s <= 0):
        parser.error("--episode_length_s must be finite and positive")
    if args.export and args.no_export:
        parser.error("--export and --no_export conflict")

    args.sim_device = args.rl_device

    if ":" in args.rl_device:
        args.sim_device_type, device_id = args.rl_device.split(":")
        args.sim_device_id = int(device_id)
    else:
        args.sim_device_type = args.rl_device
        args.sim_device_id = 0

    return args
