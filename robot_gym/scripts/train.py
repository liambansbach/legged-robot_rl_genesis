from robot_gym.envs import *  # noqa: F401,F403 - registers tasks
from robot_gym.utils import get_args, task_registry

"""
Example training command (command line call) with all arguments specified:

    PowerShell:
    python -m robot_gym.scripts.train `
        --task dodo `
        --experiment_name daimao_walking `
        --run_name run_01 `
        --num_envs 4096 `
        --max_iterations 1500 `
        --seed 1 `
        --rl_device cuda:0 `
        --headless

    To resume from the latest checkpoint:
    python -m robot_gym.scripts.train `
        --task dodo `
        --experiment_name daimao_walking `
        --resume `
        --load_run -1 `
        --checkpoint -1 `
        --rl_device cuda:0 `
        --headless

    --task: Task name defined in task_registry envs/__init__.py
    --experiment_name: Name of the experiment, used for logging and wandb run name.
    --run_name: Name of the run. Overrides config file if provided.
    --num_envs: Number of parallel environments to use for training.
    --max_iterations: Maximum number of training iterations.
    --seed: Random seed for reproducibility.
    --rl_device: Device used by the RL algorithm (cpu, cuda, cuda:0, etc.).
    --headless: Force display off at all times (no rendering, faster training).
    --resume: Resume training from a checkpoint.
    --load_run: Name of the run to load when resume=True. If -1: load the latest run.
    --checkpoint: Saved model checkpoint number. If -1: load the latest checkpoint.

Not all arguments are required. A simple call could look like this:
    python -m robot_gym.scripts.train --task dodo --experiment_name dodo_walking_test --num_envs 4096 --max_iterations 1000 --headless
"""

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
    )


    if args.training_diagnostics:
        from pathlib import Path
        from robot_gym.utils.training_diagnostics import TrainingDiagnostics
        TrainingDiagnostics(ppo_runner, env, Path(ppo_runner.logger.log_dir) / "diagnostics.jsonl")

    ppo_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )


if __name__ == "__main__":
    args = get_args()
    train(args)
