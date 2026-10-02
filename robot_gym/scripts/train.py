"""Train a registered robot through the shared environment and native PPO runner.

Example: python -m robot_gym.scripts.train --task dodo --num_envs 4096 --headless
Use --help for all arguments and robot documentation for opt-in recipes.
"""

import importlib.metadata
import sys
from pathlib import Path
from time import perf_counter

from robot_gym.envs import *  # noqa: F401,F403 - register tasks
from robot_gym.utils import get_args, task_registry
from robot_gym.utils.helpers import update_cfg_from_args
from robot_gym.utils.diagnostics import write_json


def train(args):
    started = perf_counter()
    cfg, training = task_registry.get_cfgs(args.task)
    update_cfg_from_args(cfg, training, args)
    task_registry.get_task_class(args.task).validate_training(args, cfg, training)
    cfg.seed = training.seed
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    runner, training = task_registry.make_alg_runner(env, args=args, train_cfg=training)
    output = Path(runner.logger.log_dir)
    if not training.runner.resume and (runner.checkpoint_path is not None or runner.alg.optimizer.state):
        raise ValueError("Fresh training unexpectedly loaded learning state")
    metadata = {
        "task": args.task, "seed": training.seed, "python_executable": sys.executable,
        "packages": {name: importlib.metadata.version(name) for name in ("genesis-world", "torch", "rsl-rl-lib", "tensordict")},
        "initialization": "checkpoint" if training.runner.resume else "fresh actor, critic, normalizers, optimizer and curriculum",
        "checkpoint": runner.checkpoint_path, "planned_updates": training.runner.max_iterations,
        "initial_learning_rate": runner.alg.learning_rate, "entropy_coef": runner.alg.entropy_coef,
        **env.training_metadata(), "status": "before_updates",
    }
    write_json(output / "preparation.json", metadata)
    try:
        runner.learn(num_learning_iterations=training.runner.max_iterations, init_at_random_ep_len=True)
        metadata["status"] = "completed"
    finally:
        metadata.update(env.training_metadata())
        metadata["last_iteration_label"] = runner.current_learning_iteration
        metadata["wall_clock_s"] = perf_counter() - started
        if metadata["status"] != "completed":
            metadata["status"] = "interrupted_or_failed"
        write_json(output / "preparation.json", metadata)
    return runner


# Minimal historical imports for existing diagnostic tools.
from robot_gym.envs.go2w.training import prepare_go2w_continuation, validate_fresh_transfer  # noqa: E402,F401

if __name__ == "__main__":
    train(get_args())
