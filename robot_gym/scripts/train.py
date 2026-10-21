"""Train a registered robot through the shared environment and native PPO runner.

Example: python -m robot_gym.scripts.train --task dodo --num_envs 4096 --headless
Use --help for runtime arguments; each registered robot owns its training config.
"""

import importlib.metadata
import sys
import torch
from pathlib import Path
from time import perf_counter

from robot_gym.envs import *  # noqa: F401,F403 - register tasks
from robot_gym.utils import get_args, task_registry
from robot_gym.utils.helpers import update_cfg_from_args
from robot_gym.utils.diagnostics import write_json


def train(args):
    if getattr(args, 'velocity_source', None) is not None:
        raise ValueError("--velocity_source is replay-only; training inputs are selected by --estimator_stage")
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
    from robot_gym.utils.velocity_estimator import enabled
    from robot_gym.utils.estimator_ppo import TrainingBudgetReached, assert_frozen_parent
    if enabled(env.cfg):
        runner.alg.budget_seconds = args.training_seconds
        runner.alg.budget_started = started
        metadata["estimator"] = dict(config=env.cfg.velocity_estimator,
            initial_counters=runner.alg.stage_counts.copy(),
            optimizer_change="none" if runner.alg.stage == "warmup" else f"retain Adam moments; fixed PPO LR {runner.alg.learning_rate}",
            trajectory_continuation="new simulator/RNG episode; learning states restored")
    try:
        runner.learn(num_learning_iterations=training.runner.max_iterations, init_at_random_ep_len=True)
        metadata["status"] = "completed"
    except TrainingBudgetReached as error:
        metadata["status"] = "wall_time_cap"
        print(str(error), flush=True)
        runner.save(str(output / f"model_{runner.current_learning_iteration}.pt"))
    except (Exception, KeyboardInterrupt):
        if enabled(env.cfg):
            runner.save(str(output / f"model_{runner.current_learning_iteration}.pt"))
        raise
    finally:
        metadata.update(env.training_metadata())
        metadata["last_iteration_label"] = runner.current_learning_iteration
        metadata["wall_clock_s"] = perf_counter() - started
        if enabled(env.cfg):
            metadata["estimator"]["final_counters"] = runner.alg.stage_counts.copy()
            if runner.alg.stage == "warmup":
                parent = torch.load(runner.checkpoint_path, map_location='cpu', weights_only=False)
                assert_frozen_parent(parent, runner.alg.save())
                metadata["estimator"]["parent_state_bitwise_unchanged"] = True
        if metadata["status"] not in ("completed", "wall_time_cap"):
            metadata["status"] = "interrupted_or_failed"
        write_json(output / "preparation.json", metadata)
    return runner


if __name__ == "__main__":
    train(get_args())
