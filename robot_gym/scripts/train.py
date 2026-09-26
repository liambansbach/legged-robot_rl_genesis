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


def prepare_go2w_continuation(args, env_cfg, train_cfg):
    """Check the explicit saved parent before constructing the simulator."""
    from pathlib import Path
    from robot_gym import ROBOT_GYM_ROOT_DIR
    from robot_gym.utils.helpers import get_load_path, class_to_dict
    from robot_gym.utils.urdf_reader import URDFReader
    from robot_gym.utils.diagnostics import check_training_continuation, sha256

    if args.load_run in (None, "-1") or args.checkpoint is None or args.checkpoint < 0:
        raise ValueError(
            "Go2-W continuation requires explicit --load_run and --checkpoint"
        )
    if not args.run_name or any(c in args.run_name for c in "/\\"):
        raise ValueError(
            "Go2-W continuation requires a new --run_name (a folder name, not a path)"
        )
    root = Path(ROBOT_GYM_ROOT_DIR) / "logs" / train_cfg.runner.experiment_name
    checkpoint = Path(get_load_path(root, args.load_run, args.checkpoint)).resolve()
    config = checkpoint.with_name("config.yaml")
    if args.reference_config and Path(args.reference_config).resolve() != config:
        raise ValueError(
            "Training continuation requires the saved config beside its checkpoint"
        )
    env_cfg.asset.joint_names = URDFReader(env_cfg.asset.robot_file).joint_names
    reference = check_training_continuation(
        config, class_to_dict(env_cfg), class_to_dict(train_cfg), args.tracking_sigma_x
    )
    saved_git = checkpoint.parent / "git" / f"{Path(ROBOT_GYM_ROOT_DIR).name}.diff"
    source_snapshot = None
    if saved_git.is_file():
        # Existing RSL-RL git snapshot includes the training HEAD and dirty patch.
        source_snapshot = {
            "path": str(saved_git),
            "sha256": sha256(saved_git),
            "head": saved_git.read_text().splitlines()[1],
        }
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "saved_config": reference,
        "saved_source_snapshot": source_snapshot,
    }


def train(args):
    from pathlib import Path
    from robot_gym import ROBOT_GYM_ROOT_DIR
    from robot_gym.utils.helpers import update_cfg_from_args
    from robot_gym.utils.diagnostics import source_identity, write_json, sha256
    from robot_gym.utils.training_diagnostics import (
        TrainingDiagnostics,
        verify_resume_state,
    )

    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    update_cfg_from_args(env_cfg, train_cfg, args)
    env_cfg.seed = train_cfg.seed
    if args.training_diagnostics:
        env_cfg.env.record_command_families = True
    parent = None
    if args.task == "go2w" and train_cfg.runner.resume:
        parent = prepare_go2w_continuation(args, env_cfg, train_cfg)
    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        train_cfg=train_cfg,
        args=args,
    )
    observer = None
    if args.training_diagnostics:
        observer = TrainingDiagnostics(
            ppo_runner, env, Path(ppo_runner.logger.log_dir) / "diagnostics.jsonl"
        )
    if parent:
        if Path(ppo_runner.checkpoint_path).resolve() != Path(parent["checkpoint"]):
            raise ValueError("Runner loaded a different continuation parent")
        loaded = verify_resume_state(ppo_runner, parent["checkpoint"])
        out = Path(ppo_runner.logger.log_dir).resolve()
        metadata = {
            "parent": parent,
            "loaded_state": loaded,
            "source": source_identity(ROBOT_GYM_ROOT_DIR),
            "seed": train_cfg.seed,
            "tracking_sigma_x": env_cfg.rewards.tracking_sigma_x,
            "planned_additional_updates": train_cfg.runner.max_iterations,
            "completed_additional_updates": 0,
            "status": "verified_before_update",
            "output_directory": str(out),
            "resolved_config_sha256": sha256(out / "config.yaml"),
            "initialization": "Matched new seeded simulator; checkpoint does not restore historical simulator/RNG state",
            "iteration_labels": "RSL-RL 5.5.1 starts at saved iter; N additional updates end at saved iter + N - 1",
        }
        write_json(out / "continuation.json", metadata)
        print(
            f"Verified original learning state: {parent['checkpoint']}; LR={loaded['loaded_learning_rate']}",
            flush=True,
        )
    completed = False
    try:
        ppo_runner.learn(
            num_learning_iterations=train_cfg.runner.max_iterations,
            init_at_random_ep_len=True,
        )
        completed = True
    finally:
        if parent:
            metadata["status"] = "completed" if completed else "failed"
            metadata["completed_additional_updates"] = (
                train_cfg.runner.max_iterations
                if completed
                else observer.iteration - loaded["source_iteration"]
                if observer
                else None
            )
            metadata["last_iteration_label"] = ppo_runner.current_learning_iteration
            metadata["final_checkpoint"] = (
                str(out / f"model_{ppo_runner.current_learning_iteration}.pt")
                if completed
                else None
            )
            write_json(out / "continuation.json", metadata)
    return ppo_runner


if __name__ == "__main__":
    args = get_args()
    train(args)
