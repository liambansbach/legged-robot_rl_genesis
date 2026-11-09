"""Matched truth/estimated replay for canonical legged tasks."""

import copy
from pathlib import Path

import genesis as gs
import numpy as np
import torch

from robot_gym.utils.diagnostics import sha256, write_json
from robot_gym.utils.helpers import set_seed
from robot_gym.utils.replay import configure_nominal
from robot_gym.utils.task_registry import task_registry
from robot_gym.utils.velocity_estimator import enabled, inference_policy


def command_sequence(ranges):
    """Stand, then half of each nonzero signed limit in body x, y and yaw."""
    commands = [(0., 0., 0.)]
    for axis, name in enumerate(("lin_vel_x", "lin_vel_y", "ang_vel_yaw")):
        for limit in getattr(ranges, name):
            if limit:
                command = [0., 0., 0.]
                command[axis] = float(limit) / 2
                commands.append(tuple(command))
    return [*commands, (0., 0., 0.)]


def collect(env, runner, source, commands, path):
    """Record pre-action estimator samples and the corresponding pre-reset outcomes."""
    env.set_fixed_command(commands[0])
    env.reset()
    policy = inference_policy(runner, env, source)
    hold = max(1, round(.5 / env.dt))
    samples = {name: [] for name in ("histories", "reset", "reset_frames", "estimate_error",
                                      "tracking_error", "actions")}
    fallen = False
    with torch.inference_mode():
        for tick in range(hold * len(commands)):
            if tick % hold == 0:
                env.set_fixed_command(commands[tick // hold])
            obs = env.get_observations()
            history = obs["estimator_history"]
            samples["histories"].append(history.clone())
            samples["reset"].append(env.sensor_history.age <= 1)
            samples["reset_frames"].append(env.sensor_history.initial_frame.clone())
            actions = policy(obs)
            estimate = env.estimated_velocity if source == "estimated" else runner.alg.estimator(history)
            samples["estimate_error"].append(estimate - obs["velocity_target"])
            env.step(actions)
            state = env.transition_state
            actual = torch.cat((state["base_lin_vel"][:, :2], state["base_ang_vel"][:, 2:3]), dim=1)
            samples["tracking_error"].append(actual - state["commands"][:, :3])
            samples["actions"].append(state["actions"])
            if state["reset_buf"].any():
                fallen = bool(state["fallen"].any())
                break
    arrays = {name: torch.stack(rows).cpu().numpy() for name, rows in samples.items()}
    history = arrays.pop("histories")
    np.savez_compressed(path, histories=history.reshape(-1, *history.shape[2:]), **arrays)
    return dict(samples=len(history), fallen=fallen,
                estimator_rmse_m_s=np.sqrt(np.mean(arrays["estimate_error"] ** 2, axis=(0, 1))).tolist(),
                tracking_rmse_vx_vy_yaw=np.sqrt(np.mean(arrays["tracking_error"] ** 2, axis=(0, 1))).tolist(),
                saturation_fraction=float((np.abs(arrays["actions"]) >= env.cfg.normalization.clip_actions - 1e-6).mean()))


def evaluate_estimator(args):
    if args.task == "go2w":
        from robot_gym.envs.go2w.evaluation.estimator import evaluate_estimator as go2w_panel
        return go2w_panel(args)
    if args.frozen_checkpoint:
        raise ValueError("--frozen_checkpoint selects the Go2-W three-way panel")
    cfg, training, checkpoint = task_registry.resolve_replay(args)
    if not enabled(cfg):
        raise ValueError("An estimator comparison requires a full actor+estimator checkpoint")
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Refusing to overwrite comparison: {out}")
    commands = command_sequence(cfg.commands.ranges)
    configure_nominal(cfg, args)
    cfg.env.capture_transitions = cfg.env.play_mode = True
    args.num_envs = 1
    out.mkdir(parents=True, exist_ok=True)
    try:
        env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
        runner, _ = task_registry.make_alg_runner(env, args=args, train_cfg=training, save_config=False)
        candidate = copy.deepcopy(runner.alg.actor.state_dict())
        if args.comparison_checkpoint:
            parent = torch.load(args.comparison_checkpoint, map_location=env.device, weights_only=False)
            runner.alg.actor.load_state_dict(parent["actor_state_dict"])
        elif runner.alg.stage_counts["finetune_updates"]:
            raise ValueError("Supply the parent --comparison_checkpoint for a fine-tuned actor")
        definition = dict(task=args.task, commands=commands, policy_dt=env.dt, hold_ticks=max(1, round(.5/env.dt)),
                          seed=cfg.seed, action_bound=cfg.normalization.clip_actions, checkpoint_sha256=sha256(checkpoint),
                          source_config=args.resolved_config_path, source_config_sha256=sha256(args.resolved_config_path))
        if args.comparison_checkpoint:
            definition["parent_checkpoint_sha256"] = sha256(args.comparison_checkpoint)
        results = {}
        for source in ("truth", "estimated"):
            set_seed(cfg.seed)
            if source == "estimated":
                runner.alg.actor.load_state_dict(candidate)
            results[source] = collect(env, runner, source, commands, out / f"nominal_{source}.npz")
        report = dict(definition=definition, comparisons=results,
                      qualification="Descriptive nominal simulation comparison through the first episode.")
        write_json(out / "metrics.json", report)
        return report
    finally:
        gs.destroy()
        from robot_gym.envs.base.base_task import BaseTask
        BaseTask._gs_initialized = False
