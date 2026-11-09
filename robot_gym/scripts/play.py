"""Replay a saved policy with sampled, fixed or keyboard commands."""

import torch
import genesis as gs

from robot_gym.envs import *  # noqa: F401,F403 - register tasks
from robot_gym.utils import get_args, task_registry
from robot_gym.utils.replay import configure_fixed_command, configure_nominal


def should_export_policy(args):
    return args.export and not args.no_export


def play(args):
    manual = getattr(args, "manual_control", False)
    if manual and args.headless:
        raise ValueError("--manual_control needs the Genesis viewer; remove --headless")
    if manual and any(getattr(args, f"command_{axis}", None) is not None for axis in ("vx", "vy", "yaw")):
        raise ValueError("--manual_control cannot be combined with fixed --command_vx/vy/yaw")
    cfg, train_cfg, checkpoint = task_registry.resolve_replay(args)
    configure_nominal(cfg, args)
    cfg.env.num_envs = args.num_envs if args.num_envs is not None else (1 if manual else min(cfg.env.num_envs, 5))
    cfg.env.play_mode = True
    cfg.viewer.visualize_foot_contacts = False
    cfg.viewer.visualize_velocity_arrows = True
    cfg.viewer.ref_env = list(range(cfg.env.num_envs))
    cfg.viewer.print_debug_velocities = False
    env = None
    keyboard = None
    steps = None if manual and not args._steps_explicit else args.steps
    try:
        if manual:
            from robot_gym.utils.manual_control import KeyboardInput, scale_axes
            # Capture the effective saved/runtime ranges before command ownership changes.
            limits = tuple(tuple(getattr(cfg.commands.ranges, name))
                           for name in ("lin_vel_x", "lin_vel_y", "ang_vel_yaw"))
            keyboard = KeyboardInput(limits)
            print(f"Manual limits [vx m/s, vy m/s, yaw rad/s]: {limits}", flush=True)
        env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg,
                                        initial_command=(0.0, 0.0, 0.0) if manual else None)
        if manual:
            env.sim.viewer.realtime_factor = 1.0  # Reuse Genesis's physics-step pacing.
        configure_fixed_command(env, args)
        env.reset()
        runner, _ = task_registry.make_alg_runner(env, args=args, train_cfg=train_cfg, save_config=False)
        from robot_gym.utils.velocity_estimator import inference_policy
        policy = inference_policy(runner, env, getattr(args, "velocity_source", None))
        print(f"Replay velocity source: {env.velocity_source}. Viewer velocity arrows show simulator truth.", flush=True)
        env.compute_observations()
        obs = env.get_observations()
        session = "until Esc or window closure" if steps is None else f"{steps} policy ticks ({steps * env.dt:g} s)"
        print(f"Session: {session}; "
              f"episode timeout: {env.max_episode_length * env.dt:g} s. Falls still reset.", flush=True)
        if should_export_policy(args):
            from robot_gym.utils.export import export_policy
            from pathlib import Path
            output = Path(args.export_dir) if args.export_dir else checkpoint.parent / "exported" / checkpoint.stem
            export_policy(runner, env, output, onnx=args.onnx, parity_histories=args.parity_histories)
            print(f"Exported policy: {output}", flush=True)
        with torch.no_grad():
            tick = 0
            while steps is None or tick < steps:
                if not args.headless and not env.sim.viewer.is_alive():
                    break
                if keyboard is not None:
                    axes, quit_requested = keyboard.poll()
                    command = scale_axes(axes, keyboard.limits)
                    env.set_fixed_command((0.0, 0.0, 0.0) if quit_requested else command)
                    if quit_requested:
                        break
                    # The setter rebuilds command features without advancing phase/history.
                    obs = env.get_observations()
                    keyboard.draw(command)
                obs, _, _, _ = env.step(policy(obs).detach())
                tick += 1
    except KeyboardInterrupt:
        print("Replay interrupted.", flush=True)
    except gs.GenesisException as error:
        if "Viewer closed" not in str(error):
            raise
    finally:
        if keyboard is not None:
            keyboard.close()
        from robot_gym.envs.base.base_task import BaseTask
        if BaseTask._gs_initialized:
            gs.destroy()
            BaseTask._gs_initialized = False


if __name__ == "__main__":
    play(get_args())
