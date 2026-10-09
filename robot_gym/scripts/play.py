"""Replay a selected run's saved task configuration and checkpoint.

From the repository root in the installed Genesis environment::

    python -m robot_gym.scripts.play --task dodo --experiment_name dodo_walking --load_run RUN_DIRECTORY --checkpoint -1

Replace RUN_DIRECTORY with a directory under logs/<experiment_name>. Latest (-1)
is resolved and printed once; explicit numbers work too. --load_run selects replay
input; --run_name is a compatibility alias when --load_run is absent (and labels
training output in train.py). No historical profile is needed.

No command flags retains task sampling. Any --command_vx (m/s), --command_vy
(m/s), or --command_yaw (rad/s) fixes the body command, with omitted axes zero.
--command_vx 0 requests exact stand. --steps counts policy ticks: 900 at .02 s is
18 simulated seconds, excluding reset settling/startup. Ticks span episode resets;
--episode_length_s changes the episode timeout. Falls still reset.

--manual_control opens a focused keyboard input window and starts at zero command.
It defaults to one environment and runs until Esc/window closure unless --steps is
explicit. Hold W/S, A/D, Q/E; Shift slows to 30%, Space requests a policy stop.

--export opts into export under the selected run. --no_export remains compatible
with the default. See --help for the full CLI and Go2-W docs for concrete examples.
"""

import torch
import genesis as gs

from robot_gym.envs import *  # noqa: F401,F403 - register tasks
from robot_gym.utils import get_args, task_registry
from robot_gym.utils.replay import configure_fixed_command, configure_nominal


def should_export_policy(args):
    return args.export and not args.no_export and not getattr(args, "zero_command_brake", False)


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
            keyboard = KeyboardInput()
            print(f"Manual limits [vx m/s, vy m/s, yaw rad/s]: {limits}", flush=True)
        env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg,
                                        initial_command=(0.0, 0.0, 0.0) if manual else None)
        if manual:
            env.sim.viewer.realtime_factor = 1.0  # Reuse Genesis's physics-step pacing.
        configure_fixed_command(env, args)
        if getattr(args, "zero_command_brake", False):
            env.enable_zero_command_brake()
        env.reset()
        runner, _ = task_registry.make_alg_runner(env, args=args, train_cfg=train_cfg, save_config=False)
        policy = runner.get_inference_policy(device=env.device)
        env.compute_observations()
        obs = env.get_observations()
        session = "until Esc or window closure" if steps is None else f"{steps} policy ticks ({steps * env.dt:g} s)"
        print(f"Session: {session}; "
              f"episode timeout: {env.max_episode_length * env.dt:g} s. Falls still reset.", flush=True)
        if should_export_policy(args):
            from robot_gym.utils.export import export_policy
            output = checkpoint.parent / "exported" / checkpoint.stem
            export_policy(runner, env, output)
            print(f"Exported policy: {output}", flush=True)
        with torch.no_grad():
            tick = 0
            while steps is None or tick < steps:
                if not args.headless and not env.sim.viewer.is_alive():
                    break
                if keyboard is not None:
                    axes, quit_requested = keyboard.poll()
                    command = scale_axes(axes, limits)
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
