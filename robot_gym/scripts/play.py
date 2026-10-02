"""Replay a selected run's saved task configuration and checkpoint.

From the repository root in the installed Genesis environment::

    python -m robot_gym.scripts.play --task dodo --experiment_name dodo_walking --load_run RUN_DIRECTORY --checkpoint -1

Replace RUN_DIRECTORY with a directory under logs/<experiment_name>. Latest (-1)
is resolved and printed once; explicit numbers work too. --run_name labels
training output; --load_run selects replay input. No historical profile is needed.

No command flags retains task sampling. Any --command_vx (m/s), --command_vy
(m/s), or --command_yaw (rad/s) fixes the body command, with omitted axes zero.
--command_vx 0 requests exact stand. --steps counts policy ticks: 900 at .02 s is
18 simulated seconds, excluding reset settling/startup. Ticks span episode resets;
--episode_length_s changes the episode timeout. Falls still reset.

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
    cfg, train_cfg, checkpoint = task_registry.resolve_replay(args)
    configure_nominal(cfg, args)
    cfg.env.num_envs = args.num_envs or min(cfg.env.num_envs, 5)
    cfg.env.play_mode = True
    cfg.viewer.visualize_foot_contacts = False
    cfg.viewer.visualize_velocity_arrows = True
    cfg.viewer.ref_env = list(range(cfg.env.num_envs))
    cfg.viewer.print_debug_velocities = False
    env = None
    try:
        env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
        configure_fixed_command(env, args)
        if getattr(args, "zero_command_brake", False):
            env.enable_zero_command_brake()
        env.reset()
        runner, _ = task_registry.make_alg_runner(env, args=args, train_cfg=train_cfg, save_config=False)
        policy = runner.get_inference_policy(device=env.device)
        env.compute_observations()
        obs = env.get_observations()
        print(f"Session: {args.steps} policy ticks ({args.steps * env.dt:g} s); "
              f"episode timeout: {env.max_episode_length * env.dt:g} s. Falls still reset.", flush=True)
        if should_export_policy(args):
            from robot_gym.utils.export import export_policy
            output = checkpoint.parent / "exported" / checkpoint.stem
            export_policy(runner, env, output)
            print(f"Exported policy: {output}", flush=True)
        with torch.no_grad():
            for _ in range(args.steps):
                if not args.headless and not env.sim.viewer.is_alive():
                    break
                obs, _, _, _ = env.step(policy(obs).detach())
    except KeyboardInterrupt:
        print("Replay interrupted.", flush=True)
    except gs.GenesisException as error:
        if "Viewer closed" not in str(error):
            raise
    finally:
        from robot_gym.envs.base.base_task import BaseTask
        if BaseTask._gs_initialized:
            gs.destroy()
            BaseTask._gs_initialized = False


if __name__ == "__main__":
    play(get_args())
