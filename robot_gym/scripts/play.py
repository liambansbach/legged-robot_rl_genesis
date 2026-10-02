import os
import torch
import genesis as gs

from robot_gym import ROBOT_GYM_ROOT_DIR
from robot_gym.envs import *  # noqa: F401,F403 -> ensures task registration
from robot_gym.utils import get_args, task_registry 

"""
Replay a saved Go2-W transfer_v1 checkpoint from the repository root.
Activate the existing Genesis environment first (conda activate genesis-gpu).

    PowerShell:
    python -m robot_gym.scripts.play `
        --task go2w `
        --go2w_profile transfer_v1 `
        --experiment_name go2w_transfer_v1 `
        --load_run transfer_v1_seed1_20261002_090639_2026-10-02_09-09-17 `
        --checkpoint 1999 `
        --num_envs 1 `
        --rl_device cuda:0 `
        --no_export

This example selects the existing model_1999.pt in that run. For another run,
replace --load_run and --checkpoint with its directory name and saved model
number, and select the matching --go2w_profile. Go2-W checks the saved config
before building the simulator and requires explicit run/checkpoint selection;
omitted selectors and -1 (latest) are rejected.

    --task: Task name defined in task_registry envs/__init__.py
    --experiment_name: Name of the experiment (used to locate logs directory).
    --run_name: Logging name only; does not select the source checkpoint.
    --load_run: Training run directory under logs/<experiment_name>.
    --checkpoint: Number in model_<number>.pt, not a filename or update budget.
    --go2w_profile: Reconstruct the selected Go2-W training configuration.
    --num_envs: Number of robots to visualize (one in the replay example).
    --rl_device: Device used for inference (cpu, cuda, cuda:0, etc.).
    --headless: Force display off (no rendering). Usually disabled for visualization.
    --no_export: Replay only; skip the default policy export.

Other tasks retain their legacy latest-run/checkpoint defaults; for Dodo only:
    python -m robot_gym.scripts.play --task dodo --experiment_name dodo_walking_test
"""

EXPORT_POLICY = True


def should_export_policy(args):
    return EXPORT_POLICY and not args.zero_command_brake and not args.no_export


def configure_fixed_command(env, args):
    """Any supplied command axis enables fixed play; unspecified axes become zero."""
    values = [getattr(args, f"command_{axis}", None) for axis in ("vx", "vy", "yaw")]
    if any(value is not None for value in values):
        command = tuple(0.0 if value is None else value for value in values)
        env.set_fixed_command(command)
        print(f"Fixed command for every environment [vx, vy, yaw]: {command}", flush=True)


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)  
    if args.task == "go2w":
        from pathlib import Path
        from robot_gym.utils.helpers import update_cfg_from_args, get_load_path, class_to_dict
        from robot_gym.utils.urdf_reader import URDFReader
        from robot_gym.utils.diagnostics import check_reference_contract

        update_cfg_from_args(env_cfg, train_cfg, args)
        if args.load_run in (None, "-1") or args.checkpoint in (None, -1):
            raise ValueError("Go2-W playback requires explicit --load_run and --checkpoint")
        checkpoint = get_load_path(
            Path(ROBOT_GYM_ROOT_DIR) / "logs" / train_cfg.runner.experiment_name,
            args.load_run, args.checkpoint,
        )
        env_cfg.asset.joint_names = URDFReader(env_cfg.asset.robot_file).joint_names
        check_reference_contract(
            args.reference_config or Path(checkpoint).with_name("config.yaml"),
            class_to_dict(env_cfg), class_to_dict(train_cfg),
        )

    # ----------------------------------------------------------------------
    # Override some parameters for testing / visualization
    # ----------------------------------------------------------------------
    envs_to_visualize = 5 # define how many parallel envs to visualize (keep it low to reduce fps impact)

    env_cfg.env.num_envs = min(env_cfg.env.num_envs, envs_to_visualize)
    env_cfg.env.play_mode = True # skips reward computation and other training overhead for faster simulation during play mode 

    # disable curriculum for play mode
    env_cfg.terrain.curriculum = False
    env_cfg.commands.curriculum = False

    # noise settings for eval
    env_cfg.noise.add_noise = False
    env_cfg.init_state.joint_position_noise = 0.0
    env_cfg.init_state.joint_velocity_noise = 0.0
    env_cfg.init_state.orientation_noise = (0.0, 0.0, 0.0)
    env_cfg.init_state.linear_velocity_noise = 0.0
    env_cfg.init_state.angular_velocity_noise = 0.0

    # Domain randomization settings for eval
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_kp = False
    env_cfg.domain_rand.randomize_kd = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_com = False 
    env_cfg.domain_rand.randomize_action_delay = False
    if hasattr(env_cfg.domain_rand, "randomize_armature"):
        env_cfg.domain_rand.randomize_armature = False
        from robot_gym.utils.export import select_transfer_dynamics
        select_transfer_dynamics(env_cfg, args)

    # Use --command_vx/--command_vy/--command_yaw for fixed commands, keeping ranges intact.
    
    # Optional viewer/debug settings for play mode
    env_cfg.sim.performance_mode = False # genesis performance mode should be used for training only.
    env_cfg.sim.deterministic = True
    env_cfg.viewer.visualize_foot_contacts = False
    env_cfg.viewer.visualize_velocity_arrows = True 
    env_cfg.viewer.ref_env = list(range(args.num_envs or envs_to_visualize))
    env_cfg.viewer.print_debug_velocities = False

    # ----------------------------------------------------------------------
    # Prepare environment
    # ----------------------------------------------------------------------
    env, _ = task_registry.make_env(
        name=args.task,
        args=args,
        env_cfg=env_cfg,
    )

    configure_fixed_command(env, args)
    print(f"Episode timeout: {env.max_episode_length * env.dt:g} s; --steps spans resets. Falls still reset.", flush=True)
    if args.zero_command_brake:
        env.enable_zero_command_brake()
    obs, _ = env.reset()

    # ----------------------------------------------------------------------
    # Load trained policy
    # ----------------------------------------------------------------------
    train_cfg.runner.resume = True

    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        save_config=False, 
    )

    policy = ppo_runner.get_inference_policy(device=env.device)

    # ---------------------------------------------------------------------- 
    # Export policy as JIT
    # ----------------------------------------------------------------------
    # A composite controller needs an explicitly qualified package, not a bare neural export.
    if should_export_policy(args):
        path = os.path.join(
            ROBOT_GYM_ROOT_DIR,
            "logs",
            train_cfg.runner.experiment_name,
            "exported",
            "policies",
        )

        from robot_gym.utils.export import export_policy
        export_policy(ppo_runner, env, path)

        print(f"Exported policy as jit script to: {path}")

    # ----------------------------------------------------------------------
    # Run policy
    # ----------------------------------------------------------------------
    with torch.no_grad():
        for _ in range(args.steps):
            actions = policy(obs)
            obs, rews, dones, infos = env.step(actions.detach()) 
    gs.destroy()


if __name__ == "__main__":
    args = get_args()
    play(args)
