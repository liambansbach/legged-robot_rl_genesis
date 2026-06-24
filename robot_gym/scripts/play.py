import os
import torch

from robot_gym import ROBOT_GYM_ROOT_DIR
from robot_gym.envs import *  # noqa: F401,F403 -> ensures task registration
from robot_gym.utils import get_args, task_registry 

"""
Example play command (command line call) with all arguments specified:

    python -m robot_gym.scripts.play \
        --task dodo \                              -> Task name defined in task_registry envs/__init__.py
        --experiment_name daimao_walking \         -> Name of the experiment (used to locate logs directory).
        --run_name run_01 \                        -> Name of the run. Overrides config file if provided.
        --load_run daimao_walking \                -> Name of the run to load. If -1: will load the last run. Overrides config file if provided.
        --checkpoint -1 \                          -> Saved model checkpoint number. If -1: will load the last checkpoint.
        --rl_device cuda:0 \                       -> Device used for inference (cpu, cuda, cuda:0, etc..)
        --headless                                 -> Force display off (no rendering). Usually disabled for visualization.

Not all arguments are required. A simple call could look like this:
    python -m robot_gym.scripts.play --task dodo --experiment_name dodo_walking_test
"""

EXPORT_POLICY = True


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)  

    # ----------------------------------------------------------------------
    # Override some parameters for testing / visualization
    # ----------------------------------------------------------------------
    envs_to_visualize = 5 # define how many parallel envs to visualize (keep it low to reduce fps impact)

    env_cfg.env.num_envs = min(env_cfg.env.num_envs, envs_to_visualize)
    env_cfg.env.play_mode = True # skips reward computation and other training overhead for faster simulation during play mode 

    # disable curriculum for play mode
    env_cfg.terrain.curriculum = False
    #env_cfg.terrain.mode = "plane"

    # noise settings for eval
    env_cfg.noise.add_noise = False

    # Domain randomization settings for eval
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = True
    env_cfg.domain_rand.randomize_kp = False
    env_cfg.domain_rand.randomize_kd = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_com = False 
    env_cfg.domain_rand.randomize_action_delay = False

    #hardcode velocity to test tracking performance in play mode (optional)
    # env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    # env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    # env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0] 
    
    # Optional viewer/debug settings for play mode
    env_cfg.sim.performance_mode = False # genesis performance mode should be used for training only.
    env_cfg.viewer.visualize_foot_contacts = False
    env_cfg.viewer.visualize_velocity_arrows = True 
    env_cfg.viewer.ref_env = list(range(envs_to_visualize)) 
    env_cfg.viewer.print_debug_velocities = False

    # ----------------------------------------------------------------------
    # Prepare environment
    # ----------------------------------------------------------------------
    env, _ = task_registry.make_env(
        name=args.task,
        args=args,
        env_cfg=env_cfg,
    )

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
    if EXPORT_POLICY:
        path = os.path.join(
            ROBOT_GYM_ROOT_DIR,
            "logs",
            train_cfg.runner.experiment_name,
            "exported",
            "policies",
        )

        ppo_runner.export_policy_to_jit(
            path,
            filename="policy_1.pt",
        )

        print(f"Exported policy as jit script to: {path}")

    # ----------------------------------------------------------------------
    # Run policy
    # ----------------------------------------------------------------------
    with torch.no_grad():
        for _ in range(10 * int(env.max_episode_length)):
            actions = policy(obs)
            obs, rews, dones, infos = env.step(actions.detach()) 


if __name__ == "__main__":
    args = get_args()
    play(args)