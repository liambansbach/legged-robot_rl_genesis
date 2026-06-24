from .base_config import BaseConfig

class LeggedRobotCfg(BaseConfig):
    class env:
        num_envs = 4096
        num_observations = 36
        num_actions = 8
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds
        num_privileged_obs = None 
        play_mode = False # For training a policy this should always be False because it skips the reward computation and other overhead. For testing or visualization in "play.py" set play_mode = True to speed up the simulation by skipping reward computation and other training overhead.

    class terrain:
        # top-level selection:
        # "plane"               
        # "heightfield"        -> use a custom heightfield (e.g. from numPy array); requires setting "heightfield"
        # "mixed"              -> random tiles inside one terrain
        # "random"             -> WARNING!! randomly choose one entry from options - preferably use mixed mode to lower simulation overhead
        # or directly one Genesis terrain type, e.g. "wave_terrain"
        mode = "plane"

        # used only when mode == "random" but the list shows all possible terrain types that can manually set as "mode" or be used in the "mixed" mode:
        options = [
            "plane",
            "flat_terrain",
            "random_uniform_terrain",
            "pyramid_sloped_terrain",
            "discrete_obstacles_terrain",
            "wave_terrain",              
            "pyramid_stairs_terrain",
            "stepping_stones_terrain",
            "fractal_terrain",
            "mixed",
        ]
        probs = [0.25, 0.05, 0.1, 0.15, 0.1, 0.15, 0.15, 0.05, 0.05, 0.0] # If mode == "random", probabilities for selecting each terrain type from options list. Must be same length as options list. Values must be non-negative and will be normalized to sum to 1. If empty, uniform probabilities are assumed.

        curriculum = False

        # global geometry
        n_subterrains = (7, 7) # number should be odd to have a center terrain tile
        subterrain_size = (4.0, 4.0) # size musst be divisible by horizontal_scale!
 
        # lower horizontal_scale = finer mesh = less low-poly look
        horizontal_scale = 0.25   # basically the grid size of each tile in meters   0.25 
        vertical_scale = 0.005    # The height of each step in the subterrain in meters. 0.025

        randomize = False
        spawn_flat_radius_sub = 0
        border_flat = False

        # optional manual override; if None -> auto-center
        pos = None

        # Genesis terrain cache name; None disables caching -> creating complex terrains can be slow on first creation!
        name = None 

        # only used if mode == "heightfield"
        heightfield = None

        color = (0.18, 0.18, 0.21)

        class mixed:
            options = [
                "flat_terrain",
                "random_uniform_terrain",
                "wave_terrain",
                "pyramid_sloped_terrain",
                "discrete_obstacles_terrain",
                "pyramid_stairs_terrain",
                "stepping_stones_terrain",
                "fractal_terrain",
            ]
            probs = [0.15, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.05] # probabilities for selecting each terrain type for the mixed terrain. Must be same length as options list.

        class terrain_kwargs:
            class flat_terrain:
                pass

            class fractal_terrain:
                levels = 8
                scale = 3.5

            class random_uniform_terrain:
                min_height = -0.04
                max_height = 0.05
                step = 0.01
                downsampled_scale = 0.5

            class pyramid_sloped_terrain:
                slope = 0.25

            class discrete_obstacles_terrain: # TODO not working yet
                max_height = 0.05
                min_size = 0.25
                max_size = 1.0
                num_rects = 5

            class wave_terrain:
                num_waves = 2.0
                amplitude = 0.06

            class pyramid_stairs_terrain:
                step_width = 0.5
                step_height = -0.075

            class stepping_stones_terrain:
                stone_size = 0.5
                stone_distance = 0.075
                max_height = 0.04
                platform_size = 0.0

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 3 # default: lin_vel_x, lin_vel_y, ang_vel_yaw,
        resampling_time = 5. # time before command are changed[s]
        stand_command_probability = 0.25 # probability of sampling a "stand still" command (0 velocity) instead of a random command
        class ranges:
            lin_vel_x = [0.0, 1.0] # min max [m/s]
            lin_vel_y = [-0.3, 0.3]   # min max [m/s]
            ang_vel_yaw = [-0.75, 0.75]    # min max [rad/s]

    class init_state:
        pos = (0.0, 0.0, 0.415) # x,y,z [m]
        rot = (1.0, 0.0, 0.0, 0.0) # w,x,y,z [quat]
        default_joint_angles = { # target angles when action = 0.0
            "joint_a": 0., 
            "joint_b": 0.}

    class control:
        control_type = 'P' # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
        dof_vel_limits = {'joint_a': 6.0, 'joint_b': 6.0}   # [rad/s]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset:
        robot_file = ""
        name = "legged_robot"
        robot_name = None
        file_format = None

        merge_fixed_links = True
        links_to_keep = None

        # following values are automatically set by the URDF reader in "legged_robot.py"!
        joint_names = None
        foot_link_names = None
        


    class domain_rand:
        randomize_friction = False
        friction_range = [0.7, 1.3]

        randomize_base_mass = False
        added_mass_range = [-0.2, 0.3]

        randomize_com = False
        com_shift_range = [-0.015, 0.015]

        push_robots = False
        push_interval_s = 10
        push_torque_scale = 2.0

        randomize_kp = False
        kp_scale_range = [0.8, 1.2]

        randomize_kd = False
        kd_scale_range = [0.8, 1.2]

        randomize_action_delay = False
        action_delay_steps_range = [0, 2]  # policy steps, not sim substeps

    class rewards:
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -0.
            torques = -0.00001
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0. 
            feet_air_time =  0.0
            collision = 0.0
            feet_stumble = 0.0 
            action_rate = -0.01
            stand_still = 0.25
            feet_slide = -0.0

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_vel_limit = 1.
        soft_torque_limit = 0.9
        soft_dof_pos_limit = 0.9 # [%] percentage of urdf limits, values above this limit are penalized

        base_height_target = 1.
        max_contact_force = 100. # forces above this value are penalized#
        contact_force_threshold = 7. # [Nm] forces above this value are counted as a contact for stumble and air time rewards

    class termination:
        base_height_threshold = 0.25 # [m], if the robot's base is below this height, the episode is terminated
        roll_threshold = 0.8 # [rad], if the robot rolls more than this, the episode is terminated
        pitch_threshold = 0.8 # [rad], if the robot pitches more than this, the episode is terminated

    class normalization:
        class obs_scales:
            lin_vel = 1.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 1.0
            height_measurements = 1.0
        clip_observations = 100.
        clip_actions = 10.

    class noise:  
        add_noise = True
        noise_level = 1.0 # scales other values  
        class noise_scales:
            dof_pos = 0.015
            dof_vel = 1.25
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    # viewer camera:
    class viewer:
        max_fps = 60
        ref_env = [0]
        pos = (2.0, 0.0, 2.5) # [m]
        lookat = (0.0, 0.0, 0.5)  # [m]
        fov = 40 # [degrees]
        show_world_frame = True
        visualize_foot_contacts = False
        print_debug_velocities = False 

        # velocity vectors for debugging:
        visualize_velocity_arrows = False
        velocity_arrow_scale = 0.6
        velocity_arrow_radius = 0.03

    class sim:
        dt =  0.005
        substeps = 1
        gravity = (0., 0. , -9.81)  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z
        enable_collision = True
        enable_joint_limit = True
        enable_self_collision = False
        performance_mode = True #  When performance_mode=True, Genesis bakes static tensor shapes into compiled kernels, yielding roughly 30% faster simulation. The trade-off is that kernels must be recompiled whenever the scene changes, which can take several minutes. With performance mode off (the default), kernels are shape-generic and only compiled once — rebuilds take just a few seconds once cached. In short, keep it off during research, fast iteration, debugging, and interactive data inspection; turn it on for policy training and production deployment.

        # dont manually set these! They will automatically set if domain randomization needs it.
        batch_links_info = False
        batch_dofs_info = False


class LeggedRobotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = "OnPolicyRunner"

    class actor:
        class_name = "MLPModel"
        hidden_dims = [512, 256, 128]
        activation = "elu"
        obs_normalization = True
        distribution_cfg = {
            "class_name": "GaussianDistribution",
            "init_std": 0.75,
            "std_type": "scalar",
        }

    class critic:
        class_name = "MLPModel"
        hidden_dims = [512, 256, 128]
        activation = "elu"
        obs_normalization = True

    class algorithm:
        class_name = "PPO"
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 8.0e-4
        schedule = "adaptive"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
        normalize_advantage_per_mini_batch = False
        rnd_cfg = None
        symmetry_cfg = None
        share_cnn_encoders = False

    class runner:
        num_steps_per_env = 24
        obs_groups = {"actor": ["policy"], "critic": ["policy"]}
        save_interval = 50
        max_iterations = 1500

        logger = "wandb"
        experiment_name = "test"
        run_name = ""

        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None

        log_wandb = True
        wandb_project = "bipedal-locomotion"