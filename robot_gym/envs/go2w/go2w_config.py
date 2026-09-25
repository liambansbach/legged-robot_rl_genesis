import numpy as np

from robot_gym.envs.go2.go2_config import GO2Cfg, GO2CfgPPO


LEG_JOINTS = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]

WHEEL_JOINTS = [
    "FL_foot_joint",
    "FR_foot_joint",
    "RL_foot_joint",
    "RR_foot_joint",
]


class GO2WCfg(GO2Cfg):
    class init_state(GO2Cfg.init_state):
        pos = (0.0, 0.0, 0.45)
        joint_position_noise = 0.03
        joint_velocity_noise = 0.05
        orientation_noise = (0.02, 0.02, 0.05)
        linear_velocity_noise = 0.03
        angular_velocity_noise = 0.03
        default_joint_angles = {
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.70,
            "FL_calf_joint": -1.33,
            "FL_foot_joint": 0.0,
            "FR_hip_joint": 0.0,
            "FR_thigh_joint": 0.70,
            "FR_calf_joint": -1.33,
            "FR_foot_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.75,
            "RL_calf_joint": -1.31,
            "RL_foot_joint": 0.0,
            "RR_hip_joint": 0.0,
            "RR_thigh_joint": 0.75,
            "RR_calf_joint": -1.31,
            "RR_foot_joint": 0.0,
        }

    class env(GO2Cfg.env):
        # base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3)
        # + commands(3) + leg_pos(12) + dof_vel(16) + actions(16) = 56
        num_observations = 56
        num_actions = 16

    class terrain(GO2Cfg.terrain):
        name = "go2w_training_terrain"
        mode = "plane"

    class commands(GO2Cfg.commands):
        curriculum = False
        short_command_duration_range = [0.5, 1.0]
        sustained_command_duration_range = [1.5, 3.0]
        sustained_command_probability = 0.30
        linear_deadzone = 0.01
        yaw_deadzone = 0.01
        stand_threshold = 1e-6
        # Absolute probabilities: straight, arc, yaw, precision, lateral, mixed.
        moving_mixture_probabilities = [0.20, 0.20, 0.15, 0.10, 0.13, 0.07]
        stand_command_probability = 0.15
        # m/s; mixed retains the full signed range.
        pure_lateral_magnitude_range = [0.10, 0.30]

        class ranges(GO2Cfg.commands.ranges):
            lin_vel_x = [-0.35, 1.10]
            lin_vel_y = [-0.30, 0.30]
            ang_vel_yaw = [-1.4, 1.4]

    class control(GO2Cfg.control):
        control_type = {
            **{name: "P" for name in LEG_JOINTS},
            **{name: "V" for name in WHEEL_JOINTS},
        }

        stiffness = {
            **{name: 40.0 for name in LEG_JOINTS},
            "FL_foot_joint": 0.0,
            "FR_foot_joint": 0.0,
            "RL_foot_joint": 0.0,
            "RR_foot_joint": 0.0,
        }
        damping = {
            **{name: 1.0 for name in LEG_JOINTS},
            "FL_foot_joint": 1.0,
            "FR_foot_joint": 1.0,
            "RL_foot_joint": 1.0,
            "RR_foot_joint": 1.0,
        }
        dof_vel_limits = {
            **GO2Cfg.control.dof_vel_limits,
            "FL_foot_joint": 30.1,
            "FR_foot_joint": 30.1,
            "RL_foot_joint": 30.1,
            "RR_foot_joint": 30.1,
        }

        action_scale = {
            **{name: 0.2 for name in LEG_JOINTS},
            **{name: 18.0 for name in WHEEL_JOINTS},
        }

        wheel_velocity_target_limit = 20.0  # rad/s; URDF limit is 30.1

    class normalization(GO2Cfg.normalization):
        clip_actions = 1.0

    class domain_rand(GO2Cfg.domain_rand):
        friction_range = [0.6, 1.2]
        friction_links = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        added_mass_range = [-0.5, 1.5]
        com_shift_range = [-0.015, 0.015]
        kp_scale_range = [0.9, 1.1]
        kd_scale_range = [0.9, 1.1]
        action_delay_steps_range = [0, 1]
        push_robots = True
        push_interval_range_s = [3.0, 6.0]
        push_duration_range_s = [0.10, 0.20]
        push_force_range = [20.0, 50.0]
        push_torque_range = [0.0, 0.0]

    class noise(GO2Cfg.noise):
        class noise_scales(GO2Cfg.noise.noise_scales):
            dof_pos = 0.01
            dof_vel = 0.2  # leg rad/s
            wheel_vel = 0.5  # wheel rad/s
            lin_vel = 0.05
            ang_vel = 0.08
            gravity = 0.02

    class termination(GO2Cfg.termination):
        base_height_threshold = 0.33
        roll_threshold = 30.0 * np.pi / 180.0
        pitch_threshold = 30.0 * np.pi / 180.0

    class asset(GO2Cfg.asset):
        robot_file = "go2w_description.urdf"
        name = "go2w"
        robot_name = "go2w"
        file_format = "urdf"
        foot_link_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        contact_height = 0.086
        hip_abduction_indices = [0, 4, 8, 12]

    class rewards(GO2Cfg.rewards):
        base_height_target = 0.415
        tracking_sigma_x = 0.25  # Squared-error denominators, in (m/s)^2.
        tracking_sigma_y = 0.04
        clearance_target = 0.03
        clearance_sigma = 0.015
        contact_force_threshold = 8.0
        lateral_step_start_vel = 0.03
        lateral_step_activation_vel = 0.10
        yaw_mobility_start = 0.30
        yaw_mobility_full = 0.85
        yaw_mobility_weight = 0.80
        yaw_pose_start = 0.60
        yaw_pose_full = 1.10
        yaw_pose_weight = 0.35
        min_wheel_side_clearance = 0.045
        min_lateral_wheel_separation = 0.14

        class scales(GO2Cfg.rewards.scales):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.8
            lin_vel_z = -0.15
            ang_vel_xy = -0.12
            orientation = -1.2
            base_height = -8.0
            torques = 0.0
            normalized_effort = -0.03
            dof_vel = -0.0
            dof_acc = 0.0
            leg_acc = -2.5e-7
            wheel_acc = -1.0e-7
            action_rate = 0.0
            leg_action_rate = -0.01
            wheel_action_rate = -0.005
            termination = -10.0
            dof_pos_limits = -2.0
            dof_vel_limits = -0.0
            torque_limits = -0.5
            feet_air_time = 0.0
            stand_still = -0.5
            feet_slide = 0.0
            foot_swing_clearance = 0.08
            default_pose = -1.0
            leg_motion = -0.02
            wheel_contact = 0.0
            unnecessary_wheel_air = -0.25
            wheel_crossover = -2.0
            survive = 0.0
            collision = -0.5
            feet_stumble = 0.0

    class sim(GO2Cfg.sim):
        enable_self_collision = True


class GO2WCfgPPO(GO2CfgPPO):
    seed = 1

    class actor(GO2CfgPPO.actor):
        class_name = "MLPModel"
        hidden_dims = [512, 256, 128]
        activation = "elu"
        obs_normalization = True
        distribution_cfg = {
            "class_name": "GaussianDistribution",
            "init_std": 0.55,
            "std_type": "log",
            "std_range": [0.05, 0.8],
        }

    class critic(GO2CfgPPO.critic):
        class_name = "MLPModel"
        hidden_dims = [512, 256, 128]
        activation = "elu"
        obs_normalization = True

    class algorithm(GO2CfgPPO.algorithm):
        class_name = "PPO"
        symmetry_cfg = {
            "data_augmentation_func": "robot_gym.envs.go2w.go2w_symmetry:sagittal_augmentation",
            "use_data_augmentation": True,
            "use_mirror_loss": False,
            "mirror_loss_coeff": 0.0,
        }
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.005
        num_learning_epochs = 5
        num_mini_batches = 8
        learning_rate = 8.0e-4
        schedule = "adaptive"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner(GO2CfgPPO.runner):
        num_steps_per_env = 48
        max_iterations = 2000
        save_interval = 50
        experiment_name = "go2w"
        run_name = ""
        resume = False
        load_run = -1
        checkpoint = -1
        log_wandb = True
        wandb_project = "go2w-locomotion"
