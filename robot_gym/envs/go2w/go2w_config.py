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
        pos = (0.0, 0.0, 0.42)
        default_joint_angles = {
            "FL_hip_joint": 0.1,
            "FL_thigh_joint": 0.8,
            "FL_calf_joint": -1.5,
            "FL_foot_joint": 0.0,
            "FR_hip_joint": -0.1,
            "FR_thigh_joint": 0.8,
            "FR_calf_joint": -1.5,
            "FR_foot_joint": 0.0,
            "RL_hip_joint": 0.1,
            "RL_thigh_joint": 1.0,
            "RL_calf_joint": -1.5,
            "RL_foot_joint": 0.0,
            "RR_hip_joint": -0.1,
            "RR_thigh_joint": 1.0,
            "RR_calf_joint": -1.5,
            "RR_foot_joint": 0.0,
        }

    class env(GO2Cfg.env):
        # base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3)
        # + commands(3) + dof_pos(16) + dof_vel(16) + actions(16) = 60
        num_observations = 60
        num_actions = 16

    class terrain(GO2Cfg.terrain):
        name = "go2w_training_terrain"

    class commands(GO2Cfg.commands):
        curriculum = True
        max_curriculum = 1.5
        stand_command_probability = 0.20

        class ranges(GO2Cfg.commands.ranges):
            lin_vel_x = [-0.8, 0.8]
            lin_vel_y = [-0.3, 0.3]
            ang_vel_yaw = [-1.0, 1.0]

    class control(GO2Cfg.control):
        control_type = {
            **{name: "P" for name in LEG_JOINTS},
            **{name: "V" for name in WHEEL_JOINTS},
        }

        stiffness = {
            **GO2Cfg.control.stiffness,
            "FL_foot_joint": 0.0,
            "FR_foot_joint": 0.0,
            "RL_foot_joint": 0.0,
            "RR_foot_joint": 0.0,
        }
        damping = {
            **GO2Cfg.control.damping,
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
            **{name: 12.0 for name in WHEEL_JOINTS},
        }

    class termination(GO2Cfg.termination):
        base_height_threshold = 0.29
        roll_threshold = 35.0 * np.pi / 180.0
        pitch_threshold = 35.0 * np.pi / 180.0

    class asset(GO2Cfg.asset):
        robot_file = "go2w_description.urdf"
        name = "go2w"
        robot_name = "go2w"
        file_format = "urdf"
        foot_link_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        contact_height = 0.086
        hip_abduction_indices = [0, 4, 8, 12]

    class rewards(GO2Cfg.rewards):
        base_height_target = 0.39
        clearance_target = 0.03
        clearance_sigma = 0.015
        contact_force_threshold = 8.0
        wheeled_forward_activation_vel = 0.25
        lateral_step_activation_vel = 0.15
        yaw_step_activation_vel = 0.45

        class scales(GO2Cfg.rewards.scales):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.6
            lin_vel_z = -0.1
            ang_vel_xy = -0.05
            orientation = -0.5
            base_height = -5.0
            torques = -0.0002
            dof_vel = -0.0
            dof_acc = -2.5e-7
            action_rate = -0.01
            termination = -10.0
            dof_pos_limits = -2.0
            dof_vel_limits = -0.0
            torque_limits = -0.5
            feet_air_time = 0.0
            stand_still = 0.5
            feet_slide = 0.0
            foot_swing_clearance = 0.08
            leg_motion = -0.08
            wheel_contact = 0.2
            unnecessary_wheel_air = -0.4
            survive = 0.05
            collision = 0.0
            feet_stumble = 0.0


class GO2WCfgPPO(GO2CfgPPO):
    seed = 1

    class actor(GO2CfgPPO.actor):
        class_name = "MLPModel"
        hidden_dims = [512, 256, 128]
        activation = "elu"
        obs_normalization = True
        distribution_cfg = {
            "class_name": "GaussianDistribution",
            "init_std": 0.75,
            "std_type": "scalar",
        }

    class critic(GO2CfgPPO.critic):
        class_name = "MLPModel"
        hidden_dims = [512, 256, 128]
        activation = "elu"
        obs_normalization = True

    class algorithm(GO2CfgPPO.algorithm):
        class_name = "PPO"
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
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
