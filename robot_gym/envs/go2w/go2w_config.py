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
            **{name: 0.25 for name in LEG_JOINTS},
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
        clearance_target = 0.035
        contact_force_threshold = 8.0

        class scales(GO2Cfg.rewards.scales):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.8
            lin_vel_z = -0.1
            ang_vel_xy = -0.0
            orientation = -0.5
            base_height = -1.0
            torques = -0.0002
            dof_vel = -0.0
            dof_acc = -2.5e-7
            action_rate = -0.01
            termination = -20.0
            dof_pos_limits = -2.0
            dof_vel_limits = -0.0
            torque_limits = -1.0
            feet_air_time = 0.0
            stand_still = -5.0
            foot_swing_clearance = 0.0
            survive = 0.01
            collision = 0.0
            feet_stumble = 0.0


class GO2WCfgPPO(GO2CfgPPO):
    class actor(GO2CfgPPO.actor):
        distribution_cfg = {
            "class_name": "GaussianDistribution",
            "init_std": 0.8,
            "std_type": "scalar",
        }

    class runner(GO2CfgPPO.runner):
        experiment_name = "go2w"
        run_name = ""
        wandb_project = "go2w-locomotion"
