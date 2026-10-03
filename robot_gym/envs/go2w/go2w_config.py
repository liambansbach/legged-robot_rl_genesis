"""Go2-W recipes, physical presets and task-owned CLI validation."""

import math
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


MEASURED_URDF = "go2w_measured_ed8dc93.urdf"
MEASURED_SHA256 = "d298cc7bf4894e869840bdab9ac60f09548d998444018e61854636cff46b1d8c"

TRANSFER_V2_REVIEW_COMMANDS = {
    "forward_fast": (0.5, 0, 0),
    "lateral_strong_positive": (0, 0.3, 0), "lateral_strong_negative": (0, -0.3, 0),
    "yaw_strong_positive": (0, 0, 0.8), "yaw_strong_negative": (0, 0, -0.8),
}


def uses_event_steps(cfg):
    """Explicit shared behavior, including the unchanged legacy event profile."""
    return getattr(cfg, "go2w_behavior", getattr(cfg, "go2w_profile", None)) == "event_step_v1"

FINETUNE_COMMANDS = {
    "mixed_zero_yaw_probability": 0.50,
    "moving_long_probability": 0.05,
    "moving_long_duration_range": [8.0, 15.0],
}
FINETUNE_MOBILITY = {
    "yaw_mobility_start": 0.15,
    "yaw_mobility_full": 0.60,
    "wheel_air_relaxation": 0.90,
}
FINETUNE_PRECISION = {
    "rewards.tracking_sigma_yaw": 0.04,
    "rewards.clearance_sigma": 0.025,
    "rewards.clearance_activation_height": 0.04,
    "rewards.scales.foot_swing_clearance": 0.40,
    "domain_rand.kp_scale_range": [0.85, 1.15],
    "domain_rand.kd_scale_range": [0.85, 1.15],
    "domain_rand.action_delay_steps_range": [0, 2],
}


def apply_go2w_finetune(env_cfg, train_cfg, name):
    """Explicit continuation designs on the unchanged step-recovery action contract."""
    if name not in ("coverage", "coverage_mobility", "precision_clearance"):
        raise ValueError(f"Unknown Go2-W finetune: {name}")
    for cfg in (env_cfg, train_cfg):
        if cfg is not None:
            if getattr(cfg, "go2w_profile", None) != "step_recovery_v1":
                raise ValueError("Go2-W finetune requires --go2w_profile step_recovery_v1")
            previous = getattr(cfg, "go2w_finetune", name)
            if previous != name:
                raise ValueError("Cannot change finetune selection on an already resolved config")
            cfg.go2w_finetune = name
    if env_cfg is not None:
        for key, value in FINETUNE_COMMANDS.items():
            setattr(env_cfg.commands, key, value.copy() if isinstance(value, list) else value)
        if name == "coverage_mobility":
            for key, value in FINETUNE_MOBILITY.items():
                setattr(env_cfg.rewards, key, value)
        if name == "precision_clearance":
            for path, value in FINETUNE_PRECISION.items():
                target = env_cfg
                *parts, key = path.split(".")
                for part in parts:
                    target = getattr(target, part)
                setattr(target, key, value.copy() if isinstance(value, list) else value)


def apply_go2w_event_quality(env_cfg, name):
    """One smaller auxiliary objective; the unset profile keeps its original formula."""
    if name != "sufficient_clearance":
        raise ValueError(f"Unknown event quality profile: {name}")
    if env_cfg is not None:
        if getattr(env_cfg, "go2w_profile", None) != "event_step_v1":
            raise ValueError("Event quality selection requires event_step_v1")
        env_cfg.rewards.event_step = {**env_cfg.rewards.event_step, "quality_profile": name}
        env_cfg.rewards.scales.step_event = 0.05


def apply_go2w_profile(env_cfg, train_cfg, name):
    """One explicit candidate; never mutate the registered config or its dictionaries."""
    if name in ("transfer_v1", "transfer_v2"):
        # Reuse selected P behavior; operational continuation state is never copied.
        new_env = env_cfg if env_cfg is not None and getattr(env_cfg, "go2w_profile", None) != name else None
        new_train = train_cfg if train_cfg is not None and getattr(train_cfg, "go2w_profile", None) != name else None
        apply_go2w_profile(new_env, new_train, "event_step_v1")
        if new_env is not None:
            new_env.go2w_profile = name
            new_env.go2w_behavior = "event_step_v1"
            new_env.rewards.sagittal_stance_weight = 2.0
            new_env.asset.robot_file = MEASURED_URDF
            new_env.asset.sha256 = MEASURED_SHA256
            new_env.asset.source_commit = "ed8dc93b3065a8e2a3a5919f032ed4690529117c"
            new_env.asset.contact_height = 0.09167
            new_env.asset.default_armature = 0.01
            new_env.control.armature = {joint: 0.01 for joint in LEG_JOINTS + WHEEL_JOINTS}
            new_env.control.passive_stiffness = 0.0
            new_env.control.passive_damping = 0.0
            new_env.control.passive_frictionloss = 0.0
            new_env.domain_rand.randomize_armature = True
            new_env.domain_rand.armature_range = [0.005, 0.02]
            new_env.domain_rand.scale_base_inertia_with_mass = True
            new_env.sim.batch_dofs_info = new_env.sim.batch_links_info = True
            new_env.sim.integrator = "approximate_implicitfast"
            if name == "transfer_v2":
                new_env.rewards.scales.orientation = -4.0
                new_env.rewards.tracking_sigma_x = 0.09  # (m/s)^2, not Gaussian std.
                new_env.rewards.support_pose = {
                    "hip": {"stand": 2.0, "loaded": 1.0, "unloaded": 0.2},
                    "sagittal": {"stand": 2.0, "loaded": 0.6, "unloaded": 0.06},
                }
                new_env.rewards.scales.hip_pose = -1.0
                new_env.rewards.scales.sagittal_pose = -1.0
                new_env.rewards.dense_swing = {
                    "height_target": 0.04, "upper_tail_start": 0.07,
                    "upper_tail_width": 0.02, "reposition_target": 0.04,
                    "initial_lift_credit": 0.2,
                }
                new_env.rewards.scales.wheel_swing = 0.4
                new_env.rewards.scales.lateral_wheel_scrub = -2.0
                new_env.commands.discovery_segment_probability = 0.70
                new_env.commands.discovery_segment_duration_range = [2.0, 4.0]
        if new_train is not None:
            new_train.go2w_profile = name
            new_train.go2w_behavior = "event_step_v1"
            new_train.runner.experiment_name = f"go2w_{name}"
            new_train.runner.max_iterations = 2000
            new_train.runner.save_interval = 100
            new_train.runner.logger = "tensorboard"
            new_train.runner.resume = False
            new_train.runner.load_run = None
            new_train.runner.checkpoint = None
            new_train.runner.resume_path = None
            new_train.runner.run_name = f"{name}_seed1"
        return
    if name == "event_step_v1":
        if env_cfg is not None and getattr(env_cfg, "go2w_profile", None) != name:
            env_cfg.go2w_profile = name
            scales = {"hip": 0.30, "thigh": 0.35, "calf": 0.40, "foot": 18.0}
            env_cfg.control.action_scale = {
                joint: scales[joint.split("_")[1]] for joint in LEG_JOINTS + WHEEL_JOINTS
            }
            env_cfg.commands.long_stand_probability = 0.25
            env_cfg.commands.long_stand_duration_range = [3.0, 6.0]
            env_cfg.commands.mixed_zero_yaw_probability = 0.50
            env_cfg.commands.moving_long_probability = 0.05
            env_cfg.commands.moving_long_duration_range = [8.0, 15.0]
            env_cfg.commands.pure_lateral_magnitude_range = [0.03, 0.30]
            env_cfg.commands.ranges.lin_vel_y = [-0.50, 0.50]
            env_cfg.commands.mixed_lateral_range = [-0.30, 0.30]
            env_cfg.commands.lateral_tail_probability = 0.20
            env_cfg.commands.lateral_curriculum_updates = [500, 1500]
            env_cfg.commands.lateral_tail_high_range = [0.30, 0.50]
            env_cfg.rewards.event_step = {
                "unload_force": 6.0, "reload_force": 10.0,
                "unload_dwell": 0.04, "reload_dwell": 0.06, "prior_support": 0.12,
                "duration_range": [0.10, 0.60], "minimum_height": 0.008,
                "minimum_reposition": 0.010, "full_reposition": 0.040,
                "target_base": 0.025, "target_gate": 0.025, "limb_factor": 2.0,
                "overshoot_band": 0.020, "credit_cap": 1.0,
            }
            env_cfg.rewards.discrete_reward_names = ["step_event"]
            env_cfg.rewards.yaw_tracking_mixture = {"broad_weight": 0.25, "broad_sigma": 0.25, "precise_sigma": 0.04}
            for key, value in {
                "foot_swing_clearance": 0.0, "default_pose": 0.0,
                "hip_pose": -2.0, "sagittal_pose": -0.6, "step_event": 0.60 / 4,
                "prolonged_unloading": -0.20, "insufficient_support": -1.0,
                "lin_vel_z": -1.0, "ang_vel_xy": -0.25, "stand_still": -2.0,
            }.items():
                setattr(env_cfg.rewards.scales, key, value)
            env_cfg.domain_rand.kp_scale_range = [0.85, 1.15]
            env_cfg.domain_rand.kd_scale_range = [0.85, 1.15]
            env_cfg.domain_rand.action_delay_steps_range = [0, 2]
        if train_cfg is not None and getattr(train_cfg, "go2w_profile", None) != name:
            train_cfg.go2w_profile = name
            train_cfg.actor.distribution_cfg = {
                "class_name": "GaussianDistribution", "init_std": 0.40,
                "std_type": "log", "std_range": [0.10, 0.70], "learn_std": True,
            }
            train_cfg.algorithm.entropy_coef = 0.003
            train_cfg.algorithm.learning_rate = 3e-4
            train_cfg.algorithm.gamma = 0.995
            train_cfg.runner.num_steps_per_env = 64
            train_cfg.runner.experiment_name = "go2w_event_step_v1"
            train_cfg.runner.max_iterations = 2000
            train_cfg.runner.save_interval = 500
        return
    if name != "step_recovery_v1":
        raise ValueError(f"Unknown Go2-W profile: {name}")
    if env_cfg is not None and getattr(env_cfg, "go2w_profile", None) != name:
        env_cfg.go2w_profile = name
        scales = {"hip": 0.30, "thigh": 0.35, "calf": 0.40, "foot": 18.0}
        env_cfg.control.action_scale = {
            joint: scales[joint.split("_")[1]] for joint in LEG_JOINTS + WHEEL_JOINTS
        }
        env_cfg.commands.long_stand_probability = 0.25
        env_cfg.commands.long_stand_duration_range = [3.0, 6.0]
        env_cfg.rewards.clearance_target = 0.04
        env_cfg.rewards.clearance_sigma = 0.02
        env_cfg.rewards.clearance_activation_height = 0.01
        env_cfg.rewards.reposition_speed = 0.15
        env_cfg.rewards.scales.foot_swing_clearance = 0.12
        env_cfg.rewards.scales.stand_still = -2.0
    if train_cfg is not None and getattr(train_cfg, "go2w_profile", None) != name:
        train_cfg.go2w_profile = name
        train_cfg.actor.distribution_cfg = {
            **train_cfg.actor.distribution_cfg,
            "init_std": 0.35,
        }
        train_cfg.algorithm.entropy_coef = 0.001
        train_cfg.runner.experiment_name = "go2w_step_recovery_v1"
        train_cfg.runner.max_iterations = 1500
        train_cfg.runner.save_interval = 250


def check_target_intervals(cfg, reader=None):
    """Candidate offsets must remain at least 0.02 rad inside the authored hard limits."""
    from robot_gym.utils.urdf_reader import URDFReader

    root = (reader or URDFReader(cfg.asset.robot_file)).root
    intervals = {}
    for name in LEG_JOINTS:
        limit = root.find(f"joint[@name='{name}']/limit")
        lower, upper = float(limit.get("lower")), float(limit.get("upper"))
        offset = cfg.control.action_scale[name] * cfg.normalization.clip_actions
        nominal = cfg.init_state.default_joint_angles[name]
        low, high = nominal - offset, nominal + offset
        if not np.isfinite([lower, upper, low, high]).all() or not (
            lower + 0.02 <= low <= high <= upper - 0.02
        ):
            raise ValueError(
                f"{name}: target [{low}, {high}] violates URDF [{lower}, {upper}] with 0.02 rad margin"
            )
        intervals[name] = [low, high]
    return intervals


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
        armature_groups = [LEG_JOINTS, WHEEL_JOINTS]
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


def add_arguments(parser):
    parameters = [
        {"name": "--go2w_finetune", "choices": ["coverage", "coverage_mobility", "precision_clearance"], "default": None, "help": "Explicit step_recovery_v1 continuation/evaluation design; unset preserves sampling and rewards"},
        {"name": "--go2w_profile", "choices": ["step_recovery_v1", "event_step_v1", "transfer_v1", "transfer_v2"], "default": None, "help": "Opt-in training recipe; replay restores the saved profile automatically"},
        {"name": "--sagittal_stance_weight", "type": float, "default": None, "help": "Explicit event_step_v1 stance weight; full-demand weight stays 0.12; select the saved value for evaluation/play"},
        {"name": "--event_quality_profile", "choices": ["sufficient_clearance"], "default": None, "help": "Opt-in event quality and payment; select the saved choice for evaluation/play"},
        {"name": "--zero_command_brake", "action": "store_true", "help": "Go2-W inference only: blend wheel targets to zero for a complete zero body command"},
        {"name": "--entropy_coef", "type": float, "default": None, "help": "Go2-W entropy weight; unset preserves the registered config"},
        {"name": "--tracking_sigma_x", "type": float, "default": None, "help": "Go2-W forward squared-error denominator; unset preserves the registered config"},
        {"name": "--skip_zero_action_probe", "action": "store_true", "help": "Bank evaluation: retain all policy cases, omit the equilibrium zero-action probe"},
        {"name": "--diagnostic_trace", "action": "store_true", "help": "Read substep control forces, summed ground loads and cylinder geometry"},
        {"name": "--reference_config", "help": "Explicit audited saved config, if not next to the checkpoint"},
        {"name": "--eval_mode", "choices": ["nominal", "bank", "equilibrium", "sustained", "closed_loop", "precision_screen", "precision_dr", "transfer_screen"], "default": "nominal"},
        {"name": "--transfer_armature", "choices": ["nominal", "low", "high"], "default": None, "help": "Transfer inference-only explicit motor armature: .01/.005/.02 kg m^2"},
        {"name": "--transfer_delay", "type": int, "choices": [0, 1, 2], "default": None, "help": "Transfer inference-only held action delay in policy steps"},
        {"name": "--transfer_cases", "nargs": "+", "choices": ["stand", "forward", "reverse", "yaw_positive", "yaw_negative", "lateral_positive", "lateral_negative", "mixed", *TRANSFER_V2_REVIEW_COMMANDS], "help": "transfer_screen subset; unset runs the small complete command panel"},
        {"name": "--bank_seed", "type": int, "default": 240925, "help": "Local NumPy generator for a fixed 32-condition bank"},
    ]
    for parameter in parameters:
        parameter = parameter.copy()
        parser.add_argument(parameter.pop("name"), **parameter)


def configure(env_cfg, cfg_train, args):
    if getattr(args, "_replay_restored", False):
        for cfg in (env_cfg, cfg_train):
            if cfg is not None:
                for key in ("go2w_profile", "go2w_finetune"):
                    explicit = getattr(args, key, None)
                    if explicit is not None and explicit != getattr(cfg, key, None):
                        raise ValueError(f"Explicit --{key}={explicit!r} conflicts with saved {getattr(cfg, key, None)!r}")
        return  # Saved rewards/commands remain authoritative for replay.
    profile = getattr(args, "go2w_profile", None)
    if profile is not None:
        if args.task != "go2w":
            raise ValueError("--go2w_profile is specific to go2w")
        if getattr(args, "zero_command_brake", False):
            raise ValueError("Go2-W step recovery requires zero-command braking disabled")
        entropy = 0.003 if profile in ("event_step_v1", "transfer_v1", "transfer_v2") else 0.001
        denominator = 0.09 if profile == "transfer_v2" else 0.25
        if getattr(args, "tracking_sigma_x", None) not in (None, denominator) or getattr(args, "entropy_coef", None) not in (None, entropy):
            raise ValueError(f"{profile} fixes tracking_sigma_x={denominator} and entropy_coef={entropy}")
        apply_go2w_profile(env_cfg, cfg_train, profile)
    finetune = getattr(args, "go2w_finetune", None)
    if finetune is not None:
        if args.task != "go2w" or profile != "step_recovery_v1":
            raise ValueError("--go2w_finetune requires go2w with --go2w_profile step_recovery_v1")
        apply_go2w_finetune(env_cfg, cfg_train, finetune)
    stance_weight = getattr(args, "sagittal_stance_weight", None)
    if stance_weight is not None:
        if args.task != "go2w" or profile != "event_step_v1":
            raise ValueError("--sagittal_stance_weight requires go2w with --go2w_profile event_step_v1")
        if not math.isfinite(stance_weight) or stance_weight <= 0:
            raise ValueError("--sagittal_stance_weight must be finite and positive")
        if env_cfg is not None:
            env_cfg.rewards.sagittal_stance_weight = stance_weight
    quality_profile = getattr(args, "event_quality_profile", None)
    if quality_profile is not None:
        if args.task != "go2w" or profile != "event_step_v1":
            raise ValueError("--event_quality_profile requires go2w with --go2w_profile event_step_v1")
        apply_go2w_event_quality(env_cfg, quality_profile)
    if getattr(args, "zero_command_brake", False) and args.task != "go2w":
        raise ValueError("--zero_command_brake is specific to go2w playback/evaluation")
    entropy = getattr(args, "entropy_coef", None)
    if entropy is not None:
        if args.task != "go2w":
            raise ValueError("--entropy_coef is specific to go2w")
        if not math.isfinite(entropy) or entropy < 0:
            raise ValueError("--entropy_coef must be finite and nonnegative")
        if cfg_train is not None:
            cfg_train.algorithm.entropy_coef = entropy
    sigma_x = getattr(args, "tracking_sigma_x", None)
    if sigma_x is not None:
        if args.task != "go2w":
            raise ValueError("--tracking_sigma_x is specific to go2w")
        if not math.isfinite(sigma_x) or sigma_x <= 0:
            raise ValueError("--tracking_sigma_x must be finite and positive")
        if env_cfg is not None:
            env_cfg.rewards.tracking_sigma_x = sigma_x


def validate_fresh_transfer(args, train_cfg):
    """Training-only guard, before any simulator/runner or checkpoint construction."""
    if getattr(train_cfg, "go2w_profile", None) not in ("transfer_v1", "transfer_v2"):
        return
    forbidden = ("resume", "load_run", "checkpoint", "reference_config", "go2w_finetune",
                 "sagittal_stance_weight", "event_quality_profile", "transfer_armature", "transfer_delay", "transfer_cases")
    if any(bool(getattr(args, key, False)) if key == "resume" else getattr(args, key, None) is not None
           for key in forbidden):
        raise ValueError(f"{train_cfg.go2w_profile} training is fresh only; omit resume/load/checkpoint/continuation arguments")
    if (train_cfg.runner.resume or train_cfg.runner.load_run is not None
            or train_cfg.runner.checkpoint is not None or train_cfg.runner.resume_path is not None):
        raise ValueError(f"{train_cfg.go2w_profile} training must not inherit checkpoint initialization")


def validate_training(args, env_cfg, train_cfg):
    validate_fresh_transfer(args, train_cfg)
    if getattr(args, "zero_command_brake", False):
        raise ValueError("Zero-command braking is inference-only")
    for key in ("go2w_finetune", "sagittal_stance_weight", "event_quality_profile"):
        if getattr(args, key, None) is not None and not train_cfg.runner.resume:
            raise ValueError(f"--{key} requires explicit full-state --resume")
    if getattr(args, "go2w_profile", None) and not train_cfg.runner.resume and (
        args.load_run is not None or args.checkpoint is not None):
        raise ValueError("Fresh profile training must omit --load_run and --checkpoint")
