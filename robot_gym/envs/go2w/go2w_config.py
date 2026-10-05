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
V3_JOINT_REFERENCE = {"hip": 0.0, "thigh": 0.70, "calf": -1.40, "foot": 0.0}
V3_SPAWN_CLEARANCE = 0.003  # Reset clearance, never added to the reward height.


def v3_joint_reference():
    return {f"{side}_{joint}_joint": angle for side in ("FL", "FR", "RL", "RR")
            for joint, angle in V3_JOINT_REFERENCE.items()}


def v3_reference_geometry(with_centers=False):
    """Design geometry from the measured URDF; no simulation or equilibrium claim."""
    from robot_gym.utils.urdf_reader import URDFReader
    from robot_gym.utils.diagnostics import urdf_link_poses, wheel_cylinders
    path = URDFReader(MEASURED_URDF).robot_file_path_absolute
    names = [f"{side}_foot" for side in ("FL", "FR", "RL", "RR")]
    offset, axis, radius, width = [x.cpu().numpy() for x in wheel_cylinders(path, names)]
    poses = urdf_link_poses(path, v3_joint_reference())
    centers = np.array([poses[n][:3, 3] + poses[n][:3, :3] @ offset[i] for i, n in enumerate(names)])
    axes = np.array([poses[n][:3, :3] @ axis[i] for i, n in enumerate(names)])
    gaps = centers[:, 2] - radius * np.sqrt(np.maximum(0, 1 - axes[:, 2] ** 2)) - width * abs(axes[:, 2])
    height = float(-gaps.min())
    if np.ptp(gaps) > 1e-4:
        raise ValueError("V3 reference has inconsistent wheel floor gaps")
    dx = [float(centers[i, 0] - poses[n.replace('_foot', '_thigh')][0, 3]) for i, n in enumerate(names)]
    return (height, dx, centers.tolist()) if with_centers else (height, dx)

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


def v3_completed_updates(saved_env, iteration):
    """Native resume repeats the loaded label; preserve the actual update lineage."""
    resume = saved_env.get("training_resume")
    if resume:
        return resume["source_total_updates"] + iteration - resume["source_iteration"] + 1
    return (saved_env.get("refinement_parent") or {}).get("prior_updates", 0) + iteration + 1


def prepare_v3_resume(env_cfg, train_cfg, args):
    """Restore one saved refinement recipe and all native learning state."""
    import copy
    from pathlib import Path
    import torch
    from robot_gym.utils.task_registry import task_registry
    from robot_gym.utils.helpers import class_to_dict, update_class_from_dict

    if args.load_run is None or not Path(args.load_run).is_dir() or args.checkpoint is None:
        raise ValueError("V3 resume requires an explicit run directory and checkpoint")
    if args.max_iterations is None or args.max_iterations <= 0:
        raise ValueError("V3 resume requires --max_iterations as a positive number of additional updates")
    for flag in ("go2w_finetune", "reference_config", "sagittal_stance_weight", "event_quality_profile",
                 "transfer_armature", "transfer_delay", "transfer_cases", "episode_length_s",
                 "entropy_coef", "tracking_sigma_x", "zero_command_brake"):
        value = getattr(args, flag, None)
        conflicting = bool(value) if flag == "zero_command_brake" else value is not None
        if conflicting:
            raise ValueError(f"Same-recipe resume preserves saved settings; omit --{flag}")
    if not hasattr(args, "v3_resume_recipe"):
        selection = copy.copy(args)
        selection.resume = False
        selection.go2w_profile = selection.go2w_finetune = None
        selection.load_run = str(Path(args.load_run).resolve())
        selection.experiment_name = selection.run_name = selection.num_envs = selection.max_iterations = None
        selection.logger = selection.seed = None
        source_env, source_train, checkpoint = task_registry.resolve_replay(selection)
        if source_env.go2w_profile != "transfer_v3" or not getattr(source_env, "go2w_finetune", None):
            raise ValueError("This resume path requires a saved V3 refinement")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        args.v3_resume_recipe = {"env_cfg": class_to_dict(source_env), "train_cfg": class_to_dict(source_train),
            "checkpoint": str(checkpoint), "source_iteration": int(saved["iter"]),
            "source_total_updates": v3_completed_updates(class_to_dict(source_env), int(saved["iter"]))}
    source = args.v3_resume_recipe
    if args.seed not in (None, source["train_cfg"]["seed"]):
        raise ValueError("Same-recipe resume retains the saved seed")
    if args.num_envs not in (None, source["env_cfg"]["env"]["num_envs"]) and not (args.num_envs == 64 and args.max_iterations <= 2):
        raise ValueError("Retain the saved environment count; 64 environments are allowed only for a <=2-update smoke")
    args.resolved_checkpoint = source["checkpoint"]
    for cfg, key in ((env_cfg, "env_cfg"), (train_cfg, "train_cfg")):
        if cfg is not None:
            update_class_from_dict(cfg, copy.deepcopy(source[key]))
    if env_cfg is not None:
        env_cfg.training_resume = {k: source[k] for k in ("checkpoint", "source_iteration", "source_total_updates")}
        env_cfg.training_resume["additional_updates"] = args.max_iterations
    if train_cfg is not None:
        train_cfg.runner.resume = True
        train_cfg.runner.resume_path = source["checkpoint"]
        train_cfg.runner.checkpoint_load_cfg = {"actor": True, "critic": True, "optimizer": True, "iteration": True}
        train_cfg.runner.experiment_name = source["train_cfg"]["runner"]["experiment_name"] + "_resume"
        train_cfg.runner.run_name = f"resume_from{source['source_iteration']}_seed{source['train_cfg']['seed']}"


def prepare_sensor_smooth(env_cfg, train_cfg, args):
    """Restore an explicit V3 parent, then apply the selected refinement delta once."""
    import copy
    from pathlib import Path
    import torch
    from robot_gym.utils.task_registry import task_registry
    from robot_gym.utils.helpers import class_to_dict, update_class_from_dict

    name = args.go2w_finetune
    phase_route = name == "sensor_phase_conditioned"
    exposure_route = name == "navigation_partial_lateral"
    zero_hold_route = name == "navigation_zero_hold"
    rolling_control_route = name == "navigation_rolling_control"
    placement_route = name in ("navigation_rolling_placement", "navigation_rolling_control")
    restore_route = phase_route or exposure_route or zero_hold_route or placement_route
    if args.task != "go2w" or args.go2w_profile != "transfer_v3" or args.resume:
        raise ValueError(f"{name} requires --task go2w --go2w_profile transfer_v3; omit --resume (fresh optimizer/counter)")
    for flag in ("reference_config", "sagittal_stance_weight", "event_quality_profile", "transfer_armature",
                 "transfer_delay", "entropy_coef", "tracking_sigma_x", "zero_command_brake"):
        value = getattr(args, flag, None)
        conflicting = bool(value) if flag == "zero_command_brake" else value is not None
        if conflicting:
            raise ValueError(f"{name} preserves the parent plant and settings; omit --{flag}")
    if not hasattr(args, "sensor_parent"):
        if args.load_run is None or not Path(args.load_run).is_dir() or args.checkpoint is None:
            raise ValueError(f"{name} needs an explicit --load_run directory path and --checkpoint number (or -1)")
        selection = copy.copy(args)
        selection.go2w_profile = selection.go2w_finetune = None
        selection.load_run = str(Path(args.load_run).resolve())
        selection.experiment_name = selection.run_name = selection.num_envs = selection.max_iterations = None
        selection.logger = selection.seed = None
        parent_env, parent_train, checkpoint = task_registry.resolve_replay(selection)
        if parent_env.go2w_profile != "transfer_v3":
            raise ValueError("Sensor refinement requires a saved transfer_v3 parent")
        if not restore_route and getattr(parent_env, "go2w_finetune", None):
            raise ValueError("sensor_smooth starts from an unrefined saved transfer_v3 parent")
        if phase_route and getattr(parent_env, "phase_observation_mode", "unconditional") != "unconditional":
            raise ValueError("Phase refinement changes the unconditional observation contract once")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        args.sensor_parent = {"env_cfg": class_to_dict(parent_env), "train_cfg": class_to_dict(parent_train),
                              "checkpoint": str(checkpoint), "iteration": int(saved["iter"]),
                              "total_updates": v3_completed_updates(class_to_dict(parent_env), int(saved["iter"]))}
        print(f"Fine-tune parent iteration {saved['iter']}; preserve both models/normalizers/std, fresh optimizer and local counter")
    parent = args.sensor_parent
    args.resolved_checkpoint = parent["checkpoint"]
    parent_experiment = parent["train_cfg"]["runner"]["experiment_name"]
    if args.experiment_name == parent_experiment:
        raise ValueError(f"{name} requires a separate output experiment")
    for cfg, key in ((env_cfg, "env_cfg"), (train_cfg, "train_cfg")):
        if cfg is not None and getattr(cfg, "go2w_finetune", None) != name:
            update_class_from_dict(cfg, copy.deepcopy(parent[key]))
            cfg.go2w_finetune = name
    if env_cfg is not None and not restore_route:
        env_cfg.sensor_smooth = {"hip_weight": 3., "long_hold_probability": .04,
            "extended_hold_probability": .015, "long_hold_s": [8., 15.], "extended_hold_s": [20., 30.]}
        env_cfg.env.episode_length_s = 60.
        env_cfg.rewards.scales.lin_vel_z = 0.
        env_cfg.rewards.scales.sensor_vertical_velocity = -1.
        env_cfg.rewards.scales.ang_vel_xy = -.075
        env_cfg.refinement_parent = {"checkpoint": parent["checkpoint"], "iteration": parent["iteration"],
                                     "prior_updates": parent["iteration"] + 1}
    if env_cfg is not None and restore_route:
        env_cfg.training_resume = None  # This is deliberately a new recipe, with fresh optimizer/counter.
        if phase_route:
            env_cfg.phase_observation_mode = "command_demand"
        if exposure_route:
            env_cfg.commands.pure_lateral_magnitude_range = [.02, .3]
        if zero_hold_route:
            # Complete-zero motion cost; excludes moving commands, including partial lateral/yaw.
            env_cfg.rewards.scales.stand_still = -5.
        if placement_route:
            _, _, centers = v3_reference_geometry(with_centers=True)
            y = np.asarray(centers)[:, 1]
            env_cfg.rewards.rolling_placement = {
                "width_reference_m": (y[[0, 2]] - y[[1, 3]]).tolist(),
                "midpoint_reference_m": ((y[[0, 2]] + y[[1, 3]]) / 2).tolist(),
                "width_deadband_m": .04, "width_scale_m": .10,
                "midpoint_deadband_m": .015, "midpoint_scale_m": .05,
            }
            env_cfg.rewards.scales.rolling_placement = -.05
        if rolling_control_route:
            # Combined placement/yaw refinement; stepping keeps the saved tracking scale.
            env_cfg.rewards.phase_objective["rolling_yaw_scale"] = .07
        previous = copy.deepcopy(parent["env_cfg"].get("refinement_parent"))
        env_cfg.refinement_parent = {"checkpoint": parent["checkpoint"], "iteration": parent["iteration"],
            "parent_local_updates": parent["total_updates"] - (previous or {}).get("prior_updates", 0),
            "prior_updates": parent["total_updates"],
            "previous_generation": previous}
    if train_cfg is not None:
        train_cfg.runner.resume = True  # Loading enabled; selective native options define initialization.
        train_cfg.runner.resume_path = parent["checkpoint"]
        train_cfg.runner.checkpoint_load_cfg = {"actor": True, "critic": True, "optimizer": False, "iteration": False}
        train_cfg.runner.experiment_name = f"go2w_transfer_v3_{name}"
        train_cfg.runner.run_name = f"{name}_from{parent['iteration']}_seed1"
        train_cfg.runner.max_iterations = 500 if zero_hold_route or placement_route else 150 if exposure_route else 200 if phase_route else 300
        train_cfg.runner.save_interval = 50
        if not restore_route:
            train_cfg.algorithm.learning_rate = 5e-5
            train_cfg.algorithm.schedule = "fixed"


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
    if name == "transfer_v3":
        new_env = env_cfg if env_cfg is not None and getattr(env_cfg, "go2w_profile", None) != name else None
        new_train = train_cfg if train_cfg is not None and getattr(train_cfg, "go2w_profile", None) != name else None
        apply_go2w_profile(new_env, new_train, "transfer_v1")
        if new_env is not None:
            c = new_env
            c.go2w_profile, c.go2w_behavior = name, "phase_guided"
            c.env.num_observations = 58
            c.init_state.default_joint_angles = v3_joint_reference()
            height, dx = v3_reference_geometry()
            c.init_state.pos = [0., 0., height + V3_SPAWN_CLEARANCE]
            c.init_state.rot = [1., 0., 0., 0.]
            c.rewards.base_height_target = height
            c.phase_guidance = {"period_s": .8, "stance_fraction": .65, "apex_m": .04,
                                "offsets": {"FL_foot": 0., "FR_foot": .5, "RL_foot": .5, "RR_foot": 0.},
                                "randomize_reset": True}
            c.rewards.phase_objective = {"tracking_scales": [.25, .15, .35],
                "height_tolerance_m": .015, "height_scale_m": .05,
                "clearance_scale_m": .04, "corridor_stance_m": .04,
                "corridor_swing_m": .09, "corridor_scale_m": .05,
                "wheel_thigh_dx_reference_m": dx}
            # Replace the legacy stack, including all event/credit dependencies.
            for key in dir(c.rewards.scales):
                if not key.startswith("_"):
                    setattr(c.rewards.scales, key, 0.)
            for key, scale in {"tracking_x": 1., "tracking_y": 1., "tracking_yaw": .8,
                               "phase_clearance": -1., "phase_support": -.5,
                               "orientation": -4., "reference_height": -1., "wheel_corridor": -.5,
                               "rolling_pose": -.5, "normalized_effort": -.03,
                               "leg_action_rate": -.01, "wheel_action_rate": -.005,
                               "lin_vel_z": -.2, "ang_vel_xy": -.05,
                               "insufficient_support": -.5, "collision": -2.,
                               "dof_pos_limits": -2., "torque_limits": -.5,
                               "lateral_wheel_scrub": -.2, "termination": -5.}.items():
                setattr(c.rewards.scales, key, scale)
            c.rewards.only_positive_rewards = False
            c.rewards.discrete_reward_names = ["termination"]
            c.commands.curriculum = False
            c.commands.stand_command_probability = .15
            c.commands.moving_mixture_probabilities = [.20, .07, .20, .10, .20, .08]
            c.commands.pure_lateral_magnitude_range = [.1, .3]
            c.commands.pure_yaw_magnitude_range = [.3, .8]
            c.commands.ranges.lin_vel_x = [-.35, .6]
            c.commands.ranges.lin_vel_y = [-.3, .3]
            c.commands.ranges.ang_vel_yaw = [-.8, .8]
            c.commands.phase_mixed_ranges = [[-.3, .3], [-.2, .2], [-.5, .5]]
            c.commands.discovery_segment_probability = .8
            c.commands.discovery_segment_duration_range = [2., 4.]
        if new_train is not None:
            new_train.go2w_profile, new_train.go2w_behavior = name, "phase_guided"
            new_train.runner.experiment_name = "go2w_transfer_v3"
            new_train.runner.run_name = "transfer_v3_seed1"
        return
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
    sensor_smooth = None  # Opt-in refinement only; saved V3 remains unchanged.
    phase_observation_mode = "unconditional"  # Missing mode in older saved recipes keeps their contract.
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
        capture_precision = False  # Wheel geometry only for bounded evaluation.
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
        {"name": "--go2w_finetune", "choices": ["coverage", "coverage_mobility", "precision_clearance", "sensor_smooth", "sensor_phase_conditioned", "navigation_partial_lateral", "navigation_zero_hold", "navigation_rolling_placement", "navigation_rolling_control"], "default": None, "help": "V3 refinements selectively load explicit saved models/normalizers/std with fresh optimizer/counter; legacy choices require step_recovery_v1 resume"},
        {"name": "--go2w_profile", "choices": ["step_recovery_v1", "event_step_v1", "transfer_v1", "transfer_v2", "transfer_v3"], "default": None, "help": "Opt-in training recipe; replay restores the saved profile automatically"},
        {"name": "--sagittal_stance_weight", "type": float, "default": None, "help": "Explicit event_step_v1 stance weight; full-demand weight stays 0.12; select the saved value for evaluation/play"},
        {"name": "--event_quality_profile", "choices": ["sufficient_clearance"], "default": None, "help": "Opt-in event quality and payment; select the saved choice for evaluation/play"},
        {"name": "--zero_command_brake", "action": "store_true", "help": "Go2-W inference only: blend wheel targets to zero for a complete zero body command"},
        {"name": "--entropy_coef", "type": float, "default": None, "help": "Go2-W entropy weight; unset preserves the registered config"},
        {"name": "--tracking_sigma_x", "type": float, "default": None, "help": "Go2-W forward squared-error denominator; unset preserves the registered config"},
        {"name": "--skip_zero_action_probe", "action": "store_true", "help": "Bank evaluation: retain all policy cases, omit the equilibrium zero-action probe"},
        {"name": "--diagnostic_trace", "action": "store_true", "help": "Read substep control forces, summed ground loads and cylinder geometry"},
        {"name": "--reference_config", "help": "Explicit audited saved config, if not next to the checkpoint"},
        {"name": "--eval_mode", "choices": ["nominal", "bank", "equilibrium", "sustained", "closed_loop", "precision_screen", "precision_dr", "transfer_screen", "sensor_sustained"], "default": "nominal"},
        {"name": "--eval_phase_offset", "type": float, "default": 0., "help": "sensor_sustained only: initial phase offset in cycles [0,1); phase still advances normally"},
        {"name": "--eval_rolling_phase_zero", "action": "store_true", "help": "Isolated sensor_sustained diagnostic: zero raw phase only at zero step demand; stand/forward plus roll-lateral-roll transition; latent clock and rewards unchanged"},
        {"name": "--eval_phase_transition", "action": "store_true", "help": "sensor_sustained only: one bounded roll/partial-step/roll/stop case, using the saved phase encoding"},
        {"name": "--eval_noise_pushes", "action": "store_true", "help": "precision_dr only: add the saved observation noise and push settings to the existing eight-condition check"},
        {"name": "--transfer_armature", "choices": ["nominal", "low", "high"], "default": None, "help": "Transfer inference-only explicit motor armature: .01/.005/.02 kg m^2"},
        {"name": "--transfer_delay", "type": int, "choices": [0, 1, 2], "default": None, "help": "Transfer inference-only held action delay in policy steps"},
        {"name": "--transfer_cases", "nargs": "+", "choices": ["stand", "forward", "reverse", "yaw_positive", "yaw_negative", "lateral_positive", "lateral_negative", "mixed", *TRANSFER_V2_REVIEW_COMMANDS], "help": "transfer_screen or sensor_sustained subset; unset runs its complete panel"},
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
    if args.resume and getattr(args, "go2w_profile", None) == "transfer_v3":
        prepare_v3_resume(env_cfg, cfg_train, args)
        return
    if getattr(args, "go2w_finetune", None) in ("sensor_smooth", "sensor_phase_conditioned", "navigation_partial_lateral", "navigation_zero_hold", "navigation_rolling_placement", "navigation_rolling_control"):
        prepare_sensor_smooth(env_cfg, cfg_train, args)
        return
    profile = getattr(args, "go2w_profile", None)
    if profile is not None:
        if args.task != "go2w":
            raise ValueError("--go2w_profile is specific to go2w")
        if getattr(args, "zero_command_brake", False):
            raise ValueError("Go2-W step recovery requires zero-command braking disabled")
        entropy = 0.003 if profile in ("event_step_v1", "transfer_v1", "transfer_v2", "transfer_v3") else 0.001
        if profile == "transfer_v3" and getattr(args, "tracking_sigma_x", None) is not None:
            raise ValueError("transfer_v3 uses configured Huber tracking scales; --tracking_sigma_x is a legacy Gaussian option")
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
    if getattr(train_cfg, "go2w_profile", None) not in ("transfer_v1", "transfer_v2", "transfer_v3"):
        return
    if args.resume and getattr(args, "v3_resume_recipe", None):
        expected = {"actor": True, "critic": True, "optimizer": True, "iteration": True}
        if train_cfg.go2w_profile != "transfer_v3" or not train_cfg.runner.resume or train_cfg.runner.checkpoint_load_cfg != expected:
            raise ValueError("Same-recipe V3 resume must restore the full native learning state")
        return
    if getattr(train_cfg, "go2w_finetune", None) in ("sensor_smooth", "sensor_phase_conditioned", "navigation_partial_lateral", "navigation_zero_hold", "navigation_rolling_placement", "navigation_rolling_control"):
        expected = {"actor": True, "critic": True, "optimizer": False, "iteration": False}
        if (train_cfg.go2w_profile != "transfer_v3" or args.resume or not train_cfg.runner.resume
                or train_cfg.runner.checkpoint_load_cfg != expected or not getattr(args, "sensor_parent", None)):
            raise ValueError("Sensor refinement requires explicit selective initialization from its resolved V3 parent")
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
    if getattr(args, "v3_resume_recipe", None):
        from robot_gym.utils.helpers import class_to_dict
        source = args.v3_resume_recipe
        for key in ("commands", "rewards", "control", "domain_rand", "sim", "asset", "init_state", "noise",
                    "normalization", "phase_guidance", "phase_observation_mode", "refinement_parent", "sensor_smooth"):
            current = getattr(env_cfg, key)
            if class_to_dict(current) != source["env_cfg"][key]:
                raise ValueError(f"Same-recipe resume changed env_cfg.{key}")
        for key in ("actor", "critic", "algorithm"):
            if class_to_dict(getattr(train_cfg, key)) != source["train_cfg"][key]:
                raise ValueError(f"Same-recipe resume changed train_cfg.{key}")
    if any(getattr(args, flag, False) for flag in ("eval_rolling_phase_zero", "eval_phase_transition", "eval_noise_pushes")):
        raise ValueError("Evaluation diagnostics are inference only")
    if getattr(args, "zero_command_brake", False):
        raise ValueError("Zero-command braking is inference-only")
    for key in ("go2w_finetune", "sagittal_stance_weight", "event_quality_profile"):
        if getattr(args, key, None) is not None and not train_cfg.runner.resume:
            raise ValueError(f"--{key} requires explicit full-state --resume")
    if getattr(args, "go2w_profile", None) and not train_cfg.runner.resume and (
        args.load_run is not None or args.checkpoint is not None):
        raise ValueError("Fresh profile training must omit --load_run and --checkpoint")
