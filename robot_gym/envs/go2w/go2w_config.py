"""The single current Go2-W training configuration.

Edit parameters here for the next experiment. Every run saves its resolved
config.yaml; historical training presets are deliberately not reconstructed.
"""

import copy
import math
from functools import lru_cache
from types import SimpleNamespace

import numpy as np

from robot_gym.envs.base.base_config import BaseConfig
from .straight_motion import CRITIC_REFERENCE_FIELDS


SIDES = ("FL", "FR", "RL", "RR")
# One numeric reference for action offsets, q-q_ref observations, pose reward,
# nominal resets and measured-URDF FK. Wheel angles are excluded from pose cost.
JOINT_REFERENCE = {"hip": 0.0, "thigh": 0.70, "calf": -1.40, "foot": 0.0}
LEG_JOINTS = [f"{side}_{joint}_joint" for side in SIDES for joint in JOINT_REFERENCE if joint != "foot"]
WHEEL_JOINTS = [f"{side}_foot_joint" for side in SIDES]
MEASURED_URDF = "go2w_measured_ed8dc93.urdf"
MEASURED_SHA256 = "d298cc7bf4894e869840bdab9ac60f09548d998444018e61854636cff46b1d8c"
SPAWN_CLEARANCE = 0.003  # Reset clearance, never added to the reward height.


def joint_reference():
    """Independent named copy of the canonical reference for all four legs."""
    return {f"{side}_{joint}_joint": angle for side in SIDES
            for joint, angle in JOINT_REFERENCE.items()}


@lru_cache(maxsize=2)
def reference_geometry(with_centers=False):
    """Design geometry from the measured URDF; no simulation or equilibrium claim."""
    from robot_gym.utils.urdf_reader import URDFReader
    from robot_gym.utils.diagnostics import urdf_link_poses, wheel_cylinders
    path = URDFReader(MEASURED_URDF).robot_file_path_absolute
    names = [f"{side}_foot" for side in ("FL", "FR", "RL", "RR")]
    offset, axis, radius, width = [x.cpu().numpy() for x in wheel_cylinders(path, names)]
    poses = urdf_link_poses(path, joint_reference())
    centers = np.array([poses[n][:3, 3] + poses[n][:3, :3] @ offset[i] for i, n in enumerate(names)])
    axes = np.array([poses[n][:3, :3] @ axis[i] for i, n in enumerate(names)])
    gaps = centers[:, 2] - radius * np.sqrt(np.maximum(0, 1 - axes[:, 2] ** 2)) - width * abs(axes[:, 2])
    height = float(-gaps.min())
    if np.ptp(gaps) > 1e-4:
        raise ValueError("Go2-W reference has inconsistent wheel floor gaps")
    dx = [float(centers[i, 0] - poses[n.replace('_foot', '_thigh')][0, 3]) for i, n in enumerate(names)]
    return (height, dx, centers.tolist()) if with_centers else (height, dx)


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


_REFERENCE_HEIGHT, _REFERENCE_DX = reference_geometry()


def _own_fields(config):
    """Give each config instance its own mutable values before CLI or replay edits."""
    for key in dir(config):
        if key.startswith('_'):
            continue
        value = getattr(config, key)
        if callable(value):
            continue
        if hasattr(value, '__dict__'):
            _own_fields(value)
        else:
            setattr(config, key, copy.deepcopy(value))


class GO2WCfg(BaseConfig):
    def __init__(self):
        super().__init__()
        _own_fields(self)

    class asset:
        contact_height = 0.09167
        default_armature = 0.01
        file_format = 'urdf'
        foot_link_names = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
        hip_abduction_indices = [0, 4, 8, 12]
        joint_names = list(joint_reference())
        links_to_keep = None
        merge_fixed_links = True
        name = 'go2w'
        robot_file = MEASURED_URDF
        robot_name = 'go2w'
        sha256 = MEASURED_SHA256
        source_commit = 'ed8dc93b3065a8e2a3a5919f032ed4690529117c'

    class commands:
        # Required navigation envelope. Reserve limits belong only to sampling.
        curriculum = False
        num_commands = 3
        class ranges:
            ang_vel_yaw = [-1.0, 1.0]
            lin_vel_x = [-0.3, 1.0]
            lin_vel_y = [-0.3, 0.3]

        stand_threshold = 1e-06
        linear_deadzone = 0.01
        yaw_deadzone = 0.01
        stand_command_probability = 0.15
        moving_mixture_probabilities = [0.2, 0.07, 0.2, 0.1, 0.2, 0.08]
        # Order: straight, arc, pure yaw, precision, pure lateral, mixed.
        pure_lateral_magnitude_range = [0.02, 0.3]
        pure_yaw_magnitude_range = [0.05, 1.0]
        short_command_duration_range = [0.5, 1.0]
        sustained_command_duration_range = [1.5, 3.0]
        sustained_command_probability = 0.3
        long_stand_probability = 0.25
        long_stand_duration_range = [3.0, 6.0]
        discovery_segment_probability = 0.8
        discovery_segment_duration_range = [2.0, 4.0]
        sampling = {
            'tier_probabilities': [0.8, 0.1, 0.1],
            'reserve_ranges': {
                'lin_vel_x': [-0.4, 1.2],
                'lin_vel_y': [-0.5, 0.5],
                'ang_vel_yaw': [-1.5, 1.5],
            },
            'precision_ranges': [[-0.063, 0.108], [0.0, 0.0], [-0.16, 0.16]],
        }
        holds = {
            'eligible_families': ['stand', 'straight', 'arc', 'lateral'],
            'long_hold_probability': 0.04,
            'extended_hold_probability': 0.015,
            'long_hold_s': [8.0, 15.0],
            'extended_hold_s': [20.0, 30.0],
        }

    class control:
        # Fixed reference offsets for P legs; velocity targets for V wheels.
        action_scale = {**{name: {"hip": .30, "thigh": .35, "calf": .40}[name.split("_")[1]]
                                     for name in LEG_JOINTS}, **dict.fromkeys(WHEEL_JOINTS, 18.0)}
        armature = dict.fromkeys(LEG_JOINTS + WHEEL_JOINTS, .01)
        armature_override = None
        control_type = {**dict.fromkeys(LEG_JOINTS, "P"), **dict.fromkeys(WHEEL_JOINTS, "V")}
        damping = dict.fromkeys(LEG_JOINTS + WHEEL_JOINTS, 1.0)
        decimation = 4
        dof_vel_limits = {name: 20.07 if "calf" in name else 30.1 for name in LEG_JOINTS + WHEEL_JOINTS}
        passive_damping = 0.0
        passive_frictionloss = 0.0
        passive_stiffness = 0.0
        stiffness = {**dict.fromkeys(LEG_JOINTS, 40.0), **dict.fromkeys(WHEEL_JOINTS, 0.0)}
        wheel_velocity_target_limit = 20.0

    class domain_rand:
        action_delay_steps_range = [0, 2]
        added_mass_range = [-0.5, 1.5]
        armature_groups = [LEG_JOINTS.copy(), WHEEL_JOINTS.copy()]
        armature_range = [0.005, 0.02]
        com_shift_range = [-0.015, 0.015]
        friction_links = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
        friction_range = [0.6, 1.2]
        kd_scale_range = [0.85, 1.15]
        kp_scale_range = [0.85, 1.15]
        push_duration_range_s = [0.1, 0.2]
        # One tier, magnitude and horizontal direction per event, held at base COM.
        push_force_mixture = {
            'probabilities': [0.8, 0.15, 0.05],
            'magnitude_ranges_n': [[20.0, 50.0], [50.0, 100.0], [100.0, 150.0]],
        }
        push_interval_range_s = [5.0, 10.0]
        push_robots = True
        randomize_action_delay = True
        randomize_armature = True
        randomize_base_mass = True
        randomize_com = True
        randomize_friction = True
        randomize_kd = True
        randomize_kp = True
        scale_base_inertia_with_mass = True

    class env:
        capture_precision = False
        capture_transitions = False
        episode_length_s = 60.0
        num_actions = 16
        num_envs = 4096
        num_observations = 58
        num_privileged_obs = num_observations + len(CRITIC_REFERENCE_FIELDS)
        play_mode = False
        record_command_families = True
        send_timeouts = True

    class init_state:
        angular_velocity_noise = 0.03
        default_joint_angles = joint_reference()  # FL/FR/RL/RR all use JOINT_REFERENCE above.
        joint_position_noise = 0.03
        joint_velocity_noise = 0.05
        linear_velocity_noise = 0.03
        orientation_noise = [0.02, 0.02, 0.05]
        pos = [0.0, 0.0, _REFERENCE_HEIGHT + SPAWN_CLEARANCE]
        rot = [1.0, 0.0, 0.0, 0.0]

    class noise:
        add_noise = True
        noise_level = 1.0
        class noise_scales:
            ang_vel = 0.08
            dof_pos = 0.01
            dof_vel = 0.2
            gravity = 0.02
            height_measurements = 0.1
            lin_vel = 0.05
            wheel_vel = 0.5


    class normalization:
        clip_actions = 1.0
        clip_observations = 100.0
        class obs_scales:
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 1.0
            height_measurements = 1.0
            lin_vel = 1.0


    phase_guidance = {
        'period_s': 0.8,
        'stance_fraction': 0.65,
        'apex_m': 0.04,
        'offsets': {'FL_foot': 0.0, 'FR_foot': 0.5, 'RL_foot': 0.5, 'RR_foot': 0.0},
        'randomize_reset': True,
    }
    phase_observation_mode = 'command_demand'
    # Training supervision only. Any changed command, including speed-only,
    # starts a segment; repeated identical values never move the reference.
    straight_motion = {
        'min_speed_m_s': 0.01,
        'yaw_zero_rad_s': 1e-6,  # Numerical zero, not a small-yaw command deadzone.
        'heading_projection_min': 1e-6,
        'cross_tolerance_m': 0.005,
        'cross_scale_m': 0.10,
        'heading_scale_rad': 0.10,
        'ramp_s': 0.5,
        'critic_time_scale_s': 30.0,
    }
    class rewards:
        # Reward rates; the shared accumulator applies policy dt exactly once.
        # Termination is the sole discrete event penalty.
        base_height_target = _REFERENCE_HEIGHT
        common_tracking = {
            'broad_scales': [0.25, 0.15, 0.35],
            'precision_scales': [0.03, 0.03, 0.03],
            'beta': [0.25, 0.25, 0.25],
        }
        contact_force_threshold = 8.0
        discrete_reward_names = ['termination']
        only_positive_rewards = False
        phase_objective = {
            'height_tolerance_m': 0.015,
            'height_scale_m': 0.05,
            'clearance_scale_m': 0.04,
            'corridor_stance_m': 0.04,
            'corridor_swing_m': 0.09,
            'corridor_scale_m': 0.05,
            'wheel_thigh_dx_reference_m': list(_REFERENCE_DX),
        }
        reference_pose = {
            'scales_rad': {'hip': 0.08, 'thigh': 0.12, 'calf': 0.12},
            'lateral_full_m_s': 0.03,
            'yaw_full_rad_s': 0.25,
            'stepping_fraction': 0.05,
        }
        class scales:
            ang_vel_xy = -0.075
            contact_safety = -2.0
            dof_pos_limits = -2.0
            insufficient_support = -0.5
            lateral_wheel_scrub = -0.2
            leg_action_rate = -0.01
            normalized_effort = -0.03
            orientation = -4.0
            phase_clearance = -1.0
            phase_support = -0.5
            reference_height = -1.0
            reference_pose = -0.5
            sensor_vertical_velocity = -1.0
            straight_cross_track = -0.1
            straight_heading = -0.1
            termination = -5.0
            torque_limits = -0.5
            tracking_x = 1.0
            tracking_y = 1.0
            tracking_yaw = 0.8
            wheel_action_rate = -0.005
            wheel_corridor = -0.5
            wheel_rate_zero = -0.1

        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 0.9

    seed = 1
    class sim:
        batch_dofs_info = True
        batch_links_info = True
        constraint_timeconst = 0.01
        contact_resolution = 'convex'
        deterministic = False
        dt = 0.005
        enable_collision = True
        enable_joint_limit = True
        enable_multi_contact = True
        enable_rolling_friction = False
        enable_self_collision = True
        enable_torsional_friction = False
        friction_cone = 'pyramidal'
        friction_rolling = 0.0
        friction_torsional = 0.0
        gravity = [0.0, 0.0, -9.81]
        ground_friction = 0.1
        integrator = 'approximate_implicitfast'
        iterations = 50
        ls_iterations = 50
        performance_mode = True
        substeps = 1
        up_axis = 1

    class termination:
        base_height_threshold = 0.33
        pitch_threshold = 0.5235987755982988
        roll_threshold = 0.5235987755982988

    class terrain:
        border_flat = False
        color = [0.18, 0.18, 0.21]
        curriculum = False
        heightfield = None
        horizontal_scale = 0.25
        mixed = {
            'options': ['flat_terrain', 'random_uniform_terrain', 'wave_terrain', 'pyramid_sloped_terrain', 'pyramid_stairs_terrain', 'stepping_stones_terrain', 'fractal_terrain'],
            'probs': [0.05, 0.2, 0.2, 0.2, 0.1, 0.2, 0.05],
        }
        mode = 'plane'
        n_subterrains = [7, 7]
        name = 'go2w_training_terrain'
        options = ['plane',
         'flat_terrain',
         'random_uniform_terrain',
         'pyramid_sloped_terrain',
         'discrete_obstacles_terrain',
         'wave_terrain',
         'pyramid_stairs_terrain',
         'stepping_stones_terrain',
         'fractal_terrain',
         'mixed']
        pos = None
        probs = [0.25, 0.05, 0.1, 0.15, 0.1, 0.15, 0.15, 0.05, 0.05, 0.0]
        randomize = False
        spawn_flat_radius_sub = 0
        subterrain_size = [4.0, 4.0]
        terrain_kwargs = {
            'discrete_obstacles_terrain': {'max_height': 0.05, 'max_size': 1.0, 'min_size': 0.25, 'num_rects': 5},
            'flat_terrain': {},
            'fractal_terrain': {'levels': 8, 'scale': 3.5},
            'pyramid_sloped_terrain': {'slope': 0.25},
            'pyramid_stairs_terrain': {'step_height': -0.075, 'step_width': 0.5},
            'random_uniform_terrain': {
                'downsampled_scale': 0.5,
                'max_height': 0.05,
                'min_height': -0.04,
                'step': 0.01,
            },
            'stepping_stones_terrain': {
                'max_height': 0.04,
                'platform_size': 0.0,
                'stone_distance': 0.075,
                'stone_size': 0.5,
            },
            'wave_terrain': {'amplitude': 0.06, 'num_waves': 2.0},
        }
        vertical_scale = 0.005

    training_resume = None
    class viewer:
        fov = 40
        lookat = [0.0, 0.0, 0.5]
        max_fps = 60
        pos = [2.0, 0.0, 2.5]
        print_debug_velocities = False
        ref_env = [0]
        show_world_frame = True
        velocity_arrow_radius = 0.03
        velocity_arrow_scale = 0.6
        visualize_foot_contacts = False
        visualize_velocity_arrows = False

    config_version = 2  # Relative geometric rewards and 58+7 asymmetric critic.


class GO2WCfgPPO(BaseConfig):
    def __init__(self):
        super().__init__()
        _own_fields(self)

    class actor:
        activation = 'elu'
        class_name = 'MLPModel'
        distribution_cfg = {
            'class_name': 'GaussianDistribution',
            'init_std': 0.4,
            'std_type': 'log',
            'std_range': [0.1, 0.7],
            'learn_std': True,
        }
        hidden_dims = [512, 256, 128]
        obs_normalization = True

    class algorithm:
        class_name = 'PPO'
        clip_param = 0.2
        desired_kl = 0.01
        entropy_coef = 0.003
        gamma = 0.995
        lam = 0.95
        learning_rate = 0.0003
        max_grad_norm = 1.0
        normalize_advantage_per_mini_batch = False
        num_learning_epochs = 5
        num_mini_batches = 8
        rnd_cfg = None
        schedule = 'fixed'
        symmetry_cfg = {
            'data_augmentation_func': 'robot_gym.envs.go2w.go2w_symmetry:sagittal_augmentation',
            'use_data_augmentation': True,
            'use_mirror_loss': False,
            'mirror_loss_coeff': 0.0,
        }
        use_clipped_value_loss = True
        use_mixed_precision = False
        value_loss_coef = 1.0

    class critic:
        activation = 'elu'
        class_name = 'MLPModel'
        hidden_dims = [512, 256, 128]
        obs_normalization = True

    class runner:
        checkpoint = None
        checkpoint_load_cfg = None
        experiment_name = 'go2w'
        load_run = None
        log_wandb = True
        logger = 'tensorboard'
        max_iterations = 2000
        num_steps_per_env = 64
        obs_groups = {'actor': ['policy'], 'critic': ['critic']}
        resume = False
        resume_path = None
        run_name = ''
        save_interval = 250
        wandb_project = 'go2w-locomotion'

    runner_class_name = 'OnPolicyRunner'
    seed = 1


def add_arguments(parser):
    parameters = [
        {"name": "--resume_current_reward_scales", "action": "store_true", "help": "Go2-W training resume only: replace saved reward scales with current GO2WCfg scales; retain all other saved settings and learning state"},
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
        {"name": "--transfer_cases", "nargs": "+", "choices": ["stand", "forward", "reverse", "yaw_positive", "yaw_negative", "lateral_positive", "lateral_negative", "mixed", "forward_fast", "lateral_strong_positive", "lateral_strong_negative", "yaw_strong_positive", "yaw_strong_negative"], "help": "transfer_screen or sensor_sustained subset; unset runs its complete panel"},
        {"name": "--bank_seed", "type": int, "default": 240925, "help": "Local NumPy generator for a fixed 32-condition bank"},
    ]
    for parameter in parameters:
        parameter = parameter.copy()
        parser.add_argument(parameter.pop("name"), **parameter)


def restore_saved_config(saved):
    """Restore complete supported settings, never fill missing physics with defaults.

    Schema 2 deliberately changes training rewards and critic inputs. Old models
    require their source checkout; retained raw traces remain comparison evidence.
    """
    from robot_gym.utils.helpers import class_to_dict

    data = copy.deepcopy(saved)
    env = data["env_cfg"]
    validate_schema(env)
    active = {k for k, v in env["rewards"]["scales"].items() if v}
    expected = set(class_to_dict(GO2WCfg().rewards.scales))
    if active != expected or env.get("phase_observation_mode") != "command_demand":
        raise ValueError("Unsupported saved Go2-W reward/phase contract; use its historical source checkout")

    def restore(template, values, path):
        result = SimpleNamespace()
        for key in class_to_dict(template):
            shape = getattr(template, key)
            if key not in values:
                raise ValueError(f"Incomplete saved Go2-W config: missing {path}.{key}")
            value = values[key]
            if isinstance(shape, dict) and isinstance(value, dict) and not shape.keys() <= value.keys():
                raise ValueError(f"Incomplete saved Go2-W config: missing fields in {path}.{key}")
            if hasattr(shape, "__dict__"):
                value = restore(shape, value, path + "." + key)
            setattr(result, key, copy.deepcopy(value))
        return result

    return (restore(GO2WCfg(), env, "env_cfg"),
            restore(GO2WCfgPPO(), data["train_cfg"], "train_cfg"))


def validate_schema(saved):
    if saved.get('config_version') != GO2WCfg.config_version:
        raise ValueError("Go2-W training schema changed: use the checkpoint's historical source checkout; old rewards/critic inputs will not be replaced by current defaults")


def configure(env_cfg, train_cfg, args):
    """Restore saved settings; optionally take only reward scales from current source."""
    current_rewards = getattr(args, "resume_current_reward_scales", False)
    if current_rewards and (not args.resume or getattr(args, "_replay_restored", False)):
        raise ValueError("--resume_current_reward_scales requires training --resume")
    if getattr(args, "_replay_restored", False) or not args.resume:
        return
    if not hasattr(args, "_go2w_resume_config"):
        if args.load_run is None:
            raise ValueError("Go2-W --resume requires an explicit --load_run saved directory")
        from robot_gym.utils.helpers import class_to_dict
        from robot_gym.utils.task_registry import task_registry
        # Capture the one current source before saved-task restoration replaces defaults.
        if current_rewards:
            args._go2w_current_reward_scales = copy.deepcopy(class_to_dict(GO2WCfg().rewards.scales))
        replay_args = copy.copy(args)
        replay_args.resume = False
        replay_args.resume_current_reward_scales = False
        restored_env, restored_train, checkpoint = task_registry.resolve_replay(replay_args)
        args.resolved_checkpoint = str(checkpoint)
        restored_env.training_resume = {"checkpoint": str(checkpoint), "state": "full native learning state"}
        if current_rewards:
            import hashlib
            import json
            from pathlib import Path
            saved_scales = class_to_dict(restored_env.rewards.scales)
            current_scales = args._go2w_current_reward_scales
            if {k for k, v in saved_scales.items() if v} != {k for k, v in current_scales.items() if v}:
                raise ValueError("Current reward scales must retain the saved active reward names")
            if not all(math.isfinite(v) for v in current_scales.values()):
                raise ValueError("Current reward scales must be finite")
            diff = {k: {"old": saved_scales.get(k, 0.), "new": v}
                    for k, v in current_scales.items() if saved_scales.get(k, 0.) != v}
            print("Current reward scale diff (unscaled): " + json.dumps(diff, sort_keys=True), flush=True)
            config_path = Path(args.reference_config or checkpoint.with_name("config.yaml")).resolve()
            restored_env.training_resume.update(
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                source_config=str(config_path),
                source_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                reward_scale_override={"source": "GO2WCfg.rewards.scales", "diff": diff},
            )
            # Before environment registration: its normal reward preparation applies dt once.
            restored_env.rewards.scales = SimpleNamespace(**copy.deepcopy(current_scales))
        restored_train.runner.checkpoint_load_cfg = dict.fromkeys(("actor", "critic", "optimizer", "iteration"), True)
        args._go2w_resume_config = (restored_env, restored_train)
    for current, restored in zip((env_cfg, train_cfg), args._go2w_resume_config):
        if current is not None:
            # Replace sections rather than overlaying the current numeric defaults.
            current.__dict__.update(copy.deepcopy(vars(restored)))


def validate_training(args, env_cfg, train_cfg):
    """Fail before simulator construction for accidental checkpoint initialization."""
    if args.resume:
        expected = dict.fromkeys(("actor", "critic", "optimizer", "iteration"), True)
        if (not getattr(args, "resolved_checkpoint", None) or not train_cfg.runner.resume
                or train_cfg.runner.checkpoint_load_cfg != expected):
            raise ValueError("Go2-W resume must restore models, normalizers, std, Adam and iteration")
    else:
        if (train_cfg.runner.resume or getattr(args, "resolved_checkpoint", None)
                or any(getattr(args, k, None) is not None for k in ("load_run", "checkpoint", "reference_config"))
                or any(getattr(train_cfg.runner, k) is not None for k in
                       ("load_run", "checkpoint", "resume_path", "checkpoint_load_cfg"))):
            raise ValueError("Fresh Go2-W training must not load a checkpoint or learning state")
        if env_cfg.init_state.default_joint_angles != joint_reference():
            raise ValueError("Go2-W requires the canonical four-leg reference")
        height, _ = reference_geometry()
        if (not math.isclose(env_cfg.rewards.base_height_target, height, abs_tol=1e-9, rel_tol=0.)
                or not math.isclose(env_cfg.init_state.pos[2], height + SPAWN_CLEARANCE, abs_tol=1e-9, rel_tol=0.)):
            raise ValueError("Go2-W reference height and spawn must derive from the measured URDF")
    if (env_cfg.phase_observation_mode != "command_demand"
            or env_cfg.env.num_observations != 58 or env_cfg.env.num_actions != 16):
        raise ValueError("Go2-W requires the demand-conditioned 58/16 interface")
    if (env_cfg.env.num_privileged_obs != env_cfg.env.num_observations + len(CRITIC_REFERENCE_FIELDS)
            or train_cfg.runner.obs_groups != {'actor': ['policy'], 'critic': ['critic']}):
        raise ValueError("Go2-W requires an isolated 58-input actor and 65-input relative-state critic")
    c = env_cfg.straight_motion
    if (not all(math.isfinite(v) for v in c.values())
            or any(c[k] <= 0 for k in ('min_speed_m_s', 'heading_projection_min', 'cross_scale_m',
                                      'heading_scale_rad', 'ramp_s', 'critic_time_scale_s'))
            or not 0 <= c['yaw_zero_rad_s'] < env_cfg.commands.yaw_deadzone
            or not 0 <= c['cross_tolerance_m'] < c['cross_scale_m']):
        raise ValueError("Invalid straight-motion physical scales or numerical-zero thresholds")
    families = {'stand', 'straight', 'arc', 'yaw', 'precision', 'lateral', 'mixed'}
    if not set(env_cfg.commands.holds['eligible_families']) <= families:
        raise ValueError("Unknown long-hold command family")
    if any(getattr(args, flag, False) for flag in ("eval_rolling_phase_zero", "eval_phase_transition", "eval_noise_pushes")):
        raise ValueError("Evaluation diagnostics are inference only")
    if any(getattr(args, flag, None) is not None for flag in ("transfer_armature", "transfer_delay")):
        raise ValueError("Evaluation physics overrides are inference only")
