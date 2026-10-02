"""Go2-W recipe and diagnostic CLI options; registered with the task."""

import math


def add_arguments(parser):
    parameters = [
        {"name": "--go2w_finetune", "choices": ["coverage", "coverage_mobility", "precision_clearance"], "default": None, "help": "Explicit step_recovery_v1 continuation/evaluation design; unset preserves sampling and rewards"},
        {"name": "--go2w_profile", "choices": ["step_recovery_v1", "event_step_v1", "transfer_v1"], "default": None, "help": "Explicit Go2-W action/reward/training profile; unset preserves the baseline"},
        {"name": "--sagittal_stance_weight", "type": float, "default": None, "help": "Explicit event_step_v1 stance weight; full-demand weight stays 0.12; select the saved value for evaluation/play"},
        {"name": "--event_quality_profile", "choices": ["sufficient_clearance"], "default": None, "help": "Opt-in event quality and payment; select the saved choice for evaluation/play"},
        {"name": "--zero_command_brake", "action": "store_true", "help": "Go2-W inference only: blend wheel targets to zero for a complete zero body command"},
        {"name": "--entropy_coef", "type": float, "default": None, "help": "Go2-W entropy weight; unset preserves the registered config"},
        {"name": "--tracking_sigma_x", "type": float, "default": None, "help": "Go2-W forward squared-error denominator; unset preserves the registered config"},
        {"name": "--skip_zero_action_probe", "action": "store_true", "help": "Bank evaluation: retain all policy cases, omit the equilibrium zero-action probe"},
        {"name": "--diagnostic_trace", "action": "store_true", "help": "Read substep control forces, summed ground loads and cylinder geometry"},
        {"name": "--reference_config", "help": "Explicit audited saved config, if not next to the checkpoint"},
        {"name": "--eval_mode", "choices": ["nominal", "bank", "equilibrium", "sustained", "closed_loop", "precision_screen", "precision_dr", "transfer_screen"], "default": "nominal"},
        {"name": "--transfer_armature", "choices": ["nominal", "low", "high"], "default": None, "help": "transfer_v1 inference-only explicit motor armature: .01/.005/.02 kg m^2"},
        {"name": "--transfer_delay", "type": int, "choices": [0, 1, 2], "default": None, "help": "transfer_v1 inference-only held action delay in policy steps"},
        {"name": "--transfer_cases", "nargs": "+", "choices": ["stand", "forward", "reverse", "yaw_positive", "yaw_negative", "lateral_positive", "lateral_negative", "mixed"], "help": "transfer_screen subset; unset runs the small complete command panel"},
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
        entropy = 0.003 if profile in ("event_step_v1", "transfer_v1") else 0.001
        if getattr(args, "tracking_sigma_x", None) not in (None, 0.25) or getattr(args, "entropy_coef", None) not in (None, entropy):
            raise ValueError(f"{profile} fixes tracking_sigma_x=0.25 and entropy_coef={entropy}")
        from robot_gym.envs.go2w.go2w_config import apply_go2w_profile

        apply_go2w_profile(env_cfg, cfg_train, profile)
    finetune = getattr(args, "go2w_finetune", None)
    if finetune is not None:
        if args.task != "go2w" or profile != "step_recovery_v1":
            raise ValueError("--go2w_finetune requires go2w with --go2w_profile step_recovery_v1")
        from robot_gym.envs.go2w.go2w_config import apply_go2w_finetune

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
        from robot_gym.envs.go2w.go2w_config import apply_go2w_event_quality
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
