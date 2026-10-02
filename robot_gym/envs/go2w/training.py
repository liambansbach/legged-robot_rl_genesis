"""Go2-W training initialization rules and runtime metadata."""

def prepare_go2w_continuation(args, env_cfg, train_cfg):
    """Check the explicit saved parent before constructing the simulator."""
    from pathlib import Path
    from robot_gym import ROBOT_GYM_ROOT_DIR
    from robot_gym.utils.helpers import get_load_path, class_to_dict
    from robot_gym.utils.urdf_reader import URDFReader
    from robot_gym.utils.diagnostics import check_training_continuation, sha256

    if args.load_run in (None, "-1") or args.checkpoint is None or args.checkpoint < 0:
        raise ValueError(
            "Go2-W continuation requires explicit --load_run and --checkpoint"
        )
    if not args.run_name or any(c in args.run_name for c in "/\\"):
        raise ValueError(
            "Go2-W continuation requires a new --run_name (a folder name, not a path)"
        )
    root = Path(ROBOT_GYM_ROOT_DIR) / "logs" / train_cfg.runner.experiment_name
    checkpoint = Path(get_load_path(root, args.load_run, args.checkpoint)).resolve()
    config = checkpoint.with_name("config.yaml")
    if args.reference_config and Path(args.reference_config).resolve() != config:
        raise ValueError(
            "Training continuation requires the saved config beside its checkpoint"
        )
    env_cfg.asset.joint_names = URDFReader(env_cfg.asset.robot_file).joint_names
    reference = check_training_continuation(
        config,
        class_to_dict(env_cfg),
        class_to_dict(train_cfg),
        args.tracking_sigma_x,
        args.entropy_coef,
        getattr(args, "go2w_finetune", None),
        getattr(args, "sagittal_stance_weight", None),
        getattr(args, "event_quality_profile", None),
    )
    if (getattr(args, "sagittal_stance_weight", None) is not None
            or getattr(args, "event_quality_profile", None) is not None):
        import torch
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        progress = (state.get("infos") or {}).get("event_step_v1", {})
        count = progress.get("completed_updates")
        if type(count) is not int or count < 0:
            raise ValueError("Missing event_step_v1 completed-update state")
        # Restore runtime progress before even the new scene's first command draw.
        env_cfg._event_completed_updates = count
    saved_git = checkpoint.parent / "git" / f"{Path(ROBOT_GYM_ROOT_DIR).name}.diff"
    source_snapshot = None
    if saved_git.is_file():
        # Existing RSL-RL git snapshot includes the training HEAD and dirty patch.
        source_snapshot = {
            "path": str(saved_git),
            "sha256": sha256(saved_git),
            "head": saved_git.read_text().splitlines()[1],
        }
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "saved_config": reference,
        "saved_source_snapshot": source_snapshot,
    }


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


def training_metadata(env):
    from .step_events import lateral_high
    from .deployment import transfer_contract
    result = {"profile": getattr(env.cfg, "go2w_profile", None)}
    if env.event_step:
        result.update(completed_updates=env.completed_updates,
                      lateral_high=lateral_high(env.completed_updates, env.cfg.commands))
    if env.cfg.control.armature is not None:
        result["deployment_contract"] = transfer_contract(env)
        result["runtime_armature_min_max_kg_m2"] = [float(env.armature_samples.min()), float(env.armature_samples.max())]
    return result
