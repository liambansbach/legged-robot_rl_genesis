"""Nominal inference settings shared by replay and task evaluation."""

from robot_gym.utils.task_registry import task_registry


def configure_nominal(cfg, args):
    cfg.noise.add_noise = False
    cfg.terrain.curriculum = cfg.commands.curriculum = False
    cfg.sim.performance_mode = False
    cfg.sim.deterministic = True
    for name in ("randomize_friction", "randomize_base_mass", "randomize_com",
                 "randomize_kp", "randomize_kd", "randomize_action_delay",
                 "randomize_armature", "push_robots"):
        setattr(cfg.domain_rand, name, False)
    for name in ("joint_position_noise", "joint_velocity_noise", "linear_velocity_noise", "angular_velocity_noise"):
        setattr(cfg.init_state, name, 0.0)
    cfg.init_state.orientation_noise = (0.0, 0.0, 0.0)
    task_registry.get_task_class(args.task).configure_evaluation(cfg, args)


def configure_fixed_command(env, args):
    """Any supplied axis fixes all commands; omitted axes become exact zero."""
    values = [getattr(args, f"command_{axis}", None) for axis in ("vx", "vy", "yaw")]
    if any(value is not None for value in values):
        command = tuple(0.0 if value is None else value for value in values)
        env.set_fixed_command(command)
        print(f"Fixed body command [vx m/s, vy m/s, yaw rad/s]: {command}", flush=True)
