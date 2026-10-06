"""Go2-W nominal deployment dynamics and explicit inference sensitivity cases."""

def select_transfer_dynamics(cfg, args):
    """Logged inference overrides, applied only after saved-config validation."""
    selection = getattr(args, "transfer_armature", None)
    delay = getattr(args, "transfer_delay", None)
    cfg.domain_rand.randomize_armature = False
    cfg.phase_guidance["randomize_reset"] = False
    cfg.control.armature_override = None if selection in (None, "nominal") else cfg.domain_rand.armature_range[0 if selection == "low" else 1]
    if delay is not None:
        cfg.domain_rand.randomize_action_delay = True
        cfg.domain_rand.action_delay_steps_range = [delay, delay]


def transfer_contract(env):
    from robot_gym.utils.diagnostics import sha256
    return {
        "task": "go2w",
        "asset": {"file": env.urdf_reader.robot_file_name,
                  "sha256": sha256(env.urdf_reader.robot_file_path_absolute),
                  "source_commit": env.cfg.asset.source_commit,
                  "wheel_radius_m": env.cfg.asset.contact_height},
        "nominal_armature_kg_m2": dict(env.cfg.control.armature),
        "nominal_passive_terms": {name: {"stiffness_Nm_per_rad": env.cfg.control.passive_stiffness,
                                          "damping_Nm_s_per_rad": env.cfg.control.passive_damping,
                                          "frictionloss_Nm": env.cfg.control.passive_frictionloss} for name in env.joint_names},
        "training_randomization": {"armature_kg_m2": list(env.cfg.domain_rand.armature_range),
                                   "armature_sampling": "independent per environment leg/wheel groups; symmetric; constant across resets",
                                   "action_delay_policy_steps": list(env.cfg.domain_rand.action_delay_steps_range),
                                   "action_delay_ms": [1000 * env.dt * n for n in env.cfg.domain_rand.action_delay_steps_range]},
        "velocity": {"source": "simulator-derived", "frame": "body axes",
                     "reference_point": "authored base-link origin", "sample_timing": "current policy boundary",
                     "genesis_getter": "RigidEntity.get_vel(relative=True), world axes then inverse base quaternion",
                     "transport": "getter already transports to authored origin; no second COM transport"},
        "physics_substeps": env.cfg.sim.substeps, "decimation": env.cfg.control.decimation,
        "integrator": env.sim.rigid_solver._integrator.name,
        "solver": {"constraint_solver": str(env.sim.rigid_solver._options.constraint_solver),
                   "iterations": env.cfg.sim.iterations, "ls_iterations": env.cfg.sim.ls_iterations},
        "assumptions": "Development armature preset/range and zero passive terms, not identified hardware; mass uncertainty scales base inertia proportionally; COM shift is separate uncertainty",
    }
