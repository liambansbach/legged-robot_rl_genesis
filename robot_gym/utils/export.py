"""Portable deterministic actor: trained normalization plus the actuator action bound."""

import json
from pathlib import Path
import torch


def select_transfer_dynamics(cfg, args):
    """Logged inference overrides, applied only after saved-recipe validation."""
    selection = getattr(args, "transfer_armature", None)
    delay = getattr(args, "transfer_delay", None)
    if getattr(cfg, "go2w_profile", None) != "transfer_v1":
        if selection is not None or delay is not None:
            raise ValueError("Transfer dynamics selectors require transfer_v1")
        return
    cfg.domain_rand.randomize_armature = False
    value = {"nominal": .01, "low": .005, "high": .02}[selection or "nominal"]
    cfg.control.armature = dict.fromkeys(cfg.control.armature, value)
    if delay is not None:
        cfg.domain_rand.randomize_action_delay = True
        cfg.domain_rand.action_delay_steps_range = [delay, delay]


def transfer_contract(env):
    from robot_gym.utils.diagnostics import sha256
    return {
        "profile": "transfer_v1",
        "asset": {"file": env.urdf_reader.robot_file_name,
                  "sha256": sha256(env.urdf_reader.robot_file_path_absolute),
                  "source_commit": env.cfg.asset.source_commit,
                  "wheel_radius_m": env.cfg.asset.contact_height},
        "nominal_armature_kg_m2": dict.fromkeys(env.joint_names, .01),
        "nominal_passive_terms": {name: {"stiffness_Nm_per_rad": 0., "damping_Nm_s_per_rad": 0.,
                                          "frictionloss_Nm": 0.} for name in env.joint_names},
        "training_randomization": {"armature_kg_m2": [.005, .02],
                                   "armature_sampling": "independent per environment leg/wheel groups; symmetric; constant across resets",
                                   "action_delay_policy_steps": [0, 2], "action_delay_ms": [0, 40]},
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


class BoundedPolicy(torch.nn.Module):
    def __init__(self, actor, observation_limit: float, action_limit: float):
        super().__init__()
        self.actor = actor
        self.observation_limit = observation_limit
        self.action_limit = action_limit

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor(
            observations.clamp(-self.observation_limit, self.observation_limit)
        ).clamp(-self.action_limit, self.action_limit)


def export_policy(runner, env, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    actor = runner.alg.get_policy().as_jit().cpu().eval()
    policy = BoundedPolicy(
        actor,
        float(env.cfg.normalization.clip_observations),
        float(env.cfg.normalization.clip_actions),
    ).eval()
    torch.jit.script(policy).save(str(path / "policy_1.pt"))
    contract = {
        "observation_dim": env.num_obs,
        "action_dim": env.num_actions,
        "joint_order": env.joint_names,
        "control_type": env.cfg.control.control_type,
        "action_scale": env.action_scale[0].tolist(),
        "default_joint_angles": env.default_dof_pos[0].tolist(),
        "nominal_kp": env.base_p_gains.tolist(),
        "nominal_kd": env.base_d_gains.tolist(),
        "effort_limits": env.torque_limits.tolist(),
        "joint_velocity_limits": env.dof_vel_limits.tolist(),
        "checkpoint": str(getattr(runner, "checkpoint_path", None)),
        "action_bound": env.cfg.normalization.clip_actions,
        "wheel_velocity_target_limit": getattr(
            env.cfg.control, "wheel_velocity_target_limit", None
        ),
        "policy_dt": env.dt,
        "physics_dt": env.cfg.sim.dt,
        "observation_normalization": "embedded",
        "quaternion_convention": "wxyz",
        "observation_order": [
            "body_linear_velocity",
            "body_angular_velocity",
            "projected_gravity",
            "body_velocity_command",
            "leg_position_error"
            if env.cfg.asset.name == "go2w"
            else "joint_position_error",
            "joint_velocity",
            "previous_clipped_action",
        ],
        "observation_scales": {
            "lin_vel": env.obs_scales.lin_vel,
            "ang_vel": env.obs_scales.ang_vel,
            "dof_pos": env.obs_scales.dof_pos,
            "dof_vel": env.obs_scales.dof_vel,
        },
    }
    if getattr(env.cfg, "go2w_profile", None) == "transfer_v1":
        contract.update(transfer_contract(env))
        from robot_gym.utils.diagnostics import joint_dynamics, write_json, sha256
        contract["actor_sha256"] = sha256(path / "policy_1.pt")
        contract["checkpoint_sha256"] = sha256(runner.checkpoint_path)
        write_json(path / "runtime_dynamics.json", joint_dynamics(env.robot, env.joint_names))
    (path / "contract.json").write_text(json.dumps(contract, indent=2))
    return policy
