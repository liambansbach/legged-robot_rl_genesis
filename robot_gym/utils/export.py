"""Portable deterministic actor: trained normalization plus the actuator action bound."""

import json
from pathlib import Path
import torch


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
    (path / "contract.json").write_text(json.dumps(contract, indent=2))
    return policy
