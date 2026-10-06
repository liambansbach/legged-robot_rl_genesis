"""Go2-W reference checks and optional wheel/control measurements.

Ordinary replay restores complete saved configurations before these measurements.
"""

import json
from pathlib import Path
import numpy as np
import torch
import yaml

from robot_gym.utils.diagnostics import (
    config_differences, sha256, wheel_cylinders, summed_normal_force,
    cylinder_clearance, heading_wxyz, rotate_wxyz,
)


def check_reference_contract(config_path, env_cfg, train_cfg):
    """Check the training contract BEFORE intentional evaluation overrides.

    Reward/command-exposure differences are reported, not treated as changed physics.
    No inferred legacy defaults: a missing config requires an explicit audited file.
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        raise ValueError(
            f"Reference config missing: {config_path}; supply --reference_config with an audited config"
        )
    saved = yaml.safe_load(config_path.read_text())
    differences = config_differences(
        saved, {"env_cfg": env_cfg, "train_cfg": train_cfg}
    )
    important = (
        "env_cfg.control.",
        "env_cfg.normalization.",
        "env_cfg.sim.",
        "env_cfg.asset.",
        "env_cfg.terrain.",
        "env_cfg.init_state.default_joint_angles.",
        "env_cfg.init_state.pos",
        "env_cfg.init_state.rot",
        "env_cfg.env.num_observations",
        "env_cfg.env.num_privileged_obs",
        "env_cfg.env.num_actions",
        "train_cfg.actor.",
        "train_cfg.critic.",
        "train_cfg.runner.obs_groups.",
    )
    # These are execution choices, not actuator/observation/physics constants.
    execution = {
        "env_cfg.sim.batch_dofs_info",
        "env_cfg.sim.batch_links_info",
        "env_cfg.sim.performance_mode",
        "env_cfg.sim.deterministic",
    }
    mismatches = {
        k: v
        for k, v in differences.items()
        if k.startswith(important) and k not in execution
    }
    if mismatches:
        raise ValueError(
            "Reference actuator/observation/physics contract mismatch:\n"
            + json.dumps(mismatches, indent=2)
        )
    return {
        "path": str(config_path.resolve()),
        "sha256": sha256(config_path),
        "differences": differences,
    }


def check_continuation_output(checkpoint, output):
    parent, output = Path(checkpoint).resolve().parent, Path(output).resolve()
    if output == parent or parent in output.parents:
        raise ValueError(
            f"Continuation output must be separate from its parent run: {output}"
        )
    if output.exists():
        raise ValueError(f"Continuation output must be fresh: {output}")

class PhysicsDiagnostics:
    """No sampling, state writes, rewards or policy calls. Attached explicitly after build."""

    def __init__(self, env):
        self.env = env
        self.geometry = wheel_cylinders(
            env.urdf_reader.robot_file_path_absolute,
            env.cfg.asset.foot_link_names,
            env.device,
        )
        self.substep_forces = []
        self.normal_force = None

    def begin_step(self, raw_actions):
        self.raw_actions = raw_actions.detach().clone()
        self.substep_forces.clear()

    def after_substep(self):
        self.substep_forces.append(
            self.env.robot.get_dofs_control_force(self.env.joint_dof_idx).clone()
        )

    def contacts(self, contacts):
        self.normal_force = summed_normal_force(contacts, self.env.foot_link_indices)

    def capture(self):
        e = self.env
        q = e.robot.get_links_quat(e.foot_link_indices_local)
        scaled = e.applied_actions * e.action_scale
        leg_targets = (
            e.default_dof_pos[:, e.leg_action_indices] + scaled[:, e.leg_action_indices]
        )
        wheel_limit = e.dof_vel_limits[e.wheel_action_indices].clamp(
            max=e.cfg.control.wheel_velocity_target_limit
        )
        result = {
            "raw_actions": self.raw_actions.clone(),
            "applied_actions": e.applied_actions.clone(),
            "leg_position_targets": leg_targets,
            "wheel_velocity_targets": scaled[:, e.wheel_action_indices].clamp(
                -wheel_limit, wheel_limit
            ),
            "control_torques": e.torques.clone(),
            "control_torques_substep_max_abs": torch.stack(self.substep_forces)
            .abs()
            .amax(dim=0),
            "summed_normal_ground_force": self.normal_force.clone(),
            "legacy_force_support": e.foot_contacts.clone(),
            "wheel_clearance": cylinder_clearance(e.foot_pos, q, *self.geometry),
            "base_quat_wxyz": e.base_quat.clone(),
            "wheel_link_quat_wxyz": q.clone(),
            "base_heading": heading_wxyz(e.base_quat),
            "base_position": e.base_pos.clone(),
            "dof_position": e.dof_pos.clone(),
            "dof_velocity": e.dof_vel.clone(),
            "base_linear_velocity_body": e.base_lin_vel.clone(),
            "base_angular_velocity_body": e.base_ang_vel.clone(),
        }
        return result

def loaded_properties(env):
    """Read actual solver properties, not configured or sampled intent."""
    robot = env.robot
    geoms = [g for link in env.ankle_links for g in link.geoms]
    ratios = env.sim.rigid_solver.get_geoms_friction_ratio([g.idx for g in geoms])
    wheel = torch.stack([g.get_friction() for g in geoms])
    ground = env.ground_floor_entity.geoms[0].get_friction()
    result = {
        "mass": robot.get_links_mass().clone(),
        "com_local": robot.get_links_COM().clone(),
        "inertia_local": robot.get_links_inertia().clone(),
        "kp": robot.get_dofs_kp(env.joint_dof_idx).clone(),
        "kv": robot.get_dofs_kv(env.joint_dof_idx).clone(),
        "wheel_friction_ratio": ratios.clone(),
        "wheel_geometry_friction": wheel.clone(),
        "ground_geometry_friction": ground.clone(),
        "effective_friction": torch.maximum(ratios * wheel, ground).clamp_min(0.01),
        "friction_rule": "Genesis collider/contact.py: max(wheel geometry friction * ratio, ground geometry friction * ratio, 0.01); ground ratio=1",
    }
    if env.cfg.control.armature is not None:
        for field in ("armature", "stiffness", "damping", "frictionloss"):
            result[field] = getattr(robot, "get_dofs_" + field)(env.joint_dof_idx).clone()
    return result




class ReferenceCapture(PhysicsDiagnostics):
    """Read-only bounded 200 Hz capture through the existing substep hook."""

    def __init__(self, env):
        super().__init__(env)
        self.rows = []

    def after_substep(self):
        super().after_substep()
        e, r = self.env, self.env.robot
        quat = r.get_quat()
        inverse = quat.clone()
        inverse[..., 1:] *= -1
        pos = r.get_links_pos(e.foot_link_indices_local)
        wheel_quat = r.get_links_quat(e.foot_link_indices_local)
        contacts = r.get_contacts(exclude_self_contact=True, with_entity=e.ground_floor_entity, is_padded=True)
        links = torch.cat((contacts["link_a"], contacts["link_b"]), dim=1)
        valid = torch.cat((contacts["valid_mask"], contacts["valid_mask"]), dim=1)
        force = torch.cat((contacts["force_a"], contacts["force_b"]), dim=1)[..., 2].abs()
        robot_side = (links >= r.link_start) & (links < r.link_end)
        wheel_side = (links[..., None] == e.foot_link_indices).any(dim=-1)
        targets = (e.applied_actions * e.action_scale).clone()
        targets[:, e.leg_action_indices] += e.default_dof_pos[:, e.leg_action_indices]
        values = {
            "q": r.get_dofs_position(e.joint_dof_idx), "dq": r.get_dofs_velocity(e.joint_dof_idx),
            "base_pos": r.get_pos(), "base_quat": quat,
            "linear_body": rotate_wxyz(inverse, r.get_vel()),
            "angular_body": rotate_wxyz(inverse, r.get_ang()),
            "targets": targets, "control_force": self.substep_forces[-1],
            "wheel_normal_force": summed_normal_force(contacts, e.foot_link_indices),
            "clearance": cylinder_clearance(pos, wheel_quat, *self.geometry),
            "nonwheel_normal_force": (force * (valid & robot_side & ~wheel_side)).sum(dim=1, keepdim=True),
        }
        self.rows.append({k: v[0].detach().clone() for k, v in values.items()})


def spectrum(values, frequency):
    if len(values) < 4:
        return None
    values = np.asarray(values, dtype=float).reshape(len(values), -1)
    values = values - values.mean(axis=0)
    power = (np.abs(np.fft.rfft(values * np.hanning(len(values))[:, None], axis=0)) ** 2).sum(axis=1)
    power[0] = 0
    frequencies = np.fft.rfftfreq(len(values), 1 / frequency)
    peak = int(np.argmax(power))
    near = (frequencies >= 24) & (frequencies <= min(26, frequency / 2))
    return {"peak_Hz": float(frequencies[peak]), "sample_Hz": frequency,
            "resolution_Hz": frequency / len(values),
            "near_25Hz_power_fraction": float(power[near].sum() / power.sum()) if power.sum() else 0}
