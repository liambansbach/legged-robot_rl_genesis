"""Go2-W reference checks and optional wheel/control measurements.

Historical checks remain explicitly callable; ordinary replay restores saved recipes.
"""

import json
from pathlib import Path
import numpy as np
import torch
import yaml

from robot_gym.utils.diagnostics import (
    config_differences, sha256, wheel_cylinders, summed_normal_force,
    cylinder_clearance, heading_wxyz,
)


def check_reference_contract(config_path, env_cfg, train_cfg, finetune_continuation=False):
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
        "env_cfg.go2w_profile",
        "train_cfg.go2w_profile",
        "env_cfg.go2w_behavior",
        "train_cfg.go2w_behavior",
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
    # Historical audits compare the explicitly selected saved finetune design.
    # Training checks its exact allowed transition separately below.
    if not finetune_continuation:
        for key in ("env_cfg.go2w_finetune", "train_cfg.go2w_finetune",
                    "env_cfg.rewards.sagittal_stance_weight", "env_cfg.rewards.event_step.quality_profile"):
            if key in differences:
                mismatches[key] = differences[key]
        if any(cfg.get("rewards", {}).get("event_step", {}).get("quality_profile")
               for cfg in (env_cfg, saved["env_cfg"])):
            mismatches.update({k: v for k, v in differences.items()
                               if k.startswith("env_cfg.rewards.event_step.")
                               or k == "env_cfg.rewards.scales.step_event"})
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

def check_training_continuation(
    config_path, env_cfg, train_cfg, sigma_x=None, entropy_coef=None, finetune=None,
    sagittal_stance_weight=None, event_quality_profile=None,
):
    """Go2-W continuation: every unexplained config difference is an error."""
    reference = check_reference_contract(config_path, env_cfg, train_cfg, finetune_continuation=True)
    allowed = {
        "train_cfg.runner.run_name",
        "train_cfg.runner.resume",
        "train_cfg.runner.load_run",
        "train_cfg.runner.checkpoint",
        "train_cfg.runner.max_iterations",
        "train_cfg.runner.logger",
        "env_cfg.env.record_command_families",  # Read-only diagnostic labels.
        "env_cfg.sim.batch_dofs_info",  # Populated during Genesis build for DR.
        "env_cfg.sim.batch_links_info",
    }
    if finetune is not None:
        from robot_gym.envs.go2w.go2w_config import FINETUNE_COMMANDS, FINETUNE_MOBILITY, FINETUNE_PRECISION

        if finetune not in ("coverage", "coverage_mobility", "precision_clearance"):
            raise ValueError(f"Unknown Go2-W finetune: {finetune}")
        expected = {
            "env_cfg.go2w_finetune": finetune,
            "train_cfg.go2w_finetune": finetune,
            **{f"env_cfg.commands.{key}": value for key, value in FINETUNE_COMMANDS.items()},
        }
        if finetune == "coverage_mobility":
            expected.update({f"env_cfg.rewards.{key}": value for key, value in FINETUNE_MOBILITY.items()})
        if finetune == "precision_clearance":
            saved = yaml.safe_load(Path(config_path).read_text())
            if any(saved[key].get("go2w_finetune") != "coverage" for key in ("env_cfg", "train_cfg")):
                raise ValueError("precision_clearance requires the saved coverage A parent")
            parent_values = {
                "rewards.tracking_sigma_yaw": None,
                "rewards.clearance_sigma": .02,
                "rewards.clearance_activation_height": .01,
                "rewards.scales.foot_swing_clearance": .12,
                "domain_rand.kp_scale_range": [.9, 1.1],
                "domain_rand.kd_scale_range": [.9, 1.1],
                "domain_rand.action_delay_steps_range": [0, 1],
            }
            for path, value in parent_values.items():
                actual = saved["env_cfg"]
                for part in path.split("."):
                    actual = actual.get(part)
                if actual != value:
                    raise ValueError(f"Coverage parent value mismatch: {path}")
            expected.update({f"env_cfg.{key}": value for key, value in FINETUNE_PRECISION.items()})
            # Coverage is already present in A: never permit a sampler migration here.
            for key in FINETUNE_COMMANDS:
                if saved["env_cfg"]["commands"].get(key) != FINETUNE_COMMANDS[key]:
                    raise ValueError(f"Coverage parent sampler mismatch: {key}")
        resolved = {"env_cfg": env_cfg, "train_cfg": train_cfg}
        if any(cfg.get("go2w_profile") != "step_recovery_v1" for cfg in resolved.values()):
            raise ValueError("Finetune continuation requires the step_recovery_v1 profile")
        for key, value in expected.items():
            current = resolved
            for part in key.split("."):
                current = current.get(part, {}) if isinstance(current, dict) else None
            if current != value:
                raise ValueError(f"Finetune override differs from explicit {finetune}: {key}")
        allowed.update(expected)
    if sigma_x is not None and env_cfg["rewards"]["tracking_sigma_x"] == sigma_x:
        allowed.add("env_cfg.rewards.tracking_sigma_x")
    if (
        entropy_coef is not None
        and train_cfg["algorithm"]["entropy_coef"] == entropy_coef
    ):
        allowed.add("train_cfg.algorithm.entropy_coef")
    if finetune == "precision_clearance":
        allowed.discard("env_cfg.rewards.tracking_sigma_x")
        allowed.discard("train_cfg.algorithm.entropy_coef")
    if sagittal_stance_weight is not None:
        saved = yaml.safe_load(Path(config_path).read_text())
        if (not np.isfinite(sagittal_stance_weight) or sagittal_stance_weight <= 0
                or any(cfg.get("go2w_profile") != "event_step_v1"
                       for cfg in (env_cfg, train_cfg, saved["env_cfg"], saved["train_cfg"]))
                or env_cfg["rewards"].get("sagittal_stance_weight") != sagittal_stance_weight):
            raise ValueError("Invalid declared event_step_v1 sagittal stance continuation")
        allowed.add("env_cfg.rewards.sagittal_stance_weight")
    if event_quality_profile is not None:
        saved = yaml.safe_load(Path(config_path).read_text())
        if (event_quality_profile != "sufficient_clearance"
                or any(cfg.get("go2w_profile") != "event_step_v1"
                       for cfg in (env_cfg, train_cfg, saved["env_cfg"], saved["train_cfg"]))
                or env_cfg["rewards"]["event_step"].get("quality_profile") != event_quality_profile
                or env_cfg["rewards"]["scales"]["step_event"] != 0.05):
            raise ValueError("Invalid declared event quality continuation")
        allowed.update(("env_cfg.rewards.event_step.quality_profile", "env_cfg.rewards.scales.step_event"))
        # This continuation changes only event quality, even if a stance flag is supplied.
        allowed.discard("env_cfg.rewards.sagittal_stance_weight")
    # Only the agreed two-update smoke may reduce the source batch size.
    if env_cfg["env"]["num_envs"] == 64 and train_cfg["runner"]["max_iterations"] == 2:
        allowed.add("env_cfg.env.num_envs")
    unexpected = {k: v for k, v in reference["differences"].items() if k not in allowed}
    if unexpected:
        raise ValueError(
            "Unexplained continuation config differences:\n"
            + json.dumps(unexpected, indent=2)
        )
    return reference

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
        brake = getattr(e, "zero_command_brake", None)
        if brake is not None:
            result["issued_actions"] = e.actions.clone()
            result["zero_command_brake_alpha"] = brake.alpha.clone()
        if getattr(e, "step_recovery", False):
            result["loaded_wheels"] = e.loaded_wheels.clone()
            result["wheel_reposition_velocity_body"] = e.wheel_reposition_velocity_body.clone()
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
    if getattr(env.cfg, "go2w_profile", None) == "transfer_v1":
        for field in ("armature", "stiffness", "damping", "frictionloss"):
            result[field] = getattr(robot, "get_dofs_" + field)(env.joint_dof_idx).clone()
    return result
