"""Opt-in, read-only measurements and explicit reference provenance for Go2-W."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value.resolve())
    return value


def write_json(path, value):
    Path(path).write_text(
        json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n"
    )


def config_differences(saved, current, prefix=""):
    saved, current = json_safe(saved), json_safe(current)
    if isinstance(saved, dict) and isinstance(current, dict):
        result = {}
        for key in sorted(saved.keys() | current.keys()):
            result.update(
                config_differences(
                    saved.get(key), current.get(key), f"{prefix}.{key}".strip(".")
                )
            )
        return result
    return {} if saved == current else {prefix: {"saved": saved, "current": current}}


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
        "env_cfg.go2w_profile",
        "train_cfg.go2w_profile",
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


def check_training_continuation(
    config_path, env_cfg, train_cfg, sigma_x=None, entropy_coef=None
):
    """Go2-W continuation: every unexplained config difference is an error."""
    reference = check_reference_contract(config_path, env_cfg, train_cfg)
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
    if sigma_x is not None and env_cfg["rewards"]["tracking_sigma_x"] == sigma_x:
        allowed.add("env_cfg.rewards.tracking_sigma_x")
    if (
        entropy_coef is not None
        and train_cfg["algorithm"]["entropy_coef"] == entropy_coef
    ):
        allowed.add("train_cfg.algorithm.entropy_coef")
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


def source_identity(root):
    def git(*args):
        return subprocess.check_output(
            ["git", *args], cwd=root, stdin=subprocess.DEVNULL, stderr=subprocess.PIPE
        )

    diff = git("diff", "HEAD", "--binary")
    untracked = {}
    for name in (
        git("ls-files", "--others", "--exclude-standard", "-z").decode().split("\0")
    ):
        if name:
            untracked[name] = sha256(Path(root) / name)
    return {
        "head": git("rev-parse", "HEAD").decode().strip(),
        "branch": git("branch", "--show-current").decode().strip(),
        "status": git("status", "--short").decode(),
        "dirty_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "untracked_sha256": untracked,
    }


def manifest(root, checkpoint, urdf, env_cfg, train_cfg, overrides, reference):
    versions = {"python": platform.python_version()}
    for name in ("torch", "genesis-world", "rsl-rl-lib"):
        versions[name] = importlib.metadata.version(name)
    return {
        "schema_version": 1,
        "checkpoint": {
            "path": str(Path(checkpoint).resolve()),
            "sha256": sha256(checkpoint),
        },
        "source": source_identity(root),
        "urdf": {"path": str(Path(urdf).resolve()), "sha256": sha256(urdf)},
        "versions": versions,
        "resolved_env_config": env_cfg,
        "resolved_train_config": train_cfg,
        "eval_overrides": overrides,
        "saved_reference_config": reference,
        "conventions": {
            "quaternion": "wxyz; link to world",
            "base_rpy_legacy": "Genesis intrinsic XYZ, radians; NOT ROS extrinsic xyz RPY",
            "control_torques": "Genesis get_dofs_control_force: recomputed current-state control force, force-range clamped; not stored integration impulse",
            "substeps": "max absolute control-force API reading after each of four 0.005 s scene steps",
            "heading": "atan2 of world projection of body +X",
            "clearance": "URDF finite cylinder envelope to z=0; not rounded hardware tire or exact solver manifold",
        },
    }


def summed_normal_force(contacts, wheel_link_indices):
    """Sum |world Fz| on the wheel side of ground-only contact pairs, before thresholding."""
    links = torch.cat((contacts["link_a"], contacts["link_b"]), dim=1)
    forces = torch.cat((contacts["force_a"], contacts["force_b"]), dim=1)[..., 2].abs()
    valid = torch.cat((contacts["valid_mask"], contacts["valid_mask"]), dim=1)
    match = links[..., None] == wheel_link_indices.view(1, 1, -1)
    return (forces[..., None] * (match & valid[..., None])).sum(dim=1)


def rotate_wxyz(q, v):
    """Active rotation; broadcasting over arbitrary leading dimensions."""
    qv, v = torch.broadcast_tensors(q[..., 1:], v)
    uv = torch.linalg.cross(qv, v)
    return v + 2 * (q[..., :1] * uv + torch.linalg.cross(qv, uv))


def heading_wxyz(q):
    forward = rotate_wxyz(q, q.new_tensor([1.0, 0.0, 0.0]))
    return torch.atan2(forward[..., 1], forward[..., 0])


def cylinder_clearance(link_pos, link_quat, offset, local_axis, radius, half_width):
    center = link_pos + rotate_wxyz(link_quat, offset)
    axis = rotate_wxyz(link_quat, local_axis)
    uz = axis[..., 2].clamp(-1, 1)
    return (
        center[..., 2]
        - radius * (1 - uz.square()).clamp_min(0).sqrt()
        - half_width * uz.abs()
    )


def link_reposition_velocity(link_pos, link_vel, base_pos, base_quat, base_vel_body, base_ang_body):
    """Body derivative of link-origin position; all velocities refer to matching origins."""
    inverse = base_quat.clone()
    inverse[:, 1:] *= -1
    r_body = rotate_wxyz(inverse[:, None], link_pos - base_pos[:, None])
    return (
        rotate_wxyz(inverse[:, None], link_vel) - base_vel_body[:, None]
        - torch.linalg.cross(base_ang_body[:, None].expand_as(r_body), r_body)
    )


def wheel_cylinders(urdf, names, device="cpu"):
    root = ET.parse(urdf).getroot()
    offsets, axes, radii, widths = [], [], [], []
    for name in names:
        collisions = root.findall(f"link[@name='{name}']/collision")
        if len(collisions) != 1 or collisions[0].find("geometry/cylinder") is None:
            raise ValueError(f"Expected one cylinder collision for {name}")
        collision = collisions[0]
        origin, cylinder = collision.find("origin"), collision.find("geometry/cylinder")
        offsets.append([float(x) for x in origin.get("xyz", "0 0 0").split()])
        # URDF rpy is EXTRINSIC xyz, unlike Genesis' intrinsic XYZ helpers.
        axes.append(
            Rotation.from_euler(
                "xyz", [float(x) for x in origin.get("rpy", "0 0 0").split()]
            ).apply([0, 0, 1])
        )
        radii.append(float(cylinder.get("radius")))
        widths.append(float(cylinder.get("length")) / 2)
    return tuple(
        torch.tensor(np.asarray(x), dtype=torch.float32, device=device)
        for x in (offsets, axes, radii, widths)
    )


def urdf_link_poses(urdf, joint_angles):
    """CPU FK in the authored root frame, shared by kinematic diagnostic checks."""
    root = ET.parse(urdf).getroot()
    joints = list(root.findall("joint"))
    children = {j.find("child").get("link") for j in joints}
    roots = {link.get("name") for link in root.findall("link")} - children
    if len(roots) != 1:
        raise ValueError("Expected a single URDF tree root")
    poses = {roots.pop(): np.eye(4)}
    while joints:
        pending = []
        for joint in joints:
            parent, child = (
                joint.find("parent").get("link"),
                joint.find("child").get("link"),
            )
            if parent not in poses:
                pending.append(joint)
                continue
            origin = joint.find("origin")
            matrix = np.eye(4)
            if origin is not None:
                matrix[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                matrix[:3, :3] = Rotation.from_euler(
                    "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
                ).as_matrix()
            angle = joint_angles.get(joint.get("name"), 0.0)
            if joint.get("type") in ("revolute", "continuous"):
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                matrix[:3, :3] = (
                    matrix[:3, :3] @ Rotation.from_rotvec(angle * axis).as_matrix()
                )
            poses[child] = poses[parent] @ matrix
        if len(pending) == len(joints):
            raise ValueError("Disconnected URDF tree")
        joints = pending
    return poses


def nominal_support_heights(urdf, joint_angles, wheel_names):
    """URDF FK with the root at z=0. Required base heights per wheel envelope."""
    poses = urdf_link_poses(urdf, joint_angles)
    positions = torch.tensor(
        np.stack([poses[n][:3, 3] for n in wheel_names]),
        dtype=torch.float32,
        device="cpu",
    )
    quats = torch.tensor(
        Rotation.from_matrix(
            np.stack([poses[n][:3, :3] for n in wheel_names])
        ).as_quat()[:, [3, 0, 1, 2]],
        dtype=torch.float32,
        device="cpu",
    )
    return (
        -cylinder_clearance(positions, quats, *wheel_cylinders(urdf, wheel_names))
    ).tolist()


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
    return {
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
