"""Shared read-only physics geometry, measurements and provenance helpers."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import torch
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


def wheel_axles_body(base_quat, wheel_quat, joint_axes):
    """Joint axes in the base frame; invariant to wheel spin about those axes."""
    inverse = base_quat.clone()
    inverse[..., 1:] *= -1
    return rotate_wxyz(inverse[..., None, :], rotate_wxyz(wheel_quat, joint_axes))


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


def joint_dynamics(robot, names):
    """Effective post-build SI parameters through public installed Genesis getters.

    None means unavailable, never inferred zero. Armature is joint-reflected
    rotational inertia (kg m^2), distinct from link inertia and motor inertia.
    """
    indices = [robot.get_joint(n).dofs_idx_local[0] for n in names]
    values, sources = {}, {}
    for field in ("armature", "kp", "kv", "stiffness", "damping", "frictionloss"):
        getter = getattr(robot, f"get_dofs_{field}", None)
        sources[field] = f"RigidEntity.get_dofs_{field}" if getter else "unavailable"
        if getter:
            actual = getter(indices).detach().cpu()
            values[field] = (actual[0] if actual.ndim == 2 else actual).tolist()
        else:
            values[field] = [None] * len(names)
    for field, getter_name in (("force_range", "get_dofs_force_range"), ("position_limit", "get_dofs_limit")):
        lo, hi = getattr(robot, getter_name)(indices)
        if lo.ndim == 2:
            lo, hi = lo[0], hi[0]
        values[field] = list(zip(lo.detach().cpu().reshape(-1).tolist(), hi.detach().cpu().reshape(-1).tolist()))
        sources[field] = f"RigidEntity.{getter_name}"
    result = {}
    for i, name in enumerate(names):
        row = {k: v[i] for k, v in values.items()}
        row['position_limit'] = [x if np.isfinite(x) else None for x in row['position_limit']]
        row['solver_parameters'] = json_safe(robot.get_joint(name).get_sol_params())
        result[name] = row
    return {"joints": result, "sources": sources, "environment_index_if_batched": 0,
            "units": {"armature": "kg m^2 (joint-reflected)", "kp": "N m/rad", "kv": "N m s/rad (active)",
                      "stiffness": "N m/rad (passive)", "damping": "N m s/rad (passive)",
                      "frictionloss": "N m", "force_range": "N m", "position_limit": "rad"}}


def __getattr__(name):
    # Preserve historical diagnostic imports without a second implementation.
    if name in {"check_reference_contract",
                "check_continuation_output", "PhysicsDiagnostics", "loaded_properties"}:
        from robot_gym.envs.go2w import diagnostics
        return getattr(diagnostics, name)
    raise AttributeError(name)
