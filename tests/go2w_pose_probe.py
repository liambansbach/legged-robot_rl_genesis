"""Bounded fixed-target calibration, without a learned controller.

Run manually from the repository root in genesis-gpu:
    python -m tests.go2w_pose_probe --output evaluation/new_pose_check

This retains the failed preparation's two identical baseline environments. It
stops at the first terminal state and never calls a failed stand an equilibrium.
It neither trains a policy nor tests coordinated unloading.
"""

import argparse
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from scipy.spatial.transform import Rotation

import robot_gym.envs  # noqa: F401 -- registers existing tasks
from robot_gym.envs.go2w.diagnostics import PhysicsDiagnostics
from robot_gym.envs.go2w.go2w_config import apply_go2w_profile
from robot_gym.utils import get_args, task_registry
from robot_gym.utils.diagnostics import (
    joint_dynamics, urdf_link_poses, wheel_cylinders, write_json,
)
from robot_gym.utils.helpers import class_to_dict
from robot_gym.utils.replay import configure_nominal
from robot_gym.utils.urdf_reader import URDFReader


def pose_geometry(cfg):
    path = URDFReader(cfg.asset.robot_file).robot_file_path_absolute
    names = cfg.asset.foot_link_names
    offsets, axes, radii, halfwidths = [
        value.cpu().numpy() for value in wheel_cylinders(path, names)
    ]
    poses = urdf_link_poses(path, cfg.init_state.default_joint_angles)
    centers = np.array([
        poses[name][:3, 3] + poses[name][:3, :3] @ offsets[i]
        for i, name in enumerate(names)
    ])
    world_axes = np.array([
        poses[name][:3, :3] @ axes[i] for i, name in enumerate(names)
    ])
    gaps = (centers[:, 2] - radii * np.sqrt(np.maximum(0, 1 - world_axes[:, 2] ** 2))
            - halfwidths * abs(world_axes[:, 2]))
    thighs = np.array([poses[name.replace("_foot", "_thigh")][:3, 3] for name in names])
    height = float(-min(gaps))
    return {
        "base_height_m": height,
        "gaps_at_geometric_height_m": (gaps + height).tolist(),
        "wheel_centers_base_m": centers.tolist(),
        "thigh_origins_base_m": thighs.tolist(),
        "wheel_minus_thigh_x_m": (centers - thighs)[:, 0].tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--spawn_clearance", type=float, default=.003, help="Floor gap in m")
    options = parser.parse_args()
    if not np.isfinite(options.spawn_clearance) or not 0 <= options.spawn_clearance <= .03:
        parser.error("spawn_clearance must be finite and within [0, .03] m")
    options.output.mkdir(parents=True, exist_ok=False)
    with patch.object(sys, "argv", [
        "pose_probe", "--task", "go2w", "--go2w_profile", "transfer_v2",
        "--num_envs", "2", "--seed", "1", "--headless", "--rl_device", "cuda:0",
    ]):
        args = get_args()
    cfg, train = task_registry.get_cfgs("go2w")
    apply_go2w_profile(cfg, train, "transfer_v2")
    configure_nominal(cfg, args)
    cfg.init_state.default_joint_angles = {
        f"{side}_{joint}_joint": angle
        for side in ("FL", "FR", "RL", "RR")
        for joint, angle in (("hip", 0.), ("thigh", .70), ("calf", -1.40), ("foot", 0.))
    }
    cfg.init_state.rot = [1., 0., 0., 0.]
    geometry = pose_geometry(cfg)
    cfg.init_state.pos = [0., 0., geometry["base_height_m"] + options.spawn_clearance]
    cfg.env.play_mode = True
    cfg.env.episode_length_s = 30
    cfg.env.capture_transitions = cfg.env.capture_precision = cfg.env.capture_closed_loop = True
    report = {
        "geometry": geometry, "spawn_clearance_m": options.spawn_clearance,
        "spawn_height_m": cfg.init_state.pos[2], "configuration": class_to_dict(cfg),
        "policy": "none; nominal leg position / zero wheel velocity targets",
        "status": "not built", "stable_loaded_height_m": None,
        "coordinated_lift_result": None,
    }
    records = []
    import genesis as gs
    try:
        env, _ = task_registry.make_env("go2w", args=args, env_cfg=cfg)
        env.set_fixed_command((0, 0, 0))
        env.physics_diagnostics = PhysicsDiagnostics(env)
        report["dynamics_after_reset"] = joint_dynamics(env.robot, env.joint_names)
        report["force_limits_Nm"] = env.torque_limits.cpu().tolist()
        actions = torch.zeros((2, 16), device=env.device)
        for _ in range(150):
            env.step(actions)
            state = env.transition_state  # Captured before the environment's automatic reset.
            row = {key: state[key].clone() for key in (
                "base_pos", "base_quat", "base_lin_vel", "base_ang_vel", "dof_pos",
                "dof_vel", "wheel_clearance", "wheel_normal_force", "actions",
                "applied_actions", "reset_buf", "time_out_buf", "nonfoot_contact_count",
                "control_torques_substep_max_abs",
            )}
            contacts = env.robot.get_contacts(with_entity=env.robot, is_padded=True)
            # This getter follows auto-reset; terminal self-contact is unavailable.
            row["self_contacts"] = torch.where(
                state["reset_buf"], -1, contacts["valid_mask"].sum(1)
            )
            records.append(row)
            if state["reset_buf"].any() or not torch.isfinite(state["dof_pos"]).all():
                raise RuntimeError("Fixed-target stand failed; no equilibrium or unloading result")
        report["status"] = "completed three seconds; inspect motion before declaring equilibrium"
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        if records:
            data = {key: torch.stack([r[key] for r in records]).cpu().numpy() for key in records[0]}
            np.savez_compressed(options.output / "pose_probe.npz", **data)
            angles = Rotation.from_quat(
                data["base_quat"].reshape(-1, 4)[:, [1, 2, 3, 0]]
            ).as_euler("xyz").reshape(-1, 2, 3)
            report.update(
                recorded_duration_s=len(records) * .02,
                terminal_height_m=data["base_pos"][-1, :, 2].tolist(),
                terminal_roll_pitch_deg=np.rad2deg(angles[-1, :, :2]).tolist(),
                max_control_force_fraction=float((data["control_torques_substep_max_abs"]
                                                  / np.array(report["force_limits_Nm"])).max()),
                nonwheel_contact_max=int(data["nonfoot_contact_count"].max()),
                self_contact_max=int(data["self_contacts"].max()),
                terminal_self_contact_available=not bool(data["reset_buf"][-1].any()),
            )
        write_json(options.output / "pose_probe.json", report)
        gs.destroy()


if __name__ == "__main__":
    main()
