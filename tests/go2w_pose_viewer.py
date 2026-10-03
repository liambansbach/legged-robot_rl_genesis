"""Frozen imported geometry, without advancing dynamics or constructing an RL env.

python -m tests.go2w_pose_viewer --duration 60
python -m tests.go2w_pose_viewer --headless --output evaluation/pose_reference.json

Duration is viewer wall-clock seconds. Height offset changes the displayed pose,
not h_ref. Joint setters write state once; absence of scene.step freezes it.
"""

import argparse
import time
from pathlib import Path

import genesis as gs
import torch
from scipy.spatial.transform import Rotation

from robot_gym.envs.go2w.go2w_config import (
    MEASURED_URDF, V3_SPAWN_CLEARANCE, v3_joint_reference,
)
from robot_gym.utils.diagnostics import cylinder_clearance, rotate_wxyz, wheel_cylinders, write_json
from robot_gym.utils.urdf_reader import URDFReader


def inspect_pose(scene, robot, path, height_offset):
    requested = v3_joint_reference()
    ids = [robot.get_joint(name).dofs_idx_local[0] for name in requested]
    target = torch.tensor(list(requested.values()), device=gs.device)
    # These are motor DOF indices, not floating-root qpos indices.
    robot.set_dofs_position(target, ids, zero_velocity=True)
    robot.set_quat([1., 0., 0., 0.], relative=True, zero_velocity=True)
    robot.set_pos([0., 0., 1.], relative=True, zero_velocity=True)
    names = [f"{s}_foot" for s in ("FL", "FR", "RL", "RR")]
    feet = [robot.get_link(n).idx_local for n in names]
    thighs = [robot.get_link(n.replace("_foot", "_thigh")).idx_local for n in names]
    geometry = wheel_cylinders(path, names, gs.device)

    def measure():
        pos, quat = robot.get_links_pos(feet, relative=True), robot.get_links_quat(feet, relative=True)
        centers = pos + rotate_wxyz(quat, geometry[0])
        return centers, cylinder_clearance(pos, quat, *geometry)

    _, initial_gaps = measure()
    h_ref = float(robot.get_pos(relative=True)[2] - initial_gaps.min())
    robot.set_pos([0., 0., h_ref], relative=True, zero_velocity=True)
    centers, gaps = measure()
    base = robot.get_pos(relative=True)
    quat = robot.get_quat(relative=True)
    inverse = quat.clone()
    inverse[1:] *= -1
    centers_body = rotate_wxyz(inverse, centers - base)
    dx = rotate_wxyz(inverse, centers - robot.get_links_pos(thighs, relative=True))[:, 0]
    actual = robot.get_dofs_position(ids)
    lower, upper = robot.get_dofs_limit(ids)
    if not torch.isfinite(actual).all() or (actual < lower).any() or (actual > upper).any():
        raise ValueError("Invalid imported motor joint state/limits")
    if (actual - target).abs().max() > 1e-5 or gaps.abs().max() > 1e-4:
        raise ValueError("Imported reference differs from requested joints or has inconsistent floor gaps")
    # Public vertex bounds catch obvious ground penetration, not exact self-contact.
    nonwheel_min = min(float(g.get_AABB()[0, 2]) for link in robot.links
                       if link.name not in names for g in link.geoms)
    report = {
        "asset": str(path), "dynamics_steps_after_build": 0,
        "joint_angles_rad": dict(zip(requested, actual.cpu().tolist())),
        "maximum_joint_error_rad": float((actual - target).abs().max()),
        "imposed_authored_base_quaternion_wxyz": quat,
        "imposed_readback_roll_pitch_deg": Rotation.from_quat(quat.cpu().numpy()[[1, 2, 3, 0]]).as_euler("xyz", degrees=True)[:2],
        "h_ref_m": h_ref, "suggested_spawn_height_m": h_ref + V3_SPAWN_CLEARANCE,
        "display_height_offset_m": height_offset, "display_height_m": h_ref + height_offset,
        "nonwheel_collision_vertex_min_z_m": nonwheel_min,
        "self_contact": "not evaluated; no dynamics/contact solve",
        "wheels": {n: {"gap_m": float(gaps[i]), "center_base_m": centers_body[i],
                       "wheel_minus_thigh_origin_x_m": float(dx[i])} for i, n in enumerate(names)},
        "interpretation": "Imposed frozen geometry, not a loaded equilibrium or locomotion result",
    }
    if nonwheel_min < -1e-4:
        raise ValueError(f"Nonwheel geometry penetrates floor: {nonwheel_min} m")
    robot.set_pos([0., 0., h_ref + height_offset], relative=True, zero_velocity=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=60, help="Viewer wall-clock seconds")
    parser.add_argument("--height-offset", type=float, default=0, help="Displayed z offset in m; h_ref is unchanged")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0 < args.duration < float("inf") or not abs(args.height_offset) < float("inf"):
        parser.error("duration must be finite/positive and height-offset finite")
    path = URDFReader(MEASURED_URDF).robot_file_path_absolute
    print(f"Asset: {path}", flush=True)
    gs.init(backend=gs.gpu, logging_level="warning")
    try:
        scene = gs.Scene(sim_options=gs.options.SimOptions(dt=.005), show_viewer=not args.headless,
                         viewer_options=gs.options.ViewerOptions(camera_pos=(1.6, -1.6, 1.0), camera_lookat=(0, 0, .3)))
        scene.add_entity(gs.morphs.Plane())
        robot = scene.add_entity(gs.morphs.URDF(file=str(path), align=False, fixed=False,
                                               merge_fixed_links=True, default_armature=.01))
        scene.build()
        report = inspect_pose(scene, robot, path, args.height_offset)
        print(f"h_ref={report['h_ref_m']:.9f} m; spawn={report['suggested_spawn_height_m']:.9f} m; max joint error={report['maximum_joint_error_rad']:.2g} rad")
        print("Wheel     floor gap m     center in base axes m                 dx from thigh m")
        for name, wheel in report["wheels"].items():
            print(f"{name:8s} {wheel['gap_m']: .8f}       {wheel['center_base_m'].cpu().tolist()}   {wheel['wheel_minus_thigh_origin_x_m']:.8f}")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, report)
        if not args.headless:
            deadline = time.monotonic() + args.duration
            while scene.viewer.is_alive() and time.monotonic() < deadline:
                scene.visualizer.update(force=True)  # Refresh rendering only; never scene.step().
                time.sleep(1 / 30)
    except KeyboardInterrupt:
        pass
    finally:
        gs.destroy()


if __name__ == "__main__":
    main()
