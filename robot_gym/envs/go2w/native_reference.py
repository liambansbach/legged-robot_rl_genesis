"""Bounded published-Actor companion to evaluate; no runner, training or export."""

import argparse
import copy
import importlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import tempfile
from time import perf_counter
from types import SimpleNamespace
from scipy.spatial.transform import Rotation

import numpy as np
import torch
import yaml
import genesis as gs

from robot_gym import ROBOT_GYM_ROOT_DIR
from robot_gym.envs import *  # noqa: F401,F403
from robot_gym.utils import task_registry
from robot_gym.utils.helpers import class_to_dict, update_cfg_from_args, get_args
from robot_gym.utils.diagnostics import (
    PhysicsDiagnostics, check_reference_contract, config_differences, joint_dynamics,
    loaded_properties, manifest, rotate_wxyz, sha256, summed_normal_force,
    cylinder_clearance, write_json, json_safe,
)
from robot_gym.utils.urdf_reader import URDFReader

ROOT = Path(ROBOT_GYM_ROOT_DIR)
PACKAGE = ROOT / "ressources/pretrained/go2w"
ASSETS = {
    "original": ("go2w_reference_original.urdf", "495401f076d408858f322dd5227f3166be4fd63e65b8bc648fcbdeffd4c98d44"),
    "measured": ("go2w_measured_ed8dc93.urdf", "d298cc7bf4894e869840bdab9ac60f09548d998444018e61854636cff46b1d8c"),
}


def restore_config(target, saved):
    """Restore the complete recorded configuration into the existing config type."""
    for key, value in saved.items():
        current = getattr(target, key, None)
        if isinstance(value, dict) and current is not None and not isinstance(current, dict):
            restore_config(current, value)
        else:
            setattr(target, key, copy.deepcopy(value))


def publication_check():
    """Run unchanged publication checks against its original asset snapshot."""
    spec = importlib.util.spec_from_file_location("go2w_publication_check", PACKAGE / "check_policy.py")
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    provenance = json.loads((PACKAGE / "provenance.json").read_text())
    with tempfile.TemporaryDirectory(prefix="go2w-publication-") as tmp:
        reference_root = Path(tmp)
        for relative in provenance["assets"]:
            source = ROOT / relative
            if relative.endswith("/go2w_description.urdf"):
                source = ROOT / "ressources/robots/go2w/urdf" / ASSETS["original"][0]
            destination = reference_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        contract, _ = checker.check_files(PACKAGE, reference_root)
        checker.check_interface(contract, reference_root, np)
    actor = torch.jit.load(str(PACKAGE / "policy.pt"), map_location="cpu").eval()
    before = {k: v.clone() for k, v in actor.state_dict().items()}
    with np.load(PACKAGE / "parity_samples.npz", allow_pickle=False) as samples:
        with torch.no_grad():
            actual = actor(torch.from_numpy(samples["observations"])).numpy()
        cpu_error = float(np.max(np.abs(actual - samples["expected_clipped_actions"])))
        gpu_error = float(np.max(np.abs(actual - samples["recorded_gpu_raw_actions"].clip(-1, 1))))
    if cpu_error > 2e-6 or gpu_error > 5e-5:
        raise ValueError("Published Actor parity failed")
    if not all(torch.equal(before[k], v) for k, v in actor.state_dict().items()):
        raise ValueError("Published Actor state changed during parity")
    for name, (filename, expected) in ASSETS.items():
        path = ROOT / "ressources/robots/go2w/urdf" / filename
        if sha256(path) != expected:
            raise ValueError(f"{name} immutable asset identity mismatch")
        if URDFReader(str(path)).joint_names != contract["joint_order"]:
            raise ValueError(f"{name} joint order mismatch")
    return contract, {"cpu_max_error": cpu_error, "gpu_recorded_max_error": gpu_error,
                      "publication_checker": "unchanged check_files/check_interface; original snapshot root; no hash rewrite"}


def prepare(asset):
    # Standard profile resolution/checks precede all diagnostic overrides.
    previous = sys.argv
    sys.argv = [previous[0], "--task", "go2w", "--go2w_profile", "event_step_v1",
                "--sagittal_stance_weight", "2.0", "--num_envs", "1", "--seed", "1", "--headless"]
    try:
        standard_args = get_args()
    finally:
        sys.argv = previous
    cfg, train = task_registry.get_cfgs("go2w")
    update_cfg_from_args(cfg, train, standard_args)
    cfg.asset.joint_names = URDFReader(cfg.asset.robot_file).joint_names
    checked = check_reference_contract(PACKAGE / "config.yaml", class_to_dict(cfg), class_to_dict(train))
    saved = yaml.safe_load((PACKAGE / "config.yaml").read_text())
    restore_config(cfg, saved["env_cfg"])
    restore_config(train, saved["train_cfg"])
    baseline = class_to_dict(cfg)
    check_reference_contract(PACKAGE / "config.yaml", baseline, class_to_dict(train))
    cfg.asset.robot_file = str(ROOT / "ressources/robots/go2w/urdf" / ASSETS[asset][0])
    cfg.env.num_envs = 1
    cfg.env.capture_transitions = cfg.env.capture_closed_loop = cfg.env.capture_joint_dynamics = True
    cfg.env.play_mode = True
    cfg.noise.add_noise = False
    cfg.terrain.curriculum = cfg.commands.curriculum = False
    for field in ("randomize_friction", "randomize_base_mass", "randomize_com", "randomize_kp",
                  "randomize_kd", "randomize_action_delay", "push_robots"):
        setattr(cfg.domain_rand, field, False)
    for field in ("joint_position_noise", "joint_velocity_noise", "linear_velocity_noise", "angular_velocity_noise"):
        setattr(cfg.init_state, field, 0.0)
    cfg.init_state.orientation_noise = (0.0, 0.0, 0.0)
    cfg.viewer.visualize_velocity_arrows = cfg.viewer.visualize_foot_contacts = False
    cfg.viewer.ref_env = [0]
    # Retain saved performance_mode=True, deterministic=False and batching.
    return cfg, train, baseline, checked


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


def interval_metrics(trace, start, end, leg_indices):
    time = trace["physics_time"]
    mask = (time > start + 1e-7) & (time <= end + 1e-7)
    if not mask.any():
        return {"complete": False, "requested_interval_s": [start, end]}
    def rms(value):
        return float(np.sqrt(np.mean(value ** 2)))
    body, dq = trace["angular_body"][mask], trace["dq"][mask][:, leg_indices]
    linear = trace["linear_body"][mask]
    return {"complete": bool(time[-1] >= end - 1e-6), "samples": int(mask.sum()),
            "planar_speed_rms_m_s": float(np.sqrt(np.mean(np.sum(linear[:, :2] ** 2, axis=1)))),
            "yaw_rate_rms_rad_s": rms(body[:, 2]), "leg_dq_rms_rad_s": rms(dq),
            "body_roll_pitch_rate_rms_rad_s": rms(body[:, :2]),
            "height_mean_m": float(trace["base_pos"][mask, 2].mean()),
            "mean_velocity": np.column_stack((linear[:, :2], body[:, 2])).mean(axis=0).tolist(),
            "leg_dq_spectrum": spectrum(dq, 200), "body_rate_spectrum": spectrum(body[:, :2], 200),
            "held_target_spectrum": spectrum(trace["targets"][mask][:, leg_indices], 200),
            "wheel_support_mean_N": trace["wheel_normal_force"][mask].mean(axis=0).tolist(),
            "minimum_clearance_m": float(trace["clearance"][mask].min())}


def longest_run(mask, dt):
    maximum = current = 0
    for value in mask:
        current = current + 1 if value else 0
        maximum = max(maximum, current)
    return maximum * dt


def metrics(trace, env, case, failure, initial):
    duration = 10.0 if case == "stand" else 14.0
    result = {"failure": failure, "recorded_seconds": float(trace["physics_time"][-1]),
              "initial_0_2": interval_metrics(trace, 0, 2, env.leg_action_indices),
              "settled_2_10": interval_metrics(trace, 2, 10, env.leg_action_indices),
              "final_2s": interval_metrics(trace, duration - 2, duration, env.leg_action_indices)}
    t = trace["physics_time"]
    def displacement(start, end):
        if t[-1] < end - 1e-6:
            return None
        a = initial if start == 0 else trace["base_pos"][np.flatnonzero(t <= start + 1e-7)[-1]]
        rows = trace["base_pos"][(t > start + 1e-7) & (t <= end + 1e-7)]
        path = np.vstack((a, rows))
        return {"signed_xyz_m": (rows[-1] - a).tolist(),
                "planar_displacement_m": float(np.linalg.norm((rows[-1] - a)[:2])),
                "planar_path_m": float(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1).sum())}
    result["last_5s_zero_drift"] = displacement(duration - 5, duration)
    bounds = env.dof_pos_limits.detach().cpu().numpy()[env.leg_action_indices]
    q = trace["q"][:, env.leg_action_indices]
    result["joint_minimum_margin_rad"] = np.minimum(q - bounds[:, 0], bounds[:, 1] - q).min(axis=0).tolist()
    result["issued_action_at_clip_fraction_per_joint"] = (np.abs(trace["actions"]) >= 1 - 1e-6).mean(axis=0).tolist()
    actions = trace["actions"][trace["policy_time"] >= 2][:, env.leg_action_indices]
    result["actor_spectrum"] = spectrum(actions, 50) if len(actions) else None
    result["actor_period_two_amplitude_per_leg"] = (
        np.abs(((actions - actions.mean(axis=0)) * (-1.0) ** np.arange(len(actions))[:, None]).mean(axis=0)).tolist()
        if len(actions) else None)
    result["maximum_force_effort_ratio_per_joint"] = (np.abs(trace["control_force"]) / env.torque_limits.detach().cpu().numpy()).max(axis=0).tolist()
    result["nonwheel_force_above_1N_longest_s"] = longest_run(trace["nonwheel_normal_force"][:, 0] > 1, .005)
    result["clearance_below_minus_2cm_longest_s"] = longest_run((trace["clearance"] < -.02).any(axis=1), .005)
    result["unloaded_above_1mm_fraction_per_wheel"] = (
        (trace["wheel_normal_force"] < 1) & (trace["clearance"] > .001)).mean(axis=0).tolist()
    result["support_mean_N"] = float(trace["wheel_normal_force"].sum(axis=1).mean())
    result["credible_support"] = (failure is None and result["nonwheel_force_above_1N_longest_s"] <= .1
                                  and result["clearance_below_minus_2cm_longest_s"] <= .1)
    final = result["final_2s"]
    result["development_targets"] = {"complete": final["complete"],
        "planar_rms_below_005": final.get("planar_speed_rms_m_s", float("inf")) < .05,
        "yaw_rms_below_010": final.get("yaw_rate_rms_rad_s", float("inf")) < .1,
        "zero_drift_below_010": result["last_5s_zero_drift"] is not None and result["last_5s_zero_drift"]["planar_displacement_m"] < .1}
    if case == "sequence":
        forward = (t > 5) & (t <= 8)
        result["forward_5_8"] = interval_metrics(trace, 5, 8, env.leg_action_indices)
        result["forward_5_8"]["vx_rmse_m_s"] = float(np.sqrt(np.mean((trace["linear_body"][forward, 0] - .2) ** 2))) if forward.any() else None
        result["forward_3_8_displacement"] = displacement(3, 8)
        result["body_forward_integrated_progress_3_8_m"] = float(trace["linear_body"][(t > 3) & (t <= 8), 0].sum() * .005)
        result["post_stop_8_14_displacement"] = displacement(8, 14)
        stop_time = t[t > 8] - 8
        stopped = ((np.linalg.norm(trace["linear_body"][t > 8, :2], axis=1) < .05)
                   & (np.abs(trace["angular_body"][t > 8, 2]) < .10))
        count = 0
        result["post_stop_first_half_second_settle_start_s"] = None
        for index, quiet in enumerate(stopped):
            count = count + 1 if quiet else 0
            if count >= 100:
                result["post_stop_first_half_second_settle_start_s"] = float(stop_time[index] - .5)
                break
    return result


def run(args):
    started = perf_counter()
    out = Path(args.output)
    if out.exists():
        raise ValueError(f"Refusing to overwrite evidence: {out}")
    contract, parity = publication_check()
    cfg, train, baseline, reference = prepare(args.asset)
    if args.preflight_only:
        print(json.dumps({"parity": parity, "contract": reference, "asset": args.asset}, indent=2))
        return
    out.mkdir(parents=True)
    launch = SimpleNamespace(num_envs=1, seed=1, sim_device="cuda:0", headless=not args.viewer)
    env, _ = task_registry.make_env("go2w", args=launch, env_cfg=cfg)
    initialized = joint_dynamics(env.robot, env.joint_names)
    if args.zero_armature:
        if not any((row["armature"] or 0) > 0 for row in initialized["joints"].values()):
            raise ValueError("Zero-armature test requires verified nonzero source armature")
        env.robot.set_dofs_armature(torch.zeros(16, device=env.device), env.joint_dof_idx)
        if any(row["armature"] != 0 for row in joint_dynamics(env.robot, env.joint_names)["joints"].values()):
            raise ValueError("Zero-armature effective readback failed")
    env.set_fixed_command((0, 0, 0))
    obs, _ = env.reset()
    reset_properties = joint_dynamics(env.robot, env.joint_names)
    if args.zero_armature and any(row["armature"] != 0 for row in reset_properties["joints"].values()):
        raise ValueError("Zero armature did not survive reset/control initialization")
    actual = loaded_properties(env)
    base = env.base_link_idx
    base_properties = {key: json_safe(actual[key][:, base]) for key in ("mass", "com_local", "inertia_local")}
    # Installed _init_link_fields feeds this read-only description quaternion to
    # the solver. No inertia/COM randomization is active. The inertia getter uses
    # the inertial frame, not the authored base_link frame.
    inertia_quat = np.asarray(env.robot.base_link.desc.inertial_quat)
    rotation = Rotation.from_quat(inertia_quat[[1, 2, 3, 0]]).as_matrix()
    inertia = actual["inertia_local"][0, base].detach().cpu().numpy()
    base_properties.update({
        "inertial_frame_quat_wxyz": inertia_quat.tolist(),
        "inertia_about_COM_in_base_frame_kg_m2": (rotation @ inertia @ rotation.T).tolist(),
        "frame_source": "RigidLink.desc.inertial_quat, fed into installed RigidSolver._init_link_fields; fixed during this no-DR run",
        "inertia_local_frame": "Genesis inertial principal frame",
    })
    solver = env.sim.rigid_solver
    settings = {"resolved_rigid_options": str(solver._options), "integrator": str(solver._integrator),
                "physics_dt_s": env.cfg.sim.dt, "internal_substeps": env.cfg.sim.substeps, "policy_dt_s": env.dt,
                "decimation": env.cfg.control.decimation, "action_delay_ticks": json_safe(env.action_delay_steps),
                "normal_play_overrides": {"performance_mode": False, "deterministic": True},
                "this_run_retains_saved_performance_mode": env.cfg.sim.performance_mode,
                "this_run_retains_saved_deterministic": env.cfg.sim.deterministic,
                "resolved_integrator_name": solver._integrator.name}
    policy = torch.jit.load(str(PACKAGE / "policy.pt"), map_location=env.device).eval()
    before = {k: v.clone() for k, v in policy.state_dict().items()}
    env.physics_diagnostics = ReferenceCapture(env)
    initial = env.base_pos[0].detach().cpu().numpy().copy()
    rows, failure = [], None
    duration = 10 if args.case == "stand" else 14
    rollout_start = perf_counter()
    with torch.no_grad():
        for step in range(round(duration / env.dt)):
            command = (0.2, 0, 0) if args.case == "sequence" and 3 <= step * env.dt < 8 else (0, 0, 0)
            env.set_fixed_command(command)
            obs = env.get_observations()
            actions = policy(obs["policy"])
            rows.append({"actions": actions[0].clone(), "observations": obs["policy"][0].clone(),
                         "commands": env.commands[0].clone()})
            env.step(actions)
            state = env.transition_state
            if not all(torch.isfinite(state[k]).all() for k in ("base_pos", "dof_pos", "dof_vel")):
                failure = {"kind": "nonfinite", "time_s": (step + 1) * env.dt}
            elif bool(state["reset_buf"][0]):
                failure = {"kind": "timeout" if bool(state["time_out_buf"][0]) else "fall",
                           "time_s": (step + 1) * env.dt, "pre_reset_state": json_safe(state)}
            if failure:
                break
    rollout_wall = perf_counter() - rollout_start
    if not all(torch.equal(before[k], v) for k, v in policy.state_dict().items()):
        raise ValueError("Frozen Actor state changed")
    trace = {k: torch.stack([r[k] for r in env.physics_diagnostics.rows]).cpu().numpy()
             for k in env.physics_diagnostics.rows[0]}
    trace.update({k: torch.stack([r[k] for r in rows]).cpu().numpy() for k in rows[0]})
    trace["physics_time"] = np.arange(1, len(trace["q"]) + 1) * .005
    trace["policy_time"] = np.arange(len(rows)) * .02
    np.savez_compressed(out / "trace.npz", **trace)
    result = {"schema": "go2w_native_reference_v1", "candidate": args.candidate, "asset": args.asset,
              "case": args.case, "zero_armature": args.zero_armature, "parity": parity,
              "manifest": manifest(ROOT, PACKAGE / "policy.pt", env.urdf_reader.robot_file_path_absolute,
                  class_to_dict(env.cfg), class_to_dict(train), config_differences(baseline, class_to_dict(env.cfg)), reference),
              "joint_order": env.joint_names, "after_build": env.joint_dynamics_after_build,
              "after_control_initialization": initialized, "after_reset": reset_properties,
              "properties": json_safe(actual), "merged_base": base_properties, "settings": settings,
              "metrics": metrics(trace, env, args.case, failure, initial),
              "trace": {"filename": "trace.npz", "sha256": sha256(out / "trace.npz"),
                        "physics_fields": list(env.physics_diagnostics.rows[0]), "policy_fields": list(rows[0]),
                        "semantics": "Post-physics 200Hz state; pre-step 50Hz Actor input/action; terminal physics rows before automatic reset. Control force getter recomputes clamped force, not measured integration impulse."},
              "setup_integration_s": .04, "rollout_wall_s": rollout_wall,
              "imports": {name: importlib.import_module(name).__file__ for name in ("torch", "genesis", "rsl_rl", "tensordict", "numpy")},
              "python_executable": sys.executable}
    genesis_root = Path(gs.__file__).parent
    result["installed_genesis_source_sha256"] = {
        p: sha256(genesis_root / p) for p in (
            "__init__.py", "options/morphs.py", "options/solvers.py", "utils/urdf.py",
            "engine/entities/rigid_entity/description.py", "engine/entities/rigid_entity/rigid_entity.py",
            "engine/solvers/rigid/rigid_solver.py")}
    result["additional_versions"] = {p: importlib.metadata.version(p) for p in (
        "tensordict", "numpy", "scipy", "quadrants", "trimesh", "pyglet")}
    gs.destroy()
    result["process_internal_wall_s"] = perf_counter() - started
    write_json(out / "result.json", result)
    print(json.dumps(result["metrics"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=["P_ck2498"], required=True)
    parser.add_argument("--asset", choices=list(ASSETS), required=True)
    parser.add_argument("--case", choices=["stand", "sequence"], default="stand")
    parser.add_argument("--output", required=True)
    parser.add_argument("--zero-armature", action="store_true")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.zero_armature and args.case != "stand":
        parser.error("Armature sensitivity is restricted to zero-command stand")
    run(args)


if __name__ == "__main__":
    main()
