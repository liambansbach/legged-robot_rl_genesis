"""Frozen 2026-10-09 Actor: bounded source-side yaw comparison, no learning."""

import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import genesis as gs
import numpy as np
import torch
import yaml

from robot_gym import ROBOT_GYM_ROOT_DIR
from robot_gym.utils import task_registry
from robot_gym.utils.diagnostics import (
    config_differences, heading_wxyz, joint_dynamics, json_safe, sha256,
    source_identity, write_json,
)
from robot_gym.utils.helpers import class_to_dict
from robot_gym.utils.replay import configure_nominal
from .diagnostics import PhysicsDiagnostics, ReferenceCapture, loaded_properties
from .go2w_config import restore_saved_config


BUNDLE = Path(ROBOT_GYM_ROOT_DIR) / "ressources/handoffs/go2w_2026-10-09/inference/go2w"
ACTOR_SHA = "639a443f3da3d9fa15d9e532b85959df55c7bd5618564797740adb66b1d2bfcb"
URDF_SHA = "d298cc7bf4894e869840bdab9ac60f09548d998444018e61854636cff46b1d8c"
YAWS = (.3, -.3, .6, -.6, .8, -.8, 1., -1.)


class YawCapture(ReferenceCapture):
    """Use the existing recorder for one second after yaw onset only."""

    def __init__(self, env, high_rate):
        super().__init__(env, contact_details=True)
        self.high_rate = high_rate
        self.tick = -1

    def after_substep(self):
        if self.high_rate and 100 <= self.tick < 150:
            super().after_substep()
        else:
            PhysicsDiagnostics.after_substep(self)


def material_readback(env):
    geoms = list(env.sim.rigid_solver.geoms)
    ids = [g.idx for g in geoms]
    return {
        "geom_indices": ids,
        "link_names": [g.link.name for g in geoms],
        "coefficients": torch.stack([g.get_friction() for g in geoms]),
        "ratios": env.sim.rigid_solver.get_geoms_friction_ratio(ids).clone(),
        "wheel_geom_indices": [g.idx for link in env.ankle_links for g in link.geoms],
    }


def reset_case(env, friction):
    """Same explicit reset as BaseTask.reset, with optional wheel-only ratio."""
    env.set_commands((0., 0., 0.))
    env.reset_idx(env.all_env_ids)
    if friction != 1.:
        env.robot.set_friction_ratio(
            torch.full((1, 4), friction, device=env.device), env.foot_link_indices_local)
    _, _, done, _ = env.step(torch.zeros_like(env.actions))
    if bool(done.any()):
        raise RuntimeError("Failure during explicit reset warmup")
    torch.testing.assert_close(env.phase, torch.full_like(env.phase, .025), atol=1e-6, rtol=0)
    assert not bool(env.actions.any()) and not bool(env.action_history.any())
    assert not bool(env.action_delay_steps.any())


def rollout(env, actor, yaw, capture):
    """400 native environment steps; retain terminal state before automatic reset."""
    rows, failure = [], None
    with torch.no_grad():
        for tick in range(400):
            command = (0., 0., yaw if 100 <= tick < 300 else 0.)
            env.set_commands(command)
            env.compute_observations()
            obs = env.get_observations()["policy"]
            pre = {"observation": obs.clone(), "phase": env.phase.clone(),
                   "previous_issued_action": env.actions.clone()}
            for name in ("base_pos", "base_quat", "base_lin_vel", "base_ang_vel", "dof_pos", "dof_vel"):
                pre[name] = getattr(env, name).clone()
            if not bool(torch.isfinite(obs).all()):
                failure = "nonfinite_observation_before_action"
                break
            action = actor(obs)
            if not bool(torch.isfinite(action).all()):
                failure = "nonfinite_action_before_step"
                break
            capture.tick = tick
            env.step(action)
            row = {"pre_" + k: v[0].detach().clone() for k, v in pre.items()}
            row.update({"post_" + k: v[0].detach().clone() for k, v in env.transition_state.items()})
            # transition phase is the phase used for action/reward, before its increment.
            row["post_next_phase"] = env.phase[0].clone()
            rows.append(row)
            if not all(bool(torch.isfinite(v).all()) for v in row.values()):
                failure = "nonfinite_state"
            elif bool(env.transition_state["fallen"].any()):
                failure = "fall"
            elif bool(env.transition_state["reset_buf"].any()):
                failure = "timeout_or_reset"
            if failure:
                break
    data = {key: torch.stack([r[key] for r in rows]).cpu().numpy() for key in rows[0]} if rows else {}
    data["pre_time_s"] = np.arange(len(rows)) * env.dt
    data["post_time_s"] = np.arange(1, len(rows) + 1) * env.dt
    if capture.rows:
        for key in capture.rows[0]:
            data["physics_" + key] = torch.stack([r[key] for r in capture.rows]).cpu().numpy()
        data["physics_time_s"] = 2. + np.arange(1, len(capture.rows) + 1) * env.cfg.sim.dt
    return data, failure


def metrics(data, yaw, contract, failure):
    """Only complete windows are scored; stop displacement includes its boundary."""
    n = len(data["post_time_s"])
    result = {"command_rad_s": yaw, "failure": failure, "actor_ticks": n,
              "sim_seconds_excluding_reset": n * .02, "windows": {}}
    if not n:
        return result
    legs, wheels = contract["leg_action_indices"], contract["wheel_action_indices"]
    v = np.c_[data["post_base_lin_vel"][:, :2], data["post_base_ang_vel"][:, 2]]
    target = data["post_wheel_velocity_targets"]
    wheel_speed = data["post_dof_vel"][:, wheels]
    leg_error = data["post_dof_pos"][:, legs] - data["post_leg_position_targets"]
    q = data["post_dof_pos"][:, legs]
    limits = np.asarray(contract["hard_joint_position_limits_rad"], dtype=float)[legs]
    result.update(
        first_actor_phase=float(data["pre_phase"][0]),
        yaw_onset_phase=float(data["pre_phase"][100]) if n > 100 else None,
        history_max_error=float(np.max(np.abs(data["pre_previous_issued_action"][1:] - data["post_actions"][:-1]))) if n > 1 else 0.,
        nonwheel_contact_steps=int(np.count_nonzero(data["post_nonfoot_ground_link_count"])),
        minimum_height_m=float(data["post_base_pos"][:, 2].min()),
    )
    for name, a, b in (("first_second", 100, 150), ("last_second", 250, 300),
                       ("full_hold", 100, 300), ("stop_full", 300, 400), ("stop_last_second", 350, 400)):
        if n < b or (failure is not None and n == b):
            result["windows"][name] = {"censored": True}
            continue
        s = slice(a, b)
        desired = [0, 0, yaw if a < 300 else 0]
        result["windows"][name] = {
            "censored": False, "pre_action_window_s": [a * .02, b * .02],
            "mean_body_vx_vy_wz": v[s].mean(axis=0),
            "rmse_vx_vy_wz": np.sqrt(np.mean((v[s] - desired) ** 2, axis=0)),
            "planar_rms_m_s": float(np.sqrt(np.mean(np.sum(v[s, :2] ** 2, axis=1)))),
            "wheel_target_mean_rad_s": target[s].mean(axis=0),
            "wheel_speed_mean_rad_s": wheel_speed[s].mean(axis=0),
            "wheel_tracking_rmse_rad_s": np.sqrt(np.mean((target[s] - wheel_speed[s]) ** 2, axis=0)),
            "leg_tracking_rmse_rad": np.sqrt(np.mean(leg_error[s] ** 2, axis=0)),
            "leg_action_clipped_fraction": (np.abs(data["post_raw_actions"][s][:, legs]) >= 1.).mean(axis=0),
            "wheel_action_clipped_fraction": (np.abs(data["post_raw_actions"][s][:, wheels]) >= 1.).mean(axis=0),
            "joint_limit_min_margin_rad": np.minimum(q[s] - limits[:, 0], limits[:, 1] - q[s]).min(axis=0),
            "wheel_load_fraction_gt8N": (data["post_summed_normal_ground_force"][s] > 8.).mean(axis=0),
            "wheel_normal_force_mean_N": data["post_summed_normal_ground_force"][s].mean(axis=0),
            "wheel_gap_gt1mm_fraction": (data["post_wheel_clearance"][s] > .001).mean(axis=0),
        }
    if n >= 300:
        for name, a, b in (("hold_motion", 100, 300), ("stopping", 300, 400)):
            if n < b:
                continue
            pos = np.concatenate((data["pre_base_pos"][a:a+1], data["post_base_pos"][a:b]))
            quat = np.concatenate((data["pre_base_quat"][a:a+1], data["post_base_quat"][a:b]))
            heading = np.unwrap(heading_wxyz(torch.from_numpy(quat)).numpy())
            result[name] = {"world_xy_displacement_m": pos[-1, :2] - pos[0, :2],
                            "xy_displacement_m": float(np.linalg.norm(pos[-1, :2] - pos[0, :2])),
                            "xy_path_length_m": float(np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1).sum()),
                            "heading_change_rad": float(heading[-1] - heading[0])}
        if n == 400 and failure is None:
            inside = (np.linalg.norm(v[300:, :2], axis=1) <= .02) & (np.abs(v[300:, 2]) <= .03)
            outside = np.flatnonzero(~inside)
            start = int(outside[-1] + 1) if len(outside) else 0
            result["stopping"]["settled_after_s"] = (start + 1) * .02 if 100 - start >= 13 else None
            result["stopping"]["settling_rule"] = "planar <=.02m/s and abs(body wz)<=.03rad/s, remaining inside through end for >=.25s; null means not observed"
    return json_safe(result)


def evaluate_source_yaw(args):
    started = perf_counter()
    if args.task != "go2w" or args.num_envs not in (None, 1) or args.seed not in (None, 1):
        raise ValueError("Frozen source yaw comparison requires Go2-W, one robot, seed1")
    if any((args.export, args.resume, args.manual_control, args.eval_noise_pushes,
            args.transfer_delay is not None, args.transfer_armature not in (None, "nominal"),
            args.eval_phase_offset, args.eval_rolling_phase_zero,
            args.eval_phase_transition, args.reference_config is not None,
            args.load_run is not None, args.checkpoint is not None)):
        raise ValueError("source_yaw fixes the frozen Actor/config and nominal inference settings")
    out = Path(args.output).resolve()
    if out.exists():
        raise ValueError(f"Refusing to overwrite evidence: {out}")
    contract = json.loads((BUNDLE / "contract.json").read_text())
    if sha256(BUNDLE / "policy.pt") != ACTOR_SHA:
        raise ValueError("Frozen Actor hash mismatch")
    saved = yaml.safe_load((BUNDLE / "config.yaml").read_text())
    cfg, train_cfg = restore_saved_config(saved)
    original = class_to_dict(cfg)
    args.seed, args.num_envs = 1, 1
    configure_nominal(cfg, args)
    cfg.env.num_envs, cfg.env.episode_length_s = 1, 15.
    cfg.env.capture_transitions = cfg.env.capture_closed_loop = cfg.env.capture_precision = True
    out.mkdir(parents=True)
    report = {"actor_sha256": ACTOR_SHA, "source": source_identity(ROBOT_GYM_ROOT_DIR),
              "frozen_bundle": str(BUNDLE), "selected_checkpoint_label": 1999,
              "friction": args.source_yaw_friction, "cases": {},
              "settings_changes_from_saved": config_differences(original, class_to_dict(cfg)),
              "runtime": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                          "packages": {k: importlib.metadata.version(k) for k in
                                       ("genesis-world", "torch", "numpy", "rsl-rl-lib", "quadrants", "PyYAML")},
                          "imports": {m.__name__: m.__file__ for m in (gs, torch, np, yaml)},
                          "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda},
              "semantics": {"time": "t=0 after one reset zero-action20ms tick; pre[t]=before Actor, post[t]=after its four physics steps",
                            "phase": "post_phase is the action/reward phase; post_next_phase is the advanced clock",
                            "velocity": "body axes at authored base-link origin; quaternion wxyz",
                            "contact": "Public get_contacts force_a/b are full world-frame contact forces; XY is tangential to the flat ground. Normal resultants alone are not tire shear/slip. No material-point slip measurement.",
                            "control_force": "public getter recomputes clipped controller force at current state; not solver motor impulse",
                            "high_rate": "optional --diagnostic_trace: 200Hz in2..3s for +/- .8 only; no missing data filled with zero"}}
    write_json(out / "metrics.json", report)
    env = None
    try:
        env, _ = task_registry.make_env("go2w", args=args, env_cfg=cfg, initial_command=(0., 0., 0.))
        assert sha256(env.urdf_reader.robot_file_path_absolute) == URDF_SHA
        assert env.joint_names == contract["joint_order"]
        assert env.dt == .02 and env.cfg.sim.dt == .005 and env.cfg.control.decimation == 4
        actor = torch.jit.load(str(BUNDLE / "policy.pt"), map_location=env.device).eval()
        actor_before = {k: v.clone() for k, v in actor.state_dict().items()}
        initialized = loaded_properties(env)
        for field, expected in (("armature", [.01] * 16), ("kp", contract["nominal_kp"]),
                                ("kv", contract["nominal_kd"]), ("stiffness", [0.] * 16),
                                ("damping", [0.] * 16), ("frictionloss", [0.] * 16)):
            actual = initialized[field].reshape(-1)
            torch.testing.assert_close(actual, actual.new_tensor(expected), atol=1e-7, rtol=0)
        torch.testing.assert_close(env.action_scale.reshape(-1), env.action_scale.new_tensor(contract["action_scale"]))
        torch.testing.assert_close(env.torque_limits, env.torque_limits.new_tensor(contract["effort_limits"]))
        materials = material_readback(env)
        write_json(out / "initialized.json", {"properties": initialized, "materials": materials,
                   "joint_dynamics": joint_dynamics(env.robot, env.joint_names),
                   "config": class_to_dict(env.cfg), "solver_options": {
                       k: v.value if hasattr(v, "value") else v
                       for k, v in env.sim.rigid_solver._options.model_dump().items()}})
        report.update(joint_order=env.joint_names, wheel_order=env.cfg.asset.foot_link_names,
                      urdf_sha256=URDF_SHA, physics_dt=.005, actor_dt=.02, display="headless" if args.headless else "existing Genesis viewer")
        for yaw in YAWS if args.source_yaw_friction == 1. else (.8, -.8):
            capture = YawCapture(env, args.diagnostic_trace and abs(yaw) == .8)
            env.physics_diagnostics = capture
            reset_case(env, args.source_yaw_friction)
            properties, actual_materials = loaded_properties(env), material_readback(env)
            for field in ("mass", "com_local", "inertia_local", "kp", "kv", "armature", "stiffness", "damping", "frictionloss"):
                torch.testing.assert_close(properties[field], initialized[field], rtol=0, atol=0)
            torch.testing.assert_close(properties["effective_friction"], torch.full_like(properties["effective_friction"], args.source_yaw_friction))
            torch.testing.assert_close(actual_materials["coefficients"], materials["coefficients"], rtol=0, atol=0)
            nonwheel = [i for i, g in enumerate(materials["geom_indices"]) if g not in materials["wheel_geom_indices"]]
            torch.testing.assert_close(actual_materials["ratios"][..., nonwheel], materials["ratios"][..., nonwheel], rtol=0, atol=0)
            name = f"yaw_{yaw:+.1f}"
            write_json(out / (name + "_reset.json"), {"properties": properties, "materials": actual_materials,
                       "joint_dynamics": joint_dynamics(env.robot, env.joint_names)})
            data, failure = rollout(env, actor, yaw, capture)
            np.savez_compressed(out / (name + ".npz"), **data)
            report["cases"][name] = metrics(data, yaw, contract, failure)
            write_json(out / "metrics.json", report)
            print(name + ": " + json.dumps(report["cases"][name]["windows"].get("last_second", {})), flush=True)
        report["actor_state_unchanged"] = all(torch.equal(actor_before[k], v) for k, v in actor.state_dict().items())
        assert report["actor_state_unchanged"]
    finally:
        if env is not None:
            gs.destroy()
        report["native_seconds_module_through_destroy"] = perf_counter() - started
        report["trace_sha256"] = {p.name: sha256(p) for p in out.glob("*.npz")}
        write_json(out / "metrics.json", report)
