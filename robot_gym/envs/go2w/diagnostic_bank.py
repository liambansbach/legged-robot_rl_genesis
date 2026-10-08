"""Small explicit reset/DR bank and standing probe; called by evaluate --eval_mode."""

import json
from time import perf_counter
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from robot_gym.utils.diagnostics import (
    loaded_properties,
    write_json,
    nominal_support_heights,
    sha256,
    heading_wxyz,
    rotate_wxyz,
)


def condition_bank(seed=240925):
    """Local RNG only. Every realized condition is saved; samples are not IID claims."""
    rng = np.random.default_rng(seed)
    nominal = dict(
        heading=0.0,
        roll=0.0,
        pitch=0.0,
        joint_offset=[0.0] * 16,
        velocity=[0.0] * 6,
        friction=1.0,
        added_mass=0.0,
        com=[0.0] * 3,
        kp=1.0,
        kv=1.0,
        delay=0,
    )
    result = [dict(nominal, id="nominal")]
    for sign in (-1, 1):
        result.append(
            dict(
                nominal,
                id=f"mass_gain_corner_{sign:+d}",
                added_mass=-0.5 if sign < 0 else 1.5,
                kp=1.1 if sign < 0 else 0.9,
                kv=1.1 if sign < 0 else 0.9,
                com=[sign * 0.015, -sign * 0.015, sign * 0.015],
            )
        )
    for i, heading in enumerate((np.pi / 2, -np.pi / 2, np.pi, 0.0, 0.0)):
        result.append(
            dict(
                nominal,
                id=f"heading_friction_delay_corner_{i}",
                heading=heading,
                friction=0.6 if i % 2 == 0 else 1.2,
                delay=i % 2,
            )
        )
    for i in range(24):
        result.append(
            dict(
                id=f"seed_{seed}_{i:02d}",
                heading=float(rng.uniform(-np.pi, np.pi)),
                roll=float(rng.uniform(-0.02, 0.02)),
                pitch=float(rng.uniform(-0.02, 0.02)),
                joint_offset=rng.uniform(-0.03, 0.03, 16).tolist(),
                velocity=rng.uniform(-0.03, 0.03, 6).tolist(),
                friction=float(rng.uniform(0.6, 1.2)),
                added_mass=float(rng.uniform(-0.5, 1.5)),
                com=rng.uniform(-0.015, 0.015, 3).tolist(),
                kp=float(rng.uniform(0.9, 1.1)),
                kv=float(rng.uniform(0.9, 1.1)),
                delay=int(rng.integers(0, 2)),
            )
        )
    return result


def apply_conditions(env, conditions, nominal):
    """Explicit setters after reset; no random draws and no extra command lag."""

    def tensor(values):
        return torch.tensor(values, dtype=torch.float32, device=env.device)

    env.reset_idx(env.all_env_ids)
    base = [env.base_link_idx]
    env.robot.set_links_mass(
        nominal["mass"][:, base]
        + tensor([c["added_mass"] for c in conditions])[:, None],
        base,
    )
    env.robot.set_links_COM(
        nominal["com_local"][:, base] + tensor([c["com"] for c in conditions])[:, None],
        base,
    )
    env.robot.set_friction_ratio(
        tensor([c["friction"] for c in conditions])[:, None].expand(-1, 4),
        env.foot_link_indices_local,
    )
    env.robot.set_dofs_kp(
        env.base_p_gains[None] * tensor([c["kp"] for c in conditions])[:, None],
        env.joint_dof_idx,
    )
    env.robot.set_dofs_kv(
        env.base_d_gains[None] * tensor([c["kv"] for c in conditions])[:, None],
        env.joint_dof_idx,
    )
    # Extrinsic xyz yields world heading after tilt. Genesis reset itself remains unchanged.
    q = Rotation.from_euler(
        "xyz", [[c["roll"], c["pitch"], c["heading"]] for c in conditions]
    ).as_quat()[:, [3, 0, 1, 2]]
    env.robot.set_quat(tensor(q))
    offsets = (
        tensor([c["joint_offset"] for c in conditions]) * env.position_control_mask
    )
    env.robot.set_dofs_position(
        env.default_dof_pos + offsets, env.joint_dof_idx, zero_velocity=True
    )
    velocities = torch.zeros((env.num_envs, env.robot.n_dofs), device=env.device)
    velocities[:, :6] = tensor([c["velocity"] for c in conditions])
    env.robot.set_dofs_velocity(velocities)
    env.action_delay_steps[:] = torch.tensor(
        [c["delay"] for c in conditions], device=env.device
    )
    env._update_robot_state()
    env.compute_observations()


def summarize(
    data, commands, done, dt, transition, leg_ids, limits, default_pos, action_scale
):
    """Per-condition outcomes kept alongside survivor-only errors; units remain explicit."""
    velocity = np.concatenate(
        (
            data["base_linear_velocity_body"][..., :2],
            data["base_angular_velocity_body"][..., 2:3],
        ),
        axis=-1,
    )
    result = []
    for i in range(done.shape[1]):
        fallen = bool(done[:, i].any())
        end = int(np.flatnonzero(done[:, i])[0] + 1) if fallen else len(done)
        support = data["legacy_force_support"][:end, i]
        final = slice(max(0, end - round(2 / dt)), end)
        target = data["leg_position_targets"][final, i]
        margin = action_scale[leg_ids] - np.abs(target - default_pos[leg_ids])
        item = {
            "fell": fallen,
            "reset_count": int(done[:, i].sum()),
            "nonwheel_ground_contact_steps": int(
                (data["nonfoot_contact_count"][:end, i] > 0).sum()
            )
            if "nonfoot_contact_count" in data
            else None,
            "measurement_end_step_exclusive": end,
            "physical_summary_window": "Up to first fall inclusive; final window is its last <=2 seconds",
            "absolute_world_xyz_displacement_m": np.abs(
                data["base_position"][end - 1, i] - data["base_position"][0, i]
            ).tolist(),
            "survivor_velocity_rmse_vx_vy_yaw": None
            if fallen
            else np.sqrt(((velocity[:, i] - commands) ** 2).mean(axis=0)).tolist(),
            "final_velocity_all_six_axes": np.r_[
                data["base_linear_velocity_body"][final, i].mean(axis=0),
                data["base_angular_velocity_body"][final, i].mean(axis=0),
            ].tolist(),
            "legacy_wheel_sample_false_fraction": float((~support).mean()),
            "legacy_any_wheel_false_fraction": float((~support).any(axis=-1).mean()),
            "geometric_clearance_min_max_m": [
                float(data["wheel_clearance"][:end, i].min()),
                float(data["wheel_clearance"][:end, i].max()),
            ],
            "geometric_gap_gt_1mm_wheel_sample_fraction": float(
                (data["wheel_clearance"][:end, i] > 0.001).mean()
            ),
            "final_mean_normal_ground_force_N": data["summed_normal_ground_force"][
                final, i
            ]
            .mean(axis=0)
            .tolist(),
            "final_mean_control_torque_Nm": data["control_torques"][final, i]
            .mean(axis=0)
            .tolist(),
            "all_substeps_peak_abs_control_torque_Nm": data[
                "control_torques_substep_max_abs"
            ][:end, i]
            .max(axis=0)
            .tolist(),
            "torque_limit_margin_Nm": (
                limits - data["control_torques_substep_max_abs"][:end, i].max(axis=0)
            ).tolist(),
            "final_leg_target_offset_margin_rad": margin.min(axis=0).tolist(),
            "final_leg_position_error_rad": (
                data["dof_position"][final, i][:, leg_ids] - default_pos[leg_ids]
            )
            .mean(axis=0)
            .tolist(),
            "final_height_m": float(data["base_position"][final, i, 2].mean()),
            "post_stop_absolute_drift": None,
        }
        if transition is not None and not fallen:
            # Boundary is the state immediately before the first zero command.
            pos = data["base_position"][transition - 1 :, i]
            heading = np.unwrap(data["base_heading"][transition - 1 :, i])
            item["post_stop_absolute_drift"] = {
                "absolute_world_xyz_m": np.abs(pos[-1] - pos[0]).tolist(),
                "xy_displacement_m": float(np.linalg.norm(pos[-1, :2] - pos[0, :2])),
                "xy_path_length_m": float(
                    np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1).sum()
                ),
                "absolute_heading_rad": float(abs(heading[-1] - heading[0])),
                "heading_total_variation_rad": float(np.abs(np.diff(heading)).sum()),
            }
        result.append(item)
    return result


def verify_selective_reset(env, out):
    before = loaded_properties(env)
    # Bank: a condition with every property varied. Three-case probe: mass/gain corner.
    reset_id = 8 if env.num_envs > 8 else 1
    env.reset_idx(torch.tensor([reset_id], device=env.device))
    after = loaded_properties(env)
    unchanged, reset_changes = {}, {}
    for key, value in before.items():
        if torch.is_tensor(value):
            if value.ndim and value.shape[0] == env.num_envs:
                keep = torch.arange(env.num_envs, device=value.device) != reset_id
                unchanged[key] = bool(torch.equal(value[keep], after[key][keep]))
                reset_changes[key] = float(
                    (value[reset_id] - after[key][reset_id]).abs().max()
                )
            else:
                unchanged[key] = bool(torch.equal(value, after[key]))
    write_json(
        out / "selective_reset_readback.json",
        {
            "reset_env_id": reset_id,
            "before": before,
            "after": after,
            "non_reset_unchanged": unchanged,
            "reset_env_max_absolute_change": reset_changes,
        },
    )
    if not all(unchanged.values()):
        raise AssertionError(
            f"Selective reset changed non-reset properties: {unchanged}"
        )


def sustained_schedule():
    """Fixed diagnostic holds, not a training distribution or robustness bank."""
    commands = [
        ("yaw_p040", (0, 0, 0.4)), ("yaw_n040", (0, 0, -0.4)),
        ("yaw_p075", (0, 0, 0.75)), ("yaw_n075", (0, 0, -0.75)),
        ("lateral_p020", (0, 0.2, 0)), ("lateral_n020", (0, -0.2, 0)),
        ("straight_p050", (0.5, 0, 0)),
        ("diagonal_p010", (0.7, 0.1, 0)), ("diagonal_n010", (0.7, -0.1, 0)),
    ]
    return {name: [((0, 0, 0), 2), (command, 25), ((0, 0, 0), 6)]
            for name, command in commands}


def rollout_sequence(env, policy, schedule, stop_on_reset=False):
    """Capture pre-reset transitions; optionally censor at the first terminal step."""
    history, dones, commands, phases = {}, [], [], []
    stopped, start = False, 0
    with torch.no_grad():
        for command, seconds in schedule:
            count = round(seconds / env.dt)
            phase = {"command": command, "start": start, "steps": count, "recorded_steps": 0}
            start += count
            phases.append(phase)
            for _ in range(0 if stopped else count):
                env.set_commands(command)
                env.compute_observations()
                _, _, done, _ = env.step(policy(env.get_observations()))
                for key, value in env.transition_state.items():
                    history.setdefault(key, []).append(value.clone())
                dones.append(done.clone())
                commands.append(command)
                phase["recorded_steps"] += 1
                if stop_on_reset and bool(done.any()):
                    stopped = True
                    break
            phase["censored"] = phase["recorded_steps"] < count or stopped
    data = {key: torch.stack(values).cpu().numpy() for key, values in history.items()}
    data["done"] = torch.stack(dones).cpu().numpy()
    data["command_stream"] = np.asarray(commands)
    return data, phases


CLOSED_LOOP_BOUNDS = ((-0.25, -0.20, -0.50), (0.70, 0.20, 0.50))
CLOSED_LOOP_CASES = {"straight": (0.5, 0.0), "diagonal_left": (0.5, 0.1),
                     "diagonal_right": (0.5, -0.1)}


def closed_loop_reference(seconds):
    """Phase, integrated speed and speed; 2/12/3/3 s, with no moving reanchor."""
    if seconds < 2:
        return 0, 0.0, 0.0
    if seconds >= 14:
        return (2 if seconds < 17 else 3), 11.0, 0.0
    t = seconds - 2
    if t <= 1:
        return 1, 0.5 * t * t, t
    if t <= 11:
        return 1, t - 0.5, 1.0
    d = t - 11
    return 1, 10.5 + d - 0.5 * d * d, 1.0 - d


def closed_loop_request(position, quaternion, reference, velocity_world, heading):
    """Fixed 1/s position and 1.5/s heading feedback; clip commands only."""
    correction = reference - position[..., :2]
    inverse = quaternion * quaternion.new_tensor([1, -1, -1, -1])
    zeros = torch.zeros_like(correction[..., :1])
    body = rotate_wxyz(inverse, torch.cat((velocity_world + correction, zeros), dim=-1))
    error = heading - heading_wxyz(quaternion)
    yaw = 1.5 * torch.atan2(torch.sin(error), torch.cos(error))
    request = torch.cat((body[..., :2], yaw[..., None]), dim=-1)
    low, high = request.new_tensor(CLOSED_LOOP_BOUNDS).unbind()
    return request, request.clamp(low, high), correction


def rollout_closed_loop(env, policy, nominal_velocity):
    """GPU tensor accumulation, one packed CPU copy, and first-failure censoring."""
    if (env.num_envs != 1 or abs(env.dt - .02) > 1e-9
            or env.cfg.control.decimation != 4 or abs(env.cfg.sim.dt - .005) > 1e-9):
        raise ValueError("Closed loop requires 50 Hz policy, four 5 ms substeps and 10 Hz = five policy steps")
    rows, layout, failure = [], {}, None
    command = torch.zeros_like(env.commands)
    request, correction = command.clone(), command[:, :2].clone()
    p0, psi0, nominal_world = None, None, None
    initial_episode_step = int(env.episode_length_buf[0])
    with torch.no_grad():
        for step in range(1000):
            phase, distance, speed = closed_loop_reference(step * env.dt)
            if step == 100:
                p0 = env.base_pos[:, :2].clone()
                psi0 = heading_wxyz(env.base_quat).clone()
                vx, vy = nominal_velocity
                nominal_world = torch.stack((vx * psi0.cos() - vy * psi0.sin(),
                                             vx * psi0.sin() + vy * psi0.cos()), dim=-1)
            update = phase in (1, 2) and step % 5 == 0
            if update:
                request, command, correction = closed_loop_request(
                    env.base_pos, env.base_quat, p0 + nominal_world * distance,
                    nominal_world * speed, psi0)
            elif phase in (0, 3):
                command.zero_()
                request = torch.zeros_like(command)
                correction = torch.zeros_like(correction)
            env.set_commands(command)
            env.compute_observations()
            observations = env.get_observations()
            policy_observation = observations["policy"].clone()
            raw = policy(observations)
            if not bool(torch.isfinite(raw).all()):
                failure = "nonfinite_policy_before_step"
                break
            env.step(raw)
            state = env.transition_state
            _, end_distance, end_speed = closed_loop_reference((step + 1) * env.dt)
            # Settling has no anchored reference; its placeholder is excluded from errors.
            reference = torch.zeros_like(correction) if p0 is None else p0 + nominal_world * end_distance
            values = {k: state[k] for k in (
                "base_pos", "base_quat", "base_lin_vel", "base_ang_vel", "dof_pos", "dof_vel",
                "torques", "actions", "applied_actions", "foot_contacts", "nonfoot_contact_count",
                "fallen", "time_out_buf", "reset_buf", "episode_length_buf")}
            for key in ("wheel_normal_force", "loaded_wheels"):
                if key in state:
                    values[key] = state[key]
            if getattr(env, "physics_diagnostics", None) is not None:
                values.update(state)
            values.update(policy_observation=policy_observation, raw_actions=raw,
                          issued_commands=command, unclipped_commands=request,
                          position_correction_world=correction, reference_position_world=reference,
                          reference_velocity_world=torch.zeros_like(correction) if p0 is None else nominal_world * end_speed,
                          reference_heading=torch.zeros(1, device=env.device) if psi0 is None else psi0,
                          phase=command.new_tensor([phase]), outer_update=command.new_tensor([update]))
            if not layout:
                offset = 0
                for key, value in values.items():
                    layout[key] = (offset, offset + value.numel(), tuple(value.shape[1:]))
                    offset += value.numel()
            row = torch.cat([v.reshape(-1).float() for v in values.values()])
            rows.append(row)
            if not bool(torch.isfinite(row).all()):
                failure = "nonfinite_state"
            elif bool(state["fallen"].any()):
                failure = "fall"
            elif bool(state["time_out_buf"].any()):
                failure = "timeout"
            elif bool(state["reset_buf"].any()) or int(state["episode_length_buf"][0]) != initial_episode_step + step + 1:
                failure = "reset"
            if failure:
                break
    packed = torch.stack(rows).cpu().numpy() if rows else np.empty((0, 0))
    data = {key: packed[:, a:b].reshape((len(rows), *shape)) for key, (a, b, shape) in layout.items()}
    data["time_s"] = np.arange(1, len(rows) + 1) * env.dt
    return data, failure


def closed_loop_metrics(data, failure, dt):
    """Physical summaries with incomplete windows left censored, not called recovery."""
    count = len(data["time_s"])
    result = {"recorded_steps": count, "duration_s": count * dt, "failure": failure,
              "uninterrupted": failure is None and count == 1000,
              "phases": [], "endpoint_error_m": None, "exact_zero": None, "windows": {}}
    for phase, (name, start, end) in enumerate((("settle", 0, 100), ("move", 100, 700),
                                               ("endpoint_feedback", 700, 850), ("exact_zero", 850, 1000))):
        result["phases"].append({"name": name, "start_s": start * dt, "planned_steps": end - start,
                                 "recorded_steps": max(0, min(count, end) - start),
                                 "censored": count < end or (failure is not None and start < count <= end)})
    if not count:
        return result
    result.update(fall_count=int(data["fallen"].sum()), timeout_count=int(data["time_out_buf"].sum()),
                  reset_after_sample_indices=np.flatnonzero(data["reset_buf"]).tolist(),
                  nonwheel_contact_steps=int((data["nonfoot_contact_count"] > 0).sum()))
    finite = np.isfinite(np.concatenate([v.reshape(count, -1) for v in data.values()], axis=1)).all(axis=1)
    valid_end = int(np.flatnonzero(~finite)[0]) if not finite.all() else count
    if valid_end <= 100:
        return result
    position = data["base_pos"][:valid_end, :2]
    heading = heading_wxyz(torch.from_numpy(data["base_quat"][:valid_end])).numpy()
    error = data["reference_position_world"][:valid_end] - position
    heading_error = np.arctan2(np.sin(data["reference_heading"][:valid_end] - heading),
                              np.cos(data["reference_heading"][:valid_end] - heading))
    direction = data["reference_velocity_world"][100]
    direction = direction / np.linalg.norm(direction)
    cross = (position - position[99]) @ np.array([-direction[1], direction[0]])
    body = np.concatenate((data["base_lin_vel"], data["base_ang_vel"]), axis=1)

    def distribution(values):
        return {"rms": float(np.sqrt(np.mean(values ** 2))),
                "p95_abs": float(np.percentile(np.abs(values), 95)), "max_abs": float(np.max(np.abs(values)))}

    # Include moving/hold transients, one-second windows and a distinct late hold.
    windows = [("move", 100, 700), ("endpoint_feedback", 700, 850), ("feedback_all", 100, 850),
               ("late_hold", 750, 850), *[(f"second_{i:02d}", i * 50, (i + 1) * 50) for i in range(2, 20)]]
    for name, start, end in windows:
        if end > valid_end:
            result["windows"][name] = {"censored": True}
            continue
        s = slice(start, end)
        commands = data["issued_commands"][s]
        request = data["unclipped_commands"][s]
        velocity_error = body[s][:, [0, 1, 5]] - commands
        result["windows"][name] = {
            "censored": False, "position_error_m": distribution(np.linalg.norm(error[s], axis=1)),
            "cross_track_m": distribution(cross[s]), "heading_error_rad": distribution(heading_error[s]),
            "body_velocity_mean_six_axes": body[s].mean(axis=0).tolist(),
            "body_velocity_std_six_axes": body[s].std(axis=0).tolist(),
            "velocity_bias_vx_vy_yaw": velocity_error.mean(axis=0).tolist(),
            "velocity_rmse_vx_vy_yaw": np.sqrt((velocity_error ** 2).mean(axis=0)).tolist(),
            "command_mean": commands.mean(axis=0).tolist(), "command_std": commands.std(axis=0).tolist(),
            "command_min": commands.min(axis=0).tolist(), "command_max": commands.max(axis=0).tolist(),
            "command_clipped_fraction": (np.abs(request - commands) > 1e-6).mean(axis=0).tolist(),
            "command_at_bound_fraction": ((np.abs(commands - np.asarray(CLOSED_LOOP_BOUNDS[0])) < 1e-6)
                                          | (np.abs(commands - np.asarray(CLOSED_LOOP_BOUNDS[1])) < 1e-6)).mean(axis=0).tolist(),
            "position_error_xy_peak_to_peak_m": np.ptp(error[s], axis=0).tolist(),
            "heading_error_peak_to_peak_rad": float(np.ptp(heading_error[s])),
            "position_correction_m_s": distribution(np.linalg.norm(data["position_correction_world"][s], axis=1)),
            "yaw_correction_rad_s": distribution(request[:, 2]),
        }
        # Analysis tolerance only; the issued controller has no deadband.
        result["windows"][name]["command_sign_reversals_above_001"] = [
            int(np.sum(np.diff(np.sign(axis[np.abs(axis) > .01])) != 0)) for axis in commands.T]
        result["windows"][name]["velocity_sign_reversals_above_001"] = [
            int(np.sum(np.diff(np.sign(axis[np.abs(axis) > .01])) != 0)) for axis in body[s][:, [0, 1, 5]].T]
    if valid_end >= 850:
        result["endpoint_error_m"] = float(np.linalg.norm(error[849]))
    if valid_end == 1000 and failure is None:
        path = position[849:]
        yaw_change = float(np.unwrap(heading[849:])[-1] - heading[849])
        result["exact_zero"] = {
            "last_second_xy_rms_m_s": float(np.sqrt((body[-50:, :2] ** 2).sum(axis=1).mean())),
            "last_second_yaw_rms_rad_s": float(np.sqrt((body[-50:, 5] ** 2).mean())),
            "displacement_m": float(np.linalg.norm(path[-1] - path[0])),
            "path_m": float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
            "signed_heading_change_rad": yaw_change, "absolute_heading_change_rad": abs(yaw_change),
            "includes_braking": True,
        }
    return result


def evaluate_closed_loop(env, policy, out):
    """Three fixed nominal probes; no search, policy adaptation or export."""
    if abs(env.max_episode_length * env.dt - 30) > 1e-6:
        raise ValueError("Closed-loop episode limit must be 30 s")
    report = {"mode": "closed_loop", "tests": {}, "joint_order": env.joint_names,
              "position_limits": [[float(v) if np.isfinite(v) else None for v in pair]
                                  for pair in env.dof_pos_limits.cpu().tolist()],
              "position_limits_semantics": "null denotes an unbounded continuous joint",
              "effort_limits": env.torque_limits.cpu().tolist(),
              "controller": {"position_gain_per_s": 1.0, "heading_gain_per_s": 1.5,
                             "command_bounds": CLOSED_LOOP_BOUNDS, "policy_hz": 50, "feedback_hz": 10,
                             "physics_step_s": .005, "physics_steps_per_policy": 4,
                             "nominal_velocities_initial_heading_frame": CLOSED_LOOP_CASES},
              "heavy_recorder": getattr(env, "physics_diagnostics", None) is not None,
              "trace_semantics": "Post-step, pre-reset state; reference at sample time. Commands and requests held five steps. "
                                 "raw_actions: actor proposal; actions: clipped before delay; applied_actions: after delay. "
                                 "torques: existing policy-rate control-force getter, not substep maxima. "
                                 "policy_observation: pre-step scaled actor input, before trained normalization. "
                                 "Reference anchors once after settling; no valid reference in phase 0.",
              "wall_clock_s": {"rollout": 0.0, "postprocessing": 0.0}}
    for name, velocity in CLOSED_LOOP_CASES.items():
        started = perf_counter()
        env.reset()
        data, failure = rollout_closed_loop(env, policy, velocity)
        report["wall_clock_s"]["rollout"] += perf_counter() - started
        started = perf_counter()
        result = closed_loop_metrics(data, failure, env.dt)
        np.savez_compressed(out / f"{name}.npz", **data, dt=env.dt)
        result.update(trace=f"{name}.npz", normal_reset_before_sequence=True)
        report["tests"][name] = result
        report["wall_clock_s"]["postprocessing"] += perf_counter() - started
        write_json(out / "metrics.json", report)
        print(f"Closed loop {name}: {result['recorded_steps']} steps, failure={failure}, "
              f"endpoint error={result['endpoint_error_m']}", flush=True)
        if failure:
            break
    return report


def evaluate_sustained(env, policy, out):
    """One environment, nine uninterrupted sequences, with normal resets between them."""
    import xml.etree.ElementTree as ET

    if env.num_envs != 1 or env.max_episode_length * env.dt < 40:
        raise ValueError("Sustained sequences require one environment and a >=40 s timeout")
    tree = ET.parse(env.urdf_reader.robot_file_path_absolute).getroot()
    wheel_joints = [env.joint_names[i] for i in env.wheel_action_indices]
    axes = [[float(x) for x in tree.find(f"joint[@name='{name}']/axis").get("xyz").split()]
            for name in wheel_joints]
    report = {
        "dt": env.dt, "episode_timeout_s": env.max_episode_length * env.dt,
        "joint_order": env.joint_names, "wheel_order": env.cfg.asset.foot_link_names,
        "wheel_joint_axes": axes, "wheel_indices": env.wheel_action_indices,
        "leg_indices": env.leg_action_indices, "tests": {},
        "trace_semantics": "raw_actions: actor proposal; actions: issued clipped action before delay; "
                           "applied_actions: after delay. Terminal samples precede reset; no post-reset samples follow.",
    }
    for name, schedule in sustained_schedule().items():
        env.reset()
        initial = {key: getattr(env, key).cpu().numpy().copy()
                   for key in ("base_pos", "base_quat", "foot_pos", "episode_length_buf")}
        data, phases = rollout_sequence(env, policy, schedule, stop_on_reset=True)
        np.savez_compressed(out / f"{name}.npz", **data, dt=env.dt,
                            **{f"initial_{key}": value for key, value in initial.items()})
        report["tests"][name] = {
            "phases": phases, "trace": f"{name}.npz",
            "fall_count": int(data["fallen"].sum()),
            "timeout_count": int(data["time_out_buf"].sum()),
            "reset_after_sample_indices": np.flatnonzero(data["done"][:, 0]).tolist(),
            "normal_reset_before_sequence": True,
            "nonwheel_contact_steps": int((data["nonfoot_contact_count"] > 0).sum()),
            "uninterrupted": not bool(data["done"].any()),
        }
        write_json(out / "metrics.json", report)
        print(f"{name}: {report['tests'][name]}", flush=True)
    return report


def evaluate_restarts(env, policy, out, prefix="brake"):
    """Two fixed restart traces using the same policy, step and reset paths."""
    report = {}
    old_timeout = env.max_episode_length
    env.max_episode_length = max(old_timeout, round(25 / env.dt))
    try:
        with torch.no_grad():
            for name, positive, negative in (
                (f"{prefix}_restart_vx", (0.1, 0, 0), (-0.1, 0, 0)),
                (f"{prefix}_restart_yaw", (0, 0, 0.4), (0, 0, -0.4)),
            ):
                env.reset()
                schedule = [(command, 6 if command == (0, 0, 0) else 3)
                            for command in ((0, 0, 0), positive, (0, 0, 0), negative, (0, 0, 0))]
                data, phases = rollout_sequence(env, policy, schedule)
                np.savez_compressed(
                    out / f"{name}.npz", **data, dt=env.dt,
                )
                report[name] = {
                    "phases": phases,
                    "episode_timeout_s": env.max_episode_length * env.dt,
                    "fall_count": int(data["done"].sum()),
                    "nonwheel_contact_steps": int((data["nonfoot_contact_count"] > 0).sum()),
                    "trace": f"{name}.npz",
                }
                print(f"{name}: {report[name]}", flush=True)
    finally:
        env.max_episode_length = old_timeout
    return report


def evaluate_bank(env, runner, args, out):
    conditions = condition_bank(args.bank_seed)
    if args.eval_mode == "equilibrium":
        conditions = conditions[:3]
    if env.num_envs != len(conditions):
        raise ValueError(f"{args.eval_mode} requires --num_envs {len(conditions)}")
    nominal = loaded_properties(env)
    apply_conditions(env, conditions, nominal)
    write_json(out / "conditions.json", conditions)
    metadata = json.loads((out / "manifest.json").read_text())
    metadata["eval_overrides"]["condition_bank"] = {
        "file": "conditions.json",
        "sha256": sha256(out / "conditions.json"),
        "count": len(conditions),
        "policy_steps_per_phase": max(300, args.steps),
        "navigation_command_hz": 10,
    }
    write_json(out / "manifest.json", metadata)
    write_json(out / "condition_properties.json", loaded_properties(env))
    verify_selective_reset(env, out)
    policy = runner.get_inference_policy(device=env.device)
    hold = max(
        300, args.steps
    )  # >=6 s per phase, regardless of short nominal smoke steps.
    sequences = [
        ("zero_action_stand", (0, 0, 0), (0, 0, 0)),
        ("stand", (0, 0, 0), (0, 0, 0)),
        ("precision", (0.1, 0, 0), (0.1, 0, 0)),
        ("vx_to_stop", (0.5, 0, 0), (0, 0, 0)),
        ("yaw_positive_to_stop", (0, 0, 0.4), (0, 0, 0)),
        ("yaw_negative_to_stop", (0, 0, -0.4), (0, 0, 0)),
        ("lateral_positive", (0, 0.25, 0), (0, 0.25, 0)),
        ("lateral_negative", (0, -0.25, 0), (0, -0.25, 0)),
        ("navigation_10hz", (0, 0, 0), (0, 0, 0)),
    ]
    if args.eval_mode == "equilibrium":
        sequences = [
            ("zero_action_stand", (0, 0, 0), (0, 0, 0)),
            ("policy_stand", (0, 0, 0), (0, 0, 0)),
        ]
    report = {
        "dt": env.dt,
        "conditions": conditions,
        "tests": {},
        "tracking_axes": ["body_vx", "body_vy", "body_yaw_rate"],
        "nominal_joint_pose_required_base_height_per_wheel_m": nominal_support_heights(
            env.urdf_reader.robot_file_path_absolute,
            env.cfg.init_state.default_joint_angles,
            env.cfg.asset.foot_link_names,
        ),
        "reward_height_target_m": env.cfg.rewards.base_height_target,
        "scope": "Fixed stress bank; no IID robustness claim. Failed conditions retain outcomes and raw traces; tracking/drift summaries exclude them.",
        "payload_model": "Existing DR semantics: base mass and local COM setters; inertia unchanged, explicitly read back.",
        "navigation": "10 Hz zero-order-held command stream; only configured 0/1 policy-step actuator delay; no additional dynamics filter",
    }
    with torch.no_grad():
        for name, before, after in sequences:
            if name == "zero_action_stand" and args.skip_zero_action_probe:
                continue
            apply_conditions(env, conditions, nominal)
            history, dones, commands = {}, [], []
            for step in range(2 * hold):
                cmd = before if step < hold else after
                if name == "navigation_10hz":
                    t = (step // round(0.1 / env.dt)) * 0.1
                    cmd = (
                        (
                            0.3 + 0.2 * np.sin(0.7 * t),
                            0.08 * np.sin(0.5 * t),
                            0.5 * np.sin(0.9 * t),
                        )
                        if step < hold
                        else (0, 0, 0)
                    )
                env.set_commands(cmd)
                env.compute_observations()
                action = (
                    torch.zeros_like(env.actions)
                    if name == "zero_action_stand"
                    else policy(env.get_observations())
                )
                _, _, done, _ = env.step(action)
                for key, value in env.transition_state.items():
                    history.setdefault(key, []).append(value.clone())
                dones.append(done.clone())
                commands.append(cmd)
            data = {
                key: torch.stack(values).cpu().numpy()
                for key, values in history.items()
            }
            done = torch.stack(dones).cpu().numpy()
            commands = np.asarray(commands)
            np.savez_compressed(
                out / f"{name}.npz",
                **data,
                done=done,
                command_stream=commands,
                dt=env.dt,
                condition_ids=[c["id"] for c in conditions],
            )
            stop = hold if before != after or name == "navigation_10hz" else None
            metrics = summarize(
                data,
                commands,
                done,
                env.dt,
                stop,
                env.leg_action_indices,
                env.torque_limits.cpu().numpy(),
                env.default_dof_pos[0].cpu().numpy(),
                env.action_scale.cpu().numpy().reshape(-1),
            )
            for condition, result in zip(conditions, metrics):
                result["condition_id"] = condition["id"]
            report["tests"][name] = metrics
            write_json(out / "metrics.json", report)
            print(
                f"{name}: {sum(m['fell'] for m in metrics)}/{len(metrics)} fell; saved per-condition results",
                flush=True,
            )


def precision_schedule(dr=False):
    zero = (0., 0., 0.)
    if dr:
        return {"dynamics": [(2, zero), (4, (.5, 0, 0)), (4, (0, 0, .4)),
                             (4, (0, 0, -.4)), (4, (0, .15, 0)), (4, (0, -.15, 0)), (4, zero)]}
    cases = {"rolling_reverse": [(2, zero), (3, (.1, 0, 0)), (4, (.5, 0, 0)),
                                  (3, (-.25, 0, 0)), (3, zero)]}
    for sign, name in ((1, "positive"), (-1, "negative")):
        cases[f"yaw_{name}"] = [(2, zero), (3, (0, 0, sign*.03)), (3, (0, 0, sign*.1)),
                                (4, (0, 0, sign*.4)), (4, (0, 0, sign*.75)), (3, zero)]
        cases[f"lateral_{name}"] = [(2, zero), (4, (0, sign*.1, 0)), (4, (0, sign*.2, 0)), (3, zero)]
    result = {name: cases[name] for name in ("rolling_reverse", "yaw_positive", "yaw_negative", "lateral_positive", "lateral_negative")}
    return result


def precision_conditions():
    nominal = condition_bank()[0]
    changes = [{}, {"friction": .6, "heading": np.pi/2}, {"friction": 1.2, "heading": -np.pi/2},
               {"added_mass": 1.5, "com": [.015, 0, 0]}, {"kp": .85, "kv": .85},
               {"kp": 1.15, "kv": 1.15}, {"delay": 1}, {"delay": 2}]
    names = ("nominal", "friction_low_heading", "friction_high_heading", "payload", "gain_low", "gain_high", "delay_1", "delay_2")
    return [dict(nominal, **dict(change, id=name)) for name, change in zip(names, changes)]


def rolling_phase_observation(observation, commands):
    """Inference diagnostic only: constant raw clock at zero demand, physical slots intact."""
    from .phase import demand
    observation = observation.clone()
    observation["policy"][:, 56:58] *= (demand(commands) > 0)[:, None]
    return observation


def rollout_precision(env, policy, schedule, sensor=False, rolling_phase_zero=False):
    """Policy-rate tensor capture; terminal rows retained, later rows explicitly invalid."""
    rows, layout = [], {}
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    episode = env.episode_length_buf.clone()
    step = 0
    with torch.no_grad():
        for phase, (seconds, command) in enumerate(schedule):
            for _ in range(round(seconds / env.dt)):
                env.set_commands(command)
                env.compute_observations()
                observation = env.get_observations()
                if rolling_phase_zero:
                    observation = rolling_phase_observation(observation, env.commands)
                recorded_observation = observation["policy"].clone() if sensor else None
                raw = policy(observation)
                finite_policy = torch.isfinite(raw).all(dim=1)
                # A nonfinite proposal is censored before it can enter the solver.
                env.step(torch.where(finite_policy[:, None], raw, torch.zeros_like(raw)))
                state = env.transition_state
                values = {key: state[key] for key in (
                    "commands", "base_pos", "base_quat", "base_lin_vel", "base_ang_vel", "rpy",
                    "dof_pos", "dof_vel", "torques", "actions", "applied_actions", "foot_pos",
                    "foot_contacts", "wheel_normal_force", "loaded_wheels", "wheel_clearance",
                    "wheel_link_quat", "wheel_reposition_velocity_body", "fallen", "time_out_buf",
                    "reset_buf", "episode_length_buf", "nonfoot_contact_count")}
                if "wheel_center_lateral_speed" in state:
                    values["wheel_center_lateral_speed"] = state["wheel_center_lateral_speed"]
                for key in ('undesired_self_pair_count', 'nonfoot_ground_link_count'):
                    if key in state:
                        values[key] = state[key]
                values.update({key: value for key, value in state.items() if key.startswith('straight_')})
                if sensor:
                    values["observations"] = recorded_observation
                if hasattr(env, "active_push_force") and env.cfg.domain_rand.push_robots:
                    values["active_push_force_world_N"] = env.active_push_force.clone()
                if getattr(env, "phase_guided", False):
                    values.update(gait_phase=state["phase"], desired_clearance=state["desired_clearance"],
                                  desired_swing=state["desired_swing"], wheel_thigh_dx=state["wheel_thigh_dx"],
                                  wheel_center_body=state["wheel_center_body"])
                finite = torch.stack([torch.isfinite(v).reshape(env.num_envs, -1).all(1) for v in values.values()]).all(0) & finite_policy
                reset = state["reset_buf"].bool() | (state["episode_length_buf"] != episode + step + 1)
                failed = ~finite | reset | state["fallen"].bool() | (state["nonfoot_contact_count"] > 0)
                values.update(raw_actions=raw, valid=active.clone(), failure=active & failed,
                              nonfinite=~finite, phase=env.commands.new_full((env.num_envs,), phase))
                if not layout:
                    offset = 0
                    for key, value in values.items():
                        width = value[0].numel()
                        layout[key] = (offset, offset + width, tuple(value.shape[1:]))
                        offset += width
                rows.append(torch.cat([v.reshape(env.num_envs, -1).float() for v in values.values()], dim=1))
                active &= ~failed
                step += 1
                if not bool(active.any()):
                    break
            if not bool(active.any()):
                break
    packed = torch.stack(rows).cpu().numpy()
    return {key: packed[:, :, a:b].reshape((len(rows), env.num_envs, *shape))
            for key, (a, b, shape) in layout.items()}


def precision_metrics(data, schedule, dt, metadata, initial):
    from robot_gym.envs.go2w.evaluate import swing_events, velocity_window_metrics
    from robot_gym.utils.diagnostics import wheel_axles_body
    from robot_gym.envs.go2w.go2w_symmetry import joint_reflection

    results = []
    for index in range(data["valid"].shape[1]):
        d = {key: value[:, index][data["valid"][:, index].astype(bool)] for key, value in data.items()}
        n = len(d["phase"])
        if d["nonfinite"].any():
            results.append({"condition_index": index, "recorded_steps": n, "duration_s": n*dt,
                            "failure": True, "nonfinite": True, "phases": [], "swings": {},
                            "reason": "Nonfinite terminal data retained in NPZ; tracking not summarized"})
            continue
        velocity = np.concatenate((d["base_lin_vel"], d["base_ang_vel"]), axis=-1)
        world_velocity = rotate_wxyz(torch.from_numpy(d["base_quat"]), torch.from_numpy(d["base_lin_vel"])).numpy()
        heading = np.unwrap(heading_wxyz(torch.from_numpy(d["base_quat"])).numpy())
        # Conventional body roll/pitch from the authored base, independent of visual meshes.
        body_rpy = Rotation.from_quat(d["base_quat"][:, [1, 2, 3, 0]]).as_euler("xyz")
        inverse = torch.from_numpy(d["base_quat"]).clone(); inverse[:, 1:] *= -1
        feet = rotate_wxyz(inverse[:, None], torch.from_numpy(d["foot_pos"] - d["base_pos"][:, None])).numpy()
        if "wheel_center_body" in d:
            feet = d["wheel_center_body"]  # Already includes the collision-cylinder offset.
        axles = wheel_axles_body(torch.from_numpy(d["base_quat"]), torch.from_numpy(d["wheel_link_quat"]),
                                torch.tensor(metadata["wheel_joint_axes"], dtype=torch.float32, device="cpu")).numpy()
        result = {"condition_index": index, "recorded_steps": n, "duration_s": n*dt,
                  "failure": bool(d["failure"].any()), "falls": int(d["fallen"].sum()),
                  "nonwheel_contact_steps": int((d["nonfoot_contact_count"] > 0).sum()),
                  "timeouts": int(d["time_out_buf"].sum()), "nonfinite": bool(d["nonfinite"].any()),
                  "phases": [], "swings": {}}
        if 'undesired_self_pair_count' in d:
            result['undesired_self_contact_boundaries'] = int((d['undesired_self_pair_count'] > 0).sum())
        offset = 0
        for phase, (seconds, command) in enumerate(schedule):
            count = round(seconds/dt); end = min(offset+count, n)
            if end <= offset:
                result["phases"].append({"phase": phase, "command": command, "censored": True, "samples": 0})
                offset += count
                continue
            sl = slice(offset, end); tail = slice(max(offset, end-round(1/dt)), end)
            start_position = initial["base_pos"][index] if offset == 0 else d["base_pos"][offset-1]
            start_heading = float(heading_wxyz(torch.from_numpy(initial["base_quat"][index]))) if offset == 0 else heading[offset-1]
            path = np.concatenate((start_position[None, :2], d["base_pos"][sl, :2]))
            legs = metadata["leg_indices"]
            error = d["dof_pos"][tail][:, legs] - np.asarray(metadata["nominal_position"])[legs]
            perm, signs = joint_reflection(metadata["joint_order"])
            mirror_error = d["dof_pos"][tail] - d["dof_pos"][tail][:, perm]*signs
            wheels = metadata["wheel_indices"]
            force = d["torques"][sl][:, wheels]
            targets = d["applied_actions"][sl]*np.asarray(metadata["action_scale"])
            targets[:, wheels] = np.clip(targets[:, wheels], -metadata["wheel_target_limit"], metadata["wheel_target_limit"])
            p = {"phase": phase, "command": command, "samples": end-offset,
                 "censored": end < offset+count or bool(d["failure"][sl].any()),
                 "last_second": velocity_window_metrics(velocity[tail], np.asarray(command)),
                 "whole_phase": velocity_window_metrics(velocity[sl], np.asarray(command)),
                 "xy_rms": float(np.sqrt(np.mean(np.sum(velocity[tail, :2]**2, axis=-1)))),
                 "yaw_rms": float(np.sqrt(np.mean(velocity[tail, 5]**2))),
                 "displacement_xy_m": float(np.linalg.norm(path[-1]-path[0])),
                 "path_xy_m": float(np.linalg.norm(np.diff(path, axis=0), axis=-1).sum()),
                 "heading_change_rad": float(heading[end-1]-start_heading),
                 "heading_change_abs_rad": float(abs(heading[end-1]-start_heading)),
                 "leg_error_rms_per_joint": np.sqrt(np.mean(error**2, axis=0)).tolist(),
                 "stance_width_front_rear_m": (feet[tail][:, [0, 2], 1]-feet[tail][:, [1, 3], 1]).mean(0).tolist(),
                 "actual_reference_pose_rms_rad": float(np.sqrt(np.mean(error**2))),
                 "tilt_rms_rad": np.sqrt((d["rpy"][tail, :2]**2).mean(0)).tolist(),
                 "wheel_camber_mean_rad": np.arcsin(np.clip(axles[tail, :, 2], -1, 1)).mean(0).tolist(),
                 "wheel_toe_mean_rad": np.arctan2(-axles[tail, :, 0], axles[tail, :, 1]).mean(0).tolist(),
                 "world_velocity_mean": world_velocity[tail].mean(0).tolist(),
                 "front_rear_pose_rms_rad": [float(np.sqrt(np.mean(error[:, group]**2))) for group in (slice(0,6),slice(6,12))],
                 "front_rear_mirror_rms_rad": [float(np.sqrt(np.mean(mirror_error[:, group]**2))) for group in (legs[:6], legs[6:])],
                 "physical_target_slew_rms_per_joint": np.sqrt(np.mean((np.diff(targets, axis=0)/dt)**2, axis=0)).tolist() if len(targets)>1 else None,
                 "wheel_opposing_force_left_right_mean_Nm": (np.minimum(np.abs(force[:, :2]), np.abs(force[:, 2:]))*(force[:, :2]*force[:, 2:]<0)).mean(0).tolist(),
                 "raw_action_saturation_per_joint": (np.abs(d["raw_actions"][sl]) >= 1).mean(0).tolist(),
                 "force_limit_fraction_per_joint": (np.abs(d["torques"][sl]) >= .99*np.asarray(metadata["force_limits"])).mean(0).tolist(),
                 "max_abs_force_ratio": float(np.max(np.abs(d["torques"][sl])/np.asarray(metadata["force_limits"])))}
            p["last_second"]["mean_error_vx_vy_yaw"] = (velocity[tail][:, [0,1,5]].mean(0)-command).tolist()
            p["body_roll_pitch_mean_rad"] = body_rpy[tail, :2].mean(0).tolist()
            p["body_roll_pitch_rms_rad"] = np.sqrt((body_rpy[tail, :2]**2).mean(0)).tolist()
            p["body_roll_pitch_peak_to_peak_rad"] = np.ptp(body_rpy[sl, :2], axis=0).tolist()
            if "wheel_center_body" in d:
                p["wheelbase_center_mean_m"] = float((feet[tail, :2, 0].mean(1)-feet[tail, 2:, 0].mean(1)).mean())
                p["wheel_pair_midpoints_xy_m"] = ((feet[tail][:, [0, 2], :2]+feet[tail][:, [1, 3], :2])/2).mean(0).tolist()
            if "wheel_thigh_dx" in d:
                p["full_phase_posture"] = {
                    "height_mean_m": float(d["base_pos"][sl, 2].mean()),
                    "height_min_max_m": [float(d["base_pos"][sl, 2].min()), float(d["base_pos"][sl, 2].max())],
                    "roll_pitch_mean_rad": body_rpy[sl, :2].mean(0).tolist(),
                    "roll_pitch_rms_rad": np.sqrt(np.mean(body_rpy[sl, :2]**2, axis=0)).tolist(),
                    "wheel_thigh_dx_mean_m": d["wheel_thigh_dx"][sl].mean(0).tolist(),
                    "wheel_thigh_dx_max_abs_m": np.abs(d["wheel_thigh_dx"][sl]).max(0).tolist(),
                    "wheel_contact_duty": (d["wheel_normal_force"][sl] > 8).mean(0).tolist(),
                    "wheel_peak_clearance_m": d["wheel_clearance"][sl].max(0).tolist(),
                }
            p["counter_command_peak_vx_vy_yaw"] = np.maximum(
                0, -velocity[sl][:, [0, 1, 5]] * np.sign(command)).max(0).tolist()
            if "wheel_center_lateral_speed" in d:
                p["loaded_center_lateral_rms_per_wheel_m_s"] = np.sqrt(np.mean(
                    d["loaded_wheels"][sl] * d["wheel_center_lateral_speed"][sl]**2, axis=0)).tolist()
            p["one_second_windows"] = [velocity_window_metrics(velocity[start:min(start+round(1/dt),end)], np.asarray(command))
                                       for start in range(offset, end, round(1/dt))]
            result["phases"].append(p)
            offset += count
        for limb, name in enumerate(metadata["wheel_order"]):
            events = swing_events(d["wheel_normal_force"][:, limb], d["wheel_clearance"][:, limb], feet[:, limb], dt)
            complete = [e for e in events if e["completed"] and e["geometric_lift"]]
            summary = {"events": events, "completed_geometric_count": len(complete),
                       "rate_hz": len(complete)/(n*dt),
                       "completed_load_only_count": sum(e["completed"] and not e["geometric_lift"] for e in events)}
            for key in ("peak_clearance_m", "horizontal_displacement_body_m", "duration_s"):
                summary[key+"_p50_p90_max"] = np.percentile([e[key] for e in complete], [50,90,100]).tolist() if complete else None
            summary["fraction_ge_2_3_4cm"] = [float(np.mean([e["peak_clearance_m"] >= h for e in complete])) for h in (.02,.03,.04)] if complete else None
            if "event_valid" in d:
                summary["fraction_ge_5cm"] = float(np.mean([e["peak_clearance_m"] >= .05 for e in complete])) if complete else None
                selected = d["event_valid"][:, limb].astype(bool)
                summary["command_qualified"] = {
                    "count": int(selected.sum()),
                    "censored_count": int(d["event_censored_count"][-1, limb]),
                    "events": [{key: float(d["event_" + key][i, limb])
                                for key in ("peak_actual", "peak_use", "duration", "reposition", "quality", "payment", "gate")}
                               for i in np.flatnonzero(selected)],
                }
            result["swings"][name] = summary
        results.append(result)
    return results


def transfer_schedule():
    """Small direct command panel for checkpoints from the continuous fresh run."""
    zero = (0., 0., 0.)
    result = {"stand": [(10, zero)]}
    for name, command in (("forward", (.2, 0, 0)), ("reverse", (-.1, 0, 0)),
                          ("yaw_positive", (0, 0, .4)), ("yaw_negative", (0, 0, -.4)),
                          ("lateral_positive", (0, .1, 0)), ("lateral_negative", (0, -.1, 0)),
                          ("mixed", (.2, .1, .3))):
        result[name] = [(3, zero), (5, command), (6, zero)]
    review_commands = {
        "forward_fast": (.5, 0, 0),
        "lateral_strong_positive": (0, .3, 0), "lateral_strong_negative": (0, -.3, 0),
        "yaw_strong_positive": (0, 0, .8), "yaw_strong_negative": (0, 0, -.8),
    }
    result.update({name: [(3, zero), (5, command), (6, zero)]
                   for name, command in review_commands.items()})
    return result


def sensor_schedule():
    """Long holds, exact stops; no changes to command-conditioned phase dynamics."""
    names = ("forward", "forward_fast", "lateral_strong_positive", "lateral_strong_negative",
             "yaw_strong_positive", "yaw_strong_negative", "mixed")
    short = transfer_schedule()
    return {"stand": [(30, (0., 0., 0.))],
            **{n: [(3, (0., 0., 0.)), (30, short[n][1][1]), (8, (0., 0., 0.))] for n in names}}


def sensor_metrics(data, schedule, dt, metadata):
    """Raw rigid-body motion: world-up is not the sensor optical axis. No derivatives."""
    from .go2w_env import rigid_sensor_state
    from .evaluate import swing_events
    d = {k: v[:, 0] for k, v in data.items()}
    names = ("base", "front_realsense", "radar", "front_realsense_body")
    offsets = torch.tensor([[0., 0., 0.]] + [metadata["sensor_frames"][n]["translation_m"] for n in names[1:]], device="cpu")
    positions, velocities = rigid_sensor_state(*[torch.from_numpy(d[k]) for k in
        ("base_pos", "base_quat", "base_lin_vel", "base_ang_vel")], offsets)
    positions, velocities = positions.numpy(), velocities.numpy()
    rpy = Rotation.from_quat(d["base_quat"][:, [1, 2, 3, 0]]).as_euler("xyz")
    angular_world = rotate_wxyz(torch.from_numpy(d["base_quat"]), torch.from_numpy(d["base_ang_vel"])).numpy()
    hip = [i for i, n in enumerate(metadata["joint_order"]) if "_hip_joint" in n]
    qerror = d["dof_pos"] - np.array(metadata["nominal_position"])
    windows = []
    if len(schedule) == 1:
        windows.append(("full_stand", 0, 30, (0., 0., 0.)))
        onset, command = 0, (0., 0., 0.)
    else:
        onset, command = 3, schedule[1][1]
        windows += [("full_command", 3, 33, command), ("stop", 33, 41, (0., 0., 0.))]
    windows += [(name, onset+a, onset+b, command) for name, a, b in
                (("early", 1, 5), ("middle", 10, 20), ("late", 20, 30))]
    result = {}
    for label, begin, end, command in windows:
        ids = np.flatnonzero((np.arange(len(rpy))*dt >= begin) & (np.arange(len(rpy))*dt < end)
                            & d["valid"].astype(bool))
        if not len(ids):
            result[label] = {"complete": False, "samples": 0}
            continue
        v = np.c_[d["base_lin_vel"][ids, :2], d["base_ang_vel"][ids, 2]]
        error = v - command
        q = qerror[ids]
        count = max(1, round(1/dt))
        loads = d["wheel_normal_force"][ids]
        entry = {"complete": len(ids) == round((end-begin)/dt) and not d["failure"][ids].any(),
                 "samples": len(ids), "window_s": [begin, end], "command": command,
                 "velocity_mean": v.mean(0), "velocity_error_mean": error.mean(0),
                 "velocity_rmse": np.sqrt(np.mean(error**2, axis=0)),
                 "integrated_body_axis_tracking_error": error.sum(0)*dt,
                 "world_displacement_m": d["base_pos"][ids[-1]]-d["base_pos"][ids[0]],
                 "roll_pitch_mean_rad": rpy[ids, :2].mean(0),
                 "roll_pitch_std_rad": rpy[ids, :2].std(0),
                 "roll_pitch_rate_rms_rad_s": np.sqrt(np.mean(d["base_ang_vel"][ids, :2]**2, axis=0)),
                 "rigid_sensor_angular_world_rms_rad_s": np.sqrt(np.mean(angular_world[ids]**2, axis=0)),
                 "hip_signed_error_rad": q[:, hip].mean(0),
                 "hip_error_rms_rad": np.sqrt(np.mean(q[:, hip]**2, axis=0)),
                 "hip_error_abs_p95_rad": np.percentile(np.abs(q[:, hip]), 95, axis=0),
                 "hip_error_drift_rad": q[-count:, hip].mean(0)-q[:count, hip].mean(0),
                 "leg_error_rms_rad": np.sqrt(np.mean(q[:, metadata["leg_indices"]]**2, axis=0)),
                 "wheel_thigh_dx_mean_m": d["wheel_thigh_dx"][ids].mean(0),
                 "wheel_thigh_dx_abs_max_m": np.abs(d["wheel_thigh_dx"][ids]).max(0),
                 "clearance_peak_m": d["wheel_clearance"][ids].max(0),
                 "contact_duty": (loads > 8).mean(0), "load_peak_N": loads.max(0),
                 "all_unloaded_fraction": ((loads > 8).sum(1) == 0).mean(),
                 "nonwheel_contact_samples": int((d["nonfoot_contact_count"][ids] > 0).sum()),
                 "raw_action_clip_fraction": (np.abs(d["raw_actions"][ids]) >= 1).mean(0),
                 "effort_limit_fraction": (np.abs(d["torques"][ids]) >= .99*np.array(metadata["force_limits"])).mean(0),
                 "sensors": {}, "cycles": {}}
        for i, name in enumerate(names):
            z, vz = positions[ids, i, 2], velocities[ids, i, 2]
            time = np.arange(len(ids))*dt
            trend = np.polyfit(time, z, 1) if len(ids)>1 else [0., z[0]]
            residual = z - np.polyval(trend, time)
            harmonics = {}
            for h in (1, 2, 3, 4):
                angle = 2*np.pi*(h/.8)*time
                basis = np.c_[np.sin(angle), np.cos(angle), time, np.ones(len(time))]
                fit = np.linalg.lstsq(basis, z, rcond=None)[0]
                oscillation = basis[:, :2] @ fit[:2]
                harmonics[str(h)] = {"frequency_hz": h/.8, "amplitude_m": float(np.linalg.norm(fit[:2])),
                    "detrended_variance_fraction": float(np.var(oscillation)/max(np.var(residual), 1e-18))}
            entry["sensors"][name] = {"z_mean_m": z.mean(), "z_drift_m": z[-1]-z[0],
                "z_trend_m_s": trend[0], "z_std_m": z.std(), "z_peak_to_peak_m": np.ptp(z),
                "z_detrended_rms_m": np.sqrt(np.mean(residual**2)), "world_vz_mean_m_s": vz.mean(),
                "world_vz_rms_m_s": np.sqrt(np.mean(vz**2)), "world_vz_peak_m_s": np.abs(vz).max(),
                "phase_harmonics": harmonics}
        for i, name in enumerate(metadata["wheel_order"]):
            events = swing_events(loads[:, i], d["wheel_clearance"][ids, i],
                                  d["wheel_center_body"][ids, i], dt)
            complete = [e for e in events if e["completed"] and e["geometric_lift"]]
            entry["cycles"][name] = {"completed_2mm": len(complete),
                "completed_2cm": sum(e["peak_clearance_m"] >= .02 for e in complete)}
        result[label] = entry
    return result


def evaluate_precision(env, policy, out, dr=False, transfer=False, case_names=None,
                       sensor=False, high_rate=False, phase_offset=0., rolling_phase_zero=False,
                       phase_transition=False, high_rate_cases=None):
    import xml.etree.ElementTree as ET

    tree = ET.parse(env.urdf_reader.robot_file_path_absolute).getroot()
    metadata = {"joint_order": env.joint_names, "wheel_order": env.cfg.asset.foot_link_names,
                "leg_indices": env.leg_action_indices, "wheel_indices": env.wheel_action_indices,
                "wheel_joint_axes": [[float(x) for x in tree.find(f"joint[@name='{env.joint_names[i]}']/axis").get("xyz").split()]
                                     for i in env.wheel_action_indices],
                "nominal_position": env.default_dof_pos[0].cpu().tolist(),
                "action_scale": env.action_scale.reshape(-1).cpu().tolist(),
                "force_limits": env.torque_limits.cpu().tolist(),
                "position_limits": [[float(v) if np.isfinite(v) else None for v in pair]
                                    for pair in env.dof_pos_limits.cpu().tolist()],
                "position_limits_semantics": "null denotes an unbounded continuous joint",
                "wheel_target_limit": env.cfg.control.wheel_velocity_target_limit}
    schedules = sensor_schedule() if sensor else transfer_schedule() if transfer else precision_schedule(dr)
    if rolling_phase_zero:
        schedules = {name: schedules[name] for name in ("stand", "forward_fast")}
        schedules["phase_transition"] = [(3, (0., 0., 0.)), (6, (.5, 0., 0.)),
            (5, (0., .3, 0.)), (6, (.5, 0., 0.)), (4, (0., 0., 0.))]
        print("Diagnostic schedule: " + json.dumps(schedules), flush=True)
    if phase_transition:
        schedules = {"phase_transition": [(3, (0., 0., 0.)), (5, (.5, 0., 0.)),
            (3, (0., .03, 0.)), (4, (0., .1, 0.)), (4, (.5, 0., 0.)),
            (3, (0., 0., .175)), (4, (0., 0., -.175)), (3, (0., -.03, 0.)),
            (4, (.2, 0., 0.)), (8, (0., 0., 0.))]}
        print("Phase transition schedule: " + json.dumps(schedules), flush=True)
    if sensor:
        from .go2w_env import fixed_sensor_frames
        metadata["sensor_frames"] = fixed_sensor_frames(env.urdf_reader.robot_file_path_absolute)
    if case_names:
        if not transfer:
            raise ValueError("--transfer_cases requires transfer_screen")
        schedules = {name: schedules[name] for name in case_names}
    report = {"mode": "sensor_sustained" if sensor else "transfer_screen" if transfer else "precision_dr" if dr else "precision_screen", "dt": env.dt, **metadata,
              "schedule": schedules, "tests": {}, "heavy_recorder": bool(high_rate if sensor else transfer and getattr(env, "physics_diagnostics", None) is not None),
              "trace_semantics": "Post-step pre-reset. valid includes first terminal row only; later rows invalid. Policy-rate forces, no substep maxima. Targets = delayed action * saved scale (+ nominal for legs), wheel targets limited as recorded. Stop displacement/path includes braking.",
              "swing_semantics": "6/10 N hysteresis, >=2 mm geometric peak, no dwell filter; completed and boundary-censored events separate",
              "wall_clock_s": {"rollout": 0., "postprocessing": 0.}}
    if rolling_phase_zero:
        report["diagnostic_phase_observation"] = "Diagnostic override: raw slots 56:58 zero at zero demand, unconditional clock at positive demand. Latent clock and reward targets unchanged."
    nominal = loaded_properties(env) if dr else None
    if transfer:
        from robot_gym.utils.export import transfer_contract
        report["transfer_dynamics"] = transfer_contract(env)
        report["physics_recording"] = {
            "enabled": report["heavy_recorder"], "dt_s": env.cfg.sim.dt,
            "semantics": "Post-physics state and held targets; reset warmup excluded; first terminal physics rows retained. Control-force getter recomputes force, not an integration impulse.",
        }
        write_json(out / "applied_properties.json", loaded_properties(env))
    for name, schedule in schedules.items():
        started = perf_counter()
        if sensor:
            env.physics_diagnostics = None
        env.reset()
        if sensor:
            env.phase.add_(phase_offset).remainder_(1.)  # Explicit episode-initial phase comparison only.
            report["reset_phase_offset_cycles"] = phase_offset
            if high_rate and name in (high_rate_cases if high_rate_cases is not None else ("forward_fast", "lateral_strong_positive")):
                from .diagnostics import ReferenceCapture
                env.physics_diagnostics = ReferenceCapture(env)
        if transfer:
            torch.testing.assert_close(env.robot.get_dofs_armature(env.joint_dof_idx), env.armature_samples)
        physics_recorder = getattr(env, "physics_diagnostics", None) if transfer else None
        if physics_recorder is not None:
            physics_recorder.rows.clear()
        if dr:
            conditions = precision_conditions()
            apply_conditions(env, conditions, nominal)
            report["conditions"] = conditions
            applied = loaded_properties(env)
            write_json(out / "applied_properties.json", applied)
            def tensor(values):
                return torch.tensor(values, device=env.device, dtype=torch.float32)
            base = [env.base_link_idx]
            torch.testing.assert_close(applied["mass"][:, base], nominal["mass"][:, base] + tensor([c["added_mass"] for c in conditions])[:, None])
            torch.testing.assert_close(applied["com_local"][:, base], nominal["com_local"][:, base] + tensor([c["com"] for c in conditions])[:, None])
            torch.testing.assert_close(applied["wheel_friction_ratio"], tensor([c["friction"] for c in conditions])[:, None].expand_as(applied["wheel_friction_ratio"]))
            for key, factor, gains in (("kp", "kp", env.base_p_gains), ("kv", "kv", env.base_d_gains)):
                torch.testing.assert_close(applied[key], tensor([c[factor] for c in conditions])[:, None]*gains)
            assert env.action_history.shape[1] >= 3 and not bool(env.action_history.any())
            assert env.action_delay_steps.tolist() == [c["delay"] for c in conditions]
        initial = {key: getattr(env, key).cpu().numpy().copy() for key in ("base_pos", "base_quat")}
        data = rollout_precision(env, policy, schedule, sensor=sensor, rolling_phase_zero=rolling_phase_zero)
        report["wall_clock_s"]["rollout"] += perf_counter()-started
        started = perf_counter()
        physics = ({key: torch.stack([row[key] for row in physics_recorder.rows]).cpu().numpy()
                    for key in physics_recorder.rows[0]} if physics_recorder is not None else {})
        np.savez_compressed(out / f"{name}.npz", **data, **{f"initial_{key}": value for key,value in initial.items()},
                            **{"physics_" + key: value for key,value in physics.items()}, dt=env.dt, physics_dt=env.cfg.sim.dt)
        report["tests"][name] = precision_metrics(data, schedule, env.dt, metadata, initial)
        if sensor and name != "phase_transition":
            report["tests"][name][0]["sensor_windows"] = sensor_metrics(data, schedule, env.dt, metadata)
            report["tests"][name][0]["recording_hz"] = {"policy": 1/env.dt, "physics": 1/env.cfg.sim.dt if physics else None}
            if physics:
                from .go2w_env import rigid_sensor_state
                offsets = torch.tensor([[0., 0., 0.]] + [metadata["sensor_frames"][n]["translation_m"]
                                       for n in ("front_realsense", "radar", "front_realsense_body")], device="cpu")
                _, velocity = rigid_sensor_state(*[torch.from_numpy(physics[k]) for k in
                    ("base_pos", "base_quat", "linear_body", "angular_body")], offsets)
                for window in report["tests"][name][0]["sensor_windows"].values():
                    if not window["samples"]:
                        continue
                    a, b = (round(t/env.cfg.sim.dt) for t in window["window_s"])
                    vz = velocity[a:b, :, 2].numpy()
                    window["physics_200hz"] = {
                        "sensor_order": ["base", "front_realsense", "radar", "front_realsense_body"],
                        "world_vz_rms_m_s": np.sqrt(np.mean(vz**2, axis=0)),
                        "world_vz_abs_peak_m_s": np.abs(vz).max(0),
                        "wheel_load_peak_N": physics["wheel_normal_force"][a:b].max(0),
                        "control_force_abs_peak_Nm": np.abs(physics["control_force"][a:b]).max(0)}
            if name == "forward_fast":
                from tensordict import TensorDict
                observations = torch.from_numpy(data["observations"][200:1650:25, 0]).to(env.device)
                with torch.no_grad():
                    original = policy(TensorDict({"policy": observations}, batch_size=[len(observations)]))
                    reflected = observations.clone()
                    reflected[:, -2:] *= -1
                    other = policy(TensorDict({"policy": reflected}, batch_size=[len(observations)]))
                report["phase_only_action_counterfactual"] = {
                    "definition": "Same recorded states, commands and issued history; only sin/cos negated (half cycle); not a dynamical trial",
                    "raw_action_rms_difference": ((other-original).square().mean(0).sqrt()).cpu().tolist(),
                    "clipped_action_rms_difference": ((other.clamp(-1,1)-original.clamp(-1,1)).square().mean(0).sqrt()).cpu().tolist()}
        if transfer:
            from robot_gym.envs.go2w.diagnostics import spectrum
            for index, result in enumerate(report["tests"][name]):
                for phase in result["phases"]:
                    if not phase["samples"]:
                        continue
                    mask = data["valid"][:, index].astype(bool) & (data["phase"][:, index] == phase["phase"])
                    actions = data["raw_actions"][mask, index]
                    rates = data["base_ang_vel"][mask, index, :2]
                    phase["oscillation"] = {
                        "body_roll_pitch_rate_rms_rad_s": np.sqrt(np.mean(rates**2, axis=0)).tolist(),
                        "joint_velocity_rms_rad_s": np.sqrt(np.mean(data["dof_vel"][mask, index]**2, axis=0)).tolist(),
                        "period_two_action_amplitude": np.abs(np.mean((actions-actions.mean(0)) * (-1.)**np.arange(len(actions))[:, None], axis=0)).tolist(),
                        "action_spectrum": spectrum(actions, 1/env.dt),
                        "body_spectrum": spectrum(rates, 1/env.dt),
                        "minimum_actual_clearance_m": float(data["wheel_clearance"][mask, index].min()),
                    }
                    velocity = data["base_lin_vel"][mask, index]
                    angular = data["base_ang_vel"][mask, index]
                    tail = min(len(velocity), round(2/env.dt))
                    position = data["base_pos"][mask, index]
                    phase["final_two_seconds"] = {
                        "complete": len(velocity) >= round(2/env.dt) and not phase["censored"],
                        "planar_speed_rms_m_s": float(np.sqrt(np.mean(np.sum(velocity[-tail:,:2]**2,axis=-1)))),
                        "yaw_rate_rms_rad_s": float(np.sqrt(np.mean(angular[-tail:,2]**2))),
                        "tracking_rmse_vx_vy_yaw": np.sqrt(np.mean((np.c_[velocity[-tail:,:2], angular[-tail:,2]]-phase["command"])**2,axis=0)).tolist(),
                    }
                    phase["last_five_second_drift_m"] = (float(np.linalg.norm(position[-1,:2]-position[-round(5/env.dt)-1,:2]))
                        if len(position) >= round(5/env.dt)+1 and not phase["censored"] else None)
                    if physics:
                        physical_mask = np.repeat(mask, env.cfg.control.decimation)
                        rates = physics["angular_body"][physical_mask, :2]
                        phase["oscillation"]["physics_rate"] = {
                            "frequency_hz": 1/env.cfg.sim.dt,
                            "body_roll_pitch_rate_rms_rad_s": np.sqrt(np.mean(rates**2,axis=0)).tolist(),
                            "joint_velocity_rms_rad_s": np.sqrt(np.mean(physics["dq"][physical_mask]**2,axis=0)).tolist(),
                            "body_spectrum": spectrum(rates,1/env.cfg.sim.dt),
                            "joint_spectrum": spectrum(physics["dq"][physical_mask],1/env.cfg.sim.dt),
                        }
        report["wall_clock_s"]["postprocessing"] += perf_counter()-started
        write_json(out / "metrics.json", report)
        print(f"{name}: steps={[r['recorded_steps'] for r in report['tests'][name]]}, failures={[r['failure'] for r in report['tests'][name]]}", flush=True)
    return report
