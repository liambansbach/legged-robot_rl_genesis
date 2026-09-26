"""Small explicit reset/DR bank and standing probe; called by evaluate --eval_mode."""

import json
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from robot_gym.utils.diagnostics import (
    loaded_properties,
    write_json,
    nominal_support_heights,
    sha256,
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


def evaluate_brake_restarts(env, policy, out):
    """Two fixed restart traces using the same policy, step and reset paths."""
    report = {}
    old_timeout = env.max_episode_length
    env.max_episode_length = max(old_timeout, round(25 / env.dt))
    try:
        with torch.no_grad():
            for name, positive, negative in (
                ("brake_restart_vx", (0.1, 0, 0), (-0.1, 0, 0)),
                ("brake_restart_yaw", (0, 0, 0.4), (0, 0, -0.4)),
            ):
                env.reset()
                history, dones, commands, phases = {}, [], [], []
                for command in ((0, 0, 0), positive, (0, 0, 0), negative, (0, 0, 0)):
                    count = round((6 if command == (0, 0, 0) else 3) / env.dt)
                    phases.append({"command": command, "start": len(dones), "steps": count})
                    for _ in range(count):
                        env.commands[:] = torch.tensor(command, device=env.device)
                        env.compute_observations()
                        _, _, done, _ = env.step(policy(env.get_observations()))
                        for key, value in env.transition_state.items():
                            history.setdefault(key, []).append(value.clone())
                        dones.append(done.clone())
                        commands.append(command)
                data = {key: torch.stack(values).cpu().numpy() for key, values in history.items()}
                done = torch.stack(dones).cpu().numpy()
                np.savez_compressed(
                    out / f"{name}.npz", **data, done=done,
                    command_stream=np.asarray(commands), dt=env.dt,
                )
                report[name] = {
                    "phases": phases,
                    "episode_timeout_s": env.max_episode_length * env.dt,
                    "fall_count": int(done.sum()),
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
                env.commands[:] = torch.tensor(cmd, device=env.device)
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
