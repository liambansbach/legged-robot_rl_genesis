"""Deterministic flat Go2-W command/step/push tests; saves raw traces and JSON metrics."""

import json
from pathlib import Path
import numpy as np
import torch
import genesis as gs
from genesis.utils.geom import inv_quat, transform_by_quat
from robot_gym.envs import *  # noqa: F401,F403
from robot_gym.utils import task_registry
from robot_gym.utils.helpers import get_args


POSE_RECOVERY_CASES = {
    "yaw_to_stop",
    "lateral_positive_to_stop",
    "lateral_negative_to_stop",
}
MIRROR_PAIRS = [
    ("yaw_-0.4", "yaw_0.4"),
    ("yaw_-1.0", "yaw_1.0"),
    ("yaw_-1.25", "yaw_1.25"),
    ("vy_-0.1", "vy_0.1"),
    ("vy_-0.25", "vy_0.25"),
    ("arc_0.5_-0.8", "arc_0.5_0.8"),
    ("arc_1.0_-1.0", "arc_1.0_1.0"),
]


def cases():
    fixed = [("stand", (0, 0, 0))]
    fixed += [(f"vx_{x}", (x, 0, 0)) for x in [0.1, 0.5, 1.0, -0.25]]
    fixed += [(f"yaw_{z}", (0, 0, z)) for z in [-1.25, -1.0, -0.4, 0.4, 1.0, 1.25]]
    fixed += [
        (f"arc_{x}_{z}", (x, 0, z))
        for x, z in [(0.5, -0.8), (0.5, 0.8), (1.0, -1.0), (1.0, 1.0), (-0.25, 0.8)]
    ]
    fixed += [(f"vy_{y}", (0, y, 0)) for y in [-0.25, -0.1, 0.1, 0.25]]
    result = [(name, cmd, cmd, False) for name, cmd in fixed]
    result += [
        ("step_forward", (0, 0, 0), (0.8, 0, 0), False),
        ("stop", (0.8, 0, 0), (0, 0, 0), False),
        ("reverse", (0.8, 0, 0), (-0.25, 0, 0), False),
        ("yaw_reverse", (0, 0, 1.0), (0, 0, -1.0), False),
        ("straight_to_arc", (0.8, 0, 0), (0.5, 0, 0.8), False),
        ("precision", (0.8, 0, 0), (0.1, 0, 0), False),
        ("push_stand", (0, 0, 0), (0, 0, 0), True),
        ("push_forward", (0.5, 0, 0), (0.5, 0, 0), True),
        ("yaw_to_stop", (0, 0, 1.0), (0, 0, 0), False),
        ("lateral_positive_to_stop", (0, 0.25, 0), (0, 0, 0), False),
        ("lateral_negative_to_stop", (0, -0.25, 0), (0, 0, 0), False),
    ]
    return result


def mirror_pair_summary(tests):
    """Component-wise mismatch in final-window [vx, vy, yaw]; keep physical units."""
    result = {}
    for minus, plus in MIRROR_PAIRS:
        a = tests[minus]["diagnostics"]["final_window"]
        b = tests[plus]["diagnostics"]["final_window"]
        mismatch = (
            None
            if a is None or b is None
            else dict(
                zip(
                    ("vx_m_s", "vy_m_s", "yaw_rad_s"),
                    np.abs(
                        np.asarray(a["mean_velocity"])
                        - np.asarray(b["mean_velocity"]) * [1, -1, -1]
                    ).tolist(),
                )
            )
        )
        result[f"{minus} / {plus}"] = {
            "minus_mean_velocity": None if a is None else a["mean_velocity"],
            "mirrored_plus_mean_velocity": None
            if b is None
            else (np.asarray(b["mean_velocity"]) * [1, -1, -1]).tolist(),
            "absolute_mirror_mismatch": mismatch,
        }
    return result


def pose_recovery_metrics(
    data, detail, stand_reference, hip_indices, dt, transition_step
):
    """Return to stand's final-window RMS + 0.05 rad, held through the end for >=0.25 s.

    Reference and recovery are per environment. Null means fallen, no valid stand
    reference, or not recovered within the recorded horizon; this is not a reward.
    """
    survivors = ~(data[:, :, 9] > 0).any(axis=0) & stand_reference["survivors"]
    error = detail["leg_position_error"][transition_step:]
    leg = np.sqrt((error**2).mean(axis=2))
    hip = np.sqrt((error[:, :, hip_indices] ** 2).mean(axis=2))
    leg_limit = stand_reference["leg_rms"] + 0.05
    hip_limit = stand_reference["hip_rms"] + 0.05
    inside = (leg <= leg_limit) & (hip <= hip_limit)
    times = []
    for i, valid in enumerate(survivors):
        outside = np.flatnonzero(~inside[:, i])
        start = int(outside[-1] + 1) if len(outside) else 0
        recovered = valid and len(inside) - start >= max(1, int(np.ceil(0.25 / dt)))
        times.append(float((start + 1) * dt) if recovered else None)
    return {
        "reference": "Per-environment final min(1 s, stand duration) RMS error from nominal",
        "tolerance_rad": 0.05,
        "minimum_hold_s": 0.25,
        "leg_rms_threshold_rad": leg_limit.tolist(),
        "hip_rms_threshold_rad": hip_limit.tolist(),
        "eligible_environments": int(survivors.sum()),
        "time_s_per_environment": times,
        "recovered_fraction": sum(t is not None for t in times) / int(survivors.sum())
        if survivors.any()
        else None,
    }


def wheel_mobility_metrics(contacts, wheel_link_height, contact_height):
    """Contact transitions and nominal-height proxy for an explicitly selected window."""
    result = {
        "airborne_wheel_clearance_above_nominal_m": None,
        "airborne_wheel_samples": 0,
        "wheel_contact_transition_count_mean_per_environment": None,
        "clearance_definition": "max(0, wheel_link_z - nominal contact height) when contact flag is false; nominal-height proxy, not tilted-cylinder ground gap; false flags can mean unloading",
    }
    if contacts.shape[1] == 0:
        return result
    clearance = np.maximum(0, wheel_link_height - contact_height)[~contacts]
    result["airborne_wheel_samples"] = int(clearance.size)
    if clearance.size:
        result["airborne_wheel_clearance_above_nominal_m"] = {
            "mean": float(clearance.mean()),
            "p90": float(np.percentile(clearance, 90)),
            "maximum": float(clearance.max()),
        }
    result["wheel_contact_transition_count_mean_per_environment"] = (
        (contacts[1:] != contacts[:-1]).sum(axis=0).mean(axis=0).tolist()
    )
    return result


def lateral_metrics(data, detail, dt, contact_height, initial_world_y=None):
    """Full-case lateral diagnostics from raw traces; exclude any environment that fell.

    Clearance is wheel-link height above the nominal contact height, conditioned
    on the force-based contact flag being false. It is a nominal-height proxy,
    not the ground gap of a tilted cylinder; unloading alone can clear the flag.
    """
    survivors = ~(data[:, :, 9] > 0).any(axis=0)
    n = max(1, round(1.0 / dt))
    windows = {
        "first_second": slice(0, n),
        "second_second": slice(n, 2 * n),
        "final_second": slice(-n, None),
    }
    result = {
        "surviving_environments": int(survivors.sum()),
        "mean_vy_m_s": {},
        "window_duration_s": {},
        "displacement_start_s": 0.0 if initial_world_y is not None else dt,
        "displacement_end_s": len(data) * dt,
        "total_lateral_displacement_world_y_m": None,
    }
    result.update(
        wheel_mobility_metrics(
            detail["foot_contacts"][:, survivors],
            detail["wheel_link_height"][:, survivors],
            contact_height,
        )
    )
    for name, window in windows.items():
        velocity = data[window, survivors, 4]
        result["window_duration_s"][name] = len(velocity) * dt
        result["mean_vy_m_s"][name] = float(velocity.mean()) if velocity.size else None
    if not survivors.any():
        return result
    start_y = data[0, :, 17] if initial_world_y is None else initial_world_y
    result["total_lateral_displacement_world_y_m"] = float(
        (data[-1, survivors, 17] - start_y[survivors]).mean()
    )
    return result


def response_metrics(velocity, target, dt, start):
    """10–90% rise and 63.2% crossing are descriptive, not a fitted dynamics model."""
    result = {}
    for i, axis in enumerate(["vx", "vy", "yaw"]):
        delta = target[i] - start[i]
        if abs(delta) < 0.05:
            continue
        progress = (velocity[:, i] - start[i]) / delta

        def crossing(threshold):
            ids = np.flatnonzero(progress >= threshold)
            return float((ids[0] + 1) * dt) if len(ids) else None

        t10, t90 = crossing(0.1), crossing(0.9)
        tol = max(0.03, 0.1 * abs(delta))
        outside = np.flatnonzero(abs(velocity[:, i] - target[i]) > tol)
        settle = int(outside[-1] + 1) if len(outside) else 0
        result[axis] = {
            "rise_time_10_90_s": None if t10 is None or t90 is None else t90 - t10,
            "settling_time_s": float(settle * dt)
            if settle < len(velocity) - int(0.25 / dt)
            else None,
            "response_63_percent_s": crossing(0.632),
            "overshoot": float(max(0, progress.max() - 1) * abs(delta)),
        }
    return result


def diagnostic_metrics(
    data, detail, target, dt, transition_step, height_target, hip_indices
):
    """Use a fixed final one-second window, and exclude environments that fell in this case."""
    survivors = ~(data[:, :, 9] > 0).any(axis=0)
    window = min(len(data) - transition_step, max(1, round(1.0 / dt)))
    result = {
        "final_window_s": window * dt,
        "surviving_environments": int(survivors.sum()),
    }
    if not survivors.any():
        return {**result, "final_window": None, "post_transition": None}
    final = data[-window:, survivors]
    post = data[transition_step:, survivors]
    leg_error = detail["leg_position_error"][transition_step:, survivors]
    hips = leg_error[:, :, hip_indices]
    contacts = detail["foot_contacts"][transition_step:, survivors]
    ncontacts = contacts.sum(axis=2)
    wheels = detail["wheel_velocities"][transition_step:, survivors]
    actions = detail["wheel_actions"][transition_step:, survivors]
    y = detail["foot_positions_body"][transition_step:, survivors, :, 1]
    result["final_window"] = {
        "mean_velocity": final[:, :, 3:6].mean(axis=(0, 1)).tolist(),
        "velocity_rmse": np.sqrt(
            ((final[:, :, 3:6] - target) ** 2).mean(axis=(0, 1))
        ).tolist(),
        "velocity_std": final[:, :, 3:6].std(axis=(0, 1)).tolist(),
        "last_velocity": final[-1, :, 3:6].mean(axis=0).tolist(),
        "mean_base_height": float(final[:, :, 8].mean()),
        "mean_base_height_error": float(final[:, :, 8].mean() - height_target),
    }
    result["post_transition"] = {
        "leg_position_error_rms": float(np.sqrt((leg_error**2).mean())),
        "hip_abduction_error_rms": float(np.sqrt((hips**2).mean())),
        "hip_abduction_error_max_abs": float(np.abs(hips).max()),
        "wheel_lateral_positions_body_mean": y.mean(axis=(0, 1)).tolist(),
        "left_wheel_lateral_position_mean": float(y[:, :, [0, 2]].mean()),
        "right_wheel_lateral_position_mean": float(y[:, :, [1, 3]].mean()),
        "stance_width_mean": float((y[:, :, [0, 2]] - y[:, :, [1, 3]]).mean()),
        "wheel_contact_fraction": contacts.mean(axis=(0, 1)).tolist(),
        "simultaneous_contacts_count": {
            str(k): int((ncontacts == k).sum()) for k in range(5)
        },
        "simultaneous_contacts_fraction": {
            str(k): float((ncontacts == k).mean()) for k in range(5)
        },
        "any_wheel_airborne_fraction": float((ncontacts < 4).mean()),
        "wheel_action_mean": actions.mean(axis=(0, 1)).tolist(),
        "wheel_velocity_mean": wheels.mean(axis=(0, 1)).tolist(),
        # Positive difference produces positive yaw for the common +Y wheel axes.
        "right_minus_left_wheel_speed": float(
            (wheels[:, :, [1, 3]] - wheels[:, :, [0, 2]]).mean()
        ),
        "leg_velocity_rms": float(np.sqrt(post[:, :, 15].mean())),
    }
    return result


def evaluate(args):
    if args.steps < 2:
        raise ValueError("--steps must be at least 2")
    if args.task != "go2w":
        raise ValueError("This command suite is specific to go2w")
    cfg, train_cfg = task_registry.get_cfgs("go2w")
    cfg.env.num_envs = args.num_envs or 8
    cfg.env.episode_length_s = max(
        20.0, 4 * args.steps * cfg.sim.dt * cfg.control.decimation
    )
    cfg.env.capture_transitions = True
    cfg.sim.performance_mode = False
    cfg.sim.deterministic = True
    cfg.noise.add_noise = False
    cfg.commands.curriculum = False
    for field in [
        "randomize_friction",
        "randomize_base_mass",
        "randomize_com",
        "randomize_kp",
        "randomize_kd",
        "randomize_action_delay",
        "push_robots",
    ]:
        setattr(cfg.domain_rand, field, False)
    cfg.init_state.joint_position_noise = cfg.init_state.joint_velocity_noise = 0.0
    cfg.init_state.orientation_noise = (0.0, 0.0, 0.0)
    cfg.init_state.linear_velocity_noise = cfg.init_state.angular_velocity_noise = 0.0
    env, _ = task_registry.make_env("go2w", args=args, env_cfg=cfg)
    env.command_resampling_enabled = False
    train_cfg.runner.resume = True
    runner, _ = task_registry.make_alg_runner(
        env, args=args, train_cfg=train_cfg, save_config=False
    )
    policy = runner.get_inference_policy(device=env.device)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "dt": env.dt,
        "seed": args.seed if args.seed is not None else train_cfg.seed,
        "checkpoint": str(runner.checkpoint_path),
        "tests": {},
        "base_height_target": cfg.rewards.base_height_target,
        "wheel_order": cfg.asset.foot_link_names,
        "leg_joint_order": [env.joint_names[i] for i in env.leg_action_indices],
        "diagnostic_windows": "Final window: last min(1 second, post-transition duration); not an automatic stability test. Post-transition diagnostics exclude any environment that fell during the case.",
        "foot_position_reference": "Wheel link origins in body frame; contact flag uses configured force threshold, so false may mean unloaded rather than geometrically lifted.",
    }
    hip_indices = [
        env.leg_action_indices.index(i) for i in cfg.asset.hip_abduction_indices
    ]
    stand_reference = None
    with torch.no_grad():
        for name, before, after, push in cases():
            env.reset()
            env.commands[:] = torch.tensor(before, device=env.device)
            env.compute_observations()
            obs = env.get_observations()
            initial_world_y = env.base_pos[:, 1].cpu().numpy().copy()
            history = []
            detail_history = {
                key: []
                for key in (
                    "leg_position_error",
                    "foot_contacts",
                    "wheel_actions",
                    "wheel_velocities",
                    "foot_positions_body",
                    "wheel_link_height",
                    "base_rpy",
                )
            }
            previous_action = torch.zeros_like(env.actions)
            previous_velocity = env.dof_vel.clone()
            for step in range(2 * args.steps):
                cmd = before if step < args.steps else after
                env.commands[:] = torch.tensor(cmd, device=env.device)
                env.compute_observations()
                obs = env.get_observations()
                if push and step == args.steps:
                    env.push_force.zero_()
                    env.push_force[:, :, 1] = 40.0
                    env.push_steps_left[:] = round(0.15 / env.cfg.sim.dt)
                action = policy(obs)
                obs, _, done, _ = env.step(action)
                state = env.transition_state
                velocity = torch.cat(
                    [state["base_lin_vel"][:, :2], state["base_ang_vel"][:, 2:3]], dim=1
                )
                wheels = env.wheel_action_indices
                foot_body = transform_by_quat(
                    (state["foot_pos"] - state["base_pos"][:, None]).reshape(-1, 3),
                    inv_quat(state["base_quat"])[:, None]
                    .expand(-1, 4, -1)
                    .reshape(-1, 4),
                ).reshape(env.num_envs, 4, 3)
                detail = {
                    "leg_position_error": (state["dof_pos"] - env.default_dof_pos)[
                        :, env.leg_action_indices
                    ],
                    "foot_contacts": state["foot_contacts"],
                    "wheel_actions": state["actions"][:, wheels],
                    "wheel_velocities": state["dof_vel"][:, wheels],
                    "foot_positions_body": foot_body,
                    "wheel_link_height": state["foot_pos"][:, :, 2],
                    "base_rpy": state["rpy"],
                }
                for key, value in detail.items():
                    detail_history[key].append(value)
                record = torch.cat(
                    [
                        state["commands"],
                        velocity,
                        state["rpy"][:, :2],
                        state["base_pos"][:, 2:3],
                        done[:, None].float(),
                        state["nonfoot_contact_count"][:, None],
                        (
                            state["dof_vel"][:, wheels].abs()
                            >= env.cfg.control.wheel_velocity_target_limit
                        )
                        .float()
                        .mean(dim=1, keepdim=True),
                        (state["torques"].abs() >= 0.99 * env.torque_limits)
                        .float()
                        .mean(dim=1, keepdim=True),
                        (state["actions"] - previous_action)
                        .square()
                        .mean(dim=1, keepdim=True),
                        (
                            (state["dof_vel"][:, wheels] - previous_velocity[:, wheels])
                            / env.dt
                        )
                        .square()
                        .mean(dim=1, keepdim=True),
                        state["dof_vel"][:, env.leg_action_indices]
                        .square()
                        .mean(dim=1, keepdim=True),
                        state["base_pos"][:, :2],
                        (state["actions"][:, wheels].abs() >= 0.99)
                        .float()
                        .mean(dim=1, keepdim=True),
                    ],
                    dim=1,
                )
                history.append(record.clone())
                previous_action.copy_(state["actions"])
                previous_velocity.copy_(state["dof_vel"])
            data = torch.stack(history).cpu().numpy()
            detail = {
                key: torch.stack(values).cpu().numpy()
                for key, values in detail_history.items()
            }
            np.savez_compressed(
                out / f"{name}.npz",
                trace=data,
                dt=env.dt,
                transition_step=args.steps,
                initial_world_y=initial_world_y,
                **detail,
            )
            # Score only surviving episodes after the transition; retain every pre-reset state in raw traces.
            post = data[args.steps :]
            valid = np.broadcast_to(~(data[:, :, 9] > 0).any(axis=0), post.shape[:2])
            error = post[:, :, 3:6] - np.asarray(after)
            rmse = (
                np.sqrt(np.mean(error[valid] ** 2, axis=0)).tolist()
                if valid.any()
                else None
            )
            fallen = (data[:, :, 9] > 0).any(axis=0)
            metrics = {
                "velocity_rmse": rmse,
                "fall_fraction": float(fallen.mean()),
                "fall_count": int(data[:, :, 9].sum()),
                "roll_pitch_rms": np.sqrt(
                    np.mean(post[:, :, 6:8][valid] ** 2, axis=0)
                ).tolist()
                if valid.any()
                else None,
                "height_variance": float(np.var(post[:, :, 8][valid]))
                if valid.any()
                else None,
                "nonwheel_contact_steps": int((data[:, :, 10] > 0).sum()),
                "wheel_speed_saturation_fraction": float(post[:, :, 11].mean()),
                "torque_saturation_fraction": float(post[:, :, 12].mean()),
                "action_rate_rms": float(np.sqrt(post[:, :, 13].mean())),
                "wheel_acceleration_rms": float(np.sqrt(post[:, :, 14].mean())),
                "leg_velocity_rms": float(np.sqrt(post[:, :, 15].mean())),
                "wheel_action_saturation_fraction": float(post[:, :, 18].mean()),
                "response": response_metrics(
                    post[:, :, 3:6].mean(axis=1),
                    np.asarray(after),
                    env.dt,
                    data[args.steps - 1, :, 3:6].mean(axis=0),
                )
                if before != after and not fallen.any()
                else None,
            }
            metrics["diagnostics"] = diagnostic_metrics(
                data,
                detail,
                np.asarray(after),
                env.dt,
                args.steps,
                cfg.rewards.base_height_target,
                hip_indices,
            )
            if name in {"vy_-0.1", "vy_0.1", "vy_-0.25", "vy_0.25"}:
                metrics["lateral"] = lateral_metrics(
                    data, detail, env.dt, cfg.asset.contact_height, initial_world_y
                )
            if name.startswith("yaw_"):
                metrics["yaw_mobility"] = wheel_mobility_metrics(
                    detail["foot_contacts"][args.steps :, ~fallen],
                    detail["wheel_link_height"][args.steps :, ~fallen],
                    cfg.asset.contact_height,
                )
                metrics["yaw_mobility"]["window"] = (
                    "post_transition; surviving environments only"
                )
            if name == "stand":
                window = min(args.steps, max(1, round(1.0 / env.dt)))
                error = detail["leg_position_error"][-window:]
                stand_reference = {
                    "survivors": ~fallen,
                    "leg_rms": np.sqrt((error**2).mean(axis=(0, 2))),
                    "hip_rms": np.sqrt(
                        (error[:, :, hip_indices] ** 2).mean(axis=(0, 2))
                    ),
                }
            if name in POSE_RECOVERY_CASES:
                metrics["pose_recovery"] = pose_recovery_metrics(
                    data, detail, stand_reference, hip_indices, env.dt, args.steps
                )
            if name == "stop" and not fallen.any():
                speed = np.linalg.norm(post[:, :, 3:5].mean(axis=1), axis=1)
                outside = np.flatnonzero(speed > 0.05)
                stop = int(outside[-1] + 1) if len(outside) else 0
                metrics["stopping_time_s"] = (
                    float(stop * env.dt) if stop < len(speed) else None
                )
                metrics["stopping_distance_m"] = float(
                    np.linalg.norm(
                        np.diff(
                            data[
                                args.steps - 1 : args.steps + min(stop + 1, len(speed)),
                                :,
                                16:18,
                            ],
                            axis=0,
                        ),
                        axis=2,
                    )
                    .sum(axis=0)
                    .mean()
                )
            report["tests"][name] = metrics
            print(name, metrics, flush=True)
    report["mirror_pairs"] = mirror_pair_summary(report["tests"])
    print("mirror_pairs", report["mirror_pairs"], flush=True)
    report["trace_columns"] = [
        "cmd_vx",
        "cmd_vy",
        "cmd_yaw",
        "vx",
        "vy",
        "yaw_rate",
        "roll",
        "pitch",
        "height",
        "done",
        "nonwheel_contacts",
        "wheel_speed_saturation",
        "torque_saturation",
        "action_delta_squared",
        "wheel_acceleration_squared",
        "leg_velocity_squared",
        "world_x",
        "world_y",
        "wheel_action_saturation",
    ]
    (out / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    gs.destroy()


if __name__ == "__main__":
    evaluate(get_args())
