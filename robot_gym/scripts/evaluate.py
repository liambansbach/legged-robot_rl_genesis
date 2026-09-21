"""Deterministic flat Go2-W command/step/push tests; saves raw traces and JSON metrics."""

import json
from pathlib import Path
import numpy as np
import torch
import genesis as gs
from robot_gym.envs import *  # noqa: F401,F403
from robot_gym.utils import task_registry
from robot_gym.utils.helpers import get_args


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
    ]
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
    }
    with torch.no_grad():
        for name, before, after, push in cases():
            env.reset()
            env.commands[:] = torch.tensor(before, device=env.device)
            env.compute_observations()
            obs = env.get_observations()
            history = []
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
            np.savez_compressed(
                out / f"{name}.npz", trace=data, dt=env.dt, transition_step=args.steps
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
