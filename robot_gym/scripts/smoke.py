"""GPU regression of the registered tasks and physical Go2-W randomization."""

import json
import torch
import genesis as gs
from robot_gym.envs import *  # noqa: F401,F403
from robot_gym.utils import get_args, task_registry


def smoke(args):
    cfg, _ = task_registry.get_cfgs(args.task)
    cfg.env.num_envs = args.num_envs or 8
    if args.task == "go2w" and cfg.env.num_envs < 2:
        raise ValueError("Go2-W friction smoke needs at least two environments")
    cfg.sim.performance_mode = False
    cfg.noise.add_noise = False
    cfg.init_state.joint_position_noise = 0.0
    cfg.init_state.joint_velocity_noise = 0.0
    cfg.init_state.orientation_noise = (0.0, 0.0, 0.0)
    cfg.init_state.linear_velocity_noise = cfg.init_state.angular_velocity_noise = 0.0
    # Isolate the wrench test from mass/COM/gain differences. Friction remains randomized.
    cfg.domain_rand.randomize_base_mass = False
    cfg.domain_rand.randomize_com = False
    cfg.domain_rand.randomize_kp = False
    cfg.domain_rand.randomize_kd = False
    cfg.domain_rand.randomize_action_delay = False
    cfg.domain_rand.push_robots = False
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    env.command_resampling_enabled = False
    env.commands.zero_()
    action = torch.zeros((env.num_envs, env.num_actions), device=env.device)
    for _ in range(100):
        obs, reward, done, _ = env.step(action)
        assert obs["policy"].shape == (env.num_envs, cfg.env.num_observations)
        assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
        assert torch.isfinite(env.torques).all()
    report = {
        "task": args.task,
        "observation_shape": list(obs["policy"].shape),
        "standing_height": env.base_pos[:, 2].tolist(),
    }
    if args.task == "go2w":
        assert env.wheel_action_indices == [3, 7, 11, 15]
        assert env.foot_contacts.all(), (
            "All cylinders should contact the flat floor at nominal stand"
        )
        assert env.friction_coefficients.std() > 0.01
        # Contact-pair coefficient is max of the two geometry coefficients after DR ratios.
        geoms = [g for link in env.ankle_links for g in link.geoms]
        ratios = env.sim.rigid_solver.get_geoms_friction_ratio([g.idx for g in geoms])
        coefficients = torch.stack([g.get_friction() for g in geoms])
        ground = env.ground_floor_entity.geoms[0].get_friction()
        effective = torch.maximum(ratios * coefficients, ground)
        torch.testing.assert_close(
            effective, env.friction_coefficients.expand_as(effective)
        )
        report["effective_wheel_friction"] = effective.tolist()
        assert effective.std() > 0.01
        # Verify that a selective reset preserves the environment's actual randomized friction.
        env.reset_idx(torch.tensor([0], device=env.device))
        restored = env.sim.rigid_solver.get_geoms_friction_ratio([g.idx for g in geoms])
        torch.testing.assert_close(restored, ratios)
        for _ in range(100):
            env.step(action)
        baseline = env.base_lin_vel.clone()
        env.push_force[0, 0, 0] = 40.0
        env.push_steps_left[0] = round(0.15 / cfg.sim.dt)
        kp = env.robot.get_dofs_kp(env.joint_dof_idx).clone()
        kv = env.robot.get_dofs_kv(env.joint_dof_idx).clone()
        displacement = env.base_pos.clone()
        for _ in range(8):
            env.step(action)
        dv = env.base_lin_vel[0, 0] - baseline[0, 0]
        report["push_delta_vx"] = float(dv)
        report["push_displacement_x"] = float(env.base_pos[0, 0] - displacement[0, 0])
        assert abs(dv) > 0.01, "Known base wrench must cause measurable motion"
        assert env.push_steps_left[0] == 0
        torch.testing.assert_close(kp, env.robot.get_dofs_kp(env.joint_dof_idx))
        torch.testing.assert_close(kv, env.robot.get_dofs_kv(env.joint_dof_idx))
        # With zero leg targets and zero wheel velocity targets, public forces must still be P/V forces.
        predicted = (
            env.base_p_gains * (env.default_dof_pos - env.dof_pos)
            - env.base_d_gains * env.dof_vel
        )
        predicted[:, env.v_control_mask] = (
            -env.base_d_gains[env.v_control_mask] * env.dof_vel[:, env.v_control_mask]
        )
        torch.testing.assert_close(
            env.torques,
            predicted.clamp(-env.torque_limits, env.torque_limits),
            atol=2e-4,
            rtol=2e-4,
        )
    if args.task == "go2w":
        env.step(torch.full_like(action, 2.0))
        assert env.actions.abs().max() == 1.0
    print(json.dumps(report, indent=2), flush=True)
    gs.destroy()
    print("SMOKE PASS; clean shutdown", flush=True)


if __name__ == "__main__":
    smoke(get_args())
