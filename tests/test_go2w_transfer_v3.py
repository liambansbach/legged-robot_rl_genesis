"""Focused CPU checks for the explicit phase-guided interface and objective."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import yaml
from tensordict import TensorDict

from robot_gym.envs.base.legged_robot import LeggedRobot
from robot_gym.envs.go2w.go2w_config import apply_go2w_profile, configure, v3_joint_reference, validate_fresh_transfer
from robot_gym.envs.go2w.go2w_env import Go2WEnv
from robot_gym.envs.go2w.go2w_symmetry import mirror_observations
from robot_gym.envs.go2w.phase import targets, clearance_error, support_error, corridor_error
from robot_gym.utils import task_registry, get_args
from robot_gym.utils.helpers import class_to_dict


def recipe():
    cfg, train = task_registry.get_cfgs("go2w")
    apply_go2w_profile(cfg, train, "transfer_v3")
    return cfg, train


class TransferV3(unittest.TestCase):
    def test_fresh_recipe_and_legacy_reference(self):
        cfg, train = recipe()
        self.assertEqual((cfg.env.num_observations, cfg.env.num_actions, cfg.env.num_privileged_obs), (58, 16, None))
        self.assertEqual(cfg.init_state.default_joint_angles, v3_joint_reference())
        self.assertAlmostEqual(cfg.init_state.pos[2] - cfg.rewards.base_height_target, .003)
        self.assertEqual(cfg.go2w_behavior, "phase_guided")
        self.assertEqual(train.actor.hidden_dims, [512, 256, 128])
        self.assertEqual(train.actor.distribution_cfg["init_std"], .4)
        self.assertEqual(train.runner.num_steps_per_env, 64)
        self.assertFalse(train.runner.resume)
        self.assertIsNone(train.runner.load_run)
        validate_fresh_transfer(SimpleNamespace(), train)
        for flag, value in (("resume", True), ("checkpoint", -1), ("load_run", "anything")):
            with self.assertRaises(ValueError):
                validate_fresh_transfer(SimpleNamespace(**{flag: value}), train)
        legacy, oldtrain = task_registry.get_cfgs("go2w")
        apply_go2w_profile(legacy, oldtrain, "transfer_v2")
        self.assertEqual(legacy.env.num_observations, 56)
        self.assertEqual(legacy.init_state.default_joint_angles["RL_thigh_joint"], .75)
        self.assertEqual(legacy.rewards.base_height_target, .415)
        self.assertEqual(cfg.rewards.scales.step_event, 0)
        self.assertEqual(getattr(cfg.rewards.scales, "wheel_swing", 0), 0)
        with self.assertRaisesRegex(ValueError, "Huber"):
            configure(cfg, train, SimpleNamespace(task="go2w", go2w_profile="transfer_v3", tracking_sigma_x=.25))

    def test_phase_mirror_and_double_support(self):
        p = torch.arange(80) / 80
        command = torch.tensor([[0., .3, .8]]).expand(80, -1)
        offsets = torch.tensor([0., .5, .5, 0.])
        swing, height = targets(p, command, offsets, .65, .04)
        reflected = command * torch.tensor([1., -1., -1.])
        other, other_h = targets((p + .5) % 1, reflected, offsets, .65, .04)
        torch.testing.assert_close(other, swing[:, [1, 0, 3, 2]], atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(other_h, height[:, [1, 0, 3, 2]], atol=1e-6, rtol=1e-5)
        self.assertTrue(((swing > 1e-6).sum(1) <= 2).all())
        self.assertTrue(((swing == 0).sum(1) == 4).any())
        _, stop = targets(p, torch.zeros_like(command), offsets, .65, .04)
        self.assertEqual(stop.count_nonzero(), 0)
        names = list(v3_joint_reference())
        env = SimpleNamespace(num_obs=58, num_actions=16, joint_names=names,
                              leg_action_indices=[i for i, n in enumerate(names) if not n.endswith("foot_joint")])
        obs = TensorDict({"policy": torch.randn(80, 58)}, batch_size=[80])
        mirrored = mirror_observations(env, obs)
        torch.testing.assert_close(mirrored["policy"][:, 56:], -obs["policy"][:, 56:])
        torch.testing.assert_close(mirror_observations(env, mirrored)["policy"], obs["policy"])

    def test_loaded_swing_gets_continuous_feedback_and_corridor(self):
        desired = torch.tensor([[.04, 0., 0., .04]])
        swing = desired / .04
        ground = clearance_error(torch.zeros_like(desired), desired, .04)
        lifting = clearance_error(desired * .5, desired, .04)
        self.assertGreater(float(ground), float(lifting))
        self.assertGreater(float(lifting), 0)
        full_load = torch.full_like(desired, 50.)
        less_load = full_load * (1 - .5 * swing)
        self.assertGreater(float(support_error(full_load, swing, 50)), float(support_error(less_load, swing, 50)))
        reference = torch.full_like(desired, .0086)
        opposed = reference + torch.tensor([[.12, .12, -.12, -.12]])
        rolling = corridor_error(opposed, reference, torch.zeros_like(swing), .04, .09, .05)
        moving = corridor_error(opposed, reference, torch.ones_like(swing), .04, .09, .05)
        self.assertGreater(float(rolling), float(moving))
        self.assertGreater(float(moving), 0)
        # The corridor has only an x input; y spread and z lift cannot enter it.
        self.assertEqual(float(corridor_error(reference, reference, swing, .04, .09, .05)), 0)
        # At a static strong lateral demand, tracking minus peak lift/load costs
        # remains positive (.8/s), while a fall costs -5 discretely.
        self.assertGreater(2.8 - 1.5 - float(ground) - .5 * float(support_error(full_load, swing, 50)), 0)

    def test_phase_boundary_reset_and_observation(self):
        cfg, _ = recipe()
        env = Go2WEnv.__new__(Go2WEnv)
        env.cfg, env.phase_guided, env.event_step = cfg, True, False
        env.num_envs, env.device, env.dt = 2, "cpu", .02
        env.phase = torch.tensor([.2, .7])
        env.phase_offsets = torch.tensor([0., .5, .5, 0.])
        env.commands = torch.tensor([[0., .3, 0.], [0., -.3, 0.]])
        env.base_lin_vel = env.base_ang_vel = env.projected_gravity = torch.zeros(2, 3)
        env.default_dof_pos = env.dof_pos = env.dof_vel = env.actions = torch.zeros(2, 16)
        env.leg_action_indices = [i for i in range(16) if i % 4 != 3]
        env.obs_scales = SimpleNamespace(lin_vel=2., ang_vel=.25, dof_pos=1., dof_vel=.05)
        env.commands_scale = torch.ones(3)
        env.add_noise = False
        env.compute_observations()
        before = env.obs_buf.clone()
        env.compute_observations()
        torch.testing.assert_close(before, env.obs_buf)
        env.update_task_state()
        with patch.object(LeggedRobot, "_post_physics_step_callback"):
            env._post_physics_step_callback()
        torch.testing.assert_close(env.phase, torch.tensor([.225, .725]))
        env.wheel_geometry_enabled = False
        cfg.phase_guidance["randomize_reset"] = False
        with patch.object(LeggedRobot, "reset_idx"):
            env.reset_idx(torch.tensor([0]))
        torch.testing.assert_close(env.phase, torch.tensor([0., .725]))
        env.commands.zero_()
        env.update_task_state()
        self.assertEqual(env.desired_clearance.count_nonzero(), 0)
        old = env.phase.clone()
        env.compute_observations()
        base = env.obs_buf.clone()
        env.base_lin_vel = torch.ones(2, 3)
        env.compute_observations()
        torch.testing.assert_close(env.obs_buf[:, :3] - base[:, :3], torch.full((2, 3), 2.))
        torch.testing.assert_close(env.obs_buf[:, 3:], base[:, 3:])
        torch.testing.assert_close(env.phase, old)

    def test_signed_reward_scaling(self):
        cfg, _ = recipe()
        env = Go2WEnv.__new__(Go2WEnv)
        env.cfg, env.dt, env.num_envs, env.device = cfg, .02, 1, "cpu"
        env.reward_scales = class_to_dict(cfg.rewards.scales)
        env._prepare_reward_function()
        self.assertEqual(env.reward_scales["phase_clearance"], -.02)
        self.assertEqual(env.reward_scales["termination"], -5)
        self.assertNotIn("step_event", env.reward_names)
        env.event_step = False
        env.reward_names = ["phase_clearance"]
        env.reward_functions = [lambda: torch.tensor([100.])]
        env.rew_buf = torch.zeros(1)
        env.reset_buf, env.time_out_buf = torch.tensor([True]), torch.tensor([False])
        env.compute_reward()
        self.assertAlmostEqual(float(env.rew_buf[0]), -7.)
        env.time_out_buf[:] = True
        env.compute_reward()
        self.assertAlmostEqual(float(env.rew_buf[0]), -2.)

    def test_imported_origin_center_transport_in_base_axes(self):
        from robot_gym.utils.diagnostics import rotate_wxyz
        env = Go2WEnv.__new__(Go2WEnv)
        cfg, _ = recipe()
        env.cfg, env.wheel_geometry_enabled, env.phase_guided = cfg, True, True
        env.event_step, env.transfer_v2 = False, False
        env.foot_link_indices_local, env.thigh_link_indices_local = [0, 1, 2, 3], [4, 5, 6, 7]
        env.wheel_clearance = env.wheel_center_lateral_speed = env.wheel_thigh_dx = torch.zeros(1, 4)
        env.wheel_reposition_velocity_body = torch.zeros(1, 4, 3)
        env.foot_lin_vel = torch.zeros(1, 4, 3)
        env.base_lin_vel = env.base_ang_vel = torch.zeros(1, 3)
        env.base_pos = torch.tensor([[2., -1., .8]])
        offset = torch.tensor([[.02, .03, -.01]]).expand(4, -1)
        axis = torch.tensor([[0., 1., 0.]]).expand(4, -1)
        env.wheel_geometry = (offset, axis, torch.full((4,), .09167), torch.full((4,), .025))
        center_body = torch.tensor([[[.3, .2, -.3], [.3, -.2, -.3], [-.1, .2, -.3], [-.1, -.2, -.3]]])
        thigh_body = center_body - torch.tensor([.1, .04, -.2])
        for quat in ([1., 0., 0., 0.], [.5, .5, .5, .5]):
            env.base_quat = torch.tensor([quat])
            wheel_quat = env.base_quat[:, None].expand(-1, 4, -1)
            env.foot_pos = env.base_pos[:, None] + rotate_wxyz(wheel_quat, center_body - offset)
            thigh = env.base_pos[:, None] + rotate_wxyz(wheel_quat, thigh_body)
            env.robot = SimpleNamespace(get_links_quat=lambda *a: wheel_quat,
                                        get_links_ang=lambda *a: torch.zeros(1, 4, 3),
                                        get_links_pos=lambda *a, **kw: thigh)
            with patch.object(LeggedRobot, "_update_robot_state"):
                env._update_robot_state()
            torch.testing.assert_close(env.wheel_thigh_dx, torch.full((1, 4), .1), atol=3e-7, rtol=1e-5)

    def test_saved_v3_replay_interface(self):
        cfg, train = recipe()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "model_100.pt").touch()
            config = {"task": "go2w", "env_cfg": class_to_dict(cfg), "train_cfg": class_to_dict(train)}
            (path / "config.yaml").write_text(yaml.safe_dump(config))
            argv = ["play", "--task", "go2w", "--load_run", str(path), "--checkpoint", "-1"]
            with patch.object(sys, "argv", argv):
                args = get_args()
            restored, _, checkpoint = task_registry.resolve_replay(args)
            self.assertEqual(restored.env.num_observations, 58)
            self.assertEqual(checkpoint.name, "model_100.pt")
            config["env_cfg"]["env"]["num_observations"] = 56
            (path / "config.yaml").write_text(yaml.safe_dump(config))
            with patch.object(sys, "argv", argv), self.assertRaises(ValueError):
                task_registry.resolve_replay(get_args())


if __name__ == "__main__":
    unittest.main()
