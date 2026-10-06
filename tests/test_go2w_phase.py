"""Focused CPU checks for the explicit phase-guided interface and objective."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from tensordict import TensorDict

from robot_gym.envs.base.legged_robot import LeggedRobot
from robot_gym.envs.go2w.go2w_config import joint_reference
from robot_gym.envs.go2w.go2w_env import Go2WEnv
from robot_gym.envs.go2w.go2w_symmetry import mirror_observations
from robot_gym.envs.go2w.phase import targets, clearance_error, support_error, corridor_error
from robot_gym.utils import task_registry
from robot_gym.utils.helpers import class_to_dict


def registered_config():
    return task_registry.get_cfgs('go2w')


class PhaseTests(unittest.TestCase):

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
        names = list(joint_reference())
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
        cfg, _ = registered_config()
        env = Go2WEnv.__new__(Go2WEnv)
        env.cfg = cfg
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
        from robot_gym.envs.go2w.phase import targets
        env.desired_swing, env.desired_clearance = targets(env.phase, env.commands, env.phase_offsets,
            cfg.phase_guidance['stance_fraction'], cfg.phase_guidance['apex_m'])
        with patch.object(LeggedRobot, "_post_physics_step_callback"):
            env._post_physics_step_callback()
        torch.testing.assert_close(env.phase, torch.tensor([.225, .725]))
        env.wheel_clearance = env.wheel_normal_force = torch.zeros(2, 4)
        env.loaded_wheels = torch.zeros(2, 4, dtype=torch.bool)
        env.wheel_reposition_velocity_body = torch.zeros(2, 4, 3)
        env.wheel_center_lateral_speed = torch.zeros(2, 4)
        cfg.phase_guidance["randomize_reset"] = False
        with patch.object(LeggedRobot, "reset_idx"):
            env.reset_idx(torch.tensor([0]))
        torch.testing.assert_close(env.phase, torch.tensor([0., .725]))
        env.commands.zero_()
        from robot_gym.envs.go2w.phase import targets
        env.desired_swing, env.desired_clearance = targets(env.phase, env.commands, env.phase_offsets,
            cfg.phase_guidance['stance_fraction'], cfg.phase_guidance['apex_m'])
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
        cfg, _ = registered_config()
        env = Go2WEnv.__new__(Go2WEnv)
        env.cfg, env.dt, env.num_envs, env.device = cfg, .02, 1, "cpu"
        env.reward_scales = class_to_dict(cfg.rewards.scales)
        env._prepare_reward_function()
        self.assertEqual(env.reward_scales["phase_clearance"], -.02)
        self.assertEqual(env.reward_scales["termination"], -5)
        self.assertNotIn("step_event", env.reward_names)
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
        cfg, _ = registered_config()
        env.cfg = cfg
        env.sensor_offsets = torch.zeros(2, 3)
        env.sensor_vz_squared = torch.zeros(1)
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



if __name__ == "__main__":
    unittest.main()
