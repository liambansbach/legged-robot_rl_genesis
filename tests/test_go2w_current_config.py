"""Focused configuration, action, sampling and push checks without a simulator."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import torch

from robot_gym.envs.base.legged_robot import LeggedRobot
from robot_gym.envs.go2w.go2w_config import (
    validate_training, joint_reference,
    reference_geometry, SPAWN_CLEARANCE, restore_saved_config,
)
from robot_gym.envs.go2w.go2w_env import Go2WEnv, horizontal_push_force
from robot_gym.utils import get_args, task_registry
from robot_gym.utils.helpers import class_to_dict, update_cfg_from_args

AXES = ('lin_vel_x', 'lin_vel_y', 'ang_vel_yaw')


def resolved():
    with patch.object(sys, 'argv', ['train', '--task', 'go2w']):
        args = get_args()
    cfg, train = task_registry.get_cfgs('go2w')
    with patch('torch.load', side_effect=AssertionError('No checkpoint loading')):
        update_cfg_from_args(cfg, train, args)
        validate_training(args, cfg, train)
    return cfg, train, args


def sampler_fixture(n):
    cfg, _, _ = resolved()
    env = Go2WEnv.__new__(Go2WEnv)
    env.cfg, env.device, env.dt, env.num_envs = cfg, 'cpu', .02, n
    env.fixed_command = None
    env.commands = torch.zeros(n, 3)
    env.command_steps_left = torch.zeros(n, dtype=torch.long)
    env.command_hold_kind = torch.zeros_like(env.command_steps_left)
    env.command_sampling_tier = torch.zeros_like(env.command_steps_left)
    env.diagnostic_command_families = torch.zeros_like(env.command_steps_left)
    env.command_ranges = class_to_dict(cfg.commands.ranges)
    env.command_mixture = torch.tensor([cfg.commands.stand_command_probability,
                                      *cfg.commands.moving_mixture_probabilities])
    return env


class CurrentConfig(unittest.TestCase):
    def test_current_reward_list_and_removed_training_selectors(self):
        cfg, _, args = resolved()
        active = class_to_dict(cfg.rewards.scales)
        expected = {'tracking_x', 'tracking_y', 'tracking_yaw', 'reference_pose', 'reference_height',
            'orientation', 'phase_clearance', 'phase_support', 'wheel_corridor', 'insufficient_support',
            'sensor_vertical_velocity', 'ang_vel_xy', 'wheel_rate_zero', 'contact_safety',
            'normalized_effort', 'leg_action_rate', 'wheel_action_rate', 'lateral_wheel_scrub',
            'dof_pos_limits', 'torque_limits', 'termination'}
        self.assertEqual(set(active), expected)
        self.assertTrue(all(active.values()))
        self.assertFalse(any(key.startswith('go2w_') for key in vars(args)))
        self.assertNotIn('_control_dofs', Go2WEnv.__dict__)  # Shared P/V target contract.
        self.assertNotIn('step', Go2WEnv.__dict__)  # No task-specific action override.

    def test_config_copies_do_not_share_mutable_parameters(self):
        first, training, _ = resolved()
        second, other, _ = resolved()
        first.init_state.default_joint_angles['FL_calf_joint'] = 0.
        first.commands.sampling['tier_probabilities'][0] = 0.
        training.actor.distribution_cfg['init_std'] = 1.
        self.assertEqual(second.init_state.default_joint_angles, joint_reference())
        self.assertEqual(second.commands.sampling['tier_probabilities'], [.8, .1, .1])
        self.assertEqual(other.actor.distribution_cfg['init_std'], .4)

    def test_saved_config_is_authoritative_and_missing_fields_fail(self):
        import copy
        cfg, train, _ = resolved()
        saved = dict(env_cfg=class_to_dict(cfg), train_cfg=class_to_dict(train))
        saved['env_cfg']['rewards']['scales']['orientation'] = -3.125
        restored, training = restore_saved_config(saved)
        self.assertEqual(class_to_dict(restored), saved['env_cfg'])
        self.assertEqual(class_to_dict(training), saved['train_cfg'])
        # The prepared config used identical mechanisms with older structural names.
        prepared = copy.deepcopy(saved)
        e = prepared['env_cfg']
        del e['config_version']
        e.update(go2w_recipe='reference_tracking_v1', go2w_profile='transfer_v3')
        e['commands']['reference_tracking_sampling'] = e['commands'].pop('sampling')
        e['sensor_smooth'] = e['commands'].pop('holds')
        e['commands']['moving_long_duration_range'] = e['sensor_smooth']['long_hold_s'].copy()
        restored, _ = restore_saved_config(prepared)
        self.assertEqual(class_to_dict(restored), saved['env_cfg'])
        e['commands']['moving_long_duration_range'][0] += 1.
        with self.assertRaisesRegex(ValueError, 'command-duration contract'):
            restore_saved_config(prepared)
        incomplete = copy.deepcopy(saved)
        del incomplete['env_cfg']['control']['armature']
        with self.assertRaisesRegex(ValueError, 'missing env_cfg.control.armature'):
            restore_saved_config(incomplete)
        unsupported = copy.deepcopy(saved)
        del unsupported['env_cfg']['config_version']
        with self.assertRaisesRegex(ValueError, 'historical source checkout'):
            restore_saved_config(unsupported)
        with self.assertRaisesRegex(ValueError, 'reward/phase contract'):
            unsupported = copy.deepcopy(saved)
            unsupported['env_cfg']['rewards']['scales']['obsolete_objective'] = 1.
            restore_saved_config(unsupported)

    def test_canonical_reference_height(self):
        expected = {f'{leg}_{joint}_joint': value for leg in ('FL', 'FR', 'RL', 'RR')
                    for joint, value in zip(('hip', 'thigh', 'calf', 'foot'), (0., .7, -1.4, 0.))}
        self.assertEqual(joint_reference(), expected)
        height, dx = reference_geometry()
        self.assertAlmostEqual(height, .4277416561558192, places=12)
        cfg, train, args = resolved()
        self.assertEqual(cfg.init_state.default_joint_angles, expected)
        self.assertAlmostEqual(cfg.init_state.pos[2], height + SPAWN_CLEARANCE, places=12)
        self.assertEqual(cfg.rewards.base_height_target, height)
        self.assertEqual(cfg.rewards.phase_objective['wheel_thigh_dx_reference_m'], dx)
        cfg.init_state.default_joint_angles['RR_calf_joint'] = -1.31
        with self.assertRaisesRegex(ValueError, 'canonical'):
            validate_training(args, cfg, train)

    def test_clipped_delayed_fixed_reference_offsets_and_reset_observation(self):
        env = sampler_fixture(3)
        cfg = env.cfg
        names = cfg.asset.joint_names
        env.num_dof = 16
        env.default_dof_pos = torch.tensor([[cfg.init_state.default_joint_angles[n] for n in names]])
        env.action_scale = torch.tensor([cfg.control.action_scale[n] for n in names])
        torch.testing.assert_close(env.action_scale, torch.tensor([.3, .35, .4, 18.] * 4))
        self.assertEqual(cfg.normalization.clip_actions, 1.)
        env.p_control_mask = torch.tensor([cfg.control.control_type[n] == 'P' for n in names])
        env.v_control_mask = ~env.p_control_mask
        env.position_control_mask = env.p_control_mask.float()
        env.p_control_dof_idx = env.leg_action_indices = env.p_control_mask.nonzero().flatten().tolist()
        env.v_control_dof_idx = env.v_control_mask.nonzero().flatten().tolist()
        env.t_control_dof_idx = []
        env.joint_dof_idx = list(range(16))
        env.dof_vel_limits = torch.full((16,), 30.1)
        env.dof_pos_limits = torch.tensor([[-10., 10.]] * 16)
        env.dof_pos = torch.zeros(3, 16)
        env.dof_vel = torch.zeros_like(env.dof_pos)
        env.robot = Mock()
        env._randomize_pd_gains = Mock()
        with patch('robot_gym.envs.base.legged_robot.gs_rand_float',
                   side_effect=lambda low, high, shape, device: torch.full(shape, .02)):
            env._reset_dofs(torch.arange(3))
        torch.testing.assert_close(env.dof_pos, env.default_dof_pos.expand(3, -1) + .02 * env.position_control_mask)
        nominal = env.default_dof_pos.clone()
        env.action_history = torch.zeros(3, 3, 16)
        env.action_delay_steps = torch.tensor([0, 1, 2])
        env.all_env_ids = torch.arange(3)
        env.sim = Mock()
        env._apply_pushes = Mock()
        cfg.domain_rand.push_robots = False
        env.post_physics_step = Mock()
        env.obs_buf = torch.zeros(3, 58)
        env.privileged_obs_buf = None
        env.rew_buf, env.reset_buf, env.extras = torch.zeros(3), torch.zeros(3), {}
        env.get_observations = lambda: env.obs_buf
        for raw, expected in ((2., [1., 0., 0.]), (-2., [-1., 1., 0.]), (0., [0., -1., 1.]),
                              (0., [0., 0., -1.])):
            env.step(torch.full((3, 16), raw))
            applied = torch.tensor(expected)[:, None]
            torch.testing.assert_close(env.robot.control_dofs_position.call_args.args[0],
                nominal[:, env.p_control_mask] + applied * env.action_scale[env.p_control_mask])
            torch.testing.assert_close(env.robot.control_dofs_velocity.call_args.args[0],
                applied.expand(-1, 4) * 18.)
            torch.testing.assert_close(env.default_dof_pos, nominal)
        # Observations and the pose reward use that same nominal, not noisy reset positions.
        env.base_lin_vel = env.base_ang_vel = env.projected_gravity = torch.zeros(3, 3)
        env.commands_scale = torch.ones(3)
        env.obs_scales = cfg.normalization.obs_scales
        env.phase = torch.zeros(3)
        env.add_noise = False
        env.reference_pose_scales = torch.tensor([.08, .12, .12] * 4)
        env.compute_observations()
        self.assertEqual(env.obs_buf.shape, (3, 58))
        torch.testing.assert_close(env.obs_buf[:, 12:24], torch.full((3, 12), .02))
        self.assertTrue((env._reward_reference_pose() > 0).all())
        env.dof_pos[:] = nominal
        torch.testing.assert_close(env._reward_reference_pose(), torch.zeros(3))

    def test_all_family_overrides_and_core_edge_reserve(self):
        families = torch.arange(7).repeat_interleave(64)
        env = sampler_fixture(len(families))
        core = torch.tensor([env.command_ranges[n] for n in AXES])
        reserve = torch.tensor([env.cfg.commands.sampling['reserve_ranges'][n] for n in AXES])
        self.assertEqual(core.tolist(), torch.tensor([[-.3, 1.], [-.3, .3], [-1., 1.]]).tolist())
        for kind in (0, 1, 2):
            torch.manual_seed(310 + kind)
            with patch('torch.multinomial', side_effect=[families, torch.full_like(families, kind)]):
                env._resample_commands(torch.arange(len(families)))
            c, f = env.commands, families
            self.assertTrue(((c >= reserve[:, 0]) & (c <= reserve[:, 1])).all())
            outside = (c < core[:, 0]) | (c > core[:, 1])
            ordinary = (f != 0) & (f != 4)
            self.assertTrue((outside.sum(1) <= 1).all())
            self.assertTrue((outside.sum(1)[ordinary] == (1 if kind == 2 else 0)).all())
            self.assertTrue((c[f == 0] == 0).all())
            self.assertTrue((c[f == 1, 1:] == 0).all())
            self.assertTrue((c[f == 2, 1] == 0).all())
            self.assertTrue((c[f == 3, :2] == 0).all())
            self.assertTrue((c[f == 5][:, [0, 2]] == 0).all())
            self.assertTrue((c[f == 4, 0].abs() <= .108).all())
            self.assertTrue((c[f == 4, 2].abs() <= .16).all())
            self.assertTrue((c[f == 4, 1] == 0).all())
            for family, axis in ((1, 0), (3, 2), (5, 1)):
                values = c[f == family, axis]
                self.assertTrue((values > 0).any() and (values < 0).any())
                if kind == 1:
                    self.assertTrue((values == core[axis, 0]).any() and (values == core[axis, 1]).any())
            if kind == 0:
                self.assertTrue((c[f == 1, 0] > .6).any())
                self.assertTrue((c[f == 3, 2].abs() < .15).any())
                self.assertTrue((c[f == 3, 2].abs() > .8).any())
                self.assertTrue((c[f == 5, 1].abs() > .25).any())
                self.assertTrue((c[f == 6, 0] > .6).any())
                self.assertTrue((c[f == 6, 1].abs() > .2).any())
                self.assertTrue((c[f == 6, 2].abs() > .5).any())

    def test_small_commands_survive_actual_deadzone(self):
        env = sampler_fixture(6)
        c = torch.tensor([[.02, 0, 0], [-.02, 0, 0], [0, .02, 0], [0, -.02, 0],
                          [0, 0, .05], [0, 0, -.05]])
        with patch('torch.multinomial', return_value=torch.tensor([1, 1, 5, 5, 3, 3])), \
             patch('robot_gym.envs.go2w.go2w_env.navigation_commands', return_value=(c.clone(), torch.zeros(6, dtype=torch.long))):
            env._resample_commands(torch.arange(6))
        torch.testing.assert_close(env.commands, c, rtol=0, atol=0)

    def test_push_tiers_explicit_strong_branch_and_norm(self):
        cfg, _, _ = resolved()
        mixture = cfg.domain_rand.push_force_mixture
        # A deterministic stratified quantile grid exercises the exact tier masses.
        draws = torch.stack(((torch.arange(200, dtype=torch.float64) + .5) / 200,
                             torch.linspace(0., 1., 200, dtype=torch.float64),
                             torch.linspace(0., 1., 200, dtype=torch.float64)), dim=1)
        force, tier = horizontal_push_force(draws, mixture)
        self.assertEqual(torch.bincount(tier).tolist(), [160, 30, 10])
        norm = force.norm(dim=1)
        for index, (lo, hi) in enumerate(mixture['magnitude_ranges_n']):
            self.assertTrue(((norm[tier == index] >= lo - 1e-10) & (norm[tier == index] <= hi + 1e-10)).all())
        self.assertLessEqual(float(norm.max()), 150. + 1e-10)
        self.assertEqual(float(force[:, 2].abs().sum()), 0.)
        _, edges = horizontal_push_force(torch.tensor([[.799, .5, .5], [.80, .5, .5], [.949, .5, .5], [.95, .5, .5]]), mixture)
        self.assertEqual(edges.tolist(), [0, 1, 1, 2])
        endpoint = torch.ones(1000, 3)
        endpoint[:, 0] = .99
        endpoint[:, 2] = torch.linspace(0., 1., 1000)
        force, _ = horizontal_push_force(endpoint, mixture)
        self.assertTrue((force.norm(dim=1) <= 150.).all())

    def test_push_event_duration_direction_impulse_and_reapplication(self):
        env = sampler_fixture(3)
        env.push_force = torch.zeros(3, 1, 3)
        env.push_torque = torch.ones_like(env.push_force)
        env.active_push_force = torch.zeros_like(env.push_force)
        env.active_push_torque = torch.zeros_like(env.push_force)
        env.push_steps_left = torch.zeros(3, dtype=torch.long)
        env.next_push_steps = torch.ones(3, dtype=torch.long)
        env.push_event_tier = torch.full((3,), -1, dtype=torch.long)
        env.push_event_duration_steps = torch.zeros(3, dtype=torch.long)
        env.push_event_impulse_ns = torch.zeros(3)
        env.training_diagnostics = Mock()
        env.robot, env.base_link_idx = Mock(), 2
        torch.manual_seed(90)
        with patch('torch.rand', return_value=torch.tensor([[.1, .5, .0], [.82, .5, .25], [.99, .5, .5]])):
            env._sample_pushes()
        original = env.push_force.clone()
        torch.testing.assert_close(original[:, 0].norm(dim=1), torch.tensor([35., 75., 125.]))
        self.assertTrue(((env.next_push_steps >= 250) & (env.next_push_steps <= 500)).all())
        self.assertTrue(((env.push_steps_left >= 20) & (env.push_steps_left <= 40)).all())
        duration = env.push_steps_left.clone()
        impulse = torch.zeros(3)
        for tick in range(41):
            if tick % 4 == 0:
                env._sample_pushes()
            env._apply_pushes()
            call = env.robot.apply_links_external_wrench.call_args.kwargs
            self.assertEqual(call['links_idx_local'], [2])
            self.assertEqual(str(call['ref']), str(__import__('genesis').link_ref_frame.link_COM))
            self.assertTrue((call['torque'] == 0).all())
            torch.testing.assert_close(call['force'], original * (tick < duration)[:, None, None])
            impulse += call['force'][:, 0].norm(dim=1) * env.cfg.sim.dt
        torch.testing.assert_close(impulse, env.push_event_impulse_ns)
        self.assertEqual(env.training_diagnostics.add.call_count, 3)
        torch.testing.assert_close(env.push_force, original)
        self.assertEqual(env.cfg.domain_rand.push_interval_range_s, [5., 10.])


    def test_fresh_state_and_unchanged_optimization_contract(self):
        cfg, train, args = resolved()
        self.assertIsNone(getattr(args, 'resolved_checkpoint', None))
        self.assertFalse(train.runner.resume)
        for key in ('load_run', 'checkpoint', 'resume_path', 'checkpoint_load_cfg'):
            self.assertIsNone(getattr(train.runner, key))
        self.assertEqual((cfg.env.num_envs, train.runner.num_steps_per_env, train.runner.max_iterations), (4096, 64, 2000))
        self.assertEqual((train.algorithm.schedule, train.algorithm.learning_rate, train.algorithm.gamma,
                          train.algorithm.lam, train.algorithm.entropy_coef), ('fixed', 3e-4, .995, .95, .003))
        self.assertEqual(train.actor.distribution_cfg['init_std'], .4)
        self.assertEqual(train.actor.distribution_cfg['std_range'], [.1, .7])


if __name__ == '__main__':
    unittest.main()
