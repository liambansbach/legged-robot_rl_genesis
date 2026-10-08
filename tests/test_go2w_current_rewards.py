"""Focused schema2-to-current reward continuation checks; no simulator updates."""
import copy
import hashlib
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import yaml

from robot_gym.envs import *  # noqa: F401,F403
from robot_gym.envs.go2w.go2w_config import GO2WCfg, restore_saved_config, validate_training
from robot_gym.envs.go2w.go2w_env import Go2WEnv
from robot_gym.envs.go2w.phase import broad_precision_tracking, huber
from robot_gym.utils import get_args, task_registry
from robot_gym.utils.helpers import class_to_dict, update_cfg_from_args


class CurrentRewardsContinuation(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        cfg, train = task_registry.get_cfgs('go2w')
        self.saved = dict(task='go2w', env_cfg=class_to_dict(cfg), train_cfg=class_to_dict(train))
        self.saved['env_cfg']['config_version'] = 2
        self.saved['env_cfg']['rewards']['common_tracking'].pop('precision_kernel')
        self.saved['env_cfg']['rewards']['scales'].update(phase_clearance=-1., ang_vel_xy=-.075)
        self.config = self.run / 'config.yaml'
        self.config.write_text(yaml.safe_dump(self.saved))
        (self.run / 'model_1000.pt').write_bytes(b'Config-only fixture; real full state checked by smoke')
        self.digest = hashlib.sha256(self.config.read_bytes()).hexdigest()

    def args(self, *flags):
        with patch.object(sys, 'argv', ['train', '--task', 'go2w', '--load_run', str(self.run),
                                       '--checkpoint', '1000', '--max_iterations', '1000', *flags]):
            return get_args()

    def resolve(self, *flags):
        args = self.args('--resume', *flags)
        cfg, train = task_registry.get_cfgs('go2w')
        update_cfg_from_args(cfg, train, args)
        validate_training(args, cfg, train)
        return cfg, train, args

    @staticmethod
    def registered(cfg):
        env = Go2WEnv.__new__(Go2WEnv)
        env.cfg, env.device, env.num_envs, env.dt = cfg, 'cpu', 1, .02
        env.reward_scales = class_to_dict(cfg.rewards.scales)
        env._prepare_reward_function()
        return env

    def test_exact_kernel_zero_symmetry_monotonic_bounded_and_large_errors(self):
        e = torch.tensor([0., 1e-9, .003, .005, .01, .03, .06, .1, .5, 1., 2.], dtype=torch.float64)
        for scale in (.25, .15, .35):
            r = broad_precision_tracking(e, scale, .03, .25)
            expected = 1 - huber(e/scale) + .25*torch.expm1(-e/.03)
            torch.testing.assert_close(r, expected, rtol=0, atol=0)
            self.assertEqual(float(r[0]), 1.)
            self.assertTrue((r[:-1] > r[1:]).all())
            torch.testing.assert_close(r, broad_precision_tracking(-e, scale, .03, .25))
            precision = 1 - huber(e/scale) - r
            self.assertTrue(((precision >= 0) & (precision <= .25+1e-14)).all())
            self.assertAlmostEqual(float(r[-2]-r[-1]), 1/scale, places=12)
            old = broad_precision_tracking(e, scale, .03, .25, 'gaussian')
            torch.testing.assert_close(old, 1-huber(e/scale)+.25*torch.expm1(-.5*(e/.03).square()))
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            broad_precision_tracking(e, .25, .03, .25, 'typo')

    def test_ordinary_resume_and_replay_preserve_gaussian_source(self):
        cfg, _, _ = self.resolve()
        replay, _, _ = task_registry.resolve_replay(self.args())
        for resolved in (cfg, replay):
            self.assertEqual(resolved.config_version, 2)
            self.assertEqual(resolved.rewards.common_tracking['precision_kernel'], 'gaussian')
            self.assertEqual(resolved.rewards.scales.phase_clearance, -1.)
            self.assertEqual(resolved.rewards.scales.ang_vel_xy, -.075)
        self.assertEqual(hashlib.sha256(self.config.read_bytes()).hexdigest(), self.digest)

    def test_runtime_all_axes_stand_motion_and_stop_use_same_kernel(self):
        for flags, kernel in (((), 'gaussian'), (('--resume_current_rewards',), 'absolute_exponential')):
            cfg, _, _ = self.resolve(*flags)
            env = self.registered(cfg)
            for command in ((0., 0., 0.), (.5, 0., 0.), (0., .3, 0.), (0., 0., -.8), (.2, -.1, .3)):
                env.commands = torch.tensor([command])
                for error in (.003, -.003, .5, -.5):
                    env.base_lin_vel = env.commands - error
                    env.base_ang_vel = env.commands - error
                    for axis in range(3):
                        c = cfg.rewards.common_tracking
                        expected = broad_precision_tracking(torch.tensor([error]), c['broad_scales'][axis],
                                                            c['precision_scales'][axis], c['beta'][axis], kernel)
                        torch.testing.assert_close(env._tracking_axis(axis), expected)

    def test_intended_diff_only_idempotent_and_saved_effective_roundtrip(self):
        cfg, train, args = self.resolve('--resume_current_rewards')
        diff = cfg.training_resume['reward_override']['diff']
        self.assertEqual(set(diff), {'env_cfg.config_version',
            'env_cfg.rewards.common_tracking.precision_kernel',
            'env_cfg.rewards.scales.phase_clearance', 'env_cfg.rewards.scales.ang_vel_xy'})
        self.assertEqual(diff['env_cfg.rewards.scales.phase_clearance'], {'saved': -1., 'current': -2.})
        before = class_to_dict(cfg)
        update_cfg_from_args(cfg, train, args)
        self.assertEqual(class_to_dict(cfg), before)
        saved = yaml.safe_load(yaml.safe_dump(dict(env_cfg=before, train_cfg=class_to_dict(train))))
        restored, restored_train = restore_saved_config(saved)
        self.assertEqual(class_to_dict(restored), before)
        self.assertEqual(class_to_dict(restored_train), class_to_dict(train))
        self.assertEqual(hashlib.sha256(self.config.read_bytes()).hexdigest(), self.digest)

    def test_current_values_not_hidden_literals_and_single_dt_after_load(self):
        with patch.object(GO2WCfg.rewards.scales, 'phase_clearance', -2.25):
            cfg, train, args = self.resolve('--resume_current_rewards')
        env = self.registered(cfg)
        self.assertEqual(env.reward_scales['phase_clearance'], -.045)
        self.assertEqual(env.reward_scales['ang_vel_xy'], -.003)
        self.assertEqual(env.reward_scales['tracking_yaw'], .016)
        self.assertEqual(env.reward_scales['termination'], -5.)
        norm = SimpleNamespace(count=torch.tensor(17))
        runner = SimpleNamespace(checkpoint_path=str(self.run/'model_1000.pt'),
                                 alg=SimpleNamespace(actor=SimpleNamespace(obs_normalizer=norm),
                                                     critic=SimpleNamespace(obs_normalizer=norm)))
        before = copy.deepcopy(env.reward_scales)
        with patch('robot_gym.envs.go2w.training_diagnostics.verify_resume_state',
                   return_value={'exact_state_match': True}):
            env.finalize_runner_loading(runner, train, args)
            self.assertEqual(env.reward_scales, before)
            self.assertEqual(env.resume_validation['common_tracking']['precision_kernel'], 'absolute_exponential')
            env.reward_scales['phase_clearance'] *= env.dt
            with self.assertRaisesRegex(ValueError, 'dt exactly once'):
                env.finalize_runner_loading(runner, train, args)
            env.reward_scales = before
            env.cfg.rewards.common_tracking['precision_kernel'] = 'gaussian'
            with self.assertRaisesRegex(ValueError, 'overwritten'):
                env.finalize_runner_loading(runner, train, args)

    def test_unrelated_central_task_ppo_and_cli_drift_rejected(self):
        for owner, name, value in ((GO2WCfg.control, 'decimation', 5),
                                    (GO2WCfg.rewards.scales, 'tracking_yaw', 1.2)):
            with self.subTest(name=name), patch.object(owner, name, value):
                with self.assertRaisesRegex(ValueError, 'Unintended'):
                    self.resolve('--resume_current_rewards')
        from robot_gym.envs.go2w.go2w_config import GO2WCfgPPO
        with patch.object(GO2WCfgPPO.algorithm, 'learning_rate', .001):
            with self.assertRaisesRegex(ValueError, 'Unintended'):
                self.resolve('--resume_current_rewards')
        for flags in (('--num_envs', '1'), ('--seed', '2')):
            with self.assertRaisesRegex(ValueError, 'Unintended'):
                self.resolve('--resume_current_rewards', *flags)

    def test_explicit_source_schema_and_active_name_guards(self):
        for flags, message in ((('--resume_current_rewards',), 'requires training'),
                (('--resume', '--resume_current_rewards', '--checkpoint', '-1'), 'explicit'),
                (('--resume', '--resume_current_rewards', '--load_run', '-1'), 'explicit'),
                (('--resume', '--resume_current_rewards', '--resume_current_reward_scales'), 'only one')):
            args = self.args(*flags)
            cfg, train = task_registry.get_cfgs('go2w')
            with self.assertRaisesRegex(ValueError, message):
                update_cfg_from_args(cfg, train, args)
        with patch.object(GO2WCfg.rewards.scales, 'phase_clearance', 0.):
            with self.assertRaisesRegex(ValueError, 'active reward names'):
                self.resolve('--resume_current_rewards')
        bad = copy.deepcopy(self.saved)
        bad['env_cfg']['rewards']['common_tracking']['precision_kernel'] = 'absolute_exponential'
        with self.assertRaisesRegex(ValueError, 'disagree'):
            restore_saved_config(bad)
        with patch.dict(GO2WCfg.rewards.common_tracking, precision_kernel='gaussian'):
            with self.assertRaisesRegex(ValueError, 'disagree'):
                self.resolve('--resume_current_rewards')


if __name__ == '__main__':
    unittest.main()
