"""Reward-scale continuation guards; no simulator or optimizer updates."""
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

from robot_gym.envs import *  # noqa: F401,F403 - register before shared helpers
from robot_gym.envs.go2w.go2w_config import GO2WCfg, validate_training
from robot_gym.envs.go2w.go2w_env import Go2WEnv
from robot_gym.utils import get_args, task_registry
from robot_gym.utils.helpers import class_to_dict, update_cfg_from_args


class RewardScaleContinuation(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        cfg, train = task_registry.get_cfgs('go2w')
        self.saved = dict(task='go2w', env_cfg=class_to_dict(cfg), train_cfg=class_to_dict(train))
        self.saved['env_cfg']['rewards']['scales']['tracking_yaw'] = .8
        self.config = self.run / 'config.yaml'
        self.config.write_text(yaml.safe_dump(self.saved))
        # Config resolution only needs an existing checkpoint; native state is checked by the smoke.
        (self.run / 'model_7.pt').write_bytes(b'config-only fixture')
        self.digest = hashlib.sha256(self.config.read_bytes()).hexdigest()

    def args(self, *flags):
        with patch.object(sys, 'argv', ['train', '--task', 'go2w', '--load_run', str(self.run),
                                       '--checkpoint', '7', '--max_iterations', '500', *flags]):
            return get_args()

    def resolve(self, opted=False):
        args = self.args('--resume', *(['--resume_current_reward_scales'] if opted else []))
        cfg, train = task_registry.get_cfgs('go2w')
        update_cfg_from_args(cfg, train, args)
        validate_training(args, cfg, train)
        return cfg, train, args

    @staticmethod
    def registered(cfg):
        env = Go2WEnv.__new__(Go2WEnv)
        env.cfg, env.device, env.num_envs = cfg, 'cpu', 1
        env.dt = cfg.sim.dt * cfg.control.decimation
        env.reward_scales = class_to_dict(cfg.rewards.scales)
        env._prepare_reward_function()
        return env

    def test_ordinary_resume_and_replay_retain_saved_scales(self):
        cfg, train, _ = self.resolve()
        self.assertEqual(cfg.rewards.scales.tracking_yaw, .8)
        self.assertEqual(self.registered(cfg).reward_scales['tracking_yaw'], .016)
        self.assertEqual(class_to_dict(cfg.rewards), self.saved['env_cfg']['rewards'])
        self.assertEqual(train.runner.checkpoint_load_cfg,
                         dict(actor=True, critic=True, optimizer=True, iteration=True))
        replay, _, _ = task_registry.resolve_replay(self.args())
        self.assertEqual(class_to_dict(replay.rewards), self.saved['env_cfg']['rewards'])
        self.assertEqual(hashlib.sha256(self.config.read_bytes()).hexdigest(), self.digest)

    def test_opted_in_diff_is_only_yaw_and_configuration_is_idempotent(self):
        cfg, train, args = self.resolve(True)
        diff = cfg.training_resume['reward_scale_override']['diff']
        self.assertEqual(diff, {'tracking_yaw': {'old': .8, 'new': 1.2}})
        actual = class_to_dict(cfg)
        actual['training_resume'] = None
        actual['rewards']['scales']['tracking_yaw'] = .8
        self.assertEqual(actual, self.saved['env_cfg'])
        for section in ('actor', 'critic', 'algorithm'):
            self.assertEqual(class_to_dict(getattr(train, section)), self.saved['train_cfg'][section])
        before = class_to_dict(cfg)
        update_cfg_from_args(cfg, train, args)
        self.assertEqual(class_to_dict(cfg), before)
        env = self.registered(cfg)
        self.assertEqual(env.reward_scales['tracking_yaw'], .024)
        self.assertEqual(env.cfg.rewards.scales.tracking_yaw, 1.2)
        self.assertEqual(env.reward_scales['termination'], -5.)
        self.assertEqual(hashlib.sha256(self.config.read_bytes()).hexdigest(), self.digest)

    def test_value_comes_from_central_config_without_a_hidden_literal(self):
        with patch.object(GO2WCfg.rewards.scales, 'tracking_yaw', 1.3):
            cfg, _, args = self.resolve(True)
        self.assertEqual(cfg.rewards.scales.tracking_yaw, 1.3)
        self.assertAlmostEqual(self.registered(cfg).reward_scales['tracking_yaw'], .026, places=14)
        args._go2w_current_reward_scales['tracking_yaw'] = 9.
        self.assertEqual(cfg.rewards.scales.tracking_yaw, 1.3)  # Separate snapshots.

    def test_opt_in_requires_training_resume(self):
        args = self.args('--resume_current_reward_scales')
        cfg, train = task_registry.get_cfgs('go2w')
        with self.assertRaisesRegex(ValueError, 'requires training --resume'):
            update_cfg_from_args(cfg, train, args)

    def test_rejects_changed_active_names_and_nonfinite_scales(self):
        for value, message in [(0., 'active reward names'), (float('nan'), 'finite')]:
            with self.subTest(value=value), patch.object(GO2WCfg.rewards.scales, 'tracking_yaw', value):
                with self.assertRaisesRegex(ValueError, message):
                    self.resolve(True)

    def test_after_load_verification_checks_effective_scales_without_reweighting(self):
        cfg, train, args = self.resolve(True)
        env = self.registered(cfg)
        norm = SimpleNamespace(count=torch.tensor(17))
        runner = SimpleNamespace(checkpoint_path=str(self.run/'model_7.pt'),
                                 alg=SimpleNamespace(actor=SimpleNamespace(obs_normalizer=norm),
                                                     critic=SimpleNamespace(obs_normalizer=norm)))
        before = copy.deepcopy(env.reward_scales)
        with patch('robot_gym.envs.go2w.training_diagnostics.verify_resume_state',
                   return_value={'exact_state_match': True}) as verify:
            env.finalize_runner_loading(runner, train, args)
            verify.assert_called_once_with(runner, runner.checkpoint_path)
            self.assertEqual(env.reward_scales, before)
            self.assertEqual(env.resume_validation['runtime_reward_scales']['tracking_yaw'], .024)
            env.reward_scales['tracking_yaw'] *= env.dt
            with self.assertRaisesRegex(ValueError, 'dt exactly once'):
                env.finalize_runner_loading(runner, train, args)
            env.reward_scales = before
            env.cfg.rewards.scales.tracking_yaw = .8
            with self.assertRaisesRegex(ValueError, 'overwritten'):
                env.finalize_runner_loading(runner, train, args)


if __name__ == '__main__':
    unittest.main()
