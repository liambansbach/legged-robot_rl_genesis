"""Configuration-only fresh initialization and unchanged native continuation."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml

from robot_gym.envs.go2w.go2w_config import apply_go2w_profile, validate_training, v3_completed_updates
from robot_gym.envs.go2w.go2w_env import Go2WEnv
from robot_gym.utils import get_args, task_registry
from robot_gym.utils.helpers import class_to_dict, update_cfg_from_args


class FreshRollingControl(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name)
        cfg, train = task_registry.get_cfgs('go2w')
        apply_go2w_profile(cfg, train, 'transfer_v3')
        cfg.go2w_finetune = train.go2w_finetune = 'navigation_rolling_control'
        cfg.phase_observation_mode = 'command_demand'
        cfg.commands.pure_lateral_magnitude_range = [.02, .30]
        cfg.sensor_smooth = {'hip_weight': 3.}
        cfg.rewards.scales.stand_still = -5.
        cfg.rewards.scales.rolling_placement = -.05
        cfg.rewards.phase_objective['rolling_yaw_scale'] = .07
        cfg.rewards.rolling_placement = dict(width_deadband_m=.04, width_scale_m=.10,
                                            midpoint_deadband_m=.015, midpoint_scale_m=.05)
        cfg.refinement_parent = {'checkpoint': 'must_not_load.pt', 'iteration': 499}
        cfg.training_resume = {'checkpoint': 'must_not_load.pt'}
        train.algorithm.learning_rate, train.algorithm.schedule = 5e-5, 'fixed'
        train.runner.resume = True
        train.runner.resume_path = train.runner.load_run = 'must_not_load.pt'
        train.runner.checkpoint = 499
        train.runner.checkpoint_load_cfg = dict(actor=True, critic=True, optimizer=False, iteration=False)
        self.saved = yaml.safe_load(yaml.safe_dump(dict(task='go2w', env_cfg=class_to_dict(cfg), train_cfg=class_to_dict(train))))
        self.config = self.path/'config.yaml'
        self.config.write_text(yaml.safe_dump(self.saved))

    def args(self, *extra):
        with patch.object(sys, 'argv', ['train', '--task', 'go2w', '--go2w_profile', 'transfer_v3',
                                      '--go2w_fresh_recipe', str(self.config), *extra]):
            return get_args()

    def resolve(self):
        args = self.args()
        args.resolved_checkpoint = 'stale.pt'
        cfg, train = task_registry.get_cfgs('go2w')
        with patch('torch.load', side_effect=AssertionError('No checkpoint may be read')):
            update_cfg_from_args(cfg, train, args)
            first = copy.deepcopy((class_to_dict(cfg), class_to_dict(train)))
            update_cfg_from_args(cfg, train, args)
            self.assertEqual(first, (class_to_dict(cfg), class_to_dict(train)))
            validate_training(args, cfg, train)
        return cfg, train, args

    def test_complete_recipe_and_no_inherited_loading(self):
        cfg, train, args = self.resolve()
        expected = copy.deepcopy(self.saved['env_cfg'])
        expected.update(refinement_parent=None, training_resume=None, recipe_source=cfg.recipe_source, seed=1)
        expected['env']['num_envs'] = 4096
        self.assertEqual(class_to_dict(cfg), expected)
        self.assertEqual(args.fresh_task_recipe, self.saved)
        self.assertIsNone(args.resolved_checkpoint)
        self.assertFalse(train.runner.resume)
        for key in ('resume_path', 'load_run', 'checkpoint', 'checkpoint_load_cfg'):
            self.assertIsNone(getattr(train.runner, key))
        self.assertEqual((train.algorithm.learning_rate, train.algorithm.schedule, train.algorithm.desired_kl),
                         (3e-4, 'adaptive', .01))
        self.assertEqual((train.runner.max_iterations, train.runner.save_interval, train.runner.num_steps_per_env),
                         (3000, 250, 64))
        self.assertEqual(train.actor.distribution_cfg['init_std'], .4)
        cfg.rewards.phase_objective['rolling_yaw_scale'] = .1
        self.assertEqual(args.fresh_task_recipe['env_cfg']['rewards']['phase_objective']['rolling_yaw_scale'], .07)

    def test_rejects_loading_or_task_overrides(self):
        for extra in (['--resume'], ['--load_run', 'old'], ['--checkpoint', '499'],
                      ['--go2w_finetune', 'navigation_rolling_control'], ['--transfer_delay', '1'],
                      ['--experiment_name', self.saved['train_cfg']['runner']['experiment_name']]):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                cfg, train = task_registry.get_cfgs('go2w')
                update_cfg_from_args(cfg, train, self.args(*extra))

    def test_registration_preserves_config_and_saved_resume(self):
        cfg, train, args = self.resolve()
        raw = copy.deepcopy(class_to_dict(cfg.rewards))
        for _ in range(2):
            env = Go2WEnv.__new__(Go2WEnv)
            env.cfg, env.dt, env.num_envs, env.device = copy.deepcopy(cfg), .02, 1, 'cpu'
            env.reward_scales = class_to_dict(env.cfg.rewards.scales)
            env._prepare_reward_function()
            self.assertEqual(class_to_dict(env.cfg.rewards), raw)
            self.assertEqual(env.reward_scales['tracking_yaw'], .8*.02)
            self.assertEqual(env.reward_scales['stand_still'], -5*.02)
            self.assertEqual(env.reward_scales['rolling_placement'], -.05*.02)
        fresh = dict(task='go2w', env_cfg=class_to_dict(cfg), train_cfg=class_to_dict(train))
        self.config.write_text(yaml.safe_dump(fresh))
        torch.save({'iter': 1}, self.path/'model_1.pt')
        with patch.object(sys, 'argv', ['train', '--task', 'go2w', '--go2w_profile', 'transfer_v3',
                '--resume', '--load_run', str(self.path), '--checkpoint', '1', '--max_iterations', '10']):
            resume_args = get_args()
        continued, training = task_registry.get_cfgs('go2w')
        update_cfg_from_args(continued, training, resume_args)
        validate_training(resume_args, continued, training)
        self.assertEqual(training.runner.checkpoint_load_cfg, dict(actor=True, critic=True, optimizer=True, iteration=True))
        self.assertEqual(class_to_dict(training.algorithm), fresh['train_cfg']['algorithm'])
        self.assertEqual(continued.recipe_source, cfg.recipe_source)
        self.assertIsNone(continued.refinement_parent)
        self.assertEqual(continued.training_resume['source_total_updates'], 2)
        self.assertEqual(v3_completed_updates(fresh['env_cfg'], 1), 2)
        resume_args.resume = False
        replay, replay_train, _ = task_registry.resolve_replay(resume_args)
        self.assertEqual(class_to_dict(replay.rewards), raw)
        self.assertEqual(replay.phase_observation_mode, 'command_demand')
        self.assertEqual(class_to_dict(replay_train.algorithm), fresh['train_cfg']['algorithm'])


if __name__ == '__main__':
    unittest.main()
