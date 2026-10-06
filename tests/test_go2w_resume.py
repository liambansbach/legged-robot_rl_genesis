"""Focused saved-recipe restoration checks; no simulator or optimizer updates."""
import json
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import yaml
from robot_gym.envs.go2w.go2w_config import apply_go2w_profile, validate_training, v3_completed_updates
from robot_gym.envs.go2w.go2w_env import Go2WEnv
from robot_gym.utils import get_args, task_registry
from robot_gym.utils.helpers import class_to_dict, update_cfg_from_args


class V3Resume(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name)
        cfg, train = task_registry.get_cfgs('go2w')
        apply_go2w_profile(cfg, train, 'transfer_v3')
        cfg.go2w_finetune = train.go2w_finetune = 'navigation_partial_lateral'
        cfg.phase_observation_mode = 'command_demand'
        cfg.commands.pure_lateral_magnitude_range = [.02,.3]
        cfg.sensor_smooth = {'hip_weight':2.7}
        cfg.refinement_parent = {'checkpoint':'phase_parent.pt','iteration':199,'prior_updates':2200}
        train.algorithm.learning_rate, train.algorithm.schedule = 5e-5, 'fixed'
        train.runner.checkpoint_load_cfg = {'actor':True,'critic':True,'optimizer':False,'iteration':False}
        self.saved = {'task':'go2w','env_cfg':class_to_dict(cfg),'train_cfg':class_to_dict(train)}
        (self.path/'config.yaml').write_text(yaml.safe_dump(self.saved))
        torch.save({'iter':199},self.path/'model_199.pt')

    def args(self, *extra):
        with patch.object(sys,'argv',['train','--task','go2w','--go2w_profile','transfer_v3','--resume',
                '--load_run',str(self.path),'--checkpoint','199','--max_iterations','200',*extra]):
            return get_args()

    def test_authoritative_recipe_and_full_load_flags(self):
        args = self.args()
        cfg, train = task_registry.get_cfgs('go2w')
        update_cfg_from_args(cfg, train, args)
        before = class_to_dict(cfg)
        update_cfg_from_args(cfg, train, args)
        self.assertEqual(before,class_to_dict(cfg))
        validate_training(args,cfg,train)
        for key in ('rewards','commands','control','domain_rand','noise','phase_observation_mode','refinement_parent'):
            self.assertEqual(json.loads(json.dumps(class_to_dict(getattr(cfg,key)))),self.saved['env_cfg'][key])
        self.assertEqual(cfg.sensor_smooth,{'hip_weight':2.7})
        self.assertEqual(class_to_dict(train.algorithm),self.saved['train_cfg']['algorithm'])
        self.assertEqual(train.runner.checkpoint_load_cfg,{'actor':True,'critic':True,'optimizer':True,'iteration':True})
        self.assertEqual(cfg.training_resume['source_iteration'],199)
        self.assertEqual(cfg.training_resume['source_total_updates'],2400)
        self.assertEqual(train.runner.max_iterations,200)
        self.assertNotEqual(train.runner.experiment_name,self.saved['train_cfg']['runner']['experiment_name'])
        cfg.commands.pure_lateral_magnitude_range = [.1,.3]
        with self.assertRaisesRegex(ValueError,'env_cfg.commands'): validate_training(args,cfg,train)

    def test_rejects_new_recipe_and_population_override(self):
        for extra in (['--go2w_finetune','navigation_partial_lateral'],['--num_envs','128'],['--seed','2'],['--entropy_coef','0']):
            cfg, train = task_registry.get_cfgs('go2w')
            with self.assertRaises(ValueError): update_cfg_from_args(cfg,train,self.args(*extra))

    def test_native_label_lineage_and_historical_replay(self):
        resumed = dict(self.saved['env_cfg'],training_resume={'source_iteration':199,'source_total_updates':2400})
        self.assertEqual(v3_completed_updates(resumed,199),2401)
        self.assertEqual(v3_completed_updates(resumed,200),2402)
        self.assertEqual(v3_completed_updates(self.saved['env_cfg'],199),2400)
        self.saved['env_cfg']['phase_observation_mode'] = 'unconditional'
        (self.path/'config.yaml').write_text(yaml.safe_dump(self.saved))
        args = self.args()
        args.resume = False
        args.go2w_profile = None
        cfg, train, checkpoint = task_registry.resolve_replay(args)
        self.assertEqual(cfg.phase_observation_mode,'unconditional')
        self.assertFalse(train.runner.checkpoint_load_cfg['optimizer'])
        self.assertEqual(checkpoint,self.path/'model_199.pt')

    def test_resume_marker_does_not_require_output_for_inference(self):
        env = Go2WEnv.__new__(Go2WEnv)
        env.cfg = SimpleNamespace(training_resume={'source_iteration':199})
        env.phase_guided = env.event_step = False
        runner = SimpleNamespace(logger=SimpleNamespace(log_dir=None))
        env.setup_runner(runner)

    def test_fixed_lr_opt_in_preserves_recipe_and_survives_ordinary_resume(self):
        torch.save({'iter':199, 'optimizer_state_dict':{'param_groups':[{'lr':.00086}]}},
                   self.path/'model_199.pt')
        source = yaml.safe_load((self.path/'config.yaml').read_text())
        args = self.args('--go2w_resume_fixed_lr', '.0003')
        cfg, train = task_registry.get_cfgs('go2w')
        update_cfg_from_args(cfg, train, args)
        validate_training(args, cfg, train)
        expected = dict(source['train_cfg']['algorithm'], learning_rate=.0003, schedule='fixed')
        self.assertEqual(class_to_dict(train.algorithm), expected)
        actual_env = json.loads(json.dumps(class_to_dict(cfg)))
        actual_env.pop('training_resume')
        if 'training_resume' in source['env_cfg']:
            actual_env['training_resume'] = source['env_cfg']['training_resume']
        self.assertEqual(actual_env, source['env_cfg'])
        self.assertEqual(args.v3_resume_recipe['env_cfg'], source['env_cfg'])
        for key in ('actor', 'critic', 'algorithm'):
            self.assertEqual(args.v3_resume_recipe['train_cfg'][key], source['train_cfg'][key])
        self.assertEqual(cfg.training_resume['optimization_change']['source_optimizer_learning_rates'], [.00086])
        self.assertEqual(yaml.safe_load((self.path/'config.yaml').read_text()), source)
        before = class_to_dict(cfg), class_to_dict(train)
        update_cfg_from_args(cfg, train, args)
        self.assertEqual(before, (class_to_dict(cfg), class_to_dict(train)))
        # The next generation uses its immediate saved config, without reapplying the flag.
        (self.path/'config.yaml').write_text(yaml.safe_dump(dict(task='go2w', env_cfg=before[0], train_cfg=before[1])))
        next_args = self.args()
        next_cfg, next_train = task_registry.get_cfgs('go2w')
        update_cfg_from_args(next_cfg, next_train, next_args)
        validate_training(next_args, next_cfg, next_train)
        self.assertEqual(class_to_dict(next_train.algorithm), expected)
        self.assertEqual(next_cfg.training_resume['source_total_updates'], 2401)
        self.assertEqual(next_cfg.training_resume['previous_generation'], cfg.training_resume)

    def test_fixed_lr_is_validated_and_task_changes_are_rejected(self):
        for value in ('0', '-.1', 'nan', 'inf'):
            cfg, train = task_registry.get_cfgs('go2w')
            with self.assertRaisesRegex(ValueError, 'finite positive LR'):
                update_cfg_from_args(cfg, train, self.args('--go2w_resume_fixed_lr', value))
        args = self.args('--go2w_resume_fixed_lr', '.0003')
        args.resume = False
        with self.assertRaisesRegex(ValueError, 'training'):
            update_cfg_from_args(cfg, train, args)
        torch.save({'iter':199, 'optimizer_state_dict':{'param_groups':[{'lr':.00086}]}},
                   self.path/'model_199.pt')
        args = self.args('--go2w_resume_fixed_lr', '.0003')
        cfg, train = task_registry.get_cfgs('go2w')
        update_cfg_from_args(cfg, train, args)
        train.algorithm.entropy_coef += .01
        with self.assertRaisesRegex(ValueError, 'train_cfg.algorithm'):
            validate_training(args, cfg, train)
        args = self.args('--go2w_resume_fixed_lr', '.0003', '--experiment_name',
                         self.saved['train_cfg']['runner']['experiment_name'])
        with self.assertRaisesRegex(ValueError, 'distinct output experiment'):
            update_cfg_from_args(cfg, train, args)

    def test_post_load_override_only_changes_lr_in_every_adam_group(self):
        parameters = [torch.nn.Parameter(torch.ones(2)), torch.nn.Parameter(torch.ones(3))]
        optimizer = torch.optim.Adam([{'params':[parameters[0]], 'lr':.008},
                                      {'params':[parameters[1]], 'lr':.002}])
        for p in parameters:
            optimizer.state[p] = dict(step=torch.tensor(10040.), exp_avg=torch.ones_like(p),
                                      exp_avg_sq=torch.full_like(p, 2.))
        before = copy.deepcopy(optimizer.state_dict())
        runner = SimpleNamespace(checkpoint_path='source.pt', cfg={'algorithm':{}},
            alg=SimpleNamespace(schedule='adaptive', learning_rate=.008, optimizer=optimizer))
        train = SimpleNamespace(algorithm=SimpleNamespace(schedule='fixed', learning_rate=.0003),
            runner=SimpleNamespace(checkpoint_load_cfg=dict(actor=True, critic=True, optimizer=True, iteration=True)))
        env = Go2WEnv.__new__(Go2WEnv)
        env.finalize_runner_loading(runner, train, SimpleNamespace(go2w_resume_fixed_lr=.0003))
        self.assertEqual((runner.alg.schedule, runner.alg.learning_rate), ('fixed', .0003))
        self.assertEqual(runner.cfg['algorithm'], dict(schedule='fixed', learning_rate=.0003))
        after = optimizer.state_dict()
        for index in before['state']:
            for key, value in before['state'][index].items():
                self.assertTrue(torch.equal(value, after['state'][index][key]))
        for old, new in zip(before['param_groups'], after['param_groups']):
            self.assertEqual(new, dict(old, lr=.0003))
        runner.alg.schedule, runner.alg.learning_rate = 'adaptive', .008
        env.finalize_runner_loading(runner, train, SimpleNamespace(go2w_resume_fixed_lr=None))
        self.assertEqual((runner.alg.schedule, runner.alg.learning_rate), ('adaptive', .008))
        train.runner.checkpoint_load_cfg['optimizer'] = False
        with self.assertRaisesRegex(ValueError, 'full-state'):
            env.finalize_runner_loading(runner, train, SimpleNamespace(go2w_resume_fixed_lr=.0003))


if __name__ == '__main__': unittest.main()
