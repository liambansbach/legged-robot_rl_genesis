"""Focused saved-recipe restoration checks; no simulator or optimizer updates."""
import json
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


if __name__ == '__main__': unittest.main()
