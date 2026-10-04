"""Focused rigid-sensor geometry and optional refinement checks; no PPO updates."""

import unittest
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation

from robot_gym.envs.go2w.go2w_config import MEASURED_URDF, apply_go2w_profile, validate_training
from robot_gym.envs.go2w.go2w_env import Go2WEnv, fixed_sensor_frames, rigid_sensor_state
from robot_gym.envs.go2w.diagnostic_bank import sensor_schedule
from robot_gym.utils.urdf_reader import URDFReader
from robot_gym.utils import get_args, task_registry
from robot_gym.utils.helpers import class_to_dict, update_cfg_from_args


def sampler(cfg, n=4096):
    cfg.env.record_command_families = True
    env = Go2WEnv.__new__(Go2WEnv)
    env.cfg, env.num_envs, env.device, env.dt = cfg, n, 'cpu', .02
    env.joint_names = list(cfg.init_state.default_joint_angles)
    env.joint_dof_idx = list(range(len(env.joint_names)))
    env._build_control_tensors()
    env.commands = torch.zeros(n, 3)
    env.command_steps_left = torch.zeros(n, dtype=torch.long)
    env.sensor_hold_kind = torch.zeros(n, dtype=torch.long)
    env.command_ranges = class_to_dict(cfg.commands.ranges)
    env.command_resampling_enabled = True
    return env


class SensorMotion(unittest.TestCase):
    def test_conditioned_clock_and_reflection_preserve_stepping(self):
        from robot_gym.envs.go2w.phase import clock_observation, demand
        phases = torch.tensor([.2,.3,.4,.5,.6])
        commands = torch.tensor([[0.,0.,0.],[.5,0.,0.],[0.,.03,0.],[0.,.3,0.],[0.,0.,-.8]])
        legacy = clock_observation(phases, commands)
        conditioned = clock_observation(phases, commands, 'command_demand')
        torch.testing.assert_close(conditioned[:2], torch.zeros(2,2))
        torch.testing.assert_close(conditioned[2], legacy[2]*.5)
        torch.testing.assert_close(conditioned[3:], legacy[3:])
        mirrored = commands*torch.tensor([1.,-1.,-1.])
        torch.testing.assert_close(demand(mirrored), demand(commands))
        torch.testing.assert_close(clock_observation((phases+.5)%1,mirrored,'command_demand'), -conditioned, atol=1e-6, rtol=1e-5)

    def test_phase_recipe_restores_sensor_parent_without_reweighting(self):
        parent, training = task_registry.get_cfgs('go2w')
        apply_go2w_profile(parent, training, 'transfer_v3')
        parent.go2w_finetune = training.go2w_finetune = 'sensor_smooth'
        parent.sensor_smooth = {'hip_weight':2.7,'long_hold_probability':.04,'extended_hold_probability':.015,
                                'long_hold_s':[8.,15.],'extended_hold_s':[20.,30.]}
        parent.rewards.scales.sensor_vertical_velocity = -1.
        parent.rewards.scales.lin_vel_z = 0.
        parent.env.episode_length_s = 60.
        parent.refinement_parent = {'checkpoint':'original.pt','iteration':1499,'prior_updates':1500}
        training.algorithm.learning_rate, training.algorithm.schedule = 5e-5, 'fixed'
        training.runner.experiment_name = 'previous_sensor_output'
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path/'config.yaml').write_text(yaml.safe_dump({'task':'go2w','env_cfg':class_to_dict(parent),'train_cfg':class_to_dict(training)}))
            torch.save({'iter':499},path/'model_499.pt')
            with patch.object(sys,'argv',['train','--task','go2w','--go2w_profile','transfer_v3',
                    '--go2w_finetune','sensor_phase_conditioned','--load_run',str(path),'--checkpoint','499']):
                args = get_args()
            cfg, train = task_registry.get_cfgs('go2w')
            update_cfg_from_args(cfg, train, args)
            first = class_to_dict(cfg)
            update_cfg_from_args(cfg, train, args)
            self.assertEqual(first, class_to_dict(cfg))
            validate_training(args, cfg, train)
            for key in ('rewards','commands','control','domain_rand','sim','init_state','sensor_smooth','phase_guidance'):
                self.assertEqual(json.loads(json.dumps(class_to_dict(getattr(cfg,key)))),
                                 json.loads(json.dumps(class_to_dict(getattr(parent,key)))))
            self.assertEqual((cfg.phase_observation_mode,cfg.env.num_observations,cfg.env.num_actions),('command_demand',58,16))
            self.assertEqual((train.algorithm.learning_rate,train.algorithm.schedule,train.runner.max_iterations,train.runner.save_interval),(5e-5,'fixed',200,50))
            self.assertEqual(train.runner.checkpoint_load_cfg,{'actor':True,'critic':True,'optimizer':False,'iteration':False})
            self.assertEqual(cfg.refinement_parent['prior_updates'],2000)
            self.assertEqual(cfg.refinement_parent['previous_generation'],parent.refinement_parent)
            self.assertEqual(cfg.sensor_smooth['hip_weight'],2.7)

    def test_rolling_phase_diagnostic_changes_only_clock_inputs(self):
        from tensordict import TensorDict
        from robot_gym.envs.go2w.diagnostic_bank import rolling_phase_observation
        obs = TensorDict({'policy': torch.randn(4, 58)}, batch_size=[4])
        commands = torch.tensor([[0.,0.,0.],[.5,0.,0.],[0.,.3,0.],[0.,0.,-.8]])
        before = obs.clone()
        changed = rolling_phase_observation(obs, commands)
        torch.testing.assert_close(obs['policy'], before['policy'])
        torch.testing.assert_close(changed['policy'][:, :56], obs['policy'][:, :56])
        torch.testing.assert_close(changed['policy'][:2, 56:], torch.zeros(2,2))
        torch.testing.assert_close(changed['policy'][2:, 56:], obs['policy'][2:, 56:])

    def test_fixed_lr_kl_uses_original_distribution_without_forward(self):
        from tensordict import TensorDict
        from robot_gym.envs.go2w.training_diagnostics import TrainingDiagnostics

        old_mean = torch.zeros(2, 16)
        old_std = torch.ones(2, 16)
        mean = torch.cat((torch.full((2, 16), .1), torch.full((2, 16), 10.)))
        batch = SimpleNamespace(observations=TensorDict({'policy': torch.zeros(2, 58)}, batch_size=[2]),
                                old_distribution_params=(old_mean, old_std), old_actions_log_prob=torch.zeros(2, 1))
        def divergence(old, current):
            m0, s0 = old
            m1, s1 = current
            return (torch.log(s1/s0)+(s0.square()+(m0-m1).square())/(2*s1.square())-.5).sum(-1)
        log_prob_result = torch.zeros(4, requires_grad=True)
        actor = SimpleNamespace(get_output_log_prob=lambda actions: log_prob_result,
                                get_kl_divergence=divergence, output_distribution_params=(mean, torch.ones_like(mean)))
        alg = SimpleNamespace(actor=actor, act=lambda obs: None, process_env_step=lambda *args: None,
                              update=lambda: None, clip_param=.2,
                              storage=SimpleNamespace(mini_batch_generator=lambda: iter([batch])))
        diagnostic = TrainingDiagnostics.__new__(TrainingDiagnostics)
        diagnostic.measured_kl, diagnostic.kl, diagnostic.ppo_clip = [], [], []
        diagnostic.current_batch = None
        diagnostic.install(alg)
        rng = torch.get_rng_state().clone()
        generator = alg.storage.mini_batch_generator()
        next(generator)
        # Native symmetry duplicates log-probs but retains original distribution parameters.
        batch.observations = torch.cat((batch.observations, batch.observations), dim=0)
        batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(2, 1)
        self.assertIs(actor.get_output_log_prob(torch.zeros(4,16)), log_prob_result)
        self.assertAlmostEqual(float(diagnostic.measured_kl[0]), .08, places=6)
        self.assertEqual(diagnostic.kl, [])
        self.assertFalse(diagnostic.measured_kl[0].requires_grad)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        with self.assertRaises(StopIteration): next(generator)
        self.assertIsNone(diagnostic.current_batch)

    def test_authored_fixed_geometry(self):
        frames = fixed_sensor_frames(URDFReader(MEASURED_URDF).robot_file_path_absolute)
        np.testing.assert_allclose(frames["front_realsense"]["translation_m"], [.33881, .0475, .111], atol=1e-6)
        np.testing.assert_allclose(frames["front_realsense"]["rotation"], np.eye(3), atol=1e-7)
        np.testing.assert_allclose(frames["radar"]["translation_m"], [.28945, 0, -.046825], atol=1e-6)
        self.assertFalse(np.allclose(frames["radar"]["rotation"], np.eye(3)))
        self.assertGreater(len(frames["front_realsense"]["fixed_chain"]), 1)

    def test_transport_and_world_up(self):
        p = torch.zeros(1, 3)
        q = torch.tensor([[1., 0., 0., 0.]])
        r = torch.tensor([[.3, .05, .1]])
        _, v = rigid_sensor_state(p, q, torch.tensor([[.5, 0., 0.]]), torch.tensor([[0., 2., 0.]]), r)
        torch.testing.assert_close(v, torch.tensor([[[.7, 0., -.6]]]))
        # Pure commanded yaw around a level base moves the point horizontally only.
        _, v = rigid_sensor_state(p, q, torch.zeros(1, 3), torch.tensor([[0., 0., .8]]), r)
        self.assertEqual(float(v[0, 0, 2]), 0.)

    def test_symmetric_virtual_rig_cost(self):
        reflection = np.diag([1., -1., 1.])
        rotation = Rotation.from_euler('xyz', [.2, -.1, .4]).as_matrix()
        other = reflection @ rotation @ reflection
        quats = torch.tensor(Rotation.from_matrix(np.stack([rotation, other])).as_quat()[:, [3, 0, 1, 2]], dtype=torch.float)
        vel = torch.tensor([[.4, .2, .1], [.4, -.2, .1]])
        ang = torch.tensor([[.6, -.2, .8], [-.6, -.2, -.8]])
        offsets = torch.tensor([[.33881, .0475, .111], [.33881, -.0475, .111]])
        _, v = rigid_sensor_state(torch.zeros(2, 3), quats, vel, ang, offsets)
        torch.testing.assert_close(v[:, :, 2].square().mean(1)[0], v[:, :, 2].square().mean(1)[1])

    def test_sustained_schedule(self):
        schedule = sensor_schedule()
        self.assertEqual(len(schedule), 8)
        self.assertEqual(schedule['stand'], [(30, (0., 0., 0.))])
        for name, phases in schedule.items():
            if name != 'stand':
                self.assertEqual([x[0] for x in phases], [3, 30, 8])
                self.assertEqual(phases[-1][1], (0., 0., 0.))

    def test_sensor_reward_dt_once_and_signed_sum(self):
        cfg, train = task_registry.get_cfgs('go2w')
        apply_go2w_profile(cfg, train, 'transfer_v3')
        cfg.rewards.scales.lin_vel_z = 0.
        cfg.rewards.scales.sensor_vertical_velocity = -1.
        e = Go2WEnv.__new__(Go2WEnv)
        e.cfg, e.num_envs, e.dt, e.device = cfg, 1, .02, 'cpu'
        e.reward_scales = class_to_dict(cfg.rewards.scales)
        e._prepare_reward_function()
        self.assertNotIn('lin_vel_z', e.reward_names)
        self.assertEqual(e.reward_scales['sensor_vertical_velocity'], -.02)
        self.assertEqual(e.reward_scales['termination'], -5.)
        e.sensor_vz_squared = torch.tensor([.04])
        e.reward_names = ['sensor_vertical_velocity']
        e.reward_functions = [e._reward_sensor_vertical_velocity]
        e.rew_buf = torch.zeros(1)
        e.reset_buf, e.time_out_buf = torch.tensor([False]), torch.tensor([False])
        e.event_step = False
        e.compute_reward()
        torch.testing.assert_close(e.rew_buf, torch.tensor([-.0008]))

    def test_saved_parent_and_selective_initialization_recipe(self):
        parent, parent_train = task_registry.get_cfgs('go2w')
        apply_go2w_profile(parent, parent_train, 'transfer_v3')
        parent.commands.short_command_duration_range = [.4, .9]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path/'config.yaml').write_text(yaml.safe_dump({'task': 'go2w',
                'env_cfg': class_to_dict(parent), 'train_cfg': class_to_dict(parent_train)}))
            torch.save({'iter': 42}, path/'model_42.pt')
            argv = ['train', '--task', 'go2w', '--go2w_profile', 'transfer_v3',
                    '--go2w_finetune', 'sensor_smooth', '--load_run', str(path), '--checkpoint', '-1']
            with patch.object(sys, 'argv', argv):
                args = get_args()
            cfg, train = task_registry.get_cfgs('go2w')
            with patch.object(task_registry, 'resolve_replay', wraps=task_registry.resolve_replay) as resolve:
                update_cfg_from_args(cfg, train, args)
                update_cfg_from_args(None, train, args)
                self.assertEqual(resolve.call_count, 1)
            validate_training(args, cfg, train)
            for key in ('asset', 'control', 'domain_rand', 'sim', 'noise', 'init_state', 'phase_guidance'):
                self.assertEqual(json.loads(json.dumps(class_to_dict(getattr(cfg, key)))),
                                 json.loads(json.dumps(class_to_dict(getattr(parent, key)))))
            self.assertEqual(cfg.commands.short_command_duration_range, [.4, .9])
            self.assertEqual((cfg.env.num_observations, cfg.env.num_actions), (58, 16))
            self.assertEqual(train.runner.checkpoint_load_cfg,
                             {'actor': True, 'critic': True, 'optimizer': False, 'iteration': False})
            self.assertEqual((train.algorithm.learning_rate, train.algorithm.schedule), (5e-5, 'fixed'))
            self.assertEqual((train.runner.max_iterations, train.runner.save_interval), (300, 50))
            self.assertEqual(cfg.refinement_parent['prior_updates'], 43)
            args.resume = True
            with self.assertRaises(ValueError):
                update_cfg_from_args(cfg, train, args)

    def test_sampler_dispatch_isolation_and_soft_hip_cost(self):
        cfg, train = task_registry.get_cfgs('go2w')
        apply_go2w_profile(cfg, train, 'transfer_v3')
        cfg.go2w_finetune = 'sensor_smooth'
        cfg.sensor_smooth = {'hip_weight': 3., 'long_hold_probability': .04, 'extended_hold_probability': .015,
                             'long_hold_s': [8., 15.], 'extended_hold_s': [20., 30.]}
        e = sampler(cfg, 20000)
        self.assertFalse(hasattr(e, 'diagnostic_long_moving_commands'))
        torch.manual_seed(1)
        e._resample_commands(torch.arange(e.num_envs))
        family = e.diagnostic_command_families
        self.assertTrue((e.sensor_hold_kind[family > 2] == 0).all())
        self.assertGreater(int((e.sensor_hold_kind == 2).sum()), 70)
        self.assertTrue(((e.command_steps_left[e.sensor_hold_kind == 2] >= 1000)
                         & (e.command_steps_left[e.sensor_hold_kind == 2] <= 1500)).all())
        # sensor_smooth must never activate legacy mixed-zero-yaw sampling.
        self.assertLess(float((e.commands[family == 6, 2] == 0).float().mean()), .06)
        before = (e.commands.clone(), e.command_steps_left.clone(), e.sensor_hold_kind.clone())
        e._resample_commands(torch.arange(100))
        for old, new in zip(before, (e.commands, e.command_steps_left, e.sensor_hold_kind)):
            torch.testing.assert_close(old[100:], new[100:])
        e.default_dof_pos = torch.zeros(1, 16)
        e.dof_pos = torch.full((e.num_envs, 16), .1)
        e.rolling_pose_weights = torch.tensor([3. if e.joint_names[i].endswith('_hip_joint') else 1.
                                             for i in e.leg_action_indices])
        e.commands.zero_()
        torch.testing.assert_close(e._reward_rolling_pose(), torch.full((e.num_envs,), .01*20/12))
        e.commands[:, 1] = .3
        self.assertEqual(float(e._reward_rolling_pose().sum()), 0.)


if __name__ == '__main__':
    unittest.main()
