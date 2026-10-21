"""Focused causal-history, native PPO and deployment-contract regressions."""

import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO

from robot_gym.envs.go2w.go2w_config import GO2WCfg, GO2WCfgPPO, restore_saved_config, joint_reference
from robot_gym.envs.go2w.go2w_env import Go2WEnv
from robot_gym.envs.go2w.go2w_symmetry import mirror_observations
from robot_gym.utils.helpers import class_to_dict
from robot_gym.utils.velocity_estimator import SensorHistory, default_config, enabled, EstimatedPolicy
from robot_gym.utils.estimator_ppo import assert_frozen_parent


def fake_env(n=4):
    names = list(joint_reference())
    env = SimpleNamespace(num_envs=n, num_actions=16, num_obs=58, cfg=GO2WCfg(), joint_names=names,
        leg_action_indices=[i for i, name in enumerate(names) if not name.endswith('_foot_joint')],
        obs_scales=SimpleNamespace(lin_vel=2., ang_vel=.25, dof_pos=1., dof_vel=.05),
        commands_scale=torch.tensor([2., 2., .25]), dt=.02)
    env.spec = Go2WEnv.velocity_estimator_spec(env)
    return env


def algorithm(stage='warmup'):
    env = fake_env()
    c = default_config()
    c.update(stage=stage, hidden_dims=[16], gradient_steps=1, batch_size=8)
    obs = TensorDict(dict(policy=torch.randn(4, 58), critic=torch.randn(4, 65),
        estimator_history=torch.randn(4, 5, 55), velocity_target=torch.randn(4, 3)), batch_size=[4])
    cfg = class_to_dict(GO2WCfgPPO())
    cfg.update(cfg.pop('runner'))
    cfg.update(multi_gpu=None, num_steps_per_env=2)
    cfg['actor']['hidden_dims'] = cfg['critic']['hidden_dims'] = [16]
    cfg['algorithm'].update(class_name='robot_gym.utils.estimator_ppo:EstimatorPPO',
        estimator_config=c, estimator_spec=env.spec, observation_limit=100.,
        num_learning_epochs=1, num_mini_batches=1)
    alg = PPO.construct_algorithm(obs, env, cfg, 'cpu')
    alg.train_mode()
    return env, obs, alg


class VelocityEstimatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_old_saved_config_stays_disabled(self):
        saved = dict(env_cfg=class_to_dict(GO2WCfg()), train_cfg=class_to_dict(GO2WCfgPPO()))
        saved['env_cfg'].pop('velocity_estimator')
        cfg, _ = restore_saved_config(saved)
        self.assertFalse(enabled(cfg))

    def test_history_getter_reset_delay_and_controller_refresh(self):
        env = fake_env()
        cfg = default_config()
        cfg['sensor_delay_steps'] = 1
        h = SensorHistory(4, env.spec, cfg, 'cpu')
        clean = torch.arange(232.).reshape(4, 58)
        noise = torch.ones(58) * .1
        h.observe(clean, noise, 1)
        first = h.history.clone()
        rng = torch.random.get_rng_state()
        h.observe(clean, noise, 1)
        torch.testing.assert_close(h.history, first)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        clean[:, 9] = 900.
        h.observe(clean, noise, 1)
        torch.testing.assert_close(h.history[:, :-1], first[:, :-1])
        self.assertTrue((h.history[:, -1, 6] == 900).all())
        h.reset(torch.tensor([1]))
        h.observe(clean + 20, noise * 0, 2)
        torch.testing.assert_close(h.history[1], h.history[1, -1:].expand(5, 55))
        torch.testing.assert_close(h.history[0, -1, :6], first[0, -1, :6])
        self.assertFalse(torch.equal(h.history[0, 0], h.history[1, 0]))

    def test_no_privileged_leak_and_full_history_symmetry(self):
        env, obs, alg = algorithm()
        mirrored = mirror_observations(env, obs)
        torch.testing.assert_close(mirror_observations(env, mirrored), obs)
        torch.testing.assert_close(mirrored['velocity_target'], obs['velocity_target'] * torch.tensor([1, -1, 1]))
        torch.testing.assert_close(mirrored['estimator_history'][..., :3],
                                   obs['estimator_history'][..., :3] * torch.tensor([-1, 1, -1]))
        model = EstimatedPolicy(alg.velocity_input, alg.actor.as_jit(), 1.).eval()
        a, v = model(obs['estimator_history'])
        obs['policy'].fill_(999)
        obs['critic'].fill_(999)
        obs['velocity_target'].fill_(999)
        b, w = model(obs['estimator_history'])
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(v, w)
        self.assertLessEqual(float(a.detach().abs().max()), 1.)

    def test_frozen_warmup_and_full_reload(self):
        _, obs, alg = algorithm()
        before = copy.deepcopy(alg.save())
        with torch.inference_mode():
            alg.act(obs)
            self.assertTrue(torch.isfinite(alg.actor.output_std).all())
            expected = alg.samples[0]['velocity_target'].clone()
            obs['velocity_target'].fill_(100)
            torch.testing.assert_close(alg.samples[0]['velocity_target'], expected)
        alg.update()
        assert_frozen_parent(before, alg.save())
        self.assertEqual(alg.stage_counts['warmup_collections'], 1)
        _, _, reloaded = algorithm()
        reloaded.load(alg.save())
        torch.testing.assert_close(alg.estimator(obs['estimator_history']), reloaded.estimator(obs['estimator_history']))
        self.assertEqual(reloaded.stage_counts, alg.stage_counts)
        self.assertTrue(reloaded.estimator_optimizer.state)
        _, _, changed_stage = algorithm('finetune')
        with self.assertRaisesRegex(ValueError, 'explicit --estimator_stage'):
            changed_stage.load(alg.save())
        changed_stage.requested_stage = 'finetune'
        changed_stage.load(alg.save())
        changed_stage.start_stage()
        self.assertEqual(changed_stage.learning_rate, 3e-5)
        self.assertEqual(changed_stage.stage_counts['warmup_collections'], 1)
        actor_only = {k: v for k, v in before.items() if k != 'velocity_estimator'}
        with self.assertRaisesRegex(ValueError, 'Actor-only checkpoint'):
            reloaded.load(actor_only)

    def test_native_ppo_stores_effective_inputs_and_recomputes_after_update(self):
        _, obs, alg = algorithm('finetune')
        estimator_before = copy.deepcopy(alg.estimator.state_dict())
        with torch.inference_mode():
            expected, _ = alg.velocity_input(obs['estimator_history'])
            for _ in range(2):
                alg.act(obs)
                stored = alg.transition.observations['policy'].clone()
                torch.testing.assert_close(stored, expected)
                obs['policy'].fill_(99.)
                torch.testing.assert_close(alg.transition.observations['policy'], stored)
                alg.process_env_step(obs, torch.ones(4), torch.zeros(4), {})
            alg.compute_returns(obs)
        alg.update()
        self.assertEqual(alg.stage_counts['finetune_updates'], 1)
        self.assertFalse(torch.equal(estimator_before['network.0.weight'], alg.estimator.state_dict()['network.0.weight']))
        with torch.inference_mode():
            alg.act(obs)  # Same runner-retained history, NEW estimate.
            new, _ = alg.velocity_input(obs['estimator_history'])
            torch.testing.assert_close(alg.transition.observations['policy'], new)

    def test_export_graph_onnx_and_torchscript(self):
        from robot_gym.utils.export import validate_onnx
        _, obs, alg = algorithm()
        model = EstimatedPolicy(alg.velocity_input, alg.actor.as_jit(), 1.).eval()
        with TemporaryDirectory() as folder:
            filename = Path(folder) / 'policy.onnx'
            history = obs['estimator_history']
            torch.onnx.export(model, (history,), str(filename), input_names=['history'],
                output_names=['clipped_actions', 'velocity_m_s'], dynamo=False, opset_version=17,
                dynamic_axes={'history': {0: 'batch'}, 'clipped_actions': {0: 'batch'}, 'velocity_m_s': {0: 'batch'}})
            report = validate_onnx(model, filename, history.numpy())
            self.assertEqual(report['samples'], 4)
            torch.testing.assert_close(torch.jit.script(model)(history), model(history))

    def test_portable_clock_and_history(self):
        from robot_gym.utils.deployment_history import PolicyHistory, phase_features
        from robot_gym.envs.go2w.phase import clock_observation
        commands = np.array([[0., .02, .14], [0., -.3, -1.], [0., 0., 0.]], dtype=np.float32)
        phase = np.array([.1, .4, .7], dtype=np.float32)
        portable = phase_features(phase, commands, [1, 2], [[.01, .05], [.1, .25]])
        actual = clock_observation(torch.from_numpy(phase), torch.from_numpy(commands), 'command_demand')
        np.testing.assert_allclose(portable, actual.numpy(), atol=2e-7, rtol=0.)
        state = PolicyHistory(3, 5, 4, [3])
        frame = np.arange(12, dtype=np.float32).reshape(3, 4)
        state.append(frame)
        np.testing.assert_array_equal(state.values, np.repeat(frame[:, None], 5, axis=1))
        state.append(frame+1)
        before = state.values.copy()
        state.refresh(frame+2)
        np.testing.assert_array_equal(before[:, :-1], state.values[:, :-1])
        state.reset([1])
        state.append(frame+3)
        np.testing.assert_array_equal(state.values[1], np.repeat((frame+3)[1:2], 5, axis=0))


if __name__ == '__main__':
    unittest.main()
