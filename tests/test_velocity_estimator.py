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
from robot_gym.utils.velocity_estimator import inference_policy
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

    def test_canonical_layout_keeps_clean_critic_and_issued_action_history(self):
        from robot_gym.envs import task_registry
        for task, joints in (("dodo", 8), ("go2", 12)):
            with self.subTest(task=task):
                env = object.__new__(task_registry.get_task_class(task))
                env.cfg, _ = task_registry.get_cfgs(task)
                env.num_envs, env.num_actions, env.num_obs = 2, joints, 12 + 3*joints
                env.obs_scales = env.cfg.normalization.obs_scales
                env.commands_scale = torch.tensor([2., 2., .25])
                env.device, env.headless, env.dt = torch.device("cpu"), True, .02
                env.joint_names = [str(i) for i in range(joints)]
                spec = env.velocity_estimator_spec()
                env.sensor_history = SensorHistory(2, spec, default_config(), env.device)
                env.common_step_counter, env.add_noise = 1, True
                env.noise_scale_vec = torch.ones(env.num_obs)
                env.base_lin_vel = torch.tensor([[.2, -.4, .1], [-.6, .3, -.2]])
                env.base_ang_vel = env.base_lin_vel + 1
                env.projected_gravity = env.base_lin_vel + 2
                env.commands = env.base_lin_vel + 3
                env.dof_pos = torch.arange(2*joints).view(2, joints).float()
                env.default_dof_pos = torch.ones(1, joints)
                env.dof_pos_obs_mask = torch.arange(joints) % 2 == 0
                env.dof_vel, env.actions = env.dof_pos + 4, env.dof_pos / 4
                env.applied_actions = torch.full_like(env.actions, -19.)
                env.compute_observations()
                obs = env.get_observations()
                clean = torch.cat((env.base_lin_vel * env.obs_scales.lin_vel,
                    env.base_ang_vel * env.obs_scales.ang_vel, env.projected_gravity,
                    env.commands * env.commands_scale,
                    (env.dof_pos - 1) * env.dof_pos_obs_mask * env.obs_scales.dof_pos,
                    env.dof_vel * env.obs_scales.dof_vel, env.actions), dim=1)
                torch.testing.assert_close(obs["critic"], clean, rtol=0., atol=0.)
                torch.testing.assert_close(obs["velocity_target"], env.base_lin_vel)
                self.assertFalse(torch.equal(obs["critic"][:, 3:6], obs["policy"][:, 3:6]))
                history = obs["estimator_history"]
                torch.testing.assert_close(history[:, -1], obs["policy"][:, 3:])
                torch.testing.assert_close(history[:, -1, -joints:], env.actions)
                self.assertEqual(spec["live_feature_indices"], [6, 7, 8, *range(env.num_obs-3-joints, env.num_obs-3)])
                self.assertEqual(spec["feature_indices"], list(range(3, env.num_obs)))
                self.assertEqual(history.shape, (2, 5, env.num_obs-3))

    def test_generic_evaluation_uses_signed_task_ranges(self):
        from robot_gym.utils.estimator_evaluation import command_sequence
        ranges = SimpleNamespace(lin_vel_x=(-.2, 1.), lin_vel_y=(0., .04), ang_vel_yaw=(-1., .6))
        self.assertEqual(command_sequence(ranges), [(0., 0., 0.), (-.1, 0., 0.), (.5, 0., 0.),
                         (0., .02, 0.), (0., 0., -.5), (0., 0., .3), (0., 0., 0.)])

    def test_estimator_init_restores_parent_then_persists_critic_width(self):
        from unittest.mock import patch
        from robot_gym.envs import task_registry
        from robot_gym.utils.helpers import get_args, update_cfg_from_args
        import yaml
        root = Path(__file__).resolve().parents[1] / "ressources" / "pretrained"
        for task in ("dodo", "go2"):
            run = root / task / f"{task}_walking"
            with self.subTest(task=task), patch("sys.argv", ["train", "--task", task, "--resume",
                    "--load_run", str(run), "--checkpoint", "1499", "--estimator_init"]):
                args = get_args()
                cfg, training = task_registry.get_cfgs(task)
                update_cfg_from_args(cfg, training, args)
                update_cfg_from_args(cfg, None, args)  # Environment construction repeats CLI overrides.
                self.assertEqual(cfg.env.num_privileged_obs, cfg.env.num_observations)
                self.assertEqual(training.runner.obs_groups, {"actor": ["policy"], "critic": ["critic"]})
                self.assertTrue(enabled(cfg))
                with self.assertRaisesRegex(ValueError, "training-only"):
                    task_registry.resolve_replay(args)
                with TemporaryDirectory() as directory:
                    saved = dict(task=task, env_cfg=class_to_dict(cfg), train_cfg=class_to_dict(training))
                    source = Path(directory) / "config.yaml"
                    (Path(directory) / "model_1499.pt").touch()
                    args.load_run, args.estimator_init, args.resume = directory, False, False
                    def save():
                        source.write_text(yaml.safe_dump(task_registry._make_yaml_safe(saved)))
                    save()
                    loaded, _, _ = task_registry.resolve_replay(args)
                    self.assertEqual(loaded.env.num_privileged_obs, cfg.env.num_observations)
                    saved["train_cfg"]["runner"]["obs_groups"]["critic"] = ["policy"]
                    save()
                    with self.assertRaisesRegex(ValueError, "clean critic groups"):
                        task_registry.resolve_replay(args)
                    saved["env_cfg"]["velocity_estimator"] = None
                    save()
                    with self.assertRaisesRegex(ValueError, "num_privileged_obs"):
                        task_registry.resolve_replay(args)

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
        unchanged = {key: getattr(h, key).clone() for key in ('history', 'age', 'delay', 'held', 'bias')}
        h.reset(torch.tensor([1]))
        h.observe(clean, noise, 1)  # Selective precision_dr reset at the same tick.
        for key, value in unchanged.items():
            torch.testing.assert_close(getattr(h, key)[[0, 2, 3]], value[[0, 2, 3]], rtol=0., atol=0.)
        torch.testing.assert_close(h.history[1], h.history[1, -1:].expand(5, 55))
        self.assertTrue(h.valid.all())
        rng = torch.random.get_rng_state()
        h.observe(clean, noise, 1)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
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
        env.cfg.velocity_estimator = default_config()
        env.device = torch.device('cpu')
        def privileged_path(**kwargs):
            raise AssertionError('Estimated dispatch called native actor-only inference')
        runner = SimpleNamespace(alg=alg, get_inference_policy=privileged_path)
        model = inference_policy(runner, env, 'estimated')
        a, v = model(obs), env.estimated_velocity.clone()
        obs['policy'].fill_(999)
        obs['critic'].fill_(999)
        obs['velocity_target'].fill_(999)
        b, w = model(obs), env.estimated_velocity.clone()
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(v, w)
        self.assertLessEqual(float(a.detach().abs().max()), 1.)
        from robot_gym.envs.go2w.evaluation.evaluate import validate_velocity_mode
        for mode in ('bank', 'equilibrium', 'precision_dr', 'transfer_screen'):
            self.assertEqual(validate_velocity_mode(env.cfg, SimpleNamespace(eval_mode=mode)), 'estimated')
        with self.assertRaisesRegex(ValueError, 'phase counterfactuals'):
            validate_velocity_mode(env.cfg, SimpleNamespace(eval_mode='sensor_sustained'))

    def test_stop_metric_definitions_and_reset_censoring(self):
        from robot_gym.envs.go2w.evaluation.estimator import stop_metrics
        data = dict(velocity=np.ones((100, 1, 3)), angular=np.zeros((100, 1, 3)),
            rpy=np.zeros((100, 1, 3)), position=np.zeros((100, 1, 3)), valid=np.ones((100, 1), dtype=bool))
        data['rpy'][:, 0, 2] = (np.arange(100)*.1 + np.pi) % (2*np.pi) - np.pi
        row = stop_metrics(data, 0, 10, 100, .02)
        self.assertEqual(row['final_component_drift_rms_m_s'], 1.)
        self.assertAlmostEqual(row['final_planar_speed_rms_m_s'], 2**.5)
        self.assertAlmostEqual(row['heading_change_rad'], 9.)
        self.assertIsNone(row['settled_after_s'])
        data['valid'][60:] = False
        self.assertEqual(stop_metrics(data, 0, 10, 100, .02), {'censored': True})

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
        with self.assertRaisesRegex(ValueError, 'full parent learning state'):
            reloaded.load(actor_only, {"actor": True})
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
        state.refresh(frame+2)
        np.testing.assert_array_equal(state.values[[0, 2], :-1], before[[0, 2], :-1])
        np.testing.assert_array_equal(state.values[1], np.repeat((frame+2)[1:2], 5, axis=0))
        state.reset([1])
        state.append(frame+3)
        np.testing.assert_array_equal(state.values[1], np.repeat((frame+3)[1:2], 5, axis=0))


if __name__ == '__main__':
    unittest.main()
