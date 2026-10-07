"""Focused geometric task, transition, critic and export contracts; no simulator."""

import copy
import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.algorithms import PPO

from robot_gym.envs.base.legged_robot import LeggedRobot
from robot_gym.envs.go2.go2_env import Go2Env
from robot_gym.envs.go2w.go2w_config import GO2WCfg, GO2WCfgPPO, restore_saved_config
from robot_gym.envs.go2w.go2w_env import Go2WEnv
from robot_gym.envs.go2w.go2w_symmetry import mirror_observations
from robot_gym.envs.go2w.straight_motion import StraightMotionReference, projected_heading, wrap_angle
from robot_gym.utils.helpers import class_to_dict
from tests.test_go2w_current_config import sampler_fixture


def quaternion(yaw):
    yaw = torch.as_tensor(yaw)
    return torch.stack(((yaw / 2).cos(), torch.zeros_like(yaw), torch.zeros_like(yaw),
                        (yaw / 2).sin()), dim=-1)


def reference(commands, heading=0.):
    commands = torch.tensor(commands, dtype=torch.float32)
    n = len(commands)
    ref = StraightMotionReference(n, 'cpu', GO2WCfg().straight_motion)
    p = torch.zeros(n, 3)
    q = quaternion(torch.full((n,), heading))
    ref.set_command(commands, p, q, torch.arange(n))
    return ref, p, q, commands


class StraightMotionTests(unittest.TestCase):
    def test_ideal_lines_and_arbitrary_along_line_lag(self):
        commands = [[.5, 0, 0], [-.2, 0, 0], [0, .3, 0], [0, -.3, 0], [.2, .3, 0]]
        for progress in (0., -3., .1, 5.):
            ref, p, q, cmd = reference(commands)
            p[:, :2] = cmd[:, :2] * progress
            ref.advance(p, q, 4.)
            torch.testing.assert_close(ref.cross_track_cost(), torch.zeros(5), atol=1e-12, rtol=0)
            torch.testing.assert_close(ref.heading_cost(), torch.zeros(5))
        # The line objective alone intentionally cannot enforce requested progress.

    def test_error_scales_ramp_and_unbounded_broad_sensitivity(self):
        ref, p, q, _ = reference([[.5, 0, 0]] * 4)
        p[:, 1] = torch.tensor([.005, .05, .2, 2.])
        q = quaternion(torch.tensor([0., .035, .2, 1.]))
        ref.advance(p, q, .25)
        torch.testing.assert_close(ref.activation(), torch.full((4,), .5))
        self.assertEqual(float(ref.cross_track_cost()[0]), 0.)
        self.assertTrue((torch.diff(ref.cross_track_cost()) > 0).all())
        self.assertTrue((torch.diff(ref.heading_cost()) > 0).all())
        half = ref.cross_track_cost().clone()
        ref.advance(p, q, .25)
        torch.testing.assert_close(ref.cross_track_cost(), 2 * half)
        self.assertGreater(float(ref.cross_track_cost()[-1]), 10.)

    def test_translation_rotation_invariance_and_reflection_all_directions(self):
        commands = [[.5, 0, 0], [-.2, 0, 0], [0, .3, 0], [0, -.3, 0], [.2, -.3, 0]]
        ref, p, q, cmd = reference(commands, heading=.7)
        end = torch.tensor([[.3, .2, .4]]).repeat(5, 1)
        end_q = quaternion(torch.full((5,), .9))
        ref.advance(end, end_q, 2.)
        angle, offset = 1.2, torch.tensor([23., -18., 4.])
        rotation = torch.tensor([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        shifted = StraightMotionReference(5, 'cpu', ref.cfg)
        p2 = p + offset
        shifted.set_command(cmd, p2, quaternion(torch.full((5,), .7 + angle)), torch.arange(5))
        end2 = end.clone()
        end2[:, :2] = end[:, :2] @ rotation.T
        shifted.advance(end2 + offset, quaternion(torch.full((5,), .9 + angle)), 2.)
        torch.testing.assert_close(shifted.cross_track, ref.cross_track, atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(shifted.heading_error, ref.heading_error)
        reflected, _, _, _ = reference((cmd * torch.tensor([1, -1, -1])).tolist(), heading=-.7)
        reflected.advance(end * torch.tensor([1, -1, 1]), quaternion(torch.full((5,), -.9)), 2.)
        a = ref.critic_features(torch.full((5,), 100), True, .02)
        b = reflected.critic_features(torch.full((5,), 100), True, .02)
        signs = torch.tensor([-1, -1, 1, 1, 1, 1, 1])
        torch.testing.assert_close(b, a * signs)
        env = sampler_fixture(5)
        env.num_obs, env.num_actions = 58, 16
        env.joint_names = env.cfg.asset.joint_names
        env.leg_action_indices = [i for i, n in enumerate(env.joint_names) if 'foot' not in n]
        policy = torch.randn(5, 58)
        obs = TensorDict(dict(policy=policy, critic=torch.cat((policy, a), -1)), [5])
        mirror = mirror_observations(env, obs)
        torch.testing.assert_close(mirror['critic'][:, 58:], b)
        torch.testing.assert_close(mirror['critic'][:, :58], mirror['policy'])
        torch.testing.assert_close(mirror['policy'][:, 56:58], -policy[:, 56:58])
        torch.testing.assert_close(mirror_observations(env, mirror), obs)

    def test_numerical_zero_small_yaw_and_angle_wrapping(self):
        ref, p, q, _ = reference([[0, 0, 0], [1e-12, 0, 0], [.02, 0, 0],
                                  [.5, 0, .00001], [.5, 0, .05], [0, 0, .1]])
        self.assertEqual(ref.valid.tolist(), [False, False, True, False, False, False])
        ref.advance(p, q, .02)
        self.assertTrue(torch.isfinite(ref.critic_features(torch.zeros(6), True, .02)).all())
        torch.testing.assert_close(wrap_angle(torch.tensor([-math.pi-.01, math.pi+.01])),
                                   torch.tensor([math.pi-.01, -math.pi+.01]))
        ref, p, q, _ = reference([[.5, 0, 0]], math.pi-.01)
        ref.advance(p, quaternion(torch.tensor([-math.pi+.01])), .02)
        self.assertAlmostEqual(float(ref.heading_error), .02, places=5)
        # Projected full-quaternion forward axis: tilted body angular-z is not used.
        from scipy.spatial.transform import Rotation
        r = Rotation.from_euler('xyz', [.3, -.2, .7])
        h, _ = projected_heading(torch.tensor(r.as_quat()[[3, 0, 1, 2]]))
        self.assertAlmostEqual(float(h), .7)

    def test_identical_messages_and_getters_keep_anchor_speed_change_relatches(self):
        env = sampler_fixture(2)
        env.compute_observations = Mock()
        env.set_fixed_command([.5, 0, 0])
        env.base_pos[:, 1] = .2
        env.straight_reference.advance(env.base_pos, env.base_quat, 2.)
        original = env.straight_reference.origin.clone()
        for _ in range(3):
            env.set_fixed_command([.5, 0, 0])
            env.straight_reference.critic_features(env.command_steps_left, False, env.dt)
        torch.testing.assert_close(env.straight_reference.origin, original)
        torch.testing.assert_close(env.straight_reference.age, torch.full((2,), 2.))
        env.set_fixed_command([.6, 0, 0])
        torch.testing.assert_close(env.straight_reference.origin, env.base_pos[:, :2])
        torch.testing.assert_close(env.straight_reference.age, torch.zeros(2))
        self.assertTrue((env.straight_reference.critic_features(env.command_steps_left, False, .02)[:, -1] == -1).all())

    def test_partial_reset_uses_new_simulator_pose_and_leaves_other_reference(self):
        env = sampler_fixture(2)
        env.commands[:] = torch.tensor([.5, 0, 0])
        env.straight_reference.set_command(env.commands, env.base_pos, env.base_quat, env.all_env_ids)
        env.base_pos[:, 1] = .2
        env.straight_reference.advance(env.base_pos, env.base_quat, 2.)
        env.phase = torch.zeros(2)
        for name in ('wheel_clearance', 'wheel_normal_force', 'loaded_wheels',
                     'wheel_reposition_velocity_body', 'wheel_center_lateral_speed'):
            setattr(env, name, torch.zeros(2, 4))
        env.robot = Mock()
        env.robot.get_pos.return_value = torch.tensor([[7., 4., .43], [0., .2, .43]])
        env.robot.get_quat.return_value = env.base_quat.clone()
        with patch.object(Go2Env, 'reset_idx'):
            env.reset_idx(torch.tensor([0]))
        torch.testing.assert_close(env.straight_reference.origin, torch.tensor([[7., 4.], [0., 0.]]))
        torch.testing.assert_close(env.straight_reference.age, torch.tensor([0., 2.]))
        torch.testing.assert_close(env.straight_reference.cross_track, torch.tensor([0., .2]))

    def test_fixed_replay_and_evaluator_batch_changes_share_reference_semantics(self):
        fixed, evaluation = sampler_fixture(2), sampler_fixture(2)
        fixed.compute_observations = Mock()
        for command in ([.5, 0, 0], [.5, 0, 0], [.7, 0, 0], [0, -.3, 0], [0, -.3, .001], [0, 0, 0]):
            for env in (fixed, evaluation):
                env.base_pos[:, :2] += torch.tensor([.1, -.02])
                env.straight_reference.advance(env.base_pos, env.base_quat, .02)
            fixed.set_fixed_command(command)
            evaluation.set_commands(command)
            for key in ('origin', 'normal', 'heading', 'age', 'valid', 'cross_track', 'heading_error'):
                torch.testing.assert_close(getattr(fixed.straight_reference, key),
                                           getattr(evaluation.straight_reference, key))

    def test_score_then_change_command_and_native_timeout_bootstrap(self):
        env = sampler_fixture(1)
        env.commands[:] = torch.tensor([.5, 0, 0])
        env.straight_reference.set_command(env.commands, env.base_pos, env.base_quat, env.all_env_ids)
        env.base_pos[:, 1] = .2
        env.command_steps_left[:] = 1
        env.episode_length_buf = torch.zeros(1, dtype=torch.long)
        env.common_step_counter = 0
        env.reset_buf = torch.zeros(1, dtype=torch.bool)
        env.time_out_buf = torch.zeros(1, dtype=torch.bool)
        env.phase = torch.zeros(1)
        env.headless = True
        env.actions = env.last_actions = env.dof_vel = env.last_dof_vel = torch.zeros(1, 16)
        env.extras = {}
        env._update_robot_state = env.check_termination = env.reset_idx = Mock()
        env.update_task_state = lambda: env.straight_reference.advance(env.base_pos, env.base_quat, env.dt)
        recorded = []
        env.compute_reward = lambda: recorded.append(env.straight_reference.cross_track.clone())
        def change(ids):
            env.commands[:] = torch.tensor([0., .3, 0])
            env.straight_reference.set_command(env.commands, env.base_pos, env.base_quat, ids)
        env._resample_commands = change
        env.compute_observations = lambda: recorded.append(env.straight_reference.cross_track.clone())
        LeggedRobot.post_physics_step(env)
        torch.testing.assert_close(recorded[0], torch.tensor([.2]))
        torch.testing.assert_close(recorded[1], torch.zeros(1))
        # Installed native timeout path uses the pre-step value, not reset obs.
        alg = PPO.__new__(PPO)
        alg.rnd, alg.device, alg.gamma = None, 'cpu', .995
        transition = SimpleNamespace(values=torch.tensor([[2.]]), clear=Mock())
        alg.transition = transition
        alg.storage, alg.actor, alg.critic = Mock(), Mock(), Mock()
        alg.process_env_step(TensorDict({'critic': torch.full((1, 65), 999.)}, [1]),
                             torch.tensor([.1]), torch.tensor([True]), {'time_outs': torch.tensor([True])})
        torch.testing.assert_close(transition.rewards, torch.tensor([.1 + .995*2]))

    def test_reward_registration_weights_dt_once(self):
        env = sampler_fixture(1)
        env.reward_scales = class_to_dict(env.cfg.rewards.scales)
        env._prepare_reward_function()
        self.assertAlmostEqual(env.reward_scales['tracking_yaw'], .016)
        for name in ('straight_cross_track', 'straight_heading'):
            self.assertAlmostEqual(env.reward_scales[name], -.002)
        self.assertEqual(env.reward_scales['termination'], -5.)

    def test_actor_normalizer_and_export_are_isolated_from_reference(self):
        torch.manual_seed(13)
        cfg = GO2WCfgPPO()
        policy = torch.randn(8, 58)
        obs = TensorDict({'policy': policy, 'critic': torch.cat((policy, torch.randn(8, 7)), -1)}, [8])
        opts = class_to_dict(cfg.actor); opts.pop('class_name')
        actor = MLPModel(obs, cfg.runner.obs_groups, 'actor', 16, **opts)
        opts = class_to_dict(cfg.critic); opts.pop('class_name')
        critic = MLPModel(obs, cfg.runner.obs_groups, 'critic', 1, **opts)
        self.assertEqual((actor.obs_dim, critic.obs_dim), (58, 65))
        self.assertEqual((int(actor.obs_normalizer.count), int(critic.obs_normalizer.count)), (0, 0))
        changed = obs.clone(); changed['critic'][:, 58:] += 5
        actor2 = copy.deepcopy(actor)
        actor.update_normalization(obs); actor2.update_normalization(changed)
        for key, value in actor.obs_normalizer.state_dict().items():
            torch.testing.assert_close(value, actor2.obs_normalizer.state_dict()[key], rtol=0, atol=0)
        actor.eval(); critic.eval()
        torch.testing.assert_close(actor(obs), actor(changed), rtol=0, atol=0)
        self.assertFalse(torch.allclose(critic(obs), critic(changed)))
        scripted = torch.jit.script(actor.as_jit().eval())
        torch.testing.assert_close(scripted(policy), actor(obs))
        self.assertEqual(actor.as_onnx(False).get_dummy_inputs()[0].shape, (1, 58))
        torch.testing.assert_close(actor.distribution.log_std_param.exp(), torch.full((16,), .4))

    def test_persistent_reference_has_no_push_or_rollout_dependency(self):
        ref, p, q, cmd = reference([[0, .3, 0]])
        for tick in range(128):  # Two nominal 64-tick boundaries, without relatching.
            p[:, 0] += .001
            ref.advance(p, q, .02)
            ref.critic_features(torch.tensor([500-tick]), True, .02)
            if tick in (20, 64):  # A disturbance changes pose, never the command/reference.
                p[:, 0] += .1
        torch.testing.assert_close(ref.origin, torch.zeros(1, 2))
        self.assertAlmostEqual(float(ref.age), 2.56, places=5)
        self.assertAlmostEqual(float(ref.cross_track), -.328, places=5)

    def test_saved_schema_is_explicit_and_complete(self):
        cfg, train = GO2WCfg(), GO2WCfgPPO()
        saved = dict(env_cfg=class_to_dict(cfg), train_cfg=class_to_dict(train))
        e, t = restore_saved_config(saved)
        self.assertEqual(class_to_dict(e), saved['env_cfg'])
        self.assertEqual(class_to_dict(t), saved['train_cfg'])
        old = copy.deepcopy(saved); old['env_cfg']['config_version'] = 1
        with self.assertRaisesRegex(ValueError, 'historical source checkout'):
            restore_saved_config(old)
        del saved['env_cfg']['straight_motion']['ramp_s']
        with self.assertRaisesRegex(ValueError, 'missing fields'):
            restore_saved_config(saved)


if __name__ == '__main__':
    unittest.main()
