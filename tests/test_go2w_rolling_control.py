"""Conditional V3 yaw precision; no simulation or policy modification."""
from types import SimpleNamespace
import unittest
import torch

from robot_gym.envs.go2w.go2w_env import Go2WEnv
from robot_gym.envs.go2w.phase import huber, rolling_command_mask


class RollingControl(unittest.TestCase):
    def setUp(self):
        env = Go2WEnv.__new__(Go2WEnv)
        env.commands = torch.tensor([[0., 0., 0.], [.5, 0., 0.], [-.2, 0., 0.],
            [0., .03, 0.], [0., 0., .2], [0., 0., -.2], [0., 1e-6, 0.], [0., .9e-6, 0.]])
        env.base_lin_vel = torch.full_like(env.commands, 2.)  # Measured motion does not choose the gate.
        env.base_ang_vel = torch.zeros_like(env.commands)
        env.base_ang_vel[:, 2] = torch.tensor([.02, .04, -.03, .02, .2, -.17, .04, .04])
        env.cfg = SimpleNamespace(commands=SimpleNamespace(stand_threshold=1e-6),
            rewards=SimpleNamespace(phase_objective={'tracking_scales': [.25, .15, .35]},
                                    only_positive_rewards=False))
        env.num_envs, env.device, env.dt = len(env.commands), 'cpu', .02
        env.rew_buf = torch.zeros(env.num_envs)
        env.event_step = False
        self.env = env

    def test_old_config_and_conditional_commanded_rate(self):
        env = self.env
        error = env.commands[:, 2] - env.base_ang_vel[:, 2]
        old = env._reward_tracking_yaw()
        torch.testing.assert_close(old, 1-huber(error/.35), rtol=0, atol=0)
        xy = [env._tracking_axis(i).clone() for i in (0, 1)]
        env.cfg.rewards.phase_objective['rolling_yaw_scale'] = .07
        torch.testing.assert_close(rolling_command_mask(env.commands, 1e-6),
                                   torch.tensor([True, True, True, False, False, False, False, True]))
        expected = 1-huber(error/torch.tensor([.07, .07, .07, .35, .35, .35, .35, .07]))
        torch.testing.assert_close(env._reward_tracking_yaw(), expected)
        self.assertEqual(float(expected[4]), 1.)  # Intentional +.2 yaw is tracked, not replaced with zero.
        for i in (0, 1): torch.testing.assert_close(env._tracking_axis(i), xy[i], rtol=0, atol=0)
        env.commands *= torch.tensor([1., -1., -1.])
        env.base_ang_vel *= torch.tensor([-1., 1., -1.])
        torch.testing.assert_close(env._reward_tracking_yaw(), expected)
        env.cfg.rewards.phase_objective.pop('rolling_yaw_scale')
        torch.testing.assert_close(env._reward_tracking_yaw(), old)

    def test_native_registration_weights_both_terms_once(self):
        env = self.env
        env.cfg.rewards.phase_objective['rolling_yaw_scale'] = .07
        env.cfg.rewards.rolling_placement = dict(width_deadband_m=.04, width_scale_m=.10,
                                                midpoint_deadband_m=.015, midpoint_scale_m=.05)
        env.placement_reference = torch.tensor([[.3802, .3802], [0., 0.]])
        env.wheel_center_body = torch.zeros(env.num_envs, 4, 3)
        env.wheel_center_body[:, :, 1] = torch.tensor([.28, -.24, .20, -.18])
        raw_yaw, raw_placement = env._reward_tracking_yaw(), env._reward_rolling_placement()
        env.reward_scales = dict(tracking_yaw=.8, rolling_placement=-.05)
        env._prepare_reward_function()
        env.compute_reward()
        torch.testing.assert_close(env.rew_buf, .02*(.8*raw_yaw-.05*raw_placement))
        torch.testing.assert_close(env.episode_sums['tracking_yaw'], .02*.8*raw_yaw)
        torch.testing.assert_close(env.episode_sums['rolling_placement'], -.02*.05*raw_placement)
        self.assertEqual(env.reward_names.count('tracking_yaw'), 1)
        self.assertEqual(env.reward_names.count('rolling_placement'), 1)


if __name__ == '__main__':
    unittest.main()
