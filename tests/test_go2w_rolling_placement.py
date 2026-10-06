"""Focused opt-in placement geometry, gating and saved-recipe checks."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml

from robot_gym.envs.go2w.go2w_config import apply_go2w_profile, v3_reference_geometry, validate_training
from robot_gym.envs.go2w.phase import rolling_placement_error
from robot_gym.utils import get_args, task_registry
from robot_gym.utils.helpers import class_to_dict, update_cfg_from_args


class RollingPlacement(unittest.TestCase):
    def test_narrow_crossed_wide_and_displaced_pairs(self):
        # Formula/geometry regression, not a statement about policy gradients.
        cfg = dict(width_deadband_m=.04, width_scale_m=.10,
                   midpoint_deadband_m=.015, midpoint_scale_m=.05)
        reference = torch.tensor([[.3802, .3802], [0., 0.]])
        widths = torch.tensor([[.3802, .3802], [.30, .3802], [.20, .3802],
                               [.10, .3802], [.10, .10], [-.10, -.10],
                               [.6604, .3802], [.3802, .3802]])
        centers = torch.zeros(len(widths), 4, 3)
        centers[:, [0, 2], 1] = widths / 2
        centers[:, [1, 3], 1] = -widths / 2
        centers[-1, :2, 1] += .065
        commands = torch.tensor([[.5, 0., 0.]]).repeat(len(widths), 1)
        raw = rolling_placement_error(centers, commands, reference, cfg, 1e-6)
        torch.testing.assert_close(raw, torch.tensor([
            0., .040401, .451, .951, 1.902, 3.902, .951, .25]))
        self.assertTrue(torch.all(raw[1:6] > raw[:5]))
        weighted_rate, weighted_tick = -.05 * raw, -.05 * .02 * raw
        self.assertTrue(torch.all(weighted_rate[1:6] < weighted_rate[:5]))
        torch.testing.assert_close(weighted_tick, weighted_rate * .02)
        # Even small intentional lateral/yaw commands exclude placement.
        for command in ([.5, .02, 0.], [.5, 0., .03], [0., 0., -.5]):
            commands[:] = torch.tensor(command)
            self.assertFalse(rolling_placement_error(
                centers, commands, reference, cfg, 1e-6).any())

    def test_measured_reference_and_partial_command_gate(self):
        height, dx, centers = v3_reference_geometry(with_centers=True)
        self.assertEqual((height, dx), v3_reference_geometry())
        centers = torch.tensor(centers).repeat(7, 1, 1)
        width = centers[0, [0, 2], 1] - centers[0, [1, 3], 1]
        torch.testing.assert_close(width, torch.tensor([.38020, .38020]))
        reference = torch.stack((width, torch.zeros(2)))
        cfg = dict(width_deadband_m=.04, width_scale_m=.10, midpoint_deadband_m=.015, midpoint_scale_m=.05)
        commands = torch.tensor([[0., 0., 0.], [.6, 0., 0.], [0., .02, 0.],
                                 [0., -.03, 0.], [0., 0., .03], [0., 0., -.175], [0., .3, .8]])
        self.assertFalse(rolling_placement_error(centers, commands, reference, cfg, 1e-6).any())
        # A width deadband applies to pair separation, not independently to each wheel.
        centers[:, 0, 1] += .02
        centers[:, 1, 1] -= .02
        self.assertLess(float(rolling_placement_error(centers, commands, reference, cfg, 1e-6).max()), 1e-12)
        centers[:, 0, 1] += .10
        centers[:, 1, 1] -= .10
        # Move the entire rear pair too; width alone must not hide a pair displacement.
        centers[:, 2:, 1] += .065
        cost = rolling_placement_error(centers, commands, reference, cfg, 1e-6)
        torch.testing.assert_close(cost[:2], torch.ones(2))
        self.assertFalse(cost[2:].any())
        reflected = centers[:, [1, 0, 3, 2]].clone()
        reflected[:, :, 1] *= -1
        torch.testing.assert_close(cost, rolling_placement_error(reflected, commands*torch.tensor([1., -1., -1.]), reference, cfg, 1e-6))

    def test_new_recipe_restores_parent_and_historical_replay(self):
        self.check_saved_recipe('navigation_rolling_placement')

    def test_combined_recipe_restores_parent_and_historical_replay(self):
        self.check_saved_recipe('navigation_rolling_control')

    def check_saved_recipe(self, recipe):
        parent, training = task_registry.get_cfgs('go2w')
        apply_go2w_profile(parent, training, 'transfer_v3')
        parent.go2w_finetune = training.go2w_finetune = 'navigation_zero_hold'
        parent.phase_observation_mode = 'command_demand'
        parent.commands.pure_lateral_magnitude_range = [.02, .3]
        parent.rewards.scales.stand_still = -5.
        parent.sensor_smooth = {'hip_weight': 3.}
        parent.refinement_parent = {'checkpoint': 'partial.pt', 'iteration': 199, 'prior_updates': 2400}
        training.algorithm.learning_rate, training.algorithm.schedule = 5e-5, 'fixed'
        source = {'task': 'go2w', 'env_cfg': class_to_dict(parent), 'train_cfg': class_to_dict(training)}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path/'config.yaml').write_text(yaml.safe_dump(source))
            torch.save({'iter': 499}, path/'model_499.pt')
            with patch.object(sys, 'argv', ['train', '--task', 'go2w', '--go2w_profile', 'transfer_v3',
                    '--go2w_finetune', recipe, '--load_run', str(path), '--checkpoint', '499']):
                args = get_args()
            cfg, train = task_registry.get_cfgs('go2w')
            update_cfg_from_args(cfg, train, args)
            before = class_to_dict(cfg)
            update_cfg_from_args(cfg, train, args)
            self.assertEqual(before, class_to_dict(cfg))
            validate_training(args, cfg, train)
            expected = copy.deepcopy(source['env_cfg']['rewards'])
            expected['scales']['rolling_placement'] = -.05
            expected['rolling_placement'] = cfg.rewards.rolling_placement
            if recipe == 'navigation_rolling_control':
                expected['phase_objective']['rolling_yaw_scale'] = .07
            self.assertEqual(expected, class_to_dict(cfg.rewards))
            for key in ('commands', 'control', 'domain_rand', 'sim', 'init_state', 'noise', 'phase_observation_mode', 'sensor_smooth'):
                self.assertEqual(json.loads(json.dumps(class_to_dict(getattr(cfg, key)))), json.loads(json.dumps(source['env_cfg'][key])))
            self.assertEqual(class_to_dict(train.algorithm), source['train_cfg']['algorithm'])
            self.assertEqual(cfg.refinement_parent['prior_updates'], 2900)
            self.assertEqual(cfg.refinement_parent['previous_generation'], source['env_cfg']['refinement_parent'])
            self.assertEqual(train.runner.checkpoint_load_cfg, dict(actor=True, critic=True, optimizer=False, iteration=False))
            self.assertEqual(train.runner.max_iterations, 500)
            with patch.object(sys, 'argv', ['play', '--task', 'go2w', '--load_run', str(path), '--checkpoint', '499']):
                replay_args = get_args()
            old, _, _ = task_registry.resolve_replay(replay_args)
            self.assertEqual(class_to_dict(old.rewards), source['env_cfg']['rewards'])


if __name__ == '__main__':
    unittest.main()
