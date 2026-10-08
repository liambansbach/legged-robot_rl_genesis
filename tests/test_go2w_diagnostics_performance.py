"""Fixed-size diagnostics must preserve samples and leave the task/RNG untouched."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import robot_gym.envs  # Register tasks before shared diagnostic imports.
from genesis.utils.geom import quat_to_xyz
from robot_gym.envs.go2w.phase import step_demand
from robot_gym.envs.go2w.training_diagnostics import TrainingDiagnostics


class DiagnosticsReductionsTest(unittest.TestCase):
    def make_diagnostics(self, count=37):
        d = TrainingDiagnostics.__new__(TrainingDiagnostics)
        d.family_ids = torch.arange(len(d.families))
        d.command_families = torch.arange(count) % 8 - 1
        d.env = SimpleNamespace(
            command_hold_kind=torch.arange(count) % 3,
            base_quat=torch.nn.functional.normalize(torch.randn(count, 4), dim=1),
            reset_buf=torch.arange(count) % 11 == 0,
            nonfoot_contact_count=(torch.arange(count) % 9 == 0).long(),
            wheel_center_lateral_speed=torch.randn(count, 4),
            loaded_wheels=torch.randn(count, 4) > 0,
        )
        d.reset_aggregates()
        return d

    @patch('genesis.EPS', 1e-12)  # Geometry arithmetic only; no simulator initialization.
    def test_family_reward_and_posture_reductions(self):
        torch.manual_seed(12)
        d = self.make_diagnostics()
        raw, commands = torch.randn(37), torch.randn(37, 3)
        before = {k: v.clone() for k, v in vars(d.env).items() if torch.is_tensor(v)}
        rng = torch.get_rng_state().clone()
        d.reward(raw, commands)
        d.reward(raw, commands)
        masks = [torch.ones(37, dtype=torch.bool), d.command_families == -1]
        masks.extend(d.command_families == i for i in range(7))
        expected = []
        for mask in masks:
            values = torch.where(mask, raw, 0)
            expected.append(torch.stack((mask.sum(), values.sum(), (values < 0).sum(),
                                         (-values.clamp_max(0)).sum())).float() * 2)
        torch.testing.assert_close(d.rewards, torch.stack(expected))
        angles = quat_to_xyz(d.env.base_quat, rpy=True)[:, :2]
        valid = ~d.env.reset_buf.bool() & (d.env.nonfoot_contact_count == 0)
        expected = []
        for family in range(7):
            mask = valid & (d.command_families == family)
            samples = torch.where(mask[:, None], angles, 0)
            expected.append(torch.cat((mask.sum().reshape(1), samples.sum(0),
                                       samples.square().sum(0))) * 2)
        torch.testing.assert_close(d.posture, torch.stack(expected))
        gate = step_demand(commands)
        moving = valid & (gate > 0)
        speed2 = d.env.wheel_center_lateral_speed.square() * d.env.loaded_wheels
        expected_scrub = torch.stack((moving.sum(), (moving & (raw < 0)).sum(),
                                     (speed2.mean(1) * moving).sum(),
                                     (speed2.mean(1) * gate * moving * (raw < 0)).sum())) * 2
        torch.testing.assert_close(d.scrubbing, expected_scrub)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        for key, value in before.items():
            self.assertTrue(torch.equal(value, getattr(d.env, key)), key)

    def test_masked_slew_counts_and_all_excluded(self):
        d = self.make_diagnostics()
        value = torch.randn(37, 16)
        mask = torch.arange(37) % 3 == 0
        d.add('slew', value, mask)
        d.add('slew', value * 2, mask)
        total, count = d.rollout['slew']
        self.assertEqual(int(count), 2 * int(mask.sum()))
        torch.testing.assert_close(total / count, 1.5 * value[mask].mean(0))
        d.add('excluded', value, torch.zeros_like(mask))
        self.assertEqual(int(d.rollout['excluded'][1]), 0)
        self.assertFalse(bool(d.rollout['excluded'][0].any()))
        d.add('unmasked', value)
        self.assertEqual(d.rollout['unmasked'][1], 37)
        torch.testing.assert_close(d.rollout['unmasked'][0], value.sum(0))

    def test_reward_totals_and_dt_are_unchanged(self):
        d = self.make_diagnostics()
        raw = torch.linspace(-2, 2, 37)
        weighted = raw * -.1 * .02
        for _ in range(64):
            d.reward_term('geometric', raw, weighted)
        count, total, contribution = d.term_totals['geometric']
        self.assertEqual(count, 37 * 64)
        expected_raw = torch.zeros(())
        expected_weighted = torch.zeros(())
        for _ in range(64):
            expected_raw += raw.sum()
            expected_weighted += weighted.sum()
        torch.testing.assert_close(total, expected_raw, rtol=0, atol=0)
        torch.testing.assert_close(contribution, expected_weighted, rtol=0, atol=0)
        self.assertTrue(torch.equal(raw, torch.linspace(-2, 2, 37)))

    def test_record_actions_excludes_reset_boundaries(self):
        d = self.make_diagnostics(4)
        d.env.cfg = SimpleNamespace(normalization=SimpleNamespace(clip_actions=1.),
                                    control=SimpleNamespace(wheel_velocity_target_limit=18.))
        d.env.action_scale = torch.tensor([.3, .35, .4, 18.] * 4)
        d.env.wheel_action_indices = [3, 7, 11, 15]
        d.env.dt = .02
        d.previous_mean = d.previous_sample = None
        d.previous_valid = torch.tensor([True, False, True, False])
        first, second = torch.randn(4, 16), torch.randn(4, 16)
        d.record_actions(first, first / 2)
        d.record_actions(second, second / 2)
        expected = ((d.targets(second) - d.targets(first))[d.previous_valid] / .02).square()
        total, count = d.rollout['sampled_target_slew_squared_per_s2']
        self.assertEqual(int(count), 2)
        torch.testing.assert_close(total / count, expected.mean(0))


if __name__ == '__main__':
    unittest.main()
