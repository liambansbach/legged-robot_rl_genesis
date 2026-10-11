"""Offline window/censoring checks for the frozen source comparison."""

import json
import unittest

import numpy as np
import torch

from robot_gym.envs.go2w.source_yaw import BUNDLE, metrics, reset_case


class SourceYawTests(unittest.TestCase):
    def setUp(self):
        self.contract = json.loads((BUNDLE / "contract.json").read_text())
        n = 400
        self.data = {
            "post_time_s": np.arange(1, n + 1) * .02,
            "post_base_lin_vel": np.zeros((n, 3)),
            "post_base_ang_vel": np.zeros((n, 3)),
            "post_wheel_velocity_targets": np.zeros((n, 4)),
            "post_dof_vel": np.zeros((n, 16)),
            "post_dof_pos": np.tile(self.contract["default_joint_angles"], (n, 1)),
            "post_leg_position_targets": np.tile(np.asarray(self.contract["default_joint_angles"])[self.contract["leg_action_indices"]], (n, 1)),
            "pre_phase": (np.arange(n) * .025 + .025) % 1,
            "pre_previous_issued_action": np.zeros((n, 16)),
            "post_actions": np.zeros((n, 16)), "post_raw_actions": np.zeros((n, 16)),
            "post_nonfoot_ground_link_count": np.zeros(n),
            "post_base_pos": np.tile([0., 0., .43], (n, 1)),
            "pre_base_pos": np.tile([0., 0., .43], (n, 1)),
            "post_base_quat": np.tile([1., 0., 0., 0.], (n, 1)),
            "pre_base_quat": np.tile([1., 0., 0., 0.], (n, 1)),
            "post_summed_normal_ground_force": np.full((n, 4), 50.),
            "post_wheel_clearance": np.zeros((n, 4)),
        }

    def test_boundary_samples_and_stop_displacement(self):
        self.data["post_base_ang_vel"][100:150, 2] = .2
        self.data["post_base_ang_vel"][150:250, 2] = .5
        self.data["post_base_ang_vel"][250:300, 2] = .7
        self.data["pre_base_pos"][300, 0] = .1
        self.data["post_base_pos"][300:, 0] = .12
        result = metrics(self.data, .8, self.contract, None)
        self.assertAlmostEqual(result["windows"]["first_second"]["mean_body_vx_vy_wz"][2], .2)
        self.assertAlmostEqual(result["windows"]["last_second"]["mean_body_vx_vy_wz"][2], .7)
        self.assertAlmostEqual(result["windows"]["full_hold"]["rmse_vx_vy_wz"][2], np.sqrt((.6**2 + 2*.3**2 + .1**2)/4))
        self.assertAlmostEqual(result["stopping"]["xy_displacement_m"], .02)

    def test_failure_censors_window_and_does_not_score_reset_tail(self):
        data = {k: v[:280] for k, v in self.data.items()}
        result = metrics(data, .8, self.contract, "fall")
        self.assertTrue(result["windows"]["last_second"]["censored"])
        self.assertTrue(result["windows"]["full_hold"]["censored"])
        self.assertNotIn("stopping", result)
        self.assertFalse(result["windows"]["first_second"]["censored"])

    def test_settling_requires_remaining_quiet_horizon(self):
        self.data["post_base_ang_vel"][390, 2] = .04
        result = metrics(self.data, .8, self.contract, None)
        self.assertIsNone(result["stopping"]["settled_after_s"])

    def test_explicit_reset_one_zero_tick_and_wheel_only_sensitivity(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        env = SimpleNamespace(set_commands=Mock(), reset_idx=Mock(), all_env_ids=torch.tensor([0]),
                              robot=SimpleNamespace(set_friction_ratio=Mock()),
                              device="cpu", foot_link_indices_local=[3, 7, 11, 15],
                              actions=torch.zeros(1, 16), action_history=torch.zeros(1, 1, 16),
                              action_delay_steps=torch.zeros(1), phase=torch.tensor([.025]),
                              step=Mock(return_value=(None, None, torch.tensor([False]), None)))
        reset_case(env, .7)
        env.reset_idx.assert_called_once()
        env.step.assert_called_once()
        self.assertFalse(bool(env.step.call_args.args[0].any()))
        self.assertEqual(env.robot.set_friction_ratio.call_args.args[1], [3, 7, 11, 15])


if __name__ == "__main__":
    unittest.main()
