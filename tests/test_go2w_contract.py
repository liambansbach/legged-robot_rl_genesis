"""Contract tests independent of the GPU solver (python -m unittest discover -s tests)."""

import unittest
from unittest.mock import Mock, patch
import torch
from robot_gym.envs.go2w.go2w_env import Go2WEnv
from robot_gym.envs.go2w.go2w_config import GO2WCfg


class ContractTests(unittest.TestCase):
    def make_env(self, count=8):
        e = Go2WEnv.__new__(Go2WEnv)
        e.cfg = GO2WCfg()
        e.device = torch.device("cpu")
        e.num_envs = count
        e.num_obs = 56
        e.dt = 0.02
        e.joint_names = list(e.cfg.init_state.default_joint_angles)
        e.joint_dof_idx = list(range(6, 22))
        e._build_control_tensors()
        e.commands = torch.zeros(count, 3)
        e.command_steps_left = torch.zeros(count, dtype=torch.long)
        e.command_ranges = vars(e.cfg.commands.ranges) or {
            n: getattr(e.cfg.commands.ranges, n)
            for n in ["lin_vel_x", "lin_vel_y", "ang_vel_yaw"]
        }
        return e

    def test_mixed_controller_and_physical_saturation(self):
        e = self.make_env()
        e.default_dof_pos = torch.tensor(
            [[e.cfg.init_state.default_joint_angles[n] for n in e.joint_names]]
        )
        e.dof_vel_limits = torch.tensor(
            [e.cfg.control.dof_vel_limits[n] for n in e.joint_names]
        )
        e.robot = Mock()
        e._control_dofs(torch.ones(8, 16))
        torch.testing.assert_close(
            e.robot.control_dofs_position.call_args.args[0],
            (e.default_dof_pos[:, e.p_control_mask] + 0.2).expand(8, -1),
        )
        torch.testing.assert_close(
            e.robot.control_dofs_velocity.call_args.args[0], torch.full((8, 4), 18.0)
        )
        e._control_dofs(torch.full((8, 16), 10.0))
        self.assertEqual(e.robot.control_dofs_velocity.call_args.args[0].max(), 20)
        e.robot.control_dofs_force.assert_not_called()
        self.assertEqual(e.wheel_action_indices, [3, 7, 11, 15])

    def test_command_mixture_and_low_yaw_gate(self):
        torch.manual_seed(4)
        e = self.make_env(20000)
        torch.testing.assert_close(
            e.command_mixture, torch.tensor([0.15, 0.20, 0.20, 0.15, 0.10, 0.13, 0.07])
        )
        self.assertAlmostEqual(e.command_mixture.sum().item(), 1.0)
        e._resample_commands(torch.arange(e.num_envs))
        c = e.commands
        self.assertTrue(((c[:, 0] >= -0.35) & (c[:, 0] <= 1.1)).all())
        self.assertTrue((c[:, 1].abs() <= 0.3).all())
        self.assertTrue((c[:, 2].abs() <= 1.4).all())
        self.assertTrue(0.14 < (c == 0).all(dim=1).float().mean() < 0.17)
        self.assertTrue(0.19 < (c[:, 1] != 0).float().mean() < 0.21)
        pure = (c[:, 0] == 0) & (c[:, 2] == 0) & (c[:, 1] != 0)
        self.assertTrue(0.12 < pure.float().mean() < 0.14)
        y = c[pure, 1]
        low, high = e.cfg.commands.pure_lateral_magnitude_range
        self.assertTrue(((y.abs() >= low) & (y.abs() <= high)).all())
        self.assertAlmostEqual((y > 0).float().mean().item(), 0.5, delta=0.035)
        self.assertAlmostEqual(y.abs().mean().item(), (low + high) / 2, delta=0.005)
        e.commands[:] = torch.tensor([0.0, 0.0, 0.3])
        self.assertEqual(e._mobility_gate().sum(), 0)

    def test_mixed_command_durations(self):
        torch.manual_seed(5)
        e = self.make_env(20000)
        e._resample_commands(torch.arange(e.num_envs))
        cfg = e.cfg.commands
        self.assertEqual(cfg.short_command_duration_range, [0.5, 1.0])
        self.assertEqual(cfg.sustained_command_duration_range, [1.5, 3.0])
        self.assertEqual(cfg.sustained_command_probability, 0.30)
        durations = e.command_steps_left
        short_low, short_high = [
            round(t / e.dt) for t in cfg.short_command_duration_range
        ]
        long_low, long_high = [
            round(t / e.dt) for t in cfg.sustained_command_duration_range
        ]
        short = (durations >= short_low) & (durations <= short_high)
        sustained = (durations >= long_low) & (durations <= long_high)
        self.assertTrue(short.any() and sustained.any())
        self.assertTrue((short | sustained).all())
        self.assertAlmostEqual(
            sustained.float().mean().item(),
            cfg.sustained_command_probability,
            delta=0.015,
        )
        # Stand and lateral commands retain the same prevalence in both duration modes.
        for mode in (short, sustained):
            commands = e.commands[mode]
            self.assertTrue(0.13 < (commands == 0).all(dim=1).float().mean() < 0.18)
            self.assertTrue(0.18 < (commands[:, 1] != 0).float().mean() < 0.22)
        untouched = durations[1::2].clone()
        e._reset_command_timer(torch.arange(0, e.num_envs, 2))
        torch.testing.assert_close(durations[1::2], untouched)

    def test_pure_lateral_is_configured_and_mixed_keeps_full_range(self):
        e = self.make_env(20000)
        # Force each family to test its conditional distribution without guessing labels.
        for family in (5, 6):
            torch.manual_seed(9)
            e.cfg.commands.pure_lateral_magnitude_range = [0.12, 0.28]
            with patch(
                "torch.multinomial", return_value=torch.full((e.num_envs,), family)
            ):
                e._resample_commands(torch.arange(e.num_envs))
            y = e.commands[:, 1]
            if family == 5:
                self.assertTrue(((y.abs() >= 0.12) & (y.abs() <= 0.28)).all())
                self.assertEqual(e.commands[:, [0, 2]].abs().sum(), 0)
                self.assertAlmostEqual((y > 0).float().mean().item(), 0.5, delta=0.015)
                self.assertAlmostEqual(y.abs().mean().item(), 0.20, delta=0.003)
            else:
                self.assertTrue(((y >= -0.30) & (y <= 0.30)).all())
                self.assertLess(y.min(), -0.29)
                self.assertGreater(y.max(), 0.29)
                self.assertAlmostEqual(
                    (y.abs() < 0.10).float().mean().item(), 1 / 3, delta=0.015
                )
                self.assertAlmostEqual(y.mean().item(), 0.0, delta=0.005)

    def test_lateral_tracking_incentive_and_longitudinal_tolerance(self):
        e = self.make_env(5)
        e.base_lin_vel = torch.zeros(5, 3)
        e.commands[:, :2] = torch.tensor(
            [[0.1, 0.0], [0.0, 0.1], [0.0, 0.25], [0.0, 0.02], [0.0, -0.1]]
        )
        rewards = e._reward_tracking_lin_vel()
        torch.testing.assert_close(
            rewards,
            torch.tensor([0.9607894, 0.7788008, 0.2096114, 0.9900498, 0.7788008]),
        )

    def test_gait_allowance_is_smooth_symmetric_and_keeps_regularization(self):
        e = self.make_env(10)
        e.commands[:, 1:] = torch.tensor(
            [
                [0.0, 0.4],
                [0.0, -0.4],
                [0.0, 0.85],
                [0.0, -1.0],
                [0.03, 0.0],
                [0.065, 0.0],
                [-0.25, 0.0],
                [0.065, 1.1],
                [0.0, 1.25],
                [0.0, 0.3],
            ]
        )
        torch.testing.assert_close(
            e._lateral_gait_gate(),
            torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0]),
        )
        torch.testing.assert_close(
            e._yaw_mobility_gate(),
            torch.tensor([2 / 11, 2 / 11, 1, 1, 0, 0, 0, 1, 1, 0]),
        )
        mobility = torch.tensor([1.6 / 11, 1.6 / 11, 0.8, 0.8, 0, 0.5, 1, 0.8, 0.8, 0])
        pose = torch.tensor([0, 0, 0.175, 0.28, 0, 0.5, 1, 0.5, 0.35, 0])
        torch.testing.assert_close(e._mobility_gate(), mobility)
        torch.testing.assert_close(e._pose_relaxation_gate(), pose)
        e.default_dof_pos = torch.zeros(1, 16)
        e.dof_pos = e.dof_vel = torch.ones(10, 16)
        e.foot_contacts = torch.zeros(10, 4, dtype=torch.bool)
        e.current_ankle_heights = torch.full(
            (10, 4), e.cfg.asset.contact_height + e.cfg.rewards.clearance_target
        )
        torch.testing.assert_close(e._reward_default_pose(), 1 - 0.7 * pose)
        torch.testing.assert_close(e._reward_leg_motion(), 1 - 0.7 * mobility)
        torch.testing.assert_close(
            e._reward_unnecessary_wheel_air(), 1 - 0.75 * mobility
        )
        torch.testing.assert_close(e._reward_foot_swing_clearance(), mobility)
        torch.testing.assert_close(
            e._reward_default_pose()[[0, 3, 8, 6]],
            torch.tensor([1.0, 0.804, 0.755, 0.30]),
        )
        self.assertTrue((e._reward_unnecessary_wheel_air() > 0).all())

    def test_gate_continuity_and_lateral_behavior_are_preserved(self):
        e = self.make_env(1001)
        e.commands[:, 1] = torch.linspace(-0.3, 0.3, e.num_envs)
        lateral = ((e.commands[:, 1].abs() - 0.03) / 0.07).clamp(0, 1)
        torch.testing.assert_close(e._mobility_gate(), lateral)
        torch.testing.assert_close(e._pose_relaxation_gate(), lateral)
        e.commands[:, 1] = 0
        e.commands[:, 2] = torch.linspace(-1.5, 1.5, e.num_envs)
        for gate in (e._mobility_gate(), e._pose_relaxation_gate()):
            torch.testing.assert_close(gate, gate.flip(0))
            self.assertLess((gate[1:] - gate[:-1]).abs().max(), 0.005)

    def test_push_duration_does_not_change_actuator_modes(self):
        e = self.make_env()
        e.robot = Mock()
        e.base_link_idx = 0
        e.push_steps_left = torch.full((8,), 3)
        e.push_force = torch.full((8, 1, 3), 20.0)
        e.push_torque = torch.zeros_like(e.push_force)
        e.active_push_force = torch.zeros_like(e.push_force)
        e.active_push_torque = torch.zeros_like(e.push_force)
        for _ in range(3):
            e._apply_pushes()
            self.assertEqual(
                e.robot.apply_links_external_wrench.call_args.kwargs["force"].max(), 20
            )
        e._apply_pushes()
        self.assertEqual(
            e.robot.apply_links_external_wrench.call_args.kwargs["force"].max(), 0
        )
        e.robot.control_dofs_force.assert_not_called()

    def test_observations_exclude_continuous_wheel_angle(self):
        e = self.make_env()
        e.obs_scales = e.cfg.normalization.obs_scales
        e.commands_scale = torch.ones(3)
        e.base_lin_vel = e.base_ang_vel = e.projected_gravity = torch.zeros(8, 3)
        e.default_dof_pos = torch.zeros(1, 16)
        e.dof_pos = torch.zeros(8, 16)
        e.dof_pos[:, e.wheel_action_indices] = float("nan")
        e.dof_vel = e.actions = torch.zeros(8, 16)
        e.add_noise = False
        e.compute_observations()
        self.assertEqual(e.obs_buf.shape, (8, 56))
        self.assertTrue(torch.isfinite(e.obs_buf).all())


if __name__ == "__main__":
    unittest.main()
