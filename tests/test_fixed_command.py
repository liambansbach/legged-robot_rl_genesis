"""Exercise real command/reset/observation paths with only physics calls stubbed."""

from types import SimpleNamespace
from unittest.mock import Mock, patch
import sys
import unittest

import torch

from robot_gym.envs import *  # noqa: F401,F403
from robot_gym.scripts.play import configure_fixed_command
from robot_gym.utils import get_args, task_registry
from robot_gym.utils.helpers import class_to_dict
from robot_gym.envs.go2w.zero_command_brake import ZeroCommandBrake


COMMANDS = [
    (0, 0, 0),
    (0.5, 0, 0),
    (0, 0.25, 0),
    (0, 0, 0.4),
    (0, 0, 1.0),
    (0.5, 0, 0.8),
]


class FixedCommandTests(unittest.TestCase):
    def test_brake_ramps_clip_before_blend_and_preserve_legs_and_inputs(self):
        # Deliberately permuted mapping, with raw wheel proposals well beyond the bound.
        wheels = [12, 1, 6, 9]
        legs = [i for i in range(16) if i not in wheels]
        brake = ZeroCommandBrake(7, wheels, 0.02, "cpu")
        raw = torch.linspace(-4, 4, 16).repeat(7, 1)
        original = raw.clone()
        commands = torch.tensor([
            [0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.4],
            [0, 0, -0.4], [0.8e-6, 0.8e-6, 0], [0, 0, 1.1e-6],
        ])
        for step in range(10):
            issued = brake.apply(raw, commands)
            self.assertAlmostEqual(float(brake.alpha[0]), (step + 1) / 10, places=6)
            torch.testing.assert_close(issued[:, legs], raw.clamp(-1, 1)[:, legs])
            torch.testing.assert_close(issued[1:], raw[1:].clamp(-1, 1))
            torch.testing.assert_close(issued[0, wheels], raw[0, wheels].clamp(-1, 1) * (1 - (step + 1) / 10))
        self.assertEqual(float(brake.alpha[0]), 1)
        self.assertEqual(issued[0, wheels].abs().max(), 0)
        commands[0, 0] = -0.1
        for step in range(5):
            issued = brake.apply(raw, commands)
            self.assertAlmostEqual(float(brake.alpha[0]), 1 - (step + 1) / 5, places=6)
        self.assertEqual(float(brake.alpha[0]), 0)
        torch.testing.assert_close(issued, raw.clamp(-1, 1))
        torch.testing.assert_close(raw, original)
        brake.apply(raw, torch.full((7, 3), 0.5e-6))
        brake.reset([1, 5])
        torch.testing.assert_close(brake.alpha, torch.tensor([.1, 0, .1, .1, .1, 0, .1]))
        brake.reset()
        self.assertEqual(brake.alpha.count_nonzero(), 0)

    def test_brake_step_history_observations_and_reset_semantics(self):
        e = self.make_env("go2w")
        e.num_actions, e.all_env_ids = 16, torch.arange(3)
        e.action_delay_steps = torch.tensor([0, 1, 0])
        e.privileged_obs_buf = None
        e.rew_buf = torch.zeros(3)
        e._control_dofs = Mock()
        e._apply_pushes = Mock()
        e.cfg.domain_rand.push_robots = False
        e.set_fixed_command((0, 0, 0))
        raw = torch.linspace(-4, 4, 16).repeat(3, 1)
        original = raw.clone()
        # Disabled mode is the ordinary clipped action/history path.
        e.step(raw)
        torch.testing.assert_close(e.actions, raw.clamp(-1, 1))
        e.enable_zero_command_brake()
        e.action_history.zero_()
        e.step(raw)
        expected = raw.clamp(-1, 1)
        expected[:, e.wheel_action_indices] *= .9
        torch.testing.assert_close(e.actions, expected)
        torch.testing.assert_close(e.action_history[:, 0], expected)
        torch.testing.assert_close(e.obs_buf[:, -16:], expected)
        torch.testing.assert_close(e.applied_actions[0], expected[0])
        self.assertEqual(e.applied_actions[1].count_nonzero(), 0)
        self.assertEqual(e._control_dofs.call_count, 2 * e.cfg.control.decimation)
        torch.testing.assert_close(e.zero_command_brake.alpha, torch.full((3,), .1))
        torch.testing.assert_close(raw, original)
        e.reset_idx(torch.tensor([1]))
        torch.testing.assert_close(e.zero_command_brake.alpha, torch.tensor([.1, 0, .1]))
        e.reset()
        self.assertEqual(e.zero_command_brake.alpha.count_nonzero(), 0)
        self.assertEqual(e.actions.count_nonzero(), 0)

    def test_brake_cli_default_and_training_rejection(self):
        from robot_gym.scripts.train import train
        from robot_gym.utils.helpers import update_cfg_from_args

        with patch.object(sys, "argv", ["play", "--task", "go2w"]):
            args = get_args()
        self.assertFalse(args.zero_command_brake)
        env, cfg = task_registry.get_cfgs("go2w")
        before = (class_to_dict(env), class_to_dict(cfg))
        args.zero_command_brake = True
        update_cfg_from_args(env, cfg, args)
        self.assertEqual(before, (class_to_dict(env), class_to_dict(cfg)))
        with self.assertRaisesRegex(ValueError, "inference-only"):
            train(args)
        args.task = "go2"
        with self.assertRaisesRegex(ValueError, "specific to go2w"):
            update_cfg_from_args(env, cfg, args)

    def make_env(self, task):
        cls = task_registry.get_task_class(task)
        e = cls.__new__(cls)
        e.cfg, _ = task_registry.get_cfgs(task)
        e.cfg.commands.curriculum = False
        e.cfg.domain_rand.randomize_friction = False
        e.cfg.env.play_mode = True
        e.cfg.env.capture_transitions = False
        e.cfg.env.send_timeouts = False
        e.cfg.viewer.print_debug_velocities = False
        e.device, e.num_envs, e.dt = torch.device("cpu"), 3, 0.02
        e.num_obs = e.cfg.env.num_observations
        e.joint_names = list(e.cfg.init_state.default_joint_angles)
        e.joint_dof_idx = list(range(len(e.joint_names)))
        e._build_control_tensors()
        e.commands, e.commands_scale = torch.zeros(3, 3), torch.tensor([2.0, 3.0, 4.0])
        e.command_ranges = class_to_dict(e.cfg.commands.ranges)
        e.command_steps_left = torch.zeros(3, dtype=torch.long)
        e.command_resampling_enabled = True
        e.obs_scales = e.cfg.normalization.obs_scales
        e.base_lin_vel = e.base_ang_vel = e.projected_gravity = torch.zeros(3, 3)
        e.base_pos, e.base_quat = (
            torch.zeros(3, 3),
            torch.tensor([[1.0, 0, 0, 0]]).expand(3, -1),
        )
        e.default_dof_pos = torch.zeros(1, len(e.joint_names))
        e.dof_pos = torch.zeros(3, len(e.joint_names))
        for name in (
            "dof_vel",
            "actions",
            "last_actions",
            "last_dof_vel",
            "applied_actions",
        ):
            setattr(e, name, torch.zeros_like(e.dof_pos))
        e.feet_air_time = torch.zeros(3, 4)
        e.prev_foot_contacts = torch.zeros(3, 4, dtype=torch.bool)
        e.action_history = torch.zeros(3, 2, len(e.joint_names))
        e.push_force = torch.zeros(3, 1, 3)
        e.push_torque = torch.zeros_like(e.push_force)
        for name in (
            "episode_length_buf",
            "reset_buf",
            "push_steps_left",
            "next_push_steps",
        ):
            setattr(e, name, torch.zeros(3, dtype=torch.long))
        e.episode_sums, e.extras = {}, {}
        e.sim, e.velocity_arrow_visualizer = Mock(), Mock()
        for name in (
            "_reset_dofs",
            "_reset_root_states",
            "_update_robot_state",
            "_sample_action_delay",
            "check_termination",
        ):
            setattr(e, name, Mock())
        e.common_step_counter, e.headless, e.add_noise = 0, False, False
        return e

    def test_fixed_commands_survive_resampling_reset_and_reach_policy_and_viewer(self):
        for task in ("dodo", "go2", "go2w"):
            e = self.make_env(task)
            ranges = class_to_dict(e.cfg.commands.ranges)
            for command in COMMANDS:
                with self.subTest(task=task, command=command):
                    configure_fixed_command(
                        e,
                        SimpleNamespace(
                            command_vx=command[0],
                            command_vy=command[1],
                            command_yaw=command[2],
                        ),
                    )
                    expected = torch.tensor(command, dtype=torch.float).expand(3, -1)
                    torch.testing.assert_close(e.commands, expected)
                    torch.testing.assert_close(
                        e.obs_buf[:, 9:12], expected * e.commands_scale
                    )
                    self.assertFalse(e.command_resampling_enabled)
                    # Exercise actual sampler guards: the family sampler must never run.
                    with patch(
                        "torch.multinomial",
                        side_effect=AssertionError("family sampling in fixed mode"),
                    ):
                        e._resample_commands(torch.arange(3))
                        e.reset_idx(torch.tensor([0, 2]))
                        e.post_physics_step()  # includes selective reset, observation rebuild and arrows
                    torch.testing.assert_close(e.commands, expected)
                    torch.testing.assert_close(
                        e.obs_buf[:, 9:12], expected * e.commands_scale
                    )
                    torch.testing.assert_close(
                        e.velocity_arrow_visualizer.update.call_args.kwargs["commands"],
                        expected,
                    )
            self.assertEqual(class_to_dict(e.cfg.commands.ranges), ranges)

    def test_cli_omitted_axes_zero_command_and_disabled_mode(self):
        e = self.make_env("go2w")
        before = e.commands.clone()
        with patch.object(sys, "argv", ["play"]):
            configure_fixed_command(e, get_args())
        self.assertTrue(e.command_resampling_enabled)
        self.assertIsNone(getattr(e, "_fixed_command", None))
        torch.testing.assert_close(e.commands, before)
        for flags, expected in [
            (["--command_yaw", "0.4"], (0, 0, 0.4)),
            (["--command_vx", "0"], (0, 0, 0)),
        ]:
            with patch.object(sys, "argv", ["play", *flags]):
                configure_fixed_command(e, get_args())
            torch.testing.assert_close(
                e.commands, torch.tensor(expected, dtype=torch.float).expand(3, -1)
            )
        e.set_fixed_command(None)
        self.assertTrue(e.command_resampling_enabled)
        with patch.object(e, "_reset_command_timer") as timer:
            e._resample_commands(torch.arange(3))
        timer.assert_called_once()
        for command in [(1, 2), (0, 0, float("nan")), (float("inf"), 0, 0)]:
            with self.assertRaises(ValueError):
                e.set_fixed_command(command)
