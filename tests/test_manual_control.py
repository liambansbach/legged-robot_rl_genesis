"""Small keyboard checks plus the real shared command/observation/reset paths."""

from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import sys
import unittest

import torch

import test_fixed_command
from robot_gym.scripts.play import play
from robot_gym.utils import get_args, task_registry
from robot_gym.utils.manual_control import KeyboardInput, held_axes, scale_axes


def arguments(*flags):
    with patch.object(sys, "argv", ["play", *flags]):
        return get_args()


class ManualControlTests(unittest.TestCase):
    def test_signs_asymmetric_limits_cancel_slow_and_stop(self):
        limits = ((-.3, 1.), (-.2, .4), (-.8, .6))
        for key, expected in (("w", (1., 0, 0)), ("s", (-.3, 0, 0)),
                              ("a", (0, .4, 0)), ("d", (0, -.2, 0)),
                              ("q", (0, 0, .6)), ("e", (0, 0, -.8))):
            self.assertEqual(scale_axes(held_axes({key}), limits), expected)
        self.assertEqual(scale_axes(held_axes(set("wsadqe")), limits), (0., 0., 0.))
        self.assertEqual(scale_axes(held_axes(set()), limits), (0., 0., 0.))
        self.assertEqual(scale_axes(held_axes({"w", "d", "q", "shift"}), limits), (.3, -.06, .18))
        self.assertEqual(scale_axes(held_axes({"w", "q", "space"}), limits), (0., 0., 0.))
        self.assertEqual(scale_axes((-1, 1, 0), ((0, 1), (-1, 0), (1, 2))), (0., 0., 0.))
        self.assertEqual(scale_axes((0, 0, 0), ((1, 2), (-2, -1), (0, 0))), (0., 0., 0.))

    def test_held_state_release_focus_and_quit(self):
        # Only pygame's event/focus/key reads are stubbed; no event-driven movement state.
        names = ("QUIT", "WINDOWCLOSE", "KEYDOWN", "WINDOWFOCUSLOST", "K_ESCAPE",
                 "K_w", "K_s", "K_a", "K_d", "K_q", "K_e", "K_SPACE", "K_LSHIFT", "K_RSHIFT")
        pg = SimpleNamespace(**{name: name for name in names})
        pressed = defaultdict(bool)
        pg.event = SimpleNamespace(get=Mock(return_value=[]))
        pg.key = SimpleNamespace(get_focused=Mock(return_value=True), get_pressed=Mock(return_value=pressed))
        keyboard = KeyboardInput.__new__(KeyboardInput)
        keyboard.pg, keyboard.quit, keyboard.wait_for_release = pg, False, True
        self.assertEqual(keyboard.poll(), ((0., 0., 0.), False))
        pressed[pg.K_w] = True
        self.assertEqual(keyboard.poll()[0], (1., 0., 0.))
        for shift in (pg.K_LSHIFT, pg.K_RSHIFT):
            pressed[shift] = True
            self.assertEqual(keyboard.poll()[0], (.3, 0., 0.))
            pressed[shift] = False
        pressed.clear()
        self.assertEqual(keyboard.poll()[0], (0., 0., 0.))
        pressed[pg.K_w] = True
        pg.key.get_focused.return_value = False
        self.assertEqual(keyboard.poll()[0], (0., 0., 0.))
        pg.key.get_focused.return_value = True
        self.assertEqual(keyboard.poll()[0], (0., 0., 0.))  # Held across focus return.
        pressed.clear()
        keyboard.poll()
        pressed[pg.K_w] = True
        self.assertEqual(keyboard.poll()[0], (1., 0., 0.))
        pg.event.get.return_value = [SimpleNamespace(type=pg.WINDOWFOCUSLOST)]
        self.assertEqual(keyboard.poll()[0], (0., 0., 0.))  # Lost/regained between ticks.
        for event in (SimpleNamespace(type=pg.QUIT), SimpleNamespace(type=pg.WINDOWCLOSE),
                      SimpleNamespace(type=pg.KEYDOWN, key=pg.K_ESCAPE)):
            keyboard.quit = False
            pg.event.get.return_value = [event]
            self.assertTrue(keyboard.poll()[1])

    def test_factory_claims_zero_before_initial_reset_for_each_robot(self):
        for task in ("dodo", "go2", "go2w"):
            e = test_fixed_command.FixedCommandTests().make_env(task)
            cls = type(e)
            cfg = e.cfg
            def reset():
                self.assertFalse(e.command_resampling_enabled)
                self.assertEqual(torch.count_nonzero(e.commands), 0)
                e.reset_idx(torch.arange(3))
            e.reset = Mock(side_effect=reset)
            with patch.object(cls, "__new__", return_value=e), patch.object(cls, "__init__", return_value=None):
                task_registry.make_env(task, args=arguments("--task", task), env_cfg=cfg,
                                       initial_command=(0., 0., 0.))
            e.reset.assert_called_once()

    def test_replay_applies_input_before_actual_policy_observation_for_each_robot(self):
        for task, count, extra in (("dodo", 1, []), ("go2", 3, ["--num_envs", "3"]),
                                   ("go2w", 1, ["--steps", "3"])):
            e = test_fixed_command.FixedCommandTests().make_env(task)
            cfg, train = e.cfg, task_registry.get_cfgs(task)[1]
            limits = tuple(tuple(getattr(cfg.commands.ranges, name))
                           for name in ("lin_vel_x", "lin_vel_y", "ang_vel_yaw"))
            e.sim.viewer.is_alive.return_value = True
            e.max_episode_length = 1000
            e.reset = Mock()
            if task == "go2w":
                e.phase[:] = .17
            history = e.action_history.clone()
            axes = ((-.3, 0., .3), (0., 1., 0.), (0., 0., 0.))
            expected = [scale_axes(value, limits) for value in axes]
            seen = []
            def policy(obs):
                command = torch.tensor(expected[len(seen)]).expand(3, -1)
                torch.testing.assert_close(obs["policy"][:, 9:12], command * e.commands_scale)
                torch.testing.assert_close(e.action_history, history)
                if task == "go2w":
                    from robot_gym.envs.go2w.phase import clock_observation
                    torch.testing.assert_close(e.phase, torch.full((3,), .17))
                    torch.testing.assert_close(obs["policy"][:, -2:],
                                               clock_observation(e.phase, command, cfg.phase_observation_mode))
                seen.append(command.clone())
                return torch.zeros_like(e.actions)
            def step(action):
                # Force a resampling boundary and real selective reset between inferences.
                e.command_steps_left[:] = 0
                with patch("torch.multinomial", side_effect=AssertionError("manual family sampling")):
                    e._post_physics_step_callback()
                    e._resample_commands(torch.arange(3))
                    e.reset_idx(torch.tensor([0, 2]))
                torch.testing.assert_close(e.commands, seen[-1])
                if task == "go2w":
                    e.phase[:] = .17
                e.compute_observations()
                return e.get_observations(), None, None, {}
            e.step = Mock(side_effect=step)
            keyboard = Mock()
            keyboard.poll.side_effect = [(value, False) for value in axes] + [((0., 0., 0.), True)]
            runner = Mock()
            runner.get_inference_policy.return_value = policy
            def make_env(*unused, **kwargs):
                self.assertEqual(kwargs["env_cfg"].env.num_envs, count)
                self.assertEqual(kwargs["env_cfg"].viewer.ref_env, list(range(count)))
                self.assertEqual(kwargs["initial_command"], (0., 0., 0.))
                e.set_fixed_command(kwargs["initial_command"])
                return e, cfg
            with patch.object(task_registry, "resolve_replay", return_value=(cfg, train, Path("model_1999.pt"))), \
                 patch.object(task_registry, "make_env", side_effect=make_env), \
                 patch.object(task_registry, "make_alg_runner", return_value=(runner, train)), \
                 patch("robot_gym.utils.manual_control.KeyboardInput", return_value=keyboard):
                play(arguments("--task", task, "--manual_control", *extra))
            self.assertEqual(len(seen), 3)
            keyboard.close.assert_called_once()

    def test_conflicts_defaults_and_optional_dependency(self):
        self.assertEqual(arguments().steps, 1000)
        self.assertFalse(arguments("--manual_control")._steps_explicit)
        self.assertTrue(arguments("--manual_control", "--steps=1000")._steps_explicit)
        for flags in (("--headless",), ("--command_vx", "0"), ("--command_vy", "1"), ("--command_yaw", "0")):
            with self.assertRaisesRegex(ValueError, "--manual_control"):
                play(arguments("--manual_control", *flags))
        with patch.dict(sys.modules, {"pygame": None}):
            with self.assertRaisesRegex(RuntimeError, "python -m pip install pygame"):
                KeyboardInput()
            # Exercise ordinary replay and CLI help while pygame cannot be imported.
            from test_shared_pipeline import ReplayTests
            ReplayTests().test_play_refreshes_first_observation_after_task_state_load()
            with self.assertRaises(SystemExit) as result:
                arguments("--help")
            self.assertEqual(result.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
