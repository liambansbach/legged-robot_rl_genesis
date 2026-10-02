"""Saved recipes and replay selectors, without constructing native physics."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch
import yaml
import torch

from robot_gym.envs import *  # noqa: F401,F403
from robot_gym.utils import task_registry, get_args
from robot_gym.utils.helpers import class_to_dict
from robot_gym.utils.replay import configure_nominal


def arguments(task, run, *extra):
    with patch.object(sys, "argv", ["play", "--task", task, "--load_run", str(run), *extra]):
        return get_args()


class ReplayTests(unittest.TestCase):
    def test_each_registered_task_restores_saved_values_and_latest_once(self):
        for task in ("dodo", "go2", "go2w"):
            with self.subTest(task=task), TemporaryDirectory() as directory:
                path = Path(directory)
                cfg, train = task_registry.get_cfgs(task)
                saved = {"env_cfg": class_to_dict(cfg), "train_cfg": class_to_dict(train)}
                saved["env_cfg"]["rewards"]["scales"]["orientation"] = -7.123
                for section, fields in (("control", ("armature", "armature_override", "passive_damping")),
                                        ("asset", ("default_armature",)),
                                        ("domain_rand", ("randomize_armature",))):
                    for field in fields:
                        saved["env_cfg"][section].pop(field, None)
                (path / "config.yaml").write_text(yaml.safe_dump(task_registry._make_yaml_safe(saved)))
                for number in (9, 100, 1999):
                    (path / f"model_{number}.pt").touch()
                args = arguments(task, path, "--checkpoint", "-1", "--num_envs", "1", "--steps", "900", "--episode_length_s", "30")
                restored, training, checkpoint = task_registry.resolve_replay(args)
                self.assertEqual(checkpoint.name, "model_1999.pt")
                self.assertEqual(args.resolved_checkpoint, str(checkpoint))
                self.assertEqual(restored.rewards.scales.orientation, -7.123)
                self.assertEqual(restored.env.episode_length_s, 30)
                self.assertEqual(args.steps * restored.sim.dt * restored.control.decimation, 18)
                self.assertIsNone(restored.control.armature)
                self.assertIsNone(restored.asset.default_armature)
                self.assertFalse(restored.domain_rand.randomize_armature)
                self.assertTrue(hasattr(restored.rewards, "scales"))
                self.assertEqual(training.runner.obs_groups, train.runner.obs_groups)
                configure_nominal(restored, args)
                self.assertFalse(restored.noise.add_noise)
                args.load_run, args.run_name = None, str(path)
                self.assertEqual(task_registry.resolve_replay(args)[2], checkpoint)

    def test_real_transfer_run_does_not_need_profile_and_restores_original_reward(self):
        run = Path("logs/go2w_transfer_v1/transfer_v1_seed1_20261002_090639_2026-10-02_09-09-17").resolve()
        if not run.is_dir():
            self.skipTest("Owner run is local evidence")
        args = arguments("go2w", run, "--checkpoint", "300")
        cfg, train, checkpoint = task_registry.resolve_replay(args)
        self.assertEqual(checkpoint.name, "model_300.pt")
        self.assertEqual(cfg.go2w_profile, "transfer_v1")
        self.assertEqual(cfg.rewards.scales.orientation, -1.2)
        self.assertEqual(cfg.rewards.tracking_sigma_x, .25)
        self.assertEqual(cfg.env.num_observations, 56)
        self.assertEqual(train.runner.obs_groups, {"actor": ["policy"], "critic": ["policy"]})
        with self.assertRaisesRegex(ValueError, "conflicts"):
            task_registry.resolve_replay(arguments("go2", run))
        with self.assertRaisesRegex(ValueError, "conflicts"):
            task_registry.resolve_replay(arguments("go2w", run, "--go2w_profile", "event_step_v1"))

    def test_export_default_and_explicit_selection(self):
        from robot_gym.scripts.play import should_export_policy
        self.assertFalse(should_export_policy(arguments("dodo", "unused")))
        self.assertTrue(should_export_policy(arguments("dodo", "unused", "--export")))
        self.assertFalse(should_export_policy(arguments("go2w", "unused", "--no_export")))

    def test_play_refreshes_first_observation_after_task_state_load(self):
        from robot_gym.scripts.play import play
        cfg, train = task_registry.get_cfgs("dodo")
        args = arguments("dodo", "unused", "--steps", "1", "--headless")
        env = Mock(dt=.02, max_episode_length=1000, device="cpu")
        current = {"policy": torch.zeros(1, cfg.env.num_observations)}
        env.reset.return_value = (current.copy(), {})
        env.get_observations.side_effect = lambda: current.copy()
        env.step.return_value = (current, None, None, {})
        policy = Mock(return_value=torch.zeros(1, cfg.env.num_actions))
        runner = Mock()
        runner.get_inference_policy.return_value = policy
        def load(*unused, **kwargs):
            current["policy"] = torch.ones_like(current["policy"])
            return runner, train
        with patch.object(task_registry, "resolve_replay", return_value=(cfg, train, Path("model_1.pt"))), \
             patch.object(task_registry, "make_env", return_value=(env, cfg)), \
             patch.object(task_registry, "make_alg_runner", side_effect=load), \
             patch("robot_gym.scripts.play.gs.destroy"):
            play(args)
        self.assertTrue((policy.call_args.args[0]["policy"] == 1).all())


if __name__ == "__main__":
    unittest.main()
