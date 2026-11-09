"""Saved configurations and replay selectors, without constructing native physics."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch
import yaml
import torch

from robot_gym.envs import *  # noqa: F401,F403
from robot_gym.utils import task_registry, get_args
from robot_gym.utils.helpers import class_to_dict, update_cfg_from_args
from robot_gym.utils.replay import configure_nominal


def arguments(task, run, *extra):
    with patch.object(sys, "argv", ["play", "--task", task, "--load_run", str(run), *extra]):
        return get_args()


class ReplayTests(unittest.TestCase):
    def test_runs_own_their_mutable_settings(self):
        for task in ("dodo", "go2", "go2w"):
            with self.subTest(task=task):
                cfg, train = task_registry.get_cfgs(task)
                other, other_train = task_registry.get_cfgs(task)
                cfg.init_state.default_joint_angles.clear()
                train.actor.hidden_dims.clear()
                self.assertTrue(other.init_state.default_joint_angles)
                self.assertTrue(other_train.actor.hidden_dims)

    def test_resume_starts_from_each_packaged_configuration(self):
        packages = Path(__file__).resolve().parents[1] / "ressources" / "pretrained"
        for task in ("dodo", "go2"):
            with self.subTest(task=task):
                folder = packages / task / f"{task}_walking"
                saved = yaml.safe_load((folder / "config.yaml").read_text())
                args = arguments(task, folder, "--resume", "--checkpoint", "1499",
                                 "--num_envs", "2", "--max_iterations", "1", "--run_name", "continued")
                cfg, train = task_registry.get_cfgs(task)
                update_cfg_from_args(cfg, train, args)
                self.assertEqual(cfg.asset.robot_file, saved["env_cfg"]["asset"]["robot_file"])
                self.assertEqual(cfg.control.stiffness, saved["env_cfg"]["control"]["stiffness"])
                self.assertEqual(class_to_dict(cfg.rewards.scales), saved["env_cfg"]["rewards"]["scales"])
                self.assertEqual(train.algorithm.learning_rate, saved["train_cfg"]["algorithm"]["learning_rate"])
                self.assertEqual(train.actor.distribution_cfg["class_name"], "GaussianDistribution")
                self.assertEqual(args.resolved_checkpoint, str(folder / "model_1499.pt"))
                self.assertEqual((cfg.env.num_envs, train.runner.max_iterations, train.runner.run_name),
                                 (2, 1, "continued"))

    def test_original_walking_objective_and_reset(self):
        packages = Path(__file__).resolve().parents[1] / "ressources/pretrained"
        for task in ("dodo", "go2"):
            with self.subTest(task=task):
                folder = packages / task / f"{task}_walking"
                args = arguments(task, folder, "--checkpoint", "1499")
                cfg, train, _ = task_registry.resolve_replay(args)
                self.assertEqual(cfg.commands.stand_command_probability, 0.0)
                self.assertEqual(cfg.commands.resampling_time, 10.0)
                self.assertIsNone(cfg.commands.resampling_time_range)
                self.assertFalse(cfg.domain_rand.randomize_action_delay)
                self.assertEqual(cfg.domain_rand.action_delay_steps_range, [0, 0])
                self.assertFalse(cfg.domain_rand.push_robots)
                # A subsequent save contains the translated settings explicitly.
                with TemporaryDirectory() as directory:
                    run = Path(directory)
                    saved = dict(env_cfg=class_to_dict(cfg), train_cfg=class_to_dict(train))
                    (run / "config.yaml").write_text(yaml.safe_dump(task_registry._make_yaml_safe(saved)))
                    (run / "model_1.pt").touch()
                    cfg, _, _ = task_registry.resolve_replay(arguments(task, run, "--checkpoint", "1"))
                self.assertTrue(cfg.rewards.legacy_walking)
                self.assertEqual(cfg.init_state.relative_joint_position_noise, 0.1)

                cls = task_registry.get_task_class(task)
                env = object.__new__(cls)
                env.cfg, env.device, env.robot = cfg, "cpu", Mock()
                env.num_dof = cfg.env.num_actions
                ref = torch.tensor([[cfg.init_state.default_joint_angles[n] for n in cfg.asset.joint_names]])
                env.default_dof_pos = ref
                env.position_control_mask = torch.ones_like(ref)
                env.dof_pos = ref.repeat(3, 1)
                env.dof_vel = torch.full_like(env.dof_pos, -0.25)
                env.dof_pos_limits = torch.tensor([[-3., 3.]]).repeat(env.num_dof, 1)
                env.joint_dof_idx = list(range(env.num_dof))
                env.commands = torch.tensor([[0., 0., 0.], [0., 0., 0.05], [0.2, 0., 0.]])
                env.dof_pos[:, 0] = torch.tensor([2.8, 3.1, -3.2])
                hard_limit_cost = torch.tensor([0., 0.1, 0.2])
                torch.testing.assert_close(env._reward_dof_pos_limits(), hard_limit_cost)
                cost = (env.dof_pos - ref).abs().sum(1) + 0.1 * env.dof_vel.abs().sum(1)
                torch.testing.assert_close(env._reward_stand_still(), cost * torch.tensor([1., 1., 0.]))
                if task == "dodo":
                    env.foot_euler = torch.arange(18).reshape(3, 2, 3) * 0.01
                    error = env.foot_euler[:, :, :2].square().sum((1, 2))
                    torch.testing.assert_close(env._reward_flat_feet(),
                                               torch.exp(-error / (2 * cfg.rewards.flat_foot_sigma**2 + 1e-8)))

                cfg.domain_rand.randomize_kp = cfg.domain_rand.randomize_kd = False
                ids = torch.tensor([0, 2])
                untouched = env.dof_pos[1].clone()
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(17)
                    expected = ref * torch.empty(2, env.num_dof).uniform_(0.9, 1.1)
                    torch.manual_seed(17)
                    env._reset_dofs(ids)
                torch.testing.assert_close(env.dof_pos[ids], expected, rtol=0, atol=0)
                torch.testing.assert_close(env.dof_pos[1], untouched)
                configure_nominal(cfg, args)
                env._reset_dofs(ids)
                torch.testing.assert_close(env.dof_pos[ids], ref.expand(2, -1), rtol=0, atol=0)

    def test_each_registered_task_restores_saved_values_and_latest_once(self):
        for task in ("dodo", "go2"):
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
                # The shipped early checkpoints identify their omitted distribution
                # name through native Gaussian state; saved numeric settings win.
                saved['train_cfg']['actor']['distribution_cfg'] = dict(init_std=.23, std_type='scalar')
                (path / 'config.yaml').write_text(yaml.safe_dump(task_registry._make_yaml_safe(saved)))
                torch.save({'actor_state_dict': {'distribution.std_param': torch.full((cfg.env.num_actions,), .23)}}, checkpoint)
                legacy, legacy_training, _ = task_registry.resolve_replay(args)
                self.assertIsNone(legacy.velocity_estimator)
                self.assertEqual(legacy_training.actor.distribution_cfg,
                                 dict(class_name='GaussianDistribution', init_std=.23, std_type='scalar'))


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
