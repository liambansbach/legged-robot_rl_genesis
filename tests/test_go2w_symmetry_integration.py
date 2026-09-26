"""Opt-in real GPU runner: GO2W_GPU_TESTS=1 python -m unittest discover -s tests -p test_go2w_symmetry_integration.py."""

import math
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import genesis as gs
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from robot_gym.envs import *  # noqa: F401,F403 - task registration
from robot_gym.envs.go2w.go2w_symmetry import sagittal_augmentation
from robot_gym.utils import get_args, task_registry


@unittest.skipUnless(
    os.environ.get("GO2W_GPU_TESTS") == "1", "opt-in Genesis GPU integration"
)
class SymmetryIntegrationTests(unittest.TestCase):
    def test_real_runner_two_updates_and_tensorboard_diagnostic(self):
        argv = [
            "symmetry-smoke",
            "--task",
            "go2w",
            "--num_envs",
            "64",
            "--headless",
            "--training_diagnostics",
            "--seed",
            "1",
            "--logger",
            "tensorboard",
            "--max_iterations",
            "2",
            "--experiment_name",
            "go2w_diagnostics_smoke",
            "--run_name",
            "symmetry",
        ]
        with patch.object(sys, "argv", argv):
            args = get_args()
        try:
            env, _ = task_registry.make_env("go2w", args=args)
            runner, _ = task_registry.make_alg_runner(env, "go2w", args=args)
            from robot_gym.utils.training_diagnostics import TrainingDiagnostics

            diagnostic_path = Path(runner.logger.log_dir) / "diagnostics.jsonl"
            TrainingDiagnostics(runner, env, diagnostic_path)
            symmetry = runner.alg.symmetry
            self.assertIs(symmetry.env, env)
            self.assertIs(symmetry.data_augmentation_func, sagittal_augmentation)
            self.assertTrue(symmetry.use_data_augmentation)
            self.assertFalse(symmetry.use_mirror_loss)
            self.assertEqual(symmetry.mirror_loss_coeff, 0.0)
            expected_names = [
                f"{side}_{joint}_joint"
                for side in ("FL", "FR", "RL", "RR")
                for joint in ("hip", "thigh", "calf", "foot")
            ]
            self.assertEqual(env.joint_names, expected_names)
            self.assertEqual(
                env.joint_dof_idx,
                [env.robot.get_joint(n).dofs_idx_local[0] for n in expected_names],
            )
            obs = env.get_observations()
            self.assertEqual(set(obs.keys()), {"policy"})
            aug_obs, aug_actions = sagittal_augmentation(
                env, obs, torch.zeros_like(env.actions)
            )
            self.assertEqual(aug_obs["policy"].shape, (128, 56))
            self.assertEqual(aug_actions.shape, (128, 16))
            torch.testing.assert_close(aug_obs[:64], obs)
            runner.learn(num_learning_iterations=2, init_at_random_ep_len=True)
            rows = [
                json.loads(line) for line in diagnostic_path.read_text().splitlines()
            ]
            self.assertEqual([r["iteration"] for r in rows], [0, 1])
            for row in rows:
                self.assertNotIn("unlabelled_initial", row["nonterminal_reward"])
                self.assertEqual(len(row["scheduler_kl_per_minibatch"]), 40)
                self.assertEqual(len(row["ppo_clip_fraction_per_minibatch"]), 40)
                self.assertEqual(len(row["action_vectors"]["raw_mean"]), 16)
                self.assertFalse(row["source"]["mirror_loss_enabled"])
            log_dir = Path(runner.logger.log_dir)
            events = EventAccumulator(str(log_dir)).Reload()
            symmetry_events = events.Scalars("Loss/symmetry")
            self.assertEqual([event.step for event in symmetry_events], [0, 1])
            self.assertTrue(
                all(
                    math.isfinite(event.value) and event.value >= 0
                    for event in symmetry_events
                )
            )
            self.assertTrue((log_dir / "model_1.pt").is_file())
            print(
                f"SYMMETRY INTEGRATION PASS: {log_dir}; Loss/symmetry={[e.value for e in symmetry_events]}",
                flush=True,
            )
            # Check fixed commands with real physics/policy observations, including resets.
            policy = runner.get_inference_policy(device=env.device)
            for command in [
                (0, 0, 0),
                (0.5, 0, 0),
                (0, 0.25, 0),
                (0, 0, 0.4),
                (0, 0, 1.0),
                (0.5, 0, 0.8),
            ]:
                env.set_fixed_command(command)
                obs, _ = env.reset()
                expected = torch.tensor(command, device=env.device).expand(
                    env.num_envs, -1
                )
                with torch.no_grad():
                    for _ in range(3):
                        torch.testing.assert_close(
                            obs["policy"][:, 9:12], expected * env.commands_scale
                        )
                        obs, _, _, _ = env.step(policy(obs))
                        torch.testing.assert_close(
                            env.commands, expected.to(env.commands.dtype)
                        )
                env.reset_idx(torch.tensor([0, 2], device=env.device))
                env.compute_observations()
                torch.testing.assert_close(
                    env.get_observations()["policy"][:, 9:12],
                    expected * env.commands_scale,
                )
            env.set_fixed_command(None)
            self.assertTrue(env.command_resampling_enabled)
            print(
                "FIXED COMMAND GPU PASS: six commands, full/selective resets, same-step policy observations",
                flush=True,
            )
        finally:
            gs.destroy()


@unittest.skipUnless(
    os.environ.get("GO2W_RESUME_SMOKE_SIGMA"), "opt-in original CK800 resume smoke"
)
class ContinuationIntegrationTests(unittest.TestCase):
    def test_original_ck800_two_additional_updates(self):
        from robot_gym.scripts.train import train
        from robot_gym.utils.training_diagnostics import verify_resume_state
        from robot_gym.utils.diagnostics import sha256, write_json
        import yaml

        sigma = float(os.environ["GO2W_RESUME_SMOKE_SIGMA"])
        self.assertIn(sigma, (0.25, 0.09))
        arm = "control" if sigma == 0.25 else "009"
        argv = [
            "resume-smoke",
            "--task",
            "go2w",
            "--num_envs",
            "64",
            "--headless",
            "--training_diagnostics",
            "--seed",
            "1",
            "--logger",
            "tensorboard",
            "--max_iterations",
            "2",
            "--resume",
            "--checkpoint",
            "800",
            "--experiment_name",
            "go2w_flat_pilot_v2_4_yaw_mobility",
            "--load_run",
            "augmentation_seed1_2026-09-25_16-27-23",
            "--run_name",
            f"xtracking_{arm}_smoke_seed1",
            "--tracking_sigma_x",
            str(sigma),
        ]
        with patch.object(sys, "argv", argv):
            args = get_args()
        try:
            runner = train(
                args
            )  # Includes exact comparison to original CK800 before learn.
            out = Path(runner.logger.log_dir)
            meta = json.loads((out / "continuation.json").read_text())
            cfg = yaml.safe_load((out / "config.yaml").read_text())
            self.assertEqual(meta["status"], "completed")
            self.assertTrue(meta["loaded_state"]["exact_state_match"])
            self.assertEqual(
                meta["parent"]["checkpoint_sha256"],
                "d0da829b95c683ce977323c4af88922c2a86ac9e680ff4501af302704c0cfcc1",
            )
            self.assertEqual(meta["completed_additional_updates"], 2)
            self.assertEqual(cfg["env_cfg"]["rewards"]["tracking_sigma_x"], sigma)
            self.assertEqual(cfg["env_cfg"]["rewards"]["tracking_sigma_y"], 0.04)
            self.assertEqual(cfg["train_cfg"]["runner"]["num_steps_per_env"], 48)
            self.assertEqual(cfg["train_cfg"]["runner"]["save_interval"], 50)
            rows = [
                json.loads(line)
                for line in (out / "diagnostics.jsonl").read_text().splitlines()
            ]
            self.assertEqual([r["iteration"] for r in rows], [800, 801])
            for row in rows:
                self.assertEqual(len(row["scheduler_kl_per_minibatch"]), 40)
                self.assertEqual(len(row["ppo_clip_fraction_per_minibatch"]), 40)
                self.assertEqual(len(row["action_vectors"]["raw_mean"]), 16)
                self.assertEqual(
                    row["nonterminal_reward"]["all"]["sample_count"], 64 * 48
                )
                self.assertFalse(row["source"]["mirror_loss_enabled"])
                json.dumps(row, allow_nan=False)
            events = EventAccumulator(str(out)).Reload()
            losses = {
                tag: [event.value for event in events.Scalars(tag)]
                for tag in events.Tags()["scalars"]
                if tag.startswith("Loss/")
            }
            self.assertTrue(losses)
            for values in losses.values():
                self.assertEqual(len(values), 2)
                self.assertTrue(all(math.isfinite(v) for v in values))
            checkpoint = Path(meta["final_checkpoint"])
            self.assertEqual(checkpoint.name, "model_801.pt")
            verify_resume_state(
                runner, checkpoint
            )  # Saved state equals completed runner.
            runner.load(checkpoint)
            reloaded = verify_resume_state(runner, checkpoint)
            for component in runner.alg.save().values():
                if isinstance(component, dict):
                    for value in component.values():
                        if torch.is_tensor(value):
                            self.assertTrue(torch.isfinite(value).all())
            # Inference only after training/reload; no extra training-time policy calls.
            with torch.no_grad():
                action = runner.get_inference_policy()(runner.env.get_observations())
            self.assertTrue(torch.isfinite(action).all())
            self.assertEqual(
                sha256(meta["parent"]["checkpoint"]),
                meta["parent"]["checkpoint_sha256"],
            )
            write_json(
                out / "smoke_validation.json",
                {
                    "argv": argv,
                    "losses": losses,
                    "save_reload": reloaded,
                    "final_checkpoint_sha256": sha256(checkpoint),
                    "finite_policy_output": True,
                },
            )
            print(
                f"CK800 RESUME SMOKE PASS: {out}; sigma_x={sigma}; two additional updates",
                flush=True,
            )
        finally:
            gs.destroy()


if __name__ == "__main__":
    unittest.main()
