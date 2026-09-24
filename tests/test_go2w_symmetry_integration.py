"""Opt-in real GPU runner: GO2W_GPU_TESTS=1 python -m unittest discover -s tests -p test_go2w_symmetry_integration.py."""

import math
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
            "--seed",
            "1",
            "--logger",
            "tensorboard",
            "--max_iterations",
            "2",
            "--experiment_name",
            "go2w_v2_2_smoke",
            "--run_name",
            "symmetry",
        ]
        with patch.object(sys, "argv", argv):
            args = get_args()
        try:
            env, _ = task_registry.make_env("go2w", args=args)
            runner, _ = task_registry.make_alg_runner(env, "go2w", args=args)
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
        finally:
            gs.destroy()


if __name__ == "__main__":
    unittest.main()
