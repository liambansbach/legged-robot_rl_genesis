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
        profile = os.environ.get("GO2W_SMOKE_PROFILE")
        if profile:
            self.assertEqual(profile, "step_recovery_v1")
            argv += ["--go2w_profile", profile]
            argv[argv.index("--experiment_name") + 1] = "go2w_step_recovery_v1_smoke"
            argv[argv.index("--run_name") + 1] = "step_recovery_v1_smoke_seed1"
        with patch.object(sys, "argv", argv):
            args = get_args()
        try:
            if profile:
                from robot_gym.scripts.train import train

                # Capture pre-update state without forwarding the policy or consuming RNG.
                original = task_registry.make_alg_runner

                def make_runner(*args, **kwargs):
                    runner, cfg = original(*args, **kwargs)
                    self.assertIsNone(runner.checkpoint_path)
                    self.assertEqual(runner.current_learning_iteration, 0)
                    self.assertFalse(runner.alg.optimizer.state)
                    self.assertEqual(runner.alg.learning_rate, 8e-4)
                    self.assertEqual(runner.alg.entropy_coef, 0.001)
                    self.assertEqual(runner.cfg["num_steps_per_env"], 48)
                    torch.testing.assert_close(
                        runner.alg.actor.distribution.log_std_param.exp(),
                        torch.full((16,), 0.35, device=runner.device),
                    )
                    for model in (runner.alg.actor, runner.alg.critic):
                        self.assertEqual(model.obs_normalizer.count, 0)
                        self.assertEqual(model.obs_normalizer._mean.count_nonzero(), 0)
                        torch.testing.assert_close(
                            model.obs_normalizer._var,
                            torch.ones_like(model.obs_normalizer._var),
                        )
                    self.assertFalse(hasattr(runner.env, "physics_diagnostics"))
                    self.assertFalse(hasattr(runner.env, "zero_command_brake"))
                    # Check the live geometry/reward throughout this one rollout.
                    reward = runner.env._reward_foot_swing_clearance

                    def checked_reward():
                        from robot_gym.utils.diagnostics import cylinder_clearance

                        e = runner.env
                        expected = cylinder_clearance(
                            e.foot_pos,
                            e.robot.get_links_quat(e.foot_link_indices_local),
                            *e.wheel_geometry,
                        )
                        torch.testing.assert_close(e.wheel_clearance, expected)
                        for value in (
                            e.obs_buf,
                            e.dof_pos,
                            e.dof_vel,
                            e.wheel_reposition_velocity_body,
                            e.wheel_normal_force,
                        ):
                            self.assertTrue(torch.isfinite(value).all())
                        result = reward()
                        self.assertTrue(torch.isfinite(result).all())
                        self.assertTrue(((result >= 0) & (result <= 1)).all())
                        return result

                    index = runner.env.reward_names.index("foot_swing_clearance")
                    runner.env.reward_functions[index] = checked_reward
                    return runner, cfg

                with patch.object(task_registry, "make_alg_runner", make_runner):
                    runner = train(args)
                env = runner.env
            else:
                env, _ = task_registry.make_env("go2w", args=args)
                runner, _ = task_registry.make_alg_runner(env, "go2w", args=args)
            from robot_gym.utils.training_diagnostics import TrainingDiagnostics

            diagnostic_path = Path(runner.logger.log_dir) / "diagnostics.jsonl"
            if not profile:
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
            if not profile:
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
            if profile:
                from robot_gym.utils.training_diagnostics import verify_resume_state
                from robot_gym.utils.diagnostics import sha256, write_json
                import yaml

                self.assertEqual(env.num_envs, 64)
                self.assertEqual(env.num_obs, 56)
                self.assertEqual(env.num_actions, 16)
                self.assertFalse(hasattr(env, "physics_diagnostics"))
                metadata = json.loads((log_dir / "preparation.json").read_text())
                self.assertEqual(metadata["completed_updates"], 2)
                self.assertEqual(metadata["status"], "completed")
                cfg = yaml.safe_load((log_dir / "config.yaml").read_text())
                self.assertEqual(cfg["env_cfg"]["go2w_profile"], profile)
                self.assertEqual(cfg["train_cfg"]["runner"]["save_interval"], 250)
                for row in rows:
                    json.dumps(row, allow_nan=False)
                    self.assertEqual(
                        row["nonterminal_reward"]["all"]["sample_count"], 64 * 48
                    )
                    self.assertAlmostEqual(
                        sum(
                            v["fraction"] for v in row["command_time_exposure"].values()
                        ),
                        1,
                    )
                losses = {
                    tag: [e.value for e in events.Scalars(tag)]
                    for tag in events.Tags()["scalars"]
                    if tag.startswith("Loss/")
                }
                self.assertTrue(losses)
                for values in losses.values():
                    self.assertEqual(len(values), 2)
                    self.assertTrue(all(math.isfinite(v) for v in values))
                checkpoint = log_dir / "model_1.pt"
                verify_resume_state(runner, checkpoint)
                runner.load(checkpoint)
                reloaded = verify_resume_state(runner, checkpoint)
                with torch.no_grad():
                    action = runner.get_inference_policy()(env.get_observations())
                self.assertTrue(torch.isfinite(action).all())
                write_json(
                    log_dir / "smoke_validation.json",
                    {
                        "argv": argv,
                        "completed_updates": 2,
                        "losses": losses,
                        "save_reload": reloaded,
                        "final_checkpoint_sha256": sha256(checkpoint),
                        "geometry_reward_without_physics_recorder": True,
                        "fresh_learning_state_verified": True,
                    },
                )
                print(
                    f"STEP RECOVERY SMOKE PASS: {log_dir}; two fresh updates",
                    flush=True,
                )
                return
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
    os.environ.get("GO2W_RESUME_SMOKE_SIGMA") or os.environ.get("GO2W_RESUME_SMOKE_FINETUNE"),
    "opt-in original checkpoint resume smoke",
)
class ContinuationIntegrationTests(unittest.TestCase):
    def precision_dr_readback(self, args):
        from robot_gym.utils.diagnostics import loaded_properties

        env, _ = task_registry.make_env("go2w", args=args)
        self.assertEqual(tuple(env.action_history.shape), (64, 3, 16))
        before = loaded_properties(env)
        delays = env.action_delay_steps.clone()
        self.assertEqual(set(delays.tolist()), {0, 1, 2})
        self.assertTrue(torch.all((before["effective_friction"] >= .6) & (before["effective_friction"] <= 1.2)))
        for key, nominal in (("kp", env.base_p_gains), ("kv", env.base_d_gains)):
            active = nominal != 0
            factors = before[key][:, active] / nominal[active]
            self.assertTrue(torch.all((factors >= .85) & (factors <= 1.15)))
            torch.testing.assert_close(factors, factors[:, :1].expand_as(factors))
        env.reset_idx(torch.tensor([0], device=env.device))
        after = loaded_properties(env)
        for key in ("mass", "com_local", "inertia_local", "wheel_friction_ratio"):
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
        for key in ("kp", "kv"):
            torch.testing.assert_close(before[key][1:], after[key][1:], rtol=0, atol=0)
        torch.testing.assert_close(delays[1:], env.action_delay_steps[1:], rtol=0, atol=0)
        env.cfg.env.capture_transitions = True
        env.cfg.env.capture_closed_loop = True
        env.action_delay_steps[:3] = torch.arange(3, device=env.device)
        env.action_history.zero_()
        for tick in range(3):
            env.step(torch.full_like(env.actions, (tick + 1) * .1))
            expected = torch.tensor([max(0, tick + 1 - delay) * .1 for delay in range(3)], device=env.device)
            torch.testing.assert_close(env.transition_state["applied_actions"][:3], expected[:, None].expand(-1, 16))
        env.reset_idx(torch.tensor([0], device=env.device))
        self.assertTrue((env.action_history[0] == 0).all())
        report = {"before": before, "after_selective_reset": after,
                  "sampled_delay": delays, "three_tick_history_verified": True,
                  "initialization": "Auxiliary simulator destroyed; train rebuilds and reseeds before native load"}
        gs.destroy()
        from robot_gym.envs.base.base_task import BaseTask
        BaseTask._gs_initialized = False
        BaseTask._gs_backend = None
        return report

    def test_original_checkpoint_two_additional_updates(self):
        from robot_gym.scripts.train import train
        from robot_gym.utils.training_diagnostics import verify_resume_state
        from robot_gym.utils.diagnostics import sha256, write_json
        import yaml

        finetune = os.environ.get("GO2W_RESUME_SMOKE_FINETUNE")
        sigma = float(os.environ.get("GO2W_RESUME_SMOKE_SIGMA", "0.25"))
        precision = finetune == "precision_clearance"
        iteration = 1798 if precision else 1499 if finetune else 800
        parent_hash = (
            "02fe2fe814b2b82e35253097b2ff8976cf699b0f42be06dd70c03f202f2eeb25" if precision else
            "080ac621c286250fa959d269a5e6299761af25a850fbbd449155a6f13f315ae7" if finetune
            else "d0da829b95c683ce977323c4af88922c2a86ac9e680ff4501af302704c0cfcc1"
        )
        self.assertIn(sigma, (0.25, 0.09))
        arm = "control" if sigma == 0.25 else "009"
        entropy = os.environ.get("GO2W_RESUME_SMOKE_ENTROPY")
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
        if entropy is not None:
            self.assertEqual(float(entropy), 0.001)
            self.assertEqual(sigma, 0.25)
            argv[argv.index("--run_name") + 1] = "entropy_001_smoke_seed1"
            argv += ["--entropy_coef", entropy]
        if finetune:
            self.assertIn(finetune, ("coverage", "coverage_mobility", "precision_clearance"))
            self.assertIsNone(entropy)
            argv[argv.index("--checkpoint") + 1] = str(iteration)
            argv[argv.index("--experiment_name") + 1] = "go2w_step_recovery_v1"
            argv[argv.index("--load_run") + 1] = "step_recovery_v1_seed1_20260926_165356_2026-09-26_16-55-18"
            argv[argv.index("--run_name") + 1] = f"{finetune}_smoke_seed1_{os.environ['GO2W_COMPARISON_TAG']}"
            argv += ["--go2w_profile", "step_recovery_v1", "--go2w_finetune", finetune]
            if precision:
                argv[argv.index("--load_run") + 1] = "coverage_seed1_20260926_231000_2026-09-26_23-15-39"
        with patch.object(sys, "argv", argv):
            args = get_args()
        try:
            dr_readback = self.precision_dr_readback(args) if precision else None
            runner = train(
                args
            )  # Includes exact comparison to the original parent before learn.
            out = Path(runner.logger.log_dir)
            meta = json.loads((out / "continuation.json").read_text())
            cfg = yaml.safe_load((out / "config.yaml").read_text())
            self.assertEqual(meta["status"], "completed")
            self.assertTrue(meta["loaded_state"]["exact_state_match"])
            self.assertEqual(
                meta["parent"]["checkpoint_sha256"],
                parent_hash,
            )
            self.assertEqual(meta["completed_additional_updates"], 2)
            if finetune:
                self.assertEqual(meta["parent"]["saved_config"]["sha256"],
                                 "8f88bb3ab44423e01aa3c2b7a19be9f49e82bd377862315fa47825627bb644d7" if precision else
                                 "916691c276ba5112fe947b00e5c59b527d0390fa68596421e97906c827a501d6")
                self.assertEqual(cfg["env_cfg"]["go2w_finetune"], finetune)
                self.assertEqual(meta["explicit_overrides"]["go2w_finetune"], finetune)
                self.assertEqual(runner.alg.entropy_coef, 0.001)
                self.assertTrue(torch.isfinite(runner.env.obs_buf).all())
                self.assertTrue(torch.isfinite(runner.env.rew_buf).all())
            if entropy is not None:
                self.assertEqual(runner.alg.entropy_coef, 0.001)
                self.assertEqual(meta["entropy_coef"], 0.001)
                self.assertEqual(meta["explicit_overrides"]["entropy_coef"], 0.001)
                self.assertEqual(cfg["train_cfg"]["algorithm"]["entropy_coef"], 0.001)
            self.assertEqual(cfg["env_cfg"]["rewards"]["tracking_sigma_x"], sigma)
            self.assertEqual(cfg["env_cfg"]["rewards"]["tracking_sigma_y"], 0.04)
            self.assertEqual(cfg["train_cfg"]["runner"]["num_steps_per_env"], 48)
            self.assertEqual(cfg["train_cfg"]["runner"]["save_interval"], 250 if finetune else 50)
            rows = [
                json.loads(line)
                for line in (out / "diagnostics.jsonl").read_text().splitlines()
            ]
            self.assertEqual([r["iteration"] for r in rows], [iteration, iteration + 1])
            for row in rows:
                self.assertEqual(len(row["scheduler_kl_per_minibatch"]), 40)
                self.assertEqual(len(row["ppo_clip_fraction_per_minibatch"]), 40)
                self.assertEqual(len(row["action_vectors"]["raw_mean"]), 16)
                self.assertEqual(
                    row["nonterminal_reward"]["all"]["sample_count"], 64 * 48
                )
                self.assertFalse(row["source"]["mirror_loss_enabled"])
                if finetune:
                    self.assertIn("coverage_time_exposure", row)
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
            self.assertEqual(checkpoint.name, f"model_{iteration + 1}.pt")
            verify_resume_state(
                runner, checkpoint
            )  # Saved state equals completed runner.
            runner.load(checkpoint)
            reloaded = verify_resume_state(runner, checkpoint)
            if entropy is not None:
                self.assertEqual(runner.alg.entropy_coef, 0.001)
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
                    "dr_readback": dr_readback,
                },
            )
            print(
                f"CK{iteration} RESUME SMOKE PASS: {out}; finetune={finetune}; sigma_x={sigma}; two additional updates",
                flush=True,
            )
        finally:
            gs.destroy()


if __name__ == "__main__":
    unittest.main()
