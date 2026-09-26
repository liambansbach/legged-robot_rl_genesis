"""CPU contracts for diagnostics: physical geometry, provenance and no new RNG draws."""

import copy
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import genesis as gs
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml
from genesis.utils.geom import xyz_to_quat, quat_to_xyz

from robot_gym import ROBOT_GYM_ROOT_DIR
from robot_gym.envs.go2w.go2w_config import GO2WCfg, GO2WCfgPPO
from robot_gym.utils import task_registry
from robot_gym.utils.helpers import class_to_dict, get_args, update_cfg_from_args
from robot_gym.utils.diagnostics import (
    summed_normal_force,
    cylinder_clearance,
    wheel_cylinders,
    heading_wxyz,
    check_reference_contract,
    check_training_continuation,
    check_continuation_output,
    manifest,
    PhysicsDiagnostics,
    write_json,
    nominal_support_heights,
)
from robot_gym.utils.training_diagnostics import (
    std_parameters,
    TrainingDiagnostics,
    verify_resume_state,
)
from robot_gym.scripts.diagnostic_bank import condition_bank
from rsl_rl.modules.distribution import GaussianDistribution


URDF = Path(ROBOT_GYM_ROOT_DIR) / "ressources/robots/go2w/urdf/go2w_description.urdf"


class DiagnosticsTests(unittest.TestCase):
    def test_entropy_override_preserves_defaults_and_validates_values(self):
        for value in (None, "0", "0.001", "0.005", "-0.001", "nan", "inf", "-inf"):
            argv = ["train", "--task", "go2w"]
            if value is not None:
                argv += [f"--entropy_coef={value}"]
            with patch.object(sys, "argv", argv):
                args = get_args()
            env, train = task_registry.get_cfgs("go2w")
            before = class_to_dict(train)
            if value in ("-0.001", "nan", "inf", "-inf"):
                with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                    update_cfg_from_args(env, train, args)
            else:
                update_cfg_from_args(env, train, args)
                self.assertEqual(
                    train.algorithm.entropy_coef,
                    0.005 if value is None else float(value),
                )
                if value is None:
                    self.assertIsNone(args.entropy_coef)
                    self.assertEqual(before, class_to_dict(train))
            self.assertEqual(
                task_registry.get_cfgs("go2w")[1].algorithm.entropy_coef, 0.005
            )
            self.assertEqual(env.rewards.tracking_sigma_x, 0.25)
        args.task, args.entropy_coef = "go2", 0.001
        with self.assertRaisesRegex(ValueError, "specific to go2w"):
            update_cfg_from_args(env, train, args)

    def test_entropy_continuation_exception_requires_exact_explicit_request(self):
        env, train = map(class_to_dict, task_registry.get_cfgs("go2w"))
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.yaml"
            config.write_text(yaml.safe_dump(dict(env_cfg=env, train_cfg=train)))
            train["algorithm"]["entropy_coef"] = 0.001
            for requested in (None, 0.005):
                with self.assertRaisesRegex(ValueError, "entropy_coef"):
                    check_training_continuation(
                        config, env, train, entropy_coef=requested
                    )
            check_training_continuation(config, env, train, entropy_coef=0.001)
            train["algorithm"]["learning_rate"] = 0.001
            with self.assertRaisesRegex(ValueError, "learning_rate"):
                check_training_continuation(config, env, train, entropy_coef=0.001)

    def test_x_override_default_validation_and_registry_isolation(self):
        for value in (None, "0.25", "0.09", "0", "-0.1", "nan", "inf", "-inf"):
            argv = ["train", "--task", "go2w"]
            if value is not None:
                argv += [f"--tracking_sigma_x={value}"]
            with patch.object(sys, "argv", argv):
                args = get_args()
            env, cfg = task_registry.get_cfgs("go2w")
            before = class_to_dict(env)
            if value in ("0", "-0.1", "nan", "inf", "-inf"):
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    update_cfg_from_args(env, cfg, args)
            else:
                update_cfg_from_args(env, cfg, args)
                self.assertEqual(
                    env.rewards.tracking_sigma_x,
                    0.25 if value is None else float(value),
                )
                self.assertEqual(env.rewards.tracking_sigma_y, 0.04)
                if value is None:
                    self.assertIsNone(args.tracking_sigma_x)
                    self.assertEqual(before, class_to_dict(env))
            self.assertEqual(
                task_registry.get_cfgs("go2w")[0].rewards.tracking_sigma_x, 0.25
            )
        args.task, args.tracking_sigma_x = "go2", 0.09
        with self.assertRaisesRegex(ValueError, "specific to go2w"):
            update_cfg_from_args(env, cfg, args)

    def test_training_contract_rejects_unexplained_behavior_changes(self):
        env, train = map(class_to_dict, task_registry.get_cfgs("go2w"))
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.yaml"
            config.write_text(yaml.safe_dump(dict(env_cfg=env, train_cfg=train)))
            current_env, current_train = copy.deepcopy(env), copy.deepcopy(train)
            current_env["rewards"]["tracking_sigma_x"] = 0.09
            current_env["env"]["record_command_families"] = True
            current_env["sim"]["batch_dofs_info"] = True
            current_train["runner"].update(
                run_name="xtracking_009_seed1",
                resume=True,
                load_run="original",
                checkpoint=800,
                max_iterations=300,
            )
            check_training_continuation(config, current_env, current_train, 0.09)
            with self.assertRaisesRegex(ValueError, "tracking_sigma_x"):
                check_training_continuation(config, current_env, current_train)
            changes = [
                ("env_cfg.rewards.tracking_sigma_y", 0.09),
                ("env_cfg.rewards.scales.default_pose", -0.1),
                ("env_cfg.commands.stand_command_probability", 0.2),
                ("env_cfg.domain_rand.push_robots", False),
                ("env_cfg.init_state.joint_position_noise", 0.0),
                ("env_cfg.noise.add_noise", False),
                ("env_cfg.sim.deterministic", True),
                ("env_cfg.sim.performance_mode", False),
                ("train_cfg.algorithm.entropy_coef", 0.001),
                ("train_cfg.algorithm.learning_rate", 0.001),
                ("train_cfg.runner.num_steps_per_env", 24),
                ("train_cfg.runner.save_interval", 25),
                ("train_cfg.seed", 2),
            ]
            for path, value in changes:
                changed = copy.deepcopy(
                    dict(env_cfg=current_env, train_cfg=current_train)
                )
                node = changed
                keys = path.split(".")
                for key in keys[:-1]:
                    node = node[key]
                node[keys[-1]] = value
                with self.subTest(path=path), self.assertRaises(ValueError):
                    check_training_continuation(
                        config, changed["env_cfg"], changed["train_cfg"], 0.09
                    )
            current_env["env"]["num_envs"] = 64
            with self.assertRaisesRegex(ValueError, "num_envs"):
                check_training_continuation(config, current_env, current_train, 0.09)
            current_train["runner"].update(max_iterations=2, logger="tensorboard")
            check_training_continuation(config, current_env, current_train, 0.09)
            with self.assertRaisesRegex(ValueError, "Reference config missing"):
                check_training_continuation(Path(tmp) / "missing.yaml", env, train)

    def test_continuation_requires_explicit_parent_and_separate_fresh_output(self):
        from robot_gym.scripts.train import prepare_go2w_continuation

        with patch.object(sys, "argv", ["train", "--task", "go2w", "--resume"]):
            args = get_args()
        env, train = task_registry.get_cfgs("go2w")
        for run, checkpoint in ((None, 800), ("-1", 800), ("original", -1)):
            args.load_run, args.checkpoint = run, checkpoint
            with self.assertRaisesRegex(ValueError, "explicit"):
                prepare_go2w_continuation(args, env, train)
        args.load_run, args.checkpoint = "original", 800
        with self.assertRaisesRegex(ValueError, "new --run_name"):
            prepare_go2w_continuation(args, env, train)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args.run_name = "xtracking_control_seed1"
            with (
                patch("robot_gym.ROBOT_GYM_ROOT_DIR", tmp),
                self.assertRaises(ValueError),
            ):
                prepare_go2w_continuation(args, env, train)
            parent = root / "original"
            parent.mkdir()
            checkpoint = parent / "model_800.pt"
            for output in (parent, parent / "child", root):
                with self.assertRaises(ValueError):
                    check_continuation_output(checkpoint, output)
            check_continuation_output(checkpoint, root / "xtracking_control_seed1_new")

    def test_fk_uses_cpu_even_if_torch_default_device_changes(self):
        cfg = GO2WCfg()
        with torch.device("meta"):
            heights = nominal_support_heights(
                URDF, cfg.init_state.default_joint_angles, cfg.asset.foot_link_names
            )
        np.testing.assert_allclose(
            heights, [0.431848824, 0.431848824, 0.433668286, 0.433668286], atol=1e-7
        )

    def test_contact_total_is_invariant_to_point_splitting(self):
        def contacts(values):
            count = len(values)
            forces = torch.zeros(1, count, 3)
            forces[0, :, 2] = torch.tensor(values)
            return dict(
                link_a=torch.full((1, count), 4),
                link_b=torch.zeros(1, count, dtype=torch.long),
                force_a=forces,
                force_b=-forces,
                valid_mask=torch.ones(1, count, dtype=torch.bool),
            )

        for forces in ([12.0], [6.0, 6.0], [3.0, 3.0, 3.0, 3.0]):
            c = contacts(forces)
            torch.testing.assert_close(
                summed_normal_force(c, torch.tensor([4, 5])),
                torch.tensor([[12.0, 0.0]]),
            )
        # The retained legacy predicate intentionally differs under point splitting.
        self.assertTrue((contacts([12.0])["force_a"][..., 2] > 8).any())
        self.assertFalse((contacts([6.0, 6.0])["force_a"][..., 2] > 8).any())
        c = contacts([12.0, 99.0])
        c["valid_mask"][0, 1] = False
        torch.testing.assert_close(
            summed_normal_force(c, torch.tensor([4])), torch.tensor([[12.0]])
        )

    def test_finite_cylinder_signed_camber_offset_heading_and_spin(self):
        geometry = wheel_cylinders(URDF, GO2WCfg().asset.foot_link_names)
        offsets, axes, radii, widths = geometry
        torch.testing.assert_close(
            offsets[:, 1], torch.tensor([0.0481, -0.0481, 0.0481, -0.0481])
        )
        for camber in (-0.3, 0.0, 0.3):
            for heading in (0.0, 0.7, -2.0):
                # Heading then camber; wheel spin about local y leaves the envelope invariant.
                for spin in (0.0, 1.3):
                    rot = (
                        Rotation.from_euler("z", heading)
                        * Rotation.from_euler("x", camber)
                        * Rotation.from_euler("y", spin)
                    )
                    q = torch.tensor(
                        rot.as_quat()[[3, 0, 1, 2]], dtype=torch.float32
                    ).expand(4, -1)
                    p = torch.tensor([0.0, 0.0, 0.2]).expand(4, -1)
                    expected = (
                        0.2
                        + offsets[:, 1] * np.sin(camber)
                        - radii * np.cos(camber)
                        - widths * abs(np.sin(camber))
                    )
                    torch.testing.assert_close(
                        cylinder_clearance(p, q, *geometry),
                        expected,
                        atol=1e-7,
                        rtol=1e-6,
                    )

    def test_genesis_intrinsic_and_ros_extrinsic_conventions(self):
        angles = torch.tensor([[0.2, -0.3, 0.7]], dtype=torch.float64)
        q = xyz_to_quat(angles)
        expected = Rotation.from_euler("XYZ", angles.numpy()).as_quat()[:, [3, 0, 1, 2]]
        np.testing.assert_allclose(q.numpy(), expected, atol=1e-12)
        self.assertGreater(
            np.max(
                abs(
                    expected
                    - Rotation.from_euler("xyz", angles.numpy()).as_quat()[
                        :, [3, 0, 1, 2]
                    ]
                )
            ),
            0.01,
        )
        with patch.object(gs, "EPS", 1e-12, create=True):
            torch.testing.assert_close(quat_to_xyz(q), angles)
        matrix = Rotation.from_quat(q.numpy()[:, [1, 2, 3, 0]]).as_matrix()
        np.testing.assert_allclose(
            heading_wxyz(q).numpy(),
            np.arctan2(matrix[:, 1, 0], matrix[:, 0, 0]),
            atol=1e-12,
        )
        self.assertGreater(abs(float(heading_wxyz(q)[0]) - 0.7), 0.01)

    def test_reference_contract_fails_on_physics_and_observation_changes(self):
        env, train = class_to_dict(GO2WCfg()), class_to_dict(GO2WCfgPPO())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            # JSON roundtrip also removes tuple YAML tags, like registry's saved config.
            path.write_text(
                yaml.safe_dump(
                    json.loads(json.dumps(dict(env_cfg=env, train_cfg=train)))
                )
            )
            check_reference_contract(path, env, train)
            for section, key, value in (
                ("control", "wheel_velocity_target_limit", 19.0),
                ("sim", "dt", 0.01),
                ("normalization", "clip_actions", 2.0),
                ("env", "num_observations", 57),
            ):
                changed = copy.deepcopy(env)
                changed[section][key] = value
                with self.assertRaisesRegex(ValueError, "contract mismatch"):
                    check_reference_contract(path, changed, train)
            env["rewards"]["tracking_sigma_x"] = 0.09
            self.assertIn(
                "env_cfg.rewards.tracking_sigma_x",
                check_reference_contract(path, env, train)["differences"],
            )
            with self.assertRaisesRegex(ValueError, "Reference config missing"):
                check_reference_contract(Path(tmp) / "missing.yaml", env, train)

    def test_manifest_is_explicit_and_json_serializable(self):
        with tempfile.TemporaryDirectory() as tmp:
            ck = Path(tmp) / "model.pt"
            ck.write_bytes(b"checkpoint")
            m = manifest(
                ROBOT_GYM_ROOT_DIR, ck, URDF, {"a": 1}, {"b": 2}, {"seed": 1}, {}
            )
            for key in (
                "checkpoint",
                "source",
                "urdf",
                "versions",
                "resolved_env_config",
                "resolved_train_config",
                "eval_overrides",
                "conventions",
            ):
                self.assertIn(key, m)
            self.assertEqual(len(m["checkpoint"]["sha256"]), 64)
            self.assertEqual(len(m["source"]["dirty_diff_sha256"]), 64)
            write_json(Path(tmp) / "manifest.json", m)
            self.assertIn("NOT ROS", m["conventions"]["base_rpy_legacy"])

    def test_trace_fields_and_substep_max_do_not_draw_randomness(self):
        cfg = GO2WCfg()
        e = SimpleNamespace(
            cfg=cfg,
            device="cpu",
            urdf_reader=SimpleNamespace(robot_file_path_absolute=URDF),
            robot=Mock(),
            joint_dof_idx=list(range(16)),
            foot_link_indices_local=[0, 1, 2, 3],
            leg_action_indices=[i for i in range(16) if i % 4 != 3],
            wheel_action_indices=[3, 7, 11, 15],
            applied_actions=torch.zeros(1, 16),
            action_scale=torch.ones(16),
            default_dof_pos=torch.zeros(1, 16),
            dof_vel_limits=torch.full((16,), 30.0),
            torques=torch.ones(1, 16),
            foot_contacts=torch.ones(1, 4, dtype=torch.bool),
            foot_pos=torch.zeros(1, 4, 3),
            base_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            base_pos=torch.zeros(1, 3),
            dof_pos=torch.zeros(1, 16),
            dof_vel=torch.zeros(1, 16),
            base_lin_vel=torch.zeros(1, 3),
            base_ang_vel=torch.zeros(1, 3),
        )
        e.robot.get_links_quat.return_value = e.base_quat[:, None].expand(-1, 4, -1)
        probe = PhysicsDiagnostics(e)
        rng = torch.get_rng_state().clone()
        probe.begin_step(torch.zeros(1, 16))
        for value in (1.0, -4.0, 2.0, 3.0):
            e.robot.get_dofs_control_force.return_value = torch.full((1, 16), value)
            probe.after_substep()
        probe.normal_force = torch.ones(1, 4)
        trace = probe.capture()
        self.assertEqual(trace["control_torques_substep_max_abs"].max(), 4.0)
        for key in (
            "raw_actions",
            "leg_position_targets",
            "wheel_velocity_targets",
            "control_torques",
            "summed_normal_ground_force",
            "wheel_clearance",
            "base_quat_wxyz",
            "wheel_link_quat_wxyz",
        ):
            self.assertIn(key, trace)
        torch.testing.assert_close(torch.get_rng_state(), rng)
        self.assertNotIn("zero_command_brake_alpha", trace)
        from robot_gym.envs.go2w.zero_command_brake import ZeroCommandBrake

        e.zero_command_brake = ZeroCommandBrake(1, e.wheel_action_indices, .02, "cpu")
        raw = torch.full((1, 16), 4.0)
        e.actions = e.zero_command_brake.apply(raw, torch.zeros(1, 3))
        probe.begin_step(raw)
        probe.after_substep()
        # The delayed command is still the previous zero action.
        trace = probe.capture()
        torch.testing.assert_close(trace["raw_actions"], raw)
        torch.testing.assert_close(trace["issued_actions"], e.actions)
        torch.testing.assert_close(trace["zero_command_brake_alpha"], torch.tensor([.1]))
        self.assertEqual(trace["applied_actions"].count_nonzero(), 0)
        self.assertEqual(trace["wheel_velocity_targets"].count_nonzero(), 0)
        torch.testing.assert_close(trace["leg_position_targets"], torch.zeros(1, 12))
        torch.testing.assert_close(torch.get_rng_state(), rng)

    def test_bank_is_explicit_reproducible_and_does_not_change_global_rng(self):
        np.random.seed(123)
        before = np.random.get_state()
        bank = condition_bank()
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(bank, condition_bank())
        self.assertEqual(len(bank), 32)
        self.assertEqual(len({c["id"] for c in bank}), 32)
        self.assertEqual({c["delay"] for c in bank}, {0, 1})
        self.assertGreater(len({c["friction"] for c in bank}), 20)

    def test_std_reports_raw_outside_forward_clamp(self):
        distribution = GaussianDistribution(16, std_type="log", std_range=(0.05, 0.8))
        with torch.no_grad():
            distribution.log_std_param.fill_(1.0)
        summary = std_parameters(distribution)
        torch.testing.assert_close(summary["raw_log_std"], torch.ones(16))
        torch.testing.assert_close(summary["effective_std"], torch.full((16,), 0.8))
        self.assertTrue(summary["strictly_outside_bounds"].all())
        distribution.update(torch.zeros(1, 16))
        distribution.entropy.sum().backward()
        self.assertEqual(distribution.log_std_param.grad.abs().max(), 0)

    def test_ppo_instrumentation_preserves_rng_losses_and_parameters(self):
        from rsl_rl.algorithms import PPO
        from rsl_rl.models import MLPModel
        from rsl_rl.storage import RolloutStorage
        from tensordict import TensorDict

        def run(path=None):
            torch.manual_seed(51)
            obs = TensorDict({"policy": torch.randn(4, 56)}, batch_size=[4])
            groups = {"actor": ["policy"], "critic": ["policy"]}
            actor = MLPModel(
                obs,
                groups,
                "actor",
                16,
                hidden_dims=[16],
                obs_normalization=True,
                distribution_cfg=dict(
                    class_name="GaussianDistribution",
                    std_type="log",
                    std_range=[0.05, 0.8],
                ),
            )
            critic = MLPModel(
                obs, groups, "critic", 1, hidden_dims=[16], obs_normalization=True
            )
            storage = RolloutStorage("rl", 4, 4, obs, [16])
            alg = PPO(actor, critic, storage, num_learning_epochs=2, num_mini_batches=2)
            env = SimpleNamespace(
                cfg=GO2WCfg(),
                device="cpu",
                num_envs=4,
                dt=0.02,
                action_scale=torch.tensor([0.2, 0.2, 0.2, 18.0] * 4),
                joint_names=list(GO2WCfg().init_state.default_joint_angles),
                leg_action_indices=[i for i in range(16) if i % 4 != 3],
                wheel_action_indices=[3, 7, 11, 15],
            )
            if path:
                observer = TrainingDiagnostics(
                    SimpleNamespace(alg=alg, current_learning_iteration=0), env, path
                )
                observer.command_families[:] = torch.tensor([0, 1, 2, 3])
            with torch.no_grad():
                for _ in range(4):
                    alg.act(obs)
                    reward = torch.tensor([-0.2, 0.1, -0.1, 0.4])
                    if path:
                        observer.reward(reward, torch.zeros(4, 3))
                    alg.process_env_step(
                        obs, reward.clamp_min(0), torch.zeros(4, dtype=torch.bool), {}
                    )
                alg.compute_returns(obs)
            losses = alg.update()
            return (
                losses,
                actor.state_dict(),
                critic.state_dict(),
                torch.get_rng_state(),
                alg,
            )

        with tempfile.TemporaryDirectory() as tmp:
            baseline = run()
            path = Path(tmp) / "diagnostics.jsonl"
            instrumented = run(path)
            self.assertEqual(baseline[0], instrumented[0])
            for a, b in zip(baseline[1:3], instrumented[1:3]):
                for key in a:
                    self.assertTrue(torch.equal(a[key], b[key]), key)
            self.assertTrue(torch.equal(baseline[3], instrumented[3]))
            row = json.loads(path.read_text())
            self.assertEqual(len(row["scheduler_kl_per_minibatch"]), 4)
            self.assertEqual(len(row["ppo_clip_fraction_per_minibatch"]), 4)
            self.assertEqual(len(row["action_vectors"]["raw_mean"]), 16)
            self.assertAlmostEqual(
                row["nonterminal_reward"]["all"]["negative_fraction"], 0.5
            )
            self.assertAlmostEqual(
                row["nonterminal_reward"]["all"][
                    "discarded_negative_magnitude_per_sample"
                ],
                0.075,
            )

            # Exercise the native full-state loader with populated Adam state.
            from rsl_rl.runners import OnPolicyRunner

            runner = OnPolicyRunner.__new__(OnPolicyRunner)
            runner.alg, runner.logger = baseline[4], Mock()
            runner.current_learning_iteration = 800
            runner.alg.learning_rate = 0.0003844335937499999
            runner.alg.optimizer.param_groups[0]["lr"] = runner.alg.learning_rate
            with torch.no_grad():
                runner.alg.actor.distribution.log_std_param[0] = 1.0
                runner.alg.actor.obs_normalizer.count.fill_(1234)
                runner.alg.critic.obs_normalizer.count.fill_(789)
            checkpoint = Path(tmp) / "model_800.pt"
            runner.save(checkpoint)
            with torch.no_grad():
                for model in (runner.alg.actor, runner.alg.critic):
                    for tensor in model.state_dict().values():
                        tensor.zero_()
            runner.alg.optimizer.state.clear()
            runner.alg.optimizer.param_groups[0]["lr"] = 0.0008
            runner.alg.learning_rate = 0.0008
            runner.current_learning_iteration = 0
            runner.load(checkpoint, map_location="cpu")
            rng = torch.get_rng_state().clone()
            verified = verify_resume_state(runner, checkpoint)
            self.assertEqual(verified["source_iteration"], 800)
            self.assertEqual(verified["loaded_learning_rate"], 0.0003844335937499999)
            self.assertEqual(runner.alg.actor.distribution.log_std_param[0], 1.0)
            self.assertEqual(runner.alg.actor.obs_normalizer.count, 1234)
            self.assertEqual(runner.alg.critic.obs_normalizer.count, 789)
            self.assertGreater(len(runner.alg.optimizer.state), 0)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            mutations = [
                lambda: setattr(runner.alg, "learning_rate", 0.0008),
                lambda: runner.alg.optimizer.param_groups[0].update(lr=0.0008),
                lambda: runner.alg.actor.distribution.log_std_param.fill_(-0.3),
                lambda: runner.alg.actor.obs_normalizer.count.zero_(),
                lambda: runner.alg.critic.obs_normalizer._mean.fill_(2),
                lambda: next(iter(runner.alg.optimizer.state.values()))["step"].zero_(),
            ]
            for mutate in mutations:
                runner.load(checkpoint, map_location="cpu")
                with torch.no_grad():
                    mutate()
                with self.assertRaisesRegex(ValueError, "differs from checkpoint"):
                    verify_resume_state(runner, checkpoint)


if __name__ == "__main__":
    unittest.main()
