"""Contract tests independent of the GPU solver (python -m unittest discover -s tests)."""

import unittest
import copy
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation
from robot_gym.envs.go2w.go2w_env import Go2WEnv
from robot_gym.envs.go2w.go2w_config import (
    GO2WCfg,
    apply_go2w_profile,
    check_target_intervals,
)
from robot_gym.envs.go2.go2_env import Go2Env
from robot_gym.envs.go2w.go2w_symmetry import joint_reflection
from robot_gym.utils import task_registry, get_args
from robot_gym.utils.helpers import class_to_dict, update_cfg_from_args
from robot_gym.utils.diagnostics import (
    check_reference_contract,
    check_training_continuation,
    urdf_link_poses,
    link_reposition_velocity,
    wheel_cylinders,
)


URDF = (
    Path(__file__).resolve().parents[1]
    / "ressources/robots/go2w/urdf/go2w_description.urdf"
)


class ContractTests(unittest.TestCase):
    def make_env(self, count=8, profile=None):
        e = Go2WEnv.__new__(Go2WEnv)
        e.cfg = GO2WCfg()
        if profile:
            apply_go2w_profile(e.cfg, None, profile)
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

    def test_profile_opt_in_contract_and_registry_isolation(self):
        baseline = tuple(map(class_to_dict, task_registry.get_cfgs("go2w")))
        for profile in (None, "step_recovery_v1"):
            argv = ["train", "--task", "go2w"]
            if profile:
                argv += ["--go2w_profile", profile]
            with patch.object(sys, "argv", argv):
                args = get_args()
            env, train = task_registry.get_cfgs("go2w")
            rng = torch.get_rng_state().clone()
            update_cfg_from_args(env, train, args)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(
                baseline, tuple(map(class_to_dict, task_registry.get_cfgs("go2w")))
            )
            if profile is None:
                self.assertIsNone(args.go2w_profile)
                self.assertEqual(baseline, (class_to_dict(env), class_to_dict(train)))
                continue
            self.assertEqual(train.actor.distribution_cfg["init_std"], 0.35)
            self.assertEqual(train.algorithm.entropy_coef, 0.001)
            self.assertEqual(train.runner.max_iterations, 1500)
            self.assertEqual(train.runner.save_interval, 250)
            self.assertFalse(train.runner.resume)
            self.assertFalse(args.zero_command_brake)
            # Train, eval and play all apply the same idempotent selection before build.
            resolved = (class_to_dict(env), class_to_dict(train))
            update_cfg_from_args(env, train, args)
            self.assertEqual(resolved, (class_to_dict(env), class_to_dict(train)))
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "config.yaml"
                path.write_text(
                    yaml.safe_dump(
                        json.loads(
                            json.dumps(dict(env_cfg=baseline[0], train_cfg=baseline[1]))
                        )
                    )
                )
                with self.assertRaisesRegex(ValueError, "contract mismatch"):
                    check_training_continuation(path, *resolved)
                path.write_text(
                    yaml.safe_dump(
                        json.loads(
                            json.dumps(dict(env_cfg=resolved[0], train_cfg=resolved[1]))
                        )
                    )
                )
                check_reference_contract(path, *resolved)
                check_training_continuation(path, *resolved)
                with self.assertRaisesRegex(ValueError, "contract mismatch"):
                    check_reference_contract(path, *baseline)
                changed = copy.deepcopy(resolved[0])
                changed["commands"]["long_stand_probability"] = 0.3
                with self.assertRaisesRegex(ValueError, "long_stand_probability"):
                    check_training_continuation(path, changed, resolved[1])
            args.zero_command_brake = True
            with self.assertRaisesRegex(ValueError, "braking disabled"):
                update_cfg_from_args(env, train, args)
            args.task = "go2"
            with self.assertRaisesRegex(ValueError, "specific to go2w"):
                update_cfg_from_args(env, train, args)

    def test_profile_target_mapping_symmetry_and_continuous_lift_path(self):
        e = self.make_env(profile="step_recovery_v1")
        intervals = check_target_intervals(e.cfg)
        e.joint_names = list(reversed(e.joint_names))
        e._build_control_tensors()
        for i, name in enumerate(e.joint_names):
            self.assertAlmostEqual(
                float(e.action_scale[0, i]),
                {"hip": 0.3, "thigh": 0.35, "calf": 0.4, "foot": 18}[
                    name.split("_")[1]
                ],
                places=6,
            )
        perm, signs = joint_reflection(e.joint_names)
        torch.testing.assert_close(e.action_scale, e.action_scale[:, perm])
        self.assertEqual(signs, [-1 if "hip" in n else 1 for n in e.joint_names])
        nominal = e.cfg.init_state.default_joint_angles
        initial = urdf_link_poses(URDF, nominal)
        # Test-only sagittal IK: lift 4 cm, reposition 1 cm, then return continuously.
        offsets = [(0, h) for h in np.linspace(0, 0.04, 21)]
        offsets += [(x, 0.04) for x in np.linspace(0, 0.01, 11)[1:]]
        offsets += [(x, 4 * x) for x in np.linspace(0.01, 0, 21)[1:]]
        previous = dict(nominal)
        for dx, dz in offsets:
            angles = dict(nominal)
            for side in ("FL", "FR", "RL", "RR"):
                thigh, calf = f"{side}_thigh_joint", f"{side}_calf_joint"
                t, c = nominal[thigh], nominal[calf]
                l1, l2 = 0.213, 0.2264
                x = -l1 * np.sin(t) - l2 * np.sin(t + c) + dx
                z = -l1 * np.cos(t) - l2 * np.cos(t + c) + dz
                c_new = -np.arccos((x * x + z * z - l1 * l1 - l2 * l2) / (2 * l1 * l2))
                t_new = np.arctan2(-x, -z) - np.arctan2(
                    l2 * np.sin(c_new), l1 + l2 * np.cos(c_new)
                )
                angles[thigh], angles[calf] = t_new, c_new
                if dx == 0 and dz == 0.04:
                    expected = (
                        [0.1453, -0.2726] if side.startswith("F") else [0.1541, -0.2747]
                    )
                    np.testing.assert_allclose(
                        [t_new - t, c_new - c], expected, atol=6e-5
                    )
            poses = urdf_link_poses(URDF, angles)
            for wheel in e.cfg.asset.foot_link_names:
                np.testing.assert_allclose(
                    poses[wheel][:3, 3] - initial[wheel][:3, 3], [dx, 0, dz], atol=1e-7
                )
            for name, (low, high) in intervals.items():
                self.assertTrue(low <= angles[name] <= high, name)
                self.assertLess(abs(angles[name] - previous[name]), 0.02, name)
            previous = angles
        e.cfg.control.action_scale["FL_calf_joint"] = 1
        with self.assertRaisesRegex(ValueError, "FL_calf_joint"):
            check_target_intervals(e.cfg)

    def test_profile_reward_geometry_reposition_support_and_reset(self):
        e = self.make_env(2, "step_recovery_v1")
        e.foot_pos = torch.tensor(
            [
                [
                    [0.2, 0.2, 0.126],
                    [0.2, -0.2, 0.086],
                    [-0.2, 0.2, 0.086],
                    [-0.2, -0.2, 0.086],
                ]
            ]
        ).repeat(2, 1, 1)
        e.base_pos = torch.tensor([[0.0, 0.0, 0.42]]).repeat(2, 1)
        e.base_quat = torch.tensor([[1.0, 0, 0, 0]]).repeat(2, 1)
        e.base_lin_vel = e.base_ang_vel = torch.zeros(2, 3)
        e.foot_lin_vel = torch.zeros_like(e.foot_pos)
        e.foot_lin_vel[:, 0, 0] = 0.15
        e.robot = Mock()
        e.foot_link_indices_local = [0, 1, 2, 3]
        e.robot.get_links_quat.return_value = (
            e.base_quat[:, None].expand(-1, 4, -1).clone()
        )
        e.wheel_geometry = wheel_cylinders(URDF, e.cfg.asset.foot_link_names)
        e.wheel_clearance = torch.zeros(2, 4)
        e.wheel_normal_force = torch.zeros(2, 4)
        e.loaded_wheels = torch.tensor([[False, True, True, True]]).repeat(2, 1)
        e.wheel_reposition_velocity_body = torch.zeros(2, 4, 3)

        def refresh():
            with patch.object(Go2Env, "_update_robot_state"):
                e._update_robot_state()

        e.commands[:] = torch.tensor([0, 0.2, 0])
        refresh()
        torch.testing.assert_close(
            e.wheel_clearance[:, 0], torch.full((2,), 0.04), atol=1e-7, rtol=0
        )
        good = e._reward_foot_swing_clearance().clone()
        torch.testing.assert_close(good, torch.ones(2))
        e.foot_pos[:, 0, 2] = 0.091
        refresh()
        self.assertTrue((e._reward_foot_swing_clearance() < good).all())
        e.foot_pos[:, 0, 2] = 0.086
        refresh()
        self.assertEqual(e._reward_foot_swing_clearance().sum(), 0)
        e.foot_pos[:, 0, 2] = 0.126
        refresh()
        for command in ([0, 0, 0], [0.5, 0, 0], [-0.3, 0, 0]):
            e.commands[:] = torch.tensor(command)
            self.assertEqual(e._reward_foot_swing_clearance().sum(), 0)
        e.commands[:] = torch.tensor([0, 0.2, 0])
        e.loaded_wheels.zero_()
        self.assertEqual(e._reward_foot_swing_clearance().sum(), 0)
        e.loaded_wheels[:, 1:] = True
        # Rigid body translation/yaw gives no repositioning, including wheel spin.
        e.base_lin_vel = torch.tensor([[0.3, -0.1, 0.0]]).repeat(2, 1)
        e.base_ang_vel = torch.tensor([[0.0, 0, 0.7]]).repeat(2, 1)
        e.foot_lin_vel = e.base_lin_vel[:, None] + torch.linalg.cross(
            e.base_ang_vel[:, None].expand(-1, 4, -1), e.foot_pos - e.base_pos[:, None]
        )
        e.robot.get_links_quat.return_value[:, :, 0] = np.cos(0.6)
        e.robot.get_links_quat.return_value[:, :, 2] = np.sin(0.6)
        refresh()
        torch.testing.assert_close(
            e._reward_foot_swing_clearance(), torch.zeros(2), atol=1e-7, rtol=0
        )
        # Mirror lateral command and permute wheels; no side-dependent reward.
        e.wheel_reposition_velocity_body[:, 0, 1] = 0.15
        e.commands[1, 1] *= -1
        for name in (
            "wheel_clearance",
            "loaded_wheels",
            "wheel_reposition_velocity_body",
        ):
            value = getattr(e, name)
            value[1] = value[0, [1, 0, 3, 2]]
        e.wheel_reposition_velocity_body[1, :, 1] *= -1
        torch.testing.assert_close(
            e._reward_foot_swing_clearance()[0], e._reward_foot_swing_clearance()[1]
        )
        with patch.object(Go2Env, "reset_idx"):
            before = e.wheel_clearance[1].clone()
            e.reset_idx(torch.tensor([0]))
            self.assertEqual(e.wheel_clearance[0].count_nonzero(), 0)
            self.assertEqual(e.loaded_wheels[0].count_nonzero(), 0)
            self.assertEqual(e.wheel_reposition_velocity_body[0].count_nonzero(), 0)
            torch.testing.assert_close(e.wheel_clearance[1], before)
        self.assertFalse(hasattr(e, "physics_diagnostics"))

    def test_profile_long_stands_only_during_normal_stand_resampling(self):
        e = self.make_env(20000, "step_recovery_v1")
        e.cfg.env.record_command_families = True
        e._build_control_tensors()
        torch.manual_seed(24)
        e._resample_commands(torch.arange(e.num_envs))
        families, durations = e.diagnostic_command_families, e.command_steps_left * e.dt
        stand = families == 0
        self.assertFalse(((durations > 3) & ~stand).any())
        self.assertAlmostEqual(
            float((durations[stand] > 3).float().mean()), 0.25, delta=0.03
        )
        self.assertLessEqual(durations.max(), 6)
        self.assertAlmostEqual(
            float(durations[stand].sum() / durations.sum()), 0.23, delta=0.015
        )
        self.assertAlmostEqual(
            float(durations[families >= 5].sum() / durations.sum()), 0.18, delta=0.015
        )
        # Ordinary post-movement resampling also receives long stands.
        e.commands[:] = torch.tensor([0.5, 0, 0])
        e.command_resampling_enabled = True
        e.command_steps_left[:] = 1
        e._post_physics_step_callback()
        self.assertTrue((e.command_steps_left > 150).any())
        # Fixed commands consume no new random draws and retain their timer.
        e.compute_observations = Mock()
        e.set_fixed_command((0, 0, 0))
        before = e.command_steps_left.clone()
        rng = torch.get_rng_state().clone()
        e._resample_commands(torch.arange(e.num_envs))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        torch.testing.assert_close(e.command_steps_left, before)
        self.assertEqual(e.commands.count_nonzero(), 0)

    def test_link_origin_reposition_matches_finite_difference(self):
        # Arbitrary tilted base, yaw and translation, with and without relative motion.
        rot = Rotation.from_euler("xyz", [0.2, -0.3, 0.7])
        omega = np.array([0.2, -0.1, 0.8])
        base = np.array([1.0, 2.0, 0.4])
        velocity = np.array([0.3, -0.2, 0.1])
        r = np.array([[0.2, 0.3, -0.3], [-0.2, -0.3, -0.25]])
        dr = np.array([[0.0, 0.0, 0.0], [0.15, -0.07, 0.01]])

        def world(t):
            rotation = rot * Rotation.from_rotvec(t * omega)
            return base + t * velocity + rotation.apply(r + t * dr)

        eps = 1e-5
        link_vel = (world(eps) - world(-eps)) / (2 * eps)
        result = link_reposition_velocity(
            torch.tensor(world(0))[None],
            torch.tensor(link_vel)[None],
            torch.tensor(base)[None],
            torch.tensor(rot.as_quat()[[3, 0, 1, 2]])[None],
            torch.tensor(rot.inv().apply(velocity))[None],
            torch.tensor(omega)[None],
        )
        np.testing.assert_allclose(result[0], dr, atol=1e-9)

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

    def test_optional_family_labels_preserve_sampler_and_rng(self):
        baseline, instrumented = self.make_env(), self.make_env()
        instrumented.cfg.env.record_command_families = True
        instrumented._build_control_tensors()
        torch.manual_seed(123)
        baseline._resample_commands(torch.arange(8))
        rng = torch.get_rng_state().clone()
        torch.manual_seed(123)
        instrumented._resample_commands(torch.arange(8))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(baseline.commands, instrumented.commands))
        self.assertTrue(
            torch.equal(baseline.command_steps_left, instrumented.command_steps_left)
        )
        self.assertTrue((instrumented.diagnostic_command_families >= 0).all())

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
