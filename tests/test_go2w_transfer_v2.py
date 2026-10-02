"""Focused v2 reward checks; optional one native probe and one fresh PPO smoke."""

import math
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import torch

from tests import test_go2w_transfer as transfer
from tests.test_go2w_contract import ContractTests
from tests.test_go2w_event_step import EventFixture
from robot_gym.envs.go2w.go2w_config import apply_go2w_profile, check_target_intervals
from robot_gym.envs.go2w.go2w_env import lateral_wheel_center_velocity
from robot_gym.envs.go2w.step_events import WheelStepEvents
from robot_gym.utils import get_args, task_registry
from robot_gym.utils.helpers import class_to_dict
from robot_gym.utils.diagnostics import (
    cylinder_clearance,
    wheel_cylinders,
    urdf_link_poses,
    write_json,
)
from robot_gym.utils.urdf_reader import URDFReader

OUT = Path(__file__).resolve().parents[1] / "evaluation/transfer_v2_preparation"


def recipe():
    cfg, train = task_registry.get_cfgs("go2w")
    apply_go2w_profile(cfg, train, "transfer_v2")
    return cfg, train


class SwingFixture(EventFixture):
    def __init__(self):
        super().__init__()
        self.cfg, _ = recipe()
        self.geometry = wheel_cylinders(
            URDFReader(self.cfg.asset.robot_file).robot_file_path_absolute,
            self.cfg.asset.foot_link_names,
        )
        self.tracker = WheelStepEvents(
            1, "cpu", self.cfg.rewards.event_step, self.geometry
        )
        self.wheels[..., 2] = self.cfg.asset.contact_height
        self.clearance = torch.zeros(1, 4)

    def score(self, **kwargs):
        self.sample(**kwargs)
        # Fixture poses contain level cylinders; base-only motion also changes actual height.
        actual = torch.zeros(1, 4)
        actual[:, self.legs] = kwargs.get("height", 0.0)
        actual += kwargs.get("heave", 0.0)
        return float(
            self.tracker.dense_swing(
                actual, self.commands, self.cfg.rewards.dense_swing
            )
        )


def reachable_targets(root_height=None, clearance=0.04):
    """Offline bounded FK solve for one probe target per limb; never an action controller."""
    cfg, _ = recipe()
    asset = URDFReader(cfg.asset.robot_file).robot_file_path_absolute
    targets = {}
    for wheel in cfg.asset.foot_link_names:
        side = wheel.removesuffix("_foot")
        names = [f"{side}_{part}_joint" for part in ("hip", "thigh", "calf")]
        nominal = np.array([cfg.init_state.default_joint_angles[n] for n in names])
        scales = np.array([cfg.control.action_scale[n] for n in names])
        geometry = tuple(x.double() for x in wheel_cylinders(asset, [wheel]))

        def position(action):
            angles = dict(cfg.init_state.default_joint_angles)
            angles.update(zip(names, nominal + scales * action))
            pose = urdf_link_poses(asset, angles)[wheel]
            xyz = torch.tensor(pose[:3, 3], device="cpu")[None]
            quat = torch.tensor(
                Rotation.from_matrix(pose[:3, :3]).as_quat(scalar_first=True),
                device="cpu",
            )[None]
            center = pose[:3, 3] + pose[:3, :3] @ geometry[0][0].numpy()
            return np.r_[center[:2], float(cylinder_clearance(xyz, quat, *geometry)[0])]

        start = position(np.zeros(3))
        desired = start + [0.0, 0.03 if side in ("FL", "RL") else -0.03, 0.04]
        if root_height is not None:
            desired[2] = clearance - root_height
        solution = least_squares(
            lambda action: position(action) - desired,
            np.zeros(3),
            bounds=(-1.0, 1.0),
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )
        targets[wheel] = {
            "joint_names": names,
            "normalized_action": solution.x.tolist(),
            "geometry_delta_m": (position(solution.x) - start).tolist(),
            "residual_m": float(np.linalg.norm(solution.fun)),
        }
    return targets


class TransferV2CPU(unittest.TestCase):
    def test_opt_in_recipe_and_fresh_guard(self):
        cfg, train = recipe()
        old, old_train = transfer.recipe()
        for key in (
            "asset",
            "control",
            "domain_rand",
            "sim",
            "noise",
            "normalization",
            "init_state",
            "env",
        ):
            self.assertEqual(
                class_to_dict(getattr(cfg, key)), class_to_dict(getattr(old, key)), key
            )
        for key in ("actor", "critic", "algorithm"):
            self.assertEqual(
                class_to_dict(getattr(train, key)),
                class_to_dict(getattr(old_train, key)),
                key,
            )
        self.assertEqual((cfg.env.num_observations, cfg.env.num_actions), (56, 16))
        self.assertEqual(
            train.runner.obs_groups, {"actor": ["policy"], "critic": ["policy"]}
        )
        self.assertEqual(cfg.rewards.scales.orientation, -4.0)
        self.assertEqual(
            (cfg.rewards.tracking_sigma_x, cfg.rewards.tracking_sigma_y), (0.09, 0.04)
        )
        self.assertEqual(cfg.rewards.event_step, old.rewards.event_step)
        self.assertEqual(cfg.rewards.scales.step_event, 0.15)
        self.assertEqual(cfg.rewards.discrete_reward_names, ["step_event"])
        self.assertEqual(old.rewards.scales.orientation, -1.2)
        for flag in (("--resume",), ("--checkpoint", "1999"), ("--load_run", "old")):
            args = transfer.args_for(64, "--go2w_profile", "transfer_v2", *flag)
            with patch.object(task_registry, "make_env") as construct:
                with self.assertRaisesRegex(ValueError, "fresh only"):
                    transfer.train(args)
                construct.assert_not_called()

    def test_support_posture_and_group_mapping(self):
        env = ContractTests().make_env(3, "transfer_v2")
        env.dof_pos = torch.ones(3, 16) * 0.1
        env.default_dof_pos = torch.zeros(1, 16)
        env.loaded_wheels = torch.tensor([[True] * 4, [True] * 4, [False] * 4])
        env.commands[1:, 1] = 0.3
        torch.testing.assert_close(
            env._reward_hip_pose(), torch.tensor([2.0, 1.0, 0.2]) * 0.01
        )
        torch.testing.assert_close(
            env._reward_sagittal_pose(), torch.tensor([2.0, 0.6, 0.06]) * 0.01
        )
        env.loaded_wheels[2, 0] = True
        self.assertAlmostEqual(
            float(env._reward_sagittal_pose()[2]),
            (0.6 * 2 + 0.06 * 6) / 8 * 0.01,
            places=7,
        )

    def test_dense_swing_excludes_startup_loaded_motion_base_motion_and_hover(self):
        startup = SwingFixture()
        self.assertEqual(startup.score(load=0.0, height=0.05, move=0.04), 0)
        f = SwingFixture()
        for _ in range(15):
            self.assertEqual(f.score(load=30.0, move=0.04, spin=0.8), 0)
        f.score(load=0.0)
        self.assertAlmostEqual(f.score(load=0.0, heave=0.05, spin=1.0), 0.0, places=6)
        f = SwingFixture()
        for _ in range(15):
            f.score()
        f.score(load=0.0)
        tiny = f.score(load=0.0, height=0.001)
        cm = f.score(load=0.0, height=0.04, move=0.04)
        self.assertGreater(tiny, 0.0)
        self.assertGreater(cm, tiny)
        self.assertAlmostEqual(cm, 0.25, places=5)  # One of four wheels, full demand.
        for _ in range(40):
            last = f.score(load=0.0, height=0.05, move=0.04)
        self.assertEqual(last, 0.0)
        failed = SwingFixture()
        for _ in range(15):
            failed.score()
        failed.score(load=0.0)
        self.assertEqual(
            failed.score(load=0.0, height=0.04, move=0.04, failed=True), 0.0
        )
        rocking = SwingFixture()
        for _ in range(15):
            rocking.score()
        rocking.score(load=0.0)
        self.assertAlmostEqual(
            rocking.score(load=0.0, heave=0.05, rotation=(0.04, 0.06, 0.1)),
            0.0,
            places=6,
        )

    def test_real_cycle_has_separate_continuous_and_discrete_contributions(self):
        rows = {}
        for name, apex, heave in (
            ("tiny", 0.001, False),
            ("centimeter_cycle", 0.04, False),
            ("base_heave", 0.04, True),
        ):
            f = SwingFixture()
            dense = 0.0
            for tick in range(60):
                t = tick * 0.02
                phase = min(1.0, max(0.0, (t - 0.4) / 0.4))
                lift = apex * math.sin(math.pi * phase) ** 2
                dense += (
                    0.4
                    * 0.02
                    * f.score(
                        load=0.0 if 0.4 <= t < 0.8 - 1e-8 else 30.0,
                        height=0.0 if heave else lift,
                        heave=lift if heave else 0.0,
                        move=0.0 if heave else 0.04 * phase,
                    )
                )
            rows[name] = {
                "dense": dense,
                "completed_event": f.total,
                "qualified_count": f.valid,
            }
        self.assertGreater(rows["centimeter_cycle"]["dense"], rows["tiny"]["dense"])
        self.assertGreater(rows["centimeter_cycle"]["completed_event"], 0.0)
        self.assertEqual(rows["tiny"]["qualified_count"], 0)
        self.assertAlmostEqual(rows["base_heave"]["dense"], 0.0, places=7)
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / "synthetic_swing_terms.json", rows)

    def test_scrubbing_transport_and_reflection(self):
        cfg, _ = recipe()
        geometry = wheel_cylinders(
            URDFReader(cfg.asset.robot_file).robot_file_path_absolute,
            cfg.asset.foot_link_names,
        )
        q = torch.tensor([1.0, 0, 0, 0]).expand(1, 4, 4)
        velocity = torch.zeros(1, 4, 3)
        velocity[..., 0] = 0.5
        angular = torch.zeros_like(velocity)
        angular[..., 1] = 8.0
        torch.testing.assert_close(
            lateral_wheel_center_velocity(q, velocity, angular, geometry),
            torch.zeros(1, 4),
            atol=1e-6,
            rtol=0,
        )
        velocity[..., 1] = 0.3
        u = lateral_wheel_center_velocity(q, velocity, angular, geometry)
        self.assertTrue(torch.allclose(u.abs(), torch.full_like(u, 0.3), atol=1e-6))
        env = ContractTests().make_env(1, "transfer_v2")
        env.commands[:, 1] = 0.3
        env.loaded_wheels = torch.ones(1, 4, dtype=torch.bool)
        env.wheel_center_lateral_speed = u
        self.assertAlmostEqual(
            float(env._reward_lateral_wheel_scrub()) * -2 * 0.02, -0.0036, places=7
        )
        env.wheel_center_lateral_speed = -u
        self.assertAlmostEqual(float(env._reward_lateral_wheel_scrub()), 0.09, places=6)
        env.loaded_wheels.zero_()
        self.assertEqual(float(env._reward_lateral_wheel_scrub()), 0.0)
        # Nonzero offset: omega_z cross +x offset gives +y velocity exactly once.
        offset = geometry[0].clone()
        offset[:, 0] = 0.1
        geometry = (offset, *geometry[1:])
        velocity.zero_()
        angular.zero_()
        angular[..., 2] = 2.0
        torch.testing.assert_close(
            lateral_wheel_center_velocity(q, velocity, angular, geometry).abs(),
            torch.full((1, 4), 0.2),
            atol=1e-6,
            rtol=0,
        )

    def test_reachability_and_review_panel(self):
        cfg, _ = recipe()
        check_target_intervals(cfg)
        targets = reachable_targets()
        for target in targets.values():
            self.assertLess(target["residual_m"], 1e-4)
            self.assertLessEqual(max(abs(x) for x in target["normalized_action"]), 1)
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / "reachable_targets.json", targets)
        from robot_gym.envs.go2w.diagnostic_bank import transfer_schedule

        schedule = transfer_schedule("transfer_v2")
        self.assertEqual(len(schedule), 13)
        self.assertEqual(schedule["lateral_strong_negative"][1], (5, (0, -0.3, 0)))
        self.assertEqual(len(transfer_schedule()), 8)

    def test_discovery_segments_preserve_commands_and_long_holds(self):
        envs = [
            ContractTests().make_env(20000, profile)
            for profile in ("transfer_v1", "transfer_v2")
        ]
        for env in envs:
            env.command_ranges = class_to_dict(env.cfg.commands.ranges)
            env.diagnostic_command_families = torch.zeros(
                env.num_envs, dtype=torch.long
            )
            env.diagnostic_long_moving_commands = torch.zeros(
                env.num_envs, dtype=torch.bool
            )
            torch.manual_seed(3)
            env._resample_commands(torch.arange(env.num_envs))
        old, new = envs
        torch.testing.assert_close(old.commands, new.commands, atol=0, rtol=0)
        families = old.diagnostic_command_families
        untouched = ((families != 3) & (families != 5) & (families != 6)) | (
            old.command_steps_left >= 400
        )
        torch.testing.assert_close(
            old.command_steps_left[untouched], new.command_steps_left[untouched]
        )
        from robot_gym.envs.go2w.step_events import step_demand

        short = (
            ~untouched
            & (old.command_steps_left <= 50)
            & (step_demand(old.commands) > 0)
        )
        fraction = float((new.command_steps_left[short] >= 100).float().mean())
        self.assertAlmostEqual(fraction, 0.70, delta=0.03)
        self.assertTrue((new.command_steps_left[short] <= 200).all())

    def test_full_reward_accumulation_and_positive_clipping(self):
        """Idealized state fixture exercises every active term, not learned behavior."""
        env = ContractTests().make_env(1, "transfer_v2")
        env.default_dof_pos = torch.tensor(
            [[env.cfg.init_state.default_joint_angles[n] for n in env.joint_names]]
        )
        env.dof_pos = env.default_dof_pos.clone()
        env.dof_vel = env.last_dof_vel = torch.zeros(1, 16)
        env.actions = env.last_actions = torch.zeros(1, 16)
        env.torques = torch.zeros(1, 16)
        env.torque_limits = torch.full((16,), 23.7)
        root = URDFReader(env.cfg.asset.robot_file).root
        env.dof_pos_limits = torch.tensor(
            [
                [
                    float(root.find(f"joint[@name='{n}']/limit").get(k, default))
                    for k, default in (("lower", "-inf"), ("upper", "inf"))
                ]
                for n in env.joint_names
            ]
        )
        env.base_pos = torch.tensor([[0.0, 0.0, 0.415]])
        env.base_quat = torch.tensor([[1.0, 0, 0, 0]])
        env.base_ang_vel = torch.zeros(1, 3)
        env.base_lin_vel = torch.tensor([[0.0, 0.3, 0.0]])
        env.projected_gravity = torch.tensor([[0.0, 0.0, -1.0]])
        env.nonfoot_contact_count = torch.zeros(1)
        env.reset_buf = env.time_out_buf = torch.zeros(1, dtype=torch.bool)
        env.rew_buf = torch.zeros(1)
        env.reward_scales = class_to_dict(env.cfg.rewards.scales)
        env._prepare_reward_function()
        self.assertEqual(env.reward_scales["wheel_swing"], 0.4 * 0.02)
        self.assertEqual(env.reward_scales["lateral_wheel_scrub"], -2 * 0.02)
        self.assertEqual(env.reward_scales["step_event"], 0.15)
        rows = {}
        for index, (name, lift, loaded, scrub, tilt) in enumerate(
            (
                ("loaded_slide", 0.0, True, 0.3, 0.0),
                ("unloaded_cm", 0.04, False, 0.0, 0.0),
                ("base_rock", 0.0, False, 0.0, 0.15),
                ("loaded_extreme_scrub", 0.0, True, 10.0, 0.0),
            )
        ):
            f = SwingFixture()
            for _ in range(15):
                f.score()
            if not loaded:
                f.score(load=0.0)
                f.score(load=0.0, height=lift, move=0.04 if lift else 0.0)
            env.step_events = f.tracker
            env.commands[:] = torch.tensor([0.0, 0.3, 0.0])
            env.loaded_wheels = torch.tensor([[loaded, True, True, True]])
            env.wheel_normal_force = env.loaded_wheels.float() * 40
            env.wheel_clearance = torch.tensor([[lift, 0.0, 0.0, 0.0]])
            env.wheel_center_lateral_speed = torch.full((1, 4), scrub)
            env.foot_pos = f.wheels
            env.projected_gravity[:] = torch.tensor(
                [math.sin(tilt), 0.0, -math.cos(tilt)]
            )
            env.common_step_counter = index
            raw = {
                n: float(fn()) for n, fn in zip(env.reward_names, env.reward_functions)
            }
            weighted = {n: value * env.reward_scales[n] for n, value in raw.items()}
            env.compute_reward()
            self.assertAlmostEqual(
                float(env.rew_buf), max(0.0, sum(weighted.values())), places=6
            )
            rows[name] = {
                "raw": raw,
                "weighted": weighted,
                "sum_before_clip": sum(weighted.values()),
                "total_after_clip": float(env.rew_buf),
            }
        self.assertLess(rows["base_rock"]["weighted"]["orientation"], 0.0)
        self.assertEqual(rows["base_rock"]["weighted"]["wheel_swing"], 0.0)
        self.assertGreater(rows["unloaded_cm"]["weighted"]["wheel_swing"], 0.0)
        self.assertLess(rows["loaded_slide"]["weighted"]["lateral_wheel_scrub"], 0.0)
        self.assertEqual(rows["loaded_extreme_scrub"]["total_after_clip"], 0.0)
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / "synthetic_full_rewards.json", rows)


@unittest.skipUnless(
    os.environ.get("GO2W_V2_NATIVE") == "probe",
    "one bounded native posture/geometry session",
)
class TransferV2Plant(unittest.TestCase):
    def test_geometry_and_owner_commands(self):
        import genesis as gs
        from robot_gym.envs.base.legged_robot import LeggedRobot
        from robot_gym.utils.replay import configure_nominal
        from robot_gym.utils.diagnostics import loaded_properties, joint_dynamics

        cfg, _ = recipe()
        args = transfer.args_for(1, "--go2w_profile", "transfer_v2")
        configure_nominal(cfg, args)
        cfg.env.capture_transitions = cfg.env.capture_precision = (
            cfg.env.capture_closed_loop
        ) = True
        cfg.env.episode_length_s = 30.0
        records, report = {}, {}
        original_step = LeggedRobot.step

        def first_step(env, actions):
            if "before_first_physics_step" not in report:
                q = env.robot.get_dofs_position(env.joint_dof_idx)
                dq = env.robot.get_dofs_velocity(env.joint_dof_idx)
                torch.testing.assert_close(q, env.default_dof_pos)
                torch.testing.assert_close(dq, torch.zeros_like(dq))
                report["before_first_physics_step"] = {
                    "q": q.cpu(),
                    "dq": dq.cpu(),
                    "nominal_applied": True,
                }
            return original_step(env, actions)

        env = None
        try:
            with patch.object(LeggedRobot, "step", first_step):
                env, _ = task_registry.make_env("go2w", args=args, env_cfg=cfg)
            report["dynamics"] = joint_dynamics(env.robot, env.joint_names)
            report["body_properties"] = loaded_properties(env)
            nominal = torch.zeros((1, 16), device=env.device)
            columns = (
                "base_pos",
                "base_quat",
                "base_lin_vel",
                "base_ang_vel",
                "dof_pos",
                "dof_vel",
                "wheel_clearance",
                "wheel_normal_force",
                "wheel_center_lateral_speed",
                "nonfoot_contact_count",
                "reset_buf",
            )

            def sample(action, rows):
                env.step(action)
                state = env.transition_state
                rows.append(
                    {key: state[key].detach().cpu().numpy().copy() for key in columns}
                )
                return bool(state["reset_buf"].any()) or not bool(
                    torch.isfinite(env.obs_buf).all()
                )

            def pack(rows):
                return {key: np.stack([row[key][0] for row in rows]) for key in columns}

            env.set_fixed_command((0.0, 0.0, 0.0))
            for _ in range(150):
                env.step(nominal)
            height = float(env.base_pos[0, 2])
            targets, landing = (
                reachable_targets(height),
                reachable_targets(height, clearance=0.0),
            )
            report["settled_height_m"] = height
            report["loaded_height_fk_targets"] = targets
            self.assertTrue(all(v["residual_m"] < 1e-4 for v in targets.values()))
            report["plant"] = {}
            for limb, wheel in enumerate(cfg.asset.foot_link_names):
                env.reset()
                env.set_fixed_command((0.0, 0.3, 0.0))
                for _ in range(150):
                    env.step(nominal)
                ids = [env.joint_names.index(n) for n in targets[wheel]["joint_names"]]
                apex = torch.tensor(
                    targets[wheel]["normalized_action"], device=env.device
                )
                touchdown = torch.tensor(
                    landing[wheel]["normalized_action"], device=env.device
                )
                rows = []
                for tick in range(75):
                    t = tick * env.dt
                    action = nominal.clone()
                    if t < 0.25:
                        value = apex * (0.5 - 0.5 * math.cos(math.pi * t / 0.25))
                    elif t < 0.50:
                        u = 0.5 - 0.5 * math.cos(math.pi * (t - 0.25) / 0.25)
                        value = apex * (1 - u) + touchdown * u
                    else:
                        value = touchdown
                    action[:, ids] = value
                    if sample(action, rows):
                        break
                data = pack(rows)
                for key, value in data.items():
                    records[f"plant_{wheel}_{key}"] = value
                unloaded = data["wheel_normal_force"][:, limb] <= 6
                supported = data["wheel_normal_force"] > 10
                report["plant"][wheel] = {
                    "peak_clearance_m": float(data["wheel_clearance"][:, limb].max()),
                    "unloaded_ticks": int(unloaded.sum()),
                    "two_other_supports_while_unloaded": bool(
                        (supported[unloaded].sum(1) >= 2).all()
                    )
                    if unloaded.any()
                    else False,
                    "reloaded_final": bool(supported[-10:, limb].all()),
                    "failure": bool(data["reset_buf"].any()),
                    "max_body_roll_pitch_deg": np.abs(
                        Rotation.from_quat(data["base_quat"][:, [1, 2, 3, 0]]).as_euler(
                            "xyz", degrees=True
                        )[:, :2]
                    )
                    .max(0)
                    .tolist(),
                }
            # Inference-only old Actor in identical nominal physical/control settings.
            # V2 rewards never feed actions; no checkpoint initializes fresh training.
            env.cfg.env.play_mode = True
            run = Path(
                "logs/go2w_transfer_v1/transfer_v1_seed1_20261002_090639_2026-10-02_09-09-17"
            ).resolve()
            with patch.object(
                sys,
                "argv",
                [
                    "probe",
                    "--task",
                    "go2w",
                    "--load_run",
                    str(run),
                    "--checkpoint",
                    "1999",
                    "--num_envs",
                    "1",
                    "--seed",
                    "1",
                    "--headless",
                ],
            ):
                replay_args = get_args()
            _, training, checkpoint = task_registry.resolve_replay(replay_args)
            runner, _ = task_registry.make_alg_runner(
                env, args=replay_args, train_cfg=training, save_config=False
            )
            policy = runner.get_inference_policy(device=env.device)
            report["actor_checkpoint"] = str(checkpoint)
            report["owner_command_capture"] = {}
            with torch.no_grad():
                for name, command in (
                    ("forward_05", (0.5, 0, 0)),
                    ("lateral_03", (0, 0.3, 0)),
                    ("yaw_08", (0, 0, 0.8)),
                ):
                    env.reset()
                    env.set_fixed_command(command)
                    rows = []
                    for _ in range(300):
                        if sample(policy(env.get_observations()), rows):
                            break
                    data = pack(rows)
                    for key, value in data.items():
                        records[f"{name}_{key}"] = value
                    angles = Rotation.from_quat(
                        data["base_quat"][:, [1, 2, 3, 0]]
                    ).as_euler("xyz")
                    velocity = np.c_[
                        data["base_lin_vel"][:, :2], data["base_ang_vel"][:, 2]
                    ]
                    tail = slice(100, None)
                    report["owner_command_capture"][name] = {
                        "command": command,
                        "ticks": len(rows),
                        "window_s": [2, len(rows) * env.dt],
                        "mean_vx_vy_yaw": velocity[tail].mean(0).tolist(),
                        "rmse_vx_vy_yaw": np.sqrt(
                            np.mean((velocity[tail] - command) ** 2, axis=0)
                        ).tolist(),
                        "signed_mean_roll_pitch_deg": np.rad2deg(
                            angles[tail, :2].mean(0)
                        ).tolist(),
                        "rms_roll_pitch_deg": np.rad2deg(
                            np.sqrt(np.mean(angles[tail, :2] ** 2, axis=0))
                        ).tolist(),
                        "loaded_scrub_rms_m_s": float(
                            np.sqrt(
                                np.mean(
                                    (data["wheel_normal_force"][tail] > 8)
                                    * data["wheel_center_lateral_speed"][tail] ** 2
                                )
                            )
                        ),
                        "failure": bool(data["reset_buf"].any()),
                    }
        finally:
            OUT.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(OUT / "native_probe.npz", **records)
            write_json(OUT / "native_probe.json", report)
            gs.destroy()


class TransferV2Smoke(transfer.TransferSmoke):
    profile = "transfer_v2"
    output = OUT


if __name__ == "__main__":
    unittest.main()
