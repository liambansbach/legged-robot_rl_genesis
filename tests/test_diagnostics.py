"""CPU contracts for diagnostics: physical geometry, provenance and no new RNG draws."""

import copy
import json
from pathlib import Path
import tempfile
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
from robot_gym.utils.helpers import class_to_dict
from robot_gym.utils.diagnostics import (
    summed_normal_force,
    cylinder_clearance,
    wheel_cylinders,
    heading_wxyz,
    check_reference_contract,
    manifest,
    PhysicsDiagnostics,
    write_json,
    nominal_support_heights,
)
from robot_gym.utils.training_diagnostics import std_parameters, TrainingDiagnostics
from robot_gym.scripts.diagnostic_bank import condition_bank
from rsl_rl.modules.distribution import GaussianDistribution


URDF = Path(ROBOT_GYM_ROOT_DIR) / "ressources/robots/go2w/urdf/go2w_description.urdf"


class DiagnosticsTests(unittest.TestCase):
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
            )

        with tempfile.TemporaryDirectory() as tmp:
            baseline = run()
            path = Path(tmp) / "diagnostics.jsonl"
            instrumented = run(path)
            self.assertEqual(baseline[0], instrumented[0])
            for a, b in zip(baseline[1:3], instrumented[1:3]):
                for key in a:
                    self.assertTrue(torch.equal(a[key], b[key]), key)
            self.assertTrue(torch.equal(baseline[-1], instrumented[-1]))
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


if __name__ == "__main__":
    unittest.main()
