import unittest
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from robot_gym.utils.export import BoundedPolicy
from robot_gym.scripts.evaluate import (
    response_metrics,
    diagnostic_metrics,
    lateral_metrics,
    mirror_pair_summary,
    pose_recovery_metrics,
    cases,
    MIRROR_PAIRS,
    POSE_RECOVERY_CASES,
)
import numpy as np


class InferenceTests(unittest.TestCase):
    def test_lateral_windows_displacement_clearance_and_contact_transitions(self):
        data = np.zeros((150, 2, 19))
        data[:50, 0, 4], data[50:100, 0, 4], data[100:, 0, 4] = 0.2, 0.04, 0.001
        data[:, 0, 17] = np.linspace(10.001, 10.25, 150)
        data[:, 1, 4] = 100  # fallen replicas must not pollute metrics
        data[75, 1, 9] = 1
        detail = {
            "foot_contacts": np.ones((150, 2, 4), dtype=bool),
            "wheel_link_height": np.full((150, 2, 4), 0.086),
        }
        detail["foot_contacts"][20:30, 0, 0] = False
        detail["foot_contacts"][60:70, 0, 0] = False
        detail["wheel_link_height"][20:30, 0, 0] += 0.02
        detail["wheel_link_height"][60:70, 0, 0] += 0.04
        result = lateral_metrics(data, detail, 0.02, 0.086, np.array([10.0, 20.0]))
        np.testing.assert_allclose(
            list(result["mean_vy_m_s"].values()), [0.2, 0.04, 0.001]
        )
        self.assertEqual(
            result["window_duration_s"],
            dict.fromkeys(["first_second", "second_second", "final_second"], 1.0),
        )
        self.assertEqual(result["total_lateral_displacement_world_y_m"], 0.25)
        self.assertEqual(result["displacement_start_s"], 0)
        self.assertEqual(result["airborne_wheel_samples"], 20)
        np.testing.assert_allclose(
            list(result["airborne_wheel_clearance_above_nominal_m"].values()),
            [0.03, 0.04, 0.04],
        )
        self.assertEqual(
            result["wheel_contact_transition_count_mean_per_environment"], [4, 0, 0, 0]
        )
        detail["foot_contacts"][:] = True
        result = lateral_metrics(data, detail, 0.02, 0.086)
        self.assertIsNone(result["airborne_wheel_clearance_above_nominal_m"])
        self.assertEqual(result["displacement_start_s"], 0.02)
        self.assertAlmostEqual(result["total_lateral_displacement_world_y_m"], 0.249)
        data[10, 0, 9] = 1
        result = lateral_metrics(data, detail, 0.02, 0.086)
        self.assertIsNone(result["total_lateral_displacement_world_y_m"])
        self.assertTrue(all(v is None for v in result["mean_vy_m_s"].values()))

    def test_mirror_pairs_and_pose_recovery_cases(self):
        commands = {name: (before, after) for name, before, after, _ in cases()}
        self.assertEqual(len(commands), 31)
        tests = {}
        for minus, plus in MIRROR_PAIRS:
            np.testing.assert_allclose(
                commands[minus][1], np.asarray(commands[plus][1]) * [1, -1, -1]
            )
            tests[minus] = {
                "diagnostics": {"final_window": {"mean_velocity": [0.5, -0.2, -0.8]}}
            }
            tests[plus] = {
                "diagnostics": {"final_window": {"mean_velocity": [0.6, 0.3, 1.0]}}
            }
        summary = mirror_pair_summary(tests)
        self.assertEqual(len(summary), 7)
        for pair in summary.values():
            np.testing.assert_allclose(
                list(pair["absolute_mirror_mismatch"].values()), [0.1, 0.1, 0.2]
            )
        tests["yaw_0.4"]["diagnostics"]["final_window"] = None
        self.assertIsNone(
            mirror_pair_summary(tests)["yaw_-0.4 / yaw_0.4"]["absolute_mirror_mismatch"]
        )
        for name in POSE_RECOVERY_CASES:
            self.assertEqual(commands[name][1], (0, 0, 0))
        self.assertEqual(commands["yaw_to_stop"][0], (0, 0, 1.0))
        self.assertEqual(commands["lateral_positive_to_stop"][0], (0, 0.25, 0))
        self.assertEqual(commands["lateral_negative_to_stop"][0], (0, -0.25, 0))

    def test_pose_recovery_requires_leg_and_hip_settling_and_valid_reference(self):
        data = np.zeros((100, 4, 19))
        detail = {"leg_position_error": np.full((100, 4, 12), 0.1)}
        detail["leg_position_error"][50:60, 0] = 0.3
        detail["leg_position_error"][60:70, 0, [0, 3, 6, 9]] = 0.2
        detail["leg_position_error"][95:, 1] = 0.3  # returns outside: not recovered
        data[60, 2, 9] = 1  # reset must not produce spurious recovery
        reference = {
            "survivors": np.array([True, True, True, False]),
            "leg_rms": np.full(4, 0.1),
            "hip_rms": np.full(4, 0.1),
        }
        result = pose_recovery_metrics(data, detail, reference, [0, 3, 6, 9], 0.02, 50)
        self.assertEqual(result["eligible_environments"], 2)
        self.assertEqual(result["time_s_per_environment"], [0.42, None, None, None])
        self.assertEqual(result["recovered_fraction"], 0.5)
        detail["leg_position_error"][50:95, 0] = 0.3
        result = pose_recovery_metrics(data, detail, reference, [0, 3, 6, 9], 0.02, 50)
        self.assertIsNone(result["time_s_per_environment"][0])  # <0.25 s at end

    def test_diagnostics_windows_contacts_and_failed_episodes(self):
        data = np.zeros((100, 2, 19))
        data[:, :, 8] = 0.415
        data[50:75, 0, 3] = 0.5
        data[75:, 0, 3] = 1.0
        data[50:, 0, 15] = 4.0
        data[:, 1, 3] = 100.0
        data[60, 1, 9] = 1.0
        detail = {
            "leg_position_error": np.full((100, 2, 12), 0.1),
            "foot_contacts": np.ones((100, 2, 4), dtype=bool),
            "wheel_velocities": np.tile([1.0, 3.0, 1.0, 3.0], (100, 2, 1)),
            "wheel_actions": np.tile([0.1, 0.3, 0.1, 0.3], (100, 2, 1)),
            "foot_positions_body": np.zeros((100, 2, 4, 3)),
        }
        detail["foot_contacts"][50:75, 0, 0] = False
        detail["leg_position_error"][:, :, [0, 3, 6, 9]] = 0.2
        detail["foot_positions_body"][:, :, :, 1] = [0.2, -0.2, 0.2, -0.2]
        result = diagnostic_metrics(
            data, detail, np.array([1.0, 0.0, 0.0]), 0.02, 50, 0.415, [0, 3, 6, 9]
        )
        self.assertEqual(result["surviving_environments"], 1)
        self.assertAlmostEqual(result["final_window"]["mean_velocity"][0], 0.75)
        self.assertAlmostEqual(
            result["final_window"]["velocity_rmse"][0], np.sqrt(0.125)
        )
        self.assertAlmostEqual(result["final_window"]["mean_base_height_error"], 0.0)
        post = result["post_transition"]
        self.assertAlmostEqual(post["leg_position_error_rms"], np.sqrt(0.02))
        self.assertAlmostEqual(post["hip_abduction_error_rms"], 0.2)
        self.assertAlmostEqual(post["hip_abduction_error_max_abs"], 0.2)
        self.assertEqual(post["simultaneous_contacts_count"]["3"], 25)
        self.assertEqual(post["simultaneous_contacts_fraction"]["4"], 0.5)
        self.assertEqual(post["wheel_contact_fraction"], [0.5, 1.0, 1.0, 1.0])
        self.assertAlmostEqual(post["stance_width_mean"], 0.4)
        self.assertEqual(post["right_minus_left_wheel_speed"], 2.0)
        self.assertEqual(post["leg_velocity_rms"], 2.0)
        data[60, 0, 9] = 1.0
        failed = diagnostic_metrics(
            data, detail, np.zeros(3), 0.02, 50, 0.415, [0, 3, 6, 9]
        )
        self.assertIsNone(failed["final_window"])
        self.assertIsNone(failed["post_transition"])

    def test_normalization_and_bounds_survive_script_export(self):
        torch.manual_seed(2)
        observations = TensorDict({"policy": torch.randn(64, 56) + 3}, batch_size=[64])
        actor = MLPModel(
            observations,
            {"actor": ["policy"]},
            "actor",
            16,
            hidden_dims=[32, 16],
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.55,
                "std_type": "log",
                "std_range": [0.05, 0.8],
            },
        )
        actor.obs_normalizer.update(observations["policy"])
        actor.eval()
        exported = torch.jit.script(BoundedPolicy(actor.as_jit(), 100.0, 1.0))
        with torch.no_grad():
            expected = actor(observations).clamp(-1, 1)
            actual = exported(observations["policy"])
        torch.testing.assert_close(actual, expected)
        self.assertTrue((actual.abs() <= 1).all())
        self.assertGreater(actor.obs_normalizer.count, 0)

    def test_first_order_response_measurement(self):
        dt = 0.02
        t = np.arange(1, 201) * dt
        velocity = np.zeros((200, 3))
        velocity[:, 0] = 1 - np.exp(-t / 0.3)
        result = response_metrics(velocity, np.array([1.0, 0.0, 0.0]), dt, np.zeros(3))[
            "vx"
        ]
        self.assertAlmostEqual(result["response_63_percent_s"], 0.3, delta=dt)
        self.assertAlmostEqual(result["rise_time_10_90_s"], 0.3 * np.log(9), delta=dt)
        self.assertEqual(result["overshoot"], 0)


if __name__ == "__main__":
    unittest.main()
