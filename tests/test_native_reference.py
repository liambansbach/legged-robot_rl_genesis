"""CPU checks for explicit identity selection and oscillation/failure metrics."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from robot_gym.scripts.native_reference import interval_metrics, spectrum, metrics
from robot_gym.utils.urdf_reader import URDFReader
from robot_gym.utils.diagnostics import joint_dynamics


class NativeReferenceTests(unittest.TestCase):
    def test_explicit_selection_and_ambiguous_fallback(self):
        xml = '<robot name="test"><link name="root"/><link name="tip"/><joint name="drive" type="continuous"><parent link="root"/><child link="tip"/></joint></robot>'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for directory in ("a/urdf", "b/urdf"):
                (root / directory).mkdir(parents=True)
                (root / directory / "same.urdf").write_text(xml)
            with patch("robot_gym.utils.urdf_reader.ROBOTS_DIR", root):
                with self.assertRaisesRegex(ValueError, "Ambiguous"):
                    URDFReader("same.urdf")
                selected = URDFReader(str(root / "b/urdf/same.urdf"))
                self.assertEqual(selected.robot_file_path_absolute, root / "b/urdf/same.urdf")
                self.assertEqual(selected.joint_names, ["drive"])
                with self.assertRaises(FileNotFoundError):
                    URDFReader(str(root / "missing.urdf"))

    def test_effective_getters_do_not_invent_missing_zero(self):
        class Robot:
            def get_joint(self, name):
                return SimpleNamespace(dofs_idx_local=[0], get_sol_params=lambda: torch.ones(7))
            def get_dofs_kp(self, indices):
                return torch.tensor([[40.]])
            def get_dofs_kv(self, indices):
                return torch.tensor([[1.]])
            def get_dofs_force_range(self, indices):
                return torch.tensor([-23.7]), torch.tensor([23.7])
            def get_dofs_limit(self, indices):
                return torch.tensor([-float("inf")]), torch.tensor([float("inf")])
        result = joint_dynamics(Robot(), ["drive"])
        self.assertIsNone(result["joints"]["drive"]["armature"])
        self.assertEqual(result["sources"]["armature"], "unavailable")
        self.assertEqual(result["joints"]["drive"]["kp"], 40)
        self.assertEqual(result["joints"]["drive"]["position_limit"], [None, None])

    def test_quiet_mean_does_not_hide_oscillation_or_incomplete_interval(self):
        time = np.arange(1, 2001) * .005
        alternating = np.sin(2 * np.pi * 25 * time)
        trace = {
            "physics_time": time, "linear_body": np.column_stack((alternating, time * 0, time * 0)),
            "angular_body": np.column_stack((alternating, time * 0, alternating)),
            "dq": alternating[:, None], "base_pos": np.column_stack((time * 0, time * 0, time * 0 + .4)),
            "targets": alternating[:, None], "wheel_normal_force": np.ones((2000, 4)) * 50,
            "clearance": np.zeros((2000, 4)),
        }
        result = interval_metrics(trace, 2, 10, [0])
        self.assertLess(abs(result["mean_velocity"][0]), 1e-6)
        self.assertGreater(result["planar_speed_rms_m_s"], .7)
        self.assertEqual(result["body_rate_spectrum"]["peak_Hz"], 25)
        self.assertGreater(result["body_rate_spectrum"]["near_25Hz_power_fraction"], .99)
        trace = {key: value[:200] for key, value in trace.items()}
        self.assertFalse(interval_metrics(trace, 2, 10, [0])["complete"])
        self.assertIsNone(spectrum(np.empty((0, 2)), 50))
        trace.update({"q": np.zeros((200, 1)), "actions": np.zeros((50, 1)),
                      "policy_time": np.arange(50) * .02, "control_force": np.zeros((200, 1)),
                      "nonwheel_normal_force": np.zeros((200, 1))})
        env = SimpleNamespace(leg_action_indices=[0], dof_pos_limits=torch.tensor([[-1., 1.]]),
                              torque_limits=torch.tensor([23.7]))
        failure = {"kind": "fall", "time_s": 1.}
        reported = metrics(trace, env, "stand", failure, trace["base_pos"][0])
        self.assertEqual(reported["failure"], failure)
        self.assertFalse(reported["credible_support"])
        self.assertFalse(reported["final_2s"]["complete"])
        self.assertIsNone(reported["last_5s_zero_drift"])
        self.assertIsNone(reported["actor_period_two_amplitude_per_leg"])


if __name__ == "__main__":
    unittest.main()
