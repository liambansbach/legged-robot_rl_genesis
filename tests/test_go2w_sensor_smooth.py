"""Focused rigid-sensor geometry and optional refinement checks; no PPO updates."""

import unittest

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from robot_gym.envs.go2w.go2w_config import MEASURED_URDF
from robot_gym.envs.go2w.go2w_env import fixed_sensor_frames, rigid_sensor_state
from robot_gym.envs.go2w.diagnostic_bank import sensor_schedule
from robot_gym.utils.urdf_reader import URDFReader


class SensorMotion(unittest.TestCase):
    def test_authored_fixed_geometry(self):
        frames = fixed_sensor_frames(URDFReader(MEASURED_URDF).robot_file_path_absolute)
        np.testing.assert_allclose(frames["front_realsense"]["translation_m"], [.33881, .0475, .111], atol=1e-6)
        np.testing.assert_allclose(frames["front_realsense"]["rotation"], np.eye(3), atol=1e-7)
        np.testing.assert_allclose(frames["radar"]["translation_m"], [.28945, 0, -.046825], atol=1e-6)
        self.assertFalse(np.allclose(frames["radar"]["rotation"], np.eye(3)))
        self.assertGreater(len(frames["front_realsense"]["fixed_chain"]), 1)

    def test_transport_and_world_up(self):
        p = torch.zeros(1, 3)
        q = torch.tensor([[1., 0., 0., 0.]])
        r = torch.tensor([[.3, .05, .1]])
        _, v = rigid_sensor_state(p, q, torch.tensor([[.5, 0., 0.]]), torch.tensor([[0., 2., 0.]]), r)
        torch.testing.assert_close(v, torch.tensor([[[.7, 0., -.6]]]))
        # Pure commanded yaw around a level base moves the point horizontally only.
        _, v = rigid_sensor_state(p, q, torch.zeros(1, 3), torch.tensor([[0., 0., .8]]), r)
        self.assertEqual(float(v[0, 0, 2]), 0.)

    def test_symmetric_virtual_rig_cost(self):
        reflection = np.diag([1., -1., 1.])
        rotation = Rotation.from_euler('xyz', [.2, -.1, .4]).as_matrix()
        other = reflection @ rotation @ reflection
        quats = torch.tensor(Rotation.from_matrix(np.stack([rotation, other])).as_quat()[:, [3, 0, 1, 2]], dtype=torch.float)
        vel = torch.tensor([[.4, .2, .1], [.4, -.2, .1]])
        ang = torch.tensor([[.6, -.2, .8], [-.6, -.2, -.8]])
        offsets = torch.tensor([[.33881, .0475, .111], [.33881, -.0475, .111]])
        _, v = rigid_sensor_state(torch.zeros(2, 3), quats, vel, ang, offsets)
        torch.testing.assert_close(v[:, :, 2].square().mean(1)[0], v[:, :, 2].square().mean(1)[1])

    def test_sustained_schedule(self):
        schedule = sensor_schedule()
        self.assertEqual(len(schedule), 8)
        self.assertEqual(schedule['stand'], [(30, (0., 0., 0.))])
        for name, phases in schedule.items():
            if name != 'stand':
                self.assertEqual([x[0] for x in phases], [3, 30, 8])
                self.assertEqual(phases[-1][1], (0., 0., 0.))


if __name__ == '__main__':
    unittest.main()
