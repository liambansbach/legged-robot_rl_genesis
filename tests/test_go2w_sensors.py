"""Current phase observation and rigid-sensor transport regression checks."""
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from robot_gym.envs.go2w.go2w_config import MEASURED_URDF
from robot_gym.envs.go2w.go2w_env import fixed_sensor_frames, rigid_sensor_state
from robot_gym.utils.urdf_reader import URDFReader


class SensorTests(unittest.TestCase):

    def test_conditioned_clock_and_reflection_preserve_stepping(self):
        from robot_gym.envs.go2w.phase import clock_observation, demand
        phases = torch.tensor([.2,.3,.4,.5,.6])
        commands = torch.tensor([[0.,0.,0.],[.5,0.,0.],[0.,.03,0.],[0.,.3,0.],[0.,0.,-.8]])
        legacy = clock_observation(phases, commands)
        conditioned = clock_observation(phases, commands, 'command_demand')
        torch.testing.assert_close(conditioned[:2], torch.zeros(2,2))
        torch.testing.assert_close(conditioned[2], legacy[2]*.5)
        torch.testing.assert_close(conditioned[3:], legacy[3:])
        mirrored = commands*torch.tensor([1.,-1.,-1.])
        torch.testing.assert_close(demand(mirrored), demand(commands))
        torch.testing.assert_close(clock_observation((phases+.5)%1,mirrored,'command_demand'), -conditioned, atol=1e-6, rtol=1e-5)

    def test_fixed_lr_kl_uses_original_distribution_without_forward(self):
        from tensordict import TensorDict
        from robot_gym.envs.go2w.training_diagnostics import TrainingDiagnostics

        old_mean = torch.zeros(2, 16)
        old_std = torch.ones(2, 16)
        mean = torch.cat((torch.full((2, 16), .1), torch.full((2, 16), 10.)))
        batch = SimpleNamespace(observations=TensorDict({'policy': torch.zeros(2, 58)}, batch_size=[2]),
                                old_distribution_params=(old_mean, old_std), old_actions_log_prob=torch.zeros(2, 1))
        def divergence(old, current):
            m0, s0 = old
            m1, s1 = current
            return (torch.log(s1/s0)+(s0.square()+(m0-m1).square())/(2*s1.square())-.5).sum(-1)
        log_prob_result = torch.zeros(4, requires_grad=True)
        actor = SimpleNamespace(get_output_log_prob=lambda actions: log_prob_result,
                                get_kl_divergence=divergence, output_distribution_params=(mean, torch.ones_like(mean)))
        alg = SimpleNamespace(actor=actor, act=lambda obs: None, process_env_step=lambda *args: None,
                              update=lambda: None, clip_param=.2,
                              storage=SimpleNamespace(mini_batch_generator=lambda: iter([batch])))
        diagnostic = TrainingDiagnostics.__new__(TrainingDiagnostics)
        diagnostic.measured_kl, diagnostic.kl, diagnostic.ppo_clip = [], [], []
        diagnostic.current_batch = None
        diagnostic.install(alg)
        rng = torch.get_rng_state().clone()
        generator = alg.storage.mini_batch_generator()
        next(generator)
        # Native symmetry duplicates log-probs but retains original distribution parameters.
        batch.observations = torch.cat((batch.observations, batch.observations), dim=0)
        batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(2, 1)
        self.assertIs(actor.get_output_log_prob(torch.zeros(4,16)), log_prob_result)
        self.assertAlmostEqual(float(diagnostic.measured_kl[0]), .08, places=6)
        self.assertEqual(diagnostic.kl, [])
        self.assertFalse(diagnostic.measured_kl[0].requires_grad)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        with self.assertRaises(StopIteration): next(generator)
        self.assertIsNone(diagnostic.current_batch)

    def test_authored_fixed_geometry(self):
        frames = fixed_sensor_frames(URDFReader(MEASURED_URDF).robot_file_path_absolute)
        np.testing.assert_allclose(frames["front_realsense"]["translation_m"], [.33881, .0475, .111], atol=1e-6)
        np.testing.assert_allclose(frames["front_realsense_body"]["translation_m"], [.31736, 0., .111], atol=1e-6)
        np.testing.assert_allclose(frames["front_realsense_body"]["rotation"], np.eye(3), atol=1e-6)
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
