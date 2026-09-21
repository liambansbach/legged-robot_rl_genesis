import unittest
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from robot_gym.utils.export import BoundedPolicy
from robot_gym.scripts.evaluate import response_metrics
import numpy as np


class InferenceTests(unittest.TestCase):
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
