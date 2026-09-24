"""CPU reflection contracts; the separate opt-in integration test uses real GPU PPO."""

from pathlib import Path
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import torch
import yaml
from tensordict import TensorDict
from rsl_rl.extensions import Symmetry
from rsl_rl.models import MLPModel

from robot_gym.envs.go2w.go2w_config import GO2WCfg, GO2WCfgPPO
from robot_gym.envs.go2w.go2w_env import Go2WEnv
from robot_gym.envs.go2w.go2w_symmetry import (
    mirror_actions,
    mirror_observations,
    sagittal_augmentation,
)
from robot_gym.utils import task_registry
from robot_gym.utils.helpers import class_to_dict


# Explicit expected output->input maps in the documented runtime order.
ACTION_PERM = [4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11]
ACTION_SIGN = [-1, 1, 1, 1] * 4
LEG_PERM = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
LEG_SIGN = [-1, 1, 1] * 4


class SymmetryTests(unittest.TestCase):
    def setUp(self):
        self.cfg = GO2WCfg()
        self.env = SimpleNamespace(
            num_obs=56,
            num_actions=16,
            joint_names=list(self.cfg.init_state.default_joint_angles),
            leg_action_indices=[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14],
        )

    def test_urdf_nominal_and_actuator_conventions(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "ressources/robots/go2w/urdf/go2w_description.urdf"
        )
        joints = [
            j
            for j in ET.parse(path).getroot().findall("joint")
            if j.get("type") != "fixed"
        ]
        self.assertEqual([j.get("name") for j in joints], self.env.joint_names)
        for joint in joints:
            name = joint.get("name")
            hip, wheel = name.endswith("_hip_joint"), name.endswith("_foot_joint")
            self.assertEqual(
                [float(v) for v in joint.find("axis").get("xyz").split()],
                [1, 0, 0] if hip else [0, 1, 0],
            )
            self.assertEqual(
                [float(v) for v in joint.find("origin").get("rpy").split()], [0, 0, 0]
            )
            self.assertEqual(self.cfg.control.control_type[name], "V" if wheel else "P")
            self.assertEqual(
                self.cfg.control.action_scale[name], 18.0 if wheel else 0.2
            )
        nominal = torch.tensor(
            [
                [
                    self.cfg.init_state.default_joint_angles[n]
                    for n in self.env.joint_names
                ]
            ]
        )
        torch.testing.assert_close(mirror_actions(self.env, nominal), nominal)

    def test_unique_vectors_joints_and_actions(self):
        # Build through the real observation method, so changed block layouts fail this test.
        env = Go2WEnv.__new__(Go2WEnv)
        env.__dict__.update(vars(self.env))
        env.obs_scales = self.cfg.normalization.obs_scales
        env.commands_scale = torch.ones(3)
        env.base_lin_vel = torch.tensor([[1.0, 2.0, 3.0]])
        env.base_ang_vel = torch.tensor([[4.0, 5.0, 6.0]])
        env.projected_gravity = torch.tensor([[7.0, 8.0, 9.0]])
        env.commands = torch.tensor([[10.0, 11.0, 12.0]])
        env.default_dof_pos = torch.zeros(1, 16)
        env.dof_pos = torch.arange(1.0, 17.0).reshape(1, 16)
        env.dof_vel = env.dof_pos + 20
        env.actions = env.dof_pos / 16  # unique valid bounded policy actions
        env.add_noise = False
        env.compute_observations()
        obs = TensorDict({"policy": env.obs_buf}, batch_size=[1])
        mirrored = mirror_observations(env, obs)["policy"]
        torch.testing.assert_close(
            mirrored[:, :12],
            torch.tensor(
                [[1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, 10.0, -11.0, -12.0]]
            ),
        )
        leg = env.dof_pos[:, env.leg_action_indices]
        torch.testing.assert_close(
            mirrored[:, 12:24], leg[:, LEG_PERM] * torch.tensor(LEG_SIGN)
        )
        torch.testing.assert_close(
            mirrored[:, 24:40], env.dof_vel[:, ACTION_PERM] * torch.tensor(ACTION_SIGN)
        )
        expected_actions = env.actions[:, ACTION_PERM] * torch.tensor(ACTION_SIGN)
        torch.testing.assert_close(mirrored[:, 40:56], expected_actions)
        torch.testing.assert_close(mirror_actions(env, env.actions), expected_actions)

    def test_involution_shape_none_and_input_immutability(self):
        torch.manual_seed(17)
        for dtype in (torch.float32, torch.float64):
            obs = TensorDict(
                {"policy": torch.randn(7, 56, dtype=dtype)}, batch_size=[7]
            )
            actions = torch.rand(7, 16, dtype=dtype) * 2 - 1
            original_obs, original_actions = obs.clone(), actions.clone()
            torch.testing.assert_close(
                mirror_observations(self.env, mirror_observations(self.env, obs)), obs
            )
            torch.testing.assert_close(
                mirror_actions(self.env, mirror_actions(self.env, actions)), actions
            )
            for o, a in ((obs, actions), (obs, None), (None, actions), (None, None)):
                aug_o, aug_a = sagittal_augmentation(env=self.env, obs=o, actions=a)
                if o is None:
                    self.assertIsNone(aug_o)
                else:
                    self.assertEqual(aug_o.batch_size, torch.Size([14]))
                    torch.testing.assert_close(aug_o[:7], original_obs)
                    torch.testing.assert_close(
                        aug_o[7:], mirror_observations(self.env, obs)
                    )
                if a is None:
                    self.assertIsNone(aug_a)
                else:
                    self.assertEqual(aug_a.shape, (14, 16))
                    torch.testing.assert_close(aug_a[:7], original_actions)
                    torch.testing.assert_close(
                        aug_a[7:], mirror_actions(self.env, actions)
                    )
            torch.testing.assert_close(obs, original_obs)
            torch.testing.assert_close(actions, original_actions)

    def test_nominal_observation_and_reordered_runtime_joints(self):
        obs = TensorDict({"policy": torch.zeros(2, 56)}, batch_size=[2])
        obs["policy"][:, 8] = -1  # upright gravity; zero joint errors and commands
        torch.testing.assert_close(mirror_observations(self.env, obs), obs)
        # A valid runtime reorder must still use names, not the numeric maps above.
        self.env.joint_names = self.env.joint_names[::2] + self.env.joint_names[1::2]
        self.env.leg_action_indices = [
            i
            for i, n in enumerate(self.env.joint_names)
            if not n.endswith("_foot_joint")
        ]
        actions = torch.arange(16.0).reshape(1, 16)
        mirrored = mirror_actions(self.env, actions)
        opposite = {"FL": "FR", "FR": "FL", "RL": "RR", "RR": "RL"}
        for i, name in enumerate(self.env.joint_names):
            source = self.env.joint_names.index(opposite[name[:2]] + name[2:])
            self.assertEqual(
                mirrored[0, i], actions[0, source] * (-1 if "_hip_" in name else 1)
            )
        torch.testing.assert_close(
            mirror_observations(self.env, mirror_observations(self.env, obs)), obs
        )

    def test_reject_changed_contract(self):
        with self.assertRaises(ValueError):
            mirror_observations(
                self.env, TensorDict({"critic": torch.zeros(2, 56)}, batch_size=[2])
            )
        with self.assertRaises(ValueError):
            mirror_actions(self.env, torch.zeros(2, 12))
        self.env.joint_names[0] = "unknown_joint"
        with self.assertRaises(ValueError):
            mirror_actions(self.env, torch.zeros(2, 16))

    def test_native_extension_detaches_diagnostic_and_config_is_serializable(self):
        cfg = class_to_dict(GO2WCfgPPO().algorithm)["symmetry_cfg"]
        self.assertEqual(yaml.safe_load(yaml.safe_dump(cfg)), cfg)
        symmetry = Symmetry(env=self.env, **cfg)
        self.assertIs(symmetry.data_augmentation_func, sagittal_augmentation)
        self.assertTrue(symmetry.use_data_augmentation)
        self.assertFalse(symmetry.use_mirror_loss)
        self.assertEqual(symmetry.mirror_loss_coeff, 0.0)
        obs = TensorDict({"policy": torch.randn(4, 56)}, batch_size=[4])
        actor = MLPModel(obs, {"actor": ["policy"]}, "actor", 16, hidden_dims=[16])
        augmented, _ = sagittal_augmentation(self.env, obs=obs)
        batch = SimpleNamespace(observations=augmented)
        loss = symmetry.compute_loss(actor, batch, 4)
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(loss.requires_grad)
        symmetry.use_mirror_loss = True
        self.assertTrue(symmetry.compute_loss(actor, batch, 4).requires_grad)
        for name in ("dodo", "go2"):
            _, train_cfg = task_registry.get_cfgs(name)
            self.assertIsNone(train_cfg.algorithm.symmetry_cfg)


if __name__ == "__main__":
    unittest.main()
