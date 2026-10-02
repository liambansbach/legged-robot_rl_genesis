"""Focused transfer_v1 CPU regressions and opt-in bounded native preparation."""

import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
import torch
import yaml

from robot_gym.envs import *  # noqa: F401,F403
from robot_gym.envs.go2w.go2w_config import apply_go2w_profile, uses_event_steps, verify_measured_asset
from robot_gym.envs.go2w.go2w_env import Go2WEnv
from robot_gym.scripts.train import train, validate_fresh_transfer
from robot_gym.utils import get_args, task_registry
from robot_gym.utils.helpers import class_to_dict, update_cfg_from_args
from robot_gym.utils.diagnostics import loaded_properties, nominal_support_heights, sha256, write_json
from robot_gym.utils.export import export_policy, select_transfer_dynamics
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evaluation/transfer_v1_preparation_20261001"


def args_for(count=1, *extra):
    argv = ["transfer-preparation", "--task", "go2w", "--go2w_profile", "transfer_v1",
            "--num_envs", str(count), "--seed", "1", "--headless", "--logger", "tensorboard", *extra]
    with patch.object(sys, "argv", argv):
        return get_args()


def recipe():
    cfg, training = task_registry.get_cfgs("go2w")
    apply_go2w_profile(cfg, training, "transfer_v1")
    return cfg, training


class TransferCPU(unittest.TestCase):
    def test_selected_p_behavior_and_fresh_recipe(self):
        cfg, training = recipe()
        saved = yaml.safe_load((ROOT / "ressources/pretrained/go2w/config.yaml").read_text())
        for field in ("commands", "rewards", "noise", "normalization", "init_state"):
            self.assertEqual(json.loads(json.dumps(class_to_dict(getattr(cfg, field)))), saved["env_cfg"][field], field)
        for field in ("actor", "critic", "algorithm"):
            self.assertEqual(class_to_dict(getattr(training, field)), saved["train_cfg"][field], field)
        self.assertEqual((cfg.env.num_observations, cfg.env.num_actions, cfg.env.num_privileged_obs), (56, 16, None))
        self.assertEqual(training.runner.obs_groups, {"actor": ["policy"], "critic": ["policy"]})
        self.assertTrue(uses_event_steps(cfg))
        self.assertEqual((training.runner.num_steps_per_env, training.runner.max_iterations, training.runner.save_interval), (64, 2000, 100))
        self.assertFalse(training.runner.resume)
        self.assertIsNone(training.runner.load_run)
        self.assertIsNone(training.runner.checkpoint)
        self.assertIsNone(training.runner.resume_path)
        self.assertFalse(hasattr(cfg, "_event_completed_updates"))
        validate_fresh_transfer(args_for(), training)
        before = class_to_dict(cfg)
        apply_go2w_profile(cfg, training, "transfer_v1")
        self.assertEqual(before, class_to_dict(cfg))
        legacy, _ = task_registry.get_cfgs("go2w")
        self.assertIsNone(legacy.control.armature)

    def test_reject_initialization_before_environment(self):
        for options in (("--resume",), ("--load_run", "old"), ("--checkpoint", "0"),
                        ("--reference_config", "old/config.yaml")):
            with patch.object(task_registry, "make_env") as construct:
                with self.assertRaisesRegex(ValueError, "fresh only|checkpoint initialization"):
                    train(args_for(64, *options))
                construct.assert_not_called()
        for option in ("--go2w_finetune", "--event_quality_profile"):
            with self.assertRaises(ValueError):
                cfg, training = recipe()
                value = "coverage" if option == "--go2w_finetune" else "sufficient_clearance"
                update_cfg_from_args(cfg, training, args_for(64, option, value))

    def test_measured_geometry_and_nominal_fk(self):
        cfg, _ = recipe()
        self.assertEqual(verify_measured_asset(), .09167)
        path = ROOT / "ressources/robots/go2w/urdf" / cfg.asset.robot_file
        mass = sum(float(e.get("value")) for e in ET.parse(path).getroot().findall("link/inertial/mass"))
        self.assertAlmostEqual(mass, 19.68371, places=4)
        support = nominal_support_heights(path, cfg.init_state.default_joint_angles, cfg.asset.foot_link_names)
        clearance = cfg.init_state.pos[2] - np.asarray(support)
        self.assertTrue((clearance > 0).all())
        self.assertTrue((clearance < .08).all())
        self.assertEqual(cfg.rewards.base_height_target, .415)
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / "geometry.json", {"mass_kg": mass, "wheel_radius_m": .09167,
                   "nominal_support_base_heights_m": support, "spawn_clearances_m": clearance,
                   "reward_height_m": cfg.rewards.base_height_target})

    def test_observation_velocity_slice_and_noise(self):
        cfg, _ = recipe()
        env = Go2WEnv.__new__(Go2WEnv)
        env.cfg, env.device, env.num_obs = cfg, torch.device("cpu"), 56
        env.joint_names = list(cfg.init_state.default_joint_angles)
        env.leg_action_indices = [i for i, n in enumerate(env.joint_names) if not n.endswith("_foot_joint")]
        env.wheel_action_indices = [3, 7, 11, 15]
        env.obs_scales = cfg.normalization.obs_scales
        env.commands_scale = torch.ones(3)
        env.base_lin_vel = torch.zeros(2, 3)
        env.base_ang_vel = torch.zeros(2, 3)
        env.projected_gravity = torch.tensor([[0., 0., -1.]]).expand(2, -1)
        env.commands = torch.zeros(2, 3)
        env.dof_pos = env.default_dof_pos = torch.zeros(2, 16)
        env.dof_vel = torch.zeros(2, 16)
        env.actions = torch.linspace(-1, 1, 16).expand(2, -1)
        noise = env._get_noise_scale_vec(cfg)
        torch.testing.assert_close(noise[:3], torch.full((3,), .05))
        torch.testing.assert_close(noise[9:12], torch.zeros(3))
        torch.testing.assert_close(noise[40:], torch.zeros(16))
        torch.testing.assert_close(noise[24 + torch.tensor(env.wheel_action_indices)], torch.full((4,), .5))
        env.add_noise = False
        env.compute_observations()
        before = env.obs_buf.clone()
        env.base_lin_vel[:] = torch.tensor([.2, -.1, .3])
        env.compute_observations()
        torch.testing.assert_close(env.obs_buf[:, :3] - before[:, :3], env.base_lin_vel * env.obs_scales.lin_vel)
        torch.testing.assert_close(env.obs_buf[:, 3:], before[:, 3:], rtol=0, atol=0)
        torch.testing.assert_close(env.obs_buf[:, 40:], env.actions)

    def test_explicit_evaluation_dynamics(self):
        for selection, expected in (("nominal", .01), ("low", .005), ("high", .02)):
            cfg, _ = recipe()
            select_transfer_dynamics(cfg, args_for(1, "--transfer_armature", selection, "--transfer_delay", "2"))
            self.assertFalse(cfg.domain_rand.randomize_armature)
            self.assertEqual(set(cfg.control.armature.values()), {.01})
            self.assertEqual(cfg.control.armature_override, None if selection == "nominal" else expected)
            self.assertEqual(cfg.domain_rand.action_delay_steps_range, [2, 2])

    def test_nominal_export_metadata_and_panel(self):
        from robot_gym.utils.export import transfer_contract
        from robot_gym.utils.urdf_reader import URDFReader
        from robot_gym.scripts.diagnostic_bank import transfer_schedule
        from robot_gym.scripts.evaluate import use_physics_diagnostics
        cfg, _ = recipe()
        select_transfer_dynamics(cfg, args_for(1, "--transfer_armature", "low"))
        options = SimpleNamespace(constraint_solver="Newton")
        env = SimpleNamespace(cfg=cfg, joint_names=list(cfg.init_state.default_joint_angles),
                              urdf_reader=URDFReader(cfg.asset.robot_file),
                              sim=SimpleNamespace(rigid_solver=SimpleNamespace(_integrator=SimpleNamespace(name="approximate_implicitfast"), _options=options)))
        metadata = transfer_contract(env)
        self.assertEqual(set(metadata["nominal_armature_kg_m2"].values()), {.01})
        self.assertEqual(metadata["training_randomization"]["armature_kg_m2"], [.005,.02])
        self.assertEqual(metadata["velocity"]["reference_point"], "authored base-link origin")
        self.assertEqual(metadata["asset"]["sha256"], cfg.asset.sha256)
        panel = transfer_schedule()
        self.assertEqual(len(panel), 8)
        self.assertEqual(panel["forward"], [(3,(0.,0.,0.)),(5,(.2,0,0)),(6,(0.,0.,0.))])
        self.assertTrue(all(sum(t for t,_ in schedule)<=14 for schedule in panel.values()))
        args = args_for(1,"--eval_mode","transfer_screen","--transfer_cases","stand","forward")
        self.assertEqual(args.transfer_cases,["stand","forward"])
        self.assertFalse(use_physics_diagnostics(args))
        args.diagnostic_trace = True
        self.assertTrue(use_physics_diagnostics(args))


@unittest.skipUnless(os.environ.get("GO2W_TRANSFER_GPU") == "plant", "bounded native plant only")
class TransferPlant(unittest.TestCase):
    def test_nominal_endpoints_and_selective_reset(self):
        import genesis as gs
        from robot_gym.utils.diagnostics import PhysicsDiagnostics, rotate_wxyz, cylinder_clearance, summed_normal_force
        cfg, _ = recipe()
        args = args_for(3)
        cfg.env.play_mode = True
        cfg.env.capture_transitions = True
        cfg.noise.add_noise = False
        for key in ("randomize_armature", "randomize_friction", "randomize_base_mass", "randomize_com",
                    "randomize_kp", "randomize_kd", "randomize_action_delay", "push_robots"):
            setattr(cfg.domain_rand, key, False)
        for key in ("joint_position_noise", "joint_velocity_noise", "linear_velocity_noise", "angular_velocity_noise"):
            setattr(cfg.init_state, key, 0.)
        cfg.init_state.orientation_noise = (0., 0., 0.)
        env, _ = task_registry.make_env("go2w", args=args, env_cfg=cfg)
        try:
            torch.testing.assert_close(env.robot.get_dofs_armature(env.joint_dof_idx), torch.full((3,16), .01, device=env.device))
            self.assertEqual(env.robot.get_dofs_armature(list(range(6))).count_nonzero(), 0)
            values = torch.tensor([.01, .005, .02], device=env.device)[:, None].expand(-1, 16)
            env.armature_samples.copy_(values)
            env.robot.set_dofs_armature(values, env.joint_dof_idx)
            env.reset()
            before = env.robot.get_dofs_armature(env.joint_dof_idx).clone()
            env.reset_idx(torch.tensor([1], device=env.device))
            torch.testing.assert_close(env.robot.get_dofs_armature(env.joint_dof_idx), before, rtol=0, atol=0)
            properties = loaded_properties(env)
            torch.testing.assert_close(properties["kp"], env.base_p_gains.expand(3,-1))
            torch.testing.assert_close(properties["kv"], env.base_d_gains.expand(3,-1))
            for field in ("stiffness", "damping", "frictionloss"):
                self.assertEqual(getattr(env.robot, "get_dofs_" + field)(env.joint_dof_idx).count_nonzero(), 0)
            self.assertAlmostEqual(float(env.robot.get_links_mass()[0].sum()), 19.68371, places=4)
            env.set_fixed_command((0.,0.,0.))
            rows = []
            class Capture(PhysicsDiagnostics):
                def after_substep(recorder):
                    super().after_substep()
                    r = env.robot
                    inverse = r.get_quat().clone(); inverse[:,1:] *= -1
                    contact = r.get_contacts(with_entity=env.ground_floor_entity, exclude_self_contact=True, is_padded=True)
                    rows.append(torch.cat((r.get_pos(), rotate_wxyz(inverse,r.get_vel()), rotate_wxyz(inverse,r.get_ang()),
                                           r.get_dofs_position(env.joint_dof_idx), r.get_dofs_velocity(env.joint_dof_idx),
                                           cylinder_clearance(r.get_links_pos(env.foot_link_indices_local), r.get_links_quat(env.foot_link_indices_local), *recorder.geometry),
                                           summed_normal_force(contact, env.foot_link_indices)), dim=1).clone())
            env.physics_diagnostics = Capture(env)
            actions = torch.zeros((3,16), device=env.device)
            for tick in range(450):
                actions.zero_()
                if 250 <= tick < 300:
                    actions[:, [1,5,9,13]] = .01/.35
                env.step(actions)
                self.assertTrue(torch.isfinite(env.obs_buf).all())
                self.assertFalse(bool(env.transition_state["reset_buf"].any()), "Plant failure: retain evidence and consider the single smaller-step diagnostic")
            trace = torch.stack(rows).cpu().numpy()
            np.savez_compressed(OUT / "plant.npz", states=trace, physics_dt=cfg.sim.dt,
                                armature=values.cpu().numpy())
            result = {"armature_kg_m2": values, "properties": properties,
                      "integrator": str(env.sim.rigid_solver._options.integrator),
                      "settled_height_m": trace[-400:,:,2].mean(0),
                      "final_planar_speed_rms_m_s": np.sqrt(np.mean(np.sum(trace[-400:,:,3:5]**2,axis=-1),axis=0)),
                      "final_joint_velocity_rms_rad_s": np.sqrt(np.mean(trace[-400:,:,25:41]**2,axis=(0,2))),
                      "final_body_rate_rms_rad_s": np.sqrt(np.mean(trace[-400:,:,6:8]**2,axis=(0,2))),
                      "final_support_N": trace[-400:,:,45:49].sum(-1).mean(0),
                      "minimum_clearance_m": trace[:,:,41:45].min(axis=(0,2)), "steps":450, "status":"passed"}
            self.assertTrue((result["final_joint_velocity_rms_rad_s"] < .1).all(), "Fixed-target plant oscillation")
            self.assertTrue((result["final_body_rate_rms_rad_s"] < .1).all(), "Fixed-target body oscillation")
            write_json(OUT / "plant.json", result)
            print("TRANSFER PLANT", json.dumps({k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in result.items() if k not in ("properties","armature_kg_m2")}))
        finally:
            gs.destroy()


@unittest.skipUnless(os.environ.get("GO2W_TRANSFER_GPU") == "smoke", "one fresh two-update smoke only")
class TransferSmoke(unittest.TestCase):
    def test_two_updates_serialization_and_export(self):
        import genesis as gs
        from robot_gym.utils.training_diagnostics import verify_resume_state
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        original = task_registry.make_alg_runner
        measurements = {"transitions":0, "optimizer_steps":0}
        def construct(*args, **kwargs):
            runner, cfg = original(*args, **kwargs)
            env = runner.env
            self.assertIsNone(runner.checkpoint_path)
            self.assertFalse(runner.alg.optimizer.state)
            self.assertEqual(runner.current_learning_iteration, 0)
            self.assertEqual(env.completed_updates, 0)
            torch.testing.assert_close(runner.alg.actor.distribution.log_std_param.exp(), torch.full((16,), .4, device=env.device))
            self.assertEqual(runner.cfg["obs_groups"], {"actor":["policy"], "critic":["policy"]})
            self.assertIsNot(runner.alg.actor.obs_normalizer, runner.alg.critic.obs_normalizer)
            for model in (runner.alg.actor,runner.alg.critic):
                self.assertEqual(model.obs_normalizer.count,0)
                self.assertEqual(model.obs_normalizer._mean.count_nonzero(),0)
                torch.testing.assert_close(model.obs_normalizer._var,torch.ones_like(model.obs_normalizer._var))
            arms = env.robot.get_dofs_armature(env.joint_dof_idx).clone()
            self.assertTrue(((arms>=.005)&(arms<=.02)).all())
            for group in (env.leg_action_indices, env.wheel_action_indices):
                torch.testing.assert_close(arms[:,group], arms[:,group[:1]].expand(-1,len(group)))
            mass = env.robot.get_links_mass([env.base_link_idx])
            inertia = env.robot.get_links_inertia([env.base_link_idx])
            ratio = (mass/env.nominal_base_mass)[...,None,None]
            torch.testing.assert_close(inertia, env.nominal_base_inertia * ratio)
            self.assertTrue(((mass-env.nominal_base_mass>=-.5)&(mass-env.nominal_base_mass<=1.5)).all())
            measurements["properties"] = loaded_properties(env)
            measurements["armature"] = arms
            observation = env.get_observations()["policy"].clone()
            measurements["actual_initial_observation"] = observation
            step = env.step
            def checked_step(actions):
                result = step(actions)
                self.assertTrue(torch.isfinite(result[1]).all())
                self.assertTrue(torch.isfinite(env.obs_buf).all())
                measurements["transitions"] += 1
                return result
            env.step = checked_step
            def optimizer(*_):
                for group in runner.alg.optimizer.param_groups:
                    for p in group["params"]:
                        self.assertTrue(torch.isfinite(p).all())
                        if p.grad is not None: self.assertTrue(torch.isfinite(p.grad).all())
                measurements["optimizer_steps"] += 1
            runner.alg.optimizer.register_step_post_hook(optimizer)
            return runner,cfg
        try:
            args = args_for(64, "--max_iterations", "2", "--training_diagnostics",
                            "--experiment_name", "go2w_transfer_v1_smoke", "--run_name", "transfer_v1_smoke_seed1_20261001")
            with patch.object(task_registry,"make_alg_runner",construct):
                runner = train(args)
            env = runner.env
            self.assertEqual((measurements["transitions"],measurements["optimizer_steps"]), (128,80))
            self.assertEqual(env.completed_updates,2)
            out = Path(runner.logger.log_dir)
            checkpoint = out / "model_1.pt"
            verify_resume_state(runner,checkpoint)
            # Serialization reload is inference-only. It is never a training parent.
            runner.load(checkpoint)
            runner.checkpoint_path = str(checkpoint)
            before = loaded_properties(env)
            delays = env.action_delay_steps.clone()
            env.reset_idx(torch.tensor([0],device=env.device))
            after = loaded_properties(env)
            for key in ("mass","com_local","inertia_local","wheel_friction_ratio"):
                torch.testing.assert_close(before[key],after[key],rtol=0,atol=0)
            for key in ("kp","kv"):
                torch.testing.assert_close(before[key][1:],after[key][1:],rtol=0,atol=0)
            torch.testing.assert_close(delays[1:],env.action_delay_steps[1:],rtol=0,atol=0)
            torch.testing.assert_close(measurements["armature"],env.robot.get_dofs_armature(env.joint_dof_idx),rtol=0,atol=0)
            env._update_robot_state(); env.compute_observations()
            observations = torch.cat((env.obs_buf.clone(), measurements["actual_initial_observation"], torch.randn((32,56),device=env.device)))
            policy = runner.get_inference_policy(device=env.device)
            from tensordict import TensorDict
            with torch.no_grad():
                expected = policy(TensorDict({"policy":observations}, batch_size=[len(observations)])).clamp(-1,1).cpu()
            export_policy(runner,env,out / "exported")
            actor = torch.jit.load(str(out / "exported/policy_1.pt"),map_location="cpu").eval()
            with torch.no_grad(): actual = actor(observations.cpu())
            torch.testing.assert_close(actual,expected,rtol=5e-5,atol=5e-5)
            contract = json.loads((out / "exported/contract.json").read_text())
            self.assertEqual((contract["observation_dim"],contract["action_dim"]),(56,16))
            self.assertEqual(set(contract["nominal_armature_kg_m2"].values()),{.01})
            events = EventAccumulator(str(out)).Reload()
            losses = {k:[x.value for x in events.Scalars(k)] for k in events.Tags()["scalars"] if k.startswith("Loss/")}
            self.assertTrue(losses)
            self.assertTrue(all(len(v)==2 and np.isfinite(v).all() for v in losses.values()))
            measurements.update(run=str(out), checkpoint_sha256=sha256(checkpoint), losses=losses,
                                export_max_error=float((actual-expected).abs().max()),
                                completed_updates=env.completed_updates, status="passed")
            write_json(OUT / "smoke.json",measurements)
            print("TRANSFER SMOKE PASS",out, "parity",measurements["export_max_error"],flush=True)
        finally:
            gs.destroy()
