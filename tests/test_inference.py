import unittest
import json
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from robot_gym.envs import *  # noqa: F401,F403 -> register tasks before utils imports
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
from types import SimpleNamespace
from robot_gym.scripts.diagnostic_bank import sustained_schedule, rollout_sequence
from robot_gym.scripts.diagnostic_bank import (
    closed_loop_reference, closed_loop_request, rollout_closed_loop, closed_loop_metrics,
    CLOSED_LOOP_CASES,
)
from robot_gym.scripts.evaluate import use_physics_diagnostics
from robot_gym.scripts.evaluate import swing_events, velocity_window_metrics
from robot_gym.utils.diagnostics import rotate_wxyz, wheel_axles_body
from scipy.spatial.transform import Rotation


class InferenceTests(unittest.TestCase):
    def test_precision_schedules_and_terminal_alignment(self):
        from robot_gym.scripts.diagnostic_bank import precision_schedule, precision_conditions, rollout_precision
        schedules = precision_schedule()
        self.assertEqual([sum(t for t,c in v) for v in schedules.values()], [15,19,19,13,13])
        self.assertEqual(sum(t for t,c in precision_schedule(True)["dynamics"]), 26)
        for family in ("yaw", "lateral"):
            plus, minus = schedules[family+"_positive"], schedules[family+"_negative"]
            for (ta, a), (tb, b) in zip(plus, minus):
                self.assertEqual(ta, tb); np.testing.assert_array_equal(a, -np.asarray(b))
        conditions = precision_conditions()
        self.assertEqual([c["delay"] for c in conditions], [0,0,0,0,0,0,1,2])
        env = SimpleNamespace(dt=.02, device="cpu", num_envs=2, commands=torch.zeros(2,3),
                              episode_length_buf=torch.zeros(2, dtype=torch.long))
        env.compute_observations = lambda: None
        env.get_observations = lambda: torch.zeros(2,56)
        env.ticks = 0
        def step(raw):
            env.ticks += 1
            state = {key: torch.zeros(2, size) for key,size in
                     (("commands",3),("base_pos",3),("base_quat",4),("base_lin_vel",3),("base_ang_vel",3),("rpy",3),
                      ("dof_pos",16),("dof_vel",16),("torques",16),("actions",16),("applied_actions",16),
                      ("foot_pos",12),("foot_contacts",4),("wheel_normal_force",4),("loaded_wheels",4),
                      ("wheel_clearance",4),("wheel_link_quat",16),("wheel_reposition_velocity_body",12))}
            state.update({k: torch.zeros(2) for k in ("fallen","time_out_buf","reset_buf","nonfoot_contact_count")})
            state["episode_length_buf"] = torch.full((2,), env.ticks)
            state["commands"] = env.commands.clone()
            state["base_pos"][:] = env.ticks
            state["actions"] = raw.clamp(-1,1)
            state["applied_actions"] = state["actions"]
            if env.ticks == 2:
                state["reset_buf"][0] = state["fallen"][0] = 1
            env.transition_state = state
        env.step = step
        data = rollout_precision(env, lambda obs: torch.full((2,16),1.2), [(.06,(.1,0,0)),(.04,(0,0,0))])
        np.testing.assert_array_equal(data["valid"][:,0], [1,1,0,0,0])
        np.testing.assert_array_equal(data["valid"][:,1], 1)
        self.assertEqual(data["base_pos"][1,0,0], 2)
        np.testing.assert_array_equal(data["phase"][:,0], [0,0,0,1,1])
        np.testing.assert_allclose(data["raw_actions"], 1.2)
        np.testing.assert_allclose(data["actions"], 1.)
        np.testing.assert_allclose(data["commands"][:3,:,0], .1)
        from robot_gym.scripts.diagnostic_bank import precision_metrics
        names = [f"{side}_{joint}_joint" for side in ("FL", "FR", "RL", "RR")
                 for joint in ("hip", "thigh", "calf", "foot")]
        data["foot_pos"] = data["foot_pos"].reshape(5,2,4,3)
        data["wheel_link_quat"] = data["wheel_link_quat"].reshape(5,2,4,4)
        quaternion = np.array([np.cos(.15), 0, 0, np.sin(.15)], dtype=np.float32)
        data["base_quat"][:] = quaternion
        data["wheel_link_quat"][:] = quaternion
        metadata = dict(joint_order=names, wheel_order=["FL","FR","RL","RR"],
                        wheel_joint_axes=[[0,1,0]]*4, leg_indices=[i for i in range(16) if i%4!=3],
                        wheel_indices=[3,7,11,15], nominal_position=[0]*16,
                        action_scale=[1]*16, force_limits=[100]*16, wheel_target_limit=20)
        initial = dict(base_pos=np.zeros((2,3)), base_quat=np.tile([1.,0,0,0],(2,1)))
        original_device = torch.get_default_device()
        try:
            # Genesis changes the default device; NumPy summaries must stay on CPU.
            torch.set_default_device("meta")
            result = precision_metrics(data, [(.06,(.1,0,0)),(.04,(0,0,0))], .02, metadata, initial)
        finally:
            torch.set_default_device(original_device)
        self.assertTrue(result[0]["failure"])
        self.assertTrue(result[0]["phases"][0]["censored"])
        self.assertEqual(result[0]["phases"][1]["samples"], 0)
        self.assertFalse(result[1]["failure"])
        np.testing.assert_allclose(result[1]["phases"][0]["wheel_toe_mean_rad"], 0, atol=1e-6)
        self.assertEqual(result[1]["phases"][0]["last_second"]["mean_error_vx_vy_yaw"], [-.1,0,0])
        json.dumps(result, allow_nan=False)

    def test_closed_loop_reference_and_frames(self):
        from unittest.mock import patch
        from robot_gym.utils.helpers import get_args

        with patch('sys.argv', ['evaluate']):
            self.assertEqual(get_args().eval_mode, 'nominal')
        with patch('sys.argv', ['evaluate', '--eval_mode', 'closed_loop']):
            self.assertEqual(get_args().eval_mode, 'closed_loop')
        for t, expected in [(0, (0, 0, 0)), (2, (1, 0, 0)), (2.5, (1, .125, .5)),
                            (3, (1, .5, 1)), (13, (1, 10.5, 1)),
                            (13.5, (1, 10.875, .5)), (14, (2, 11, 0)),
                            (17, (3, 11, 0)), (20, (3, 11, 0))]:
            np.testing.assert_allclose(closed_loop_reference(t), expected)
        times = np.linspace(2, 14, 12001)
        speed = [closed_loop_reference(t)[2] for t in times]
        self.assertAlmostEqual(float(np.trapezoid(speed, times)), 11)
        self.assertEqual(CLOSED_LOOP_CASES['diagonal_left'], (.5, .1))
        self.assertEqual(CLOSED_LOOP_CASES['diagonal_right'], (.5, -.1))
        rotation = Rotation.from_euler('xyz', [.15, -.2, .4])
        q = torch.tensor(rotation.as_quat()[[3, 0, 1, 2]])[None]
        position = torch.tensor([[1., 2., .4]], dtype=torch.float64)
        ref = torch.tensor([[1.01, 2.02]], dtype=torch.float64)
        velocity = torch.tensor([[.3, -.1]], dtype=torch.float64)
        heading = torch.tensor([.45], dtype=torch.float64)
        request, issued, correction = closed_loop_request(position, q, ref, velocity, heading)
        np.testing.assert_allclose(request[0, :2], rotation.inv().apply([.31, -.08, 0])[:2], atol=1e-12)
        self.assertAlmostEqual(float(request[0, 2]), .075)
        np.testing.assert_allclose(correction, [[.01, .02]], atol=1e-12)
        np.testing.assert_array_equal(request, issued)
        # A large wrapped heading difference must use the short turn and command bounds.
        q = torch.tensor(Rotation.from_euler('z', np.pi-.01).as_quat()[[3, 0, 1, 2]])[None]
        request, issued, _ = closed_loop_request(position, q, ref+10, velocity, torch.tensor([-np.pi+.01]))
        self.assertAlmostEqual(float(request[0, 2]), .03, places=6)
        self.assertGreater(float((request-issued).abs().max()), 1)

    def test_closed_loop_hold_censoring_and_light_recording(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import Mock, patch
        from robot_gym.scripts.diagnostic_bank import evaluate_closed_loop

        for failure in (None, 'fall', 'timeout', 'reset', 'nonfinite_state'):
            env = SimpleNamespace(dt=.02, device='cpu', num_envs=1, commands=torch.zeros(1, 3),
                                  base_pos=torch.tensor([[0., 0., .4]]), base_quat=torch.tensor([[1., 0., 0., 0.]]),
                                  episode_length_buf=torch.zeros(1, dtype=torch.long),
                                  cfg=SimpleNamespace(control=SimpleNamespace(decimation=4), sim=SimpleNamespace(dt=.005)))
            env.compute_observations = lambda: None
            env.get_observations = lambda: TensorDict({"policy": torch.arange(56).float()[None]}, batch_size=[1])

            def step(action):
                env.episode_length_buf += 1
                env.base_pos[:, 0] += .001
                terminal = int(env.episode_length_buf[0]) == 175 and failure is not None
                state = {k: getattr(env, k).clone() for k in ('base_pos', 'base_quat', 'episode_length_buf')}
                state.update(base_lin_vel=torch.zeros(1, 3), base_ang_vel=torch.zeros(1, 3),
                             dof_pos=torch.zeros(1, 16), dof_vel=torch.zeros(1, 16), torques=torch.zeros(1, 16),
                             actions=action.clamp(-1, 1), applied_actions=action.clamp(-1, 1),
                             foot_contacts=torch.ones(1, 4), nonfoot_contact_count=torch.zeros(1),
                             fallen=torch.tensor([terminal and failure=='fall']),
                             time_out_buf=torch.tensor([terminal and failure=='timeout']),
                             reset_buf=torch.tensor([terminal and failure!='nonfinite_state']))
                if terminal and failure=='nonfinite_state':
                    state['base_pos'][0, 0] = float('nan')
                env.transition_state = state
                if terminal:
                    env.base_pos[:] = -999  # automatic reset must not replace terminal evidence

            env.step = step
            data, reason = rollout_closed_loop(env, lambda obs: torch.ones(1, 16)*1.2, (.5, .1))
            self.assertEqual(reason, failure)
            self.assertEqual(len(data['time_s']), 175 if failure else 1000)
            self.assertNotEqual(float(data['base_pos'][-1, 0]), -999)
            np.testing.assert_allclose(data['raw_actions'], 1.2)
            np.testing.assert_array_equal(data['policy_observation'], np.tile(np.arange(56), (len(data['time_s']), 1)))
            np.testing.assert_allclose(data['actions'], 1)
            np.testing.assert_array_equal(data['issued_commands'][:100], 0)
            for start in range(100, len(data['time_s'])-4, 5):
                np.testing.assert_array_equal(data['issued_commands'][start:start+5],
                                              np.tile(data['issued_commands'][start], (5, 1)))
            result = closed_loop_metrics(data, reason, env.dt)
            self.assertEqual(result['uninterrupted'], failure is None)
            self.assertEqual(result['phases'][1]['censored'], failure is not None)
            if failure:
                self.assertIsNone(result['exact_zero'])
            else:
                self.assertEqual(int(data['outer_update'].sum()), 150)
                np.testing.assert_array_equal(data['issued_commands'][850:], 0)
                np.testing.assert_allclose(data['reference_position_world'][849], [.1+5.5, 1.1], atol=1e-6)
                self.assertIsNotNone(result['endpoint_error_m'])
                self.assertIn('velocity_sign_reversals_above_001', result['windows']['late_hold'])
            env.dt = .01
            with self.assertRaisesRegex(ValueError, '50 Hz'):
                rollout_closed_loop(env, lambda obs: torch.zeros(1, 16), (.5, 0))
        env.dt, env.max_episode_length = .02, 1500
        env.joint_names = [str(i) for i in range(16)]
        env.dof_pos_limits = torch.tensor([[-float('inf'), float('inf')]] * 16)
        env.torque_limits = torch.ones(16)
        env.reset = Mock()
        with TemporaryDirectory() as out, patch(
                'robot_gym.scripts.diagnostic_bank.rollout_closed_loop', return_value=(data, reason)) as rollout:
            result = evaluate_closed_loop(env, None, Path(out))
            self.assertEqual(list(result['tests']), ['straight'])
            self.assertEqual(result['position_limits'], [[None, None]] * 16)
            self.assertEqual(rollout.call_count, 1)
            env.reset.assert_called_once()
        args = SimpleNamespace(eval_mode='closed_loop', diagnostic_trace=False,
                               go2w_profile='step_recovery_v1', zero_command_brake=False)
        self.assertFalse(use_physics_diagnostics(args))
        args.diagnostic_trace = True
        self.assertTrue(use_physics_diagnostics(args))
        args.diagnostic_trace = False
        for args.eval_mode in ('nominal', 'sustained', 'bank'):
            self.assertTrue(use_physics_diagnostics(args))

    def test_sustained_schedule_and_censoring(self):
        schedule = sustained_schedule()
        self.assertEqual(len(schedule), 9)
        for phases in schedule.values():
            self.assertEqual([p[1] for p in phases], [2, 25, 6])
            self.assertEqual(phases[0][0], (0, 0, 0))
            self.assertEqual(phases[-1][0], (0, 0, 0))
        for positive, negative in (("yaw_p040", "yaw_n040"), ("yaw_p075", "yaw_n075"),
                                   ("lateral_p020", "lateral_n020"), ("diagonal_p010", "diagonal_n010")):
            np.testing.assert_allclose(np.asarray(schedule[positive][1][0]) * [1, -1, -1], schedule[negative][1][0])
        for terminal_at, timeout in ((None, False), (175, False), (175, True)):
            env = SimpleNamespace(dt=.02, device="cpu", commands=torch.zeros(1, 3), steps=0)
            env.compute_observations = lambda: None
            env.get_observations = lambda: torch.zeros(1, 56)

            def step(action):
                env.steps += 1
                done = torch.tensor([env.steps == terminal_at])
                env.transition_state = {"commands": env.commands.clone(), "reset_buf": done,
                                        "time_out_buf": done & timeout, "fallen": done & (not timeout)}
                return None, None, done, {}

            env.step = step
            data, phases = rollout_sequence(env, lambda obs: torch.zeros(1, 16), schedule["yaw_p040"], True)
            self.assertEqual(env.steps, terminal_at or 1650)
            self.assertEqual([p["start"] for p in phases], [0, 100, 1350])
            self.assertEqual(phases[1]["censored"], terminal_at is not None)
            if terminal_at is not None:
                self.assertEqual(phases[2]["recorded_steps"], 0)
                self.assertEqual(int(data["time_out_buf"].sum()), int(timeout))
                self.assertEqual(int(data["fallen"].sum()), int(not timeout))
            np.testing.assert_array_equal(data["commands"][:, 0], data["command_stream"].astype(np.float32))

    def test_body_world_and_spin_invariant_axles(self):
        base = Rotation.from_euler("xyz", [.12, -.18, -.3])
        q = torch.tensor(base.as_quat()[[3, 0, 1, 2]])
        velocity = torch.tensor([.7, .1, 0.], dtype=torch.float64)
        world = rotate_wxyz(q, velocity)
        np.testing.assert_allclose(world.numpy(), base.apply(velocity.numpy()))
        self.assertLess(float(world[1]), 0)  # positive body vy can coexist with negative world vy
        axis = torch.tensor([[0., 1., 0.]], dtype=torch.float64)
        expected = Rotation.from_euler("xyz", [.2, 0, -.1])
        results = []
        for spin in (0., 1.7, -2.8):
            wheel = base * expected * Rotation.from_rotvec([0, spin, 0])
            wq = torch.tensor(wheel.as_quat()[[3, 0, 1, 2]])[None]
            results.append(wheel_axles_body(q, wq, axis).numpy())
        for result in results:
            np.testing.assert_allclose(result, expected.apply(axis.numpy()), atol=1e-12)

    def test_swing_events_separate_unloading_and_boundaries(self):
        loads = np.array([0, 0, 12, 5, 8, 9, 12, 5, 12, 5, 0.])
        height = np.array([.03, .02, 0, .01, .04, .03, 0, .001, 0, .03, .04])
        position = np.zeros((len(loads), 3))
        position[:, 0] = np.arange(len(loads)) * .01
        events = swing_events(loads, height, position, .02)
        self.assertEqual([e["completed"] for e in events], [False, True, True, False])
        self.assertEqual([e["geometric_lift"] for e in events], [True, True, False, True])
        self.assertAlmostEqual(events[1]["duration_s"], .06)
        self.assertAlmostEqual(events[1]["horizontal_displacement_body_m"], .04)
        self.assertEqual(events[1]["peak_clearance_m"], .04)
        data = np.tile([.5, .2, .03, .01, -.02, .4], (50, 1))
        stats = velocity_window_metrics(data, [.5, .1, .4])
        np.testing.assert_allclose(stats["rmse_vx_vy_yaw"], [0, .1, 0], atol=1e-12)
        self.assertEqual(len(stats["std_six_axes"]), 6)

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
