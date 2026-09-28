"""Focused CPU checks and two separately selected, bounded GPU preparation checks."""

import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import root
from scipy.spatial.transform import Rotation
import torch
import yaml

from tests import test_go2w_contract as contract
from robot_gym.envs.go2w.go2w_config import apply_go2w_profile, check_target_intervals
from robot_gym.envs.go2w.step_events import WheelStepEvents, step_demand, lateral_high, project_log_std
from robot_gym.utils import get_args, task_registry
from robot_gym.utils.helpers import class_to_dict
from robot_gym.utils.diagnostics import (
    cylinder_clearance, wheel_cylinders, rotate_wxyz, urdf_link_poses,
    check_reference_contract, write_json, sha256,
)

AUDIT = Path(__file__).resolve().parents[1] / ".migration-audit/event-step-20260927"
URDF = contract.URDF


def config():
    e, t = task_registry.get_cfgs("go2w")
    apply_go2w_profile(e, t, "event_step_v1")
    return e, t


class EventFixture:
    def __init__(self, dt=.02, legs=(0,), gate_command=(0, .05, 0)):
        self.dt, self.legs = dt, list(legs)
        self.cfg = config()[0]
        self.geometry = wheel_cylinders(URDF, self.cfg.asset.foot_link_names)
        self.tracker = WheelStepEvents(1, "cpu", self.cfg.rewards.event_step, self.geometry)
        self.base = torch.tensor([[0., 0., .415]])
        self.quat = torch.tensor([[1., 0., 0., 0.]])
        self.wheels = torch.tensor([[[.2, .15, .086], [.2, -.15, .086], [-.2, .15, .086], [-.2, -.15, .086]]])
        self.wheel_quat = self.quat[:, None].repeat(1, 4, 1)
        self.commands = torch.tensor([gate_command], dtype=torch.float32)
        self.total = 0.
        self.valid = 0
        self.payments = []

    def sample(self, load=30., height=0., move=.0, heave=0., failed=False, spin=0., rotation=None):
        pos = self.wheels.clone()
        pos[:, self.legs, 2] += height
        pos[:, self.legs, 1] += move
        pos[..., 2] += heave
        base = self.base.clone(); base[:, 2] += heave
        q = self.quat.clone()
        wq = torch.tensor([math.cos(spin/2), 0., math.sin(spin/2), 0.]).expand_as(self.wheel_quat).clone()
        if rotation is not None:
            from genesis.utils.geom import transform_quat_by_quat
            q = torch.tensor(Rotation.from_euler("xyz", rotation).as_quat(scalar_first=True), dtype=torch.float32)[None]
            pos = base[:, None] + rotate_wxyz(q[:, None], pos - base[:, None])
            wq = transform_quat_by_quat(wq, q[:, None].expand_as(wq))
        loads = torch.full((1, 4), 30.)
        loads[:, self.legs] = load
        self.tracker.update(self.dt, self.commands, base, q, pos, wq, loads, torch.tensor([failed]))
        payment = float(.15 * self.tracker.payment.sum())
        self.total += payment
        self.valid += int(self.tracker.valid.sum())
        if payment:
            self.payments.append(payment)
        return payment

    def cycle(self, apex=.05, distance=.04, heave=False, back=False, fail=False, switch=False):
        for k in range(round(1.2/self.dt)):
            time = k*self.dt
            swing = .4 <= time < .8 - 1e-8
            phase = min(1., max(0., (time-.4)/.4))
            height = apex * math.sin(math.pi*phase)**2
            move = distance * (math.sin(math.pi*phase)**2 if back else phase)
            if switch and time >= .6:
                self.commands[:] = 0
            self.sample(0. if swing else 30., height=0. if heave else height,
                        move=0. if heave else move, heave=height if heave else 0.,
                        failed=fail and .6 <= time < .6+self.dt)
        return self.total


def solve_leg(side, delta, cfg=None):
    """Full URDF FK with test-only three-joint IK; cylinder height includes camber."""
    cfg = cfg or config()[0]
    nominal = cfg.init_state.default_joint_angles
    names = [f"{side}_{j}_joint" for j in ("hip", "thigh", "calf")]
    wheel = f"{side}_foot"
    geometry = tuple(g.double() for g in wheel_cylinders(URDF, [wheel]))

    def position(angles):
        pose = urdf_link_poses(URDF, dict(nominal, **dict(zip(names, angles))))[wheel]
        pos = torch.tensor(pose[:3, 3], dtype=torch.float64)[None]
        quat = torch.tensor(Rotation.from_matrix(pose[:3, :3]).as_quat(scalar_first=True))[None]
        h = float(cylinder_clearance(pos, quat, *geometry))
        return np.array([pose[0, 3], pose[1, 3], h])

    original = np.array([nominal[n] for n in names])
    target = position(original) + delta
    result = root(lambda x: position(x)-target, original, tol=1e-9)
    if not result.success and np.max(np.abs(position(result.x)-target)) > 1e-7:
        raise AssertionError(result.message)
    return names, result.x, position(result.x)-target


def cycle_table():
    """One 1.2 s cycle, perfect equal tracking, nominal trunk except explicit bob."""
    rows = []
    for legs in (1, 2):
        for name, apex, hip, sagittal, bob in (("shuffle", .010, .12, .05, False),
                                               ("step_5cm", .050, .04, .20, False),
                                               ("rigid_bob", .050, 0., 0., True)):
            f = EventFixture(legs=tuple(range(legs)))
            event = f.cycle(apex=apex, heave=bob)
            # Pose errors are sinusoidal over the 0.4 s swing; other legs stay nominal.
            row = {"case": name, "legs": legs, "linear_tracking": 1.2, "yaw_tracking": .96,
                   "hip_pose": -hip**2 * .4/2 * legs/4,
                   "sagittal_pose": -.12 * sagittal**2 * .4/2 * legs/4,
                   "leg_motion": 0., "wheel_air": 0., "prolonged_unloading": 0.,
                   "insufficient_support": 0., "vertical_velocity": -(.05*math.pi/.4)**2*.4/2 if bob else 0.,
                   "roll_pitch_rate": 0., "orientation": 0.,
                   "height": -8*.05**2*.4*3/8 if bob else 0., "stand": 0.,
                   "old_clearance": 0., "old_default_pose": 0., "event": event}
            row["affected_total"] = sum(v for k,v in row.items() if k not in ("case", "legs"))
            rows.append(row)
    return rows


class EventStepCPU(unittest.TestCase):
    def test_gate_and_yaw(self):
        commands = torch.tensor([[1.,0,0], [-.3,0,0], [0,0,.03], [1.1,.05,0],
                                 [0,.03,0], [0,0,.25], [0,0,-.175], [0,-.01,-.1]])
        torch.testing.assert_close(step_demand(commands), torch.tensor([0.,0,0,1,.5,1,.5,0]))
        e = contract.ContractTests().make_env(8, "event_step_v1")
        e.commands = commands
        e.base_ang_vel = torch.zeros(8, 3)
        expected = .25*torch.exp(-commands[:,2]**2/.25)+.75*torch.exp(-commands[:,2]**2/.04)
        torch.testing.assert_close(e._reward_tracking_ang_vel(), expected)

    def test_light_screen_keeps_physical_and_qualified_counts_separate(self):
        from tests.test_inference import InferenceTests
        from robot_gym.scripts import diagnostic_bank as bank
        old=bank.precision_schedule(); new=bank.precision_schedule(event_step=True)
        self.assertEqual({k:new[k] for k in old},old)
        self.assertEqual(len(new),7)
        metric=bank.precision_metrics
        def with_events(data,*args):
            for key in ("valid","censored_count","peak_actual","peak_use","duration","reposition","quality","payment","gate"):
                data["event_"+key]=np.zeros((5,2,4),dtype=np.float32)
            data["event_valid"][2,1,0]=1
            data["event_peak_actual"][2,1,0]=.05
            result=metric(data,*args)
            wheel=result[1]["swings"]["FL"]
            self.assertEqual(wheel["command_qualified"]["count"],1)
            self.assertIsNone(wheel["fraction_ge_5cm"])
            self.assertIn("fraction_ge_2_3_4cm",wheel)
            return result
        with patch.object(bank,"precision_metrics",with_events):
            InferenceTests().test_precision_schedules_and_terminal_alignment()

    def test_profile_contract_and_named_joints(self):
        original = tuple(map(class_to_dict, task_registry.get_cfgs("go2w")))
        e,t = config()
        self.assertEqual(tuple(map(class_to_dict, task_registry.get_cfgs("go2w"))), original)
        self.assertEqual((e.env.num_observations, e.env.num_actions), (56,16))
        self.assertEqual(t.runner.num_steps_per_env, 64)
        self.assertEqual(t.algorithm.gamma, .995)
        self.assertEqual(t.actor.distribution_cfg["std_range"], [.1,.7])
        e = contract.ContractTests().make_env(2, "event_step_v1")
        e.joint_names.reverse(); e._build_control_tensors()
        self.assertEqual({e.joint_names[i] for i in e.hip_indices}, {f"{s}_hip_joint" for s in ("FL","FR","RL","RR")})
        self.assertEqual(len(e.sagittal_indices), 8)
        e.default_dof_pos = torch.tensor([[e.cfg.init_state.default_joint_angles[n] for n in e.joint_names]]).repeat(2,1)
        e.dof_pos = e.default_dof_pos + .1
        e.dof_vel = torch.ones_like(e.dof_pos)
        e.commands[:,1] = .05
        torch.testing.assert_close(e._reward_hip_pose(), torch.full((2,), .005))
        torch.testing.assert_close(e._reward_sagittal_pose(), torch.full((2,), .002))
        self.assertEqual(float(e._reward_leg_motion().sum()), 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"config.yaml"
            path.write_text(yaml.safe_dump(json.loads(json.dumps(dict(env_cfg=original[0],train_cfg=original[1])))))
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                check_reference_contract(path, class_to_dict(e.cfg), class_to_dict(t))

    def test_legacy_sampler_exact_rng(self):
        # Execute only the reviewed sampler function against the same fixture.
        source = subprocess.check_output(["git", "show", "2479ffd:robot_gym/envs/go2w/go2w_env.py"], text=True, stderr=subprocess.PIPE)
        old = source[source.index("    def _resample_commands"):source.index("    def compute_observations")]
        import textwrap
        namespace = {"torch": torch}
        exec(textwrap.dedent(old), namespace)
        for profile, finetune in ((None,None), ("step_recovery_v1",None), ("step_recovery_v1","coverage"),
                                  ("step_recovery_v1","coverage_mobility"), ("step_recovery_v1","precision_clearance")):
            a = contract.ContractTests().make_env(200, profile)
            if finetune:
                from robot_gym.envs.go2w.go2w_config import apply_go2w_finetune
                apply_go2w_finetune(a.cfg, None, finetune)
            b = copy.deepcopy(a)
            ids = torch.arange(200)
            torch.manual_seed(123); a._resample_commands(ids); rng = torch.get_rng_state()
            torch.manual_seed(123); namespace["_resample_commands"](b, ids)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            torch.testing.assert_close(a.commands, b.commands, rtol=0, atol=0)
            torch.testing.assert_close(a.command_steps_left,b.command_steps_left,rtol=0,atol=0)

    def test_sampler_curriculum(self):
        e = contract.ContractTests().make_env(20000, "event_step_v1")
        e.command_ranges = {n:getattr(e.cfg.commands.ranges,n) for n in ("lin_vel_x","lin_vel_y","ang_vel_yaw")}
        e.command_mixture[:] = 0; e.command_mixture[5] = 1
        ids = torch.arange(e.num_envs)
        for update, high in ((0,.3),(500,.3),(1000,.4),(1500,.5),(2000,.5)):
            e.completed_updates = update
            self.assertAlmostEqual(lateral_high(update,e.cfg.commands), high)
            torch.manual_seed(1); e._resample_commands(ids)
            v = e.commands[:,1]
            self.assertTrue(((v.abs() >= .03) & (v.abs() <= high+1e-6)).all())
            self.assertLess(abs(float((v>0).float().mean())-.5), .02)
            self.assertLess(abs(float((v.abs()>=.3-1e-7).float().mean())-.2), .02)
        e.command_mixture[:] = 0; e.command_mixture[6] = 1
        e._resample_commands(ids)
        self.assertLessEqual(float(e.commands[:,1].abs().max()), .3)
        self.assertLess(abs(float((e.commands[:,2]==0).float().mean())-.5), .02)
        e._fixed_command = torch.tensor([0.,.05,0.])
        before = torch.get_rng_state(); e._resample_commands(ids)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        torch.testing.assert_close(e.commands, e._fixed_command.expand_as(e.commands))

    def test_valid_completed_and_simultaneous(self):
        one, two = EventFixture(), EventFixture(legs=(0,2))
        self.assertGreater(one.cycle(), 0)
        self.assertAlmostEqual(two.cycle(), 2*one.total)
        self.assertEqual((one.valid,two.valid), (1,2))
        self.assertEqual(len(one.payments),1)
        self.assertAlmostEqual(float(one.tracker.duration[0,0]), .4, places=6)
        self.assertAlmostEqual(float(one.tracker.reposition[0,0]), .04, places=6)

    def test_counterexamples(self):
        for kwargs in ({"apex":0.}, {"distance":0.}, {"back":True}, {"heave":True}, {"fail":True}, {"switch":True}):
            f = EventFixture(); self.assertEqual(f.cycle(**kwargs), 0., kwargs)
        for command in ((0,0,0),(1,0,0),(0,0,.03)):
            self.assertEqual(EventFixture(gate_command=command).cycle(), 0.)
        self.assertEqual(EventFixture(legs=(0,1,2)).cycle(),0.)

    def test_failed_touchdown_nonfinite_and_settling(self):
        for failure in ("terminal", "nonfinite"):
            f = EventFixture()
            for _ in range(20): f.sample()
            f.sample(load=0.)
            for _ in range(19): f.sample(load=0.,height=.05,move=.04)
            f.sample(move=.04); f.sample(move=.04)
            if failure == "nonfinite": f.wheels[0,1,0] = float("nan")
            f.sample(move=.04,failed=failure == "terminal")
            self.assertEqual(f.total,0.)
            self.assertEqual(float(f.tracker.credit.sum()),0.)
        f = EventFixture()
        for _ in range(15): f.sample(load=0.,height=.05,move=.04)
        for _ in range(3): f.sample(move=.04)
        self.assertEqual(f.total,0.)

    def test_unknown_initial_support_needs_reload(self):
        f=EventFixture()
        for _ in range(20): f.sample(load=8.)
        self.assertFalse(f.tracker.supported[0,0])
        f.sample(load=6.)
        self.assertFalse(f.tracker.active[0,0])
        for _ in range(20): f.sample(load=10.)
        f.sample(load=6.)
        self.assertTrue(f.tracker.active[0,0])

    def test_graded_apex_and_overshoot(self):
        values = [EventFixture().cycle(apex=h) for h in (.005,.009,.025,.05,.10)]
        self.assertEqual(values[0],0.)
        self.assertTrue(0 < values[1] < values[2] < values[3])
        self.assertLess(values[4],values[3])
        half=EventFixture(gate_command=(0,.03,0))
        self.assertAlmostEqual(half.cycle(apex=.0375)*2,values[3],places=6)

    def test_family_and_duration_coverage(self):
        e=contract.ContractTests().make_env(30000,"event_step_v1")
        e.command_ranges=class_to_dict(e.cfg.commands.ranges)
        e.diagnostic_command_families=torch.full((e.num_envs,),-1,dtype=torch.long)
        torch.manual_seed(21); e._resample_commands(torch.arange(e.num_envs))
        families=e.diagnostic_command_families; duration=e.command_steps_left*e.dt
        torch.testing.assert_close(torch.bincount(families,minlength=7).float()/e.num_envs,e.command_mixture,atol=.01,rtol=0)
        moving=families!=0
        for measured,expected in (((duration[moving]<=1).float().mean(),.70),
                                  (((duration[moving]>=1.5)&(duration[moving]<=3)).float().mean(),.25),
                                  ((duration[moving]>=8).float().mean(),.05)):
            self.assertLess(abs(float(measured)-expected),.015)
        self.assertTrue((e.commands[~moving]==0).all())
        self.assertLess(float(e.commands[:,0].min()),-.34)
        self.assertGreater(float(e.commands[:,0].max()),1.09)

    def test_flicker_hover_crossings_prolonged(self):
        f = EventFixture()
        for _ in range(20): f.sample()
        for _ in range(10):
            f.sample(load=0.,height=.05,move=.04); f.sample(load=30.)
        self.assertEqual(f.total,0.)
        for _ in range(20): f.sample()
        for i in range(20):
            f.sample(load=0.,height=0. if i==0 else .05 if i%2 else .02,move=.04*i/20)
            self.assertEqual(f.total,0.)
        for _ in range(3): f.sample(move=.04)
        self.assertEqual(f.valid,1)
        self.assertEqual(len(f.payments),1)
        g = EventFixture()
        for _ in range(20): g.sample()
        for _ in range(50): g.sample(load=0.,height=.05,move=.04)
        self.assertGreater(float(g.tracker.unloaded_time[0,0]),.9)
        for _ in range(4): g.sample(move=.04)
        self.assertEqual(g.total,0.)

    def test_hysteresis_onset_reload_and_command_censor(self):
        f = EventFixture()
        for _ in range(20): f.sample()
        f.sample(load=6.)
        start = f.tracker.takeoff.clone()
        f.sample(load=8.,height=.025,move=.02)
        self.assertTrue(f.tracker.confirmed[0,0])
        torch.testing.assert_close(start,f.tracker.takeoff)
        for _ in range(8): f.sample(load=0.,height=.05,move=.04)
        f.sample(load=10.,move=.04)
        f.sample(load=8.,move=.10)
        f.sample(load=8.,move=.20)
        self.assertEqual(f.valid,1)
        self.assertAlmostEqual(float(f.tracker.reposition[0,0]),.04,places=6)
        g = EventFixture()
        for _ in range(20): g.sample()
        for _ in range(4): g.sample(load=0.,height=.03)
        g.tracker.command_changed(g.commands.clone())
        self.assertTrue(g.tracker.active[0,0])
        g.commands[:,1] *= -1
        g.tracker.command_changed(g.commands)
        self.assertEqual(int(g.tracker.censored_count.sum()),1)
        self.assertEqual(float(g.tracker.credit.sum()),0)

    def test_frozen_frame_and_spin(self):
        f = EventFixture()
        for _ in range(20): f.sample()
        f.sample(load=0.)
        for k in range(1,12):
            f.sample(load=0.,heave=.05,rotation=(.1,-.1,.05),spin=k*.8)
        self.assertLess(float(f.tracker.peak_use.max()),1e-6)
        f.sample(); f.sample(); f.sample()
        self.assertEqual(f.total,0.)
        self.assertLess(float(f.tracker.reposition.max()),1e-6)

    def test_mirrored_events(self):
        left, right = EventFixture(legs=(0,2)), EventFixture(legs=(1,3),gate_command=(0,-.05,0))
        self.assertAlmostEqual(left.cycle(), right.cycle(distance=-.04), places=6)

    def test_time_budget_dt_and_frequency(self):
        values = [EventFixture(dt=dt).cycle() for dt in (.01,.02,.04)]
        self.assertLess(max(values)-min(values), .15*.04+1e-6)
        for cycles in (1,2,4):
            f = EventFixture()
            for _ in range(cycles): f.cycle()
            self.assertLessEqual(f.total,.15*cycles*1.2+1e-6)
        self.assertEqual(len(EventFixture().payments),0)
        # Same credited duration, split in any number of unit-quality completions.
        for count in (1,2,4):
            self.assertAlmostEqual(sum(.15*.8/count for _ in range(count)),.12)

    def test_reset_and_pure_reward_reads(self):
        f = EventFixture()
        for _ in range(20): f.sample()
        for _ in range(5): f.sample(load=0.,height=.03)
        e = contract.ContractTests().make_env(1,"event_step_v1"); e.step_events = f.tracker
        before = {k:v.clone() for k,v in vars(f.tracker).items() if torch.is_tensor(v)}
        e._reward_step_event(); e._reward_step_event(); e._reward_prolonged_unloading()
        for k,v in before.items(): torch.testing.assert_close(getattr(f.tracker,k),v)
        f.tracker.reset(torch.tensor([0]))
        self.assertEqual(float(f.tracker.credit.sum()),0.)
        self.assertFalse(f.tracker.active.any())
        self.assertEqual(float(f.tracker.support_time.sum()),0.)

    def test_discrete_accumulator(self):
        e = contract.ContractTests().make_env(1,"event_step_v1")
        e.cfg.rewards.only_positive_rewards = True
        e.reward_scales = {"step_event":.15,"lin_vel_z":-1.,"termination":-10.}
        e._prepare_reward_function()
        self.assertEqual(e.reward_scales,{"step_event":.15,"lin_vel_z":-.02,"termination":-.2})
        e.step_events = EventFixture().tracker; e.step_events.payment[0,0] = .8
        e.base_lin_vel = torch.zeros(1,3); e.rew_buf = torch.zeros(1)
        e.reset_buf = torch.tensor([False]); e.time_out_buf = torch.tensor([False]); e.common_step_counter=1
        e.compute_reward(); saved = {k:v.clone() for k,v in e.episode_sums.items()}
        self.assertAlmostEqual(float(e.rew_buf),.12,places=6)
        e.compute_reward()
        for k,v in saved.items(): torch.testing.assert_close(e.episode_sums[k],v)

    def test_std_projection_finite_and_boundary_recovery(self):
        from rsl_rl.modules.distribution import GaussianDistribution
        d = GaussianDistribution(16,init_std=.4,std_type="log",std_range=(.1,.7))
        optimizer = torch.optim.Adam(d.parameters(),lr=.1)
        project_log_std(optimizer,d)
        with torch.no_grad(): d.log_std_param.fill_(math.log(.1))
        for _ in range(4):
            optimizer.zero_grad(); d.update(torch.zeros(2,16))
            loss = -d.entropy.mean(); loss.backward()
            self.assertTrue(torch.isfinite(d.log_std_param.grad).all()); optimizer.step()
        self.assertTrue((d.log_std_param > math.log(.1)).all())
        with torch.no_grad(): d.log_std_param.fill_(math.log(.7))
        optimizer.zero_grad(); (-d.log_std_param.sum()).backward(); optimizer.step()
        self.assertTrue((d.log_std_param <= math.log(.7)).all())
        self.assertTrue(torch.isfinite(d.log_std_param).all())
        self.assertEqual(int(optimizer.state[d.log_std_param]["step"]),5)

    def test_cycle_ranking_and_kinematics(self):
        AUDIT.mkdir(parents=True,exist_ok=True)
        table = cycle_table()
        for rows in (table[:3],table[3:]):
            self.assertGreater(rows[1]["affected_total"],rows[0]["affected_total"])
            self.assertGreater(rows[0]["affected_total"],rows[2]["affected_total"])
        cfg,_ = config(); intervals = check_target_intervals(cfg)
        report = []
        import xml.etree.ElementTree as ET
        tree = ET.parse(URDF).getroot()
        for side in ("FL","FR","RL","RR"):
            for height in (.05,.06,.08,.10):
                names, angles, error = solve_leg(side,np.array([0.,0.,height]),cfg)
                offsets = angles-np.array([cfg.init_state.default_joint_angles[n] for n in names])
                target_margin = min(min(a-intervals[n][0],intervals[n][1]-a) for n,a in zip(names,angles))
                hard_margin = min(min(a-float(tree.find(f"joint[@name='{n}']/limit").get("lower")),
                                      float(tree.find(f"joint[@name='{n}']/limit").get("upper"))-a) for n,a in zip(names,angles))
                report.append(dict(leg=side,height=height,offsets=offsets,target_margin=target_margin,hard_margin=hard_margin,error=error))
                if height == .05: self.assertGreater(target_margin,0.)
            # Inward full displacement at the apex is a separate, restrictive probe.
            names, angles, error = solve_leg(side,np.array([0.,-.02 if side.endswith("L") else .02,.05]),cfg)
            report.append(dict(leg=side,height=.05,inward_at_apex=True,
                               target_margin=min(min(a-intervals[n][0],intervals[n][1]-a) for n,a in zip(names,angles))))
            previous = np.array([cfg.init_state.default_joint_angles[f"{side}_{j}_joint"] for j in ("hip","thigh","calf")])
            for s in np.linspace(0,1,41):
                smooth = 3*s*s-2*s*s*s
                names, angles, error = solve_leg(side,np.array([0.,.02*smooth,.05*math.sin(math.pi*s)**2]),cfg)
                self.assertLess(np.abs(angles-previous).max(),.04)
                for n,a in zip(names,angles): self.assertTrue(intervals[n][0]<=a<=intervals[n][1])
                previous=angles
        write_json(AUDIT/"cpu_preflight.json",dict(cycle_table=table,kinematics=report,
                   reference_hashes={str(p):sha256(p) for p in [URDF,
                   Path("logs/go2w_step_recovery_v1/coverage_seed1_20260926_231000_2026-09-26_23-15-39/model_1798.pt"),
                   Path("logs/go2w_step_recovery_v1/coverage_seed1_20260926_231000_2026-09-26_23-15-39/config.yaml")]}))


class EventRefactorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Compare against the reviewed implementation, without keeping a second tracker.
        cls.original = {}
        for name, path in (("events", "robot_gym/envs/go2w/step_events.py"),
                           ("diagnostics", "robot_gym/utils/training_diagnostics.py")):
            source = subprocess.run(
                ["git", "show", f"4733792200e10d41660e786f78bb7296bc614405:{path}"],
                cwd=URDF.parents[4], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, check=True,
            ).stdout
            namespace = {}
            exec(compile(source, path, "exec"), namespace)
            cls.original[name] = namespace

    def assert_state_equal(self, old, new):
        self.assertEqual(vars(old).keys(), vars(new).keys())
        for name, value in vars(old).items():
            if torch.is_tensor(value):
                torch.testing.assert_close(getattr(new, name), value, rtol=0, atol=0, equal_nan=True)
        self.assertEqual(old.update_count, new.update_count)

    def test_tracker_masks_cache_and_edges(self):
        f = EventFixture()
        old = self.original["events"]["WheelStepEvents"](4, "cpu", f.tracker.cfg, f.geometry)
        new = WheelStepEvents(4, "cpu", f.tracker.cfg, f.geometry)
        for i in range(160):
            command = f.commands.repeat(4, 1)
            if 55 <= i < 75:
                command[1] = 0
            if i >= 110:
                command[2, 1] *= -1
            base, quat = f.base.repeat(4, 1), f.quat.repeat(4, 1)
            pos, wq = f.wheels.repeat(4, 1, 1), f.wheel_quat.repeat(4, 1, 1)
            loads = torch.full((4, 4), 30.)
            phase = i % 50
            if 20 <= phase < 40:
                loads[:, 0] = 0
                pos[:, 0, 2] += .05 * math.sin(math.pi * (phase-20)/20)**2
                pos[:, 0, 1] += .04 * (phase-20)/20
                loads[2, 1:3] = 0  # Support loss; independent simultaneous attempts.
            elif phase >= 40:
                pos[:, 0, 1] += .04
            if phase == 10:
                loads[3, 0] = 0  # Flicker before unloading dwell.
            if phase == 25:
                loads[1, 0] = 8  # Hysteresis retains unloading.
            failed = torch.tensor([False, i == 88, False, False])
            if i == 87:
                base[3, 2] = float("nan")
            actual = cylinder_clearance(pos, wq, *f.geometry)
            old.update(.02, command, base, quat, pos, wq, loads, failed)
            new.update(.02, command, base, quat, pos, wq, loads, failed, actual_clearance=actual)
            self.assert_state_equal(old, new)
            if i in (51, 90, 140):
                old.command_changed(command); new.command_changed(command)
                self.assert_state_equal(old, new)
                ids = torch.tensor([1, 3]) if i != 51 else torch.empty(0, dtype=torch.long)
                old.reset(ids); new.reset(ids)
                self.assert_state_equal(old, new)

    def test_reward_cache_hook_and_named_indices(self):
        e = contract.ContractTests().make_env(4, "event_step_v1")
        from types import SimpleNamespace
        from unittest.mock import Mock
        e.step_events = SimpleNamespace(update=Mock())
        for name, shape in (("dof_pos", (4,16)), ("dof_vel", (4,16)), ("actions", (4,16)),
                            ("torques", (4,16)), ("base_lin_vel", (4,3)), ("base_ang_vel", (4,3)),
                            ("base_pos", (4,3)), ("base_quat", (4,4)), ("foot_pos", (4,4,3)),
                            ("wheel_link_quat", (4,4,4)), ("wheel_normal_force", (4,4)),
                            ("wheel_clearance", (4,4)), ("reset_buf", (4,)), ("nonfoot_contact_count", (4,))):
            setattr(e, name, torch.zeros(shape))
        e._update_step_events()
        self.assertIs(e.step_events.update.call_args.kwargs["actual_clearance"], e.wheel_clearance)
        self.assertEqual(e.step_events.update.call_count, 1)
        self.assertEqual(e.hip_indices.device, e.commands.device)
        self.assertEqual(e.sagittal_indices.tolist(), [1,2,5,6,9,10,13,14])
        e.reset_idx(torch.empty(0, dtype=torch.long))

    def test_diagnostic_masks_preserve_counts_and_sums(self):
        from robot_gym.utils.training_diagnostics import TrainingDiagnostics
        env = contract.ContractTests().make_env(17, "event_step_v1")
        pair = []
        for cls in (self.original["diagnostics"]["TrainingDiagnostics"], TrainingDiagnostics):
            d = cls.__new__(cls)
            d.env = env
            d.command_families = torch.arange(17) % 7
            d.previous_mean = d.previous_sample = None
            d.previous_valid = torch.ones(17, dtype=torch.bool)
            d.reset_aggregates()
            pair.append(d)
        values = torch.linspace(-1.7, 1.7, 17*16).reshape(17, 16)
        for i in range(8):
            for d in pair:
                d.previous_valid = (torch.arange(17) % 3 != 0) if i != 3 else torch.zeros(17, dtype=torch.bool)
                d.record_actions(values * (1 + i*.01), values * .9)
                d.reward(torch.linspace(-.1, .2, 17), env.commands)
            for name, (total, count) in pair[0].rollout.items():
                actual, actual_count = pair[1].rollout[name]
                self.assertEqual(int(actual_count), int(count))
                torch.testing.assert_close(actual, total, rtol=5e-7, atol=1e-5)
            for name, expected in pair[0].rewards.items():
                torch.testing.assert_close(pair[1].rewards[name], expected, rtol=5e-7, atol=1e-6)


def gpu_args(count, smoke=False):
    argv = ["event-preparation", "--task", "go2w", "--go2w_profile", "event_step_v1",
            "--num_envs", str(count), "--headless", "--seed", "1", "--logger", "tensorboard"]
    if smoke:
        argv += ["--max_iterations", "2", "--training_diagnostics", "--run_name",
                 "event_step_smoke_seed1_20260927", "--experiment_name", "go2w_event_step_v1_smoke"]
    with patch.object(sys,"argv",argv):
        return get_args()


def destroy_genesis():
    import genesis as gs
    from robot_gym.envs.base.base_task import BaseTask
    gs.destroy()
    BaseTask._gs_initialized = False
    BaseTask._gs_backend = None


@unittest.skipUnless(os.environ.get("GO2W_EVENT_GPU") == "feasibility", "one bounded physical process")
class EventStepFeasibility(unittest.TestCase):
    def test_four_nominal_limbs(self):
        from robot_gym.utils.diagnostics import loaded_properties
        cfg,_ = config()
        cfg.env.num_envs = 1
        cfg.env.capture_transitions = cfg.env.capture_closed_loop = cfg.env.capture_precision = True
        cfg.noise.add_noise = False
        for name in ("randomize_friction","randomize_base_mass","randomize_com","randomize_kp",
                     "randomize_kd","randomize_action_delay","push_robots"):
            setattr(cfg.domain_rand,name,False)
        cfg.init_state.joint_position_noise = cfg.init_state.joint_velocity_noise = 0.
        cfg.init_state.orientation_noise = (0.,0.,0.)
        cfg.init_state.linear_velocity_noise = cfg.init_state.angular_velocity_noise = 0.
        # Targets only: 1 s settle, 0.8 s load transfer, 0.5 s swing, return and settle.
        paths = {}
        for side in ("FL","FR","RL","RR"):
            actions=[]
            sx = -.025 if side.startswith("F") else .025
            sy = -.030 if side.endswith("L") else .030
            for k in range(250):
                time=k*.02
                u=np.clip((time-1.)/.8,0,1)
                down=np.clip((time-2.5)/.8,0,1)
                weight=(3*u*u-2*u**3)*(1-(3*down*down-2*down**3))
                swing=np.clip((time-1.8)/.5,0,1)
                reset=np.clip((time-3.3)/.7,0,1)
                shift=np.array([sx,sy,0.])*weight
                angles=dict(cfg.init_state.default_joint_angles)
                for limb in ("FL","FR","RL","RR"):
                    delta=-shift.copy()
                    if limb==side:
                        delta[1] += (.02 if side.endswith("L") else -.02)*(3*swing*swing-2*swing**3)*(1-(3*reset*reset-2*reset**3))
                        delta[2] += .05*math.sin(math.pi*swing)**2
                    names,values,_=solve_leg(limb,delta,cfg)
                    angles.update(dict(zip(names,values)))
                action=np.array([(angles[n]-cfg.init_state.default_joint_angles[n])/cfg.control.action_scale[n]
                                 for n in cfg.init_state.default_joint_angles])
                self.assertLessEqual(np.abs(action).max(),1.,(side,time,action))
                actions.append(action)
            paths[side]=np.array(actions)
        report={"nominal":True,"free_base":True,"gravity_contacts_limits_retained":True,
                "setup":"Support targets shift the trunk 25 mm away longitudinally and 30 mm laterally; no external hold or root teleport during trials",
                "trials":[],"total_simulated_s":0.}
        try:
            env,_=task_registry.make_env("go2w",args=gpu_args(1),env_cfg=cfg)
            report["properties"]=loaded_properties(env)
            self.assertFalse(hasattr(env,"physics_diagnostics"))
            env.set_fixed_command((0,.05,0))
            original_step=env.sim.step
            maxima=torch.zeros_like(env.torques)
            def substep(*args,**kwargs):
                result=original_step(*args,**kwargs)
                maxima.copy_(torch.maximum(maxima,env.robot.get_dofs_control_force(env.joint_dof_idx).abs()))
                return result
            env.sim.step=substep
            report["total_simulated_s"] += env.dt
            for side, path in paths.items():
                env.reset(); report["total_simulated_s"] += env.dt
                data=[]
                for action in path:
                    maxima.zero_()
                    env.step(torch.tensor(action,dtype=torch.float32,device=env.device)[None])
                    s=env.transition_state
                    data.append({key:s[key].cpu().numpy().copy() for key in
                                 ("base_pos","rpy","dof_pos","wheel_clearance","wheel_normal_force","reset_buf","nonfoot_contact_count")}
                                | {"target":(env.default_dof_pos+env.action_scale*torch.tensor(action,device=env.device)).cpu().numpy(),
                                   "substep_force_max":maxima.cpu().numpy().copy()})
                    report["total_simulated_s"]+=env.dt
                    if bool(s["reset_buf"].any()): break
                d={key:np.concatenate([r[key] for r in data],axis=0) for key in data[0]}
                np.savez_compressed(AUDIT/f"feasibility_{side}.npz",**d,dt=env.dt)
                index=env.cfg.asset.foot_link_names.index(f"{side}_foot")
                ids=[env.joint_names.index(f"{side}_{j}_joint") for j in ("hip","thigh","calf")]
                row={"leg":side,"steps":len(data),"failed":bool(d["reset_buf"].any()),
                     "apex_m":float(d["wheel_clearance"][:,index].max()),
                     "loaded_after_swing":bool((d["wheel_normal_force"][min(120,len(data)-1):,index]>=10).any()),
                     "swing_min_load_N":float(d["wheel_normal_force"][min(90,len(data)-1):min(116,len(data)),index].min()),
                     "tracking_rms_rad":np.sqrt(np.mean((d["dof_pos"][:,ids]-d["target"][:,ids])**2,axis=0)),
                     "max_roll_pitch_rad":np.abs(d["rpy"][:,:2]).max(0),
                     "base_z_range_m":[float(d["base_pos"][:,2].min()),float(d["base_pos"][:,2].max())],
                     "substep_max_force_ratio":float(np.max(d["substep_force_max"]/env.torque_limits.cpu().numpy())),
                     "substep_max_force_per_joint":d["substep_force_max"].max(0),
                     "minimum_loaded_count":int((d["wheel_normal_force"]>6).sum(1).min()),
                     "nonwheel_contact_steps":int((d["nonfoot_contact_count"]>0).sum())}
                report["trials"].append(row)
                print("LIMB TRIAL",side,row,flush=True)
            self.assertLessEqual(report["total_simulated_s"],24.)
            write_json(AUDIT/"feasibility.json",report)
        finally:
            destroy_genesis()


@unittest.skipUnless(os.environ.get("GO2W_EVENT_GPU") == "smoke", "one fresh two-update process")
class EventStepSmoke(unittest.TestCase):
    def test_two_fresh_updates(self):
        from robot_gym.scripts.train import train
        from robot_gym.utils.diagnostics import loaded_properties
        from robot_gym.utils.training_diagnostics import verify_resume_state, std_parameters
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        original=task_registry.make_alg_runner
        measurements={"optimizer_steps":0,"transitions":0}
        def create(*args,**kwargs):
            runner,cfg=original(*args,**kwargs)
            env=runner.env
            self.assertIsNone(runner.checkpoint_path)
            self.assertFalse(runner.alg.optimizer.state)
            self.assertEqual(runner.alg.learning_rate,3e-4)
            self.assertEqual(runner.alg.entropy_coef,.003)
            self.assertEqual(runner.cfg["num_steps_per_env"],64)
            self.assertEqual(env.completed_updates,0)
            torch.testing.assert_close(runner.alg.actor.distribution.log_std_param.exp(),torch.full((16,),.4,device=env.device))
            for model in (runner.alg.actor,runner.alg.critic):
                self.assertEqual(model.obs_normalizer.count,0)
                self.assertEqual(model.obs_normalizer._mean.count_nonzero(),0)
                torch.testing.assert_close(model.obs_normalizer._var,torch.ones_like(model.obs_normalizer._var))
            self.assertEqual(tuple(env.action_history.shape),(64,3,16))
            self.assertEqual(set(env.action_delay_steps.tolist()),{0,1,2})
            measurements["dr"]=loaded_properties(env)
            measurements["initial_delays"]=env.action_delay_steps.clone()
            read=measurements["dr"]
            nominal=json.loads((AUDIT/"feasibility.json").read_text())["properties"]
            base=env.base_link_idx
            mass=read["mass"][:,base]-torch.tensor(nominal["mass"],device=env.device).reshape(-1,env.robot.n_links)[0,base]
            com=read["com_local"][:,base]-torch.tensor(nominal["com_local"],device=env.device).reshape(-1,env.robot.n_links,3)[0,base]
            self.assertTrue(((mass>=-.5)&(mass<=1.5)).all())
            self.assertTrue((com.abs()<=.015001).all())
            measurements["mass_delta_range"]=[float(mass.min()),float(mass.max())]
            measurements["maximum_abs_com_delta"]=float(com.abs().max())
            self.assertTrue(((read["effective_friction"]>=.6)&(read["effective_friction"]<=1.2)).all())
            for key,base in (("kp",env.base_p_gains),("kv",env.base_d_gains)):
                ids=base!=0; factors=read[key][:,ids]/base[ids]
                self.assertTrue(((factors>=.85)&(factors<=1.15)).all())
                torch.testing.assert_close(factors,factors[:,:1].expand_as(factors))
            self.assertFalse(hasattr(env,"physics_diagnostics"))
            self.assertFalse(hasattr(env,"zero_command_brake"))
            step=env.step
            def checked_step(actions):
                count=env.step_events.update_count
                old=env.action_history.clone()
                delay=env.action_delay_steps.clone()
                expected=torch.cat((actions.clamp(-1,1)[:,None],old[:,:-1]),dim=1)[env.all_env_ids,delay]
                result=step(actions)
                self.assertEqual(env.step_events.update_count,count+1)
                self.assertTrue(torch.isfinite(result[1]).all())
                self.assertTrue(torch.isfinite(env.obs_buf).all())
                torch.testing.assert_close(env.applied_actions[~env.reset_buf],expected[~env.reset_buf])
                reward=env.rew_buf.clone(); sums={k:v.clone() for k,v in env.episode_sums.items()}
                env.get_observations(); env.compute_reward(); env._reward_step_event()
                torch.testing.assert_close(reward,env.rew_buf)
                for k,v in sums.items(): torch.testing.assert_close(env.episode_sums[k],v)
                self.assertEqual(env.step_events.update_count,count+1)
                measurements["transitions"]+=1
                return result
            env.step=checked_step
            def checked_optimizer(*_):
                for group in runner.alg.optimizer.param_groups:
                    for p in group["params"]:
                        self.assertTrue(torch.isfinite(p).all())
                        if p.grad is not None:self.assertTrue(torch.isfinite(p.grad).all())
                d=runner.alg.actor.distribution
                self.assertTrue(((d.log_std_param>=d.log_std_range[0])&(d.log_std_param<=d.log_std_range[1])).all())
                measurements["optimizer_steps"]+=1
            runner.alg.optimizer.register_step_post_hook(checked_optimizer)
            return runner,cfg
        try:
            with patch.object(task_registry,"make_alg_runner",create):
                runner=train(gpu_args(64,True))
            env=runner.env
            self.assertEqual((measurements["transitions"],measurements["optimizer_steps"]),(128,80))
            self.assertEqual(env.completed_updates,2)
            out=Path(runner.logger.log_dir); checkpoint=out/"model_1.pt"
            verify_resume_state(runner,checkpoint)
            env.completed_updates=0
            runner.load(checkpoint)
            measurements["native_reload"]=verify_resume_state(runner,checkpoint)
            self.assertEqual(env.completed_updates,2)
            self.assertIn(runner.event_std_hook.id,runner.alg.optimizer._optimizer_step_post_hooks)
            with self.assertRaisesRegex(ValueError,"historical checkpoints"):
                runner.load("logs/go2w_step_recovery_v1/coverage_seed1_20260926_231000_2026-09-26_23-15-39/model_1798.pt")
            rows=[json.loads(row) for row in (out/"diagnostics.jsonl").read_text().splitlines()]
            self.assertEqual([r["event_step_v1"]["completed_updates"] for r in rows],[1,2])
            events=EventAccumulator(str(out)).Reload()
            losses={k:[x.value for x in events.Scalars(k)] for k in events.Tags()["scalars"] if k.startswith("Loss/")}
            self.assertTrue(losses)
            self.assertTrue(all(len(v)==2 and np.isfinite(v).all() for v in losses.values()))
            before=loaded_properties(env); delays=env.action_delay_steps.clone()
            env.reset_idx(torch.tensor([0],device=env.device)); after=loaded_properties(env)
            for key in ("mass","com_local","inertia_local","wheel_friction_ratio"):
                torch.testing.assert_close(before[key],after[key],rtol=0,atol=0)
            for key in ("kp","kv"):torch.testing.assert_close(before[key][1:],after[key][1:],rtol=0,atol=0)
            torch.testing.assert_close(delays[1:],env.action_delay_steps[1:],rtol=0,atol=0)
            measurements.update(losses=losses,run=str(out),checkpoint_sha256=sha256(checkpoint),
                                std=std_parameters(runner.alg.actor.distribution),groups=rows[-1]["groups"],
                                completed_updates=env.completed_updates,lateral_high=lateral_high(env.completed_updates,env.cfg.commands))
            write_json(AUDIT/"smoke.json",measurements)
            print("EVENT STEP SMOKE PASS",out,flush=True)
        finally:
            destroy_genesis()


SAGITTAL_PARENT = Path(__file__).resolve().parents[1] / "logs/go2w_event_step_v1/event_step_v1_seed1_20260927_prepared_2026-09-28_10-30-59"
SAGITTAL_AUDIT = Path(__file__).resolve().parents[1] / ".migration-audit/sagittal-retention-20260928"


def sagittal_args():
    args = gpu_args(64)
    args.resume, args.max_iterations, args.training_diagnostics = True, 2, True
    args.load_run, args.checkpoint = str(SAGITTAL_PARENT), 1999
    args.run_name = "sagittal_retention_smoke_seed1_20260928"
    args.experiment_name = "go2w_event_step_v1"
    args.sagittal_stance_weight = 2.0
    return args


class SagittalContinuationCPU(unittest.TestCase):
    def test_weighted_rates_and_legacy_path(self):
        e = contract.ContractTests().make_env(4, "event_step_v1")
        e.default_dof_pos = torch.zeros(1, 16)
        e.dof_pos = torch.full((4, 16), .2)
        e.dof_pos[3] = 0
        e.commands[:, 1] = torch.tensor([0., .03, .05, .05])
        old = e._reward_sagittal_pose() * e.cfg.rewards.scales.sagittal_pose
        e.cfg.rewards.sagittal_stance_weight = 2.
        e.reward_scales = {"sagittal_pose": e.cfg.rewards.scales.sagittal_pose}
        e._prepare_reward_function()
        actual = e._reward_sagittal_pose() * e.reward_scales["sagittal_pose"]
        expected = -torch.tensor([2., 1.06, .12, .12]) * torch.tensor([.04,.04,.04,0.]) * e.dt
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-9)
        torch.testing.assert_close(actual[2:], old[2:] * e.dt, rtol=2e-6, atol=1e-9)
        self.assertTrue((actual[:3] < 0).all())
        del e.cfg.rewards.sagittal_stance_weight
        torch.testing.assert_close(e._reward_sagittal_pose() * -.6, old, rtol=0, atol=0)
        self.assertFalse(hasattr(config()[0].rewards, "sagittal_stance_weight"))

    def test_cli_and_saved_inference_choice(self):
        from robot_gym.utils.helpers import update_cfg_from_args
        args = sagittal_args()
        for bad in (0., -1., float("nan"), float("inf")):
            args.sagittal_stance_weight = bad
            with self.assertRaisesRegex(ValueError, "finite and positive"):
                update_cfg_from_args(*task_registry.get_cfgs("go2w"), args)
        args.sagittal_stance_weight, args.go2w_profile = 2., "step_recovery_v1"
        with self.assertRaisesRegex(ValueError, "requires go2w"):
            update_cfg_from_args(*task_registry.get_cfgs("go2w"), args)
        args.go2w_profile = "event_step_v1"
        e,t = update_cfg_from_args(*task_registry.get_cfgs("go2w"), args)
        saved = {"env_cfg":class_to_dict(e), "train_cfg":class_to_dict(t)}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"config.yaml"; path.write_text(yaml.safe_dump(saved))
            # Both evaluate and play use this same CLI resolution and contract check.
            for mode in ("evaluation", "playback"):
                with self.subTest(mode=mode):
                    check_reference_contract(path, class_to_dict(e), class_to_dict(t))
                    for weight in (None, 1.2):
                        args.sagittal_stance_weight = weight
                        other,train_cfg = update_cfg_from_args(*task_registry.get_cfgs("go2w"), args)
                        with self.assertRaisesRegex(ValueError, "sagittal_stance_weight"):
                            check_reference_contract(path, class_to_dict(other), class_to_dict(train_cfg))

    def test_only_declared_continuation_difference(self):
        from robot_gym.utils.diagnostics import check_training_continuation
        from robot_gym.utils.helpers import update_cfg_from_args
        args = sagittal_args()
        e,t = update_cfg_from_args(*task_registry.get_cfgs("go2w"), args)
        e.asset.joint_names = list(e.init_state.default_joint_angles)
        e.seed = t.seed
        ee,tt = class_to_dict(e),class_to_dict(t)
        path = SAGITTAL_PARENT/"config.yaml"
        check_training_continuation(path, ee, tt, sagittal_stance_weight=2.)
        from robot_gym.scripts.train import prepare_go2w_continuation
        prepare_go2w_continuation(args,e,t)
        self.assertEqual(e._event_completed_updates,2000)
        self.assertNotIn("_event_completed_updates",class_to_dict(e))
        probe=contract.ContractTests().make_env(4,"event_step_v1")
        probe.cfg._event_completed_updates=e._event_completed_updates
        probe._build_control_tensors()
        self.assertEqual(probe.completed_updates,2000)
        with self.assertRaisesRegex(ValueError, "Unexplained"):
            check_training_continuation(path, ee, tt)
        for group,parts,value in (("env",("control","wheel_velocity_target_limit"),21.),
                                  ("env",("env","num_observations"),57),
                                  ("env",("domain_rand","friction_range"),[.5,1.2]),
                                  ("env",("rewards","scales","sagittal_pose"),-2.),
                                  ("train",("algorithm","gamma"),.99),
                                  ("train",("algorithm","learning_rate"),.001)):
            a,b=copy.deepcopy(ee),copy.deepcopy(tt);target=a if group=="env" else b
            for key in parts[:-1]:target=target[key]
            target[parts[-1]]=value
            with self.subTest(path=parts), self.assertRaises(ValueError):
                check_training_continuation(path,a,b,sagittal_stance_weight=2.)


@unittest.skipUnless(os.environ.get("GO2W_EVENT_GPU") == "sagittal", "one original-parent two-update continuation")
class SagittalContinuationSmoke(unittest.TestCase):
    def test_original_parent_two_updates(self):
        from robot_gym.scripts.train import train
        from robot_gym.utils.training_diagnostics import verify_resume_state
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        original = task_registry.make_alg_runner
        measured = {"optimizer_steps":0,"transitions":0}
        def create(*args,**kwargs):
            env=kwargs["env"]
            resample=env._resample_commands
            def sampled(ids):
                self.assertEqual(env.completed_updates,2000)
                return resample(ids)
            env._resample_commands=sampled
            runner,cfg=original(*args,**kwargs)
            env._resample_commands=resample
            measured["before"]=verify_resume_state(runner,SAGITTAL_PARENT/"model_1999.pt")
            self.assertEqual(measured["before"]["optimizer_step_counts"],[80000])
            self.assertEqual(env.completed_updates,2000)
            self.assertEqual(lateral_high(env.completed_updates,env.cfg.commands),.5)
            self.assertEqual(env.cfg.rewards.sagittal_stance_weight,2.)
            commands=env.commands.clone()
            error=(env.dof_pos-env.default_dof_pos)[:,env.sagittal_indices].square().mean(1)
            for vy,weight in ((0.,2.),(.03,1.06),(.05,.12)):
                env.commands[:]=env.commands.new_tensor([0.,vy,0.])
                torch.testing.assert_close(env._reward_sagittal_pose()*env.reward_scales["sagittal_pose"],
                                           -weight*error*env.dt,rtol=2e-6,atol=1e-8)
            env.commands.copy_(commands)
            step=env.step
            def checked_step(actions):
                count=env.step_events.update_count;result=step(actions)
                self.assertEqual(env.step_events.update_count,count+1)
                self.assertTrue(torch.isfinite(result[1]).all() and torch.isfinite(env.obs_buf).all())
                measured["transitions"]+=1
                return result
            env.step=checked_step
            def checked_optimizer(*_):
                for group in runner.alg.optimizer.param_groups:
                    for p in group["params"]:
                        self.assertTrue(torch.isfinite(p).all())
                        if p.grad is not None:self.assertTrue(torch.isfinite(p.grad).all())
                distribution=runner.alg.actor.distribution
                lo,hi=distribution.log_std_range
                self.assertTrue(((distribution.log_std_param>=lo)&(distribution.log_std_param<=hi)).all())
                measured["optimizer_steps"]+=1
            runner.alg.optimizer.register_step_post_hook(checked_optimizer)
            self.assertIn(runner.event_std_hook.id,runner.alg.optimizer._optimizer_step_post_hooks)
            return runner,cfg
        try:
            self.assertEqual(sha256(SAGITTAL_PARENT/"model_1999.pt"),"eda85fba4f0b3911ed3051e9011b68ddcfb215c9c698f113f109b9b0a9b0ccc2")
            self.assertEqual(sha256(SAGITTAL_PARENT/"config.yaml"),"43a949f8084a0b0fb3fafc1928dd01ae13997d204f939b9d6124be6ec232013e")
            with patch.object(task_registry,"make_alg_runner",create):runner=train(sagittal_args())
            self.assertEqual((measured["transitions"],measured["optimizer_steps"]),(128,80))
            env=runner.env;out=Path(runner.logger.log_dir);checkpoint=out/"model_2000.pt"
            self.assertEqual(env.completed_updates,2002)
            measured["after"]=verify_resume_state(runner,checkpoint)
            self.assertEqual(measured["after"]["optimizer_step_counts"],[80080])
            env.completed_updates=0;runner.load(checkpoint)
            verify_resume_state(runner,checkpoint)
            self.assertEqual(env.completed_updates,2002)
            self.assertIn(runner.event_std_hook.id,runner.alg.optimizer._optimizer_step_post_hooks)
            rows=[json.loads(s) for s in (out/"diagnostics.jsonl").read_text().splitlines()]
            self.assertEqual([r["event_step_v1"]["completed_updates"] for r in rows],[2001,2002])
            events=EventAccumulator(str(out)).Reload()
            losses={k:[x.value for x in events.Scalars(k)] for k in events.Tags()["scalars"] if k.startswith("Loss/")}
            self.assertTrue(losses and all(len(v)==2 and np.isfinite(v).all() for v in losses.values()))
            saved=yaml.safe_load((out/"config.yaml").read_text())
            self.assertEqual(saved["env_cfg"]["rewards"]["sagittal_stance_weight"],2.)
            measured.update(run=str(out),checkpoint_sha256=sha256(checkpoint),losses=losses,
                            continuation=json.loads((out/"continuation.json").read_text()))
            SAGITTAL_AUDIT.mkdir(exist_ok=True,parents=True)
            write_json(SAGITTAL_AUDIT/"smoke.json",measured)
            print("SAGITTAL CONTINUATION SMOKE PASS",out,flush=True)
        finally:
            destroy_genesis()


if __name__ == "__main__":
    unittest.main()
