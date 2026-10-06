"""Focused CPU checks for the opt-in consolidated task; no simulator."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml

from robot_gym.envs.go2w.go2w_env import Go2WEnv, undesired_self_contacts, unique_contact_count
from robot_gym.envs.go2w.phase import broad_precision_tracking, huber, reference_pose_activation
from robot_gym.envs.go2w.go2w_config import validate_training
from robot_gym.utils import task_registry, get_args
from robot_gym.utils.helpers import class_to_dict, update_cfg_from_args

RECIPE = Path(__file__).resolve().parents[1]/'docs/go2w_reference_tracking_v1.yaml'


class ReferenceTracking(unittest.TestCase):
    def resolve(self, *extra):
        with patch.object(sys, 'argv', ['train','--task','go2w','--go2w_profile','transfer_v3',
                '--go2w_fresh_recipe',str(RECIPE),*extra]):
            args = get_args()
        cfg, train = task_registry.get_cfgs('go2w')
        with patch('torch.load', side_effect=AssertionError('Fresh initialization must not read a model')):
            update_cfg_from_args(cfg, train, args)
            before = copy.deepcopy((class_to_dict(cfg),class_to_dict(train)))
            update_cfg_from_args(cfg, train, args)
            self.assertEqual(before,(class_to_dict(cfg),class_to_dict(train)))
            validate_training(args,cfg,train)
        return cfg,train,args

    def fixture(self):
        cfg,_,_=self.resolve()
        env=Go2WEnv.__new__(Go2WEnv)
        env.cfg,env.device,env.dt,env.num_envs=cfg,'cpu',.02,6
        env.reference_tracking=env.phase_guided=True
        env.event_step=False
        env.commands=torch.tensor([[.5,0,0],[0,.03,0],[0,-.03,0],[0,0,.4],[0,0,-.4],[.2,.1,-.3]])
        env.base_lin_vel=env.commands.clone();env.base_ang_vel=torch.zeros_like(env.commands)
        env.base_ang_vel[:,2]=env.commands[:,2]
        return env

    def test_all_axes_track_requested_including_zero_cross_axes(self):
        env=self.fixture()
        for axis in range(3):
            torch.testing.assert_close(env._tracking_axis(axis),torch.ones(6))
            target=env.base_lin_vel if axis<2 else env.base_ang_vel
            for sign in (-1,1):
                target[:,axis]+=.03*sign
                self.assertTrue((env._tracking_axis(axis)<1).all())
                target[:,axis]-=.03*sign
        # A tiny lateral command does not switch the yaw precision kernel.
        env.base_ang_vel[:,2]+=.04
        before=env._tracking_axis(2)
        env.commands[:,1]=1e-7
        torch.testing.assert_close(env._tracking_axis(2),before)

    def test_bounded_precision_broad_sensitivity_and_start_stop(self):
        for scale in (.25,.15,.35):
            errors=torch.tensor([0.,.01,.03,.1,.5,1.,2.],dtype=torch.float64,requires_grad=True)
            actual=broad_precision_tracking(errors,scale,.03,.25)
            cost=(1-huber(errors/scale))-actual
            self.assertTrue(((cost>=0)&(cost<=.25+1e-14)).all())
            self.assertTrue((actual[:-1]>actual[1:]).all())
            actual.sum().backward()
            self.assertAlmostEqual(float(errors.grad[-1]),-1/scale)
            torch.testing.assert_close(actual,broad_precision_tracking(-errors,scale,.03,.25))
            # +.5 startup lag and -.5 stopping excess have identical costs.
            torch.testing.assert_close(broad_precision_tracking(torch.tensor(.5),scale,.03,.25),
                                       broad_precision_tracking(torch.tensor(-.5),scale,.03,.25))

    def test_actual_pose_signs_and_no_wheel_or_action_objective(self):
        env=self.fixture();env.commands.zero_()
        env.leg_action_indices=[i for i in range(16) if i%4!=3]
        env.default_dof_pos=torch.tensor([[0.,.7,-1.4,0.]*4])
        env.dof_pos=env.default_dof_pos.repeat(6,1)
        env.reference_pose_scales=torch.tensor([.08,.12,.12]*4)
        env.dof_pos[:,[3,7,11,15]]=123456.
        env.actions=torch.ones(6,16)  # Supporting target offsets are explicitly allowed.
        env.applied_actions=-env.actions
        torch.testing.assert_close(env._reward_reference_pose(),torch.zeros(6))
        env.dof_pos[1,env.leg_action_indices]+=torch.tensor([0.,.25,-.5]*4)  # crouch
        env.dof_pos[2,[0,4,8,12]]=torch.tensor([.2,-.2,.2,-.2])  # outward splay
        env.dof_pos[3,[0,4,8,12]]=torch.tensor([-.4,.4,.4,-.4])  # front clench
        env.dof_pos[4,env.leg_action_indices]=-env.default_dof_pos[:,env.leg_action_indices]
        cost=env._reward_reference_pose()
        self.assertTrue((cost[1:5]>cost[0]).all())
        self.assertEqual(float(cost[5]),0.)

    def test_smooth_relaxation_protects_small_signed_lateral(self):
        x=torch.linspace(-.051,.051,1003,dtype=torch.float64)
        commands=torch.zeros(len(x),3,dtype=x.dtype);commands[:,1]=x
        activation=reference_pose_activation(commands,.03,.25,.05)
        torch.testing.assert_close(activation,activation.flip(0))
        self.assertLess(float(abs(torch.diff(activation)).max()),.005)
        at=torch.tensor([[.5,0,0],[0,.02,0],[0,.03,0],[0,.05,0],[0,0,.25]])
        a=reference_pose_activation(at,.03,.25,.05)
        self.assertEqual(float(a[0]),1.)
        self.assertLess(float(a[1]),.30)
        torch.testing.assert_close(a[2:],torch.full((3,),.05))

    def test_old_kernels_exact_and_new_dt_once(self):
        env=self.fixture();env.base_ang_vel[:,2]+=.06
        env.reference_tracking=False
        old=1-huber((env.commands[:,2]-env.base_ang_vel[:,2])/.35)
        torch.testing.assert_close(env._reward_tracking_yaw(),old,rtol=0,atol=0)
        env.cfg.rewards.phase_objective['rolling_yaw_scale']=.07
        expected=old.clone();expected[0]=1-huber(torch.tensor(-.06/.07))
        torch.testing.assert_close(env._reward_tracking_yaw(),expected)
        env.reference_tracking=True
        env.reward_scales=dict(tracking_x=1.,tracking_y=1.,tracking_yaw=.8)
        raw=sum(w*env._tracking_axis(i) for i,w in enumerate([1.,1.,.8]))
        env.rew_buf=torch.zeros(6);env._prepare_reward_function();env.compute_reward()
        torch.testing.assert_close(env.rew_buf,raw*.02)

    def test_targeted_contacts_deduplicate_and_exclude_adjacent_ground(self):
        allowed=~torch.eye(4,dtype=torch.bool)
        allowed[0,1]=allowed[1,0]=False
        contacts=dict(link_a=torch.tensor([[10,12,10,10,10,10,9]]),
            link_b=torch.tensor([[12,10,12,11,13,13,12]]),
            valid_mask=torch.tensor([[True,True,True,True,True,False,True]]),
            force_a=torch.tensor([[[9.,0,0],[9.,0,0],[20.,0,0],[20.,0,0],[.1,0,0],[20.,0,0],[20.,0,0]]]))
        torch.testing.assert_close(undesired_self_contacts(contacts,10,allowed,8.),torch.tensor([1]))
        torch.testing.assert_close(unique_contact_count(torch.tensor([[2,2,3,3]]),torch.tensor([[True,True,False,False]])),torch.tensor([1]))
        env=self.fixture();env.nonfoot_ground_link_count=torch.tensor([0,1,0,1,3,0])
        env.undesired_self_pair_count=torch.tensor([0,0,1,1,3,0])
        torch.testing.assert_close(env._reward_contact_safety(),torch.tensor([0,1,1,2,4,0]))

    def test_corridor_only_steps_and_zero_wheel_term_is_separate(self):
        env=self.fixture()
        env.wheel_thigh_dx=torch.ones(6,4)*.2
        env.dx_reference=torch.tensor(env.cfg.rewards.phase_objective['wheel_thigh_dx_reference_m'])
        env.desired_swing=torch.zeros(6,4)
        self.assertEqual(float(env._reward_wheel_corridor()[0]),0.)
        self.assertTrue((env._reward_wheel_corridor()[1:]>0).all())
        env.wheel_action_indices=[3,7,11,15]
        env.dof_vel=torch.ones(6,16);env.commands.zero_()
        before=env._reward_wheel_rate_zero()
        env.base_lin_vel.fill_(50.);env.base_ang_vel.fill_(50.)
        torch.testing.assert_close(env._reward_wheel_rate_zero(),before)
        env.commands[:,1]=.02
        torch.testing.assert_close(env._reward_wheel_rate_zero(),torch.zeros(6))

    def test_recipe_fresh_fixed_complete_and_rejects_loading(self):
        cfg,train,args=self.resolve()
        self.assertIsNone(args.resolved_checkpoint)
        self.assertFalse(train.runner.resume)
        for key in ('resume_path','load_run','checkpoint','checkpoint_load_cfg'):
            self.assertIsNone(getattr(train.runner,key))
        self.assertIsNone(cfg.training_resume);self.assertIsNone(cfg.refinement_parent)
        self.assertEqual((cfg.env.num_envs,train.runner.num_steps_per_env,train.runner.max_iterations,train.runner.save_interval),(4096,64,2000,250))
        self.assertEqual((train.algorithm.schedule,train.algorithm.learning_rate,train.algorithm.gamma,train.algorithm.lam),('fixed',3e-4,.995,.95))
        self.assertEqual(train.actor.distribution_cfg['init_std'],.4)
        self.assertTrue(cfg.sim.enable_self_collision)
        self.assertEqual(cfg.phase_guidance['apex_m'],.04)
        self.assertEqual(cfg.commands.linear_deadzone,.01)
        for flags in [('--resume',),('--load_run','old'),('--checkpoint','1499')]:
            with self.subTest(flags=flags),self.assertRaises(ValueError):self.resolve(*flags)
        train.algorithm.schedule='adaptive'
        with self.assertRaises(ValueError):validate_training(args,cfg,train)

    def test_normal_full_state_resume_retains_new_recipe(self):
        cfg,train,_=self.resolve()
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            (folder/'config.yaml').write_text(yaml.safe_dump(dict(task='go2w',env_cfg=class_to_dict(cfg),train_cfg=class_to_dict(train))))
            torch.save({'iter':1},folder/'model_1.pt')
            with patch.object(sys,'argv',['train','--task','go2w','--go2w_profile','transfer_v3','--resume','--load_run',temp,'--checkpoint','1','--max_iterations','2']):
                args=get_args()
            resumed,training=task_registry.get_cfgs('go2w')
            update_cfg_from_args(resumed,training,args);validate_training(args,resumed,training)
            self.assertEqual(resumed.go2w_recipe,'reference_tracking_v1')
            self.assertEqual(training.runner.checkpoint_load_cfg,dict(actor=True,critic=True,optimizer=True,iteration=True))
            self.assertEqual(class_to_dict(resumed.rewards),class_to_dict(cfg.rewards))


if __name__=='__main__':unittest.main()
