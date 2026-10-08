"""Flat Go2-W: rolling-first rewards and a navigation-oriented command mixture."""

import torch
from genesis.utils.geom import inv_quat, transform_by_quat

from robot_gym.envs.go2.go2_env import Go2Env


def lateral_wheel_center_velocity(link_quat, link_vel, link_ang, geometry):
    """World-frame cylinder-center velocity along its horizontal axle (m/s).

    Genesis get_links_vel(relative=True) already refers to the authored origin.
    Transport once to the collision center. This is a scrubbing surrogate, not
    material-point tire slip; wheel spin about a centered axle contributes zero.
    """
    from robot_gym.utils.diagnostics import rotate_wxyz
    offset = rotate_wxyz(link_quat, geometry[0])
    center_velocity = link_vel + torch.linalg.cross(link_ang, offset)
    lateral = rotate_wxyz(link_quat, geometry[1]).clone()
    lateral[..., 2] = 0
    lateral = lateral / lateral.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return (center_velocity * lateral).sum(dim=-1)


def fixed_sensor_frames(path):
    """Authored fixed transforms; frame names do not supply RGB calibration."""
    import xml.etree.ElementTree as ET
    import numpy as np
    from robot_gym.utils.diagnostics import urdf_link_poses
    tree = ET.parse(path).getroot()
    parents = {j.find("child").get("link"): j for j in tree.findall("joint")}
    poses = urdf_link_poses(path, {})
    result = {}
    for name in ("front_realsense", "radar", "front_realsense_body"):
        chain, link = [], name
        while link != "base_link":
            joint = parents[link]
            if joint.get("type") != "fixed":
                raise ValueError(f"Sensor {name} is not rigidly attached to base_link")
            chain.append(joint.get("name"))
            link = joint.find("parent").get("link")
        transform = np.linalg.inv(poses["base_link"]) @ poses[name]
        result[name] = {"translation_m": transform[:3, 3].tolist(),
                        "rotation": transform[:3, :3].tolist(), "fixed_chain": chain[::-1]}
    return result


def rigid_sensor_state(base_position, base_quaternion, linear_body, angular_body, offsets):
    """World positions/velocities from authored-base origin state, transported once."""
    from robot_gym.utils.diagnostics import rotate_wxyz
    offsets = offsets.to(device=base_position.device, dtype=base_position.dtype)
    offset_world = rotate_wxyz(base_quaternion[:, None], offsets)
    velocity_body = linear_body[:, None] + torch.cross(angular_body[:, None].expand_as(offset_world),
                                                       offsets.expand_as(offset_world), dim=-1)
    return base_position[:, None] + offset_world, rotate_wxyz(base_quaternion[:, None], velocity_body)


def unique_contact_count(ids, valid):
    """Count a link/pair once per boundary, regardless of manifold multiplicity."""
    ordered = torch.where(valid, ids, -1).sort(dim=1).values
    first = torch.ones_like(ordered, dtype=torch.bool)
    first[:, 1:] = ordered[:, 1:] != ordered[:, :-1]
    return ((ordered >= 0) & first).sum(dim=1)


def undesired_self_contacts(contacts, link_start, allowed_pairs, force_threshold):
    """Named robot pairs only; symmetric cached mask excludes adjacent links."""
    a, b = contacts['link_a'] - link_start, contacts['link_b'] - link_start
    count = allowed_pairs.shape[0]
    valid = contacts['valid_mask'] & (a >= 0) & (a < count) & (b >= 0) & (b < count)
    valid &= contacts['force_a'].norm(dim=-1) > force_threshold
    valid &= allowed_pairs[a.clamp(0, count-1), b.clamp(0, count-1)]
    return unique_contact_count(torch.minimum(a, b) * count + torch.maximum(a, b), valid)


def navigation_commands(families, ranges, cfg):
    """Core/edge/reserve command draws; stand and precision stay inside core."""
    n, device = len(families), families.device
    sampling = cfg.sampling
    names = ('lin_vel_x', 'lin_vel_y', 'ang_vel_yaw')
    core = torch.tensor([ranges[name] for name in names], device=device)
    reserve = torch.tensor([sampling['reserve_ranges'][name] for name in names], device=device)
    cmd = core[:, 0] + torch.rand((n, 3), device=device) * (core[:, 1] - core[:, 0])
    active = torch.stack(((families == 1) | (families == 2) | (families == 4) | (families == 6),
                          (families == 5) | (families == 6),
                          (families == 2) | (families == 3) | (families == 4) | (families == 6)), dim=1)
    for family, axis, limits in ((5, 1, cfg.pure_lateral_magnitude_range),
                                 (3, 2, cfg.pure_yaw_magnitude_range)):
        u = 2 * torch.rand(n, device=device) - 1
        cmd[:, axis] = torch.where(families == family,
            (limits[0] + (limits[1] - limits[0]) * u.abs()) * torch.where(u < 0, -1., 1.), cmd[:, axis])
    precision = torch.tensor(sampling['precision_ranges'], device=device)
    cmd = torch.where((families == 4)[:, None],
        precision[:, 0] + torch.rand((n, 3), device=device) * (precision[:, 1] - precision[:, 0]), cmd)
    cmd *= active

    tier = torch.multinomial(torch.tensor(sampling['tier_probabilities'], device=device), n, replacement=True)
    tier[(families == 0) | (families == 4)] = 0
    # Select only one active axis, uniformly, then choose its signed limit.
    axis = torch.rand((n, 3), device=device).masked_fill(~active, -1).argmax(dim=1)
    sign = torch.randint(0, 2, (n,), device=device)
    edge, outer = core[axis, sign], reserve[axis, sign]
    # A reserve draw is strictly outside core, even for a zero uniform draw.
    inner = torch.nextafter(edge, outer)
    extended = inner + torch.rand(n, device=device) * (outer - inner)
    rows = torch.arange(n, device=device)
    cmd[rows, axis] = torch.where(tier == 1, edge,
                                 torch.where(tier == 2, extended, cmd[rows, axis]))
    return cmd, tier


def horizontal_push_force(draws, mixture):
    """One tier, magnitude and azimuth per event; draws are independent U[0,1)."""
    # Sum in Python precision before casting; .80 + .15 must not move the
    # float32 .95 tier boundary by an extra representable value.
    probabilities = mixture['probabilities']
    boundaries = draws.new_tensor([sum(probabilities[:i]) for i in range(1, len(probabilities))])
    limits = draws.new_tensor(mixture['magnitude_ranges_n'])
    tier = torch.bucketize(draws[:, 0].contiguous(), boundaries, right=True)
    magnitude = limits[tier, 0] + draws[:, 1] * (limits[tier, 1] - limits[tier, 0])
    angle = 2 * torch.pi * draws[:, 2] - torch.pi
    force = torch.stack((magnitude * angle.cos(), magnitude * angle.sin(), torch.zeros_like(magnitude)), dim=1)
    # Keep the vector norm inside the configured cap even at float32 roundoff.
    cap = torch.nextafter(limits[:, 1].max(), draws.new_zeros(()))
    force *= (cap / force.norm(dim=1).clamp_min(cap)).unsqueeze(1)
    return force, tier


class Go2WEnv(Go2Env):
    phase_guided = True  # Capability exposed to shared diagnostics.
    wheel_geometry_enabled = True  # Shared ground-contact callback.

    @staticmethod
    def replay_observation_dim(saved, default):
        from .go2w_config import validate_schema
        validate_schema(saved)
        return default

    @staticmethod
    def restore_saved_config(saved):
        from .go2w_config import restore_saved_config
        return restore_saved_config(saved)

    @staticmethod
    def add_arguments(parser):
        from .go2w_config import add_arguments
        add_arguments(parser)

    @staticmethod
    def configure(env_cfg, train_cfg, args):
        from .go2w_config import configure
        configure(env_cfg, train_cfg, args)

    @staticmethod
    def configure_evaluation(cfg, args):
        from .deployment import select_transfer_dynamics
        select_transfer_dynamics(cfg, args)

    @staticmethod
    def validate_training(args, env_cfg, train_cfg):
        from .go2w_config import validate_training
        validate_training(args, env_cfg, train_cfg)

    def setup_runner(self, runner):
        if getattr(self.cfg, "training_resume", None) and runner.logger.log_dir is not None:
            from .diagnostics import check_continuation_output
            check_continuation_output(runner.checkpoint_path, runner.logger.log_dir)
        from .phase import project_log_std
        runner.std_projection_hook = project_log_std(runner.alg.optimizer, runner.alg.actor.distribution)

    def finalize_runner_loading(self, runner, train_cfg, args):
        """Verify native full-state restoration and registered scales before learning."""
        if not args.resume:
            return
        import math
        from robot_gym.utils.helpers import class_to_dict
        from .training_diagnostics import verify_resume_state
        state = verify_resume_state(runner, runner.checkpoint_path)
        scales = class_to_dict(self.cfg.rewards.scales)
        if getattr(args, "resume_current_reward_scales", False):
            if scales != args._go2w_current_reward_scales:
                raise ValueError("Current reward scales were overwritten during native loading")
        if getattr(args, "resume_current_rewards", False):
            if (class_to_dict(self.cfg.rewards) != args._go2w_current_rewards
                    or self.cfg.config_version != args._go2w_current_version):
                raise ValueError("Current rewards/schema were overwritten during native loading")
        expected = {k: v if k in self.cfg.rewards.discrete_reward_names else v * self.dt
                    for k, v in scales.items() if v}
        if (self.reward_scales.keys() != expected.keys()
                or any(not math.isclose(self.reward_scales[k], v, rel_tol=0., abs_tol=1e-12)
                       for k, v in expected.items())):
            raise ValueError("Registered reward scales must apply policy dt exactly once")
        self.resume_validation = {
            "before_updates": True, "learning_state": state,
            "config_version": self.cfg.config_version,
            "common_tracking": dict(self.cfg.rewards.common_tracking),
            "unscaled_reward_scales": scales, "runtime_reward_scales": dict(self.reward_scales),
            "normalizer_counts": {name: int(getattr(runner.alg, name).obs_normalizer.count)
                                  for name in ("actor", "critic")},
        }
        print("Verified full native learning state and reward scales before updates.", flush=True)

    def install_training_diagnostics(self, runner, log_dir):
        from pathlib import Path
        from .training_diagnostics import TrainingDiagnostics
        TrainingDiagnostics(runner, self, Path(log_dir) / "diagnostics.jsonl")

    def training_metadata(self):
        from .deployment import transfer_contract
        from .straight_motion import CRITIC_REFERENCE_FIELDS
        resume = self.cfg.training_resume
        return {
            "initialization": ("full native resume: actor/critic/normalizers/std/Adam/iteration" if resume
                               else "fresh actor/critic/normalizers/std/Adam; local iteration zero"),
            "training_resume": resume,
            "resume_validation": getattr(self, "resume_validation", None),
            "parameter_source": "robot_gym/envs/go2w/go2w_config.py",
            "config_version": self.cfg.config_version,
            "common_tracking": dict(self.cfg.rewards.common_tracking),
            "geometric_supervision": {
                "actor_inputs": self.num_obs, "critic_inputs": self.num_privileged_obs,
                "critic_relative_fields": list(CRITIC_REFERENCE_FIELDS),
                "parameters": dict(self.cfg.straight_motion),
                "reference": "Fixed command-onset line/heading; any changed command (including speed) or episode reset relatches; identical values, pushes, getters and PPO boundaries do not",
                "bootstrap": "Native RSL-RL 5.5.1 timeouts use pre-step transition values with the old reference; next observations use the reset reference; rollout-end values use the current reference",
                "runtime_reward_scales": {k: self.reward_scales[k] for k in
                                          ('tracking_yaw', 'straight_cross_track', 'straight_heading')},
            },
            "phase_observation_mode": self.cfg.phase_observation_mode,
            "deployment_contract": transfer_contract(self),
            "runtime_armature_min_max_kg_m2": [float(self.armature_samples.min()), float(self.armature_samples.max())],
        }

    def export_metadata(self):
        from .deployment import transfer_contract
        metadata = {"observation_order": ["body_linear_velocity", "body_angular_velocity",
                    "projected_gravity", "body_velocity_command", "leg_position_error",
                    "joint_velocity", "previous_clipped_action"]}
        if self.cfg.control.armature is not None:
            metadata.update(transfer_contract(self))
        metadata["observation_order"] += ["phase_sin", "phase_cos"]
        metadata.update(
            config_version=self.cfg.config_version,
            training_tracking_kernel=self.cfg.rewards.common_tracking['precision_kernel'],
            actor_input_dim=self.num_obs,
            privileged_inputs_exported=False,
            height_reference_m=self.cfg.rewards.base_height_target,
            reset_spawn_clearance_m=self.cfg.init_state.pos[2] - self.cfg.rewards.base_height_target,
            phase={**self.cfg.phase_guidance,
                   "inference_reset_phase": 0., "training_reset": "uniform [0,1)",
                   "timing": "Action at boundary phase p; reward scores p after four held-target physics steps; then advance p by policy_dt/period modulo 1. Episode reset replaces phase; getters and command changes never advance/reset it. Initial environment reset includes one zero-action policy tick.",
                   "observation": "slots 56:58 = sin(2*pi*p), cos(2*pi*p), unscaled and noiseless before embedded normalizer",
                   "reflection": "sagittal: p -> p+.5, both sin/cos negate"},
            wheel_thigh_dx_reference_m=self.cfg.rewards.phase_objective["wheel_thigh_dx_reference_m"],
        )
        mode = self.cfg.phase_observation_mode
        metadata["phase"]["observation_mode"] = mode
        if mode == "command_demand":
            metadata["phase"]["observation"] = "slots 56:58 = demand(command) * [sin(2*pi*p), cos(2*pi*p)] before embedded normalizer; zero raw clock at zero demand, not necessarily normalized zero"
            metadata["phase"]["demand"] = {"lateral_start_full_m_s": [.01, .05], "yaw_start_full_rad_s": [.10, .25],
                "combination": "max of clipped linear ramps, then smoothstep d*d*(3-2*d); invariant under sagittal reflection"}
        sampling = self.cfg.commands.sampling
        from robot_gym.utils.helpers import class_to_dict
        metadata['navigation_commands'] = {
            'required_ranges': class_to_dict(self.cfg.commands.ranges),
            'training_sampling': sampling,
            'units': ['m/s', 'm/s', 'rad/s'],
            'reserve_rule': 'One selected active axis outside required; other axes inside required',
        }
        mixture = self.cfg.domain_rand.push_force_mixture
        metadata['body_push_events'] = {
            'force_mixture': mixture,
            'interval_s': self.cfg.domain_rand.push_interval_range_s,
            'duration_s': self.cfg.domain_rand.push_duration_range_s,
            'application': 'Fixed horizontal force per event at base-link COM; zero external torque',
            'impulse': 'Force norm times sampled physics ticks times physics dt; reset may truncate an event',
        }
        return metadata

    def update_task_state(self):
        self.straight_reference.advance(self.base_pos, self.base_quat, self.dt)
        diagnostics = getattr(self, 'training_diagnostics', None)
        if diagnostics is not None:
            for key, value in self.straight_reference.diagnostics().items():
                diagnostics.add(key, value)
        from .phase import targets
        c = self.cfg.phase_guidance
        self.desired_swing, self.desired_clearance = targets(
            self.phase, self.commands, self.phase_offsets, c["stance_fraction"], c["apex_m"])
        contacts = self.robot.get_contacts(exclude_self_contact=False, is_padded=True)
        self.undesired_self_pair_count = undesired_self_contacts(
            contacts, self.robot.link_start, self.self_contact_pairs,
            self.cfg.rewards.contact_force_threshold)

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        # Reward/capture used the phase observed by the action just integrated.
        # Only now advance to the next boundary. Getters never advance phase.
        self.phase.add_(self.dt / self.cfg.phase_guidance["period_s"]).remainder_(1.)

    def capture_task_state(self):
        state = self.straight_reference.diagnostics()
        if hasattr(self, 'push_event_duration_steps'):
            for name in ('push_event_tier', 'push_event_duration_steps', 'push_event_impulse_ns'):
                state[name] = getattr(self, name).clone()
        if hasattr(self, 'command_sampling_tier'):
            state['command_sampling_tier'] = self.command_sampling_tier.clone()
        state.update(undesired_self_pair_count=self.undesired_self_pair_count.clone(),
                     nonfoot_ground_link_count=self.nonfoot_ground_link_count.clone())
        state.update(phase=self.phase.clone(), desired_clearance=self.desired_clearance.clone(),
                     desired_swing=self.desired_swing.clone(), wheel_thigh_dx=self.wheel_thigh_dx.clone())
        if self.cfg.env.capture_precision:
            state["wheel_center_body"] = self.wheel_center_body.clone()
        if getattr(self.cfg.env, "capture_closed_loop", False):
            # Preserve the issued/delayed action and cached loads before reset clears them.
            state["applied_actions"] = self.applied_actions.clone()
            state["wheel_normal_force"] = self.wheel_normal_force.clone()
            state["loaded_wheels"] = self.loaded_wheels.clone()
        if getattr(self.cfg.env, "capture_precision", False):
            for name in ("wheel_clearance", "wheel_link_quat", "wheel_reposition_velocity_body", "wheel_center_lateral_speed"):
                state[name] = getattr(self, name).clone()
        return state

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        super().reset_idx(env_ids)
        # Shared reset has changed simulator pose, but cached base buffers still
        # describe the terminal state. Latch only these environments at their
        # actual new pose, after the terminal reward/capture has completed.
        self.straight_reference.set_command(
            self.commands, self.robot.get_pos(), self.robot.get_quat(), env_ids, reset=True)
        self.phase[env_ids] = (torch.rand(len(env_ids), device=self.device)
                              if self.cfg.phase_guidance["randomize_reset"] else 0.)
        # Installed Genesis 1.4.1 Scene.reset restores state, retaining DOF info
        # (including build-time armature samples) and link mass/COM/inertia.
        self.wheel_clearance[env_ids] = 0
        self.wheel_normal_force[env_ids] = 0
        self.loaded_wheels[env_ids] = False
        self.wheel_reposition_velocity_body[env_ids] = 0
        self.wheel_center_lateral_speed[env_ids] = 0

    def _build_control_tensors(self):
        super()._build_control_tensors()
        self.leg_action_indices = [
            i
            for i, n in enumerate(self.joint_names)
            if self.cfg.control.control_type[n] == "P"
        ]
        self.wheel_action_indices = [
            i
            for i, n in enumerate(self.joint_names)
            if self.cfg.control.control_type[n] == "V"
        ]
        probabilities = [
            self.cfg.commands.stand_command_probability,
            *self.cfg.commands.moving_mixture_probabilities,
        ]
        if (
            len(probabilities) != 7
            or min(probabilities) < 0
            or abs(sum(probabilities) - 1) > 1e-6
        ):
            raise ValueError(
                "Go2-W command mixture requires seven nonnegative probabilities summing to one"
            )
        self.command_mixture = torch.tensor(probabilities, device=self.device)
        if getattr(self.cfg.env, "record_command_families", False):
            self.diagnostic_command_families = torch.full(
                (self.num_envs,), -1, dtype=torch.long, device=self.device
            )
        low, high = self.cfg.commands.pure_lateral_magnitude_range
        y_min, y_max = self.cfg.commands.ranges.lin_vel_y
        if not 0 < low <= high <= min(-y_min, y_max):
            raise ValueError(
                "Pure-lateral magnitudes must fit both signs of the command range"
            )

    def _init_buffers(self):
        super()._init_buffers()
        from .straight_motion import StraightMotionReference
        self.straight_reference = StraightMotionReference(self.num_envs, self.device, self.cfg.straight_motion)
        self.command_sampling_tier = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.push_event_tier = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.push_event_duration_steps = torch.zeros_like(self.push_event_tier)
        self.push_event_impulse_ns = torch.zeros(self.num_envs, device=self.device)
        c = self.cfg.rewards.reference_pose
        self.reference_pose_scales = torch.tensor(
            [c['scales_rad'][self.joint_names[i].split('_')[1]] for i in self.leg_action_indices],
            device=self.device)
        # Resolve names/adjacency once. Physical self-collision remains enabled.
        names = {link.name: link.idx_local for link in self.robot.links}
        self.self_contact_pairs = ~torch.eye(len(names), dtype=torch.bool, device=self.device)
        for joint in self.urdf_reader.root.findall('joint'):
            a, b = joint.find('parent').get('link'), joint.find('child').get('link')
            if a in names and b in names:
                self.self_contact_pairs[names[a], names[b]] = False
                self.self_contact_pairs[names[b], names[a]] = False
        self.undesired_self_pair_count = torch.zeros(self.num_envs, device=self.device)
        self.nonfoot_ground_link_count = torch.zeros(self.num_envs, device=self.device)
        offset = fixed_sensor_frames(self.urdf_reader.robot_file_path_absolute)["front_realsense"]["translation_m"]
        self.sensor_offsets = torch.tensor([offset, [offset[0], -offset[1], offset[2]]], device=self.device)
        self.sensor_vz_squared = torch.zeros(self.num_envs, device=self.device)
        self.command_hold_kind = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        from robot_gym.utils.diagnostics import wheel_cylinders

        self.wheel_geometry = wheel_cylinders(
            self.urdf_reader.robot_file_path_absolute,
            self.cfg.asset.foot_link_names,
            self.device,
        )
        self.wheel_clearance = torch.zeros_like(self.current_ankle_heights)
        self.wheel_normal_force = torch.zeros_like(self.wheel_clearance)
        self.loaded_wheels = torch.zeros_like(self.foot_contacts)
        self.wheel_reposition_velocity_body = torch.zeros_like(self.foot_pos)
        self.wheel_center_lateral_speed = torch.zeros_like(self.wheel_clearance)
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.phase_offsets = torch.tensor([self.cfg.phase_guidance["offsets"][n]
                                           for n in self.cfg.asset.foot_link_names], device=self.device)
        self.thigh_link_indices_local = [self.robot.get_link(n.replace("_foot", "_thigh")).idx_local
                                        for n in self.cfg.asset.foot_link_names]
        self.wheel_thigh_dx = torch.zeros_like(self.wheel_clearance)
        self.desired_swing = torch.zeros_like(self.wheel_clearance)
        self.desired_clearance = torch.zeros_like(self.wheel_clearance)
        self.dx_reference = torch.tensor(self.cfg.rewards.phase_objective["wheel_thigh_dx_reference_m"], device=self.device)
        # Nominal model weight per four supports; independent of DR samples.
        self.nominal_support_load = sum(float(m.get("value")) for m in
            self.urdf_reader.root.findall("link/inertial/mass")) * abs(self.cfg.sim.gravity[2]) / 4

    def validate_asset(self):
        from .go2w_config import check_target_intervals
        check_target_intervals(self.cfg, self.urdf_reader)
        import math
        p = self.cfg.phase_guidance
        if (self.cfg.env.num_observations != 58 or self.cfg.env.num_actions != 16
                or not math.isfinite(p["period_s"]) or p["period_s"] <= self.cfg.sim.dt * self.cfg.control.decimation
                or not .5 < p["stance_fraction"] < 1 or not 0 < p["apex_m"] < float("inf")
                or set(p["offsets"]) != set(self.cfg.asset.foot_link_names)):
            raise ValueError("Go2-W requires 58/16 dimensions, finite phase timing/height and all four named wheels")
        offsets = p["offsets"]
        if (offsets["FL_foot"] != offsets["RR_foot"] or offsets["FR_foot"] != offsets["RL_foot"]
                or (offsets["FR_foot"] - offsets["FL_foot"]) % 1 != .5):
            raise ValueError("Go2-W sagittal symmetry requires half-cycle diagonal offsets")
        if self.cfg.control.armature is not None:
            radii = [float(self.urdf_reader.root.find(
                f"link[@name='{name}']/collision/geometry/cylinder").get("radius"))
                for name in self.cfg.asset.foot_link_names]
            if len(set(radii)) != 1:
                raise ValueError("Go2-W contact-height metadata requires equal wheel radii")
            self.cfg.asset.contact_height = radii[0]

    def _update_wheel_support(self, contacts):
        from robot_gym.utils.diagnostics import summed_normal_force

        # This getter is already filtered to the ground. Count named links, not points.
        links = torch.cat((contacts['link_a'], contacts['link_b']), dim=1)
        forces = torch.cat((contacts['force_a'], contacts['force_b']), dim=1)
        valid = torch.cat((contacts['valid_mask'], contacts['valid_mask']), dim=1)
        valid &= (links >= self.robot.link_start) & (links < self.robot.link_end)
        valid &= ~torch.isin(links, self.foot_link_indices)
        valid &= forces[..., 2].abs() > self.cfg.rewards.contact_force_threshold
        self.nonfoot_ground_link_count = unique_contact_count(links, valid)

        self.wheel_normal_force[:] = summed_normal_force(
            contacts, self.foot_link_indices
        )
        self.loaded_wheels[:] = (
            self.wheel_normal_force > self.cfg.rewards.contact_force_threshold
        )

    def _update_robot_state(self):
        super()._update_robot_state()
        _, velocity = rigid_sensor_state(self.base_pos, self.base_quat, self.base_lin_vel,
                                         self.base_ang_vel, self.sensor_offsets)
        self.sensor_vz_squared[:] = velocity[:, :, 2].square().mean(1)
        from robot_gym.utils.diagnostics import (
            cylinder_clearance,
            link_reposition_velocity,
        )

        wheel_quat = self.robot.get_links_quat(self.foot_link_indices_local)
        if self.cfg.env.capture_precision:
            self.wheel_link_quat = wheel_quat
        self.wheel_clearance[:] = cylinder_clearance(
            self.foot_pos,
            wheel_quat,
            *self.wheel_geometry,
        )
        # Genesis 1.4.1 defaults to authored link origins (not COM), in world axes.
        # Both get_vel and get_links_vel already include COM/origin transport.
        self.wheel_reposition_velocity_body[:] = link_reposition_velocity(
            self.foot_pos,
            self.foot_lin_vel,
            self.base_pos,
            self.base_quat,
            self.base_lin_vel,
            self.base_ang_vel,
        )
        self.wheel_center_lateral_speed[:] = lateral_wheel_center_velocity(
            wheel_quat, self.foot_lin_vel, self.robot.get_links_ang(self.foot_link_indices_local),
            self.wheel_geometry,
        )
        from robot_gym.utils.diagnostics import rotate_wxyz
        center = self.foot_pos + rotate_wxyz(wheel_quat, self.wheel_geometry[0])
        thigh = self.robot.get_links_pos(self.thigh_link_indices_local, relative=True)
        self.wheel_thigh_dx[:] = rotate_wxyz(inv_quat(self.base_quat)[:, None], center - thigh)[..., 0]
        self.wheel_center_body = rotate_wxyz(inv_quat(self.base_quat)[:, None], center - self.base_pos[:, None])

    def _reset_command_timer(self, env_ids):
        n = len(env_ids)
        if not n:
            return
        cfg = self.cfg.commands
        short = self._sample_interval_steps(
            cfg.short_command_duration_range, n, self.dt
        )
        sustained = self._sample_interval_steps(
            cfg.sustained_command_duration_range, n, self.dt
        )
        # Duration is independent of command family, balancing fast response with sustained tracking.
        use_sustained = (
            torch.rand(n, device=self.device) < cfg.sustained_command_probability
        )
        self.command_steps_left[env_ids] = torch.where(use_sustained, sustained, short)

    def _sample_pushes(self):
        mixture = self.cfg.domain_rand.push_force_mixture
        self.next_push_steps -= 1
        ids = (self.next_push_steps <= 0).nonzero().flatten()
        if not len(ids):
            return
        dr = self.cfg.domain_rand
        force, tier = horizontal_push_force(torch.rand((len(ids), 3), device=self.device), mixture)
        self.push_force[ids, 0] = force
        self.push_torque[ids] = 0.
        duration = self._sample_interval_steps(dr.push_duration_range_s, len(ids), self.cfg.sim.dt)
        self.push_steps_left[ids] = duration
        self.next_push_steps[ids] = self._sample_interval_steps(dr.push_interval_range_s, len(ids), self.dt)
        self.push_event_tier[ids] = tier
        self.push_event_duration_steps[ids] = duration
        impulse = force.norm(dim=1) * duration * self.cfg.sim.dt
        self.push_event_impulse_ns[ids] = impulse
        diagnostics = getattr(self, 'training_diagnostics', None)
        if diagnostics is not None:
            # Event-weighted records, once per draw; never once per reapplication.
            diagnostics.add('push_event_tier_fraction', torch.nn.functional.one_hot(tier, 3))
            diagnostics.add('push_event_duration_physics_ticks', duration)
            diagnostics.add('push_event_impulse_ns', impulse)
        # The inherited physics-step loop reapplies this fixed wrench at link COM.

    def _resample_commands(self, env_ids):
        if self._apply_fixed_command(env_ids):
            return
        self._reset_command_timer(env_ids)
        n = len(env_ids)
        if not n:
            return
        families = torch.multinomial(self.command_mixture, n, replacement=True)
        if hasattr(self, "diagnostic_command_families"):
            self.diagnostic_command_families[env_ids] = families
        elif getattr(self, "training_diagnostics", None) is not None:
            self.training_diagnostics.command_families[env_ids] = families
        cmd, tier = navigation_commands(families, self.command_ranges, self.cfg.commands)
        if hasattr(self, 'command_sampling_tier'):
            self.command_sampling_tier[env_ids] = tier
        cmd[:, :2] *= (
            torch.linalg.vector_norm(cmd[:, :2], dim=1)
            > self.cfg.commands.linear_deadzone
        ).unsqueeze(1)
        cmd[:, 2] *= cmd[:, 2].abs() > self.cfg.commands.yaw_deadzone
        self.commands[env_ids] = cmd
        stand_ids = env_ids[families == 0]
        long_ids = stand_ids[
            torch.rand(len(stand_ids), device=self.device)
            < self.cfg.commands.long_stand_probability
        ]
        self.command_steps_left[long_ids] = self._sample_interval_steps(
            self.cfg.commands.long_stand_duration_range, len(long_ids), self.dt
        )
        from .phase import step_demand
        cfg = self.cfg.commands
        # These are segment probabilities. Preserve already drawn 8-15 s holds.
        eligible = ((families == 3) | (families == 5) | (families == 6))
        eligible &= step_demand(self.commands[env_ids]) > 0
        eligible &= self.command_steps_left[env_ids] < round(cfg.holds["long_hold_s"][0] / self.dt)
        selected = env_ids[eligible & (torch.rand(n, device=self.device) < cfg.discovery_segment_probability)]
        self.command_steps_left[selected] = self._sample_interval_steps(
            cfg.discovery_segment_duration_range, len(selected), self.dt)
        c = self.cfg.commands.holds
        self.command_hold_kind[env_ids] = 0
        family_names = ('stand', 'straight', 'arc', 'yaw', 'precision', 'lateral', 'mixed')
        eligible_ids = torch.tensor([family_names.index(name) for name in c['eligible_families']],
                                    device=self.device)
        eligible = torch.isin(families, eligible_ids)
        draw = torch.rand(n, device=self.device)
        p = c["long_hold_probability"]
        for kind, mask, duration in (
            (1, eligible & (draw < p), c["long_hold_s"]),
            (2, eligible & (draw >= p) & (draw < p+c["extended_hold_probability"]), c["extended_hold_s"]),
        ):
            selected = env_ids[mask]
            self.command_steps_left[selected] = self._sample_interval_steps(duration, len(selected), self.dt)
            self.command_hold_kind[selected] = kind
        self.straight_reference.set_command(self.commands, self.base_pos, self.base_quat, env_ids)

    def _apply_fixed_command(self, env_ids=None):
        applied = super()._apply_fixed_command(env_ids)
        if applied:
            ids = self.all_env_ids if env_ids is None else env_ids
            self.straight_reference.set_command(self.commands, self.base_pos, self.base_quat, ids)
        return applied

    def set_commands(self, commands):
        """Diagnostic command batches; use the same boundary latch as sampling.

        Call before recomputing observations/action selection. Identical values
        do nothing to the reference. This setter never corrects a command.
        """
        self.commands[:] = torch.as_tensor(commands, dtype=self.commands.dtype, device=self.device)
        self.straight_reference.set_command(self.commands, self.base_pos, self.base_quat, self.all_env_ids)

    def set_fixed_command(self, command):
        super().set_fixed_command(command)
        if command is None:
            # Unpinning changes the critic's remaining-time convention, not the anchor.
            self.compute_observations()

    def _step_demand(self):
        from .phase import demand
        return demand(self.commands)

    def compute_observations(self):
        # Keep this construction explicit so simulator velocity can later be replaced by an estimate.
        self.obs_buf = torch.cat(
            (
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands * self.commands_scale,
                (self.dof_pos - self.default_dof_pos)[:, self.leg_action_indices]
                * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
            ),
            dim=-1,
        )
        from .phase import clock_observation
        clock = clock_observation(self.phase, self.commands, self.cfg.phase_observation_mode)
        self.obs_buf = torch.cat((self.obs_buf, clock), dim=-1)
        if self.add_noise:
            self.obs_buf += (
                2 * torch.rand_like(self.obs_buf) - 1
            ) * self.noise_scale_vec
        self.privileged_obs_buf = torch.cat((self.obs_buf, self.straight_reference.critic_features(
            self.command_steps_left, self.command_resampling_enabled, self.dt)), dim=-1)

    def _get_noise_scale_vec(self, cfg):
        self.add_noise = cfg.noise.add_noise
        n = cfg.noise.noise_scales
        scale = torch.zeros(self.num_obs, device=self.device)
        scale[:3] = n.lin_vel * self.obs_scales.lin_vel
        scale[3:6] = n.ang_vel * self.obs_scales.ang_vel
        scale[6:9] = n.gravity
        scale[12:24] = n.dof_pos * self.obs_scales.dof_pos
        scale[24:40] = n.dof_vel * self.obs_scales.dof_vel
        scale[24 + torch.tensor(self.wheel_action_indices, device=self.device)] = (
            n.wheel_vel * self.obs_scales.dof_vel
        )
        return scale * cfg.noise.noise_level

    def _tracking_axis(self, axis):
        from .phase import broad_precision_tracking
        actual = self.base_lin_vel[:, axis] if axis < 2 else self.base_ang_vel[:, 2]
        c = self.cfg.rewards.common_tracking
        return broad_precision_tracking(self.commands[:, axis] - actual, c['broad_scales'][axis],
                                        c['precision_scales'][axis], c['beta'][axis], c['precision_kernel'])

    def _compute_fallen_mask(self):
        return super()._compute_fallen_mask() | self.base_contact

    # Reward functions

    def _reward_straight_cross_track(self):
        return self.straight_reference.cross_track_cost()

    def _reward_straight_heading(self):
        return self.straight_reference.heading_cost()

    def _reward_tracking_x(self):
        return self._tracking_axis(0)

    def _reward_tracking_y(self):
        return self._tracking_axis(1)

    def _reward_tracking_yaw(self):
        return self._tracking_axis(2)

    def _reward_phase_clearance(self):
        from .phase import clearance_error
        return clearance_error(self.wheel_clearance, self.desired_clearance,
                               self.cfg.rewards.phase_objective["clearance_scale_m"])

    def _reward_phase_support(self):
        from .phase import support_error
        return support_error(self.wheel_normal_force, self.desired_swing, self.nominal_support_load)

    def _reward_wheel_corridor(self):
        from .phase import corridor_error
        c = self.cfg.rewards.phase_objective
        cost = corridor_error(self.wheel_thigh_dx, self.dx_reference, self.desired_swing,
                              c["corridor_stance_m"], c["corridor_swing_m"], c["corridor_scale_m"])
        # Repositioning guidance is active only for requested stepping.
        return self._step_demand() * cost

    def _reward_reference_height(self):
        from .phase import huber
        c = self.cfg.rewards.phase_objective
        error = torch.relu((self.base_pos[:, 2] - self.cfg.rewards.base_height_target).abs() - c["height_tolerance_m"])
        return huber(error / c["height_scale_m"])

    def _reward_reference_pose(self):
        from .phase import reference_pose_error
        legs = self.leg_action_indices
        return reference_pose_error(self.dof_pos[:, legs], self.default_dof_pos[:, legs],
                                    self.reference_pose_scales, self.commands, self.cfg.rewards.reference_pose)

    def _reward_wheel_rate_zero(self):
        # Suppress wheel spinning/slipping at rest, separately from body tracking.
        return self._stand_mask() * self.dof_vel[:, self.wheel_action_indices].square().mean(dim=1)

    def _reward_contact_safety(self):
        return (self.nonfoot_ground_link_count + self.undesired_self_pair_count).clamp(max=4)

    def _reward_sensor_vertical_velocity(self):
        # Actual left-imager location and its virtual sagittal mirror; raw world-up velocity.
        return self.sensor_vz_squared

    def _reward_lateral_wheel_scrub(self):
        return self._step_demand() * (self.loaded_wheels * self.wheel_center_lateral_speed.square()).mean(dim=1)

    def _reward_insufficient_support(self):
        return torch.relu(2 - (self.wheel_normal_force > 6).sum(dim=1)).square()

    def _reward_normalized_effort(self):
        # Clipped instantaneous P/V control effort; an effort surrogate, not measured mechanical energy.
        effort = (self.torques / self.torque_limits).square()
        return effort[:, self.leg_action_indices].mean(dim=1) + effort[
            :, self.wheel_action_indices
        ].mean(dim=1)

    def _reward_leg_action_rate(self):
        return (
            (self.actions - self.last_actions)[:, self.leg_action_indices]
            .square()
            .sum(dim=1)
        )

    def _reward_wheel_action_rate(self):
        return (
            (self.actions - self.last_actions)[:, self.wheel_action_indices]
            .square()
            .sum(dim=1)
        )
