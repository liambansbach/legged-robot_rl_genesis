"""Flat Go2-W: rolling-first rewards and a navigation-oriented command mixture."""

import torch
from genesis.utils.geom import inv_quat, transform_by_quat

from robot_gym.envs.go2.go2_env import Go2Env
from .go2w_config import uses_event_steps


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


class Go2WEnv(Go2Env):
    @staticmethod
    def add_arguments(parser):
        from .cli import add_arguments
        add_arguments(parser)

    @staticmethod
    def configure(env_cfg, train_cfg, args):
        from .cli import configure
        configure(env_cfg, train_cfg, args)

    @staticmethod
    def configure_evaluation(cfg, args):
        from .deployment import select_transfer_dynamics
        select_transfer_dynamics(cfg, args)

    @staticmethod
    def validate_training(args, env_cfg, train_cfg):
        from .training import validate_training
        validate_training(args, env_cfg, train_cfg)

    def setup_runner(self, runner):
        if self.event_step:
            from .step_events import install_event_training
            install_event_training(runner, self)

    def install_training_diagnostics(self, runner, log_dir):
        from pathlib import Path
        from .training_diagnostics import TrainingDiagnostics
        TrainingDiagnostics(runner, self, Path(log_dir) / "diagnostics.jsonl")

    def training_metadata(self):
        from .training import training_metadata
        return training_metadata(self)

    def export_metadata(self):
        from .deployment import transfer_contract
        metadata = {"observation_order": ["body_linear_velocity", "body_angular_velocity",
                    "projected_gravity", "body_velocity_command", "leg_position_error",
                    "joint_velocity", "previous_clipped_action"]}
        if self.cfg.control.armature is not None:
            metadata.update(transfer_contract(self))
        return metadata

    def update_task_state(self):
        if self.event_step:
            self._update_step_events()

    def capture_task_state(self):
        state = {}
        if getattr(self.cfg.env, "capture_closed_loop", False):
            # Preserve the issued/delayed action and cached loads before reset clears them.
            state["applied_actions"] = self.applied_actions.clone()
            if getattr(self, "wheel_geometry_enabled", getattr(self, "step_recovery", False)):
                state["wheel_normal_force"] = self.wheel_normal_force.clone()
                state["loaded_wheels"] = self.loaded_wheels.clone()
        if getattr(self.cfg.env, "capture_precision", False):
            for name in ("wheel_clearance", "wheel_link_quat", "wheel_reposition_velocity_body", "wheel_center_lateral_speed"):
                state[name] = getattr(self, name).clone()
            if getattr(self, "event_step", False):
                for name in ("completed", "valid", "censored", "censored_count", "peak_actual",
                             "peak_use", "reposition", "duration", "quality", "payment", "gate"):
                    state["event_" + name] = getattr(self.step_events, name).clone()
        return state

    def enable_zero_command_brake(self):
        if getattr(self.cfg, "go2w_profile", None) == "step_recovery_v1" or uses_event_steps(self.cfg):
            raise ValueError(
                "Go2-W step recovery requires zero-command braking disabled"
            )
        from .zero_command_brake import ZeroCommandBrake

        self.zero_command_brake = ZeroCommandBrake(
            self.num_envs,
            self.wheel_action_indices,
            self.dt,
            self.device,
            self.cfg.normalization.clip_actions,
        )

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        super().reset_idx(env_ids)
        # Installed Genesis 1.4.1 Scene.reset restores state, retaining DOF info
        # (including build-time armature samples) and link mass/COM/inertia.
        if getattr(self, "wheel_geometry_enabled", getattr(self, "step_recovery", False)):
            self.wheel_clearance[env_ids] = 0
            self.wheel_normal_force[env_ids] = 0
            self.loaded_wheels[env_ids] = False
            self.wheel_reposition_velocity_body[env_ids] = 0
            self.wheel_center_lateral_speed[env_ids] = 0
        if hasattr(self, "step_events"):
            self.step_events.reset(env_ids)
        brake = getattr(self, "zero_command_brake", None)
        if brake is not None:
            brake.reset(env_ids)

    def reset(self):
        result = super().reset()
        brake = getattr(self, "zero_command_brake", None)
        if brake is not None:
            # The base reset performs one zero-action settling step.
            brake.reset()
        return result

    def _build_control_tensors(self):
        super()._build_control_tensors()
        self.step_recovery = (
            getattr(self.cfg, "go2w_profile", None) == "step_recovery_v1"
        )
        self.event_step = uses_event_steps(self.cfg)
        self.transfer_v2 = getattr(self.cfg, "go2w_profile", None) == "transfer_v2"
        self.wheel_geometry_enabled = self.step_recovery or self.event_step
        if self.event_step:
            self.completed_updates = getattr(self.cfg, "_event_completed_updates", 0)
            self.hip_indices = torch.tensor([i for i, n in enumerate(self.joint_names)
                                             if n.endswith("_hip_joint")], device=self.device)
            self.sagittal_indices = torch.tensor([i for i, n in enumerate(self.joint_names)
                                                  if n.endswith(("_thigh_joint", "_calf_joint"))], device=self.device)
            if self.transfer_v2:
                support = {name.removesuffix("_foot"): i for i, name in enumerate(self.cfg.asset.foot_link_names)}
                self.hip_support_indices = torch.tensor([support[self.joint_names[i].rsplit("_", 2)[0]]
                                                        for i in self.hip_indices.tolist()], device=self.device)
                self.sagittal_support_indices = torch.tensor([support[self.joint_names[i].rsplit("_", 2)[0]]
                                                             for i in self.sagittal_indices.tolist()], device=self.device)
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
            if getattr(self.cfg, "go2w_finetune", None) or self.event_step:
                self.diagnostic_long_moving_commands = torch.zeros(
                    self.num_envs, dtype=torch.bool, device=self.device
                )
        low, high = self.cfg.commands.pure_lateral_magnitude_range
        y_min, y_max = self.cfg.commands.ranges.lin_vel_y
        if not 0 < low <= high <= min(-y_min, y_max):
            raise ValueError(
                "Pure-lateral magnitudes must fit both signs of the command range"
            )

    def _init_buffers(self):
        super()._init_buffers()
        if self.wheel_geometry_enabled:
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
            if self.event_step:
                from .step_events import WheelStepEvents
                self.step_events = WheelStepEvents(self.num_envs, self.device,
                                                   self.cfg.rewards.event_step, self.wheel_geometry)

    def validate_asset(self):
        from .go2w_config import check_target_intervals
        if getattr(self.cfg, "go2w_profile", None):
            check_target_intervals(self.cfg, self.urdf_reader)
        if self.cfg.control.armature is not None:
            radii = [float(self.urdf_reader.root.find(
                f"link[@name='{name}']/collision/geometry/cylinder").get("radius"))
                for name in self.cfg.asset.foot_link_names]
            if len(set(radii)) != 1:
                raise ValueError("Go2-W contact-height metadata requires equal wheel radii")
            self.cfg.asset.contact_height = radii[0]

    def _update_wheel_support(self, contacts):
        from robot_gym.utils.diagnostics import summed_normal_force

        self.wheel_normal_force[:] = summed_normal_force(
            contacts, self.foot_link_indices
        )
        self.loaded_wheels[:] = (
            self.wheel_normal_force > self.cfg.rewards.contact_force_threshold
        )

    def _update_robot_state(self):
        super()._update_robot_state()
        if self.wheel_geometry_enabled:
            from robot_gym.utils.diagnostics import (
                cylinder_clearance,
                link_reposition_velocity,
            )

            wheel_quat = self.robot.get_links_quat(self.foot_link_indices_local)
            if self.event_step or getattr(self.cfg.env, "capture_precision", False):
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
            if self.transfer_v2 or getattr(self.cfg.env, "capture_precision", False):
                self.wheel_center_lateral_speed[:] = lateral_wheel_center_velocity(
                    wheel_quat, self.foot_lin_vel, self.robot.get_links_ang(self.foot_link_indices_local),
                    self.wheel_geometry,
                )

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
        cmd = torch.rand((n, 3), device=self.device)
        # Reuse the same uniform draw: balanced sign, independent uniform magnitude.
        lateral_sample = 2 * cmd[:, 1] - 1
        for axis, name in enumerate(("lin_vel_x", "lin_vel_y", "ang_vel_yaw")):
            low, high = self.command_ranges[name]
            cmd[:, axis] = low + (high - low) * cmd[:, axis]
        low, high = self.cfg.commands.pure_lateral_magnitude_range
        pure_lateral = (low + (high - low) * lateral_sample.abs()) * torch.where(
            lateral_sample < 0, -1.0, 1.0
        )
        if self.event_step:
            from .step_events import lateral_high
            cfg = self.cfg.commands
            # One uniform supplies independent sign and a mixture quantile.
            u = lateral_sample.abs()
            split = 1 - cfg.lateral_tail_probability
            magnitude = torch.where(u < split, low + (high - low) * u / split,
                                    high + (lateral_high(self.completed_updates, cfg) - high) * (u - split) / (1 - split))
            pure_lateral = magnitude * torch.where(lateral_sample < 0, -1.0, 1.0)
            lo, hi = cfg.mixed_lateral_range
            cmd[:, 1] = lo + (hi - lo) * (lateral_sample + 1) / 2
        cmd[:, 1] = torch.where(families == 5, pure_lateral, cmd[:, 1])
        # Most commands use wheels and yaw. Lateral demand is explicit and uncommon.
        cmd[:, 1] *= families >= 5
        cmd[:, 2] *= (families != 1) & (families != 5)
        cmd[:, 0] *= (families != 3) & (families != 5)
        precision = families == 4
        cmd[precision, 0] *= 0.18
        cmd[precision, 2] *= 0.2
        cmd[families == 0] = 0
        cmd[:, :2] *= (
            torch.linalg.vector_norm(cmd[:, :2], dim=1)
            > self.cfg.commands.linear_deadzone
        ).unsqueeze(1)
        cmd[:, 2] *= cmd[:, 2].abs() > self.cfg.commands.yaw_deadzone
        self.commands[env_ids] = cmd
        if self.step_recovery or self.event_step:
            stand_ids = env_ids[families == 0]
            long_ids = stand_ids[
                torch.rand(len(stand_ids), device=self.device)
                < self.cfg.commands.long_stand_probability
            ]
            self.command_steps_left[long_ids] = self._sample_interval_steps(
                self.cfg.commands.long_stand_duration_range, len(long_ids), self.dt
            )
        if getattr(self.cfg, "go2w_finetune", None) or self.event_step:
            cfg = self.cfg.commands
            mixed_ids = env_ids[families == 6]
            zero_yaw = torch.rand(len(mixed_ids), device=self.device) < cfg.mixed_zero_yaw_probability
            self.commands[mixed_ids[zero_yaw], 2] = 0
            # Reassign one sixth of the ordinary 30% sustained MOVING segments.
            # Stand's original draws and 25% long-stand replacement stay intact.
            sustained = self.command_steps_left[env_ids] >= round(cfg.sustained_command_duration_range[0] / self.dt)
            candidates = env_ids[(families != 0) & sustained]
            long_ids = candidates[torch.rand(len(candidates), device=self.device)
                                  < cfg.moving_long_probability / cfg.sustained_command_probability]
            self.command_steps_left[long_ids] = self._sample_interval_steps(
                cfg.moving_long_duration_range, len(long_ids), self.dt
            )
            if hasattr(self, "diagnostic_long_moving_commands"):
                self.diagnostic_long_moving_commands[env_ids] = (
                    (families != 0) & (self.command_steps_left[env_ids] > round(3 / self.dt))
                )
        if self.transfer_v2:
            from .step_events import step_demand
            cfg = self.cfg.commands
            # These are segment probabilities. Preserve already drawn 8-15 s holds.
            eligible = ((families == 3) | (families == 5) | (families == 6))
            eligible &= step_demand(self.commands[env_ids]) > 0
            eligible &= self.command_steps_left[env_ids] < round(cfg.moving_long_duration_range[0] / self.dt)
            selected = env_ids[eligible & (torch.rand(n, device=self.device) < cfg.discovery_segment_probability)]
            self.command_steps_left[selected] = self._sample_interval_steps(
                cfg.discovery_segment_duration_range, len(selected), self.dt)
        if hasattr(self, "step_events"):
            self.step_events.command_changed(self.commands)

    def set_fixed_command(self, command):
        super().set_fixed_command(command)
        if hasattr(self, "step_events"):
            self.step_events.command_changed(self.commands)

    def _update_step_events(self):
        finite = torch.stack([torch.isfinite(value).reshape(self.num_envs, -1).all(dim=1)
                              for value in (self.dof_pos, self.dof_vel, self.base_lin_vel,
                                            self.base_ang_vel, self.torques, self.actions)]).all(dim=0)
        self.step_events.update(self.dt, self.commands, self.base_pos, self.base_quat,
                                self.foot_pos, self.wheel_link_quat, self.wheel_normal_force,
                                self.reset_buf.bool() | (self.nonfoot_contact_count > 0) | ~finite,
                                actual_clearance=self.wheel_clearance)

    def _step_demand(self):
        from .step_events import step_demand
        return step_demand(self.commands)

    def _reward_step_event(self):
        return self.step_events.payment.sum(dim=1)

    def compute_reward(self):
        if self.event_step:
            if getattr(self, "_event_reward_step", None) == self.common_step_counter:
                return
            self._event_reward_step = self.common_step_counter
        super().compute_reward()

    def _reward_hip_pose(self):
        error = (self.dof_pos - self.default_dof_pos)[:, self.hip_indices]
        if self.transfer_v2:
            return self._supported_pose(error, self.hip_support_indices, "hip")
        return (1 - 0.5 * self._step_demand()) * error.square().mean(dim=1)

    def _reward_sagittal_pose(self):
        error = (self.dof_pos - self.default_dof_pos)[:, self.sagittal_indices]
        if self.transfer_v2:
            return self._supported_pose(error, self.sagittal_support_indices, "sagittal")
        stance = getattr(self.cfg.rewards, "sagittal_stance_weight", None)
        if stance is not None:
            gate = self._step_demand()
            # The existing -0.6 scale and policy dt are applied by the accumulator.
            return ((stance * (1 - gate) + 0.12 * gate) / 0.6) * error.square().mean(dim=1)
        return (1 - 0.8 * self._step_demand()) * error.square().mean(dim=1)

    def _supported_pose(self, error, support_indices, group):
        coefficients = self.cfg.rewards.support_pose[group]
        demand = self._step_demand()[:, None]
        moving = torch.where(self.loaded_wheels[:, support_indices], coefficients["loaded"], coefficients["unloaded"])
        coefficient = coefficients["stand"] * (1 - demand) + moving * demand
        return (coefficient * error.square()).mean(dim=1)

    def _reward_wheel_swing(self):
        return self.step_events.dense_swing(self.wheel_clearance, self.commands, self.cfg.rewards.dense_swing)

    def _reward_lateral_wheel_scrub(self):
        return self._step_demand() * (self.loaded_wheels * self.wheel_center_lateral_speed.square()).mean(dim=1)

    def _reward_prolonged_unloading(self):
        return ((self.step_events.unloaded_time - 0.60) / 0.20).clamp(0, 1).square().mean(dim=1)

    def _reward_insufficient_support(self):
        return torch.relu(2 - (self.wheel_normal_force > 6).sum(dim=1)).square()

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
        if self.add_noise:
            self.obs_buf += (
                2 * torch.rand_like(self.obs_buf) - 1
            ) * self.noise_scale_vec

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

    def _compute_fallen_mask(self):
        return super()._compute_fallen_mask() | self.base_contact

    def _reward_tracking_lin_vel(self):
        error = self.commands[:, :2] - self.base_lin_vel[:, :2]
        # Missing a small explicit lateral command must cost more than ordinary rolling noise.
        return torch.exp(
            -error[:, 0].square() / self.cfg.rewards.tracking_sigma_x
            - error[:, 1].square() / self.cfg.rewards.tracking_sigma_y
        )

    def _lateral_gait_gate(self):
        cfg = self.cfg.rewards
        return (
            (self.commands[:, 1].abs() - cfg.lateral_step_start_vel)
            / (cfg.lateral_step_activation_vel - cfg.lateral_step_start_vel)
        ).clamp(0, 1)

    def _yaw_mobility_gate(self):
        cfg = self.cfg.rewards
        return (
            (self.commands[:, 2].abs() - cfg.yaw_mobility_start)
            / (cfg.yaw_mobility_full - cfg.yaw_mobility_start)
        ).clamp(0, 1)

    def _mobility_gate(self):
        # Allow moderate-yaw unloading without requiring a prescribed gait.
        return torch.maximum(
            self._lateral_gait_gate(),
            self.cfg.rewards.yaw_mobility_weight * self._yaw_mobility_gate(),
        )

    def _pose_relaxation_gate(self):
        cfg = self.cfg.rewards
        yaw = (
            (self.commands[:, 2].abs() - cfg.yaw_pose_start)
            / (cfg.yaw_pose_full - cfg.yaw_pose_start)
        ).clamp(0, 1)
        return torch.maximum(self._lateral_gait_gate(), cfg.yaw_pose_weight * yaw)

    def _gait_gate(self):
        # Go2's inherited swing-clearance kernel calls this hook; pose uses its own gate.
        return self._mobility_gate()

    def _reward_default_pose(self):
        err = (
            (self.dof_pos - self.default_dof_pos)[:, self.leg_action_indices]
            .square()
            .mean(dim=1)
        )
        return (1 - 0.7 * self._pose_relaxation_gate()) * err

    def _reward_leg_motion(self):
        if self.event_step:
            return (1 - self._step_demand()) * self.dof_vel[:, self.leg_action_indices].square().mean(dim=1)
        return (1 - 0.7 * self._mobility_gate()) * self.dof_vel[
            :, self.leg_action_indices
        ].square().mean(dim=1)

    def _reward_normalized_effort(self):
        # Clipped instantaneous P/V control effort; an effort surrogate, not measured mechanical energy.
        effort = (self.torques / self.torque_limits).square()
        return effort[:, self.leg_action_indices].mean(dim=1) + effort[
            :, self.wheel_action_indices
        ].mean(dim=1)

    def _reward_leg_acc(self):
        return (
            ((self.dof_vel - self.last_dof_vel)[:, self.leg_action_indices] / self.dt)
            .square()
            .sum(dim=1)
        )

    def _reward_wheel_acc(self):
        return (
            ((self.dof_vel - self.last_dof_vel)[:, self.wheel_action_indices] / self.dt)
            .square()
            .sum(dim=1)
        )

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

    def _reward_stand_still(self):
        motion = (
            self.base_lin_vel[:, :2].square().sum(dim=1)
            + self.base_ang_vel[:, 2].square()
        )
        motion += 0.02 * self.dof_vel[:, self.wheel_action_indices].square().mean(dim=1)
        return self._stand_mask() * motion

    def _reward_collision(self):
        return self.nonfoot_contact_count.clamp(max=4)

    def _reward_unnecessary_wheel_air(self):
        if self.event_step:
            return (1 - self._step_demand()) * self.step_events.unloaded.float().mean(dim=1)
        # Retain a contact preference even when stepping is allowed.
        contacts = self.loaded_wheels if self.step_recovery else self.foot_contacts
        relaxation = getattr(self.cfg.rewards, "wheel_air_relaxation", 0.75)
        return (1 - relaxation * self._mobility_gate()) * (~contacts).float().mean(dim=1)

    def _reward_tracking_ang_vel(self):
        if self.event_step:
            error = (self.commands[:, 2] - self.base_ang_vel[:, 2]).square()
            mixture = self.cfg.rewards.yaw_tracking_mixture
            weight = mixture["broad_weight"]
            return weight * torch.exp(-error / mixture["broad_sigma"]) + (1 - weight) * torch.exp(-error / mixture["precise_sigma"])
        denominator = getattr(self.cfg.rewards, "tracking_sigma_yaw", None)
        if denominator is None:
            return super()._reward_tracking_ang_vel()
        error = self.commands[:, 2] - self.base_ang_vel[:, 2]
        return torch.exp(-error.square() / denominator)

    def _reward_foot_swing_clearance(self):
        if not self.step_recovery:
            return super()._reward_foot_swing_clearance()
        cfg = self.cfg.rewards
        height = self.wheel_clearance.clamp_min(0)
        unloaded = (~self.loaded_wheels).float()
        activation = (height / cfg.clearance_activation_height).clamp(0, 1)
        kernel = torch.exp(
            -(height - cfg.clearance_target).square() / (2 * cfg.clearance_sigma**2)
        )
        speed = (
            self.wheel_reposition_velocity_body[..., :2].norm(dim=-1)
            / cfg.reposition_speed
        ).clamp(0, 1)
        reward = (unloaded * activation * kernel * speed).sum(dim=1) / unloaded.sum(
            dim=1
        ).clamp_min(1)
        return self._mobility_gate() * (self.loaded_wheels.sum(dim=1) >= 2) * reward

    def _reward_wheel_crossover(self):
        n = self.foot_pos.shape[1]
        feet = transform_by_quat(
            (self.foot_pos - self.base_pos[:, None]).reshape(-1, 3),
            inv_quat(self.base_quat)[:, None].expand(-1, n, -1).reshape(-1, 4),
        )
        y = feet.reshape(self.num_envs, n, 3)[:, :, 1]
        left, right = y[:, [0, 2]], y[:, [1, 3]]
        side = self.cfg.rewards.min_wheel_side_clearance
        separation = self.cfg.rewards.min_lateral_wheel_separation
        return (
            torch.relu(side - left).square()
            + torch.relu(right + side).square()
            + 0.5 * torch.relu(separation - (left - right)).square()
        ).sum(dim=1)
