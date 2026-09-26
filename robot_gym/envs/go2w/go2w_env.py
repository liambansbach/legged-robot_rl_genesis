"""Flat Go2-W: rolling-first rewards and a navigation-oriented command mixture."""

import torch
from genesis.utils.geom import inv_quat, transform_by_quat

from robot_gym.envs.go2.go2_env import Go2Env


class Go2WEnv(Go2Env):
    def enable_zero_command_brake(self):
        if getattr(self.cfg, "go2w_profile", None) == "step_recovery_v1":
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
        super().reset_idx(env_ids)
        if getattr(self, "step_recovery", False):
            self.wheel_clearance[env_ids] = 0
            self.wheel_normal_force[env_ids] = 0
            self.loaded_wheels[env_ids] = False
            self.wheel_reposition_velocity_body[env_ids] = 0
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
        if self.step_recovery:
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
        if self.step_recovery:
            from robot_gym.utils.diagnostics import (
                cylinder_clearance,
                link_reposition_velocity,
            )

            self.wheel_clearance[:] = cylinder_clearance(
                self.foot_pos,
                self.robot.get_links_quat(self.foot_link_indices_local),
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
        if self.step_recovery:
            stand_ids = env_ids[families == 0]
            long_ids = stand_ids[
                torch.rand(len(stand_ids), device=self.device)
                < self.cfg.commands.long_stand_probability
            ]
            self.command_steps_left[long_ids] = self._sample_interval_steps(
                self.cfg.commands.long_stand_duration_range, len(long_ids), self.dt
            )

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
        # Retain a contact preference even when stepping is allowed.
        contacts = self.loaded_wheels if self.step_recovery else self.foot_contacts
        return (1 - 0.75 * self._mobility_gate()) * (~contacts).float().mean(dim=1)

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
