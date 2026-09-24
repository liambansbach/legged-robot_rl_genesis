"""Flat Go2-W: rolling-first rewards and a navigation-oriented command mixture."""

import torch
from genesis.utils.geom import inv_quat, transform_by_quat

from robot_gym.envs.go2.go2_env import Go2Env


class Go2WEnv(Go2Env):
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

    def _resample_commands(self, env_ids):
        self._reset_command_timer(env_ids)
        n = len(env_ids)
        if not n:
            return
        families = torch.multinomial(self.command_mixture, n, replacement=True)
        cmd = torch.rand((n, 3), device=self.device)
        for axis, name in enumerate(("lin_vel_x", "lin_vel_y", "ang_vel_yaw")):
            low, high = self.command_ranges[name]
            cmd[:, axis] = low + (high - low) * cmd[:, axis]
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

    def _yaw_gait_gate(self):
        cfg = self.cfg.rewards
        return (
            (self.commands[:, 2].abs() - cfg.yaw_step_start)
            / (cfg.yaw_step_full - cfg.yaw_step_start)
        ).clamp(0, 1)

    def _gait_gate(self):
        # Prefer skid steering at low/moderate yaw; allow leg assistance at high yaw.
        return torch.maximum(
            self._lateral_gait_gate(),
            self.cfg.rewards.yaw_gait_weight * self._yaw_gait_gate(),
        )

    def _reward_default_pose(self):
        err = (
            (self.dof_pos - self.default_dof_pos)[:, self.leg_action_indices]
            .square()
            .mean(dim=1)
        )
        return (1 - 0.7 * self._gait_gate()) * err

    def _reward_leg_motion(self):
        return (1 - 0.7 * self._gait_gate()) * self.dof_vel[
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
        return (1 - 0.75 * self._gait_gate()) * (~self.foot_contacts).float().mean(
            dim=1
        )

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
