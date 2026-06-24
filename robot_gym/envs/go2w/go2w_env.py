import torch

from robot_gym.envs.go2.go2_env import Go2Env


class Go2WEnv(Go2Env):
    """
    Go2W locomotion environment.

    The wheels should handle ordinary forward/backward locomotion. Leg swing is
    only encouraged for lateral/yaw-heavy commands where rolling alone is weak.
    """

    def __init__(self, cfg, sim_params, sim_device, headless):
        super().__init__(cfg, sim_params, sim_device, headless)

    def _ramp(self, x, deadzone, full_activation_at):
        return torch.clamp(
            (x - deadzone) / (full_activation_at - deadzone + 1e-8),
            0.0,
            1.0,
        )

    def _gait_gate(self):
        """
        Activate swing-foot rewards only for commands that likely need stepping.
        Pure x velocity should be solved by wheel velocity control.
        """
        vy = torch.abs(self.commands[:, 1])
        wz = torch.abs(self.commands[:, 2])

        lateral_gate = self._ramp(
            vy,
            deadzone=0.03,
            full_activation_at=self.cfg.rewards.lateral_step_activation_vel,
        )
        yaw_gate = self._ramp(
            wz,
            deadzone=0.08,
            full_activation_at=self.cfg.rewards.yaw_step_activation_vel,
        )

        return torch.maximum(lateral_gate, yaw_gate)

    def _wheel_drive_gate(self):
        """
        Gate for commands where the robot should mainly roll with quiet legs.
        """
        vx = torch.abs(self.commands[:, 0])
        forward_gate = self._ramp(
            vx,
            deadzone=0.03,
            full_activation_at=self.cfg.rewards.wheeled_forward_activation_vel,
        )

        return forward_gate * (1.0 - self._gait_gate())

    def _quiet_leg_gate(self):
        return torch.maximum(self._stand_mask(), self._wheel_drive_gate())

    def _reward_leg_motion(self):
        """
        Penalize unnecessary leg motion during stand-still and wheel-drive phases.
        Wheel joints are excluded through the position-control mask.
        """
        leg_mask = self.position_control_mask
        num_leg_dof = leg_mask.sum(dim=1).clamp(min=1.0)

        pos_delta = torch.where(
            leg_mask.bool(),
            self.dof_pos - self.default_dof_pos,
            torch.zeros_like(self.dof_pos),
        )
        pos_err = torch.sum(torch.square(pos_delta), dim=1) / num_leg_dof
        vel_err = torch.sum(torch.square(self.dof_vel) * leg_mask, dim=1) / num_leg_dof
        action_err = torch.sum(torch.square(self.actions) * leg_mask, dim=1) / num_leg_dof

        return self._quiet_leg_gate() * (pos_err + 0.02 * vel_err + 0.1 * action_err)

    def _reward_wheel_contact(self):
        """
        Reward keeping all wheels grounded while the command can be solved by rolling.
        """
        contact_fraction = torch.mean(self.foot_contacts.float(), dim=1)
        return self._quiet_leg_gate() * contact_fraction

    def _reward_unnecessary_wheel_air(self):
        """
        Penalize lifted wheels during stand-still and pure rolling commands.
        """
        air_fraction = torch.mean((~self.foot_contacts).float(), dim=1)
        return self._quiet_leg_gate() * air_fraction
