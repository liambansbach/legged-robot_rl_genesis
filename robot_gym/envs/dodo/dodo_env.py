import torch

from robot_gym.envs.base.legged_robot import LeggedRobot


class DodoEnv(LeggedRobot):
    """Biped torso, foot orientation and hip rewards."""

    def _reward_forward_torso_pitch(self):
        pitch = self.rpy[:, 1]
        target = self.cfg.rewards.pitch_target
        sigma = self.cfg.rewards.pitch_sigma

        return torch.exp(-torch.square(pitch - target) / (2.0 * sigma**2 + 1e-8))

    def _reward_flat_feet(self):
        """Score stance-foot roll and pitch while allowing free yaw."""
        roll_pitch = self.foot_euler[:, :, :2]
        sigma = self.cfg.rewards.flat_foot_sigma
        if getattr(self.cfg.rewards, "legacy_walking", False):
            error = torch.square(roll_pitch).sum(dim=(1, 2))
            return torch.exp(-error / (2.0 * sigma**2 + 1e-8))
        err_per_foot = torch.sum(torch.square(roll_pitch), dim=2)
        rew_per_foot = torch.exp(-err_per_foot / (2.0 * sigma**2 + 1e-8))

        stance_mask = self.foot_contacts.float()
        denom = stance_mask.sum(dim=1).clamp(min=1.0)

        return torch.sum(rew_per_foot * stance_mask, dim=1) / denom

    def _reward_hip_abduction_penalty(self):
        hip_ids = self.cfg.asset.hip_abduction_indices

        hip_pos = self.dof_pos[:, hip_ids]
        hip_default = self.default_dof_pos[:, hip_ids]

        return torch.sum(torch.square(hip_pos - hip_default), dim=1)
