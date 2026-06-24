import torch

from robot_gym.envs.base.legged_robot import LeggedRobot


class DodoEnv(LeggedRobot):
    """
    Dodo-specific reward extensions on top of the generic LeggedRobot base class.

    In addition to the rewards defined in the base class, we implemted following dodo specific rewards here:

    Implemented here:
    - forward_torso_pitch
    - foot_swing_clearance
    - flat_feet
    - hip_abduction_penalty
    - survive
    """

    def __init__(self, cfg, sim_params, sim_device, headless):
        super().__init__(cfg, sim_params, sim_device, headless)

    # ---------------------------------------------------------------------
    # helper
    # ---------------------------------------------------------------------
    def _gait_gate(self):
        """
        Command-conditioned gait activation.

        0.0 -> standing / tiny commands
        1.0 -> clear locomotion command

        Useful so swing-related rewards do not fight against stand-still behavior.
        """
        def ramp(x, deadzone, full_activation_at):
            return torch.clamp(
                (x - deadzone) / (full_activation_at - deadzone + 1e-8),
                0.0,
                1.0,
            )

        vx = torch.abs(self.commands[:, 0])
        vy = torch.abs(self.commands[:, 1])
        wz = torch.abs(self.commands[:, 2])

        # forward gets a slightly wider ramp
        g_vx = ramp(vx, deadzone=0.03, full_activation_at=0.25)
        # lateral / yaw should activate stepping earlier
        g_vy = ramp(vy, deadzone=0.03, full_activation_at=0.10)
        g_wz = ramp(wz, deadzone=0.03, full_activation_at=0.10)

        return torch.maximum(torch.maximum(g_vx, g_vy), g_wz)

    # ---------------------------------------------------------------------
    # dodo-specific rewards
    # ---------------------------------------------------------------------
    def _reward_forward_torso_pitch(self):
        """
        Reward a moderate forward torso pitch.
        """
        pitch = self.rpy[:, 1]
        target = self.cfg.rewards.pitch_target
        sigma = self.cfg.rewards.pitch_sigma

        return torch.exp(-torch.square(pitch - target) / (2.0 * sigma**2 + 1e-8))

    def _reward_foot_swing_clearance(self):
        """
        Reward desired swing-foot clearance above contact height.

        Uses real Genesis foot contacts and current ankle heights.
        Only swing feet are considered.
        """
        gait_gate = self._gait_gate()

        # bool -> float mask
        swing_mask = (~self.foot_contacts).float()

        # height above nominal ground-contact height
        clearance = torch.clamp(
            self.current_ankle_heights - self.cfg.asset.contact_height,
            min=0.0,
        )
        #print("clearance:", clearance)

        target = self.cfg.rewards.clearance_target
        sigma = self.cfg.rewards.clearance_sigma

        rew_per_foot = torch.exp(
            -torch.square(clearance - target) / (2.0 * sigma**2 + 1e-8)
        ) * swing_mask

        num_swing_feet = swing_mask.sum(dim=1).clamp(min=1.0)
        rew = torch.sum(rew_per_foot, dim=1) / num_swing_feet

        # if no foot is in swing, no clearance reward
        rew *= (swing_mask.sum(dim=1) > 0.0).float()

        return gait_gate * rew

    def _reward_flat_feet(self):
        """
        Reward feet staying parallel to the ground. 
        Only applied to feet in contact with the ground.
        Uses roll/pitch of both feet, ignores yaw.
        """
        # self.foot_euler: (N, num_feet, 3) 
        roll_pitch = self.foot_euler[:, :, :2]
        err_per_foot = torch.sum(torch.square(roll_pitch), dim=2)

        sigma = self.cfg.rewards.flat_foot_sigma
        rew_per_foot = torch.exp(-err_per_foot / (2.0 * sigma**2 + 1e-8))

        stance_mask = self.foot_contacts.float()
        denom = stance_mask.sum(dim=1).clamp(min=1.0)

        return torch.sum(rew_per_foot * stance_mask, dim=1) / denom

    def _reward_hip_abduction_penalty(self):
        """
        Penalize hip ab/adduction away from the nominal/default pose.

        For Dodo we assume the first two actuated joints are left/right hip abduction joints.
        If needed later, this can be made explicit in the config.
        """
        hip_ids = self.cfg.asset.hip_abduction_indices

        hip_pos = self.dof_pos[:, hip_ids]
        hip_default = self.default_dof_pos[:, hip_ids]

        return torch.sum(torch.square(hip_pos - hip_default), dim=1)

    def _reward_survive(self):
        """
        Small alive reward as long as the robot has not fallen.
        """
        return (~self._compute_fallen_mask()).float()