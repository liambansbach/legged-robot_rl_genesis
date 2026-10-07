"""Relative geometric supervision for zero-yaw translation; never a controller."""

import torch

from .phase import huber


# Appended to the unchanged policy observation, in this order. No world pose.
CRITIC_REFERENCE_FIELDS = (
    "cross_track_over_scale", "heading_sin", "heading_cos", "reference_valid",
    "activation", "reference_age_over_time_scale", "command_remaining_over_time_scale",
)


def projected_heading(quaternion):
    """Heading of the horizontal projection of body +x, using wxyz rotation."""
    w, x, y, z = quaternion.unbind(-1)
    forward_x = 1 - 2 * (y.square() + z.square())
    forward_y = 2 * (x * y + w * z)
    return torch.atan2(forward_y, forward_x), torch.hypot(forward_x, forward_y)


def wrap_angle(angle):
    return torch.atan2(angle.sin(), angle.cos())


class StraightMotionReference:
    """One anchor per environment, replaced only by command changes or reset.

    Every numerical command change (including speed-only changes) starts a new
    segment. Repeated identical commands keep their anchor and age, even when a
    sampled hold timer is renewed. Invalid segments never acquire a moving anchor.
    Caller scores the completed transition before calling ``set_command`` for
    the next segment. Neither observation getters nor PPO boundaries mutate this.
    """

    def __init__(self, count, device, cfg):
        self.cfg = cfg
        self.command = torch.zeros(count, 3, device=device)
        self.origin = torch.zeros(count, 2, device=device)
        self.heading = torch.zeros(count, device=device)
        self.normal = torch.zeros(count, 2, device=device)
        self.initialized = torch.zeros(count, dtype=torch.bool, device=device)
        self.valid = torch.zeros_like(self.initialized)
        self.age = torch.zeros(count, device=device)
        self.cross_track = torch.zeros_like(self.age)
        self.heading_error = torch.zeros_like(self.age)

    def set_command(self, commands, position, quaternion, ids, reset=False):
        """Latch at a policy boundary; inputs are full batches, ids may be partial."""
        changed = (commands[ids] != self.command[ids]).any(-1) | ~self.initialized[ids]
        if reset:
            changed = torch.ones_like(changed)
        ids = ids[changed]
        command = commands[ids]
        heading, projection = projected_heading(quaternion[ids])
        speed = command[:, :2].norm(dim=-1)
        valid = ((speed > self.cfg['min_speed_m_s'])
                 & (command[:, 2].abs() <= self.cfg['yaw_zero_rad_s'])
                 & (projection > self.cfg['heading_projection_min']))
        direction_body = command[:, :2] / speed.clamp_min(self.cfg['min_speed_m_s'])[:, None]
        c, s = heading.cos(), heading.sin()
        dx = c * direction_body[:, 0] - s * direction_body[:, 1]
        dy = s * direction_body[:, 0] + c * direction_body[:, 1]
        self.normal[ids] = torch.stack((-dy, dx), dim=-1) * valid[:, None]
        self.origin[ids] = position[ids, :2]
        self.heading[ids] = heading
        self.command[ids] = command
        self.initialized[ids] = True
        self.valid[ids] = valid
        self.age[ids] = 0
        self.cross_track[ids] = 0
        self.heading_error[ids] = 0

    def advance(self, position, quaternion, dt):
        """Exactly once after physics, before the old segment's reward/capture."""
        self.age += self.valid * dt
        self.cross_track = ((position[:, :2] - self.origin) * self.normal).sum(-1)
        heading, _ = projected_heading(quaternion)
        self.heading_error = wrap_angle(heading - self.heading) * self.valid

    def activation(self):
        fraction = (self.age / self.cfg['ramp_s']).clamp(0, 1)
        return self.valid * fraction.square() * (3 - 2 * fraction)

    def cross_track_cost(self):
        error = (self.cross_track.abs() - self.cfg['cross_tolerance_m']).clamp_min(0)
        return self.activation() * huber(error / self.cfg['cross_scale_m'])

    def heading_cost(self):
        return self.activation() * huber(self.heading_error / self.cfg['heading_scale_rad'])

    def critic_features(self, steps_left, resampling_enabled, dt):
        """Dimensionless relative state. Remaining time -1 means a pinned command."""
        time_scale = self.cfg['critic_time_scale_s']
        remaining = (steps_left.clamp_min(0) * dt / time_scale if resampling_enabled
                     else torch.full_like(self.age, -1))
        return torch.stack((self.cross_track / self.cfg['cross_scale_m'],
                            self.heading_error.sin(), self.heading_error.cos(),
                            self.valid.float(), self.activation(), self.age / time_scale,
                            remaining), dim=-1)

    def diagnostics(self):
        return {"straight_cross_track_m": self.cross_track.clone(),
                "straight_heading_error_rad": self.heading_error.clone(),
                "straight_reference_valid": self.valid.clone(),
                "straight_reference_age_s": self.age.clone(),
                "straight_activation": self.activation()}
