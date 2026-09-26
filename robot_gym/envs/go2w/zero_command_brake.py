"""Inference-only wheel-target blend; the learned leg controller remains active."""

import math
import torch


class ZeroCommandBrake:
    def __init__(self, num_envs, wheel_indices, dt, device, action_limit=1.0):
        if not math.isclose(dt, 0.02, abs_tol=1e-9):
            raise ValueError("Go2-W zero-command braking requires 50 Hz policy updates")
        self.wheel_indices = list(wheel_indices)
        self.dt, self.action_limit = dt, action_limit
        self.alpha = torch.zeros(num_envs, device=device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.alpha.zero_()
        else:
            self.alpha[env_ids] = 0

    def apply(self, actions, commands):
        stop = (torch.linalg.vector_norm(commands[:, :2], dim=1) <= 1e-6) & (
            commands[:, 2].abs() <= 1e-6
        )
        self.alpha.add_(torch.where(stop, self.dt / 0.20, -self.dt / 0.10))
        self.alpha.clamp_(0, 1)
        # Endpoint roundoff must not add an extra policy step to either ramp.
        eps = torch.finfo(self.alpha.dtype).eps
        self.alpha[self.alpha <= eps] = 0
        self.alpha[self.alpha >= 1 - eps] = 1
        issued = actions.clamp(-self.action_limit, self.action_limit)
        issued[:, self.wheel_indices] *= 1 - self.alpha[:, None]
        return issued

    def contract(self):
        return {
            "controller": "frozen neural policy plus zero-command wheel braking",
            "policy_dt_s": self.dt,
            "xy_zero_tolerance_m_s": 1e-6,
            "yaw_zero_tolerance_rad_s": 1e-6,
            "engage_s": 0.20,
            "release_s": 0.10,
            "wheel_indices": self.wheel_indices,
            "action_limit": self.action_limit,
            "state": "one alpha per environment; initially zero; reset with that environment",
            "order": "actor -> clip -> blend wheels -> action history -> existing delay -> scaling/limits",
            "previous_action": "issued clipped/braked action before actuator delay",
            "trace": "raw_actions is actor proposal; issued_actions is pre-delay; applied_actions is post-delay",
            "fully_engaged": "zero wheel velocity target, not zero control torque",
            "scope": "inference composite; no training, hardware or sim2sim equivalence established",
        }
