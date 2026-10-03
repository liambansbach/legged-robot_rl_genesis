"""V3 observable diagonal phase targets and small dimensionless reward kernels."""

import torch


def huber(error):
    magnitude = error.abs()
    return torch.where(magnitude <= 1, .5 * error.square(), magnitude - .5)


def demand(commands):
    lateral = ((commands[:, 1].abs() - .01) / .04).clamp(0, 1)
    yaw = ((commands[:, 2].abs() - .10) / .15).clamp(0, 1)
    value = torch.maximum(lateral, yaw)
    return value.square() * (3 - 2 * value)


def targets(phase, commands, offsets, stance_fraction, apex):
    """Smooth swing envelope; all four stance during the diagonal overlap."""
    local = (phase[:, None] + offsets) % 1
    progress = ((local - stance_fraction) / (1 - stance_fraction)).clamp(0, 1)
    envelope = torch.sin(torch.pi * progress).square() * (local >= stance_fraction)
    swing = demand(commands)[:, None] * envelope
    return swing, apex * swing


def clearance_error(actual, desired, length_scale):
    # Crucially evaluated while loaded too; no event/contact eligibility gate.
    return huber((actual - desired) / length_scale).mean(dim=1)


def support_error(loads, desired_swing, nominal_load):
    normalized = loads.clamp_min(0) / nominal_load
    return (desired_swing * normalized.clamp(max=2).square()
            + (1 - desired_swing) * torch.relu(.2 - normalized).square()).mean(dim=1)


def corridor_error(dx, reference, desired_swing, stance_tolerance, swing_tolerance, length_scale):
    tolerance = stance_tolerance + (swing_tolerance - stance_tolerance) * desired_swing
    return huber(torch.relu((dx - reference).abs() - tolerance) / length_scale).mean(dim=1)
