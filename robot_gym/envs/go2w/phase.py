"""Observable diagonal phase targets and small dimensionless reward kernels."""

import torch


def huber(error):
    magnitude = error.abs()
    return torch.where(magnitude <= 1, .5 * error.square(), magnitude - .5)


def broad_precision_tracking(error, broad_scale, precision_scale, beta, precision_kernel='absolute_exponential'):
    """Dimensionless reward rate; axis weight and policy dt belong to the accumulator."""
    normalized = error / precision_scale
    if precision_kernel == 'absolute_exponential':
        exponent = -normalized.abs()  # Continuous value, with a cusp at zero.
    elif precision_kernel == 'gaussian':  # Saved schema2 semantics, not a training preset.
        exponent = -.5 * normalized.square()
    else:
        raise ValueError(f"Unknown tracking precision kernel: {precision_kernel}")
    return 1 - huber(error / broad_scale) - beta * (-torch.expm1(exponent))


def reference_pose_activation(commands, lateral_full, yaw_full, stepping_fraction):
    """Smooth requested-motion relaxation, including small lateral commands."""
    # A smooth union avoids the derivative kink of max at equal axis demands.
    lateral = (commands[:, 1].abs() / lateral_full).clamp(0, 1)
    yaw = (commands[:, 2].abs() / yaw_full).clamp(0, 1)
    lateral = lateral.square() * (3 - 2 * lateral)
    yaw = yaw.square() * (3 - 2 * yaw)
    relaxation = 1 - (1 - lateral) * (1 - yaw)
    return 1 - (1 - stepping_fraction) * relaxation


def reference_pose_error(actual_legs, reference_legs, scales, commands, cfg):
    """Actual P-joint pose only; wheel angles, actions and support targets are absent."""
    activation = reference_pose_activation(commands, cfg['lateral_full_m_s'],
                                          cfg['yaw_full_rad_s'], cfg['stepping_fraction'])
    return activation * huber((actual_legs - reference_legs) / scales).mean(dim=1)


def demand(commands):
    lateral = ((commands[:, 1].abs() - .01) / .04).clamp(0, 1)
    yaw = ((commands[:, 2].abs() - .10) / .15).clamp(0, 1)
    value = torch.maximum(lateral, yaw)
    return value.square() * (3 - 2 * value)


def clock_observation(phase, commands, mode="unconditional"):
    """Raw clock entries before empirical normalization; the latent clock is unchanged."""
    angle = 2 * torch.pi * phase
    clock = torch.stack((angle.sin(), angle.cos()), dim=-1)
    if mode == "command_demand":
        return demand(commands)[:, None] * clock
    if mode != "unconditional":
        raise ValueError(f"Unknown phase observation mode: {mode}")
    return clock


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


def step_demand(commands):
    """Unsmoothed demand for command-hold eligibility and diagnostic bins."""
    lateral = ((commands[:, 1].abs() - 0.01) / 0.04).clamp(0, 1)
    yaw = ((commands[:, 2].abs() - 0.10) / 0.15).clamp(0, 1)
    return torch.maximum(lateral, yaw)


def project_log_std(optimizer, distribution):
    """Project only raw Gaussian parameters, retaining native Adam moments and LR."""
    if distribution.std_type != "log":
        raise ValueError("Go2-W requires log std")
    def project(*_):
        with torch.no_grad():
            distribution.log_std_param.clamp_(*distribution.log_std_range)
    return optimizer.register_step_post_hook(project)
