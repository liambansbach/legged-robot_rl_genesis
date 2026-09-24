"""Sagittal Y -> -Y reflection for the Go2-W 56-observation / mixed P/V contract.

RSL-RL receives raw (pre-normalization) TensorDict observations. Both actor and
critic use the same `policy` group. Reflection is only mini-batch augmentation;
it does not alter rollout states, command sampling, or inference actions.
"""

from functools import lru_cache

import torch
from tensordict import TensorDictBase


SIDES = {"FL": "FR", "FR": "FL", "RL": "RR", "RR": "RL"}


def joint_reflection(names):
    """Output joint i takes its opposite-side value, negated only for hip joints."""
    mirrored = [SIDES[name[:2]] + name[2:] for name in names]
    return (
        [names.index(name) for name in mirrored],
        [-1 if name.endswith("_hip_joint") else 1 for name in names],
    )


@lru_cache(maxsize=8)
def _maps(joint_names, leg_names, device):
    expected = {
        f"{side}_{joint}_joint"
        for side in SIDES
        for joint in ("hip", "thigh", "calf", "foot")
    }
    if len(joint_names) != 16 or set(joint_names) != expected:
        raise ValueError(
            "Go2-W symmetry requires the 16 named hip/thigh/calf/foot joints"
        )
    if leg_names != tuple(n for n in joint_names if not n.endswith("_foot_joint")):
        raise ValueError("Go2-W leg observations must follow the runtime P-joint order")
    action_perm, action_sign = joint_reflection(joint_names)
    leg_perm, leg_sign = joint_reflection(leg_names)
    # Same block order as Go2WEnv.compute_observations. Angular velocity is axial.
    blocks = [
        (list(range(3)), [1, -1, 1]),  # body linear velocity (polar)
        (list(range(3)), [-1, 1, -1]),  # body angular velocity (axial)
        (list(range(3)), [1, -1, 1]),  # projected gravity (polar)
        (list(range(3)), [1, -1, -1]),  # commands: vx, vy, yaw
        (leg_perm, leg_sign),  # 12 leg position errors from nominal
        (action_perm, action_sign),  # 16 joint velocities
        (action_perm, action_sign),  # 16 previous actions
    ]
    obs_perm, obs_sign = [], []
    for perm, sign in blocks:
        offset = len(obs_perm)
        obs_perm.extend(offset + i for i in perm)
        obs_sign.extend(sign)
    return (
        torch.tensor(obs_perm, device=device),
        torch.tensor(obs_sign, dtype=torch.int8, device=device),
        torch.tensor(action_perm, device=device),
        torch.tensor(action_sign, dtype=torch.int8, device=device),
    )


def _env_maps(env, device):
    if env.num_obs != 56 or env.num_actions != 16:
        raise ValueError("Go2-W symmetry requires 56 observations and 16 actions")
    names = tuple(env.joint_names)
    legs = tuple(names[i] for i in env.leg_action_indices)
    return _maps(names, legs, device)


def mirror_observations(env, obs):
    """Return a new TensorDict with the reflected policy observation."""
    if not isinstance(obs, TensorDictBase) or set(obs.keys()) != {"policy"}:
        raise ValueError(
            "Go2-W symmetry expects only the TensorDict 'policy' observation group"
        )
    policy = obs["policy"]
    if policy.ndim != 2 or policy.shape[-1] != 56:
        raise ValueError("Go2-W symmetry expects a [B, 56] policy tensor")
    perm, sign, _, _ = _env_maps(env, policy.device)
    mirrored = obs.clone(recurse=False)
    mirrored["policy"] = policy.index_select(-1, perm) * sign
    return mirrored


def mirror_actions(env, actions):
    """Reflect P leg offsets and V wheel targets, preserving the common wheel sign."""
    if actions.ndim != 2 or actions.shape[-1] != 16:
        raise ValueError("Go2-W symmetry expects a [B, 16] action tensor")
    _, _, perm, sign = _env_maps(env, actions.device)
    return actions.index_select(-1, perm) * sign


def sagittal_augmentation(env, obs=None, actions=None):
    """RSL-RL 5.5.1 contract: return [original B; mirrored B], allowing either None."""
    augmented_obs = (
        None if obs is None else torch.cat((obs, mirror_observations(env, obs)), dim=0)
    )
    augmented_actions = (
        None
        if actions is None
        else torch.cat((actions, mirror_actions(env, actions)), dim=0)
    )
    return augmented_obs, augmented_actions
