"""Causal, policy-rate velocity estimation. Inference needs only PyTorch.

Features are task-declared, scaled actor inputs before actor normalization. History
is oldest first; the first measured frame pads an episode, never truth velocity.
"""

import copy
import torch
from torch import nn


def default_config():
    return dict(enabled=True, stage="warmup", history_length=5, hidden_dims=[128, 128, 128],
                learning_rate=1e-3, batch_size=1024, gradient_steps=16,
                sensor_delay_steps=0, sensor_hold_steps=1, sensor_bias_fraction=0.0,
                finetune_learning_rate=3e-5)


def enabled(cfg):
    value = getattr(cfg, "velocity_estimator", None)
    return isinstance(value, dict) and value.get("enabled", False)


def configure(env_cfg, args, train_cfg=None):
    """Explicit initialization differs from restoring an already trained pair."""
    initialize = getattr(args, "estimator_init", False)
    if initialize and train_cfg is not None:
        groups = train_cfg.runner.obs_groups
        if groups == {"actor": ["policy"], "critic": ["policy"]}:
            # Same parent critic width, but the critic keeps simulator truth during estimation.
            train_cfg.runner.obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    if env_cfg is None:
        return
    if initialize:
        if not enabled(env_cfg):
            env_cfg.velocity_estimator = default_config()
        if env_cfg.env.num_privileged_obs is None:
            env_cfg.env.num_privileged_obs = env_cfg.env.num_observations
    if enabled(env_cfg):
        cfg = env_cfg.velocity_estimator
        missing = default_config().keys() - cfg.keys()
        if missing:
            raise ValueError(f"Incomplete saved estimator configuration: {sorted(missing)}")
        for arg, key in (("estimator_stage", "stage"), ("sensor_delay_steps", "sensor_delay_steps"),
                         ("sensor_hold_steps", "sensor_hold_steps")):
            value = getattr(args, arg, None)
            if value is not None:
                cfg[key] = value
        if cfg["history_length"] < 1 or cfg["sensor_delay_steps"] < 0 or cfg["sensor_hold_steps"] < 1:
            raise ValueError("Invalid estimator history or sensor timing")
        if cfg['stage'] not in ('warmup', 'finetune') or not cfg['hidden_dims'] or min(cfg['hidden_dims']) < 1:
            raise ValueError("Invalid estimator stage or MLP widths")
    elif (getattr(args, "velocity_source", None) == "estimated" or
          any(getattr(args, name, None) is not None
              for name in ("estimator_stage", "sensor_delay_steps", "sensor_hold_steps"))):
        raise ValueError("Estimator options require a saved estimator or explicit training --estimator_init")


def feature_spec(env, blocks, live_fields):
    """Describe scaled non-velocity blocks in actor order, after the three velocity slots."""
    fields, live, start = [], [], 0
    for name, width, units, scale in blocks:
        fields.append(dict(name=name, start=start, stop=start + width, units=units, scale=scale))
        if name in live_fields:
            live.extend(range(start, start + width))
        start += width
    if start + 3 != env.num_obs:
        raise ValueError("Estimator feature widths do not match the actor observation")
    return dict(feature_indices=list(range(3, env.num_obs)), velocity_indices=[0, 1, 2],
                live_feature_indices=live, fields=fields, velocity_scale=float(env.obs_scales.lin_vel),
                target="body-frame linear velocity at authored base-link origin, m/s",
                policy_dt=env.dt, feature_joint_order=list(env.joint_names))


class SensorHistory:
    """One coherent measurement per boundary; getters refresh controller state only.

    Delays and holding affect sensor channels together. Commands, phase and issued
    actions are current controller state. Targets always describe the current tick.
    Reset invalidates only selected rows; next acquisition repeats its first frame.
    """

    def __init__(self, num_envs, spec, cfg, device):
        self.spec, self.cfg = spec, cfg
        self.indices = torch.tensor(spec["feature_indices"], device=device)
        self.live = torch.tensor(spec["live_feature_indices"], device=device)
        size = len(spec["feature_indices"])
        self.history = torch.zeros(num_envs, cfg["history_length"], size, device=device)
        self.delay = torch.zeros(num_envs, cfg["sensor_delay_steps"] + 1, size, device=device)
        self.held = torch.zeros(num_envs, size, device=device)
        self.initial_frame = torch.zeros_like(self.held)
        self.bias = torch.zeros_like(self.held)
        self.valid = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.age = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.tick = -1

    def reset(self, ids):
        self.valid[ids] = False
        self.age[ids] = 0
        self.history[ids] = 0
        self.delay[ids] = 0
        self.held[ids] = 0
        self.bias[ids] = 0

    def observe(self, clean, noise_scale, tick):
        frame = clean.index_select(-1, self.indices)
        fresh = ~self.valid
        boundary = tick != self.tick
        # Selective resets can occur after acquisition at this same global tick.
        # Only reset rows receive a first measurement; other rows do not age.
        advance = torch.ones_like(fresh) if boundary else fresh
        if bool(advance.any()):
            acquire = fresh | (advance & (self.age.remainder(self.cfg["sensor_hold_steps"]) == 0))
            scale = noise_scale.index_select(0, self.indices)
            # No random draws on repeated getters, command changes or checkpoint loads.
            if torch.any(scale != 0):
                if self.cfg["sensor_bias_fraction"]:
                    self.bias[fresh] = (2 * torch.rand_like(frame[fresh]) - 1) * scale * self.cfg["sensor_bias_fraction"]
                measured = frame[acquire] + (2 * torch.rand_like(frame[acquire]) - 1) * scale + self.bias[acquire]
            else:
                measured = frame[acquire]
            self.held[acquire] = measured
            self.delay[advance] = torch.cat((self.delay[advance, 1:], self.held[advance, None]), dim=1)
            self.delay[fresh] = self.held[fresh, None]
            received = self.delay[:, 0].clone()
            received[:, self.live] = frame[:, self.live]
            self.history[advance] = torch.cat((self.history[advance, 1:], received[advance, None]), dim=1)
            self.history[fresh] = received[fresh, None]
            self.initial_frame[fresh] = received[fresh]
            self.valid[:] = True
            self.age[advance] += 1
            self.tick = tick
        self.history[:, -1, self.live] = frame[:, self.live]
        measured_obs = clean.clone()
        measured_obs[:, self.indices] = self.history[:, -1]
        return measured_obs


class HistoryMLP(nn.Module):
    """Fixed feature statistics plus a linear, unconstrained m/s output."""

    def __init__(self, feature_dim, history_length, hidden_dims):
        super().__init__()
        self.register_buffer("feature_mean", torch.zeros(feature_dim))
        self.register_buffer("feature_std", torch.ones(feature_dim))
        layers, previous = [], feature_dim * history_length
        for width in hidden_dims:
            layers.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width
        layers.append(nn.Linear(previous, 3))
        self.network = nn.Sequential(*layers)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        normalized = (history - self.feature_mean) / self.feature_std
        return self.network(normalized.flatten(1))


class VelocityInput(nn.Module):
    """Build the actor vector exclusively from measured history and an estimate."""

    def __init__(self, estimator, spec, observation_limit):
        super().__init__()
        self.estimator = estimator
        self.observation_limit = float(observation_limit)
        self.velocity_scale = float(spec["velocity_scale"])
        # Reorder [estimate, features] into the task's original actor layout.
        order = list(spec["velocity_indices"]) + list(spec["feature_indices"])
        if sorted(order) != list(range(len(order))) or len(spec["velocity_indices"]) != 3:
            raise ValueError("Estimator spec must partition the actor inputs, with three velocity slots")
        self.register_buffer("order", torch.tensor([order.index(i) for i in range(len(order))]))

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        velocity = self.estimator(history)
        obs = torch.cat((velocity * self.velocity_scale, history[:, -1]), dim=-1)
        return obs.index_select(-1, self.order).clamp(-self.observation_limit, self.observation_limit), velocity


class EstimatedPolicy(nn.Module):
    """Stateless deployment graph: scaled history -> bounded action and velocity."""

    def __init__(self, inputs, actor, action_limit):
        super().__init__()
        self.inputs, self.actor = inputs, actor
        self.action_limit = float(action_limit)

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        obs, velocity = self.inputs(history)
        return self.actor(obs).clamp(-self.action_limit, self.action_limit), velocity


def inference_policy(runner, env, source=None):
    """The common play/evaluation composition; native inference alone is actor-only."""
    source = selected_velocity_source(env.cfg, source)
    env.velocity_source = source
    if not enabled(env.cfg):
        return runner.get_inference_policy(device=env.device)
    runner.alg.eval_mode()
    print(f"Velocity input: {source}; estimator history {env.cfg.velocity_estimator['history_length']} frames", flush=True)
    if source == "truth":
        return runner.get_inference_policy(device=env.device)
    model = EstimatedPolicy(runner.alg.velocity_input, runner.alg.actor.as_jit(),
                            env.cfg.normalization.clip_actions).to(env.device).eval()

    def act(obs):
        actions, velocity = model(obs["estimator_history"])
        env.estimated_velocity = velocity.detach()
        return actions
    return act


def selected_velocity_source(cfg, source=None):
    source = source or ("estimated" if enabled(cfg) else "truth")
    if source not in ("estimated", "truth") or (source == "estimated" and not enabled(cfg)):
        raise ValueError("Estimated inference requires a loaded velocity estimator")
    return source


def deployment_policy(runner, env):
    return EstimatedPolicy(copy.deepcopy(runner.alg.velocity_input), runner.alg.actor.as_jit(),
                           env.cfg.normalization.clip_actions).cpu().eval()
