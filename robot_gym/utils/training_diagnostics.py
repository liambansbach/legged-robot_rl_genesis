"""Read-only instrumentation of installed RSL-RL 5.5 PPO interfaces; no PPO copy.

Wraps the existing rollout/batch/log-prob/KL calls once, returns their exact results,
and never performs an extra actor forward or random draw. Unsupported paths fail.
"""

import inspect
import importlib.metadata
import json
from pathlib import Path

import torch

from robot_gym.utils.diagnostics import json_safe, sha256


def verify_resume_state(runner, checkpoint):
    """Read-only, exact comparison with the parent; no forwards or RNG draws."""
    if importlib.metadata.version("rsl-rl-lib") != "5.5.1":
        raise ValueError("Go2-W full-state continuation is validated for RSL-RL 5.5.1")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)

    def compare(expected, actual, path):
        if torch.is_tensor(expected):
            equal = (
                torch.is_tensor(actual)
                and expected.dtype == actual.dtype
                and torch.equal(expected, actual.detach().cpu())
            )
        elif isinstance(expected, dict):
            equal = isinstance(actual, dict) and expected.keys() == actual.keys()
            if equal:
                for key in expected:
                    compare(expected[key], actual[key], f"{path}.{key}")
        elif isinstance(expected, (list, tuple)):
            equal = isinstance(actual, (list, tuple)) and len(expected) == len(actual)
            if equal:
                for index, (a, b) in enumerate(zip(expected, actual)):
                    compare(a, b, f"{path}.{index}")
        else:
            equal = expected == actual
        if not equal:
            raise ValueError(
                f"Loaded continuation state differs from checkpoint: {path}"
            )

    # Native save includes both normalizers (all buffers/counters) in the models.
    state = runner.alg.save()
    for key in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict"):
        compare(saved[key], state[key], key)
    compare(saved["iter"], runner.current_learning_iteration, "iteration")
    learning_rate = saved["optimizer_state_dict"]["param_groups"][0]["lr"]
    compare(learning_rate, runner.alg.learning_rate, "algorithm.learning_rate")
    return {
        "exact_state_match": True,
        "rsl_rl_version": "5.5.1",
        "source_iteration": saved["iter"],
        "loaded_learning_rate": learning_rate,
        "optimizer_learning_rates": [
            g["lr"] for g in runner.alg.optimizer.param_groups
        ],
        "actor_and_critic_state_keys": {
            k: list(state[k]) for k in ("actor_state_dict", "critic_state_dict")
        },
        "std_before_update": json_safe(std_parameters(runner.alg.actor.distribution)),
    }


def std_parameters(distribution):
    if (
        distribution.__class__.__name__ != "GaussianDistribution"
        or distribution.std_type != "log"
    ):
        raise ValueError(
            "Diagnostics require state-independent GaussianDistribution(std_type='log')"
        )
    raw = distribution.log_std_param.detach()
    low, high = distribution.log_std_range
    return {
        "raw_log_std": raw.clone(),
        "effective_std": raw.clamp(low, high).exp(),
        "raw_log_std_bounds": [low, high],
        "effective_std_bounds": distribution.std_range,
        "at_or_below_lower_bound": raw <= low,
        "at_or_above_upper_bound": raw >= high,
        "strictly_outside_bounds": (raw < low) | (raw > high),
    }


class TrainingDiagnostics:
    families = ["stand", "straight", "arc", "yaw", "precision", "lateral", "mixed"]

    def __init__(self, runner, env, path):
        self.runner, self.env, self.path = runner, env, Path(path)
        alg = runner.alg
        source = inspect.getsource(type(alg).update)
        required = (
            "get_kl_divergence(batch.old_distribution_params, distribution_params)",
            "actions_log_prob - torch.squeeze(batch.old_actions_log_prob)",
        )
        if type(alg).__name__ != "PPO" or any(s not in source for s in required):
            raise ValueError(
                "Unsupported installed PPO path for diagnostics; inspect its batch/KL interfaces"
            )
        if (
            alg.is_multi_gpu
            or alg.actor.is_recurrent
            or alg.use_mixed_precision
            or hasattr(alg.actor, "_orig_mod")
        ):
            raise ValueError(
                "Diagnostics support eager, single-GPU, float32 MLP PPO only"
            )
        std_parameters(alg.actor.distribution)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise ValueError(f"Refusing to overwrite training diagnostics: {self.path}")
        self.command_families = getattr(env, "diagnostic_command_families", None)
        if self.command_families is None:
            self.command_families = torch.full(
                (env.num_envs,), -1, dtype=torch.long, device=env.device
            )
        self.previous_mean = self.previous_sample = None
        self.previous_valid = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        self.current_batch = None
        self.iteration = runner.current_learning_iteration
        self.reset_aggregates()
        self.source = {
            "ppo_path": inspect.getfile(type(alg)),
            "ppo_sha256": sha256(inspect.getfile(type(alg))),
            "lr_scheduler_bounds": [1e-5, 1e-2],
            "joint_order": env.joint_names,
            "mirror_loss_enabled": bool(alg.symmetry and alg.symmetry.use_mirror_loss),
        }
        self.install(alg)
        env.training_diagnostics = self

    def reset_aggregates(self):
        self.rollout = {}
        self.rewards = {}
        self.kl = []
        self.ppo_clip = []

    def add(self, key, value):
        value = value.detach().float()
        if value.numel() == 0:
            return
        total, count = self.rollout.get(key, (torch.zeros_like(value[0]), 0))
        self.rollout[key] = (total + value.sum(dim=0), count + len(value))

    def targets(self, actions):
        e = self.env
        value = (
            actions.clamp(
                -e.cfg.normalization.clip_actions, e.cfg.normalization.clip_actions
            )
            * e.action_scale
        )
        wheels = e.wheel_action_indices
        value[:, wheels] = value[:, wheels].clamp(
            -e.cfg.control.wheel_velocity_target_limit,
            e.cfg.control.wheel_velocity_target_limit,
        )
        return (
            value  # Nominal P offset cancels in slew; units rad (legs), rad/s (wheels).
        )

    def record_actions(self, actions, mean):
        clip = self.env.cfg.normalization.clip_actions
        self.add("raw_mean", mean)
        self.add("raw_mean_squared", mean.square())
        self.add("sampled_action_clipping_fraction", actions.abs() > clip)
        self.add("deterministic_mean_saturation_fraction", mean.abs() >= clip)
        mean_target, sample_target = (
            self.targets(mean.detach()),
            self.targets(actions.detach()),
        )
        if self.previous_mean is not None:
            ids = self.previous_valid
            self.add(
                "mean_target_slew_squared_per_s2",
                ((mean_target - self.previous_mean)[ids] / self.env.dt).square(),
            )
            self.add(
                "sampled_target_slew_squared_per_s2",
                ((sample_target - self.previous_sample)[ids] / self.env.dt).square(),
            )
        self.previous_mean, self.previous_sample = (
            mean_target.clone(),
            sample_target.clone(),
        )

    def reward(self, raw, commands):
        # raw is the dt-scaled nonterminal reward sum, before only_positive_rewards.
        masks = {
            "all": torch.ones_like(raw, dtype=torch.bool),
            "unlabelled_initial": self.command_families == -1,
        }
        masks.update(
            {name: self.command_families == i for i, name in enumerate(self.families)}
        )
        for name, mask in masks.items():
            values = raw[mask].detach()
            summary = torch.stack(
                (
                    mask.sum(),
                    values.sum(),
                    (values < 0).sum(),
                    (-values.clamp_max(0)).sum(),
                )
            ).float()
            self.rewards[name] = (
                self.rewards.get(name, torch.zeros_like(summary)) + summary
            )

    def install(self, alg):
        original_act, original_step, original_update = (
            alg.act,
            alg.process_env_step,
            alg.update,
        )
        original_batches = alg.storage.mini_batch_generator
        original_prob, original_kl = (
            alg.actor.get_output_log_prob,
            alg.actor.get_kl_divergence,
        )

        def act(obs):
            result = original_act(obs)
            with torch.no_grad():
                self.record_actions(result, alg.actor.output_mean.detach())
            return result

        def process_env_step(obs, rewards, dones, extras):
            result = original_step(obs, rewards, dones, extras)
            self.previous_valid = ~dones.bool()
            return result

        def batches(*args, **kwargs):
            for batch in original_batches(*args, **kwargs):
                self.current_batch = batch
                yield batch
            self.current_batch = None

        def log_prob(actions):
            result = original_prob(actions)
            if self.current_batch is not None:
                with torch.no_grad():
                    ratio = (
                        result - self.current_batch.old_actions_log_prob.squeeze()
                    ).exp()
                    self.ppo_clip.append(
                        ((ratio - 1).abs() > alg.clip_param).float().mean().detach()
                    )
            return result

        def kl(old, new):
            result = original_kl(old, new)
            self.kl.append(result.detach().mean())
            return result

        def update():
            result = original_update()
            self.flush(alg)
            return result

        alg.act, alg.process_env_step, alg.update = act, process_env_step, update
        alg.storage.mini_batch_generator = batches
        alg.actor.get_output_log_prob, alg.actor.get_kl_divergence = log_prob, kl

    def flush(self, alg):
        vectors = {key: total / count for key, (total, count) in self.rollout.items()}
        for key in list(vectors):
            if "squared_per_s2" in key:
                vectors[key.replace("squared_per_s2", "rms_per_s")] = vectors.pop(
                    key
                ).sqrt()
        std = std_parameters(alg.actor.distribution)
        groups = {}
        for name, ids in (
            ("legs", self.env.leg_action_indices),
            ("wheels", self.env.wheel_action_indices),
        ):
            groups[name] = {
                key: value[ids].float().mean()
                for key, value in {**vectors, **std}.items()
                if torch.is_tensor(value) and value.shape == (16,)
            }
        rewards = {}
        for name, values in self.rewards.items():
            count, total, negative, discarded = values.tolist()
            if count:
                rewards[name] = {
                    "sample_count": int(count),
                    "raw_nonterminal_mean": total / count,
                    "negative_fraction": negative / count,
                    "clipped_fraction": negative / count
                    if self.env.cfg.rewards.only_positive_rewards
                    else 0.0,
                    "discarded_negative_magnitude_per_sample": discarded / count
                    if self.env.cfg.rewards.only_positive_rewards
                    else 0.0,
                }
        row = {
            "iteration": self.iteration,
            "source": self.source,
            "action_vectors": vectors,
            "std_parameters_after_update": std,
            "groups": groups,
            "scheduler_kl_per_minibatch": self.kl,
            "ppo_clip_fraction_per_minibatch": self.ppo_clip,
            "learning_rate_after_update": alg.learning_rate,
            "nonterminal_reward": rewards,
            "slew_definition": "Consecutive clipped/scaled targets on sampled rollout states, excluding reset boundaries; mean path is a counterfactual at those same states; no extra policy calls",
        }
        with self.path.open("a") as stream:
            stream.write(json.dumps(json_safe(row), allow_nan=False) + "\n")
        self.iteration += 1
        self.reset_aggregates()
