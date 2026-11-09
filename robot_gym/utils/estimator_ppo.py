"""Small RSL-RL 5.5.1 extension; native PPO still owns storage and policy updates."""

import copy
from time import perf_counter
import torch
from rsl_rl.algorithms import PPO

from .velocity_estimator import HistoryMLP, VelocityInput


class TrainingBudgetReached(RuntimeError):
    pass


def assert_frozen_parent(parent, current):
    def compare(a, b):
        if torch.is_tensor(a):
            assert torch.equal(a.cpu(), b.detach().cpu()), "Frozen parent tensor changed"
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                compare(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for left, right in zip(a, b):
                compare(left, right)
        else:
            assert a == b, "Frozen parent optimizer setting changed"
    for name in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict"):
        compare(parent[name], current[name])


class EstimatorPPO(PPO):
    def __init__(self, *args, estimator_config, estimator_spec, observation_limit, **kwargs):
        super().__init__(*args, **kwargs)
        if self.is_multi_gpu or self.actor.is_recurrent or self.rnd:
            raise ValueError("Velocity estimation currently supports single-device MLP PPO without RND")
        self.estimator_config = copy.deepcopy(estimator_config)
        self.estimator_spec = copy.deepcopy(estimator_spec)
        c = self.estimator_config
        self.stage = c["stage"]
        self.estimator = HistoryMLP(len(estimator_spec["feature_indices"]), c["history_length"],
                                    c["hidden_dims"]).to(self.device)
        self.velocity_input = VelocityInput(self.estimator, estimator_spec, observation_limit).to(self.device)
        self.estimator_optimizer = torch.optim.Adam(self.estimator.parameters(), lr=c["learning_rate"])
        self.stage_counts = dict(warmup_collections=0, finetune_updates=0, supervised_steps=0)
        self.samples = []
        self.allow_estimator_init = False
        self.requested_stage = None
        self.inference_seconds = 0.0
        self.training_wall_s = 0.0
        self.budget_seconds = None
        self.budget_started = None

    def act(self, obs):
        # Clone before env.step: targets and histories can otherwise alias reset buffers.
        effective = obs.clone()
        self.samples.append(effective.select("estimator_history", "velocity_target").clone())
        if self.stage == "warmup":
            actions = self.actor(effective).detach()
            # Native runner logs output_std from the current distribution object.
            # Populate it without drawing an action or updating any learning state.
            self.actor.distribution.update(actions)
            return actions
        start = perf_counter()
        effective["policy"], _ = self.velocity_input(effective["estimator_history"])
        self.inference_seconds += perf_counter() - start
        return super().act(effective)

    def process_env_step(self, obs, rewards, dones, extras):
        if self.stage != "warmup":
            super().process_env_step(obs, rewards, dones, extras)

    def compute_returns(self, obs):
        if self.stage != "warmup":
            super().compute_returns(obs)

    def train_mode(self):
        super().train_mode()
        if self.stage == "warmup":
            self.actor.eval()
            self.critic.eval()
        self.estimator.eval()

    def update(self):
        if self.budget_seconds is not None:
            elapsed = perf_counter() - self.budget_started
            if self.training_wall_s + elapsed >= self.budget_seconds:
                raise TrainingBudgetReached("Combined estimator training wall-time cap reached")
        # Estimator is unchanged throughout collection AND PPO consumption. PPO sees
        # precisely the frozen effective observations used for old log probabilities.
        result = {} if self.stage == "warmup" else super().update()
        self.stage_counts["warmup_collections" if self.stage == "warmup" else "finetune_updates"] += 1
        data = torch.cat(self.samples, dim=0)
        self.samples.clear()
        if self.symmetry and self.symmetry.use_data_augmentation:
            data, _ = self.symmetry.data_augmentation_func(env=self.symmetry.env, obs=data, actions=None)
        # Native runner collects under inference_mode; clone here into ordinary tensors.
        history, target = data["estimator_history"].clone(), data["velocity_target"].clone()
        start = perf_counter()
        with torch.no_grad():
            error = self.estimator(history) - target
            for i, axis in enumerate("xyz"):
                result[f"estimator_rmse_{axis}"] = error[:, i].square().mean().sqrt().item()
                result[f"estimator_bias_{axis}"] = error[:, i].mean().item()
        c = self.estimator_config
        for _ in range(c["gradient_steps"]):
            ids = torch.randint(len(history), (min(c["batch_size"], len(history)),), device=self.device)
            loss = (self.estimator(history[ids]) - target[ids]).square().mean()
            self.estimator_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.estimator.parameters(), 1.0)
            self.estimator_optimizer.step()
            self.stage_counts["supervised_steps"] += 1
        result["estimator_fit_s"] = perf_counter() - start
        result["estimator_dispatch_s"] = self.inference_seconds
        self.inference_seconds = 0.0
        # Next act() recomputes from runner-retained history with the new weights.
        return result

    def save(self):
        data = super().save()
        data["velocity_estimator"] = dict(version=1, config=self.estimator_config, spec=self.estimator_spec,
            model=self.estimator.state_dict(), optimizer=self.estimator_optimizer.state_dict(),
            stage=self.stage, counters=self.stage_counts.copy(),
            training_wall_s=self.training_wall_s + (perf_counter() - self.budget_started if self.budget_started else 0.))
        return data

    def load(self, data, load_cfg=None, strict=True):
        if load_cfg is not None and not all(load_cfg.get(k) for k in ("actor", "critic", "optimizer", "iteration")):
            raise ValueError("Estimator initialization and resume require the full parent learning state")
        iteration = super().load(data, load_cfg, strict)
        saved = data.get("velocity_estimator")
        if saved is None:
            if not self.allow_estimator_init or self.stage != "warmup":
                raise ValueError("Actor-only checkpoint: use explicit --estimator_init --estimator_stage warmup")
            normalizer = self.actor.obs_normalizer
            if hasattr(normalizer, "mean"):
                ids = self.estimator_spec["feature_indices"]
                self.estimator.feature_mean.copy_(normalizer.mean[ids])
                self.estimator.feature_std.copy_((normalizer.std[ids] + normalizer.eps).clamp_min(1e-2))
        else:
            if self.allow_estimator_init:
                raise ValueError("--estimator_init cannot discard an existing estimator; omit it to resume")
            if saved["version"] != 1 or saved["spec"] != self.estimator_spec:
                raise ValueError("Saved estimator feature contract differs from this task")
            if saved['config']['stage'] != saved['stage']:
                raise ValueError("Checkpoint estimator stage metadata disagrees")
            if self.stage != saved['stage'] and self.requested_stage != self.stage:
                raise ValueError("Changing estimator stage requires explicit --estimator_stage")
            for key in ("history_length", "hidden_dims", "learning_rate", "batch_size", "gradient_steps", "finetune_learning_rate"):
                if saved["config"][key] != self.estimator_config[key]:
                    raise ValueError(f"Saved estimator {key} differs; cannot resume")
            self.estimator.load_state_dict(saved["model"], strict=True)
            self.estimator_optimizer.load_state_dict(saved["optimizer"])
            self.stage_counts = saved["counters"].copy()
            self.training_wall_s = saved.get("training_wall_s", 0.)
        return iteration

    def start_stage(self):
        """Apply declared fine-tuning LR only after exact native reload verification."""
        if self.stage == "finetune":
            self.learning_rate = self.estimator_config["finetune_learning_rate"]
            self.schedule = "fixed"
            for group in self.optimizer.param_groups:
                group["lr"] = self.learning_rate
        self.train_mode()
