import os
import copy
import yaml
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple, Type

from rsl_rl.runners import OnPolicyRunner

from robot_gym import ROBOT_GYM_ROOT_DIR
from robot_gym.envs.base.base_task import BaseTask
from robot_gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotCfgPPO,
)

from robot_gym.utils.helpers import (
    get_args,
    update_cfg_from_args,
    class_to_dict,
    update_class_from_dict,
    get_load_path,
    set_seed,
)


def restore_early_walking_config(cfg, saved):
    """Translate the two walking packages written before stand sampling and body wrenches."""
    dr = saved.get("domain_rand", {})
    if (cfg.asset.name not in ("dodo", "go2") or cfg.control.control_type != "P"
            or "stand_command_probability" in saved["commands"]
            or "push_force_range" in dr or "randomize_action_delay" in dr
            or not {"push_interval_s", "max_push_vel_xy"} <= dr.keys()):
        return
    cfg.commands.stand_command_probability = 0.0
    cfg.commands.resampling_time_range = None
    cfg.commands.linear_deadzone = cfg.commands.yaw_deadzone = cfg.commands.stand_threshold = 0.1
    cfg.domain_rand.randomize_action_delay = False
    cfg.domain_rand.action_delay_steps_range = [0, 0]
    # The old "push" wrote a joint torque after stepping. P control replaced
    # it before the next physics step, so these runs had no physical wrench.
    cfg.domain_rand.push_robots = False
    cfg.init_state.relative_joint_position_noise = 0.1
    cfg.init_state.joint_position_noise = 0.0
    cfg.rewards.legacy_walking = True


class TaskRegistry:
    """
    Central registry for environments and training configs.
    """

    def __init__(self):
        self.task_classes: dict[str, Type[BaseTask]] = {}
        self.env_cfgs: dict[str, LeggedRobotCfg] = {}
        self.train_cfgs: dict[str, LeggedRobotCfgPPO] = {}

    def register(
        self,
        name: str,
        task_class: Type[BaseTask],
        env_cfg: LeggedRobotCfg,
        train_cfg: LeggedRobotCfgPPO,
    ):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg

    def get_task_class(self, name: str) -> Type[BaseTask]:
        if name not in self.task_classes:
            raise ValueError(f"Task '{name}' is not registered.")
        return self.task_classes[name]

    def get_cfgs(self, name: str) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]:
        if name not in self.env_cfgs or name not in self.train_cfgs:
            raise ValueError(f"Configs for task '{name}' not found.")

        env_cfg = copy.deepcopy(self.env_cfgs[name])
        train_cfg = copy.deepcopy(self.train_cfgs[name])

        if hasattr(train_cfg, "seed"):
            env_cfg.seed = train_cfg.seed

        return env_cfg, train_cfg

    def load_saved_run(self, args, *, replay):
        """Read the selected checkpoint's configuration before applying runtime options."""
        defaults, training_defaults = self.get_cfgs(args.task)
        selected_run = args.load_run
        if replay and selected_run is None and args.run_name is not None:
            selected_run = args.run_name
            print("Using --run_name as a replay selector; prefer --load_run.", flush=True)
        checkpoint = Path(get_load_path(
            Path(ROBOT_GYM_ROOT_DIR) / "logs" / (args.experiment_name or training_defaults.runner.experiment_name),
            selected_run if selected_run is not None else -1,
            args.checkpoint if args.checkpoint is not None else -1,
        )).resolve()
        config_path = Path(getattr(args, "reference_config", None) or checkpoint.with_name("config.yaml")).resolve()
        with config_path.open(encoding="utf-8") as stream:
            saved = yaml.safe_load(stream)
        if not isinstance(saved, dict) or not all(isinstance(saved.get(k), dict) for k in ("env_cfg", "train_cfg")):
            raise ValueError(f"Expected env_cfg and train_cfg mappings in {config_path}")
        if saved.get("task", args.task) != args.task:
            raise ValueError(f"Checkpoint task {saved['task']!r} conflicts with --task {args.task!r}")
        asset = saved["env_cfg"].get("asset", {})
        if asset.get("name") != defaults.asset.name:
            raise ValueError(f"Saved asset {asset.get('name')!r} conflicts with task {args.task!r}")
        saved_estimator = saved["env_cfg"].get("velocity_estimator")
        has_estimator = isinstance(saved_estimator, dict) and saved_estimator.get("enabled", False)
        if has_estimator and saved["train_cfg"].get("runner", {}).get("obs_groups") != {
            "actor": ["policy"], "critic": ["critic"]
        }:
            raise ValueError("Saved estimator requires separate policy and clean critic groups")
        for key in ("num_observations", "num_actions", "num_privileged_obs"):
            expected = getattr(defaults.env, key)
            if key == "num_observations":
                expected = self.get_task_class(args.task).replay_observation_dim(saved["env_cfg"], expected)
            if key == "num_privileged_obs" and has_estimator and expected is None:
                expected = defaults.env.num_observations
                if key not in saved["env_cfg"]["env"]:
                    raise ValueError("Saved estimator is missing its clean critic dimension")
            if saved["env_cfg"].get("env", {}).get(key, expected) != expected:
                raise ValueError(f"Saved {key} conflicts with the registered {args.task} interface")
        restore = getattr(self.get_task_class(args.task), "restore_saved_config", None)
        if restore is not None:
            env_cfg, train_cfg = restore(saved)
        else:
            # Other robots retain their existing optional-field import behavior.
            env_cfg, train_cfg = defaults, training_defaults
            update_class_from_dict(env_cfg, saved["env_cfg"])
            update_class_from_dict(train_cfg, saved["train_cfg"])
            # The saved scale names define the reward function. Inheriting a newly
            # added term here would change the objective on the first resumed step.
            env_cfg.rewards.scales = SimpleNamespace(**saved["env_cfg"]["rewards"]["scales"])
            restore_early_walking_config(env_cfg, saved["env_cfg"])
            # Early Dodo/Go2 saves lost this name when RSL-RL consumed the dict.
            # Restore only the identifiable native Gaussian; keep all saved values.
            distribution = train_cfg.actor.distribution_cfg
            if (isinstance(distribution, dict) and set(distribution) == {"init_std", "std_type"}
                    and distribution["std_type"] == "scalar"):
                state = torch.load(checkpoint, map_location="cpu", weights_only=False)["actor_state_dict"]
                if [key for key in state if key.startswith("distribution.")] == ["distribution.std_param"]:
                    distribution["class_name"] = "GaussianDistribution"
                    print("Restored legacy GaussianDistribution name from saved std_param; other settings unchanged.", flush=True)
        args.resolved_config_path = str(config_path)
        return env_cfg, train_cfg, checkpoint

    def resolve_replay(self, args):
        if getattr(args, "estimator_init", False):
            raise ValueError("--estimator_init is training-only; replay requires saved estimator weights")
        env_cfg, train_cfg, checkpoint = self.load_saved_run(args, replay=True)
        args._replay_restored = True
        args.resolved_checkpoint = str(checkpoint)
        update_cfg_from_args(env_cfg, train_cfg, args)
        if args.episode_length_s is not None:
            env_cfg.env.episode_length_s = args.episode_length_s
        env_cfg.seed = args.seed if args.seed is not None else train_cfg.seed
        train_cfg.runner.resume = True
        print(f"Resolved checkpoint: {checkpoint}", flush=True)
        return env_cfg, train_cfg, checkpoint

    def make_env(
        self,
        name: str,
        args=None,
        env_cfg: LeggedRobotCfg | None = None,
        initial_command=None,
    ) -> Tuple[BaseTask, LeggedRobotCfg]:
        """Build and seed the task, then pin an optional command before the first reset."""

        if args is None:
            args = get_args()

        task_class = self.get_task_class(name)

        if env_cfg is None:
            env_cfg, train_cfg = self.get_cfgs(name)
        else:
            env_cfg = copy.deepcopy(env_cfg)
            train_cfg = None

        # Apply CLI overrides
        env_cfg, train_cfg = update_cfg_from_args(env_cfg, train_cfg, args)
        if getattr(args, "training_diagnostics", False):
            env_cfg.env.record_command_families = True

        if getattr(args, "seed", None) is not None:
            env_cfg.seed = args.seed

        if getattr(env_cfg, "seed", None) is not None:
            set_seed(env_cfg.seed)

        env = task_class(
            cfg=env_cfg,
            sim_params=env_cfg.sim,
            sim_device=args.sim_device,
            headless=args.headless,
        )

        # Claim external command ownership before reset's sampling and zero-action tick.
        if initial_command is not None:
            env.set_fixed_command(initial_command)
        env.reset()
        return env, env_cfg

    def make_alg_runner(
        self,
        env: BaseTask,
        name: str | None = None,
        args=None,
        train_cfg: LeggedRobotCfgPPO | None = None,
        log_root: str | None = "default",
        save_config: bool = True,
    ) -> Tuple[OnPolicyRunner, LeggedRobotCfgPPO]:
        """Construct native PPO and restore the selected learning state before any updates."""

        if args is None:
            args = get_args()

        # Load config
        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be provided.")
            _, train_cfg = self.get_cfgs(name)
        else:
            train_cfg = copy.deepcopy(train_cfg)
            if name is not None:
                print(f"'train_cfg' provided -> ignoring 'name={name}'")

        # Apply CLI overrides
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)
        if save_config:
            type(env).validate_training(args, env.cfg, train_cfg)

        effective_run_name = (
            train_cfg.runner.run_name
            if train_cfg.runner.run_name
            else train_cfg.runner.experiment_name
        )

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_folder = f"{effective_run_name}_{timestamp}"

        if log_root == "default":
            log_root = os.path.join(
                ROBOT_GYM_ROOT_DIR,
                "logs",
                train_cfg.runner.experiment_name,
            )
        log_dir = os.path.join(log_root, run_folder) if log_root is not None else None

        train_cfg_dict = class_to_dict(train_cfg)

        runner_cfg = train_cfg_dict.pop("runner", {})

        for key, value in runner_cfg.items():
            train_cfg_dict.setdefault(key, value)

        train_cfg_dict.setdefault("obs_groups", {"actor": ["policy"], "critic": ["policy"]})
        train_cfg_dict.setdefault("multi_gpu", False)
        train_cfg_dict.setdefault("logger", "tensorboard")
        train_cfg_dict.setdefault("torch_compile_mode", None)
        train_cfg_dict["run_name"] = effective_run_name

        from robot_gym.utils.velocity_estimator import enabled
        if enabled(env.cfg):
            groups = train_cfg_dict["obs_groups"]
            if groups.get("actor") != ["policy"] or groups.get("critic") != ["critic"]:
                raise ValueError("Estimator tasks need explicit policy actor and separate truth critic groups")
            train_cfg_dict["algorithm"].update(
                class_name="robot_gym.utils.estimator_ppo:EstimatorPPO",
                estimator_config=copy.deepcopy(env.cfg.velocity_estimator),
                estimator_spec=env.velocity_estimator_spec(),
                observation_limit=env.cfg.normalization.clip_observations)

        if str(train_cfg_dict.get("logger", "")).lower() == "wandb":
            train_cfg_dict["logger"] = {"class_name": "WandbLogWriter", "project_name": train_cfg.runner.wandb_project}

        resume_path = None

        if getattr(args, "resolved_checkpoint", None) is not None:
            resume_path = args.resolved_checkpoint
        elif train_cfg.runner.resume:
            if log_root is None:
                raise ValueError("Cannot resume when log_root is None.")

            resume_path = get_load_path(
                log_root,
                load_run=train_cfg.runner.load_run,
                checkpoint=train_cfg.runner.checkpoint,
            )

        if enabled(env.cfg):
            if save_config and (resume_path is None or not args.resume):
                raise ValueError("Estimator training requires a full parent checkpoint with --resume")
            if not save_config and (resume_path is None or getattr(args, "estimator_init", False)):
                raise ValueError("Estimator inference requires a saved estimator checkpoint; --estimator_init is training-only")

        if not save_config:
            log_dir = None  # Inference uses the selected run without creating a new training log.

        runner = OnPolicyRunner(
            env=env,
            train_cfg=train_cfg_dict,
            log_dir=log_dir,
            device=args.rl_device,
        )

        runner.add_git_repo_to_log(__file__)

        runner.checkpoint_path = resume_path
        runner.source_config_path = getattr(args, "resolved_config_path", None)
        if enabled(env.cfg):
            runner.alg.allow_estimator_init = getattr(args, "estimator_init", False)
            runner.alg.requested_stage = getattr(args, "estimator_stage", None)
        env.setup_runner(runner)
        if resume_path is not None:
            print(f"Loading model from: {resume_path}")
            runner.load(resume_path, load_cfg=train_cfg.runner.checkpoint_load_cfg if save_config else None)
            # Task-state loading may change commands/curriculum; never issue the first
            # action from a TensorDict captured before loading.
            env.compute_observations()

        if save_config:
            env.finalize_runner_loading(runner, train_cfg, args)
            if enabled(env.cfg):
                runner.alg.start_stage()
                if runner.alg.stage == "finetune":
                    train_cfg.algorithm.learning_rate = runner.alg.learning_rate
                    train_cfg.algorithm.schedule = "fixed"

        if save_config and args.training_diagnostics:
            env.install_training_diagnostics(runner, log_dir)

        # save config as yaml in log_dir for reproducibility
        if save_config and log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)

            full_cfg = {
                "task": args.task,
                "train_cfg": class_to_dict(train_cfg),
            }

            if hasattr(env, "cfg"):
                full_cfg["env_cfg"] = class_to_dict(env.cfg)

            full_cfg = self._make_yaml_safe(full_cfg)

            config_path = os.path.join(log_dir, "config.yaml")

            with open(config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    full_cfg,
                    f,
                    sort_keys=False,
                    default_flow_style=False,
                    indent=2,
                )

            print(f"Saved config to: {config_path}")

        return runner, train_cfg

    def _make_yaml_safe(self, obj):
        if isinstance(obj, dict):
            return {str(k): self._make_yaml_safe(v) for k, v in obj.items()}

        if isinstance(obj, (list, tuple)):
            return [self._make_yaml_safe(v) for v in obj]

        if isinstance(obj, np.generic):
            return obj.item()

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if torch.is_tensor(obj):
            return obj.detach().cpu().tolist()

        return obj


task_registry = TaskRegistry()
