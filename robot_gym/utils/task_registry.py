import os
import copy
import yaml
import numpy as np
import torch
from datetime import datetime
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
    get_load_path,
    set_seed,
)


class TaskRegistry:
    """
    Central registry for environments and training configs.
    """

    def __init__(self):
        self.task_classes: dict[str, Type[BaseTask]] = {}
        self.env_cfgs: dict[str, LeggedRobotCfg] = {}
        self.train_cfgs: dict[str, LeggedRobotCfgPPO] = {}

    # --------------------------------------------------------------------------
    # Registration
    # --------------------------------------------------------------------------
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

    # --------------------------------------------------------------------------
    # Getters
    # --------------------------------------------------------------------------
    def get_task_class(self, name: str) -> Type[BaseTask]:
        if name not in self.task_classes:
            raise ValueError(f"Task '{name}' is not registered.")
        return self.task_classes[name]

    def get_cfgs(self, name: str) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]:
        if name not in self.env_cfgs or name not in self.train_cfgs:
            raise ValueError(f"Configs for task '{name}' not found.")

        # IMPORTANT: deepcopy to avoid modifying registry defaults
        env_cfg = copy.deepcopy(self.env_cfgs[name])
        train_cfg = copy.deepcopy(self.train_cfgs[name])

        # Sync seed
        try:
            env_cfg.seed = train_cfg.seed
        except Exception:
            pass

        return env_cfg, train_cfg

    # --------------------------------------------------------------------------
    # Simulation Params (Genesis-specific)
    # --------------------------------------------------------------------------
    def _build_sim_params(self, env_cfg: LeggedRobotCfg) -> SimpleNamespace:
        """
        Convert cfg.sim into an object with attribute access.

        Required because BaseTask expects:
            self.sim_params.dt
        """
        sim_dict = class_to_dict(env_cfg.sim)
        return SimpleNamespace(**sim_dict)

    # --------------------------------------------------------------------------
    # Environment creation
    # --------------------------------------------------------------------------
    def make_env(
        self,
        name: str,
        args=None,
        env_cfg: LeggedRobotCfg | None = None,
    ) -> Tuple[BaseTask, LeggedRobotCfg]:
        """ Creates an environment either from a registered name or from the provided config file.

        Args:
            name (string): Name of a registered env.
            args (Args, optional): command line arguments. If None get_args() will be called. Defaults to None.
            env_cfg (Dict, optional): Environment config file used to override the registered config. Defaults to None.

        Raises:
            ValueError: Error if no registered env corresponds to 'name' 

        Returns:
            The created environment
            Dict: the corresponding config file
        """

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

        # Seed handling
        seed = None
        if train_cfg is not None and hasattr(train_cfg, "seed"):
            seed = train_cfg.seed
        elif hasattr(env_cfg, "seed"):
            seed = env_cfg.seed

        if seed is not None:
            set_seed(seed)

        # Build Genesis-compatible sim params
        sim_params = self._build_sim_params(env_cfg)

        env = task_class(
            cfg=env_cfg,
            sim_params=sim_params,
            sim_device=args.sim_device,
            headless=args.headless,
        )

        return env, env_cfg

    # --------------------------------------------------------------------------
    # Algorithm runner creation
    # --------------------------------------------------------------------------
    def make_alg_runner(
        self,
        env: BaseTask,
        name: str | None = None,
        args=None,
        train_cfg: LeggedRobotCfgPPO | None = None,
        log_root: str | None = "default",
        save_config: bool = True,
    ) -> Tuple[OnPolicyRunner, LeggedRobotCfgPPO]:
        """ Creates the training algorithm  either from a registered name or from the provided config file.

        Args:
            env (BaseTask): The environment to train on.
            name (string, optional): Name of a registered env. If None, the config file will be used instead. Defaults to None.
            args (Args, optional): command line arguments. If None get_args() will be called. Defaults to None.
            train_cfg (Dict, optional): Training config file. If None 'name' will be used to get the config file. Defaults to None.
            log_root (str, optional): Logging directory for wandb. Set to 'None' to avoid logging (at test time for example). 
                                      Logs will be saved in <log_root>/<date_time>_<run_name>. Defaults to "default"=<path_to_LEGGED_GYM>/logs/<experiment_name>.

        Raises:
            ValueError: Error if neither 'name' or 'train_cfg' are provided
            Warning: If both 'name' or 'train_cfg' are provided 'name' is ignored

        Returns:
            PPO: The created algorithm
            Dict: the corresponding config file
        """

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
        elif log_root is None:
            log_dir = None

        if log_root is not None:
            log_dir = os.path.join(log_root, run_folder) 

        train_cfg_dict = class_to_dict(train_cfg)

        runner_cfg = train_cfg_dict.pop("runner", {})
        train_cfg_dict = self._adapt_train_cfg_for_rsl_rl(train_cfg_dict)

        for key, value in runner_cfg.items():
            train_cfg_dict.setdefault(key, value)

        train_cfg_dict.setdefault("obs_groups", {"actor": ["policy"], "critic": ["policy"]})
        train_cfg_dict.setdefault("empirical_normalization", True)
        train_cfg_dict.setdefault("multi_gpu", False)
        train_cfg_dict.setdefault("logger", "tensorboard")
        train_cfg_dict.setdefault("torch_compile_mode", None)
        train_cfg_dict["run_name"] = effective_run_name

        if str(train_cfg_dict.get("logger", "")).lower() == "wandb":
            self._patch_wandb_writer_for_base_config()

        resume_path = None

        if train_cfg.runner.resume:
            if log_root is None:
                raise ValueError("Cannot resume when log_root is None.")

            resume_path = get_load_path(
                log_root,
                load_run=train_cfg.runner.load_run,
                checkpoint=train_cfg.runner.checkpoint,
            )

        runner = OnPolicyRunner(
            env=env,
            train_cfg=train_cfg_dict,
            log_dir=log_dir,
            device=args.rl_device,
        )

        runner.log_dir = log_dir

        if resume_path is not None: 
            print(f"Loading model from: {resume_path}")
            runner.load(resume_path)


        # save config as yaml in log_dir for reproducibility
        if save_config and log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)

            full_cfg = {
                "train_cfg": class_to_dict(train_cfg),
            }

            if hasattr(env, "cfg"):
                full_cfg["env_cfg"] = class_to_dict(env.cfg)

            full_cfg = self._make_yaml_safe(full_cfg)

            config_path = os.path.join(log_dir, "config.yaml")

            with open(config_path, "w") as f:
                yaml.safe_dump(
                    full_cfg,
                    f,
                    sort_keys=False,
                    default_flow_style=False,
                    indent=2,
                )

            print(f"Saved config to: {config_path}")


        return runner, train_cfg

    def _adapt_train_cfg_for_rsl_rl(self, train_cfg_dict: dict) -> dict:
        """
        Convert this repo's actor/critic config shape to the ActorCritic
        policy config expected by the installed rsl-rl OnPolicyRunner.
        """
        if "policy" not in train_cfg_dict:
            actor_cfg = train_cfg_dict.pop("actor", {})
            critic_cfg = train_cfg_dict.pop("critic", {})

            distribution_cfg = actor_cfg.get("distribution_cfg", {})

            train_cfg_dict["policy"] = {
                "class_name": "ActorCritic",
                "actor_hidden_dims": actor_cfg.get("hidden_dims", [512, 256, 128]),
                "critic_hidden_dims": critic_cfg.get(
                    "hidden_dims",
                    actor_cfg.get("hidden_dims", [512, 256, 128]),
                ),
                "activation": actor_cfg.get(
                    "activation",
                    critic_cfg.get("activation", "elu"),
                ),
                "init_noise_std": distribution_cfg.get("init_std", 1.0),
                "noise_std_type": distribution_cfg.get("std_type", "scalar"),
            }

        algorithm_cfg = train_cfg_dict.get("algorithm", {})
        # Some config keys are used by newer/custom policy builders but are not
        # accepted by rsl-rl 5.x PPO directly.
        algorithm_cfg.pop("share_cnn_encoders", None)

        return train_cfg_dict

    def _patch_wandb_writer_for_base_config(self):
        """
        rsl-rl 5.x serializes env_cfg with dataclasses.asdict(), but this repo
        uses BaseConfig-style class configs. Patch only the config logging hook
        so wandb works without modifying the installed rsl-rl package.
        """
        from rsl_rl.utils import wandb_utils

        if getattr(wandb_utils.WandbSummaryWriter, "_robot_gym_config_patch", False):
            return

        original_init = wandb_utils.WandbSummaryWriter.__init__

        def __init__(writer, log_dir: str, flush_secs: int, cfg):
            original_init(writer, log_dir, flush_secs, cfg)
            if wandb_utils.wandb.run is not None:
                wandb_utils.wandb.run.name = os.path.basename(log_dir)

        def store_config(writer, env_cfg, runner_cfg, alg_cfg, policy_cfg):
            wandb_utils.wandb.config.update({"runner_cfg": runner_cfg})
            wandb_utils.wandb.config.update({"policy_cfg": policy_cfg})
            wandb_utils.wandb.config.update({"alg_cfg": alg_cfg})
            wandb_utils.wandb.config.update({"env_cfg": class_to_dict(env_cfg)})

        wandb_utils.WandbSummaryWriter.__init__ = __init__
        wandb_utils.WandbSummaryWriter.store_config = store_config
        wandb_utils.WandbSummaryWriter._robot_gym_config_patch = True
    
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


#global task registry
task_registry = TaskRegistry()
