import torch
import genesis as gs

from .legged_robot_config import LeggedRobotCfg


class BaseTask:
    _gs_initialized = False
    _gs_backend = None

    def __init__(self, cfg: LeggedRobotCfg, sim_params, sim_device, headless):
        self.sim_params = sim_params
        self.headless = headless
        self.sim_device = sim_device

        # get device type -> robust to inputs like "cuda:0" or "cpu" (if multiple gpus are avvailable for example)
        device_type = sim_device.split(":")[0]
        backend = gs.cuda if device_type == "cuda" else gs.cpu

        # init genesis
        if not BaseTask._gs_initialized:
            gs.init(
                backend=backend,
                performance_mode=cfg.sim.performance_mode,
                )
            BaseTask._gs_initialized = True
            BaseTask._gs_backend = backend
        elif BaseTask._gs_backend != backend:
            raise RuntimeError(
                f"Genesis already initialized with backend {BaseTask._gs_backend}, "
                f"cannot reinitialize with {backend}."
            )

        self.device: torch.device = gs.device

        self.num_envs = cfg.env.num_envs
        self.num_obs = cfg.env.num_observations
        self.num_actions = cfg.env.num_actions
        self.num_commands = cfg.commands.num_commands
        self.num_privileged_obs = cfg.env.num_privileged_obs

        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        N = self.num_envs
        self.obs_buf = torch.zeros((N, self.num_obs), device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros((N,), device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones((N,), device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros((N,), device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros((N,), device=self.device, dtype=torch.bool)

        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(
                (N, self.num_privileged_obs), device=self.device, dtype=torch.float
            )
        else:
            self.privileged_obs_buf = None

        self.extras = {"observations": {}}

        self.enable_viewer_sync = True
        self.viewer = None

    def get_observations(self):
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset_idx(self, env_ids):
        raise NotImplementedError

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, _, _, extras = self.step( 
            torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False)
        )
        return obs, extras

    def step(self, actions):
        raise NotImplementedError

















