from robot_gym.envs.go2.go2_env import Go2Env


class Go2WEnv(Go2Env):
    """
    Go2W locomotion environment.

    The first version reuses the Go2 reward extensions. The wheel-specific
    behavior is configured through mixed position/velocity joint control.
    """

    def __init__(self, cfg, sim_params, sim_device, headless):
        super().__init__(cfg, sim_params, sim_device, headless)
