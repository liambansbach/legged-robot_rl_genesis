from robot_gym import ROBOT_GYM_ROOT_DIR, ROBOT_GYM_ENVS_DIR


from robot_gym.envs.base.legged_robot import LeggedRobot


from robot_gym.envs.dodo.dodo_env import DodoEnv
from robot_gym.envs.dodo.dodo_config import DodoCfg, DodoCfgPPO

from robot_gym.envs.go2.go2_env import Go2Env
from robot_gym.envs.go2.go2_config import GO2Cfg, GO2CfgPPO

from robot_gym.envs.go2w.go2w_env import Go2WEnv
from robot_gym.envs.go2w.go2w_config import GO2WCfg, GO2WCfgPPO


from robot_gym.utils.task_registry import task_registry

task_registry.register(
    "dodo",
    DodoEnv,
    DodoCfg(),
    DodoCfgPPO(),
)

task_registry.register(
    "go2",
    Go2Env,
    GO2Cfg(),
    GO2CfgPPO(),
)

task_registry.register(
    "go2w",
    Go2WEnv,
    GO2WCfg(),
    GO2WCfgPPO(),
)
