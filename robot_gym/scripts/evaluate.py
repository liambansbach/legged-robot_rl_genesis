"""Compatibility entry point for Go2-W evaluate."""

from robot_gym.envs.go2w.evaluate import *  # noqa: F401,F403

if __name__ == "__main__":
    evaluate(get_args())
