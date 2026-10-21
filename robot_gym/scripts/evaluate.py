"""Compatibility entry point for Go2-W evaluate."""

from robot_gym.envs.go2w.evaluate import *  # noqa: F401,F403

if __name__ == "__main__":
    args = get_args()
    if args.estimator_screen:
        from robot_gym.utils.estimator_evaluation import evaluate_estimator
        evaluate_estimator(args)
    else:
        evaluate(args)
