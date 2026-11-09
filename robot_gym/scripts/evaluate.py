"""Evaluate a saved policy against the selected task's command schedules."""

import robot_gym.envs  # Register task-specific CLI arguments.
from robot_gym.utils.helpers import get_args


if __name__ == "__main__":
    args = get_args()
    if args.estimator_screen:
        from robot_gym.utils.estimator_evaluation import evaluate_estimator
        evaluate_estimator(args)
    else:
        from robot_gym.envs.go2w.evaluate import evaluate
        evaluate(args)
