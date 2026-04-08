"""
grader.py — Standalone grader for Financial Fraud Defender episodes.

Matches openenv.yaml entry:  grader: environment:compute_grade

Usage:
  python grader.py                     # self-test all 3 tasks with rule-based agent
  python grader.py --task easy         # test one task
  python grader.py --task hard --seed 99
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List


def grade(task: str, episode_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Grade a completed episode.

    Args:
        task: "easy" | "medium" | "hard"
        episode_history: list of step dicts, each containing at minimum
                         {"action": int, "is_fraud": bool}  (or "was_fraud")

    Returns:
        {
          "task": str,
          "score": float,        # 0.0 – 1.0
          "passed": bool,
          "metric": str,         # "recall" | "f1" | "composite"
          "threshold": float,
          "details": {tp, fp, fn, tn, precision, recall, f1, fpr, total_steps}
        }
    """
    from environment import compute_grade as _compute
    return _compute(task, episode_history)


def _run_full_episode(task: str, seed: int = 42) -> Dict[str, Any]:
    """Run a complete episode with the rule-based agent and return grading result."""
    from environment import FraudDetectionEnv
    from inference import rule_based_action

    env = FraudDetectionEnv(task=task, seed=seed)
    obs_arr, _ = env.reset(seed=seed)
    obs = obs_arr.tolist()

    while True:
        action = rule_based_action(obs)
        obs_arr, _, terminated, truncated, _ = env.step(action)
        obs = obs_arr.tolist()
        if terminated or truncated:
            break

    return grade(task, env._episode_history)


def _self_test(task: str | None = None, seed: int = 42) -> None:
    tasks = [task] if task else ["easy", "medium", "hard"]
    all_passed = True

    print(f"\n{'Task':<8} {'Metric':<12} {'Score':>7} {'Threshold':>10} {'Status'}")
    print("-" * 50)

    for t in tasks:
        result = _run_full_episode(t, seed=seed)
        status = "PASS ✓" if result["passed"] else "FAIL ✗"
        if not result["passed"]:
            all_passed = False
        print(
            f"{t:<8} {result['metric']:<12} {result['score']:>7.4f} "
            f"{result['threshold']:>10.2f} {status}"
        )
        print(f"         details: {json.dumps(result['details'])}")

    print("-" * 50)
    print(f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grade a fraud detection episode.")
    parser.add_argument("--task", choices=["easy", "medium", "hard"], default=None,
                        help="Task to test (default: all three)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()
    _self_test(task=args.task, seed=args.seed)
