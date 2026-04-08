from __future__ import annotations

import os
import re
import sys
import json
import random
import argparse
from typing import Any, Dict, List, Optional, Tuple

# ---------------- SAFE IMPORTS ----------------
try:
    from environment import FraudDetectionEnv, compute_grade, MERCHANT_CATEGORIES
except ImportError:
    try:
        from environement import FraudDetectionEnv, compute_grade, MERCHANT_CATEGORIES
    except Exception as e:
        print(f"[FATAL ERROR] {e}", flush=True)
        print("[END] success=false steps=0 score=0.00 rewards=", flush=True)
        sys.exit(0)

try:
    from models import Action, TASK_DEFINITIONS
except Exception as e:
    print(f"[FATAL ERROR] {e}", flush=True)
    print("[END] success=false steps=0 score=0.00 rewards=", flush=True)
    sys.exit(0)


ENV_BENCHMARK = "financial-fraud-defender-v1"


# ---------------- RULE BASED ----------------
def rule_based_action(obs: List[float]) -> int:
    (_, _, location_match, txn_last_hour, is_new_merchant,
     risk_score, _, _, velocity_ratio, time_since, card_present,
     _, _, _) = obs

    if risk_score >= 8.0 or velocity_ratio > 6.0:
        return int(Action.BLOCK)

    if velocity_ratio > 3.0 and card_present < 0.5:
        return int(Action.BLOCK)

    suspicious = sum([
        risk_score >= 5.0,
        velocity_ratio > 2.5,
        location_match < 0.5,
        is_new_merchant > 0.5,
        txn_last_hour >= 4,
        time_since < 2.0,
    ])

    if suspicious >= 2:
        return int(Action.VERIFY)

    return int(Action.APPROVE)


# ---------------- EPISODE ----------------
def run_episode(task="easy", seed=None):

    try:
        env = FraudDetectionEnv(task=task, seed=seed)
        obs, _ = env.reset(seed=seed)
        obs = obs.tolist()
    except Exception as e:
        print(f"[START] task={task} env={ENV_BENCHMARK} model=rule_based", flush=True)
        print("[END] success=false steps=0 score=0.00 rewards=", flush=True)
        return

    rewards = []

    print(f"[START] task={task} env={ENV_BENCHMARK} model=rule_based", flush=True)

    for step in range(20):
        try:
            action = rule_based_action(obs)

            obs, reward, terminated, truncated, _ = env.step(action)
            obs = obs.tolist()

            done = terminated or truncated
            rewards.append(reward)

        except Exception as e:
            action = int(Action.VERIFY)
            reward = 0.0
            done = True
            rewards.append(reward)

        print(
            f"[STEP] step={step+1} action={Action(action).name} "
            f"reward={reward:.2f} done={'true' if done else 'false'} error=null",
            flush=True
        )

        if done:
            break

    # ---------------- SAFE GRADING ----------------
    try:
        history = getattr(env, "_episode_history", [])
    except:
        history = []

    try:
        grade = compute_grade(task, history)
    except:
        grade = {"passed": False, "score": 0.0}

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={'true' if grade.get('passed', False) else 'false'} "
        f"steps={len(rewards)} score={grade.get('score', 0.0):.2f} rewards={rewards_str}",
        flush=True
    )


# ---------------- MAIN ----------------
def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--task", default="easy")
        parser.add_argument("--seed", type=int, default=None)
        args = parser.parse_args()

        run_episode(task=args.task, seed=args.seed)

    except Exception as e:
        print(f"[FATAL ERROR] {e}", flush=True)
        print("[END] success=false steps=0 score=0.00 rewards=", flush=True)
        sys.exit(0)


if __name__ == "__main__":
    main()
