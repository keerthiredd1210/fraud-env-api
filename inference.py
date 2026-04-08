"""
inference.py — Run agents against the Financial Fraud Defender environment.

MANDATORY FORMAT:
[START]
[STEP]
[END]
"""

from __future__ import annotations

import os
import re
import json
import random
import argparse
from typing import Any, Dict, List, Optional, Tuple

from environment import FraudDetectionEnv, compute_grade, MERCHANT_CATEGORIES
from models import Action, TASK_DEFINITIONS


API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini")

ENV_BENCHMARK: str = "financial-fraud-defender-v1"


# ---------------- RULE-BASED ---------------- #

def rule_based_action(obs: List[float]) -> int:
    try:
        (amount, hour_norm, location_match, txn_last_hour, is_new_merchant,
         risk_score, _step, merch_cat, velocity_ratio, time_since, card_present,
         la1, la2, la3) = obs

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

    except Exception:
        return int(Action.APPROVE)


def random_action(rng: random.Random) -> int:
    return int(rng.choice([Action.APPROVE, Action.BLOCK, Action.VERIFY]))


# ---------------- MAIN EPISODE ---------------- #

def run_episode(
    task: str = "easy",
    agent: str = "rule_based",
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:

    all_rewards: List[float] = []
    success = False
    score = 0.0

    try:
        env = FraudDetectionEnv(task=task, seed=seed)
        obs_arr, _ = env.reset(seed=seed)
        obs = obs_arr.tolist()

        rng = random.Random(seed)

        if verbose:
            print(f"[START] task={task} env={ENV_BENCHMARK} model={agent}", flush=True)

        for step in range(20):

            try:
                if agent == "random":
                    action = random_action(rng)
                else:
                    action = rule_based_action(obs)

                obs_arr, reward, terminated, truncated, info = env.step(action)
                obs = obs_arr.tolist()

            except Exception:
                action = int(Action.APPROVE)
                reward = 0.0
                terminated = True
                truncated = False

            all_rewards.append(reward)
            done = terminated or truncated

            if verbose:
                print(
                    f"[STEP] step={step + 1} action={Action(action).name} "
                    f"reward={reward:.2f} done={'true' if done else 'false'} "
                    f"error=null",
                    flush=True,
                )

            if done:
                break

        # SAFE grading
        try:
            history = getattr(env, "_episode_history", [])
            grade = compute_grade(task, history)
            success = grade.get("passed", False)
            score = grade.get("score", 0.0)
        except Exception:
            success = False
            score = 0.0

    except Exception as e:
        print(f"[ERROR] {str(e)}", flush=True)

    finally:
        # MUST ALWAYS PRINT [END]
        try:
            rewards_str = ",".join(f"{r:.2f}" for r in all_rewards)
            print(
                f"[END] success={'true' if success else 'false'} "
                f"steps={len(all_rewards)} score={score:.2f} rewards={rewards_str}",
                flush=True,
            )
        except Exception:
            print(f"[END] success=false steps=0 score=0.00 rewards=", flush=True)

    return {
        "task": task,
        "total_reward": sum(all_rewards),
        "steps": len(all_rewards),
        "score": score,
        "passed": success,
    }


# ---------------- ENTRY POINT ---------------- #

def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--task", default="easy", choices=["easy", "medium", "hard"])
        parser.add_argument("--agent", default="rule_based", choices=["rule_based", "random"])
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--quiet", action="store_true")

        args = parser.parse_args()

        result = run_episode(
            task=args.task,
            agent=args.agent,
            seed=args.seed,
            verbose=not args.quiet
        )

        if args.quiet:
            print(json.dumps(result, indent=2))

    except Exception as e:
        print(f"[FATAL ERROR] {str(e)}", flush=True)


if __name__ == "__main__":
    main()
