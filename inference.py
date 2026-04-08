from __future__ import annotations

import sys
import traceback

print("[INFO] inference.py started", flush=True)

import os
import json
import random
import argparse
from typing import Any, Dict, List, Optional

# ✅ SAFE IMPORTS (NO sys.exit)
try:
    from environment import FraudDetectionEnv, compute_grade, MERCHANT_CATEGORIES
except Exception:
    try:
        from environement import FraudDetectionEnv, compute_grade, MERCHANT_CATEGORIES
    except Exception as e:
        print(f"[FATAL ERROR] Cannot import environment: {e}", flush=True)
        raise e

try:
    from models import Action, TASK_DEFINITIONS
except Exception as e:
    print(f"[FATAL ERROR] Cannot import models: {e}", flush=True)
    raise e

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str   = os.getenv("MODEL_NAME",   "gpt-4o-mini")
ENV_BENCHMARK     = "financial-fraud-defender-v1"


def rule_based_action(obs: List[float]) -> int:
    try:
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

    except Exception:
        return int(Action.APPROVE)


def random_action(rng: random.Random) -> int:
    return int(rng.choice([Action.APPROVE, Action.BLOCK, Action.VERIFY]))


def run_episode(
    task: str = "easy",
    agent: str = "rule_based",
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:

    all_rewards: List[float] = []
    success = False
    score   = 0.0
    details = {}
    history = []
    env     = None
    grade   = {}

    try:
        print("🚀 Creating environment...", flush=True)
        env = FraudDetectionEnv(task=task, seed=seed)
        print("✅ Environment created", flush=True)

        print("🔄 Resetting environment...", flush=True)
        result = env.reset(seed=seed)
        print("📦 Reset result:", result, flush=True)

        if isinstance(result, tuple):
            obs_arr = result[0]
        else:
            obs_arr = result

        if obs_arr is None:
            raise ValueError("obs_arr is None")

        if not hasattr(obs_arr, "tolist"):
            raise ValueError(f"obs_arr has no tolist(): {type(obs_arr)}")

        obs = obs_arr.tolist()
        print("✅ Observation processed", flush=True)

        rng = random.Random(seed)

        for step in range(20):
            try:
                action = random_action(rng) if agent == "random" else rule_based_action(obs)

                obs_arr, reward, terminated, truncated, info = env.step(action)
                obs = obs_arr.tolist()
                done = terminated or truncated

            except Exception as e:
                print("❌ STEP ERROR:", str(e), flush=True)
                action = int(Action.APPROVE)
                reward = 0.0
                done   = True

            all_rewards.append(reward)

            if done:
                break

        try:
            history = getattr(env, "_episode_history", [])
            grade   = compute_grade(task, history)

            success = grade.get("passed", False)
            score   = grade.get("score",  0.0)
            details = grade.get("details", {})

        except Exception as e:
            print("❌ GRADING ERROR:", str(e), flush=True)

    except Exception as exc:
        print("❌ ENVIRONMENT CRASH:", str(exc), flush=True)
        raise exc

    finally:
        rewards_str = ",".join(f"{r:.2f}" for r in all_rewards)
        print(f"[END] success={success} steps={len(all_rewards)} score={score:.2f} rewards={rewards_str}", flush=True)

    return {
        "task": task,
        "agent": agent,
        "total_reward": sum(all_rewards),
        "steps": len(all_rewards),
        "score": score,
        "passed": success,
        "details": details,
        "episode_history": history,
    }


def main() -> None:
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--task", default="easy")
        parser.add_argument("--agent", default="rule_based")
        parser.add_argument("--seed", type=int, default=None)
        args = parser.parse_args()

        run_episode(task=args.task, agent=args.agent, seed=args.seed)

    except Exception as e:
        print("❌ MAIN CRASH:", str(e), flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    main()
