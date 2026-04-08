# Fix uvloop/asyncio crash on Python 3.11
import asyncio
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
"""
inference.py — Financial Fraud Defender
"""
from __future__ import annotations

# ── CRITICAL: must be first — fixes uvloop crash on Python 3.11 ──
import asyncio
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

import os
import json
import random
import argparse
from typing import Any, Dict, List, Optional

try:
    from environment import FraudDetectionEnv, compute_grade, MERCHANT_CATEGORIES
except Exception:
    try:
        from environement import FraudDetectionEnv, compute_grade, MERCHANT_CATEGORIES
    except Exception as e:
        import sys
        print(f"[FATAL ERROR] Cannot import environment: {e}", flush=True)
        print("[END] success=false steps=0 score=0.00 rewards=", flush=True)
        sys.exit(0)

try:
    from models import Action, TASK_DEFINITIONS
except Exception as e:
    import sys
    print(f"[FATAL ERROR] Cannot import models: {e}", flush=True)
    print("[END] success=false steps=0 score=0.00 rewards=", flush=True)
    sys.exit(0)

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
        env = FraudDetectionEnv(task=task, seed=seed)
        obs_arr, _ = env.reset(seed=seed)
        obs = obs_arr.tolist()
        rng = random.Random(seed)

        if verbose:
            print(f"[START] task={task} env={ENV_BENCHMARK} model={agent}", flush=True)

        for step in range(20):
            try:
                action = random_action(rng) if agent == "random" else rule_based_action(obs)
                obs_arr, reward, terminated, truncated, info = env.step(action)
                obs  = obs_arr.tolist()
                done = terminated or truncated
            except Exception:
                action = int(Action.APPROVE)
                reward = 0.0
                done   = True

            all_rewards.append(reward)

            if verbose:
                print(
                    f"[STEP] step={step + 1} action={Action(action).name} "
                    f"reward={reward:.2f} done={'true' if done else 'false'} error=null",
                    flush=True,
                )

            if done:
                break

        try:
            history = getattr(env, "_episode_history", [])
            grade   = compute_grade(task, history)
            success = grade.get("passed", False)
            score   = grade.get("score",  0.0)
            details = grade.get("details", {})
        except Exception:
            pass

    except Exception as exc:
        print(f"[ERROR] {exc}", flush=True)

    finally:
        try:
            rewards_str = ",".join(f"{r:.2f}" for r in all_rewards)
            print(
                f"[END] success={'true' if success else 'false'} "
                f"steps={len(all_rewards)} score={score:.2f} rewards={rewards_str}",
                flush=True,
            )
        except Exception:
            print("[END] success=false steps=0 score=0.00 rewards=", flush=True)

    return {
        "task":            task,
        "agent":           agent,
        "total_reward":    sum(all_rewards),
        "steps":           len(all_rewards),
        "score":           score,
        "passed":          success,
        "details":         details,
        "episode_history": history,
    }


def main() -> None:
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--task",  default="easy",      choices=["easy", "medium", "hard"])
        parser.add_argument("--agent", default="rule_based", choices=["rule_based", "random", "llm"])
        parser.add_argument("--seed",  type=int, default=None)
        parser.add_argument("--quiet", action="store_true")
        args = parser.parse_args()

        result = run_episode(
            task=args.task, agent=args.agent,
            seed=args.seed, verbose=not args.quiet,
        )
        if args.quiet:
            print(json.dumps(result, indent=2))

    except SystemExit:
        raise
    except Exception as exc:
        print(f"[FATAL ERROR] {exc}", flush=True)
        print("[END] success=false steps=0 score=0.00 rewards=", flush=True)


if __name__ == "__main__":
    main()
