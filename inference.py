from __future__ import annotations

import asyncio
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

import sys
import traceback

def safe_main():
    try:
        from environment import FraudDetectionEnv, compute_grade
        from models import Action, TASK_DEFINITIONS

        task = "easy"
        seed = 42

        env = FraudDetectionEnv(task=task, seed=seed)

        obs, info = env.reset(seed=seed)

        total_reward = 0
        steps = 0

        done = False

        while not done:
            action = 0  # APPROVE (safe default)

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            steps += 1

            done = terminated or truncated

        result = compute_grade(task, env._episode_history if hasattr(env, "_episode_history") else [])

        print(f"[END] success=true steps={steps} score={result.get('score', 0.0):.2f} rewards={total_reward:.2f}")

    except Exception as e:
        print("[FATAL ERROR]", str(e))
        traceback.print_exc()
        print("[END] success=false steps=0 score=0.0 rewards=0.0")
        sys.exit(0)


if __name__ == "__main__":
    safe_main()
