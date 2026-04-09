"""
inference.py — Run agents against the Financial Fraud Defender environment.

Mandatory stdout format:
  [START] task=<name> env=<benchmark> model=<model>
  [STEP]  step=<n> action=<APPROVE|BLOCK|VERIFY> reward=<0.00> done=<true|false> error=<null>
  [END]   success=<true|false> steps=<n> score=<0.0010> rewards=<r1,r2,...>
"""
from __future__ import annotations

import os
import re
import sys
import json
import random
import argparse
from typing import Any, Dict, List, Optional, Tuple

try:
    from environment import FraudDetectionEnv, compute_grade, MERCHANT_CATEGORIES
except ImportError:
    try:
        from environement import FraudDetectionEnv, compute_grade, MERCHANT_CATEGORIES
    except ImportError as _e:
        print(f"[FATAL ERROR] Cannot import environment module: {_e}", flush=True)
        print("[END] success=false steps=0 score=0.0010 rewards=", flush=True)
        sys.exit(0)

try:
    from models import Action, TASK_DEFINITIONS
except ImportError as _e:
    print(f"[FATAL ERROR] Cannot import models module: {_e}", flush=True)
    print("[END] success=false steps=0 score=0.0010 rewards=", flush=True)
    sys.exit(0)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL: str     = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str       = os.getenv("MODEL_NAME", "gpt-4o-mini")
ENV_BENCHMARK: str    = "financial-fraud-defender-v1"

# ---------------------------------------------------------------------------
# Observation -> Text
# ---------------------------------------------------------------------------
def _obs_to_text(obs: List[float], step: int) -> str:
    (amount, hour_norm, location_match, txn_last_hour, is_new_merchant,
     risk_score, _step, merch_cat, velocity_ratio, time_since, card_present,
     la1, la2, la3) = obs

    hour = int(round(hour_norm * 23))
    cat_idx = max(0, min(int(round(merch_cat)), len(MERCHANT_CATEGORIES) - 1))
    category = MERCHANT_CATEGORIES[cat_idx]

    flags = []
    if risk_score >= 7.0: flags.append(f"HIGH RISK ({risk_score:.1f}/10)")
    if velocity_ratio > 3.0: flags.append(f"HIGH VELOCITY")
    if location_match < 0.5: flags.append("FOREIGN LOCATION")
    if card_present < 0.5: flags.append("CARD NOT PRESENT")

    last_acts = []
    for a in [la1, la2, la3]:
        if a >= 0:
            try: last_acts.append(Action(int(a)).name)
            except ValueError: pass
    last_acts_str = ", ".join(last_acts) if last_acts else "none"

    return (
        f"Transaction #{step + 1}: ${amount:.2f} at {hour:02d}:00 | category: {category}\n"
        f"Risk score: {risk_score:.1f}/10 | Velocity: {velocity_ratio:.2f}x avg\n"
        f"Location: {'HOME' if location_match > 0.5 else 'FOREIGN'} | Card: {'PRESENT' if card_present > 0.5 else 'NOT PRESENT'}\n"
        f"Active flags: [{', '.join(flags) if flags else 'NONE'}]\n"
        f"Last actions: [{last_acts_str}]"
    )

_SYSTEM_PROMPT_TEMPLATE = """\
You are a financial fraud detection AI making real-time decisions on transactions.

== COST MATRIX ==
  APPROVE + legit  = +0.50
  APPROVE + fraud  = -1.00
  BLOCK   + fraud  = +1.00
  BLOCK   + legit  = -0.50
  VERIFY  + fraud  = +0.70
  VERIFY  + legit  = -0.10

Missing fraud (-1.00) costs 2x more than a false block (-0.50). 
When uncertain, VERIFY.

Current task: {task} - {task_description}
RESPOND WITH A SINGLE INTEGER: 0 (APPROVE), 1 (BLOCK), or 2 (VERIFY).
"""

# ---------------------------------------------------------------------------
# Logic / Agents
# ---------------------------------------------------------------------------
def _apply_pre_rules(obs: List[float]) -> Optional[int]:
    risk_score = obs[5]
    velocity = obs[8]
    card_present = obs[10]
    if risk_score >= 8.0 or velocity > 6.0: return int(Action.BLOCK)
    if velocity > 3.0 and card_present < 0.5: return int(Action.BLOCK)
    return None

def _apply_post_rules(action: int, obs: List[float]) -> int:
    time_since = obs[9]
    if time_since < 2.0:
        if action == 0: return 2
        if action == 2: return 1
    return action

def _parse_action(raw: str) -> int:
    cleaned = raw.strip()
    m = re.search(r"([012])", cleaned)
    if m: return int(m.group(1))
    upper = cleaned.upper()
    if "APPROVE" in upper: return 0
    if "BLOCK" in upper: return 1
    return 2

def rule_based_action(obs: List[float]) -> int:
    pre = _apply_pre_rules(obs)
    if pre is not None: return pre
    risk_score = obs[5]
    if risk_score >= 5.0: return int(Action.VERIFY)
    return int(Action.APPROVE)

def _llm_action(client, task, obs, step, memory):
    pre = _apply_pre_rules(obs)
    if pre is not None: return pre, None
    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        task=task, task_description=TASK_DEFINITIONS[task].description
    )
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": _obs_to_text(obs, step)}],
            max_tokens=5, temperature=0.0
        )
        action = _parse_action(response.choices[0].message.content)
        return _apply_post_rules(action, obs), None
    except Exception as e:
        return rule_based_action(obs), str(e)

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_episode(task="easy", agent="llm", seed=None, verbose=True) -> Dict[str, Any]:
    try:
        env = FraudDetectionEnv(task=task, seed=seed)
        obs_arr, _ = env.reset(seed=seed)
        obs = obs_arr.tolist()
    except Exception:
        if verbose:
            print(f"[START] task={task} env={ENV_BENCHMARK} model={agent}", flush=True)
            print(f"[END] success=false steps=0 score=0.0010 rewards=", flush=True)
        sys.exit(0)

    client = None
    if agent == "llm":
        try:
            from openai import OpenAI
            client = OpenAI(base_url=API_BASE_URL, api_key=os.environ["API_KEY"])
        except:
            agent = "rule_based"

    all_rewards = []
    if verbose:
        print(f"[START] task={task} env={ENV_BENCHMARK} model={MODEL_NAME if agent=='llm' else agent}", flush=True)

    for step in range(20):
        if agent == "llm" and client:
            action, _ = _llm_action(client, task, obs, step, [])
        elif agent == "random":
            action = random.randint(0, 2)
        else:
            action = rule_based_action(obs)

        obs_arr, reward, terminated, truncated, _ = env.step(action)
        obs = obs_arr.tolist()
        all_rewards.append(reward)
        done = terminated or truncated

        if verbose:
            print(f"[STEP] step={step + 1} action={Action(action).name} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)
        if done: break

    history = getattr(env, "_history", [])
    try:
        grade = compute_grade(task, history)
    except:
        grade = {"passed": False, "score": 0.0010}

    rewards_str = ",".join(f"{r:.2f}" for r in all_rewards)
    if verbose:
        # ✅ FIX: Score format .4f
        print(f"[END] success={str(grade['passed']).lower()} steps={len(all_rewards)} score={grade['score']:.4f} rewards={rewards_str}", flush=True)

    return grade

def main():
    parser = argparse.ArgumentParser()
    # ✅ FIX: Default "all"
    parser.add_argument("--task", default="all", choices=["easy", "medium", "hard", "all"])
    parser.add_argument("--agent", default="llm", choices=["rule_based", "random", "llm"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    # ✅ FIX: Run loop for all tasks
    tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]
    
    for task in tasks:
        run_episode(task=task, agent=args.agent, seed=args.seed, verbose=not args.quiet)

if __name__ == "__main__":
    main()
