"""
inference.py — Run agents against the Financial Fraud Defender environment.

Mandatory stdout format:
  [START] task=<name> env=<benchmark> model=<model>
  [STEP]  step=<n> action=<APPROVE|BLOCK|VERIFY> reward=<0.00> done=<true|false> error=<null>
  [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
"""
from __future__ import annotations

import os
import re
import sys
import json
import random
import argparse
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Robust imports — handles both correct and typo spelling of environment.py
# ---------------------------------------------------------------------------
try:
    from environment import FraudDetectionEnv, compute_grade, MERCHANT_CATEGORIES
except ImportError:
    try:
        from environement import FraudDetectionEnv, compute_grade, MERCHANT_CATEGORIES  # typo fallback
    except ImportError as _e:
        print(f"[FATAL ERROR] Cannot import environment module: {_e}", flush=True)
        print("[END] success=false steps=0 score=0.00 rewards=", flush=True)
        sys.exit(0)

try:
    from models import Action, TASK_DEFINITIONS
except ImportError as _e:
    print(f"[FATAL ERROR] Cannot import models module: {_e}", flush=True)
    print("[END] success=false steps=0 score=0.00 rewards=", flush=True)
    sys.exit(0)

# ---------------------------------------------------------------------------
# Environment / model configuration
# ---------------------------------------------------------------------------

API_BASE_URL: str     = os.getenv("API_BASE_URL",     "https://api.openai.com/v1")
MODEL_NAME: str       = os.getenv("MODEL_NAME",       "gpt-4o-mini")
HF_TOKEN: str         = os.getenv("HF_TOKEN",         "")
LOCAL_IMAGE_NAME: str = os.getenv("LOCAL_IMAGE_NAME", "financial-fraud-defender")

ENV_BENCHMARK: str = "financial-fraud-defender-v1"


# ---------------------------------------------------------------------------
# Observation -> natural language (for LLM context)
# ---------------------------------------------------------------------------

def _obs_to_text(obs: List[float], step: int) -> str:
    (amount, hour_norm, location_match, txn_last_hour, is_new_merchant,
     risk_score, _step, merch_cat, velocity_ratio, time_since, card_present,
     la1, la2, la3) = obs

    hour     = int(round(hour_norm * 23))
    cat_idx  = max(0, min(int(round(merch_cat)), len(MERCHANT_CATEGORIES) - 1))
    category = MERCHANT_CATEGORIES[cat_idx]

    flags = []
    if risk_score >= 7.0:
        flags.append(f"HIGH RISK ({risk_score:.1f}/10)")
    elif risk_score >= 4.0:
        flags.append(f"MEDIUM RISK ({risk_score:.1f}/10)")
    if velocity_ratio > 3.0:
        flags.append(f"HIGH VELOCITY {velocity_ratio:.1f}x avg")
    elif velocity_ratio > 1.5:
        flags.append(f"ELEVATED VELOCITY {velocity_ratio:.1f}x avg")
    if location_match < 0.5:
        flags.append("FOREIGN LOCATION")
    if is_new_merchant > 0.5:
        flags.append("NEW MERCHANT")
    if card_present < 0.5:
        flags.append("CARD NOT PRESENT (online)")
    if txn_last_hour >= 4:
        flags.append(f"BURST: {int(txn_last_hour)} txns/hr")
    if time_since < 2.0:
        flags.append(f"RAPID FOLLOW-UP ({time_since:.1f} min ago)")

    last_acts = []
    for a in [la1, la2, la3]:
        if a >= 0:
            try:
                last_acts.append(Action(int(a)).name)
            except ValueError:
                pass
    last_acts_str = ", ".join(last_acts) if last_acts else "none"

    return (
        f"Transaction #{step + 1}: ${amount:.2f} at {hour:02d}:00 "
        f"| category: {category}\n"
        f"Risk score: {risk_score:.1f}/10 | Velocity: {velocity_ratio:.2f}x avg\n"
        f"Location: {'HOME' if location_match > 0.5 else 'FOREIGN'} "
        f"| Card: {'PRESENT' if card_present > 0.5 else 'NOT PRESENT'} "
        f"| New merchant: {'YES' if is_new_merchant > 0.5 else 'NO'}\n"
        f"Txns last hour: {int(txn_last_hour)} "
        f"| Time since last txn: {time_since:.1f} min\n"
        f"Active flags: [{', '.join(flags) if flags else 'NONE'}]\n"
        f"Last 3 agent actions: [{last_acts_str}]"
    )


# ---------------------------------------------------------------------------
# LLM system prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are a financial fraud detection AI making real-time decisions on transactions.

== COST MATRIX (memorise exactly) ==
  APPROVE + legit  = +0.50   <- correct, customer happy
  APPROVE + fraud  = -1.00   <- THE WORST - real money lost, not recoverable
  BLOCK   + fraud  = +1.00   <- best fraud catch
  BLOCK   + legit  = -0.50   <- customer inconvenienced, costs half of missed fraud
  VERIFY  + fraud  = +0.70   <- smart catch with low friction
  VERIFY  + legit  = -0.10   <- tiny cost, customer barely notices

* KEY INSIGHT: Missing fraud (-1.00) costs EXACTLY 2x more than a false block (-0.50).
  Therefore: when uncertain -> VERIFY (not APPROVE). VERIFY is almost free for real customers.
  Only APPROVE when you are CONFIDENT the transaction is legitimate.

== HARD RULES ==
  RULE A: velocity_ratio > 3.0 AND card NOT PRESENT -> always BLOCK
  RULE B: time_since_last_txn < 2 min -> escalate (APPROVE->VERIFY, VERIFY->BLOCK)
  RULE C: risk_score >= 8.0 OR velocity_ratio > 6.0 -> always BLOCK

Current task: {task} - {task_description}

RESPOND WITH A SINGLE INTEGER: 0 (APPROVE), 1 (BLOCK), or 2 (VERIFY).
No explanation. No punctuation. Just the digit.\
"""


# ---------------------------------------------------------------------------
# Hard rules - pre/post LLM
# ---------------------------------------------------------------------------

def _apply_pre_rules(obs: List[float]) -> Optional[int]:
    (_, _, _, _, _, risk_score, _, _, velocity_ratio, _, card_present,
     _, _, _) = obs
    if risk_score >= 8.0 or velocity_ratio > 6.0:
        return int(Action.BLOCK)
    if velocity_ratio > 3.0 and card_present < 0.5:
        return int(Action.BLOCK)
    return None


def _apply_post_rules(action: int, obs: List[float]) -> int:
    time_since = obs[9]
    if time_since < 2.0:
        if action == int(Action.APPROVE):
            return int(Action.VERIFY)
        elif action == int(Action.VERIFY):
            return int(Action.BLOCK)
    return action


# ---------------------------------------------------------------------------
# LLM response parser
# ---------------------------------------------------------------------------

def _parse_action(raw: str) -> int:
    cleaned = raw.strip()
    if cleaned and cleaned[0] in "012":
        return int(cleaned[0])
    m = re.search(r"\b([012])\b", cleaned)
    if m:
        return int(m.group(1))
    m = re.search(r"([012])", cleaned)
    if m:
        return int(m.group(1))
    upper = cleaned.upper()
    if "APPROVE" in upper: return int(Action.APPROVE)
    if "BLOCK"   in upper: return int(Action.BLOCK)
    if "VERIFY"  in upper: return int(Action.VERIFY)
    raise ValueError(f"Cannot parse action from: {raw!r}")


# ---------------------------------------------------------------------------
# Rule-based agent - deterministic, no API key needed
# ---------------------------------------------------------------------------

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
        bool(is_new_merchant > 0.5 and card_present < 0.5),
        txn_last_hour >= 4,
        time_since < 2.0,
    ])
    if suspicious >= 2 or (risk_score >= 4.0 and suspicious >= 1):
        return int(Action.VERIFY)

    return int(Action.APPROVE)


# ---------------------------------------------------------------------------
# Random agent
# ---------------------------------------------------------------------------

def random_action(rng: random.Random) -> int:
    return int(rng.choice([Action.APPROVE, Action.BLOCK, Action.VERIFY]))


# ---------------------------------------------------------------------------
# LLM agent
# ---------------------------------------------------------------------------

def _llm_action(
    client: Any,
    task: str,
    obs: List[float],
    step: int,
    episode_memory: List[str],
) -> Tuple[int, Optional[str]]:
    pre = _apply_pre_rules(obs)
    if pre is not None:
        return pre, None

    memory_block = ""
    if episode_memory:
        memory_block = (
            "\n\nEpisode memory (last 5 decisions):\n"
            + "\n".join(f"  {m}" for m in episode_memory[-5:])
        )

    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        task=task,
        task_description=TASK_DEFINITIONS[task].description,
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": _obs_to_text(obs, step) + memory_block},
            ],
            max_tokens=5,
            temperature=0.0,
        )
        raw    = response.choices[0].message.content
        action = _parse_action(raw)
        action = _apply_post_rules(action, obs)
        return action, None
    except Exception as exc:
        return rule_based_action(obs), str(exc)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    task:    str           = "easy",
    agent:   str           = "rule_based",
    seed:    Optional[int] = None,
    verbose: bool          = True,
) -> Dict[str, Any]:
    env = None
    try:
        env = FraudDetectionEnv(task=task, seed=seed)
        obs_arr, _ = env.reset(seed=seed)
        obs = obs_arr.tolist()
    except Exception as exc:
        if verbose:
            print(f"[START] task={task} env={ENV_BENCHMARK} model=rule_based", flush=True)
            print(f"[END] success=false steps=0 score=0.00 rewards=", flush=True)
        raise

    client = None
    # FIX 1: Corrected OpenAI Client initialization
    if agent == "llm":
        try:
            from openai import OpenAI
            client = OpenAI(
                base_url=os.environ["API_BASE_URL"],
                api_key=os.environ["API_KEY"],
            )
        except (ImportError, KeyError):
            agent = "rule_based"

    rng            = random.Random(seed)
    episode_memory: List[str]     = []
    all_rewards:    List[float]   = []
    error_flag:     Optional[str] = None
    model_label = MODEL_NAME if agent == "llm" else agent

    if verbose:
        print(f"[START] task={task} env={ENV_BENCHMARK} model={model_label}", flush=True)

    for step in range(20):
        try:
            if agent == "llm" and client is not None:
                action, err = _llm_action(client, task, obs, step, episode_memory)
                if err and not error_flag:
                    error_flag = err
            elif agent == "random":
                action = random_action(rng)
            else:
                action = rule_based_action(obs)

            obs_arr, reward, terminated, truncated, info = env.step(action)
            obs  = obs_arr.tolist()
            all_rewards.append(reward)
            done = terminated or truncated

        except Exception as exc:
            error_flag = str(exc)
            action = int(Action.VERIFY)
            done   = True
            reward = 0.0
            all_rewards.append(reward)

        if verbose:
            err_val = "null" if not error_flag else json.dumps(str(error_flag))
            print(
                f"[STEP] step={step + 1} action={Action(action).name} "
                f"reward={reward:.2f} done={'true' if done else 'false'} "
                f"error={err_val}",
                flush=True,
            )

        episode_memory.append(
            f"Step {step + 1}: {Action(action).name} -> reward {reward:+.2f}"
        )

        if done:
            break

    # FIX 3: Safe history and grading
    history = getattr(env, "_episode_history", [])
    try:
        grade = compute_grade(task, history)
    except:
        grade = {"passed": False, "score": 0.0, "details": {}}

    rewards_str = ",".join(f"{r:.2f}" for r in all_rewards)

    if verbose:
        print(
            f"[END] success={'true' if grade['passed'] else 'false'} "
            f"steps={len(all_rewards)} score={grade['score']:.2f} rewards={rewards_str}",
            flush=True,
        )

    return {
        "task":            task,
        "agent":           agent,
        "total_reward":    sum(all_rewards),
        "steps":           len(all_rewards),
        "score":           grade["score"],
        "passed":          grade["passed"],
        "details":         grade["details"],
        "episode_history": history,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        parser = argparse.ArgumentParser(
            description="Financial Fraud Defender - inference runner"
        )
        parser.add_argument("--task",  default="easy",
                            choices=["easy", "medium", "hard"])
        # FIX 2: Changed default agent to "llm"
        parser.add_argument("--agent", default="llm",
                            choices=["rule_based", "random", "llm"])
        parser.add_argument("--seed",  type=int, default=None)
        parser.add_argument("--quiet", action="store_true",
                            help="Suppress step output; print JSON result")
        args = parser.parse_args()

        result = run_episode(
            task=args.task,
            agent=args.agent,
            seed=args.seed,
            verbose=not args.quiet,
        )

        if args.quiet:
            print(json.dumps(result, indent=2))

    except SystemExit:
        raise
    except Exception as exc:
        print(f"[FATAL ERROR] {exc}", flush=True)
        print("[END] success=false steps=0 score=0.00 rewards=", flush=True)
        sys.exit(0)


if __name__ == "__main__":
    main()
