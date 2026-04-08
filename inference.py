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
import json
import random
import argparse
from typing import Any, Dict, List, Optional, Tuple

from environment import FraudDetectionEnv, compute_grade, MERCHANT_CATEGORIES
from models import Action, TASK_DEFINITIONS

# ---------------------------------------------------------------------------
# Required env vars
# ---------------------------------------------------------------------------

API_BASE_URL: str     = os.getenv("API_BASE_URL",     "https://api.openai.com/v1")
MODEL_NAME: str       = os.getenv("MODEL_NAME",       "gpt-4o-mini")
HF_TOKEN: str         = os.getenv("HF_TOKEN",         "")
LOCAL_IMAGE_NAME: str = os.getenv("LOCAL_IMAGE_NAME", "financial-fraud-defender")

ENV_BENCHMARK: str = "financial-fraud-defender-v1"


def _obs_to_text(obs: List[float], step: int) -> str:
    (amount, hour_norm, location_match, txn_last_hour, is_new_merchant,
     risk_score, _step, merch_cat, velocity_ratio, time_since, card_present,
     la1, la2, la3) = obs

    hour     = int(round(hour_norm * 23))
    cat_idx  = max(0, min(int(round(merch_cat)), len(MERCHANT_CATEGORIES) - 1))
    category = MERCHANT_CATEGORIES[cat_idx]

    flags = []
    if risk_score >= 7.0:      flags.append(f"HIGH RISK ({risk_score:.1f}/10)")
    elif risk_score >= 4.0:    flags.append(f"MEDIUM RISK ({risk_score:.1f}/10)")
    if velocity_ratio > 3.0:   flags.append(f"HIGH VELOCITY {velocity_ratio:.1f}x avg")
    elif velocity_ratio > 1.5: flags.append(f"ELEVATED VELOCITY {velocity_ratio:.1f}x avg")
    if location_match < 0.5:   flags.append("FOREIGN LOCATION")
    if is_new_merchant > 0.5:  flags.append("NEW MERCHANT")
    if card_present < 0.5:     flags.append("CARD NOT PRESENT (online)")
    if txn_last_hour >= 4:     flags.append(f"BURST: {int(txn_last_hour)} txns/hr")
    if time_since < 2.0:       flags.append(f"RAPID FOLLOW-UP ({time_since:.1f} min ago)")

    last_acts = []
    for a in [la1, la2, la3]:
        if a >= 0:
            try: last_acts.append(Action(int(a)).name)
            except ValueError: pass
    last_acts_str = ", ".join(last_acts) if last_acts else "none"

    return (
        f"Transaction #{step + 1}: ${amount:.2f} at {hour:02d}:00 | category: {category}\n"
        f"Risk score: {risk_score:.1f}/10 | Velocity: {velocity_ratio:.2f}x avg\n"
        f"Location: {'HOME' if location_match > 0.5 else 'FOREIGN'} "
        f"| Card: {'PRESENT' if card_present > 0.5 else 'NOT PRESENT'} "
        f"| New merchant: {'YES' if is_new_merchant > 0.5 else 'NO'}\n"
        f"Txns last hour: {int(txn_last_hour)} | Time since last txn: {time_since:.1f} min\n"
        f"Active flags: [{', '.join(flags) if flags else 'NONE'}]\n"
        f"Last 3 agent actions: [{last_acts_str}]"
    )


_SYSTEM_PROMPT_TEMPLATE = """\
You are a financial fraud detection AI making real-time decisions on transactions.

== COST MATRIX ==
  APPROVE + legit  = +0.50   correct, customer happy
  APPROVE + fraud  = -1.00   THE WORST -- real money lost
  BLOCK   + fraud  = +1.00   best fraud catch
  BLOCK   + legit  = -0.50   customer friction, half of missed fraud
  VERIFY  + fraud  = +0.70   smart catch with low friction
  VERIFY  + legit  = -0.10   tiny cost, customer barely notices

KEY INSIGHT: missing fraud (-1.00) costs 2x more than false block (-0.50).
When uncertain -> VERIFY. Only APPROVE when CONFIDENT.

== HARD RULES ==
  RULE A: velocity_ratio > 3.0 AND card NOT PRESENT -> BLOCK
  RULE B: time_since_last_txn < 2 min -> escalate (APPROVE->VERIFY, VERIFY->BLOCK)
  RULE C: risk_score >= 8.0 OR velocity_ratio > 6.0 -> BLOCK

== FEW-SHOT EXAMPLES ==

Example A -> BLOCK (1):
  $1,847 at 03:00 | Risk: 9.2 | Vel: 6.4x | FOREIGN | Card NOT PRESENT
  -> 1  (Rule A + Rule C fire)

Example B -> APPROVE (0):
  $89 at 14:00 | Risk: 1.0 | Vel: 0.7x | HOME | Card PRESENT | No flags
  -> 0  (all clear)

Example C -> VERIFY (2):
  $342 at 22:00 | Risk: 4.5 | Vel: 2.1x | HOME | Card NOT PRESENT | New merchant
  -> 2  (uncertain -- VERIFY is safe default)

Current task: {task} -- {task_description}

RESPOND WITH A SINGLE INTEGER: 0 (APPROVE), 1 (BLOCK), or 2 (VERIFY).
No explanation. No punctuation. Just the digit.\
"""


def _apply_pre_rules(obs: List[float]) -> Optional[int]:
    (amount, hour_norm, location_match, txn_last_hour, is_new_merchant,
     risk_score, _step, merch_cat, velocity_ratio, time_since, card_present,
     la1, la2, la3) = obs
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


def rule_based_action(obs: List[float]) -> int:
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
        bool(is_new_merchant > 0.5 and card_present < 0.5),
        txn_last_hour >= 4,
        time_since < 2.0,
    ])
    if suspicious >= 2 or (risk_score >= 4.0 and suspicious >= 1):
        return int(Action.VERIFY)
    return int(Action.APPROVE)


def random_action(rng: random.Random) -> int:
    return int(rng.choice([Action.APPROVE, Action.BLOCK, Action.VERIFY]))


def _llm_action(
    client: Any, task: str, obs: List[float],
    step: int, episode_memory: List[str],
) -> Tuple[int, Optional[str]]:
    pre = _apply_pre_rules(obs)
    if pre is not None:
        return pre, None

    description  = _obs_to_text(obs, step)
    memory_block = ""
    if episode_memory:
        recent = episode_memory[-5:]
        memory_block = "\n\nEpisode memory (last 5):\n" + "\n".join(f"  {m}" for m in recent)

    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        task=task, task_description=TASK_DEFINITIONS[task].description,
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": description + memory_block},
            ],
            max_tokens=5,
            temperature=0.0,
        )
        raw    = response.choices[0].message.content
        action = _parse_action(raw)
        if action not in (0, 1, 2):
            raise ValueError(f"Parsed action {action} out of range")
        action = _apply_post_rules(action, obs)
        return action, None
    except Exception as exc:
        return rule_based_action(obs), str(exc)


def run_episode(
    task:    str           = "easy",
    agent:   str           = "rule_based",
    seed:    Optional[int] = None,
    verbose: bool          = True,
) -> Dict[str, Any]:
    """Run one full 20-step episode. All mandatory lines go to stdout."""
    env = FraudDetectionEnv(task=task, seed=seed)
    obs_arr, _ = env.reset(seed=seed)
    obs = obs_arr.tolist()

    client = None
    if agent == "llm":
        try:
            from openai import OpenAI
            client = OpenAI(base_url=API_BASE_URL,
                            api_key=os.getenv("OPENAI_API_KEY", "dummy"))
        except ImportError:
            agent = "rule_based"

    rng            = random.Random(seed)
    episode_memory: List[str]     = []
    all_rewards:    List[float]   = []
    error_flag:     Optional[str] = None
    model_label = MODEL_NAME if agent == "llm" else agent

    # [START]
    if verbose:
        print(f"[START] task={task} env={ENV_BENCHMARK} model={model_label}", flush=True)

    for step in range(20):
        if agent == "llm" and client is not None:
            action, err = _llm_action(client, task, obs, step, episode_memory)
            if err and not error_flag:
                error_flag = err
        elif agent == "random":
            action = random_action(rng)
        else:
            action = rule_based_action(obs)

        obs_arr, reward, terminated, truncated, info = env.step(action)
        obs = obs_arr.tolist()
        all_rewards.append(reward)
        done = terminated or truncated

        # [STEP]
        if verbose:
            err_val = "null" if not error_flag else json.dumps(error_flag)
            print(
                f"[STEP] step={step + 1} action={Action(action).name} "
                f"reward={reward:.2f} done={'true' if done else 'false'} "
                f"error={err_val}",
                flush=True,
            )

        episode_memory.append(f"Step {step + 1}: {Action(action).name} -> reward {reward:+.2f}")

        if done:
            break

    history     = env._episode_history
    grade       = compute_grade(task, history)
    rewards_str = ",".join(f"{r:.2f}" for r in all_rewards)

    # [END]
    if verbose:
        print(
            f"[END] success={'true' if grade['passed'] else 'false'} "
            f"steps={len(all_rewards)} score={grade['score']:.2f} rewards={rewards_str}",
            flush=True,
        )

    return {
        "task": task, "agent": agent,
        "total_reward": sum(all_rewards), "steps": len(all_rewards),
        "score": grade["score"], "passed": grade["passed"],
        "details": grade["details"], "episode_history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Financial Fraud Defender — inference runner")
    parser.add_argument("--task",  default="easy",       choices=["easy", "medium", "hard"])
    parser.add_argument("--agent", default="rule_based", choices=["rule_based", "random", "llm"])
    parser.add_argument("--seed",  type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    result = run_episode(task=args.task, agent=args.agent,
                         seed=args.seed, verbose=not args.quiet)
    if args.quiet:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
