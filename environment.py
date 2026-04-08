"""
environment.py — Financial Fraud Defender RL Environment.

Observation space : Box(14,) float32  (via Observation.to_numpy())
Action space      : Discrete(3)       0=APPROVE 1=BLOCK 2=VERIFY
Episode length    : 20 steps
"""

from __future__ import annotations
import asyncio
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

import random
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from models import (
    Action,
    REWARD_TABLE,
    TASK_DEFINITIONS,
    TransactionFeatures,
    UserProfile,
)

NUM_STEPS    = 20
NUM_FEATURES = 14
MERCHANT_CATEGORIES = ["groceries", "travel", "entertainment", "dining",
                       "electronics", "healthcare", "fuel", "other"]
LOCATIONS = ["NYC", "LAX", "CHI", "HOU", "PHX", "PHL", "SAN", "DAL"]


@dataclass
class Observation:
    """14 gym features + derived metadata. Compatible via .tolist() / .to_numpy()."""
    amount: float                  = 0.0
    hour_of_day: float             = 0.0
    location_match: float          = 0.0
    transactions_last_hour: float  = 0.0
    is_new_merchant: float         = 0.0
    risk_score: float              = 0.0
    step_num: float                = 0.0
    merchant_category_index: float = 0.0
    velocity_ratio: float          = 0.0
    time_since_last_txn: float     = 0.0
    card_present: float            = 0.0
    last_action_1: float           = -1.0
    last_action_2: float           = -1.0
    last_action_3: float           = -1.0
    # metadata — NOT part of the 14-element gym vector
    merchant_category: str         = "other"
    echoed_message: str            = ""

    @property
    def risk_label(self) -> str:
        """LOW / MEDIUM / HIGH / CRITICAL based on risk_score 0-10."""
        if self.risk_score < 3.0:   return "LOW"
        elif self.risk_score < 6.0: return "MEDIUM"
        elif self.risk_score < 8.5: return "HIGH"
        else:                       return "CRITICAL"

    def to_numpy(self) -> np.ndarray:
        return np.array([
            self.amount, self.hour_of_day, self.location_match,
            self.transactions_last_hour, self.is_new_merchant, self.risk_score,
            self.step_num, self.merchant_category_index, self.velocity_ratio,
            self.time_since_last_txn, self.card_present,
            self.last_action_1, self.last_action_2, self.last_action_3,
        ], dtype=np.float32)

    def tolist(self) -> List[float]:
        return self.to_numpy().tolist()


def _zero_obs(step_num: float = 0.0) -> Observation:
    return Observation(step_num=step_num,
                       last_action_1=-1.0, last_action_2=-1.0, last_action_3=-1.0)


def _make_user_profile(rng: random.Random, fraud_rate: float) -> UserProfile:
    avg_amount     = rng.uniform(20.0, 500.0)
    start_hour     = rng.randint(7, 10)
    end_hour       = rng.randint(18, 22)
    usual_hours    = list(range(start_hour, end_hour + 1))
    num_cats       = rng.randint(1, 4)
    preferred_cats = rng.sample(range(len(MERCHANT_CATEGORIES)), num_cats)
    home_location  = rng.choice(LOCATIONS)
    return UserProfile(
        user_id=str(uuid.uuid4())[:8],
        avg_transaction_amount=avg_amount,
        usual_hours=usual_hours,
        preferred_categories=preferred_cats,
        home_location=home_location,
        fraud_rate=fraud_rate,
    )


def _make_transaction(
    rng: random.Random, profile: UserProfile, step: int,
    last_actions: List[int], is_fraud: bool, txn_history: List[Dict[str, Any]],
) -> Tuple[Observation, Dict[str, Any]]:
    if is_fraud:
        amount           = rng.uniform(profile.avg_transaction_amount * 2.0,
                                       profile.avg_transaction_amount * 8.0)
        hour             = rng.choice(
            [h for h in range(24) if h not in profile.usual_hours] or list(range(24))
        )
        location_match   = 0.0 if rng.random() < 0.75 else 1.0
        is_new_merchant  = float(rng.random() < 0.70)
        card_present     = float(rng.random() < 0.25)
        merchant_cat     = rng.choice(
            [c for c in range(len(MERCHANT_CATEGORIES))
             if c not in profile.preferred_categories]
            or list(range(len(MERCHANT_CATEGORIES)))
        )
    else:
        amount           = max(1.0, abs(rng.gauss(profile.avg_transaction_amount,
                                                  profile.avg_transaction_amount * 0.3)))
        hour             = (rng.choice(profile.usual_hours)
                            if rng.random() < 0.80 else rng.randint(0, 23))
        location_match   = 1.0 if rng.random() < 0.85 else 0.0
        is_new_merchant  = float(rng.random() < 0.15)
        card_present     = float(rng.random() < 0.75)
        merchant_cat     = (rng.choice(profile.preferred_categories)
                            if rng.random() < 0.75
                            else rng.randint(0, len(MERCHANT_CATEGORIES) - 1))

    recent               = txn_history[-5:] if len(txn_history) >= 5 else txn_history
    transactions_last_hr = float(len(recent))
    velocity_ratio       = amount / max(profile.avg_transaction_amount, 1.0)
    burst                = float(transactions_last_hr >= 4)
    time_since           = (rng.uniform(0.5, 5.0) if txn_history and is_fraud
                            else rng.uniform(1.0, 60.0) if txn_history
                            else rng.uniform(30.0, 1440.0))

    risk_score = _compute_risk_score(
        amount=amount, avg_amount=profile.avg_transaction_amount,
        hour=hour, usual_hours=profile.usual_hours,
        location_match=location_match, is_new_merchant=is_new_merchant,
        card_present=card_present, velocity_ratio=velocity_ratio, burst=burst,
    )

    padded = ([-1.0] * 3 + [float(a) for a in last_actions])[-3:]
    echo = (
        f"Step {step}: ${amount:.2f} at {hour:02d}:00 "
        f"| risk={risk_score:.1f}/10 | vel={velocity_ratio:.2f}x "
        f"| {'FRAUD' if is_fraud else 'LEGIT'}"
    )

    obs = Observation(
        amount=amount, hour_of_day=hour / 23.0, location_match=location_match,
        transactions_last_hour=transactions_last_hr, is_new_merchant=is_new_merchant,
        risk_score=risk_score, step_num=float(step),
        merchant_category_index=float(merchant_cat), velocity_ratio=velocity_ratio,
        time_since_last_txn=time_since, card_present=card_present,
        last_action_1=padded[0], last_action_2=padded[1], last_action_3=padded[2],
        merchant_category=MERCHANT_CATEGORIES[merchant_cat], echoed_message=echo,
    )

    metadata: Dict[str, Any] = {
        "was_fraud": is_fraud, "is_fraud": is_fraud,
        "amount": amount, "hour": hour, "location_match": location_match,
        "merchant_category": MERCHANT_CATEGORIES[merchant_cat],
        "risk_score": risk_score, "velocity_ratio": velocity_ratio,
        "card_present": card_present, "is_new_merchant": is_new_merchant,
        "burst": burst, "echoed_message": echo,
    }
    return obs, metadata


def _compute_risk_score(
    amount: float, avg_amount: float, hour: int, usual_hours: List[int],
    location_match: float, is_new_merchant: float, card_present: float,
    velocity_ratio: float, burst: float,
) -> float:
    score = 0.0
    if   velocity_ratio > 5.0: score += 3.0
    elif velocity_ratio > 3.0: score += 2.0
    elif velocity_ratio > 1.5: score += 1.0
    if hour not in usual_hours:  score += 1.5
    if location_match < 0.5:     score += 2.0
    score += is_new_merchant * 1.0
    if card_present < 0.5:       score += 0.5
    score += burst * 1.5
    return min(10.0, score)


class FraudDetectionEnv(gym.Env):
    """Financial Fraud Defender — Gymnasium-compliant RL environment."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, task: str = "easy", seed: Optional[int] = None):
        super().__init__()
        if task not in TASK_DEFINITIONS:
            raise ValueError(f"Unknown task '{task}'. Choose: {list(TASK_DEFINITIONS)}")
        self.task_name = task
        self.task_def  = TASK_DEFINITIONS[task]
        self._seed     = seed

        self.observation_space = spaces.Box(
            low=np.array([-1.0] * NUM_FEATURES, dtype=np.float32),
            high=np.array([10000.0, 1.0, 1.0, 20.0, 1.0, 10.0,
                           20.0, 7.0, 20.0, 1440.0, 1.0, 2.0, 2.0, 2.0],
                          dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

        self._rng:             random.Random            = random.Random(seed)
        self._np_rng:          np.random.Generator      = np.random.default_rng(seed)
        self._profile:         Optional[UserProfile]    = None
        self._step:            int                      = 0
        self._episode_reward:  float                    = 0.0
        self._current_obs:     Optional[Observation]    = None
        self._current_meta:    Optional[Dict[str, Any]] = None
        self._last_actions:    List[int]                = []
        self._txn_history:     List[Dict[str, Any]]     = []
        self._episode_history: List[Dict[str, Any]]     = []
        self._fraud_labels:    List[bool]               = []

    def reset(
        self,
        *,
        seed:       Optional[int]           = None,
        difficulty: Optional[str]           = None,
        options:    Optional[Dict[str, Any]] = None,
    ) -> Tuple[Observation, Dict[str, Any]]:
        if difficulty is not None:
            if difficulty not in TASK_DEFINITIONS:
                raise ValueError(f"Unknown difficulty '{difficulty}'.")
            self.task_name = difficulty
            self.task_def  = TASK_DEFINITIONS[difficulty]

        if seed is not None:
            self._seed = seed
        effective_seed = self._seed if self._seed is not None else random.randint(0, 2**31)
        self._rng    = random.Random(effective_seed)
        self._np_rng = np.random.default_rng(effective_seed)

        self._profile         = _make_user_profile(self._rng, self.task_def.fraud_rate)
        self._step            = 0
        self._episode_reward  = 0.0
        self._last_actions    = []
        self._txn_history     = []
        self._episode_history = []

        n_fraud      = int(round(NUM_STEPS * self.task_def.fraud_rate))
        fraud_labels = [True] * n_fraud + [False] * (NUM_STEPS - n_fraud)
        self._rng.shuffle(fraud_labels)
        self._fraud_labels = fraud_labels

        obs, meta = _make_transaction(
            self._rng, self._profile, self._step,
            self._last_actions, self._fraud_labels[self._step], self._txn_history,
        )
        self._current_obs  = obs
        self._current_meta = meta

        info: Dict[str, Any] = {
            "task": self.task_name, "step": self._step,
            "user_id": self._profile.user_id, "transaction": meta,
            "echoed_message": (
                f"Episode reset: task={self.task_name}, "
                f"user={self._profile.user_id}, seed={effective_seed}"
            ),
        }
        return obs, info

    def step(self, action: int) -> Tuple[Observation, float, bool, bool, Dict[str, Any]]:
        if self._profile is None:
            raise RuntimeError("Call reset() before step().")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Must be 0, 1, or 2.")

        meta        = self._current_meta
        is_fraud    = meta["was_fraud"]
        action_name = Action(action).name
        outcome     = "fraud" if is_fraud else "legit"
        reward      = REWARD_TABLE[action_name][outcome]

        self._episode_reward += reward
        self._last_actions.append(action)
        self._txn_history.append(meta)

        record: Dict[str, Any] = {
            "step": self._step, "action": action, "action_name": action_name,
            "was_fraud": is_fraud, "is_fraud": is_fraud,
            "reward": reward, "risk_score": meta["risk_score"],
            "amount": meta["amount"], "velocity_ratio": meta["velocity_ratio"],
        }
        self._episode_history.append(record)

        self._step  += 1
        terminated   = self._step >= NUM_STEPS
        truncated    = False
        echoed = (
            f"Step {self._step}: {action_name} -> {outcome} -> {reward:+.2f} "
            f"(cumulative {self._episode_reward:.2f})"
        )

        if terminated:
            obs = _zero_obs(float(NUM_STEPS))
            self._current_obs  = obs
            self._current_meta = {}
            info: Dict[str, Any] = {
                "task": self.task_name, "step": self._step,
                "was_fraud": is_fraud, "episode_reward": self._episode_reward,
                "episode_history": self._episode_history,
                "echoed_message": echoed + " [EPISODE DONE]",
            }
        else:
            obs, next_meta = _make_transaction(
                self._rng, self._profile, self._step,
                self._last_actions[-3:], self._fraud_labels[self._step],
                self._txn_history,
            )
            self._current_obs  = obs
            self._current_meta = next_meta
            info = {
                "task": self.task_name, "step": self._step,
                "was_fraud": is_fraud, "transaction": next_meta,
                "echoed_message": echoed,
            }

        return obs, reward, terminated, truncated, info

    def get_state(self) -> Dict[str, Any]:
        return {
            "step": self._step, "episode_reward": self._episode_reward,
            "task": self.task_name,
            "observation": self._current_obs.tolist() if self._current_obs else None,
            "episode_history": self._episode_history,
        }

    def grade_easy(self)   -> Dict[str, Any]: return compute_grade("easy",   self._episode_history)
    def grade_medium(self) -> Dict[str, Any]: return compute_grade("medium", self._episode_history)
    def grade_hard(self)   -> Dict[str, Any]: return compute_grade("hard",   self._episode_history)

    def render(self, mode: str = "human") -> None:
        if mode == "human" and self._current_meta:
            m = self._current_meta
            print(
                f"[Step {self._step:02d}] ${m.get('amount', 0):.2f}  "
                f"risk={m.get('risk_score', 0):.1f}  "
                f"fraud={'YES' if m.get('was_fraud') else 'NO'}  "
                f"total={self._episode_reward:.2f}"
            )


def compute_grade(task: str, episode_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Score a completed episode. Always returns score in [0.0, 1.0]."""
    task_def = TASK_DEFINITIONS[task]

    if not episode_history:
        return {
            "task": task, "score": 0.0, "passed": False,
            "metric": task_def.metric, "threshold": task_def.threshold,
            "details": {"tp": 0, "fp": 0, "fn": 0, "tn": 0,
                        "precision": 0.0, "recall": 0.0, "f1": 0.0,
                        "fpr": 0.0, "total_steps": 0},
        }

    y_true, y_pred = [], []
    for rec in episode_history:
        fraud = rec.get("was_fraud", rec.get("is_fraud", False))
        y_true.append(int(fraud))
        y_pred.append(0 if rec["action"] == Action.APPROVE else 1)

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)
    fpr       = fp / max(fp + tn, 1)

    if task == "easy":
        score, metric = recall, "recall"
    elif task == "medium":
        score, metric = f1, "f1"
    else:
        score, metric = max(0.0, f1 - 0.3 * fpr), "composite"

    score  = min(1.0, max(0.0, score))
    passed = score >= task_def.threshold

    return {
        "task": task, "score": round(score, 4), "passed": passed,
        "metric": metric, "threshold": task_def.threshold,
        "details": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "fpr": round(fpr, 4),
            "total_steps": len(episode_history),
        },
    }
