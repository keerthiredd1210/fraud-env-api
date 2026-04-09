"""
environment.py — Financial Fraud Defender RL Environment.

Observation space : Box(14,) float32  (via Observation.to_numpy())
Action space      : Discrete(3)       0=APPROVE 1=BLOCK 2=VERIFY
Episode length    : 20 steps
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from models import (
    Action,
    REWARD_TABLE,
    TASK_DEFINITIONS,
    TransactionFeatures,
    UserProfile,
)

NUM_STEPS = 20
NUM_FEATURES = 14

MERCHANT_CATEGORIES = [
    "groceries", "travel", "entertainment", "dining",
    "electronics", "healthcare", "fuel", "other"
]

LOCATIONS = ["NYC", "LAX", "CHI", "HOU", "PHX", "PHL", "SAN", "DAL"]


@dataclass
class Observation:
    amount: float = 0.0
    hour_of_day: float = 0.0
    location_match: float = 0.0
    transactions_last_hour: float = 0.0
    is_new_merchant: float = 0.0
    risk_score: float = 0.0
    step_num: float = 0.0
    merchant_category_index: float = 0.0
    velocity_ratio: float = 0.0
    time_since_last_txn: float = 0.0
    card_present: float = 0.0
    last_action_1: float = -1.0
    last_action_2: float = -1.0
    last_action_3: float = -1.0
    merchant_category: str = "other"
    echoed_message: str = ""

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
    return Observation(step_num=step_num)


def _make_user_profile(rng: random.Random, fraud_rate: float) -> UserProfile:
    avg_amount = rng.uniform(20.0, 500.0)
    return UserProfile(
        user_id=str(uuid.uuid4())[:8],
        avg_transaction_amount=avg_amount,
        usual_hours=list(range(8, 20)),
        preferred_categories=rng.sample(range(len(MERCHANT_CATEGORIES)), 2),
        home_location=rng.choice(LOCATIONS),
        fraud_rate=fraud_rate,
    )


def _compute_risk_score(amount, avg, hour, usual, loc, new, card, vel, burst):
    score = 0
    if vel > 5: score += 3
    elif vel > 3: score += 2
    elif vel > 1.5: score += 1
    if hour not in usual: score += 1.5
    if loc < 0.5: score += 2
    score += new
    if card < 0.5: score += 0.5
    score += burst * 1.5
    return min(10, score)


class FraudDetectionEnv:
    def __init__(self, task="easy", seed=None):
        import gymnasium as gym
        from gymnasium import spaces

        self.spaces = spaces

        self.task_name = task
        self.task_def = TASK_DEFINITIONS[task]
        self._rng = random.Random(seed)

        self.observation_space = self.spaces.Box(
            low=np.array([-1.0] * NUM_FEATURES, dtype=np.float32),
            high=np.array([10000.0] * NUM_FEATURES, dtype=np.float32),
            dtype=np.float32,
        )

        self.action_space = self.spaces.Discrete(3)

        self._step = 0
        self._episode_reward = 0.0
        self._history = []

    def reset(self, seed=None, options=None):
        self._step = 0
        self._episode_reward = 0.0
        self._history = []
        obs = Observation()
        return obs, {}

    def step(self, action):
        # ✅ FIX: Step logic uses REWARD_TABLE and task fraud_rate
        action_name = Action(action).name
        fraud_rate = self.task_def.fraud_rate
        is_fraud = self._rng.random() < fraud_rate
        outcome = "fraud" if is_fraud else "legit"

        reward = REWARD_TABLE[action_name][outcome]

        # ✅ FIX: Store history for grading
        self._history.append({
            "action": action_name,
            "outcome": outcome,
            "reward": reward,
        })

        self._episode_reward += reward
        self._step += 1
        done = self._step >= NUM_STEPS
        obs = Observation(step_num=float(self._step))

        return obs, reward, done, False, {}

    def get_state(self):
        return {
            "step": self._step,
            "reward": self._episode_reward,
        }


# ✅ FIX: Reward-based grading with clamping
def compute_grade(task, history):
    if len(history) == 0:
        score = 0.0010
    else:
        total_reward = sum(h["reward"] for h in history)
        max_possible = len(history) * 1.0
        score = total_reward / max_possible

    # Clamping score between 0.001 and 0.999
    score = min(0.999, max(0.001, score))

    return {
        "task": task,
        "score": score,
        "passed": score > 0.5,
        "metric": "reward_based",
        "threshold": 0.5,
        "details": {},
    }
