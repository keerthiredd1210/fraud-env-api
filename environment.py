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
        obs = Observation(step_num=0.0)
        return obs.to_numpy(), {}

    def step(self, action):
        is_fraud = self._rng.random() < self.task_def.fraud_rate
        
        # Reward logic based on action
        if action == 0:    # APPROVE
            reward = 0.5 if not is_fraud else -1.0
        elif action == 1:  # BLOCK
            reward = 1.0 if is_fraud else -0.5
        elif action == 2:  # VERIFY
            reward = 0.7 if is_fraud else -0.1
        else:
            reward = 0.0

        # Uncertainty bonus for VERIFY in mid-risk scenarios
        if action == 2 and 0.4 <= reward <= 0.7:
            reward += 0.1

        self._episode_reward += reward
        self._step += 1
        
        # Store history for grading (tracking raw action and fraud status)
        self._history.append({
            "action": action,
            "is_fraud": is_fraud,
            "reward": reward
        })

        done = self._step >= NUM_STEPS

        # Generate meaningful risk features
        current_vel = self._rng.uniform(0.5, 5)
        obs = Observation(
            amount=self._rng.uniform(10, 1000),
            risk_score=min(10, (
                3 * (1 if self._rng.choice([0.0, 1.0]) == 0 else 0) +
                2 * (current_vel > 2.5) +
                self._rng.uniform(0, 2)
            )),
            velocity_ratio=current_vel,
            location_match=self._rng.choice([0.0, 1.0]),
            transactions_last_hour=self._rng.uniform(0, 10),
            card_present=self._rng.choice([0.0, 1.0]),
            step_num=float(self._step)
        )

        return obs.to_numpy(), reward, done, False, {}

  def get_state(self):
      return {
        "step": self._step,
        "episode_reward": self._episode_reward,
        "task": self.task_name,
        "observation": None,
        "episode_history": self._history
    }


def compute_grade(task, history):
    if not history:
        return {"task": task, "score": 0.001, "passed": False}

    tp = fp = fn = tn = 0
    for entry in history:
        action = entry["action"]
        fraud = entry["is_fraud"]

        if action == 1 and fraud: 
            tp += 1
        elif action == 1 and not fraud: 
            fp += 1
        elif action == 0 and fraud: 
            fn += 1
        elif action == 0 and not fraud: 
            tn += 1
        elif action == 2 and fraud:
            tp += 0.7
        elif action == 2 and not fraud:
            fp += 0.1

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    # Grading logic based on task difficulty
    if task == "easy":
        score = recall
        threshold = 0.65
    elif task == "medium":
        score = f1
        threshold = 0.70
    else:  # hard
        fpr = fp / (fp + tn + 1e-6)
        score = f1 - 0.3 * fpr
        threshold = 0.72

    score = float(min(0.999, max(0.001, score)))

    return {
        "task": task,
        "score": score,
        "passed": score >= threshold,
        "metric": "statistical_f1",
        "threshold": threshold,
        "details": {
            "precision": round(precision, 3), 
            "recall": round(recall, 3), 
            "f1": round(f1, 3)
        },
    }
