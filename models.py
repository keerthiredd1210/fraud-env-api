"""
models.py — Pydantic data models for Financial Fraud Defender RL Environment.
"""
from __future__ import annotations

from enum import IntEnum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Action(IntEnum):
    APPROVE = 0
    BLOCK = 1
    VERIFY = 2


REWARD_TABLE: Dict[str, Dict[str, float]] = {
    "APPROVE": {"legit": +0.50, "fraud": -1.00},
    "BLOCK":   {"fraud": +1.00, "legit": -0.50},
    "VERIFY":  {"fraud": +0.70, "legit": -0.10},
}


class UserProfile(BaseModel):
    user_id: str
    avg_transaction_amount: float = Field(ge=1.0)
    usual_hours: List[int]
    preferred_categories: List[int]
    home_location: str
    fraud_rate: float = Field(ge=0.0, le=1.0)


class TransactionFeatures(BaseModel):
    amount: float
    hour_of_day: float
    location_match: float
    transactions_last_hour: float
    is_new_merchant: float
    risk_score: float
    step: float
    merchant_category_index: float
    velocity_ratio: float
    time_since_last_txn: float
    card_present: float
    last_action_1: float
    last_action_2: float
    last_action_3: float
    echoed_message: Optional[str] = None


class ResetRequest(BaseModel):
    task: str = Field(default="easy", pattern="^(easy|medium|hard)$")
    seed: Optional[int] = None


class ResetResponse(BaseModel):
    observation: List[float]
    info: Dict[str, Any]
    echoed_message: str = ""


class StepRequest(BaseModel):
    action: int = Field(ge=0, le=2)


class StepResponse(BaseModel):
    observation: List[float]
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]
    echoed_message: str = ""


class StateResponse(BaseModel):
    step: int
    episode_reward: float
    task: str
    observation: Optional[List[float]]
    episode_history: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    version: str
    tasks: List[str]


class TaskDefinition(BaseModel):
    name: str
    fraud_rate: float
    metric: str
    threshold: float
    description: str


TASK_DEFINITIONS: Dict[str, TaskDefinition] = {
    "easy": TaskDefinition(
        name="easy",
        fraud_rate=0.15,
        metric="recall",
        threshold=0.65,
        description="15% fraud rate — maximise recall >= 0.65",
    ),
    "medium": TaskDefinition(
        name="medium",
        fraud_rate=0.30,
        metric="f1",
        threshold=0.70,
        description="30% fraud rate — maximise F1 >= 0.70",
    ),
    "hard": TaskDefinition(
        name="hard",
        fraud_rate=0.50,
        metric="composite",
        threshold=0.72,
        description="50% fraud rate — composite score >= 0.72 with FP penalty",
    ),
}


class GraderRequest(BaseModel):
    task: str = Field(pattern="^(easy|medium|hard)$")
    episode_history: List[Dict[str, Any]]


class GraderResponse(BaseModel):
    task: str
    score: float
    passed: bool
    metric: str
    threshold: float
    details: Dict[str, Any]


class BaselineRequest(BaseModel):
    task: str = Field(default="easy", pattern="^(easy|medium|hard)$")
    agent: str = Field(default="rule_based", pattern="^(rule_based|random|llm)$")
    seed: Optional[int] = None


class BaselineResponse(BaseModel):
    task: str
    agent: str
    total_reward: float
    steps: int
    score: float
    passed: bool
    details: Dict[str, Any]
