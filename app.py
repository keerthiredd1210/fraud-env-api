from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from environment import FraudDetectionEnv, compute_grade
from inference import run_episode
from models import (
    Action,
    GraderRequest,
    GraderResponse,
    HealthResponse,
    ResetRequest,
    ResetResponse,
    StateResponse,
    StepRequest,
    StepResponse,
    TASK_DEFINITIONS,
)

_env: Optional[FraudDetectionEnv] = None
_current_obs: Optional[List[float]] = None


def _get_env() -> FraudDetectionEnv:
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    return _env


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("API starting...", flush=True)
    yield
    print("API stopping...", flush=True)


app = FastAPI(title="Financial Fraud Defender", version="1.0.0", lifespan=lifespan)

# ---------------- ROOT ----------------

@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse("""
    <h1>🛡️ Financial Fraud Defender</h1>
    <ul>
        <li><a href="./demo/">🚀 Launch Demo</a></li>
        <li><a href="/docs">Docs</a></li>
        <li><a href="/health">Health</a></li>
        <li><a href="/tasks">Tasks</a></li>
    </ul>
    """)

# ---------------- API ----------------

@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    global _env, _current_obs
    _env = FraudDetectionEnv(task=req.task, seed=req.seed)
    obs, info = _env.reset(seed=req.seed)
    _current_obs = obs.tolist()
    return ResetResponse(observation=_current_obs, info=info, echoed_message="")

@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    env = _get_env()
    global _current_obs
    obs, reward, terminated, truncated, info = env.step(req.action)
    _current_obs = obs.tolist()
    return StepResponse(
        observation=_current_obs,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        info=info,
        echoed_message=""
    )

@app.get("/state", response_model=StateResponse)
def state():
    env = _get_env()
    return StateResponse(**env.get_state())

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", version="1.0.0", tasks=list(TASK_DEFINITIONS.keys()))

# ✅ FIXED tasks format (validator-safe)
@app.get("/tasks")
def tasks():
    return {
        "tasks": {
            name: {
                "fraud_rate": td.fraud_rate,
                "metric": td.metric,
                "threshold": td.threshold,
                "description": td.description,
                "action_schema": {
                    "field": "action",
                    "type": "int",
                    "values": {"0": "APPROVE", "1": "BLOCK", "2": "VERIFY"}
                }
            }
            for name, td in TASK_DEFINITIONS.items()
        }
    }

@app.post("/grader", response_model=GraderResponse)
def grader(req: GraderRequest):
    if not req.episode_history:
        raise HTTPException(status_code=422, detail="Empty history.")
    return GraderResponse(**compute_grade(req.task, req.episode_history))

@app.post("/baseline")
def baseline():
    from inference import rule_based_action

    results = {}
    for task in ("easy", "medium", "hard"):
        env = FraudDetectionEnv(task=task, seed=42)
        obs, _ = env.reset(seed=42)
        obs = obs.tolist()

        for _ in range(5):
            action = rule_based_action(obs)
            obs, *_ = env.step(action)
            obs = obs.tolist()

        results[task] = compute_grade(task, env._episode_history)

    return {
    "tasks": results,
    "model": "rule-based-fallback",
    "timestamp": datetime.now(timezone.utc).isoformat()
}

# ---------------- GRADIO ----------------

def _build_gradio_app():
    import gradio as gr
    import pandas as pd

    store = {"env": None, "history": [], "rewards": []}

    def reset_fn(task, seed):
        env = FraudDetectionEnv(task=task, seed=int(seed))
        obs, _ = env.reset()

        store["env"] = env
        store["history"] = []
        store["rewards"] = []

        return (
            f"Observation:\n{obs.tolist()}",
            pd.DataFrame(columns=["Step", "Action", "Reward"]),
            pd.DataFrame({"reward": []})
        )

    def step_fn(action):
        env = store["env"]
        if env is None:
            return "Reset first", None, None

        action_map = {"APPROVE": 0, "BLOCK": 1, "VERIFY": 2}
        obs, reward, terminated, truncated, _ = env.step(action_map[action])

        step_num = len(store["history"]) + 1
        store["history"].append({
            "Step": step_num,
            "Action": action,
            "Reward": reward
        })
        store["rewards"].append(reward)

        return (
            f"{action} → {reward:+.2f}\nObs: {obs.tolist()}",
            pd.DataFrame(store["history"]),
            pd.DataFrame({"reward": store["rewards"]})
        )

    def auto_fn(task):
        res = run_episode(task=task, agent="rule_based")

        df = pd.DataFrame(res["episode_history"])
        rewards = [h["reward"] for h in res["episode_history"]]

        return (
            f"Score: {res['score']:.4f}\nReward: {res['total_reward']:.2f}",
            df,
            pd.DataFrame({"reward": rewards})
        )

    with gr.Blocks() as demo:
        gr.Markdown("# 🛡️ Financial Fraud Defender")

        with gr.Tab("Manual"):
            task = gr.Dropdown(["easy", "medium", "hard"], value="easy")
            seed = gr.Textbox(value="42")

            reset_btn = gr.Button("Reset")
            state = gr.Textbox(lines=4)

            action = gr.Dropdown(["APPROVE", "BLOCK", "VERIFY"])
            step_btn = gr.Button("Step")

            result = gr.Textbox(lines=4)

            gr.Markdown("### 📋 History")
            table = gr.Dataframe()

            gr.Markdown("### 📊 Reward Graph")
            plot = gr.LinePlot()

            reset_btn.click(reset_fn, [task, seed], [state, table, plot])
            step_btn.click(step_fn, action, [result, table, plot])

        with gr.Tab("Auto"):
            task2 = gr.Dropdown(["easy", "medium", "hard"])
            run_btn = gr.Button("Run")

            auto_out = gr.Textbox()
            auto_table = gr.Dataframe()
            auto_plot = gr.LinePlot()

            run_btn.click(auto_fn, task2, [auto_out, auto_table, auto_plot])

    return demo


import gradio as gr
app = gr.mount_gradio_app(app, _build_gradio_app(), path="/demo")

# ---------------- RUN ----------------
def serve():
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, log_level="info")

if __name__ == "__main__":
    serve()
