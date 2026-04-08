from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Body
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


def _build_gradio_app():
    import gradio as gr

    store = {"env": None}

    def reset_fn(task, seed):
        env = FraudDetectionEnv(task=task, seed=int(seed))
        env.reset()
        store["env"] = env
        return "Episode started"

    def step_fn(action):
        env = store.get("env")
        if not env:
            return "Reset first"
        action_map = {"APPROVE": 0, "BLOCK": 1, "VERIFY": 2}
        _, reward, *_ = env.step(action_map[action])
        return f"{action} → {reward:+.2f}"

    def auto_fn(task):
        res = run_episode(task=task, agent="rule_based")
        return f"Score: {res['score']:.4f}"

    with gr.Blocks() as demo:
        gr.Markdown("# 🛡️ Financial Fraud Defender")

        with gr.Tab("Manual"):
            task = gr.Dropdown(["easy", "medium", "hard"], value="easy")
            seed = gr.Textbox(value="42")
            btn = gr.Button("Reset")
            out = gr.Textbox()
            btn.click(reset_fn, [task, seed], out)

            action = gr.Dropdown(["APPROVE", "BLOCK", "VERIFY"])
            btn2 = gr.Button("Step")
            out2 = gr.Textbox()
            btn2.click(step_fn, action, out2)

        with gr.Tab("Auto"):
            task2 = gr.Dropdown(["easy", "medium", "hard"])
            btn3 = gr.Button("Run")
            out3 = gr.Textbox()
            btn3.click(auto_fn, task2, out3)

        with gr.Tab("About"):
            gr.Markdown("Fraud detection RL demo")

    return demo


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting API on port 7860...", flush=True)
    import gradio as gr
    gradio_app = _build_gradio_app()
    gr.mount_gradio_app(app, gradio_app, path="/demo")
    print("Gradio app mounted.", flush=True)
    yield
    print("Shutting down.", flush=True)


app = FastAPI(title="Financial Fraud Defender", lifespan=lifespan)


# ---------------- ROOT ----------------

@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse("""
    <h1>Financial Fraud Defender</h1>
    <ul>
      <li><a href="./demo/">Interactive Gradio Demo</a></li>
      <li><a href="/docs">API Docs</a></li>
      <li><a href="/health">Health</a></li>
      <li><a href="/tasks">Tasks</a></li>
    </ul>
    """)


# ---------------- API ----------------

@app.post("/reset", response_model=ResetResponse)
def reset(req: Optional[ResetRequest] = Body(default=None)):
    global _env, _current_obs
    if req is None:
        req = ResetRequest()
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
    return HealthResponse(
        status="ok",
        version="1.0.0",
        tasks=list(TASK_DEFINITIONS.keys())
    )


@app.get("/tasks")
def tasks():
    return {
        name: {
            "fraud_rate": td.fraud_rate,
            "metric": td.metric,
            "threshold": td.threshold,
            "description": td.description,
        }
        for name, td in TASK_DEFINITIONS.items()
    }


@app.post("/grader", response_model=GraderResponse)
def grader(req: GraderRequest):
    if not req.episode_history:
        raise HTTPException(status_code=422, detail="Empty history.")
    result = compute_grade(req.task, req.episode_history)
    return GraderResponse(**result)


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

        grade = compute_grade(task, env._episode_history)
        results[task] = grade

    return {
        "tasks": results,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# ---------------- RUN ----------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
