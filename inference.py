"""
OpenOps — Autonomous Customer Support Agent Inference Script
MANDATORY env vars: API_BASE_URL, MODEL_NAME, HF_TOKEN
STDOUT FORMAT: [START], [STEP], [END]
"""

import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai import OpenAI

from models import SupportAction, SupportObservation, ACTION_NAMES, NUM_ACTIONS
from environment import OpenOpsEnvironment

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_KEY = HF_TOKEN

BENCHMARK = "openops"
MAX_STEPS = 8
TASKS = ["easy_1", "medium_1", "hard_1"]


# ---------------------------------------------------------------------------
# Local environment wrapper
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    observation: SupportObservation
    reward: Optional[float] = None
    done: bool = False


class LocalEnv:
    def __init__(self) -> None:
        self._env = OpenOpsEnvironment()

    async def reset(self, **kwargs: Any) -> StepResult:
        obs = self._env.reset(**kwargs)
        return StepResult(observation=obs, reward=obs.reward, done=obs.done)

    async def step(self, action: SupportAction) -> StepResult:
        obs = self._env.step(action)
        return StepResult(observation=obs, reward=obs.reward, done=obs.done)

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an autonomous customer support agent. You handle support tickets by taking actions step by step.

## Actions (pick 0-9):
0: classify_email       — Identify the intent/category of the ticket
1: query_customer_db    — Look up customer history, account tier, spend
2: check_policy         — Check refund/return policy for this situation
3: approve_refund       — Approve a refund (verify eligibility first!)
4: reject_refund        — Reject a refund request
5: escalate_to_human    — Escalate to human supervisor (use only when needed)
6: send_reply           — Send a response to the customer
7: request_more_info    — Ask the customer for more details
8: apply_discount       — Offer a discount/credit as alternative resolution
9: close_ticket         — Mark ticket as resolved

## Key rules:
- ALWAYS classify the ticket first before other actions
- ALWAYS check policy before approving/rejecting refunds
- Query customer DB to understand their value and history
- Premium/enterprise customers need extra care
- Don't close without sending a reply
- Efficiency matters — fewer steps is better
- Some tickets REQUIRE human escalation (complex enterprise issues)

## Scoring:
30% customer satisfaction + 25% resolution quality + 20% policy compliance + 15% efficiency + 10% proper closure

Respond with ONLY: {"action": <0-9>, "reasoning": "<brief>"}
"""


# ---------------------------------------------------------------------------
# Observation formatting
# ---------------------------------------------------------------------------

def format_observation(obs: dict, step: int, history: List[str]) -> str:
    parts = []

    if step == 0:
        desc = obs.get("task_description", "")
        if desc:
            parts.append(f"## Task\n{desc}\n")

    parts.append(f"## Step {obs.get('step_num', 0)} / {obs.get('max_steps', 8)}")
    parts.append(f"\n## Ticket")
    parts.append(f"  ID: {obs.get('ticket_id', '')}")
    parts.append(f"  Subject: {obs.get('subject', '')}")
    parts.append(f"  Body: {obs.get('body', '')}")
    parts.append(f"  Order: {obs.get('order_id', '')} — ${obs.get('order_amount', 0):.2f}")
    parts.append(f"  Days since purchase: {obs.get('days_since_purchase', 0)}")

    if obs.get("customer_known"):
        parts.append(f"\n## Customer (queried)")
        parts.append(f"  Name: {obs.get('customer_name', '')}")
        parts.append(f"  Tier: {obs.get('account_tier', '')}")
        parts.append(f"  Tenure: {obs.get('tenure_months', 0)} months")
        parts.append(f"  Total spend: ${obs.get('total_spend', 0):,.0f}")
        parts.append(f"  Previous tickets: {obs.get('previous_tickets', 0)}")
        parts.append(f"  Previous refunds: {obs.get('previous_refunds', 0)}")
    else:
        parts.append(f"\n## Customer: Not yet queried (use action 1)")

    parts.append(f"\n## Progress")
    parts.append(f"  Classified: {obs.get('classified', False)} {('— ' + obs.get('classification', '')) if obs.get('classified') else ''}")
    parts.append(f"  Policy checked: {obs.get('policy_checked', False)} {('— ' + obs.get('policy_result', '')) if obs.get('policy_checked') else ''}")
    parts.append(f"  Refund decided: {obs.get('refund_decided', False)}")
    parts.append(f"  Reply sent: {obs.get('reply_sent', False)}")
    parts.append(f"  Escalated: {obs.get('escalated', False)}")
    parts.append(f"  Customer mood: {obs.get('estimated_satisfaction', 'neutral')}")

    msgs = obs.get("system_messages", [])
    if msgs:
        parts.append(f"\n## System Messages")
        for m in msgs:
            parts.append(f"  > {m}")

    if history:
        parts.append(f"\n## Your actions so far")
        for h in history[-4:]:
            parts.append(f"  {h}")

    parts.append('\nRespond: {"action": <0-9>, "reasoning": "..."}')
    return "\n".join(parts)


def parse_response(text: str) -> int:
    json_match = re.search(r'\{[^}]*"action"\s*:\s*(\d)[^}]*\}', text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            action = int(data.get("action", 9))
            if 0 <= action < NUM_ACTIONS:
                return action
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
    return 9  # fallback: close_ticket


# ---------------------------------------------------------------------------
# Run one task
# ---------------------------------------------------------------------------

async def run_task(env, task_id: str, client: OpenAI) -> float:
    history, messages, rewards = [], [], []
    steps_taken, score = 0, 0.5

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_id=task_id)
        obs = result.observation.model_dump()

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            user_msg = format_observation(obs, step - 1, history)
            messages.append({"role": "user", "content": user_msg})

            try:
                comp = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages[-6:],
                    max_tokens=300, temperature=0.3, stream=False,
                )
                llm_text = (comp.choices[0].message.content or "").strip()
            except Exception as exc:
                print(f"[DEBUG] LLM error: {exc}", flush=True)
                llm_text = '{"action": 9, "reasoning": "API error fallback"}'

            messages.append({"role": "assistant", "content": llm_text})
            action_idx = parse_response(llm_text)
            action_name = ACTION_NAMES.get(action_idx, str(action_idx))

            result = await env.step(SupportAction(action=action_idx))
            obs = result.observation.model_dump()
            reward = result.reward or 0.0
            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_name, reward=reward, done=result.done, error=None)
            history.append(f"Step {step}: {action_name} -> reward {reward:+.2f}")

            if result.done:
                meta = result.observation.metadata or {}
                score = meta.get("comparison_score", 0.5)
                break

        if not result.done:
            score = 0.5

    except Exception as exc:
        print(f"[DEBUG] Task {task_id} error: {exc}", flush=True)
        score = 0.0

    finally:
        score = min(max(score, 0.0), 1.0)
        log_end(success=score >= 0.5, steps=steps_taken, score=score, rewards=rewards)

    return score


# ---------------------------------------------------------------------------
# Create environment (Docker → HF Space → Local fallback)
# ---------------------------------------------------------------------------

async def create_env():
    if LOCAL_IMAGE_NAME:
        try:
            from client import OpenOpsEnv
            env = await OpenOpsEnv.from_docker_image(LOCAL_IMAGE_NAME)
            return env
        except Exception:
            pass

    hf_space_url = os.getenv("HF_SPACE_URL", "")
    if hf_space_url:
        try:
            from client import OpenOpsEnv
            env = OpenOpsEnv(base_url=hf_space_url)
            await env.connect()
            return env
        except Exception:
            pass

    return LocalEnv()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = await create_env()
    try:
        for task_id in TASKS:
            await run_task(env, task_id, client)
    finally:
        try:
            await env.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
