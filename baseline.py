"""
OpenOps: Autonomous Customer Support Agent — Baseline Policies
"""

from __future__ import annotations

import random

from models import ActionType, SupportState
from dynamics import step_dynamics
from tasks import get_task


def naive_policy(state: SupportState, step: int) -> int:
    """Fixed rotation: classify → query → check_policy → approve → send_reply → close."""
    rotation = [
        ActionType.CLASSIFY_EMAIL,
        ActionType.QUERY_CUSTOMER_DB,
        ActionType.CHECK_POLICY,
        ActionType.APPROVE_REFUND,
        ActionType.SEND_REPLY,
        ActionType.CLOSE_TICKET,
        ActionType.CLOSE_TICKET,
        ActionType.CLOSE_TICKET,
    ]
    return rotation[min(step, len(rotation) - 1)]


BASELINE_POLICIES = {"naive": naive_policy}


def run_baseline(task_id: str, policy_name: str = "naive") -> SupportState:
    task = get_task(task_id)
    state = task.initial_state.clone()
    rng = random.Random(task.seed)
    policy = BASELINE_POLICIES[policy_name]

    for step in range(task.max_steps):
        if state.ticket_closed:
            break
        action = policy(state, step)
        step_dynamics(state, action, rng)

    return state
