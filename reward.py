"""
OpenOps: Autonomous Customer Support Agent — Reward & Scoring

Evaluates the agent on:
  - Customer satisfaction (did we make them happy?)
  - Resolution quality (did we actually solve the problem?)
  - Policy compliance (did we follow the rules?)
  - Efficiency (did we resolve it quickly?)
"""

from __future__ import annotations

from models import SupportState, TicketCategory


def compute_step_reward(prev: SupportState, curr: SupportState) -> float:
    """Dense per-step reward."""
    delta_satisfaction = curr.customer_satisfaction - prev.customer_satisfaction
    delta_resolution = curr.resolution_quality - prev.resolution_quality
    delta_compliance = curr.policy_compliance - prev.policy_compliance

    reward = (
        0.35 * delta_satisfaction
        + 0.30 * delta_resolution
        + 0.20 * delta_compliance
        - 0.15 * 0.12  # efficiency cost per step
    )
    return reward


def compute_penalties(state: SupportState) -> float:
    """Penalties for bad outcomes."""
    penalty = 0.0

    # Closed without reply
    if state.ticket_closed and not state.reply_sent:
        penalty -= 0.5

    # Rejected eligible refund
    if state.refund_decided and not state.refund_approved and state.ticket.refund_eligible:
        penalty -= 0.3

    # Approved ineligible refund without checking policy
    if state.refund_approved and not state.ticket.refund_eligible and not state.policy_checked:
        penalty -= 0.4

    # Customer very angry
    if state.customer_satisfaction < 0.2:
        penalty -= 0.3

    return penalty


def compute_final_score(state: SupportState) -> float:
    """Final score [0, 1]."""
    score = (
        0.30 * state.customer_satisfaction
        + 0.25 * min(state.resolution_quality, 1.0)
        + 0.20 * state.policy_compliance
        + 0.15 * state.efficiency
        + 0.10 * (1.0 if state.ticket_closed and state.reply_sent else 0.0)
    )
    return max(0.0, min(1.0, score))


def compute_task_weighted_score(
    state: SupportState,
    weights: dict[str, float] | None = None,
) -> float:
    if weights is None:
        return compute_final_score(state)

    score = (
        weights.get("satisfaction", 0.30) * state.customer_satisfaction
        + weights.get("resolution", 0.25) * min(state.resolution_quality, 1.0)
        + weights.get("compliance", 0.20) * state.policy_compliance
        + weights.get("efficiency", 0.15) * state.efficiency
        + weights.get("closure", 0.10) * (1.0 if state.ticket_closed and state.reply_sent else 0.0)
    )
    return max(0.0, min(1.0, score))


def compute_baseline_comparison(
    agent_state: SupportState,
    baseline_state: SupportState,
    weights: dict[str, float] | None = None,
) -> float:
    agent_score = compute_task_weighted_score(agent_state, weights)
    baseline_score = compute_task_weighted_score(baseline_state, weights)
    diff = agent_score - baseline_score
    return max(0.0, min(1.0, (diff + 1.0) / 2.0))
