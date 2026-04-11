"""
OpenOps: Autonomous Customer Support Agent — Task Definitions

Three scenarios:
  Easy:   Simple refund request within policy
  Medium: Ambiguous complaint, premium customer, tricky policy edge case
  Hard:   Multi-issue ticket, churn-risk customer, requires judgment
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict

from pydantic import BaseModel, Field, model_validator

from models import (
    CustomerProfile,
    SupportState,
    Ticket,
    TicketCategory,
    TicketPriority,
)


class TaskConfig(BaseModel):
    task_id: str
    description: str
    initial_state: SupportState
    scoring_weights: Dict[str, float]
    max_steps: int = 8
    seed: int = 0

    @model_validator(mode="after")
    def _derive_seed(self) -> "TaskConfig":
        h = hashlib.sha256(self.task_id.encode()).hexdigest()
        self.seed = int(h[:8], 16)
        return self


TASKS: Dict[str, TaskConfig] = {}


def _register(cfg: TaskConfig) -> None:
    TASKS[cfg.task_id] = cfg


# --- EASY: Simple Refund ---
_register(TaskConfig(
    task_id="easy_1",
    description=(
        "TICKET: Standard Refund Request\n"
        "A customer wants a refund for a recent purchase. The request "
        "appears straightforward. Handle the ticket following proper procedure: "
        "classify, check customer info, verify policy, make a decision, "
        "and respond to the customer.\n"
        "GOAL: Resolve efficiently while following policy."
    ),
    initial_state=SupportState(
        ticket=Ticket(
            ticket_id="TKT-10042",
            customer_id="C-8821",
            subject="Refund request for order #ORD-7734",
            body=(
                "Hi, I purchased a wireless keyboard 5 days ago (order #ORD-7734, $49.99) "
                "but it stopped working after 2 days. The 'A' key doesn't register. "
                "I'd like a full refund please. I've already tried resetting it. "
                "Receipt attached."
            ),
            true_category=int(TicketCategory.REFUND),
            true_priority=int(TicketPriority.MEDIUM),
            order_id="ORD-7734",
            order_amount=49.99,
            days_since_purchase=5,
            refund_eligible=True,
            sentiment=0.4,
            requires_human=False,
        ),
        customer=CustomerProfile(
            customer_id="C-8821",
            name="Emily Rodriguez",
            account_tier="standard",
            tenure_months=8,
            total_spend=320.0,
            previous_tickets=1,
            previous_refunds=0,
            satisfaction_history=0.7,
            is_churning_risk=False,
        ),
        customer_satisfaction=0.4,
        policy_compliance=0.5,
    ),
    scoring_weights={
        "satisfaction": 0.25,
        "resolution": 0.30,
        "compliance": 0.25,
        "efficiency": 0.15,
        "closure": 0.05,
    },
))


# --- MEDIUM: Ambiguous Complaint, Premium Customer ---
_register(TaskConfig(
    task_id="medium_1",
    description=(
        "TICKET: Ambiguous Complaint from Premium Customer\n"
        "A premium customer has submitted a vague complaint about a recent order. "
        "The issue isn't clearly a refund request — it could be a quality complaint, "
        "a feature request, or a billing dispute. The customer has been loyal but "
        "seems frustrated. You need to figure out what they actually want and "
        "handle it appropriately.\n"
        "GOAL: Correctly identify the issue, handle the premium customer with care, "
        "and find the right resolution without violating policy."
    ),
    initial_state=SupportState(
        ticket=Ticket(
            ticket_id="TKT-10043",
            customer_id="C-3344",
            subject="Very disappointed with recent experience",
            body=(
                "I've been a customer for 3 years and I've never had an issue until now. "
                "I ordered the ProMax headphones 45 days ago and they're already falling apart. "
                "The ear cushion is peeling, the Bluetooth disconnects every 20 minutes, "
                "and the noise cancellation barely works. For $199, this is unacceptable. "
                "I want this made right. I've recommended your products to dozens of people "
                "and I'm starting to regret it."
            ),
            true_category=int(TicketCategory.REFUND),
            true_priority=int(TicketPriority.HIGH),
            order_id="ORD-5521",
            order_amount=199.00,
            days_since_purchase=45,
            refund_eligible=True,  # within 90-day window but only partial
            sentiment=0.2,
            requires_human=False,
        ),
        customer=CustomerProfile(
            customer_id="C-3344",
            name="David Chang",
            account_tier="premium",
            tenure_months=36,
            total_spend=4200.0,
            previous_tickets=3,
            previous_refunds=0,
            satisfaction_history=0.85,
            is_churning_risk=True,
        ),
        customer_satisfaction=0.25,
        policy_compliance=0.5,
    ),
    scoring_weights={
        "satisfaction": 0.35,    # premium customer satisfaction matters more
        "resolution": 0.25,
        "compliance": 0.20,
        "efficiency": 0.10,
        "closure": 0.10,
    },
))


# --- HARD: Multi-Issue, Churn Risk, Requires Judgment ---
_register(TaskConfig(
    task_id="hard_1",
    description=(
        "TICKET: Complex Multi-Issue Complaint\n"
        "An enterprise customer has submitted an escalation with multiple issues: "
        "a billing overcharge, a product defect, AND a complaint about previous "
        "support interactions. The customer is threatening to cancel their enterprise "
        "contract. The refund amount exceeds standard policy limits. Some issues "
        "may require human escalation while others can be handled autonomously.\n"
        "GOAL: Navigate a complex situation with competing priorities. Protect "
        "the enterprise relationship while maintaining policy compliance. "
        "Not everything can be resolved — prioritize what matters most."
    ),
    initial_state=SupportState(
        ticket=Ticket(
            ticket_id="TKT-10044",
            customer_id="C-1001",
            subject="URGENT: Multiple unresolved issues — considering contract cancellation",
            body=(
                "This is my THIRD attempt to get these issues resolved. "
                "1) I was overcharged $847 on invoice INV-2024-3391 — we agreed on "
                "enterprise pricing but I was billed retail rates. "
                "2) The batch of 50 wireless mice (order #ORD-9102, $2,450) arrived "
                "with 15 units DOA. That's a 30% defect rate. "
                "3) My previous two tickets (#TKT-9987, #TKT-10011) were both closed "
                "without resolution. I spoke to agents who promised callbacks that never came. "
                "I have 200 employees depending on this equipment and I'm evaluating "
                "competitor contracts RIGHT NOW. Fix this or we're done."
            ),
            true_category=int(TicketCategory.COMPLAINT),
            true_priority=int(TicketPriority.URGENT),
            order_id="ORD-9102",
            order_amount=2450.00,
            days_since_purchase=12,
            refund_eligible=True,
            sentiment=0.1,
            requires_human=True,  # enterprise escalation needed
        ),
        customer=CustomerProfile(
            customer_id="C-1001",
            name="Sarah Mitchell, VP Operations @ TechCorp Inc.",
            account_tier="enterprise",
            tenure_months=24,
            total_spend=87500.0,
            previous_tickets=7,
            previous_refunds=1,
            satisfaction_history=0.4,
            is_churning_risk=True,
        ),
        customer_satisfaction=0.15,
        policy_compliance=0.5,
    ),
    scoring_weights={
        "satisfaction": 0.30,
        "resolution": 0.20,
        "compliance": 0.15,
        "efficiency": 0.10,
        "closure": 0.25,    # properly closing/escalating matters most
    },
))


def get_task(task_id: str) -> TaskConfig:
    if task_id not in TASKS:
        available = ", ".join(sorted(TASKS.keys()))
        raise KeyError(f"Unknown task_id '{task_id}'. Available: {available}")
    return TASKS[task_id]


def list_tasks() -> list[dict[str, Any]]:
    return [
        {"task_id": c.task_id, "description": c.description, "max_steps": c.max_steps}
        for c in TASKS.values()
    ]
