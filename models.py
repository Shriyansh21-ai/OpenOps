"""
OpenOps: Autonomous Customer Support Agent — Data Models

All models extend OpenEnv base types (Action, Observation, State).
"""

from __future__ import annotations

import random
from enum import IntEnum
from typing import Any, Dict, List, Optional

from pydantic import Field

from openenv.core.env_server import Action, Observation, State


# ---------------------------------------------------------------------------
# Ticket categories and priorities
# ---------------------------------------------------------------------------

class TicketCategory(IntEnum):
    REFUND = 0
    TECHNICAL = 1
    BILLING = 2
    ACCOUNT = 3
    COMPLAINT = 4
    GENERAL = 5


CATEGORY_NAMES = {int(c): c.name.lower() for c in TicketCategory}

class TicketPriority(IntEnum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    URGENT = 3


# ---------------------------------------------------------------------------
# Action Space
# ---------------------------------------------------------------------------

class ActionType(IntEnum):
    CLASSIFY_EMAIL = 0       # Identify the intent/category of the ticket
    QUERY_CUSTOMER_DB = 1    # Look up customer history, order details
    CHECK_POLICY = 2         # Check company policy for this situation
    APPROVE_REFUND = 3       # Approve a refund request
    REJECT_REFUND = 4        # Reject a refund request
    ESCALATE_TO_HUMAN = 5    # Escalate to human supervisor
    SEND_REPLY = 6           # Send a response to the customer
    REQUEST_MORE_INFO = 7    # Ask the customer for more details
    APPLY_DISCOUNT = 8       # Offer a discount/credit instead of refund
    CLOSE_TICKET = 9         # Mark ticket as resolved


ACTION_NAMES: Dict[int, str] = {int(a): a.name.lower() for a in ActionType}
NUM_ACTIONS = len(ActionType)


class SupportAction(Action):
    """Agent picks one action per step."""
    action: int = Field(
        ..., ge=0, lt=NUM_ACTIONS,
        description="Action index (0-9). See ActionType enum.",
    )


# ---------------------------------------------------------------------------
# Customer profile
# ---------------------------------------------------------------------------

class CustomerProfile(State):
    """Customer information — some fields hidden until queried."""
    customer_id: str = ""
    name: str = ""
    account_tier: str = "standard"        # standard, premium, enterprise
    tenure_months: int = 0
    total_spend: float = 0.0
    previous_tickets: int = 0
    previous_refunds: int = 0
    satisfaction_history: float = 0.5     # 0-1, hidden from agent
    is_churning_risk: bool = False        # hidden from agent


# ---------------------------------------------------------------------------
# Ticket
# ---------------------------------------------------------------------------

class Ticket(State):
    """A customer support ticket."""
    ticket_id: str = ""
    customer_id: str = ""
    subject: str = ""
    body: str = ""
    true_category: int = 0                # hidden — agent must classify
    true_priority: int = 1                # hidden — agent must assess
    order_id: str = ""
    order_amount: float = 0.0
    days_since_purchase: int = 0
    refund_eligible: bool = True          # hidden — agent must check policy
    sentiment: float = 0.5               # 0=angry, 1=happy — hidden
    requires_human: bool = False          # some tickets genuinely need escalation


# ---------------------------------------------------------------------------
# Full state (ground truth)
# ---------------------------------------------------------------------------

class SupportState(State):
    """Full ground-truth state of the support interaction."""
    ticket: Ticket = Field(default_factory=Ticket)
    customer: CustomerProfile = Field(default_factory=CustomerProfile)

    # Agent's progress
    classified: bool = False
    agent_classification: int = -1
    customer_queried: bool = False
    policy_checked: bool = False
    refund_decided: bool = False
    refund_approved: bool = False
    reply_sent: bool = False
    info_requested: bool = False
    discount_offered: bool = False
    escalated: bool = False
    ticket_closed: bool = False

    # Quality metrics
    customer_satisfaction: float = Field(default=0.5, ge=0.0, le=1.0)
    resolution_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    policy_compliance: float = Field(default=0.5, ge=0.0, le=1.0)
    efficiency: float = Field(default=1.0, ge=0.0, le=1.0)  # decays with steps

    step_num: int = 0
    task_id: str = ""

    def snapshot(self) -> Dict[str, Any]:
        d = self.model_dump()
        return d

    def clone(self) -> SupportState:
        return SupportState(**self.model_dump())


# ---------------------------------------------------------------------------
# Observation (partial observability)
# ---------------------------------------------------------------------------

class SupportObservation(Observation):
    """What the agent sees — customer sentiment and eligibility are hidden."""
    # Ticket info (always visible)
    ticket_id: str = ""
    subject: str = ""
    body: str = ""
    order_id: str = ""
    order_amount: float = 0.0
    days_since_purchase: int = 0

    # Customer info (only visible after query_customer_db)
    customer_known: bool = False
    customer_name: str = ""
    account_tier: str = ""
    tenure_months: int = 0
    total_spend: float = 0.0
    previous_tickets: int = 0
    previous_refunds: int = 0

    # Agent's progress
    classified: bool = False
    classification: str = ""
    policy_checked: bool = False
    policy_result: str = ""       # what the policy says (after check_policy)
    refund_decided: bool = False
    reply_sent: bool = False
    info_requested: bool = False
    discount_offered: bool = False
    escalated: bool = False

    # Estimated satisfaction (noisy)
    estimated_satisfaction: str = "neutral"  # angry, frustrated, neutral, satisfied, happy

    step_num: int = 0
    max_steps: int = 8
    task_description: str = ""

    # Supervisor/system messages
    system_messages: List[str] = Field(default_factory=list)


def make_observation(
    state: SupportState,
    rng: random.Random,
    task_description: str = "",
    done: bool = False,
    reward: float | None = None,
    system_messages: list[str] | None = None,
) -> SupportObservation:
    """Build partially-observable view of the state."""

    # Satisfaction estimate (noisy)
    true_sat = state.customer_satisfaction
    noise = rng.gauss(0, 0.15)
    noisy_sat = max(0.0, min(1.0, true_sat + noise))
    if noisy_sat < 0.25:
        sat_label = "angry"
    elif noisy_sat < 0.4:
        sat_label = "frustrated"
    elif noisy_sat < 0.6:
        sat_label = "neutral"
    elif noisy_sat < 0.8:
        sat_label = "satisfied"
    else:
        sat_label = "happy"

    # Policy result text
    policy_result = ""
    if state.policy_checked:
        if state.ticket.refund_eligible:
            if state.ticket.days_since_purchase <= 30:
                policy_result = "ELIGIBLE: Within 30-day return window. Full refund authorized."
            elif state.ticket.days_since_purchase <= 90:
                policy_result = "PARTIAL: Within 90-day window. Store credit or 50% refund authorized."
            else:
                policy_result = "INELIGIBLE: Outside return window. Discount or escalation recommended."
        else:
            policy_result = "INELIGIBLE: Item category excluded from refund policy. Offer alternative resolution."

    # Classification text
    classification = ""
    if state.classified:
        classification = CATEGORY_NAMES.get(state.agent_classification, "unknown")

    obs = SupportObservation(
        ticket_id=state.ticket.ticket_id,
        subject=state.ticket.subject,
        body=state.ticket.body,
        order_id=state.ticket.order_id,
        order_amount=state.ticket.order_amount,
        days_since_purchase=state.ticket.days_since_purchase,
        customer_known=state.customer_queried,
        customer_name=state.customer.name if state.customer_queried else "",
        account_tier=state.customer.account_tier if state.customer_queried else "",
        tenure_months=state.customer.tenure_months if state.customer_queried else 0,
        total_spend=state.customer.total_spend if state.customer_queried else 0.0,
        previous_tickets=state.customer.previous_tickets if state.customer_queried else 0,
        previous_refunds=state.customer.previous_refunds if state.customer_queried else 0,
        classified=state.classified,
        classification=classification,
        policy_checked=state.policy_checked,
        policy_result=policy_result,
        refund_decided=state.refund_decided,
        reply_sent=state.reply_sent,
        info_requested=state.info_requested,
        discount_offered=state.discount_offered,
        escalated=state.escalated,
        estimated_satisfaction=sat_label,
        step_num=state.step_num,
        max_steps=8,
        task_description=task_description,
        system_messages=system_messages or [],
        done=done,
        reward=reward,
    )
    return obs
