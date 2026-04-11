"""
OpenOps: Autonomous Customer Support Agent — Transition Dynamics

Simulates the customer support interaction:
  - Agent actions affect ticket state, customer satisfaction, compliance
  - Customer satisfaction evolves based on agent behavior
  - Wrong decisions (refund when ineligible, reject when eligible) have consequences
  - Efficiency decays with each step (faster resolution = better)
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

from models import (
    ActionType,
    CATEGORY_NAMES,
    SupportState,
    TicketCategory,
)


# ---------------------------------------------------------------------------
# Satisfaction dynamics
# ---------------------------------------------------------------------------

def _update_satisfaction(state: SupportState, delta: float, rng: random.Random) -> None:
    """Adjust customer satisfaction with small noise."""
    noise = rng.gauss(0, 0.02)
    state.customer_satisfaction = max(0.0, min(1.0, state.customer_satisfaction + delta + noise))


# ---------------------------------------------------------------------------
# Apply action
# ---------------------------------------------------------------------------

def apply_action(
    state: SupportState,
    action: int,
    rng: random.Random,
) -> List[str]:
    """
    Apply agent action. Returns system messages (feedback to agent).
    Mutates state in place.
    """
    a = ActionType(action)
    messages: List[str] = []

    if a == ActionType.CLASSIFY_EMAIL:
        if state.classified:
            messages.append("SYSTEM: Ticket already classified. Redundant action.")
            _update_satisfaction(state, -0.02, rng)
        else:
            # Agent classifies correctly ~80% of the time based on info available
            correct = state.ticket.true_category
            if rng.random() < 0.85:
                state.agent_classification = correct
                messages.append(f"SYSTEM: Ticket classified as '{CATEGORY_NAMES[correct]}'.")
            else:
                # Misclassification
                wrong = rng.choice([c for c in TicketCategory if c != correct])
                state.agent_classification = int(wrong)
                messages.append(f"SYSTEM: Ticket classified as '{CATEGORY_NAMES[int(wrong)]}'.")
            state.classified = True

    elif a == ActionType.QUERY_CUSTOMER_DB:
        if state.customer_queried:
            messages.append("SYSTEM: Customer data already retrieved. Redundant action.")
            _update_satisfaction(state, -0.02, rng)
        else:
            state.customer_queried = True
            tier = state.customer.account_tier
            tenure = state.customer.tenure_months
            messages.append(
                f"SYSTEM: Customer profile loaded. {state.customer.name} — "
                f"{tier} tier, {tenure} months tenure, "
                f"${state.customer.total_spend:,.0f} total spend, "
                f"{state.customer.previous_tickets} prior tickets."
            )
            # Premium/enterprise customers expect faster service
            if tier in ("premium", "enterprise"):
                messages.append(
                    f"NOTE: {tier.upper()} customer — prioritize resolution quality."
                )

    elif a == ActionType.CHECK_POLICY:
        if state.policy_checked:
            messages.append("SYSTEM: Policy already checked. Redundant action.")
        else:
            state.policy_checked = True
            if state.ticket.refund_eligible:
                if state.ticket.days_since_purchase <= 30:
                    messages.append("POLICY: Full refund eligible (within 30-day window).")
                elif state.ticket.days_since_purchase <= 90:
                    messages.append("POLICY: Partial refund or store credit (30-90 day window).")
                else:
                    messages.append("POLICY: Outside refund window. Discount or escalation options available.")
            else:
                messages.append("POLICY: Item excluded from standard refund policy. Consider alternative resolution.")

    elif a == ActionType.APPROVE_REFUND:
        if state.refund_decided:
            messages.append("SYSTEM: Refund decision already made.")
        elif not state.classified:
            messages.append("WARNING: Approving refund without classifying the ticket first. Compliance risk.")
            state.policy_compliance = max(0.0, state.policy_compliance - 0.3)
        else:
            state.refund_decided = True
            state.refund_approved = True
            if state.ticket.refund_eligible:
                _update_satisfaction(state, 0.25, rng)
                state.resolution_quality += 0.3
                state.policy_compliance = min(1.0, state.policy_compliance + 0.2)
                messages.append("SYSTEM: Refund approved and processed. Customer notified.")
            else:
                # Approved a refund that shouldn't be approved
                state.policy_compliance = max(0.0, state.policy_compliance - 0.4)
                _update_satisfaction(state, 0.15, rng)  # customer is happy but policy violated
                messages.append("SYSTEM: Refund processed. WARNING: Policy violation flagged for review.")

            if not state.policy_checked:
                state.policy_compliance = max(0.0, state.policy_compliance - 0.2)
                messages.append("WARNING: Refund approved without checking policy first.")

    elif a == ActionType.REJECT_REFUND:
        if state.refund_decided:
            messages.append("SYSTEM: Refund decision already made.")
        else:
            state.refund_decided = True
            state.refund_approved = False
            if not state.ticket.refund_eligible:
                state.policy_compliance = min(1.0, state.policy_compliance + 0.2)
                _update_satisfaction(state, -0.1, rng)
                state.resolution_quality += 0.1
                messages.append("SYSTEM: Refund rejected per policy. Customer notified.")
            else:
                # Rejected an eligible refund
                _update_satisfaction(state, -0.35, rng)
                state.policy_compliance = max(0.0, state.policy_compliance - 0.3)
                messages.append("SYSTEM: Refund rejected. Customer may escalate complaint.")
                if state.customer.is_churning_risk:
                    messages.append("ALERT: Customer is flagged as churn risk. Rejection may cause account loss.")

    elif a == ActionType.ESCALATE_TO_HUMAN:
        if state.escalated:
            messages.append("SYSTEM: Already escalated.")
        else:
            state.escalated = True
            if state.ticket.requires_human:
                state.resolution_quality += 0.3
                _update_satisfaction(state, 0.05, rng)
                messages.append("SYSTEM: Ticket escalated to human supervisor. Appropriate for this case.")
            else:
                state.resolution_quality -= 0.1
                _update_satisfaction(state, -0.05, rng)
                messages.append("SYSTEM: Ticket escalated. Note: this case could have been resolved autonomously.")

    elif a == ActionType.SEND_REPLY:
        if state.reply_sent:
            messages.append("SYSTEM: Reply already sent. Sending another may confuse customer.")
            _update_satisfaction(state, -0.05, rng)
        else:
            state.reply_sent = True
            if state.classified and (state.refund_decided or state.ticket.true_category != TicketCategory.REFUND):
                _update_satisfaction(state, 0.15, rng)
                state.resolution_quality += 0.2
                messages.append("SYSTEM: Reply sent to customer.")
            elif not state.classified:
                _update_satisfaction(state, -0.05, rng)
                messages.append("SYSTEM: Reply sent, but ticket wasn't classified. Response may be off-topic.")
            else:
                _update_satisfaction(state, 0.05, rng)
                messages.append("SYSTEM: Reply sent. Consider resolving the refund request before closing.")

    elif a == ActionType.REQUEST_MORE_INFO:
        if state.info_requested:
            messages.append("SYSTEM: Already requested info. Repeated requests frustrate customers.")
            _update_satisfaction(state, -0.1, rng)
        else:
            state.info_requested = True
            # Sometimes useful, sometimes unnecessary
            if state.ticket.body and len(state.ticket.body) > 100:
                _update_satisfaction(state, -0.05, rng)
                messages.append("SYSTEM: Info request sent. Note: original ticket had sufficient detail.")
            else:
                _update_satisfaction(state, 0.0, rng)
                messages.append("SYSTEM: Additional information requested from customer.")

    elif a == ActionType.APPLY_DISCOUNT:
        if state.discount_offered:
            messages.append("SYSTEM: Discount already offered.")
        else:
            state.discount_offered = True
            if not state.ticket.refund_eligible or state.ticket.days_since_purchase > 90:
                # Good alternative when refund isn't possible
                _update_satisfaction(state, 0.15, rng)
                state.resolution_quality += 0.2
                state.policy_compliance = min(1.0, state.policy_compliance + 0.1)
                messages.append("SYSTEM: 20% discount applied to customer's next order.")
            else:
                # Offering discount when full refund was available — not ideal
                _update_satisfaction(state, 0.05, rng)
                messages.append("SYSTEM: Discount applied. Note: customer was eligible for full refund.")

    elif a == ActionType.CLOSE_TICKET:
        state.ticket_closed = True
        if state.reply_sent and (state.refund_decided or state.ticket.true_category != TicketCategory.REFUND):
            state.resolution_quality += 0.2
            messages.append("SYSTEM: Ticket closed. Resolution complete.")
        elif not state.reply_sent:
            _update_satisfaction(state, -0.2, rng)
            messages.append("SYSTEM: Ticket closed WITHOUT sending a reply. Customer was not informed of resolution.")
        else:
            messages.append("SYSTEM: Ticket closed. Some actions may have been incomplete.")

    # Efficiency decays each step
    state.efficiency = max(0.0, state.efficiency - 0.12)
    state.step_num += 1
    state.step_count += 1

    return messages


# ---------------------------------------------------------------------------
# Full step
# ---------------------------------------------------------------------------

def step_dynamics(
    state: SupportState,
    action: int,
    rng: random.Random,
) -> List[str]:
    """Execute one step. Returns system messages."""
    messages = apply_action(state, action, rng)
    return messages
