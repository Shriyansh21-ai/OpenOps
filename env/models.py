from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal


# -----------------------------
# Observation sent to the agent
# -----------------------------
class Observation(BaseModel):
    # Raw customer email text
    email: str

    # Unique identifier for customer
    customer_id: str

    # Partial observability (None until DB is queried)
    known_customer_data: Optional[Dict[str, object]] = None

    # Ticket status
    ticket_status: Literal["open", "closed"]

    # Customer sentiment
    customer_mood: Literal["neutral", "frustrated", "angry"]

    # Remaining budget
    remaining_budget: int

    # Remaining steps
    remaining_steps: int

    # Action history
    history: List[str] = Field(default_factory=list)


# -----------------------------
# Action chosen by the agent
# -----------------------------
class Action(BaseModel):
    # Allowed action space (STRICT → avoids invalid actions)
    action_type: Literal[
        "query_customer_db",
        "classify_email",
        "send_reply",
        "approve_refund",
        "reject_refund",
        "escalate_ticket",
        "close_ticket",
    ]

    # Optional payload
    content: Optional[str] = ""


# -----------------------------
# Internal environment state
# (hidden from agent)
# -----------------------------
class InternalState(BaseModel):

    # Ground truth
    true_intent: Literal["refund", "query"]

    eligible_for_refund: bool

    correct_resolution: Literal["approve", "reject"]

    # Step tracking
    step_count: int = 0

    # Budget constraint
    budget: int = 10

    # Max steps
    max_steps: int = 6

    # Termination flag
    done: bool = False

    # DB access tracking
    db_accessed: bool = False

    # Company loss flag
    company_loss: bool = False