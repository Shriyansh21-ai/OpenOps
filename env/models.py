from pydantic import BaseModel, Field
from typing import List, Optional, Dict


# -----------------------------
# Observation sent to the agent
# -----------------------------
class Observation(BaseModel):
    # Raw customer email text
    email: str

    # Unique identifier for customer
    customer_id: str

    # Partial observability:
    # This is ONLY available if agent queries DB
    known_customer_data: Optional[Dict] = None

    # Current ticket status (open/closed)
    ticket_status: str

    # Dynamic state reflecting customer sentiment
    customer_mood: str

    # Remaining budget (cost-constrained actions)
    remaining_budget: int

    # Remaining steps before episode ends
    remaining_steps: int

    # Action history (helps agent reason sequentially)
    history: List[str] = Field(default_factory=list)


# -----------------------------
# Action chosen by the agent
# -----------------------------
class Action(BaseModel):
    # Type of action (must match allowed action space)
    action_type: str

    # Optional content (e.g., classification label or reply text)
    content: Optional[str] = ""


# -----------------------------
# Internal environment state
# (hidden from agent)
# -----------------------------
class InternalState(BaseModel):

    # Ground-truth intent (used for evaluation)
    true_intent: str

    # Whether refund is actually valid
    eligible_for_refund: bool

    # Expected correct resolution (approve/reject)
    correct_resolution: str

    # Step tracking
    step_count: int = 0

    # Budget constraint (penalizes inefficient policies)
    budget: int = 10

    # Maximum allowed steps per episode
    max_steps: int = 6

    # Episode termination flag
    done: bool = False

    # Whether agent accessed database (important for reasoning)
    db_accessed: bool = False

    # Tracks incorrect refund approvals (company loss)
    company_loss: bool = False