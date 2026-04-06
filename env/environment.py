import random
from typing import Tuple, Dict, Any

from .models import Observation, Action, InternalState


class OpenOpsEnvironment:
    """
     Advanced real-world customer support simulation

    Features:
    - Partial observability
    - Noisy inputs (real-world ambiguity)
    - Budget & step constraints
    - Delayed consequences
    - Customer satisfaction modeling
    - Action dependency penalties
    """

    def __init__(self, task_name="easy"):
        self.task_name = task_name
        self._internal_state = None
        self.history = []
        self.customer_db = self._load_db()
        self.current_email = None

    # -----------------------------
    # DATABASE
    # -----------------------------
    def _load_db(self):
        return {
            "C001": {"name": "John", "orders": 3, "refund_eligible": True},
            "C002": {"name": "Alice", "orders": 1, "refund_eligible": False},
        }

    # -----------------------------
    # EMAIL SAMPLING 
    # -----------------------------
    def _sample_email(self):
        return random.choice([
            {
                "email": "I want refund",
                "customer_id": "C001",
                "intent": "refund",
                "eligible": True
            },
            {
                "email": "Refund me now!!! this is unacceptable",
                "customer_id": "C002",
                "intent": "refund",
                "eligible": False
            },
            {
                "email": "Where is my order??",
                "customer_id": "C001",
                "intent": "query",
                "eligible": False
            },
            {
                "email": "Charged twice, used product, refund?",
                "customer_id": "C002",
                "intent": "refund",
                "eligible": False
            },
            {
                # ambiguous / tricky case
                "email": "I had an issue with my order",
                "customer_id": "C001",
                "intent": "query",
                "eligible": False
            }
        ])

    # -----------------------------
    # RESET
    # -----------------------------
    def reset(self) -> Observation:
        sample = self._sample_email()
        self.current_email = sample

        self._internal_state = InternalState(
            true_intent=sample["intent"],
            eligible_for_refund=sample["eligible"],
            correct_resolution="approve" if sample["eligible"] else "reject"
        )

        # Extra hidden realism
        self._internal_state.customer_satisfaction = 0.5
        self._internal_state.delayed_penalty = 0.0

        self.history = []
        return self._get_obs()

    # -----------------------------
    # STEP
    # -----------------------------
    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:

        reward = 0.0
        error = None

        cost_map = {
            "query_customer_db": 2,
            "classify_email": 1,
            "send_reply": 2,
            "approve_refund": 5,
            "reject_refund": 1,
            "escalate_ticket": 3,
            "close_ticket": 0,
        }

        # Invalid action
        if action.action_type not in cost_map:
            return self._get_obs(), -0.3, False, {"error": "invalid_action"}

        # Deduct budget
        self._internal_state.budget -= cost_map[action.action_type]

        # Log
        self.history.append({
            "action": action.action_type,
            "content": action.content
        })

        # -----------------------------
        # ACTION ORDER VALIDATION
        # -----------------------------
        if action.action_type in ["approve_refund", "reject_refund"]:
            if not any(h["action"] == "classify_email" for h in self.history):
                reward -= 0.2  # acting without understanding

        # -----------------------------
        # ACTION LOGIC
        # -----------------------------
        if action.action_type == "query_customer_db":
            self._internal_state.db_accessed = True

            # Simulate noisy DB (real-world inconsistency)
            if random.random() < 0.1:
                reward -= 0.1  # misleading info

        elif action.action_type == "classify_email":
            if action.content == self._internal_state.true_intent:
                reward += 0.25
            else:
                reward -= 0.15

        elif action.action_type == "approve_refund":
            if self._internal_state.eligible_for_refund:
                reward += 0.4
                self._internal_state.customer_satisfaction += 0.3
            else:
                reward -= 0.5
                self._internal_state.company_loss = True
                self._internal_state.delayed_penalty += 0.2

        elif action.action_type == "reject_refund":
            if not self._internal_state.eligible_for_refund:
                reward += 0.35
            else:
                reward -= 0.4
                self._internal_state.customer_satisfaction -= 0.3

        elif action.action_type == "send_reply":
            text = action.content.lower()

            if any(k in text for k in ["sorry", "apologize", "understand"]):
                reward += 0.25
                self._internal_state.customer_satisfaction += 0.2
            else:
                reward -= 0.05

        elif action.action_type == "escalate_ticket":
            reward += 0.1  # safe fallback strategy

        elif action.action_type == "close_ticket":
            self._internal_state.done = True

            # Delayed reward based on satisfaction
            reward += self._internal_state.customer_satisfaction

        # -----------------------------
        # DYNAMICS
        # -----------------------------
        self._internal_state.step_count += 1

        mood = "neutral"

        if self._internal_state.customer_satisfaction < 0.3:
            mood = "angry"
        elif self._internal_state.customer_satisfaction > 0.7:
            mood = "happy"
        elif self._internal_state.step_count > 2:
            mood = "frustrated"

        # -----------------------------
        # TERMINATION
        # -----------------------------
        if self._internal_state.step_count >= self._internal_state.max_steps:
            self._internal_state.done = True
            reward -= 0.3

        if self._internal_state.budget <= 0:
            self._internal_state.done = True
            reward -= 0.3

        # Apply delayed penalty
        reward -= self._internal_state.delayed_penalty

        # Efficiency bonus
        reward += 0.1 * (self._internal_state.budget / 10)

        return self._get_obs(mood), reward, self._internal_state.done, {"error": error}

    # -----------------------------
    # OBSERVATION
    # -----------------------------
    def _get_obs(self, mood="neutral") -> Observation:

        data = None
        if self._internal_state.db_accessed:
            data = self.customer_db[self.current_email["customer_id"]]

        return Observation(
            email=self.current_email["email"],
            customer_id=self.current_email["customer_id"],
            known_customer_data=data,
            ticket_status="closed" if self._internal_state.done else "open",
            customer_mood=mood,
            remaining_budget=self._internal_state.budget,
            remaining_steps=self._internal_state.max_steps - self._internal_state.step_count,
            history=[h["action"] for h in self.history]
        )

    # -----------------------------
    # REQUIRED STATE FUNCTION
    # -----------------------------
    def state(self) -> Dict[str, Any]:
        """
         REQUIRED by OpenEnv spec
        """
        return {
            "internal_state": self._internal_state.dict(),
            "current_email": self.current_email,
            "history": self.history,
            "db_accessed": self._internal_state.db_accessed,
            "budget": self._internal_state.budget,
            "step_count": self._internal_state.step_count,
            "done": self._internal_state.done,
            "customer_satisfaction": getattr(self._internal_state, "customer_satisfaction", None),
        }

    # Backward compatibility
    def state_dict(self):
        return self.state()