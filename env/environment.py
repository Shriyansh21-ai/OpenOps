import random
from typing import Tuple, Dict

from .models import Observation, Action, InternalState


class OpenOpsEnvironment:
    """
    Simulates a real-world customer support workflow.

    Key properties:
    - Partial observability (DB required)
    - Budget constraints (cost per action)
    - Multi-step decision making
    - Trade-offs (customer satisfaction vs company loss)
    """

    def __init__(self, task_name="easy"):
        self.task_name = task_name
        self.state = None
        self.history = []
        self.customer_db = self._load_db()

    def _load_db(self):
        """
        Simulated customer database.
        Only accessible if agent explicitly queries it.
        """
        return {
            "C001": {"name": "John", "orders": 3, "refund_eligible": True},
            "C002": {"name": "Alice", "orders": 1, "refund_eligible": False},
        }

    def _sample_email(self):
        """
        Samples different real-world scenarios:
        - Refund (eligible / not eligible)
        - General queries
        - Edge cases (used product refund)
        """
        return random.choice([
            {"email": "I want refund", "customer_id": "C001", "intent": "refund", "eligible": True},
            {"email": "Refund me now!", "customer_id": "C002", "intent": "refund", "eligible": False},
            {"email": "Where is my order?", "customer_id": "C001", "intent": "query", "eligible": False},
            {"email": "Charged twice, used product, refund?", "customer_id": "C002", "intent": "refund", "eligible": False}
        ])

    def reset(self):
        """
        Initializes a new episode with a fresh customer query.
        """
        sample = self._sample_email()
        self.current_email = sample

        # Internal ground-truth state
        self.state = InternalState(
            true_intent=sample["intent"],
            eligible_for_refund=sample["eligible"],
            correct_resolution="approve" if sample["eligible"] else "reject"
        )

        self.history = []
        return self._get_obs()

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        """
        Executes one action in the environment.

        Returns:
        - Observation
        - Reward (continuous)
        - Done flag
        - Info dict
        """

        reward = 0.0
        error = None

        # Action cost mapping (introduces budget constraint)
        cost_map = {
            "query_customer_db": 2,
            "classify_email": 1,
            "send_reply": 2,
            "approve_refund": 5,
            "reject_refund": 1,
            "escalate_ticket": 3,
            "close_ticket": 0,
        }

        # Invalid action handling (deterministic penalty)
        if action.action_type not in cost_map:
            return self._get_obs(), -0.2, False, {"error": "invalid_action"}

        # Deduct budget
        self.state.budget -= cost_map[action.action_type]

        # Log action history for grading
        self.history.append({
            "action": action.action_type,
            "content": action.content
        })

        # ---------------- ACTION LOGIC ----------------

        if action.action_type == "query_customer_db":
            # Enables access to hidden customer data
            self.state.db_accessed = True

        elif action.action_type == "classify_email":
            # Reward correct intent classification
            if action.content == self.state.true_intent:
                reward += 0.2
            else:
                reward -= 0.1

        elif action.action_type == "approve_refund":
            # Trade-off: correct vs company loss
            if self.state.eligible_for_refund:
                reward += 0.3
            else:
                reward -= 0.4
                self.state.company_loss = True

        elif action.action_type == "reject_refund":
            if not self.state.eligible_for_refund:
                reward += 0.3
            else:
                reward -= 0.4

        elif action.action_type == "send_reply":
            # Reward empathetic responses (simple NLP heuristic)
            if any(k in action.content.lower() for k in ["sorry", "understand"]):
                reward += 0.2

        elif action.action_type == "close_ticket":
            # Encourages task completion
            self.state.done = True
            reward += 0.1

        # ---------------- DYNAMIC STATE ----------------

        self.state.step_count += 1

        # Customer mood evolves based on interaction length & quality
        if self.state.step_count > 2:
            mood = "frustrated"
        else:
            mood = "neutral"

        if reward < 0:
            mood = "angry"

        # ---------------- TERMINATION CONDITIONS ----------------

        # Step limit
        if self.state.step_count >= self.state.max_steps:
            self.state.done = True
            reward -= 0.3

        # Budget exhausted
        if self.state.budget <= 0:
            self.state.done = True
            reward -= 0.3

        # Efficiency bonus (encourages optimal policies)
        reward += 0.1 * (self.state.budget / 10)

        return self._get_obs(mood), reward, self.state.done, {"error": error}

    def _get_obs(self, mood="neutral") -> Observation:
        """
        Constructs observation visible to the agent.
        Implements partial observability.
        """

        data = None
        if self.state.db_accessed:
            data = self.customer_db[self.current_email["customer_id"]]

        return Observation(
            email=self.current_email["email"],
            customer_id=self.current_email["customer_id"],
            known_customer_data=data,
            ticket_status="closed" if self.state.done else "open",
            customer_mood=mood,
            remaining_budget=self.state.budget,
            remaining_steps=self.state.max_steps - self.state.step_count,
            history=[h["action"] for h in self.history]
        )

    def state_dict(self):
        """
        Exposes internal state for grading only (not agent).
        """
        return self.state.dict()