import random
from typing import Tuple, Dict, Any

from .models import Observation, Action, InternalState


class OpenEnvEnvironment:

    def __init__(self, task_name="easy", seed=42):
        self.task_name = task_name
        self._internal_state = None
        self.history = []
        self.customer_db = self._load_db()
        self.current_email = None

        random.seed(seed)

    def _load_db(self):
        return {
            "C001": {"name": "John", "orders": 3, "refund_eligible": True},
            "C002": {"name": "Alice", "orders": 1, "refund_eligible": False},
        }

    def _sample_email(self):
        samples = [
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
        ]

        index = len(self.history) % len(samples)
        return samples[index]

    def reset(self) -> Observation:
        sample = self._sample_email()
        self.current_email = sample

        self._internal_state = InternalState(
            true_intent=sample["intent"],
            eligible_for_refund=sample["eligible"],
            correct_resolution="approve" if sample["eligible"] else "reject"
        )

        self._internal_state.customer_satisfaction = 0.5
        self._internal_state.delayed_penalty = 0.0
        self._internal_state.company_loss = False
        self._internal_state.db_accessed = False
        self._internal_state.done = False
        self._internal_state.step_count = 0

        if self.task_name == "easy":
            self._internal_state.max_steps = 4
            self._internal_state.budget = 12
        elif self.task_name == "medium":
            self._internal_state.max_steps = 5
            self._internal_state.budget = 9
        else:
            self._internal_state.max_steps = 6
            self._internal_state.budget = 6

        self.history = []
        return self._get_obs()

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

        if action.action_type not in cost_map:
            return self._get_obs(), -0.3, False, {"error": "invalid_action"}

        self._internal_state.budget -= cost_map[action.action_type]

        self.history.append({
            "action": action.action_type,
            "content": action.content
        })

        if action.action_type in ["approve_refund", "reject_refund"]:
            if not any(h["action"] == "classify_email" for h in self.history):
                reward -= 0.2

        # -----------------------------
        # REWARD LOGIC
        # -----------------------------
        if action.action_type == "query_customer_db":
            self._internal_state.db_accessed = True
            reward += 0.10 if self.task_name == "easy" else (0.05 if self.task_name == "medium" else 0.02)

        elif action.action_type == "classify_email":
            if action.content == self._internal_state.true_intent:
                reward += 0.30 if self.task_name == "easy" else (0.25 if self.task_name == "medium" else 0.20)
            else:
                reward -= 0.20

        elif action.action_type == "approve_refund":
            if self._internal_state.eligible_for_refund:
                reward += 0.4
                self._internal_state.customer_satisfaction += 0.3

                # 🔥 HARD FIX: don't end here
                if self.task_name == "hard":
                    self._internal_state.done = False

            else:
                reward -= 0.6

        elif action.action_type == "reject_refund":
            if not self._internal_state.eligible_for_refund:
                reward += 0.35
            else:
                reward -= 0.5

        elif action.action_type == "send_reply":
            text = action.content.lower()

            if any(k in text for k in ["sorry", "apologize", "understand"]):
                reward += 0.25 if self.task_name != "hard" else 0.20
            else:
                reward -= 0.10

            # 🔥 HARD FIX: continue flow
            if self.task_name == "hard":
                self._internal_state.done = False

        elif action.action_type == "escalate_ticket":
            reward += 0.1 if self.task_name != "hard" else 0.30

        elif action.action_type == "close_ticket":

            if not any(h["action"] == "send_reply" for h in self.history):
                reward -= 0.2

            self._internal_state.done = True

            if self.task_name == "easy":
                reward += 0.30
            elif self.task_name == "medium":
                reward += 0.25
            else:
                reward += 0.20

        self._internal_state.step_count += 1

        if self._internal_state.step_count >= self._internal_state.max_steps:
            self._internal_state.done = True
            reward -= 0.3

        if self._internal_state.budget <= 0:
            self._internal_state.done = True
            reward -= 0.3

        reward -= self._internal_state.delayed_penalty

        if self.task_name == "hard":
            reward -= 0.05

        bonus = 0.1 * (self._internal_state.budget / 10)
        if self.task_name == "hard":
            bonus += 0.05

        reward += bonus

        # 🔥 FINAL HARD SAFETY CHECK
        if self.task_name == "hard":
            if not any(h["action"] == "send_reply" for h in self.history):
                self._internal_state.done = False

        return self._get_obs(), reward, self._internal_state.done, {"error": error}

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

    def state(self) -> Dict[str, Any]:
        return {
            "internal_state": self._internal_state.dict(),
            "current_email": self.current_email,
            "history": self.history,
            "db_accessed": self._internal_state.db_accessed,
            "budget": self._internal_state.budget,
            "step_count": self._internal_state.step_count,
            "done": self._internal_state.done,
            "customer_satisfaction": self._internal_state.customer_satisfaction,
        }

    def state_dict(self):
        return self.state()