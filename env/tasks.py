"""
Defines task difficulty levels.

Each task modifies:
- Step limits
- Budget constraints

This ensures:
✔ Easy → basic reasoning
✔ Medium → decision + correctness
✔ Hard → planning + optimization
"""


class ClassificationTask:
    name = "classification_easy"

    def configure(self, env):
        # Very limited interaction required
        env.state.max_steps = 3
        env.state.budget = 5


class RefundDecisionTask:
    name = "refund_decision_medium"

    def configure(self, env):
        # Requires correct reasoning + action
        env.state.max_steps = 5
        env.state.budget = 8


class ConstrainedWorkflowTask:
    name = "constrained_workflow_hard"

    def configure(self, env):
        # Full complexity:
        # planning + cost + correctness + empathy
        env.state.max_steps = 6
        env.state.budget = 10