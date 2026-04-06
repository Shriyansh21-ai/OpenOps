"""
Defines task difficulty levels.

Each task modifies:
- Step limits
- Budget constraints
- Scenario complexity

✔ Easy → basic reasoning
✔ Medium → decision + correctness
✔ Hard → planning + optimization + robustness
"""


class get_all_tasks:
    name = "base"

    def configure(self, env):
        """Override in subclasses"""
        pass


# ----------------------------------
# EASY: Classification-focused
# ----------------------------------
class ClassificationTask(get_all_tasks):
    name = "classification_easy"

    def configure(self, env):
        state = env._internal_state

        # Tight limits
        state.max_steps = 3
        state.budget = 5

        # Keep scenario simple (no override of email)
        state.true_intent = "refund"
        state.eligible_for_refund = True
        state.correct_resolution = "approve"


# ----------------------------------
# MEDIUM: Decision-making
# ----------------------------------
class RefundDecisionTask(get_all_tasks):
    name = "refund_decision_medium"

    def configure(self, env):
        state = env._internal_state

        state.max_steps = 5
        state.budget = 8

        # Ensure clean baseline
        state.company_loss = False


# ----------------------------------
# HARD: Full workflow optimization
# ----------------------------------
class ConstrainedWorkflowTask(get_all_tasks):
    name = "constrained_workflow_hard"

    def configure(self, env):
        state = env._internal_state

        state.max_steps = 6
        state.budget = 10

        # Force smart DB usage
        state.db_accessed = False

        # Introduce mild ambiguity safely
        if state.true_intent == "refund":
            # Slight uncertainty but not breaking logic
            state.customer_satisfaction = 0.4


# ----------------------------------
# TASK REGISTRY
# ----------------------------------
TASK_REGISTRY = {
    "easy": ClassificationTask(),
    "medium": RefundDecisionTask(),
    "hard": ConstrainedWorkflowTask(),
}