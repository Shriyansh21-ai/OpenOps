"""
Defines task difficulty levels.

Each task modifies:
- Step limits
- Budget constraints
- Scenario complexity

This ensures:
✔ Easy → basic reasoning
✔ Medium → decision + correctness
✔ Hard → planning + optimization + robustness
"""


class BaseTask:
    name = "base"

    def configure(self, env):
        """Override in subclasses"""
        pass


# ----------------------------------
# EASY: Classification-focused
# ----------------------------------
class ClassificationTask(BaseTask):
    name = "classification_easy"

    def configure(self, env):
        # Minimal steps, minimal cost
        env.state.max_steps = 3
        env.state.budget = 5

        # Simplify scenarios (only clear cases)
        env.current_email = {
            "email": "I want a refund",
            "customer_id": "C001",
            "intent": "refund",
            "eligible": True
        }

        # Encourage classification reward
        env.state.true_intent = "refund"
        env.state.eligible_for_refund = True
        env.state.correct_resolution = "approve"


# ----------------------------------
# MEDIUM: Decision-making
# ----------------------------------
class RefundDecisionTask(BaseTask):
    name = "refund_decision_medium"

    def configure(self, env):
        env.state.max_steps = 5
        env.state.budget = 8

        # Slight penalty pressure
        env.state.company_loss = False


# ----------------------------------
# HARD: Full workflow optimization
# ----------------------------------
class ConstrainedWorkflowTask(BaseTask):
    name = "constrained_workflow_hard"

    def configure(self, env):
        env.state.max_steps = 6
        env.state.budget = 10

        # Make environment more uncertain
        env.state.db_accessed = False

        
        if "refund" in env.current_email["email"].lower():
            # Inject ambiguity
            env.state.eligible_for_refund = False


# ----------------------------------
# TASK REGISTRY 
# ----------------------------------
TASK_REGISTRY = {
    "easy": ClassificationTask(),
    "medium": RefundDecisionTask(),
    "hard": ConstrainedWorkflowTask(),
}