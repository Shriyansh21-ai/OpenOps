"""
Deterministic grading system.

Produces score in [0.0, 1.0] using:
- Task correctness
- Decision quality
- Efficiency
- Resource usage
- Real-world penalties

Designed to:
✔ Be reproducible
✔ Reward good reasoning
✔ Penalize bad trade-offs
"""


def efficiency(steps, max_steps):
    # Fewer steps = higher efficiency
    return max(0, 1 - steps / max_steps)


def budget_score(budget):
    # More remaining budget = better policy
    return max(0, budget / 10)


class HardGrader:

    def grade(self, history, env):

        # Check correct intent classification
        classification = any(
            h["action"] == "classify_email" and h["content"] == env.state.true_intent
            for h in history
        )

        # Check correct refund decision
        correct_decision = any(
            (h["action"] == "approve_refund" and env.state.eligible_for_refund) or
            (h["action"] == "reject_refund" and not env.state.eligible_for_refund)
            for h in history
        )

        # Whether agent used DB (important for realism)
        used_db = any(h["action"] == "query_customer_db" for h in history)

        # Check for empathetic communication
        good_reply = any(
            h["action"] == "send_reply" and "sorry" in h["content"].lower()
            for h in history
        )

        # Task completion
        completed = any(h["action"] == "close_ticket" for h in history)

        # Final weighted score
        score = (
            0.2 * classification +
            0.25 * correct_decision +
            0.15 * used_db +
            0.15 * good_reply +
            0.1 * completed +
            0.1 * efficiency(env.state.step_count, env.state.max_steps) +
            0.05 * budget_score(env.state.budget)
        )

        # Penalize skipping DB in refund scenarios
        if not used_db and env.state.true_intent == "refund":
            score -= 0.2

        # Penalize company loss (critical real-world failure)
        if env.state.company_loss:
            score -= 0.2

        # Clamp score to valid range
        return max(0.0, min(score, 1.0))