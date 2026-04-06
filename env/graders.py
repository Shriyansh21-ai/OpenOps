"""
Advanced deterministic grading system.

Produces score in [0.0, 1.0] using:
- Correctness
- Decision quality
- Efficiency
- Resource optimization
- Risk awareness

Designed for:
✔ Real-world evaluation
✔ Strategy differentiation
✔ High-quality agent selection
"""


def efficiency(steps, max_steps):
    return max(0, 1 - steps / max_steps)


def budget_score(budget):
    return max(0, budget / 10)


class HardGrader:

    def grade(self, history, env):

        # -----------------------------
        # BASIC SIGNALS
        # -----------------------------
        classification = any(
            h["action"] == "classify_email" and h["content"] == env.state.true_intent
            for h in history
        )

        correct_decision = any(
            (h["action"] == "approve_refund" and env.state.eligible_for_refund) or
            (h["action"] == "reject_refund" and not env.state.eligible_for_refund)
            for h in history
        )

        used_db = any(h["action"] == "query_customer_db" for h in history)

        good_reply = any(
            h["action"] == "send_reply" and any(
                k in h["content"].lower()
                for k in ["sorry", "understand", "apologize"]
            )
            for h in history
        )

        completed = any(h["action"] == "close_ticket" for h in history)

        # -----------------------------
        # ADVANCED SIGNALS
        # -----------------------------

        # Penalize unnecessary DB usage
        unnecessary_db = (
            used_db and env.state.true_intent != "refund"
        )

        # Detect over-action 
        over_steps = env.state.step_count > (env.state.max_steps * 0.8)

        # Detect redundant actions
        action_counts = {}
        for h in history:
            action_counts[h["action"]] = action_counts.get(h["action"], 0) + 1

        redundant_actions = any(v > 1 for k, v in action_counts.items()
                                if k not in ["send_reply"])

        # Risk-aware decision
        risky_decision = (
            any(h["action"] == "approve_refund" for h in history)
            and not env.state.eligible_for_refund
        )

        # -----------------------------
        # SCORING
        # -----------------------------

        score = 0.0

        # Core correctness (highest weight)
        score += 0.25 * classification
        score += 0.30 * correct_decision

        # Behavioral quality
        score += 0.10 * good_reply
        score += 0.10 * completed

        # Resource intelligence
        score += 0.10 * efficiency(env.state.step_count, env.state.max_steps)
        score += 0.10 * budget_score(env.state.budget)

        # Smart DB usage
        if used_db:
            score += 0.05
        if unnecessary_db:
            score -= 0.05

        # -----------------------------
        # PENALTIES 
        # -----------------------------

        # Skipping DB in refund case
        if not used_db and env.state.true_intent == "refund":
            score -= 0.15

        # Company loss 
        if env.state.company_loss:
            score -= 0.25

        # Risky wrong approval
        if risky_decision:
            score -= 0.15

        # Inefficiency penalties
        if over_steps:
            score -= 0.05

        if redundant_actions:
            score -= 0.05

        # -----------------------------
        # FINAL CLAMP
        # -----------------------------
        return max(0.0, min(score, 1.0))