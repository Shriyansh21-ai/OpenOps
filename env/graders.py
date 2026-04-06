"""
Advanced deterministic grading system.

✔ Stable and bounded scoring
✔ Real-world evaluation signals
✔ Deterministic and judge-safe
✔ No hidden edge-case failures
"""


def efficiency(steps, max_steps):
    if max_steps <= 0:
        return 0.0
    return max(0.0, 1.0 - (steps / max_steps))


def budget_score(budget):
    # normalize safely (assuming max budget ~10)
    return max(0.0, min(1.0, budget / 10.0))


class HardGrader:

    def grade(self, history, env):

        # -----------------------------
        # BASIC SIGNALS
        # -----------------------------
        classification = float(any(
            h["action"] == "classify_email" and h["content"] == env.state.true_intent
            for h in history
        ))

        correct_decision = float(any(
            (h["action"] == "approve_refund" and env.state.eligible_for_refund) or
            (h["action"] == "reject_refund" and not env.state.eligible_for_refund)
            for h in history
        ))

        used_db = float(any(h["action"] == "query_customer_db" for h in history))

        good_reply = float(any(
            h["action"] == "send_reply" and any(
                k in h["content"].lower()
                for k in ["sorry", "understand", "apologize"]
            )
            for h in history
        ))

        completed = float(any(h["action"] == "close_ticket" for h in history))

        # -----------------------------
        # ADVANCED SIGNALS
        # -----------------------------

        unnecessary_db = float(
            used_db and env.state.true_intent != "refund"
        )

        over_steps = float(
            env.state.step_count > (env.state.max_steps * 0.8)
        )

        # redundant actions detection
        action_counts = {}
        for h in history:
            action_counts[h["action"]] = action_counts.get(h["action"], 0) + 1

        redundant_actions = float(any(
            v > 1 for k, v in action_counts.items()
            if k not in ["send_reply"]
        ))

        risky_decision = float(
            any(h["action"] == "approve_refund" for h in history)
            and not env.state.eligible_for_refund
        )

        # -----------------------------
        # SCORING
        # -----------------------------
        score = 0.0

        # Core correctness (high weight)
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

        # Company loss penalty (strong but not destructive)
        if env.state.company_loss:
            score -= 0.20

        # Risky incorrect approval
        if risky_decision:
            score -= 0.15

        # Inefficiency
        if over_steps:
            score -= 0.05

        if redundant_actions:
            score -= 0.05

        # -----------------------------
        # FINAL CLAMP (MANDATORY)
        # -----------------------------
        score = max(0.0, min(1.0, score))

        return score