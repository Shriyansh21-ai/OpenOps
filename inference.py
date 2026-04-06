""" OpenOps Inference Script

✔ Deterministic
✔ Minimal steps
✔ Budget optimized
✔ Judge-safe output
✔ Strong decision logic
"""

import json
import sys
from env.environment import OpenOpsEnvironment
from env.models import Action


# -----------------------------
# ELITE AGENT
# -----------------------------
class EliteAgent:

    def act(self, obs):

        history = obs.history
        email = obs.email.lower()

        is_refund = any(k in email for k in ["refund", "charged", "return"])

        remaining_budget = getattr(obs, "remaining_budget", 10)

        SAFE_EXIT_BUDGET = 2  # smaller buffer

        # ----------------------------------
        # 1. ALWAYS CLASSIFY (MANDATORY)
        # ----------------------------------
        if "classify_email" not in history:
            return Action(action_type="classify_email", content="refund" if is_refund else "query")

        # ----------------------------------
        # 2. ULTRA SAFE DECISION (DOCKER SAFE)
        # ----------------------------------

        if "approve_refund" not in history and "reject_refund" not in history:

            email_lower = email.lower()

            if obs.known_customer_data:
                refund_flag = obs.known_customer_data.get("refund_eligible") is True

                # VERY STRICT context (prevents false approvals)
                refund_context = (
                    "refund" in email_lower and
                    ("charged" in email_lower or "money" in email_lower)
                )

                if refund_flag and refund_context:
                    return Action(action_type="approve_refund", content="")
                else:
                    return Action(action_type="reject_refund", content="")

            return Action(action_type="reject_refund", content="")
        
        # ----------------------------------
        # 3. LOW BUDGET → SKIP EXTRA ACTIONS
        # ----------------------------------
        if remaining_budget <= SAFE_EXIT_BUDGET:
            return Action(action_type="close_ticket", content="")

        # ----------------------------------
        # 4. OPTIONAL REPLY (ONLY IF SAFE)
        # ----------------------------------
        if "send_reply" not in history:
            return Action(
                action_type="send_reply",
                content="Your request has been processed successfully. Thank you for your patience."
            )

        # ----------------------------------
        # 5. CLOSE
        # ----------------------------------
        return Action(action_type="close_ticket", content="")
# -----------------------------
# MAIN
# -----------------------------
def run():

    env = OpenOpsEnvironment()
    agent = EliteAgent()

    obs = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action = agent.act(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward

    state = env.state()

    # -----------------------------
    # EXACT JUDGE FORMAT
    # -----------------------------
    output = {
        "score": round(total_reward, 4),
        "steps": state["step_count"],
        "budget_left": state["budget"]
    }

    print(json.dumps(output))


# -----------------------------
# FAIL-SAFE
# -----------------------------
if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(json.dumps({
            "score": 0.0,
            "error": str(e)
        }))
        sys.exit(0)