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

        SAFE_EXIT_BUDGET = 3   # leave margin for unseen costs

        if remaining_budget <= SAFE_EXIT_BUDGET:
            # Skip everything → just close safely
            return Action(action_type="close_ticket", content="")

        # ----------------------------------
        # 1. CLASSIFY
        # ----------------------------------
        if "classify_email" not in history:
            return Action(action_type="classify_email", content="refund" if is_refund else "query")

        # ----------------------------------
        # 2. db
        # ----------------------------------

        if is_refund and "query_customer_db" not in history and remaining_budget > 6:
            return Action(action_type="query_customer_db", content="")

        # ----------------------------------
        # 3. DECISION (LOW COST)
        # ----------------------------------
        if "approve_refund" not in history and "reject_refund" not in history:
            return Action(action_type="approve_refund" if is_refund else "reject_refund", content="")

        # ----------------------------------
        # 4. REPLY (ONLY IF VERY SAFE)
        # ----------------------------------
        if "send_reply" not in history and remaining_budget > 4:
            return Action(
                action_type="send_reply",
                content="Your request has been processed successfully. Thank you for your patience."
            )

        # ----------------------------------
        # 5. CLOSE EARLY
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