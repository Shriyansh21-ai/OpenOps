from env.environment import OpenOpsEnvironment
from env.models import Action
from env.graders import HardGrader


"""
Deterministic Rule-Based Baseline Agent

Purpose:
- Provides a reproducible baseline for evaluation
- Demonstrates how agents should interact with the environment
- Optimized for high score while respecting constraints

Design Strategy:
1. Use DB early (critical for refund correctness)
2. Classify intent
3. Make correct decision (approve/reject)
4. Send empathetic response
5. Close ticket efficiently

Key Features:
- Budget-aware (avoids unnecessary actions)
- Step-efficient (minimizes steps)
- Deterministic (same input → same output)
"""


class RuleBasedAgent:
    def __init__(self):
        pass

    def act(self, obs):
        """
        Core decision logic of the agent.
        Takes observation → returns next action.
        """

        history = obs.history

        # ---------------- STEP 1: QUERY DB ----------------
        # Always query DB first if not already done
        # Critical for refund decision correctness
        if obs.known_customer_data is None:
            return Action(action_type="query_customer_db")

        # ---------------- STEP 2: CLASSIFY EMAIL ----------------
        if "classify_email" not in history:
            if "refund" in obs.email.lower():
                return Action(action_type="classify_email", content="refund")
            else:
                return Action(action_type="classify_email", content="query")

        # ---------------- STEP 3: DECISION MAKING ----------------
        # Use DB info to decide refund correctly
        if not any(a in history for a in ["approve_refund", "reject_refund"]):

            if obs.known_customer_data.get("refund_eligible", False):
                return Action(action_type="approve_refund")
            else:
                return Action(action_type="reject_refund")

        # ---------------- STEP 4: SEND EMPATHETIC RESPONSE ----------------
        if "send_reply" not in history:
            return Action(
                action_type="send_reply",
                content="Sorry for the inconvenience. I understand your concern and we are resolving it."
            )

        # ---------------- STEP 5: CLOSE TICKET ----------------
        if "close_ticket" not in history:
            return Action(action_type="close_ticket")

        # Fallback safety (should not reach here)
        return Action(action_type="close_ticket")


# ---------------- RUNNER ----------------

def run_episode(task_name="constrained_workflow_hard", verbose=False):
    """
    Runs one full episode for a given task.

    Returns:
    - final score (0.0 to 1.0)
    """

    env = OpenOpsEnvironment(task_name=task_name)
    agent = RuleBasedAgent()

    obs = env.reset()
    done = False

    step = 0

    while not done:
        action = agent.act(obs)

        obs, reward, done, info = env.step(action)

        if verbose:
            print(f"\nStep {step}")
            print(f"Action: {action}")
            print(f"Reward: {reward}")
            print(f"Remaining Budget: {obs.remaining_budget}")
            print(f"Mood: {obs.customer_mood}")

        step += 1

    # Final grading
    grader = HardGrader()
    score = grader.grade(env.history, env)

    if verbose:
        print("\nFinal History:", env.history)
        print("Final Score:", score)

    return score


def run_all_tasks():
    """
    Runs baseline across all tasks required by OpenEnv.

    Ensures:
    - Coverage of difficulty levels
    - Reproducible benchmarking
    """

    tasks = [
        "classification_easy",
        "refund_decision_medium",
        "constrained_workflow_hard"
    ]

    results = {}

    print("\nRunning Rule-Based Baseline on All Tasks\n")

    for task in tasks:
        score = run_episode(task_name=task, verbose=False)
        results[task] = score
        print(f"{task}: {score:.3f}")

    avg_score = sum(results.values()) / len(results)

    print("\nAverage Score:", round(avg_score, 3))

    return results


# ---------------- ENTRY POINT ----------------

if __name__ == "__main__":
    run_all_tasks()