from env.environment import OpenOpsEnvironment
from env.models import Action
from env.graders import HardGrader
import os
from openai import OpenAI

"""
Optional baseline using OpenAI API.

NOTE:
- Not required for evaluation
- Requires OPENAI_API_KEY
- Provided to demonstrate LLM-agent compatibility
"""

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def llm_agent(obs):
    prompt = f"""
You are an AI customer support agent.

Email: {obs.email}
Customer Data: {obs.known_customer_data}
Budget: {obs.remaining_budget}
Steps Left: {obs.remaining_steps}

Choose ONE action from:
query_customer_db, classify_email, send_reply,
approve_refund, reject_refund, close_ticket

Respond in JSON:
{{"action_type": "...", "content": "..."}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    try:
        import json
        parsed = json.loads(response.choices[0].message.content)
        return Action(**parsed)
    except:
        return Action(action_type="close_ticket")


def run_episode():
    env = OpenOpsEnvironment()
    obs = env.reset()

    done = False

    while not done:
        action = llm_agent(obs)
        obs, reward, done, _ = env.step(action)

    grader = HardGrader()
    score = grader.grade(env.history, env)

    return score


if __name__ == "__main__":
    print("Running OpenAI baseline...")
    score = run_episode()
    print(f"Score: {score}")