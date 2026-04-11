---
title: OpenOps
emoji: 🎧
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# OpenOps: Autonomous Customer Support Agent

An OpenEnv-compatible RL environment where an AI agent autonomously handles customer support tickets — classifying issues, looking up customer data, checking policies, making refund decisions, and communicating with customers.

## Why Customer Support?

- **Partial observability** — customer sentiment is hidden, refund eligibility unknown until checked
- **Policy compliance** — approving an ineligible refund is a policy violation
- **Customer context matters** — premium vs standard, churn risk, tenure
- **Every action has consequences** — rejecting an eligible refund tanks satisfaction
- **Efficiency matters** — fewer steps to resolution = better score

## Action Space (10 actions)

| ID | Action | Effect |
|---|---|---|
| 0 | classify_email | Identify ticket category |
| 1 | query_customer_db | Look up customer history/tier |
| 2 | check_policy | Verify refund eligibility |
| 3 | approve_refund | Approve (must check policy first!) |
| 4 | reject_refund | Reject refund request |
| 5 | escalate_to_human | Escalate to supervisor |
| 6 | send_reply | Respond to customer |
| 7 | request_more_info | Ask for details |
| 8 | apply_discount | Offer discount/credit alternative |
| 9 | close_ticket | Mark resolved |

## Tasks

| Task | Difficulty | Scenario |
|---|---|---|
| easy_1 | Easy | Simple refund, clear policy, standard customer |
| medium_1 | Medium | Ambiguous complaint, premium customer, edge case policy |
| hard_1 | Hard | Multi-issue enterprise escalation, churn risk, $2,450 order |

## Scoring

```
30% customer satisfaction + 25% resolution quality + 20% policy compliance + 15% efficiency + 10% proper closure
```

## Quick Start

```bash
pip install -r requirements.txt
uvicorn server.app:app --port 7860
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `API_BASE_URL` | Yes | OpenAI-compatible API endpoint |
| `MODEL_NAME` | Yes | Model identifier |
| `HF_TOKEN` | Yes | HuggingFace token / API key |
| `LOCAL_IMAGE_NAME` | Optional | Docker image for `from_docker_image()` |
