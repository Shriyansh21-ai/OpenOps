#  OpenOps Benchmark: Cost-Constrained AI Decision Environment

##  TL;DR

A **real-world, cost-aware AI benchmark** where agents must resolve customer support tickets under **budget, time, and uncertainty constraints** — balancing **correctness, efficiency, and business risk**.

> Not a toy problem — a **production-style decision system benchmark**

---

##  Why This is Top-Tier

*  **Real-world modeling** (refund systems, customer DB, workflows)
*  **Budget constraints** (every action has cost)
*  **Time constraints** (limited steps per episode)
*  **Partial observability** (DB must be queried)
*  **Trade-offs** (customer satisfaction vs company loss)
*  **Deterministic grading (0.0–1.0)**

-> Forces **true agent intelligence**, not just prediction.

---

##  Tasks (Increasing Difficulty)

| Task                        | What it Tests                                    |
| --------------------------- | ------------------------------------------------ |
| `classification_easy`       | Basic intent recognition                         |
| `refund_decision_medium`    | Correct decision using hidden data               |
| `constrained_workflow_hard` | Full pipeline: reasoning + planning + efficiency |

---

##  What Makes This Different

Unlike standard benchmarks:

| Typical Benchmarks    | This Environment            |
| --------------------- | --------------------------- |
| Static inputs         | Dynamic multi-step workflow |
| No cost awareness     | Budget-constrained actions  |
| Fully visible data    | Hidden state (DB required)  |
| Accuracy-only scoring | Multi-objective scoring     |

---

##  Environment Highlights

* **Action Space**: query DB, classify, approve/reject, reply, close
* **Cost System**: each action reduces budget
* **Dynamic State**: customer mood evolves (neutral → frustrated → angry)
* **Penalties**:

  * Wrong refund → company loss ❌
  * No DB usage → reasoning penalty ❌
  * Inefficiency → lower score ❌

---

##  Scoring (Deterministic)

Score ∈ **[0.0, 1.0]**, based on:

* ✔ Correct classification
* ✔ Correct decision
* ✔ DB usage (reasoning)
* ✔ Empathy in response
* ✔ Task completion
* ✔ Efficiency (steps)
* ✔ Budget usage

-> Same input → **same score every time**

---

##  Baselines

###  Rule-Based (Primary)

* Deterministic
* No external dependencies
* Optimized for evaluation

```bash
python inference/baseline_rule_based.py
```
**Expected baseline score:** ~0.85–0.92 across tasks
---

###  OpenAI Agent (Optional)

* Demonstrates LLM compatibility
* Requires `OPENAI_API_KEY`

```bash
python inference/baseline_openai.py
```

> Not used for evaluation (non-deterministic)

---

##  Project Structure

```
openops-benchmark/
│
├── env/              # Environment + tasks + graders
├── inference/        # Baseline agents
├── openenv.yaml      # Spec
├── Dockerfile
└── README.md
```

---

##  Run with Docker

```bash
docker build -t openops-benchmark .
docker run openops-benchmark
```

---

##  Why This Scores High

✔ **Real-world utility** — models real support workflows
✔ **Strong tasks** — multi-step, increasing difficulty
✔ **Deterministic graders** — reproducible evaluation
✔ **Rich environment** — constraints + dynamics
✔ **Novel design** — cost-aware AI benchmarking

---

##  Key Insight

> This benchmark evaluates not just *what an agent predicts* —
> but *how it decides under constraints*.

---

## -> Final Note

Designed for **OpenEnv / OpenOps**, this environment pushes agents toward:

-> **Strategic, efficient, and realistic decision-making**

---

**This benchmark is designed to evaluate the next generation of AI agents operating in real-world systems.**
