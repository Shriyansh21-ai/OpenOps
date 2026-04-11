"""
OpenOps: Autonomous Customer Support Agent — Core Environment
"""

from __future__ import annotations

import random
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server import Environment

from models import (
    SupportAction,
    SupportObservation,
    SupportState,
    ActionType,
    ACTION_NAMES,
    NUM_ACTIONS,
    make_observation,
)
from dynamics import step_dynamics
from reward import (
    compute_step_reward,
    compute_penalties,
    compute_task_weighted_score,
    compute_baseline_comparison,
)
from baseline import run_baseline
from tasks import get_task, TaskConfig


class OpenOpsEnvironment(Environment[SupportAction, SupportObservation, SupportState]):
    """OpenEnv-compatible customer support RL environment."""

    def __init__(self) -> None:
        super().__init__()
        self._task: Optional[TaskConfig] = None
        self._state: SupportState = SupportState()
        self._rng: random.Random = random.Random(42)
        self._action_history: list[int] = []
        self._cumulative_reward: float = 0.0
        self._baseline_state: Optional[SupportState] = None
        self._done: bool = False
        self._initialized: bool = False

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SupportObservation:
        task_id = kwargs.get("task_id", "easy_1")
        self._task = get_task(task_id)
        self._state = self._task.initial_state.clone()
        self._state.episode_id = episode_id or str(uuid4())
        self._state.step_count = 0
        self._state.task_id = task_id

        effective_seed = seed if seed is not None else self._task.seed
        self._rng = random.Random(effective_seed)

        self._action_history = []
        self._cumulative_reward = 0.0
        self._done = False
        self._initialized = True

        self._baseline_state = run_baseline(task_id, policy_name="naive")

        return make_observation(
            self._state, self._rng,
            task_description=self._task.description,
            done=False, reward=None,
        )

    def step(
        self,
        action: SupportAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> SupportObservation:
        if not self._initialized:
            self.reset(task_id="easy_1")
        if self._done:
            self.reset(task_id=self._state.task_id)

        action_idx = action.action
        prev_state = self._state.clone()

        # Execute
        messages = step_dynamics(self._state, action_idx, self._rng)
        self._action_history.append(action_idx)

        # Reward
        step_reward = compute_step_reward(prev_state, self._state)
        penalty = compute_penalties(self._state)
        total_reward = step_reward + penalty
        self._cumulative_reward += total_reward

        # Termination
        done = False
        info: dict[str, Any] = {
            "step": self._state.step_num,
            "action_name": ACTION_NAMES.get(action_idx, str(action_idx)),
            "step_reward": round(step_reward, 6),
            "penalty": round(penalty, 6),
        }

        if self._state.ticket_closed:
            done = True
            info["termination_reason"] = "ticket_closed"
        elif self._state.step_num >= self._task.max_steps:
            done = True
            info["termination_reason"] = "max_steps_reached"

        if done:
            self._done = True
            agent_score = compute_task_weighted_score(self._state, self._task.scoring_weights)
            baseline_score = compute_task_weighted_score(self._baseline_state, self._task.scoring_weights)
            comparison = compute_baseline_comparison(self._state, self._baseline_state, self._task.scoring_weights)
            info["agent_final_score"] = round(agent_score, 6)
            info["baseline_final_score"] = round(baseline_score, 6)
            info["comparison_score"] = round(comparison, 6)
            info["cumulative_reward"] = round(self._cumulative_reward, 6)
            info["final_state"] = self._state.snapshot()

        obs = make_observation(
            self._state, self._rng,
            task_description=self._task.description if not done else "",
            done=done, reward=total_reward,
            system_messages=messages,
        )
        obs.metadata = info
        return obs

    @property
    def state(self) -> SupportState:
        return self._state
