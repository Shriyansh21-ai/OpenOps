"""OpenOps: OpenEnv Client"""

from __future__ import annotations
from typing import Any, Dict
from openenv.core import EnvClient
from openenv.core.env_client import StepResult
from models import SupportAction, SupportObservation, SupportState


class OpenOpsEnv(EnvClient[SupportAction, SupportObservation, SupportState]):
    def _step_payload(self, action: SupportAction) -> Dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[SupportObservation]:
        obs_data = payload.get("observation", payload)
        obs = SupportObservation(**obs_data)
        return StepResult(observation=obs, reward=payload.get("reward"), done=payload.get("done", obs.done))

    def _parse_state(self, payload: Dict[str, Any]) -> SupportState:
        return SupportState(**payload)
