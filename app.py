
# Optional demo runner (not required for evaluation)

from fastapi import FastAPI
from pydantic import BaseModel
from env.environment import OpenOpsEnvironment
from env.models import Action

app = FastAPI()

# Global environment instance
env = OpenOpsEnvironment()


class ActionInput(BaseModel):
    action_type: str
    content: str = ""


@app.get("/")
def root():
    return {"status": "ok", "message": "OpenOps environment running"}


@app.post("/reset")
def reset():
    obs = env.reset()
    return obs.dict()


@app.post("/step")
def step(action: ActionInput):
    action_obj = Action(**action.dict())
    obs, reward, done, info = env.step(action_obj)

    return {
        "observation": obs.dict(),
        "reward": reward,
        "done": done,
        "info": info
    }